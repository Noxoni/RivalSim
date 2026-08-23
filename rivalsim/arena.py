"""Exact RocketSim CMF loading and shared Warp static-world meshes.

The CMF format and mesh hash intentionally mirror RocketSim at the pinned
v0.2 authority commit.  Geometry remains an external runtime dependency: this
module never downloads, extracts, rewrites, or caches arena vertices.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

UU_PER_BT = np.float32(50.0)
SOCCAR_EXTENT_X = np.float32(4096.0)
SOCCAR_HEIGHT = np.float32(2048.0)
CMF_HEADER = struct.Struct("<ii")
MAX_CMF_ELEMENTS = 1_000_000


@dataclass(frozen=True, slots=True)
class CollisionMesh:
    """One validated RocketSim collision-mesh file."""

    path: Path
    triangles: np.ndarray
    vertices_bt: np.ndarray
    sha256: str
    rocketsim_hash: int

    @property
    def vertices_uu(self) -> np.ndarray:
        return np.asarray(self.vertices_bt * UU_PER_BT, dtype=np.float32)

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_bt.shape[0])

    def metadata(self) -> dict[str, object]:
        return {
            "file": self.path.name,
            "size_bytes": self.path.stat().st_size,
            "sha256": self.sha256,
            "rocketsim_mesh_hash": f"{self.rocketsim_hash:08X}",
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
            "bounds_bt": {
                "min": self.vertices_bt.min(axis=0).astype(float).tolist(),
                "max": self.vertices_bt.max(axis=0).astype(float).tolist(),
            },
            "bounds_uu": {
                "min": self.vertices_uu.min(axis=0).astype(float).tolist(),
                "max": self.vertices_uu.max(axis=0).astype(float).tolist(),
            },
        }


@dataclass(frozen=True, slots=True)
class ArenaGeometry:
    """Deterministically concatenated Soccar meshes in Rocket League units."""

    source_root: Path
    meshes: tuple[CollisionMesh, ...]
    vertices_uu: np.ndarray
    triangles: np.ndarray

    @classmethod
    def load_soccar(cls, collision_root: str | os.PathLike[str]) -> ArenaGeometry:
        root = Path(collision_root).expanduser().resolve()
        soccar = root if root.name.casefold() == "soccar" else root / "soccar"
        if not soccar.is_dir():
            raise FileNotFoundError(f"Soccar CMF directory not found: {soccar}")
        paths = sorted(soccar.glob("*.cmf"), key=_cmf_sort_key)
        if not paths:
            raise FileNotFoundError(f"no .cmf files found in {soccar}")
        meshes = tuple(read_cmf(path) for path in paths)
        vertices: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        vertex_offset = 0
        for mesh in meshes:
            vertices.append(mesh.vertices_uu)
            triangles.append(mesh.triangles + np.int32(vertex_offset))
            vertex_offset += mesh.vertex_count
        return cls(
            source_root=root,
            meshes=meshes,
            vertices_uu=np.ascontiguousarray(np.concatenate(vertices), dtype=np.float32),
            triangles=np.ascontiguousarray(np.concatenate(triangles), dtype=np.int32),
        )

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_uu.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def bounds_min(self) -> np.ndarray:
        return self.vertices_uu.min(axis=0)

    @property
    def bounds_max(self) -> np.ndarray:
        return self.vertices_uu.max(axis=0)

    @property
    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        for mesh in self.meshes:
            digest.update(bytes.fromhex(mesh.sha256))
        digest.update(self.vertices_uu.astype("<f4", copy=False).tobytes())
        digest.update(self.triangles.astype("<i4", copy=False).tobytes())
        return digest.hexdigest().upper()

    def metadata(self) -> dict[str, object]:
        return {
            "format": "RocketSim CMF",
            "scale_uu_per_bt": float(UU_PER_BT),
            "file_count": len(self.meshes),
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
            "bounds_uu": {
                "min": self.bounds_min.astype(float).tolist(),
                "max": self.bounds_max.astype(float).tolist(),
            },
            "combined_content_sha256": self.content_sha256,
            "files": [mesh.metadata() for mesh in self.meshes],
        }


class WarpArenaMeshes:
    """One normal BVH and one cuBQL BVH shared by every batched world."""

    def __init__(self, geometry: ArenaGeometry, device: str = "cuda:0"):
        self.geometry = geometry
        self.device = str(wp.get_device(device))
        self.points = wp.array(geometry.vertices_uu, dtype=wp.vec3, device=self.device)
        self.indices = wp.array(
            geometry.triangles.reshape(-1), dtype=wp.int32, device=self.device
        )
        # Keep the conventional mesh for AABB iteration. cuBQL currently does
        # not implement mesh_query_aabb, but is benchmarked for wheel rays.
        self.default = wp.Mesh(self.points, self.indices)
        self.cubql = wp.Mesh(self.points, self.indices, bvh_constructor="cubql")

    def refit(self) -> None:
        self.default.refit()
        self.cubql.refit()


def read_cmf(path: str | os.PathLike[str]) -> CollisionMesh:
    """Read a CMF exactly, rejecting truncation, trailing bytes, and bad indices."""

    resolved = Path(path).expanduser().resolve()
    payload = resolved.read_bytes()
    if len(payload) < CMF_HEADER.size:
        raise ValueError(f"truncated CMF header: {resolved}")
    triangle_count, vertex_count = CMF_HEADER.unpack_from(payload)
    if (
        min(triangle_count, vertex_count) <= 0
        or max(triangle_count, vertex_count) > MAX_CMF_ELEMENTS
    ):
        raise ValueError(
            f"invalid CMF counts in {resolved}: triangles={triangle_count}, "
            f"vertices={vertex_count}"
        )
    index_bytes = triangle_count * 3 * np.dtype("<i4").itemsize
    vertex_bytes = vertex_count * 3 * np.dtype("<f4").itemsize
    expected_size = CMF_HEADER.size + index_bytes + vertex_bytes
    if len(payload) != expected_size:
        raise ValueError(
            f"invalid CMF size for {resolved}: {len(payload)} bytes, expected {expected_size}"
        )
    triangles = np.frombuffer(
        payload,
        dtype="<i4",
        count=triangle_count * 3,
        offset=CMF_HEADER.size,
    ).reshape(triangle_count, 3)
    if int(triangles.min()) < 0 or int(triangles.max()) >= vertex_count:
        raise ValueError(f"out-of-range triangle index in {resolved}")
    vertices = np.frombuffer(
        payload,
        dtype="<f4",
        count=vertex_count * 3,
        offset=CMF_HEADER.size + index_bytes,
    ).reshape(vertex_count, 3)
    if not np.isfinite(vertices).all():
        raise ValueError(f"non-finite vertex in {resolved}")
    triangles = np.array(triangles, dtype=np.int32, order="C", copy=True)
    vertices = np.array(vertices, dtype=np.float32, order="C", copy=True)
    return CollisionMesh(
        path=resolved,
        triangles=triangles,
        vertices_bt=vertices,
        sha256=hashlib.sha256(payload).hexdigest().upper(),
        rocketsim_hash=rocketsim_mesh_hash(vertices, triangles),
    )


def rocketsim_mesh_hash(vertices: np.ndarray, triangles: np.ndarray) -> int:
    """Reproduce ``CollisionMeshFile::UpdateHash`` with uint32 overflow."""

    mask = 0xFFFFFFFF
    value = (len(vertices) + len(triangles) * len(vertices)) & mask
    for index in np.asarray(triangles, dtype=np.int64).reshape(-1):
        for component in np.asarray(vertices[int(index)], dtype=np.float32):
            # C++ truncates float toward zero before conversion to uint32.
            current = int(np.trunc(float(component))) & mask
            for _ in range(2):
                current = (((current >> 16) ^ current) * 0x045D9F3B) & mask
            current = ((current >> 16) ^ current) & mask
            combined = (
                current + 0x9E3779B9 + ((value << 6) & mask) + (value >> 2)
            ) & mask
            value = (value ^ combined) & mask
    return value


def raycast_triangles_cpu(
    vertices: np.ndarray,
    triangles: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    max_distances: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact-triangle, two-sided Moller-Trumbore CPU reference.

    Returns hit flags, distances, geometric normals in stored winding order,
    and face indices. Miss distances are ``max_distances`` and miss faces are -1.
    """

    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins and directions must both have shape (N, 3)")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("ray directions must be non-zero")
    directions = directions / lengths[:, None]
    maximum = np.broadcast_to(np.asarray(max_distances, dtype=np.float64), (len(origins),))
    if np.any(maximum <= 0.0):
        raise ValueError("max distances must be positive")

    tri_vertices = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles, dtype=np.int64)]
    edge1 = tri_vertices[:, 1] - tri_vertices[:, 0]
    edge2 = tri_vertices[:, 2] - tri_vertices[:, 0]
    raw_normals = np.cross(edge1, edge2)
    normal_lengths = np.linalg.norm(raw_normals, axis=1)
    valid_triangles = normal_lengths > 1e-12

    hit = np.zeros(len(origins), dtype=np.int32)
    distance = maximum.astype(np.float32)
    normal = np.zeros((len(origins), 3), dtype=np.float32)
    face = np.full(len(origins), -1, dtype=np.int32)
    epsilon = 1e-10
    for ray_index, (origin, direction, max_distance) in enumerate(
        zip(origins, directions, maximum, strict=True)
    ):
        pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum("ij,ij->i", edge1, pvec)
        candidate = valid_triangles & (np.abs(determinant) > epsilon)
        inverse = np.zeros_like(determinant)
        inverse[candidate] = 1.0 / determinant[candidate]
        tvec = origin - tri_vertices[:, 0]
        u = np.einsum("ij,ij->i", tvec, pvec) * inverse
        qvec = np.cross(tvec, edge1)
        v = qvec @ direction * inverse
        t = np.einsum("ij,ij->i", edge2, qvec) * inverse
        candidate &= (u >= -epsilon) & (v >= -epsilon) & (u + v <= 1.0 + epsilon)
        candidate &= (t >= 0.0) & (t <= max_distance)
        indices = np.flatnonzero(candidate)
        if indices.size:
            nearest = int(indices[np.argmin(t[indices])])
            hit[ray_index] = 1
            distance[ray_index] = np.float32(t[nearest])
            normal[ray_index] = (raw_normals[nearest] / normal_lengths[nearest]).astype(
                np.float32
            )
            face[ray_index] = nearest
    return hit, distance, normal, face


def raycast_soccar_cpu(
    geometry: ArenaGeometry,
    origins: np.ndarray,
    directions: np.ndarray,
    max_distances: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference RocketSim Soccar rays against CMFs plus its four static planes."""

    origins64 = np.asarray(origins, dtype=np.float64)
    directions64 = np.asarray(directions, dtype=np.float64)
    directions64 /= np.linalg.norm(directions64, axis=1, keepdims=True)
    maximum = np.broadcast_to(np.asarray(max_distances, dtype=np.float64), (len(origins64),))
    hit, distance, normal, face = raycast_triangles_cpu(
        geometry.vertices_uu,
        geometry.triangles,
        origins64,
        directions64,
        maximum,
    )
    # Plane codes deliberately remain outside the triangle-face namespace.
    planes = (
        (np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.0, 1.0)), -2),
        (np.array((0.0, 0.0, float(SOCCAR_HEIGHT))), np.array((0.0, 0.0, -1.0)), -3),
        (np.array((-float(SOCCAR_EXTENT_X), 0.0, 0.0)), np.array((1.0, 0.0, 0.0)), -4),
        (np.array((float(SOCCAR_EXTENT_X), 0.0, 0.0)), np.array((-1.0, 0.0, 0.0)), -5),
    )
    for ray_index, (origin, direction) in enumerate(zip(origins64, directions64, strict=True)):
        nearest = float(distance[ray_index])
        for point, plane_normal, plane_code in planes:
            denominator = float(np.dot(direction, plane_normal))
            if abs(denominator) <= 1e-12:
                continue
            candidate = float(np.dot(point - origin, plane_normal) / denominator)
            if 0.0 <= candidate <= nearest and candidate <= maximum[ray_index]:
                hit[ray_index] = 1
                distance[ray_index] = np.float32(candidate)
                normal[ray_index] = plane_normal.astype(np.float32)
                face[ray_index] = plane_code
                nearest = candidate
    return hit, distance, normal, face


def discover_collision_root(candidates: Iterable[str | os.PathLike[str]] = ()) -> Path:
    """Resolve an explicitly configured local RocketSim asset root."""

    configured = os.environ.get("RIVALSIM_COLLISION_DIR")
    paths = ([configured] if configured else []) + list(candidates)
    for candidate in paths:
        path = Path(candidate).expanduser().resolve()
        soccar = path if path.name.casefold() == "soccar" else path / "soccar"
        if soccar.is_dir() and any(soccar.glob("*.cmf")):
            return path
    raise FileNotFoundError(
        "set RIVALSIM_COLLISION_DIR to a RocketSim collision-mesh root containing soccar/*.cmf"
    )


def _cmf_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.stem.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else 2**31 - 1, path.name.casefold())
