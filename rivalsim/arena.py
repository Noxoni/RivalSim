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
        self.indices = wp.array(geometry.triangles.reshape(-1), dtype=wp.int32, device=self.device)
        edge_angles, edge_flags = build_internal_edge_data(geometry)
        bvh_rank = build_bullet_bvh_rank(geometry)
        self.internal_edge_angles = wp.array(
            edge_angles, dtype=wp.vec3, device=self.device
        )
        self.internal_edge_flags = wp.array(
            edge_flags, dtype=wp.int32, device=self.device
        )
        self.bullet_bvh_rank = wp.array(
            bvh_rank, dtype=wp.int32, device=self.device
        )
        # Keep the conventional mesh for AABB iteration. cuBQL currently does
        # not implement mesh_query_aabb, but is benchmarked for wheel rays.
        self.default = wp.Mesh(self.points, self.indices)
        self.cubql = wp.Mesh(self.points, self.indices, bvh_constructor="cubql")

    def refit(self) -> None:
        self.default.refit()
        self.cubql.refit()


def build_internal_edge_data(
    geometry: ArenaGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Bullet's per-triangle shared-edge angles and flags.

    RocketSim calls ``btGenerateInternalEdgeInfo`` once for each source CMF.
    The resulting angles and convex/swap flags drive
    ``btAdjustInternalEdgeContacts`` on every generated mesh contact.  Keeping
    the source meshes separate here is therefore part of the collision ABI.
    """

    angles = np.full(
        (geometry.triangle_count, 3),
        np.float32(2.0 * np.pi),
        dtype=np.float32,
    )
    flags = np.zeros(geometry.triangle_count, dtype=np.int32)
    face_offset = 0
    edge_vertices = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for mesh in geometry.meshes:
        vertices = np.asarray(mesh.vertices_bt, dtype=np.float32)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        vertex_keys = [tuple(vertex.view(np.uint32).tolist()) for vertex in vertices]
        edge_map: dict[
            tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, int]]
        ] = {}
        for face, triangle in enumerate(triangles):
            for edge, (start, end, _other) in enumerate(edge_vertices):
                key_a = vertex_keys[int(triangle[start])]
                key_b = vertex_keys[int(triangle[end])]
                key = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
                edge_map.setdefault(key, []).append((face, edge))

        for entries in edge_map.values():
            if len(entries) < 2:
                continue
            for face, edge in entries:
                neighbor = next(item for item in entries if item[0] != face)
                edge_angle, edge_flags = _bullet_edge_data(
                    vertices,
                    triangles[face],
                    edge,
                    triangles[neighbor[0]],
                )
                angles[face_offset + face, edge] = edge_angle
                flags[face_offset + face] |= edge_flags
        face_offset += mesh.triangle_count
    return angles, flags


def build_bullet_bvh_rank(geometry: ArenaGeometry) -> np.ndarray:
    """Return each face's visit rank in Bullet's quantized BVH.

    ``btOptimizedBvh`` recursively partitions its leaf array in place, then
    visits the resulting leaves in that fixed depth-first order for every AABB
    query.  RocketSim owns one BVH per CMF, so ranks restart at every source
    mesh and are offset only to make the combined array deterministic.
    """

    result = np.empty(geometry.triangle_count, dtype=np.int32)
    face_offset = 0
    rank_offset = 0
    for mesh in geometry.meshes:
        vertices = np.asarray(mesh.vertices_bt, dtype=np.float32)
        triangle_vertices = vertices[np.asarray(mesh.triangles, dtype=np.int32)]
        leaf_min = np.min(triangle_vertices, axis=1).astype(np.float32)
        leaf_max = np.max(triangle_vertices, axis=1).astype(np.float32)
        thin = leaf_max - leaf_min < np.float32(0.002)
        leaf_min = np.where(thin, leaf_min - np.float32(0.001), leaf_min)
        leaf_max = np.where(thin, leaf_max + np.float32(0.001), leaf_max)

        bvh_min, _bvh_max, quantization = _bullet_quantization_bounds(
            vertices.min(axis=0), vertices.max(axis=0)
        )
        quantized_min = _bullet_quantize(leaf_min, bvh_min, quantization, False)
        quantized_max = _bullet_quantize(leaf_max, bvh_min, quantization, True)
        faces = np.arange(mesh.triangle_count, dtype=np.int32)
        _bullet_partition_leaves(
            quantized_min,
            quantized_max,
            faces,
            bvh_min,
            quantization,
            0,
            mesh.triangle_count,
        )
        for rank, face in enumerate(faces):
            result[face_offset + int(face)] = np.int32(rank_offset + rank)
        face_offset += mesh.triangle_count
        rank_offset += mesh.triangle_count
    return result


def _bullet_partition_leaves(
    quantized_min: np.ndarray,
    quantized_max: np.ndarray,
    faces: np.ndarray,
    bvh_min: np.ndarray,
    quantization: np.ndarray,
    start: int,
    end: int,
) -> None:
    """Mirror ``btQuantizedBvh::buildTree``'s in-place leaf partition."""

    count = end - start
    if count <= 1:
        return

    def center(index: int) -> np.ndarray:
        minimum = quantized_min[index].astype(np.float32) / quantization + bvh_min
        maximum = quantized_max[index].astype(np.float32) / quantization + bvh_min
        return np.asarray(np.float32(0.5) * (maximum + minimum), dtype=np.float32)

    mean = np.zeros(3, dtype=np.float32)
    for index in range(start, end):
        mean = np.asarray(mean + center(index), dtype=np.float32)
    mean = np.asarray(mean * np.float32(1.0 / count), dtype=np.float32)

    variance = np.zeros(3, dtype=np.float32)
    for index in range(start, end):
        difference = np.asarray(center(index) - mean, dtype=np.float32)
        variance = np.asarray(variance + difference * difference, dtype=np.float32)
    variance = np.asarray(
        variance * np.float32(1.0 / (count - 1)), dtype=np.float32
    )
    if variance[0] < variance[1]:
        split_axis = 2 if variance[1] < variance[2] else 1
    else:
        split_axis = 2 if variance[0] < variance[2] else 0

    split_value = mean[split_axis]
    split = start
    for index in range(start, end):
        if center(index)[split_axis] > split_value:
            if index != split:
                quantized_min[[index, split]] = quantized_min[[split, index]]
                quantized_max[[index, split]] = quantized_max[[split, index]]
                faces[[index, split]] = faces[[split, index]]
            split += 1

    balanced_range = count // 3
    if split <= start + balanced_range or split >= end - 1 - balanced_range:
        split = start + (count >> 1)
    _bullet_partition_leaves(
        quantized_min,
        quantized_max,
        faces,
        bvh_min,
        quantization,
        start,
        split,
    )
    _bullet_partition_leaves(
        quantized_min,
        quantized_max,
        faces,
        bvh_min,
        quantization,
        split,
        end,
    )


def _bullet_quantization_bounds(
    local_min: np.ndarray, local_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = np.float32(1.0)
    bvh_min = np.asarray(local_min - margin, dtype=np.float32)
    bvh_max = np.asarray(local_max + margin, dtype=np.float32)
    quantization = np.asarray(
        np.float32(65533.0) / (bvh_max - bvh_min), dtype=np.float32
    )

    quantized = _bullet_quantize(
        bvh_min[None, :], bvh_min, quantization, False
    )[0]
    unquantized = np.asarray(
        quantized.astype(np.float32) / quantization + bvh_min,
        dtype=np.float32,
    )
    bvh_min = np.minimum(bvh_min, unquantized - margin).astype(np.float32)
    quantization = np.asarray(
        np.float32(65533.0) / (bvh_max - bvh_min), dtype=np.float32
    )

    quantized = _bullet_quantize(
        bvh_max[None, :], bvh_min, quantization, True
    )[0]
    unquantized = np.asarray(
        quantized.astype(np.float32) / quantization + bvh_min,
        dtype=np.float32,
    )
    bvh_max = np.maximum(bvh_max, unquantized + margin).astype(np.float32)
    quantization = np.asarray(
        np.float32(65533.0) / (bvh_max - bvh_min), dtype=np.float32
    )
    return bvh_min, bvh_max, quantization


def _bullet_quantize(
    points: np.ndarray,
    bvh_min: np.ndarray,
    quantization: np.ndarray,
    is_maximum: bool,
) -> np.ndarray:
    scaled = np.asarray((points - bvh_min) * quantization, dtype=np.float32)
    if is_maximum:
        integers = np.asarray(scaled + np.float32(1.0), dtype=np.uint16)
        return np.asarray(integers | np.uint16(1), dtype=np.uint16)
    integers = np.asarray(scaled, dtype=np.uint16)
    return np.asarray(integers & np.uint16(0xFFFE), dtype=np.uint16)


def _bullet_edge_data(
    vertices: np.ndarray,
    triangle_a: np.ndarray,
    edge_a: int,
    triangle_b: np.ndarray,
) -> tuple[np.float32, np.int32]:
    """Mirror one shared-edge result from Bullet's connectivity processor."""

    edge_vertices = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    start_a, end_a, other_a = edge_vertices[edge_a]
    key_start = tuple(vertices[int(triangle_a[start_a])].view(np.uint32).tolist())
    key_end = tuple(vertices[int(triangle_a[end_a])].view(np.uint32).tolist())
    keys_b = [tuple(vertices[int(index)].view(np.uint32).tolist()) for index in triangle_b]
    start_b = keys_b.index(key_start)
    end_b = keys_b.index(key_end)
    other_b = 3 - start_b - end_b

    a = vertices[triangle_a]
    b = vertices[triangle_b]
    edge = _normalize_f32(a[end_a] - a[start_a])
    normal_a = _normalize_f32(np.cross(a[1] - a[0], a[2] - a[0]))
    # btConnectivityProcessor constructs tB with the shared edge reversed.
    normal_b = _normalize_f32(
        np.cross(b[start_b] - b[end_b], b[other_b] - b[end_b])
    )
    edge_cross_a = _normalize_f32(np.cross(edge, normal_a))
    if np.dot(edge_cross_a, a[other_a] - a[start_a]) < 0.0:
        edge_cross_a = -edge_cross_a
    edge_cross_b = _normalize_f32(np.cross(edge, normal_b))
    if np.dot(edge_cross_b, b[other_b] - b[start_b]) < 0.0:
        edge_cross_b = -edge_cross_b
    calculated_edge = np.cross(edge_cross_a, edge_cross_b)
    if np.dot(calculated_edge, calculated_edge) < np.float32(0.0001):
        return np.float32(0.0), np.int32(0)

    calculated_edge = _normalize_f32(calculated_edge)
    calculated_normal_a = _normalize_f32(np.cross(calculated_edge, edge_cross_a))
    angle_2 = np.float32(
        np.arctan2(
            np.dot(edge_cross_b, calculated_normal_a),
            np.dot(edge_cross_b, edge_cross_a),
        )
    )
    angle_4 = np.float32(np.pi) - angle_2
    is_convex = bool(np.dot(normal_a, edge_cross_b) < 0.0)
    corrected_angle = angle_4 if is_convex else -angle_4
    stored_angle = np.float32(-corrected_angle)

    # btConnectivityProcessor verifies the reconstructed neighboring normal
    # and records whether it must be flipped for this directed edge.
    rotation_edge = a[start_a] - a[end_a]
    computed_normal_b = _rotate_axis_angle_f32(
        normal_a, rotation_edge, stored_angle
    )
    edge_flags = np.int32(0)
    if is_convex:
        edge_flags |= np.int32(1 << edge_a)
    if np.dot(computed_normal_b, normal_b) < 0.0:
        edge_flags |= np.int32(1 << (edge_a + 3))
    return stored_angle, edge_flags


def _normalize_f32(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    length = np.float32(np.sqrt(np.dot(value, value)))
    if length <= 0.0:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(value / length, dtype=np.float32)


def _rotate_axis_angle_f32(
    value: np.ndarray, axis: np.ndarray, angle: np.float32
) -> np.ndarray:
    """Float32 Rodrigues rotation matching Bullet's axis-angle quaternion."""

    unit_axis = _normalize_f32(axis)
    sine = np.float32(np.sin(np.float32(angle)))
    cosine = np.float32(np.cos(np.float32(angle)))
    one_minus_cosine = np.float32(1.0) - cosine
    return np.asarray(
        value * cosine
        + np.cross(unit_axis, value) * sine
        + unit_axis * np.dot(unit_axis, value) * one_minus_cosine,
        dtype=np.float32,
    )


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
            f"invalid CMF counts in {resolved}: triangles={triangle_count}, vertices={vertex_count}"
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
            combined = (current + 0x9E3779B9 + ((value << 6) & mask) + (value >> 2)) & mask
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
            normal[ray_index] = (raw_normals[nearest] / normal_lengths[nearest]).astype(np.float32)
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
            if 0.0 <= candidate < nearest and candidate <= maximum[ray_index]:
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
