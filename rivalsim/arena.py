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


_BULLET_FACE_NORMAL = r"""
    // The pinned Windows build uses btVector3::normalize's _mm_rsqrt_ss
    // estimate plus one Newton step. btTriangleShape::calcNormal performs one
    // normalization for contact/internal-edge adjustment; the wheel ray path
    // normalizes that result a second time in btDefaultVehicleRaycaster.
    // Compute both immutable per-face variants on the authority CPU.
    auto op_add = [](float a, float b) -> float {
        volatile float value = a + b;
        return value;
    };
    auto op_sub = [](float a, float b) -> float {
        volatile float value = a - b;
        return value;
    };
    auto op_mul = [](float a, float b) -> float {
        volatile float value = a * b;
        return value;
    };
    struct FaceV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> FaceV3 {
        FaceV3 value = {x, y, z};
        return value;
    };
    auto sub = [&](FaceV3 a, FaceV3 b) -> FaceV3 {
        return make(op_sub(a.x, b.x), op_sub(a.y, b.y), op_sub(a.z, b.z));
    };
    auto cross = [&](FaceV3 a, FaceV3 b) -> FaceV3 {
        return make(
            op_sub(op_mul(a.y, b.z), op_mul(a.z, b.y)),
            op_sub(op_mul(a.z, b.x), op_mul(a.x, b.z)),
            op_sub(op_mul(a.x, b.y), op_mul(a.y, b.x)));
    };
    auto dot = [&](FaceV3 a, FaceV3 b) -> float {
        return op_add(op_add(op_mul(a.x, b.x), op_mul(a.y, b.y)), op_mul(a.z, b.z));
    };
    auto normalize = [&](FaceV3 value) -> FaceV3 {
        const float length_squared = dot(value, value);
        using FaceM128 = float __attribute__((__vector_size__(16)));
        FaceM128 input = {length_squared, 0.0f, 0.0f, 0.0f};
        FaceM128 estimate = __builtin_ia32_rsqrtss(input);
        float inverse_length = estimate[0];
        float correction = op_mul(op_mul(length_squared, 0.5f), inverse_length);
        correction = op_mul(correction, inverse_length);
        correction = op_sub(1.5f, correction);
        inverse_length = op_mul(inverse_length, correction);
        return make(
            op_mul(value.x, inverse_length),
            op_mul(value.y, inverse_length),
            op_mul(value.z, inverse_length));
    };
    FaceV3 a = make(v0[0], v0[1], v0[2]);
    FaceV3 b = make(v1[0], v1[1], v1[2]);
    FaceV3 c = make(v2[0], v2[1], v2[2]);
    FaceV3 normal = normalize(cross(sub(b, a), sub(c, a)));
    if (normalize_twice != 0) {
        normal = normalize(normal);
    }
    return wp::vec_t<3, wp::float32>(normal.x, normal.y, normal.z);
"""


@wp.func_native(_BULLET_FACE_NORMAL)
def _bullet_face_normal(
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, normalize_twice: int
) -> wp.vec3: ...


_BULLET_INTERNAL_EDGE_INFO = r"""
    // Direct fixed-triangle specialization of btConnectivityProcessor in
    // btInternalEdgeUtility.cpp. The Python side supplies the already
    // resolved shared-edge connectivity; this function preserves the pinned
    // Windows Bullet float32/SSE operation order that creates the cached
    // angle and convex/swap flags.
    auto op_add = [](float a, float b) -> float {
        volatile float result = a + b;
        return result;
    };
    auto op_sub = [](float a, float b) -> float {
        volatile float result = a - b;
        return result;
    };
    auto op_mul = [](float a, float b) -> float {
        volatile float result = a * b;
        return result;
    };
    auto op_div = [](float a, float b) -> float {
        volatile float result = a / b;
        return result;
    };
    struct EdgeV3 { float x; float y; float z; };
    struct EdgeQ { float x; float y; float z; float w; };
    auto make = [](float x, float y, float z) -> EdgeV3 {
        EdgeV3 result = {x, y, z};
        return result;
    };
    auto sub = [&](EdgeV3 a, EdgeV3 b) -> EdgeV3 {
        return make(op_sub(a.x, b.x), op_sub(a.y, b.y), op_sub(a.z, b.z));
    };
    auto scale = [&](EdgeV3 value, float amount) -> EdgeV3 {
        return make(
            op_mul(value.x, amount),
            op_mul(value.y, amount),
            op_mul(value.z, amount));
    };
    auto dot = [&](EdgeV3 a, EdgeV3 b) -> float {
        return op_add(
            op_add(op_mul(a.x, b.x), op_mul(a.y, b.y)),
            op_mul(a.z, b.z));
    };
    auto cross = [&](EdgeV3 a, EdgeV3 b) -> EdgeV3 {
        return make(
            op_sub(op_mul(a.y, b.z), op_mul(a.z, b.y)),
            op_sub(op_mul(a.z, b.x), op_mul(a.x, b.z)),
            op_sub(op_mul(a.x, b.y), op_mul(a.y, b.x)));
    };
    auto normalize = [&](EdgeV3 value) -> EdgeV3 {
        const float length_squared = dot(value, value);
        using EdgeM128 = float __attribute__((__vector_size__(16)));
        EdgeM128 input = {length_squared, 0.0f, 0.0f, 0.0f};
        EdgeM128 estimate = __builtin_ia32_rsqrtss(input);
        float inverse_length = estimate[0];
        float correction = op_mul(op_mul(length_squared, 0.5f), inverse_length);
        correction = op_mul(correction, inverse_length);
        correction = op_sub(1.5f, correction);
        inverse_length = op_mul(inverse_length, correction);
        return scale(value, inverse_length);
    };
    auto quaternion_product = [&](EdgeQ lhs, EdgeQ rhs) -> EdgeQ {
        EdgeQ result;
        result.x = op_add(
            op_sub(op_mul(lhs.w, rhs.x), op_mul(lhs.z, rhs.y)),
            op_add(op_mul(lhs.x, rhs.w), op_mul(lhs.y, rhs.z)));
        result.y = op_add(
            op_sub(op_mul(lhs.w, rhs.y), op_mul(lhs.x, rhs.z)),
            op_add(op_mul(lhs.y, rhs.w), op_mul(lhs.z, rhs.x)));
        result.z = op_add(
            op_sub(op_mul(lhs.w, rhs.z), op_mul(lhs.y, rhs.x)),
            op_add(op_mul(lhs.z, rhs.w), op_mul(lhs.x, rhs.y)));
        result.w = op_sub(
            op_sub(op_mul(lhs.w, rhs.w), op_mul(lhs.z, rhs.z)),
            op_add(op_mul(lhs.x, rhs.x), op_mul(lhs.y, rhs.y)));
        return result;
    };
    auto quaternion_vector_product = [&](EdgeQ quat, EdgeV3 value) -> EdgeQ {
        EdgeQ result;
        result.x = op_sub(
            op_add(op_mul(quat.w, value.x), op_mul(quat.y, value.z)),
            op_mul(quat.z, value.y));
        result.y = op_sub(
            op_add(op_mul(quat.w, value.y), op_mul(quat.z, value.x)),
            op_mul(quat.x, value.z));
        result.z = op_sub(
            op_add(op_mul(quat.w, value.z), op_mul(quat.x, value.y)),
            op_mul(quat.y, value.x));
        result.w = op_sub(
            op_sub(0.0f, op_add(op_mul(quat.x, value.x), op_mul(quat.y, value.y))),
            op_mul(quat.z, value.z));
        return result;
    };

    const EdgeV3 a[3] = {
        make(a0[0], a0[1], a0[2]),
        make(a1[0], a1[1], a1[2]),
        make(a2[0], a2[1], a2[2]),
    };
    const EdgeV3 shared_b0 = make(b0[0], b0[1], b0[2]);
    const EdgeV3 shared_b1 = make(b1[0], b1[1], b1[2]);
    const EdgeV3 other_b = make(b2[0], b2[1], b2[2]);
    const int edge_end = edge_index == 2 ? 0 : edge_index + 1;
    const int other_a = 3 - edge_index - edge_end;

    EdgeV3 edge = normalize(sub(a[edge_end], a[edge_index]));
    const EdgeV3 normal_a = normalize(cross(sub(a[1], a[0]), sub(a[2], a[0])));
    // btTriangleShape tB(sharedVertsB[1], sharedVertsB[0], otherIndexB)
    const EdgeV3 normal_b = normalize(
        cross(sub(shared_b0, shared_b1), sub(other_b, shared_b1)));

    EdgeV3 edge_cross_a = normalize(cross(edge, normal_a));
    if (dot(edge_cross_a, sub(a[other_a], a[edge_index])) < 0.0f) {
        edge_cross_a = scale(edge_cross_a, -1.0f);
    }
    EdgeV3 edge_cross_b = normalize(cross(edge, normal_b));
    if (dot(edge_cross_b, sub(other_b, shared_b0)) < 0.0f) {
        edge_cross_b = scale(edge_cross_b, -1.0f);
    }

    EdgeV3 calculated_edge = cross(edge_cross_a, edge_cross_b);
    const float length_squared = dot(calculated_edge, calculated_edge);
    if (length_squared < 0.0001f) {
        return wp::vec_t<2, wp::float32>(0.0f, 0.0f);
    }

    calculated_edge = normalize(calculated_edge);
    EdgeV3 calculated_normal_a = normalize(cross(calculated_edge, edge_cross_a));
    const float angle_2 = atan2f(
        dot(edge_cross_b, calculated_normal_a),
        dot(edge_cross_b, edge_cross_a));
    const float angle_4 = op_sub(3.1415926535897932384626433832795029f, angle_2);
    const bool is_convex = dot(normal_a, edge_cross_b) < 0.0f;
    const float corrected_angle = is_convex ? angle_4 : -angle_4;
    const float stored_angle = -corrected_angle;

    // btQuaternion(a[start] - a[end], stored_angle), then quatRotate.
    const EdgeV3 rotation_axis = sub(a[edge_index], a[edge_end]);
    const float axis_length = sqrtf(dot(rotation_axis, rotation_axis));
    const float half_angle = op_mul(stored_angle, 0.5f);
    const float quaternion_scale = op_div(sinf(half_angle), axis_length);
    const EdgeQ rotation = {
        op_mul(rotation_axis.x, quaternion_scale),
        op_mul(rotation_axis.y, quaternion_scale),
        op_mul(rotation_axis.z, quaternion_scale),
        cosf(half_angle),
    };
    const EdgeQ first = quaternion_vector_product(rotation, normal_a);
    const EdgeQ inverse = {-rotation.x, -rotation.y, -rotation.z, rotation.w};
    const EdgeQ rotated = quaternion_product(first, inverse);
    const EdgeV3 computed_normal_b = make(rotated.x, rotated.y, rotated.z);

    int result_flags = is_convex ? (1 << edge_index) : 0;
    if (dot(computed_normal_b, normal_b) < 0.0f) {
        result_flags |= 1 << (edge_index + 3);
    }
    return wp::vec_t<2, wp::float32>(stored_angle, (float)result_flags);
"""


@wp.func_native(_BULLET_INTERNAL_EDGE_INFO)
def _bullet_internal_edge_info(
    a0: wp.vec3,
    a1: wp.vec3,
    a2: wp.vec3,
    b0: wp.vec3,
    b1: wp.vec3,
    b2: wp.vec3,
    edge_index: int,
) -> wp.vec2: ...


@wp.kernel(enable_backward=False)
def _build_bullet_internal_edge_info(
    vertices_bt: wp.array(dtype=wp.vec3),
    record_vertex_indices: wp.array(dtype=wp.int32),
    record_edge_indices: wp.array(dtype=wp.int32),
    record_angles: wp.array(dtype=wp.float32),
    record_flags: wp.array(dtype=wp.int32),
):
    record = wp.tid()
    offset = record * 6
    result = _bullet_internal_edge_info(
        vertices_bt[record_vertex_indices[offset]],
        vertices_bt[record_vertex_indices[offset + 1]],
        vertices_bt[record_vertex_indices[offset + 2]],
        vertices_bt[record_vertex_indices[offset + 3]],
        vertices_bt[record_vertex_indices[offset + 4]],
        vertices_bt[record_vertex_indices[offset + 5]],
        record_edge_indices[record],
    )
    record_angles[record] = result[0]
    record_flags[record] = int(result[1])


_BULLET_INTERNAL_EDGE_STATIC_VECTOR = r"""
    // Static specialization of the two immutable vector constructions in
    // btAdjustInternalEdgeContacts/btClampNormal. The authority CPU supplies
    // Bullet's exact SSE normalization and quaternion-product operation order
    // once per CMF face/edge; the GPU consumes those source results directly.
    auto op_add = [](float a, float b) -> float {
        volatile float result = a + b;
        return result;
    };
    auto op_sub = [](float a, float b) -> float {
        volatile float result = a - b;
        return result;
    };
    auto op_mul = [](float a, float b) -> float {
        volatile float result = a * b;
        return result;
    };
    auto op_div = [](float a, float b) -> float {
        volatile float result = a / b;
        return result;
    };
    struct EdgeV3 { float x; float y; float z; };
    struct EdgeQ { float x; float y; float z; float w; };
    auto make = [](float x, float y, float z) -> EdgeV3 {
        EdgeV3 result = {x, y, z};
        return result;
    };
    auto sub = [&](EdgeV3 a, EdgeV3 b) -> EdgeV3 {
        return make(op_sub(a.x, b.x), op_sub(a.y, b.y), op_sub(a.z, b.z));
    };
    auto scale = [&](EdgeV3 value, float amount) -> EdgeV3 {
        return make(
            op_mul(value.x, amount),
            op_mul(value.y, amount),
            op_mul(value.z, amount));
    };
    auto dot = [&](EdgeV3 a, EdgeV3 b) -> float {
        return op_add(
            op_add(op_mul(a.x, b.x), op_mul(a.y, b.y)),
            op_mul(a.z, b.z));
    };
    auto cross = [&](EdgeV3 a, EdgeV3 b) -> EdgeV3 {
        return make(
            op_sub(op_mul(a.y, b.z), op_mul(a.z, b.y)),
            op_sub(op_mul(a.z, b.x), op_mul(a.x, b.z)),
            op_sub(op_mul(a.x, b.y), op_mul(a.y, b.x)));
    };
    auto normalize = [&](EdgeV3 value) -> EdgeV3 {
        const float length_squared = dot(value, value);
        using EdgeM128 = float __attribute__((__vector_size__(16)));
        EdgeM128 input = {length_squared, 0.0f, 0.0f, 0.0f};
        EdgeM128 estimate = __builtin_ia32_rsqrtss(input);
        float inverse_length = estimate[0];
        float correction = op_mul(op_mul(length_squared, 0.5f), inverse_length);
        correction = op_mul(correction, inverse_length);
        correction = op_sub(1.5f, correction);
        inverse_length = op_mul(inverse_length, correction);
        return scale(value, inverse_length);
    };
    auto quaternion_product = [&](EdgeQ lhs, EdgeQ rhs) -> EdgeQ {
        // Four-wide SSE grouping from btQuaternion::operator*.
        EdgeQ result;
        result.x = op_add(
            op_sub(op_mul(lhs.w, rhs.x), op_mul(lhs.z, rhs.y)),
            op_add(op_mul(lhs.x, rhs.w), op_mul(lhs.y, rhs.z)));
        result.y = op_add(
            op_sub(op_mul(lhs.w, rhs.y), op_mul(lhs.x, rhs.z)),
            op_add(op_mul(lhs.y, rhs.w), op_mul(lhs.z, rhs.x)));
        result.z = op_add(
            op_sub(op_mul(lhs.w, rhs.z), op_mul(lhs.y, rhs.x)),
            op_add(op_mul(lhs.z, rhs.w), op_mul(lhs.x, rhs.y)));
        result.w = op_sub(
            op_sub(op_mul(lhs.w, rhs.w), op_mul(lhs.z, rhs.z)),
            op_add(op_mul(lhs.x, rhs.x), op_mul(lhs.y, rhs.y)));
        return result;
    };
    auto quaternion_vector_product = [&](EdgeQ quat, EdgeV3 value) -> EdgeQ {
        // Specialized SSE grouping from btQuaternion::operator*(btVector3).
        EdgeQ result;
        result.x = op_sub(
            op_add(op_mul(quat.w, value.x), op_mul(quat.y, value.z)),
            op_mul(quat.z, value.y));
        result.y = op_sub(
            op_add(op_mul(quat.w, value.y), op_mul(quat.z, value.x)),
            op_mul(quat.x, value.z));
        result.z = op_sub(
            op_add(op_mul(quat.w, value.z), op_mul(quat.x, value.y)),
            op_mul(quat.y, value.x));
        result.w = op_sub(
            op_sub(0.0f, op_add(op_mul(quat.x, value.x), op_mul(quat.y, value.y))),
            op_mul(quat.z, value.z));
        return result;
    };

    const EdgeV3 vertices[3] = {
        make(v0[0], v0[1], v0[2]),
        make(v1[0], v1[1], v1[2]),
        make(v2[0], v2[1], v2[2]),
    };
    const int edge_end = edge_index == 2 ? 0 : edge_index + 1;
    const EdgeV3 edge = sub(vertices[edge_index], vertices[edge_end]);
    const EdgeV3 triangle = make(
        triangle_normal[0], triangle_normal[1], triangle_normal[2]);
    const bool convex = (edge_flags & (1 << edge_index)) != 0;
    const float swap_factor = convex ? 1.0f : -1.0f;
    const EdgeV3 normal_a = scale(triangle, swap_factor);

    if (mode == 0) {
        const EdgeV3 edge_cross = normalize(cross(edge, normal_a));
        return wp::vec_t<3, wp::float32>(
            edge_cross.x, edge_cross.y, edge_cross.z);
    }

    const float axis_length = sqrtf(dot(edge, edge));
    const float half_angle = op_mul(edge_angle, 0.5f);
    const float quaternion_scale = op_div(sinf(half_angle), axis_length);
    const EdgeQ rotation = {
        op_mul(edge.x, quaternion_scale),
        op_mul(edge.y, quaternion_scale),
        op_mul(edge.z, quaternion_scale),
        cosf(half_angle),
    };
    const EdgeQ first = quaternion_vector_product(rotation, triangle);
    const EdgeQ inverse = {-rotation.x, -rotation.y, -rotation.z, rotation.w};
    const EdgeQ rotated = quaternion_product(first, inverse);
    EdgeV3 normal_b = make(rotated.x, rotated.y, rotated.z);
    if ((edge_flags & (1 << (edge_index + 3))) != 0) {
        normal_b = scale(normal_b, -1.0f);
    }
    normal_b = scale(normal_b, swap_factor);
    return wp::vec_t<3, wp::float32>(normal_b.x, normal_b.y, normal_b.z);
"""


@wp.func_native(_BULLET_INTERNAL_EDGE_STATIC_VECTOR)
def _bullet_internal_edge_static_vector(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    triangle_normal: wp.vec3,
    edge_angle: float,
    edge_flags: int,
    edge_index: int,
    mode: int,
) -> wp.vec3: ...


@wp.kernel(enable_backward=False)
def _build_bullet_face_normals(
    vertices_bt: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    edge_angles: wp.array(dtype=wp.vec3),
    edge_flags: wp.array(dtype=wp.int32),
    wheel_face_normals: wp.array(dtype=wp.vec3),
    internal_edge_face_normals: wp.array(dtype=wp.vec3),
    internal_edge_crosses: wp.array(dtype=wp.vec3),
    internal_edge_normal_bs: wp.array(dtype=wp.vec3),
):
    face = wp.tid()
    offset = face * 3
    v0 = vertices_bt[triangle_indices[offset]]
    v1 = vertices_bt[triangle_indices[offset + 1]]
    v2 = vertices_bt[triangle_indices[offset + 2]]
    internal_edge_face_normal = _bullet_face_normal(v0, v1, v2, 0)
    internal_edge_face_normals[face] = internal_edge_face_normal
    wheel_face_normals[face] = _bullet_face_normal(
        v0,
        v1,
        v2,
        1,
    )
    for edge_index in range(3):
        output_index = face * 3 + edge_index
        internal_edge_crosses[output_index] = _bullet_internal_edge_static_vector(
            v0,
            v1,
            v2,
            internal_edge_face_normal,
            edge_angles[face][edge_index],
            edge_flags[face],
            edge_index,
            0,
        )
        internal_edge_normal_bs[output_index] = _bullet_internal_edge_static_vector(
            v0,
            v1,
            v2,
            internal_edge_face_normal,
            edge_angles[face][edge_index],
            edge_flags[face],
            edge_index,
            1,
        )


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
        # The CMF payload is already stored in Bullet coordinates.  Keep those
        # original float32 bits alongside the UU acceleration mesh: scaling a
        # UU vertex back by 0.02 is not bit-reversible for every arena value,
        # and Bullet narrowphase/manifold tie ordering observes the difference.
        vertices_bt = np.ascontiguousarray(
            np.concatenate([mesh.vertices_bt for mesh in geometry.meshes]),
            dtype=np.float32,
        )
        self.points_bt = wp.array(vertices_bt, dtype=wp.vec3, device=self.device)
        self.indices = wp.array(geometry.triangles.reshape(-1), dtype=wp.int32, device=self.device)
        points_bt_cpu = wp.array(vertices_bt, dtype=wp.vec3, device="cpu")
        indices_cpu = wp.array(
            geometry.triangles.reshape(-1), dtype=wp.int32, device="cpu"
        )
        edge_angles, edge_flags = build_internal_edge_data(geometry)
        edge_angles_cpu = wp.array(edge_angles, dtype=wp.vec3, device="cpu")
        edge_flags_cpu = wp.array(edge_flags, dtype=wp.int32, device="cpu")
        face_normals_cpu = wp.empty(
            geometry.triangle_count, dtype=wp.vec3, device="cpu"
        )
        internal_edge_face_normals_cpu = wp.empty(
            geometry.triangle_count, dtype=wp.vec3, device="cpu"
        )
        internal_edge_crosses_cpu = wp.empty(
            geometry.triangle_count * 3, dtype=wp.vec3, device="cpu"
        )
        internal_edge_normal_bs_cpu = wp.empty(
            geometry.triangle_count * 3, dtype=wp.vec3, device="cpu"
        )
        wp.launch(
            _build_bullet_face_normals,
            dim=geometry.triangle_count,
            inputs=[
                points_bt_cpu,
                indices_cpu,
                edge_angles_cpu,
                edge_flags_cpu,
                face_normals_cpu,
                internal_edge_face_normals_cpu,
                internal_edge_crosses_cpu,
                internal_edge_normal_bs_cpu,
            ],
            device="cpu",
        )
        self.bullet_face_normals = face_normals_cpu.to(self.device)
        self.internal_edge_face_normals = internal_edge_face_normals_cpu.to(
            self.device
        )
        self.internal_edge_crosses = internal_edge_crosses_cpu.to(self.device)
        self.internal_edge_normal_bs = internal_edge_normal_bs_cpu.to(self.device)
        bvh_rank = build_bullet_bvh_rank(geometry)
        face_mesh_index = build_face_mesh_index(geometry)
        self.internal_edge_angles = wp.array(
            edge_angles, dtype=wp.vec3, device=self.device
        )
        self.internal_edge_flags = wp.array(
            edge_flags, dtype=wp.int32, device=self.device
        )
        self.bullet_bvh_rank = wp.array(
            bvh_rank, dtype=wp.int32, device=self.device
        )
        self.face_mesh_index = wp.array(
            face_mesh_index, dtype=wp.int32, device=self.device
        )
        # Keep the conventional mesh for AABB iteration. cuBQL currently does
        # not implement mesh_query_aabb, but is benchmarked for wheel rays.
        self.default = wp.Mesh(self.points, self.indices)
        self.cubql = wp.Mesh(self.points, self.indices, bvh_constructor="cubql")
        # Wheel rays in RocketSim are cast against the original Bullet-unit
        # CMF vertices.  Keep matching acceleration structures instead of
        # treating the scaled UU mesh as the numerical geometry authority.
        self.default_bt = wp.Mesh(self.points_bt, self.indices)
        self.cubql_bt = wp.Mesh(
            self.points_bt,
            self.indices,
            bvh_constructor="cubql",
        )

    def refit(self) -> None:
        self.default.refit()
        self.cubql.refit()
        self.default_bt.refit()
        self.cubql_bt.refit()


def build_face_mesh_index(geometry: ArenaGeometry) -> np.ndarray:
    """Map every combined face back to RocketSim's source collision body.

    RocketSim creates one static rigid body and therefore one persistent
    manifold per CMF file.  The combined Warp mesh is only an acceleration
    structure; collision-body ownership must remain explicit because Bullet's
    four-point manifold limit applies independently to each source CMF.
    """

    # RocketSim::Init consumes std::filesystem::directory_iterator order. The
    # frozen Windows Soccar custody enumerates the names lexicographically
    # (mesh_0, mesh_1, mesh_10, ..., mesh_9), whereas RivalSim's immutable
    # combined face IDs deliberately use numeric suffix order. Encode the
    # observed source-body order without reordering any protected geometry.
    body_order = {
        mesh.path: body_index
        for body_index, mesh in enumerate(
            sorted(geometry.meshes, key=lambda value: value.path.name.casefold())
        )
    }
    result = np.empty(geometry.triangle_count, dtype=np.int32)
    face_offset = 0
    for mesh in geometry.meshes:
        next_offset = face_offset + mesh.triangle_count
        result[face_offset:next_offset] = np.int32(body_order[mesh.path])
        face_offset = next_offset
    return result


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
    vertex_offset = 0
    edge_vertices = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    record_vertex_indices: list[int] = []
    record_edge_indices: list[int] = []
    record_output_indices: list[int] = []
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
                triangle_a = triangles[face]
                triangle_b = triangles[neighbor[0]]
                start_a, end_a, _other_a = edge_vertices[edge]
                key_start = vertex_keys[int(triangle_a[start_a])]
                key_end = vertex_keys[int(triangle_a[end_a])]
                keys_b = [vertex_keys[int(index)] for index in triangle_b]
                shared_b0 = keys_b.index(key_start)
                shared_b1 = keys_b.index(key_end)
                other_b = 3 - shared_b0 - shared_b1
                record_vertex_indices.extend(
                    [
                        vertex_offset + int(triangle_a[0]),
                        vertex_offset + int(triangle_a[1]),
                        vertex_offset + int(triangle_a[2]),
                        vertex_offset + int(triangle_b[shared_b0]),
                        vertex_offset + int(triangle_b[shared_b1]),
                        vertex_offset + int(triangle_b[other_b]),
                    ]
                )
                record_edge_indices.append(edge)
                record_output_indices.append((face_offset + face) * 3 + edge)
        face_offset += mesh.triangle_count
        vertex_offset += mesh.vertex_count

    if record_edge_indices:
        vertices_bt = np.ascontiguousarray(
            np.concatenate([mesh.vertices_bt for mesh in geometry.meshes]),
            dtype=np.float32,
        )
        record_count = len(record_edge_indices)
        record_angles_cpu = wp.empty(
            record_count, dtype=wp.float32, device="cpu"
        )
        record_flags_cpu = wp.empty(record_count, dtype=wp.int32, device="cpu")
        wp.launch(
            _build_bullet_internal_edge_info,
            dim=record_count,
            inputs=[
                wp.array(vertices_bt, dtype=wp.vec3, device="cpu"),
                wp.array(
                    np.asarray(record_vertex_indices, dtype=np.int32),
                    dtype=wp.int32,
                    device="cpu",
                ),
                wp.array(
                    np.asarray(record_edge_indices, dtype=np.int32),
                    dtype=wp.int32,
                    device="cpu",
                ),
                record_angles_cpu,
                record_flags_cpu,
            ],
            device="cpu",
        )
        flat_angles = angles.reshape(-1)
        generated_angles = record_angles_cpu.numpy()
        generated_flags = record_flags_cpu.numpy()
        for record, output_index in enumerate(record_output_indices):
            flat_angles[output_index] = generated_angles[record]
            flags[output_index // 3] |= generated_flags[record]
    return angles, flags


def build_bullet_bvh_rank(geometry: ArenaGeometry) -> np.ndarray:
    """Return each face's cache-friendly visit rank in Bullet's quantized BVH.

    ``btOptimizedBvh`` recursively partitions its leaf array in place, then
    partitions the resulting tree into at-most-2048-byte subtree headers.
    Quantized AABB queries visit those headers in their post-build insertion
    order, which is not necessarily the depth-first leaf order. RocketSim owns
    one BVH per CMF, so ranks restart at every source mesh and are offset only
    to make the combined array deterministic.
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
        subtree_headers: list[tuple[int, int]] = []
        _bullet_partition_leaves(
            quantized_min,
            quantized_max,
            faces,
            bvh_min,
            quantization,
            0,
            mesh.triangle_count,
            subtree_headers,
        )
        if not subtree_headers:
            subtree_headers.append((0, mesh.triangle_count))
        cache_friendly_faces = np.concatenate(
            [faces[start:end] for start, end in subtree_headers]
        )
        if (
            cache_friendly_faces.size != mesh.triangle_count
            or np.unique(cache_friendly_faces).size != mesh.triangle_count
        ):
            raise AssertionError("Bullet subtree headers did not partition all faces")
        for rank, face in enumerate(cache_friendly_faces):
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
    subtree_headers: list[tuple[int, int]],
) -> None:
    """Mirror Bullet's leaf partition and subtree-header insertion order."""

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
        subtree_headers,
    )
    _bullet_partition_leaves(
        quantized_min,
        quantized_max,
        faces,
        bvh_min,
        quantization,
        split,
        end,
        subtree_headers,
    )
    # Quantized nodes are 16 bytes. Bullet adds children as cache-friendly
    # subtree headers only when the current subtree exceeds 2048 bytes; header
    # insertion happens after both recursive builds, so a large right child
    # can intentionally be visited before a small left child.
    if (2 * count - 1) * 16 > 2048:
        if (2 * (split - start) - 1) * 16 <= 2048:
            subtree_headers.append((start, split))
        if (2 * (end - split) - 1) * 16 <= 2048:
            subtree_headers.append((split, end))


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
