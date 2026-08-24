"""Emit an exact per-tick RivalSim/RocketSim trace for one generated v0.2.2 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.dfh_breadth import (
    build_breadth_catalog,
    cases_to_controls,
    generate_breadth_cases,
)
from rivalsim.kernels.bullet_box_triangle import (
    _BULLET_BOX_TRIANGLE_CLOSEST,
    bullet_box_triangle_closest,
    bullet_box_triangle_penetration,
    bullet_manifold_replacement,
)
from rivalsim.kernels.vehicle import (
    CONTACT_BREAKING_THRESHOLD,
    HITBOX_MARGIN,
    _authority_input_quaternion_matrix,
    _bullet_inverse_transform_point,
    _bullet_quaternion_matrix,
    _point_segment_distance_sq,
    _triangle_obb_sat,
)
from rivalsim.static_world import StaticWorldSim
from rivalsim.v022_oracle_cache import (
    RocketSimAuthorityCache,
    build_authority_identity,
)
from rivalsim.vehicle_state import MAX_MESH_CANDIDATES_PER_CAR

PAIR_TRACE_LIMIT = 16


def _instrument_pair_trace_source() -> str:
    source = _BULLET_BOX_TRIANGLE_CLOSEST

    def replace_once(old: str, new: str) -> None:
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f"pair trace insertion point is not unique: {old[:60]!r}")
        source = source.replace(old, new, 1)

    replace_once(
        "    BtPairV3 axis = pair_make(0.0f, 1.0f, 0.0f);",
        """    trace_count = 0;
    BtPairV3 axis = pair_make(0.0f, 1.0f, 0.0f);""",
    )
    replace_once(
        """        const float delta = pair_dot(axis, value);
        if (delta > 0.0f""",
        """        const float delta = pair_dot(axis, value);
        const int trace_index = trace_count++;
        const int trace_output = trace_base + trace_index;
        if (trace_index < 16) {
            trace_axis.data[trace_output] = wp::vec_t<3, wp::float32>(axis.x, axis.y, axis.z);
            trace_direction_a.data[trace_output] = wp::vec_t<3, wp::float32>(
                direction_a.x, direction_a.y, direction_a.z);
            trace_point_a.data[trace_output] = wp::vec_t<3, wp::float32>(
                point_a.x, point_a.y, point_a.z);
            trace_point_b.data[trace_output] = wp::vec_t<3, wp::float32>(
                point_b.x, point_b.y, point_b.z);
            trace_w.data[trace_output] = wp::vec_t<3, wp::float32>(
                value.x, value.y, value.z);
            trace_delta.data[trace_output] = delta;
            trace_squared_distance.data[trace_output] = squared_distance;
            trace_cached_p1.data[trace_output] = wp::vec_t<3, wp::float32>(
                cached_p1.x, cached_p1.y, cached_p1.z);
            trace_cached_p2.data[trace_output] = wp::vec_t<3, wp::float32>(
                cached_p2.x, cached_p2.y, cached_p2.z);
            trace_cached_v.data[trace_output] = wp::vec_t<3, wp::float32>(
                cached_v.x, cached_v.y, cached_v.z);
            trace_simplex_count.data[trace_output] = simplex_count;
            trace_exit.data[trace_output] = 0;
        }
        if (delta > 0.0f""",
    )
    replace_once(
        """            degenerate_simplex = 10;
            check_simplex = 1;
            break;""",
        """            degenerate_simplex = 10;
            check_simplex = 1;
            if (trace_index < 16) trace_exit.data[trace_output] = 10;
            break;""",
    )
    replace_once(
        """            degenerate_simplex = 1;
            check_simplex = 1;
            break;""",
        """            degenerate_simplex = 1;
            check_simplex = 1;
            if (trace_index < 16) trace_exit.data[trace_output] = 1;
            break;""",
    )
    replace_once(
        """            degenerate_simplex = f0 <= 0.0f ? 2 : 11;
            check_simplex = 1;
            break;""",
        """            degenerate_simplex = f0 <= 0.0f ? 2 : 11;
            check_simplex = 1;
            if (trace_index < 16) trace_exit.data[trace_output] = degenerate_simplex;
            break;""",
    )
    replace_once(
        """        if (!pair_simplex_closest()) {
            degenerate_simplex = 3;
            check_simplex = 1;
            break;
        }
        const BtPairV3 new_axis = cached_v;""",
        """        if (!pair_simplex_closest()) {
            degenerate_simplex = 3;
            check_simplex = 1;
            if (trace_index < 16) {
                trace_simplex_count.data[trace_output] = simplex_count;
                trace_exit.data[trace_output] = 3;
            }
            break;
        }
        if (trace_index < 16) {
            trace_cached_p1.data[trace_output] = wp::vec_t<3, wp::float32>(
                cached_p1.x, cached_p1.y, cached_p1.z);
            trace_cached_p2.data[trace_output] = wp::vec_t<3, wp::float32>(
                cached_p2.x, cached_p2.y, cached_p2.z);
            trace_cached_v.data[trace_output] = wp::vec_t<3, wp::float32>(
                cached_v.x, cached_v.y, cached_v.z);
            trace_simplex_count.data[trace_output] = simplex_count;
        }
        const BtPairV3 new_axis = cached_v;""",
    )
    replace_once(
        """            degenerate_simplex = 6;
            check_simplex = 1;
            break;""",
        """            degenerate_simplex = 6;
            check_simplex = 1;
            if (trace_index < 16) trace_exit.data[trace_output] = 6;
            break;""",
    )
    replace_once(
        """            degenerate_simplex = 12;
            check_simplex = 1;
            break;""",
        """            degenerate_simplex = 12;
            check_simplex = 1;
            if (trace_index < 16) trace_exit.data[trace_output] = 12;
            break;""",
    )
    replace_once(
        """        if (simplex_count == 4) {
            degenerate_simplex = 13;
            break;
        }""",
        """        if (simplex_count == 4) {
            degenerate_simplex = 13;
            if (trace_index < 16) trace_exit.data[trace_output] = 13;
            break;
        }""",
    )
    return source


@wp.func_native(_instrument_pair_trace_source())
def bullet_box_triangle_closest_trace(
    body_origin_bt: wp.vec3,
    basis: wp.mat33,
    v0_bt: wp.vec3,
    v1_bt: wp.vec3,
    v2_bt: wp.vec3,
    point_a_bt: wp.ref[wp.vec3],
    point_b_bt: wp.ref[wp.vec3],
    normal_world: wp.ref[wp.vec3],
    distance_bt: wp.ref[wp.float32],
    valid: wp.ref[wp.int32],
    degenerate_status: wp.ref[wp.int32],
    trace_base: wp.int32,
    trace_count: wp.ref[wp.int32],
    trace_axis: wp.array(dtype=wp.vec3),
    trace_direction_a: wp.array(dtype=wp.vec3),
    trace_point_a: wp.array(dtype=wp.vec3),
    trace_point_b: wp.array(dtype=wp.vec3),
    trace_w: wp.array(dtype=wp.vec3),
    trace_delta: wp.array(dtype=wp.float32),
    trace_squared_distance: wp.array(dtype=wp.float32),
    trace_cached_p1: wp.array(dtype=wp.vec3),
    trace_cached_p2: wp.array(dtype=wp.vec3),
    trace_cached_v: wp.array(dtype=wp.vec3),
    trace_simplex_count: wp.array(dtype=wp.int32),
    trace_exit: wp.array(dtype=wp.int32),
): ...


@wp.kernel(enable_backward=False)
def probe_body_transform(
    quaternion: wp.quat,
    authority_input_basis: int,
    basis_rows: wp.array(dtype=wp.vec3),
):
    basis = _bullet_quaternion_matrix(quaternion)
    if authority_input_basis != 0:
        basis = _authority_input_quaternion_matrix(quaternion)
    basis_rows[0] = wp.vec3(basis[0, 0], basis[0, 1], basis[0, 2])
    basis_rows[1] = wp.vec3(basis[1, 0], basis[1, 1], basis[1, 2])
    basis_rows[2] = wp.vec3(basis[2, 0], basis[2, 1], basis[2, 2])


@wp.kernel(enable_backward=False)
def probe_pair_iterations(
    mesh_id: wp.uint64,
    vertices_bt: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    faces: wp.array(dtype=wp.int32),
    body_origin_bt: wp.vec3,
    quaternion: wp.quat,
    authority_input_basis: int,
    trace_counts: wp.array(dtype=wp.int32),
    trace_axis: wp.array(dtype=wp.vec3),
    trace_direction_a: wp.array(dtype=wp.vec3),
    trace_point_a: wp.array(dtype=wp.vec3),
    trace_point_b: wp.array(dtype=wp.vec3),
    trace_w: wp.array(dtype=wp.vec3),
    trace_delta: wp.array(dtype=wp.float32),
    trace_squared_distance: wp.array(dtype=wp.float32),
    trace_cached_p1: wp.array(dtype=wp.vec3),
    trace_cached_p2: wp.array(dtype=wp.vec3),
    trace_cached_v: wp.array(dtype=wp.vec3),
    trace_simplex_count: wp.array(dtype=wp.int32),
    trace_exit: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    face = faces[index]
    triangle_offset = face * 3
    v0_bt = vertices_bt[triangle_indices[triangle_offset]]
    v1_bt = vertices_bt[triangle_indices[triangle_offset + 1]]
    v2_bt = vertices_bt[triangle_indices[triangle_offset + 2]]
    point_a_bt = wp.vec3(0.0, 0.0, 0.0)
    point_b_bt = wp.vec3(0.0, 0.0, 0.0)
    normal_world = wp.vec3(0.0, 0.0, 0.0)
    distance_bt = wp.float32(0.0)
    valid = wp.int32(0)
    degenerate = wp.int32(0)
    count = wp.int32(0)
    basis = _bullet_quaternion_matrix(quaternion)
    if authority_input_basis != 0:
        basis = _authority_input_quaternion_matrix(quaternion)
    bullet_box_triangle_closest_trace(
        body_origin_bt,
        basis,
        v0_bt,
        v1_bt,
        v2_bt,
        point_a_bt,
        point_b_bt,
        normal_world,
        distance_bt,
        valid,
        degenerate,
        index * PAIR_TRACE_LIMIT,
        count,
        trace_axis,
        trace_direction_a,
        trace_point_a,
        trace_point_b,
        trace_w,
        trace_delta,
        trace_squared_distance,
        trace_cached_p1,
        trace_cached_p2,
        trace_cached_v,
        trace_simplex_count,
        trace_exit,
    )
    trace_counts[index] = count


@wp.kernel(enable_backward=False)
def probe_narrowphase(
    mesh_id: wp.uint64,
    vertices_bt: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    faces: wp.array(dtype=wp.int32),
    position_uu: wp.vec3,
    body_origin_bt: wp.vec3,
    quaternion: wp.quat,
    authority_input_basis: int,
    gjk_valid: wp.array(dtype=wp.int32),
    gjk_distance: wp.array(dtype=wp.float32),
    gjk_normal: wp.array(dtype=wp.vec3),
    gjk_point_a: wp.array(dtype=wp.vec3),
    gjk_point_b: wp.array(dtype=wp.vec3),
    sat_penetration: wp.array(dtype=wp.float32),
    sat_normal: wp.array(dtype=wp.vec3),
    deep_point_a: wp.array(dtype=wp.vec3),
    deep_point_b: wp.array(dtype=wp.vec3),
    deep_normal: wp.array(dtype=wp.vec3),
    deep_distance: wp.array(dtype=wp.float32),
    deep_edge_distance_sq: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    face = faces[index]
    v0 = wp.mesh_eval_position(mesh_id, face, 1.0, 0.0)
    v1 = wp.mesh_eval_position(mesh_id, face, 0.0, 1.0)
    v2 = wp.mesh_eval_position(mesh_id, face, 0.0, 0.0)
    triangle_offset = face * 3
    v0_bt = vertices_bt[triangle_indices[triangle_offset]]
    v1_bt = vertices_bt[triangle_indices[triangle_offset + 1]]
    v2_bt = vertices_bt[triangle_indices[triangle_offset + 2]]
    basis = _bullet_quaternion_matrix(quaternion)
    if authority_input_basis != 0:
        basis = _authority_input_quaternion_matrix(quaternion)
    pair_point_a_bt = wp.vec3(0.0, 0.0, 0.0)
    pair_point_b_bt = wp.vec3(0.0, 0.0, 0.0)
    pair_normal = wp.vec3(0.0, 0.0, 0.0)
    pair_distance_bt = wp.float32(0.0)
    pair_valid = wp.int32(0)
    pair_degenerate = wp.int32(0)
    bullet_box_triangle_closest(
        body_origin_bt,
        basis,
        v0_bt,
        v1_bt,
        v2_bt,
        pair_point_a_bt,
        pair_point_b_bt,
        pair_normal,
        pair_distance_bt,
        pair_valid,
        pair_degenerate,
    )
    sat = _triangle_obb_sat(
        v0,
        v1,
        v2,
        position_uu + wp.quat_rotate(quaternion, wp.vec3(13.8757, 0.0, 20.755)),
        quaternion,
    )
    gjk_valid[index] = pair_valid
    gjk_distance[index] = pair_distance_bt * 50.0
    gjk_normal[index] = pair_normal
    gjk_point_a[index] = pair_point_a_bt * 50.0
    gjk_point_b[index] = pair_point_b_bt * 50.0
    sat_penetration[index] = sat[3]
    sat_normal[index] = wp.vec3(sat[0], sat[1], sat[2])
    catch_degenerate = pair_degenerate != 0 and (pair_distance_bt + HITBOX_MARGIN * 0.02 < 0.01)
    if pair_valid == 0 or catch_degenerate:
        point_a_bt = wp.vec3(0.0, 0.0, 0.0)
        point_b_bt = wp.vec3(0.0, 0.0, 0.0)
        normal_world = wp.vec3(0.0, 0.0, 0.0)
        distance_bt = wp.float32(0.0)
        valid = wp.int32(0)
        bullet_box_triangle_penetration(
            body_origin_bt,
            basis,
            v0_bt,
            v1_bt,
            v2_bt,
            point_a_bt,
            point_b_bt,
            normal_world,
            distance_bt,
            valid,
        )
        if valid != 0 and (pair_valid == 0 or distance_bt < pair_distance_bt):
            deep_point_a[index] = point_a_bt * 50.0
            deep_point_b[index] = point_b_bt * 50.0
            deep_normal[index] = normal_world
            deep_distance[index] = distance_bt * 50.0
            deep_edge_distance_sq[index] = wp.vec3(
                _point_segment_distance_sq(point_b_bt * 50.0, v0, v1),
                _point_segment_distance_sq(point_b_bt * 50.0, v1, v2),
                _point_segment_distance_sq(point_b_bt * 50.0, v2, v0),
            )


@wp.kernel(enable_backward=False)
def probe_manifold_sequence(
    mesh_id: wp.uint64,
    vertices_bt: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    faces: wp.array(dtype=wp.int32),
    body_origin_bt: wp.vec3,
    quaternion: wp.quat,
    authority_input_basis: int,
    selected_valid: wp.array(dtype=wp.int32),
    selected_distance_bt: wp.array(dtype=wp.float32),
    selected_local_a: wp.array(dtype=wp.vec3),
    retained_faces: wp.array(dtype=wp.int32),
):
    basis = _bullet_quaternion_matrix(quaternion)
    if authority_input_basis != 0:
        basis = _authority_input_quaternion_matrix(quaternion)
    point0 = wp.vec3(0.0, 0.0, 0.0)
    point1 = wp.vec3(0.0, 0.0, 0.0)
    point2 = wp.vec3(0.0, 0.0, 0.0)
    point3 = wp.vec3(0.0, 0.0, 0.0)
    distance0 = wp.float32(0.0)
    distance1 = wp.float32(0.0)
    distance2 = wp.float32(0.0)
    distance3 = wp.float32(0.0)
    face0 = wp.int32(-1)
    face1 = wp.int32(-1)
    face2 = wp.int32(-1)
    face3 = wp.int32(-1)
    count = wp.int32(0)

    for step in range(MAX_MESH_CANDIDATES_PER_CAR):
        if step < faces.shape[0]:
            face = faces[step]
            triangle_offset = face * 3
            v0_bt = vertices_bt[triangle_indices[triangle_offset]]
            v1_bt = vertices_bt[triangle_indices[triangle_offset + 1]]
            v2_bt = vertices_bt[triangle_indices[triangle_offset + 2]]

            point_a_bt = wp.vec3(0.0, 0.0, 0.0)
            point_b_bt = wp.vec3(0.0, 0.0, 0.0)
            normal = wp.vec3(0.0, 0.0, 0.0)
            distance_bt = wp.float32(0.0)
            valid = wp.int32(0)
            degenerate = wp.int32(0)
            bullet_box_triangle_closest(
                body_origin_bt,
                basis,
                v0_bt,
                v1_bt,
                v2_bt,
                point_a_bt,
                point_b_bt,
                normal,
                distance_bt,
                valid,
                degenerate,
            )
            catch_degenerate = degenerate != 0 and (distance_bt + HITBOX_MARGIN * 0.02 < 0.01)
            if valid == 0 or catch_degenerate:
                epa_point_a_bt = wp.vec3(0.0, 0.0, 0.0)
                epa_point_b_bt = wp.vec3(0.0, 0.0, 0.0)
                epa_normal = wp.vec3(0.0, 0.0, 0.0)
                epa_distance_bt = wp.float32(0.0)
                epa_valid = wp.int32(0)
                bullet_box_triangle_penetration(
                    body_origin_bt,
                    basis,
                    v0_bt,
                    v1_bt,
                    v2_bt,
                    epa_point_a_bt,
                    epa_point_b_bt,
                    epa_normal,
                    epa_distance_bt,
                    epa_valid,
                )
                if epa_valid != 0 and (valid == 0 or epa_distance_bt < distance_bt):
                    point_a_bt = epa_point_a_bt
                    point_b_bt = epa_point_b_bt
                    normal = epa_normal
                    distance_bt = epa_distance_bt
                    valid = 1
            selected_valid[step] = valid
            selected_distance_bt[step] = distance_bt
            candidate = _bullet_inverse_transform_point(
                body_origin_bt,
                basis,
                point_a_bt,
            )
            selected_local_a[step] = candidate
            if valid != 0 and distance_bt * 50.0 < CONTACT_BREAKING_THRESHOLD:
                if count == 0:
                    point0 = candidate
                    distance0 = distance_bt * 50.0
                    face0 = face
                elif count == 1:
                    point1 = candidate
                    distance1 = distance_bt * 50.0
                    face1 = face
                elif count == 2:
                    point2 = candidate
                    distance2 = distance_bt * 50.0
                    face2 = face
                elif count == 3:
                    point3 = candidate
                    distance3 = distance_bt * 50.0
                    face3 = face
                else:
                    replacement = bullet_manifold_replacement(
                        candidate,
                        point0,
                        point1,
                        point2,
                        point3,
                        distance_bt * 50.0,
                        distance0,
                        distance1,
                        distance2,
                        distance3,
                    )
                    if replacement == 0:
                        point0 = candidate
                        distance0 = distance_bt * 50.0
                        face0 = face
                    elif replacement == 1:
                        point1 = candidate
                        distance1 = distance_bt * 50.0
                        face1 = face
                    elif replacement == 2:
                        point2 = candidate
                        distance2 = distance_bt * 50.0
                        face2 = face
                    else:
                        point3 = candidate
                        distance3 = distance_bt * 50.0
                        face3 = face
                count = wp.min(count + 1, 4)
            retained_faces[step * 4] = face0
            retained_faces[step * 4 + 1] = face1
            retained_faces[step * 4 + 2] = face2
            retained_faces[step * 4 + 3] = face3


def narrowphase_probe(
    meshes: WarpArenaMeshes,
    faces: np.ndarray,
    position: np.ndarray,
    body_origin_bt: np.ndarray,
    quaternion: np.ndarray,
    authority_input_basis: bool,
) -> list[dict[str, Any]]:
    if faces.size == 0:
        return []
    device = meshes.device
    face_array = wp.array(faces, dtype=wp.int32, device=device)
    valid = wp.empty(faces.size, dtype=wp.int32, device=device)
    distance = wp.empty(faces.size, dtype=wp.float32, device=device)
    normal = wp.empty(faces.size, dtype=wp.vec3, device=device)
    gjk_point_a = wp.zeros(faces.size, dtype=wp.vec3, device=device)
    gjk_point_b = wp.zeros(faces.size, dtype=wp.vec3, device=device)
    penetration = wp.empty(faces.size, dtype=wp.float32, device=device)
    sat_normal = wp.empty(faces.size, dtype=wp.vec3, device=device)
    deep_point_a = wp.zeros(faces.size, dtype=wp.vec3, device=device)
    deep_point_b = wp.zeros(faces.size, dtype=wp.vec3, device=device)
    deep_normal = wp.zeros(faces.size, dtype=wp.vec3, device=device)
    deep_distance = wp.zeros(faces.size, dtype=wp.float32, device=device)
    deep_edge_distance_sq = wp.zeros(faces.size, dtype=wp.vec3, device=device)
    wp.launch(
        probe_narrowphase,
        dim=faces.size,
        inputs=[
            meshes.default.id,
            meshes.points_bt,
            meshes.indices,
            face_array,
            wp.vec3(*position.astype(float).tolist()),
            wp.vec3(*body_origin_bt.astype(float).tolist()),
            wp.quat(*quaternion.astype(float).tolist()),
            int(authority_input_basis),
            valid,
            distance,
            normal,
            gjk_point_a,
            gjk_point_b,
            penetration,
            sat_normal,
            deep_point_a,
            deep_point_b,
            deep_normal,
            deep_distance,
            deep_edge_distance_sq,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    valid_host = np.asarray(valid.numpy(), dtype=np.int32)
    distance_host = np.asarray(distance.numpy(), dtype=np.float32)
    normal_host = np.asarray(normal.numpy(), dtype=np.float32)
    gjk_point_a_host = np.asarray(gjk_point_a.numpy(), dtype=np.float32)
    gjk_point_b_host = np.asarray(gjk_point_b.numpy(), dtype=np.float32)
    penetration_host = np.asarray(penetration.numpy(), dtype=np.float32)
    sat_normal_host = np.asarray(sat_normal.numpy(), dtype=np.float32)
    deep_point_a_host = np.asarray(deep_point_a.numpy(), dtype=np.float32)
    deep_point_b_host = np.asarray(deep_point_b.numpy(), dtype=np.float32)
    deep_normal_host = np.asarray(deep_normal.numpy(), dtype=np.float32)
    deep_distance_host = np.asarray(deep_distance.numpy(), dtype=np.float32)
    deep_edge_distance_sq_host = np.asarray(deep_edge_distance_sq.numpy(), dtype=np.float32)
    return [
        {
            "face": int(face),
            "gjk_valid": bool(valid_host[index]),
            "gjk_distance_uu": float(distance_host[index]),
            "gjk_normal": normal_host[index].astype(float).tolist(),
            "gjk_point_a": gjk_point_a_host[index].astype(float).tolist(),
            "gjk_point_b": gjk_point_b_host[index].astype(float).tolist(),
            "sat_penetration_uu": float(penetration_host[index]),
            "sat_local_normal": sat_normal_host[index].astype(float).tolist(),
            "deep_point_a": deep_point_a_host[index].astype(float).tolist(),
            "deep_point_b": deep_point_b_host[index].astype(float).tolist(),
            "deep_normal": deep_normal_host[index].astype(float).tolist(),
            "deep_distance_uu": float(deep_distance_host[index]),
            "deep_edge_distance_sq_uu2": deep_edge_distance_sq_host[index].astype(float).tolist(),
        }
        for index, face in enumerate(faces)
    ]


def pair_iteration_probe(
    meshes: WarpArenaMeshes,
    faces: np.ndarray,
    body_origin_bt: np.ndarray,
    quaternion: np.ndarray,
    authority_input_basis: bool,
) -> list[dict[str, Any]]:
    if faces.size == 0:
        return []
    device = meshes.device
    face_array = wp.array(faces, dtype=wp.int32, device=device)
    trace_size = faces.size * PAIR_TRACE_LIMIT
    counts = wp.zeros(faces.size, dtype=wp.int32, device=device)
    axis = wp.zeros(trace_size, dtype=wp.vec3, device=device)
    direction_a = wp.zeros(trace_size, dtype=wp.vec3, device=device)
    point_a = wp.zeros(trace_size, dtype=wp.vec3, device=device)
    point_b = wp.zeros(trace_size, dtype=wp.vec3, device=device)
    value_w = wp.zeros(trace_size, dtype=wp.vec3, device=device)
    delta = wp.zeros(trace_size, dtype=wp.float32, device=device)
    squared_distance = wp.zeros(trace_size, dtype=wp.float32, device=device)
    cached_p1 = wp.zeros(trace_size, dtype=wp.vec3, device=device)
    cached_p2 = wp.zeros(trace_size, dtype=wp.vec3, device=device)
    cached_v = wp.zeros(trace_size, dtype=wp.vec3, device=device)
    simplex_count = wp.zeros(trace_size, dtype=wp.int32, device=device)
    exit_reason = wp.zeros(trace_size, dtype=wp.int32, device=device)
    wp.launch(
        probe_pair_iterations,
        dim=faces.size,
        inputs=[
            meshes.default.id,
            meshes.points_bt,
            meshes.indices,
            face_array,
            wp.vec3(*body_origin_bt.astype(float).tolist()),
            wp.quat(*quaternion.astype(float).tolist()),
            int(authority_input_basis),
            counts,
            axis,
            direction_a,
            point_a,
            point_b,
            value_w,
            delta,
            squared_distance,
            cached_p1,
            cached_p2,
            cached_v,
            simplex_count,
            exit_reason,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    counts_host = np.asarray(counts.numpy(), dtype=np.int32)
    vec_fields = {
        "axis": np.asarray(axis.numpy(), dtype=np.float32),
        "direction_a": np.asarray(direction_a.numpy(), dtype=np.float32),
        "point_a": np.asarray(point_a.numpy(), dtype=np.float32),
        "point_b": np.asarray(point_b.numpy(), dtype=np.float32),
        "w": np.asarray(value_w.numpy(), dtype=np.float32),
        "cached_p1": np.asarray(cached_p1.numpy(), dtype=np.float32),
        "cached_p2": np.asarray(cached_p2.numpy(), dtype=np.float32),
        "cached_v": np.asarray(cached_v.numpy(), dtype=np.float32),
    }
    scalar_fields = {
        "delta": np.asarray(delta.numpy(), dtype=np.float32),
        "squared_distance_before": np.asarray(squared_distance.numpy(), dtype=np.float32),
    }
    simplex_host = np.asarray(simplex_count.numpy(), dtype=np.int32)
    exit_host = np.asarray(exit_reason.numpy(), dtype=np.int32)
    result = []
    for face_index, face in enumerate(faces):
        count = min(int(counts_host[face_index]), PAIR_TRACE_LIMIT)
        base = face_index * PAIR_TRACE_LIMIT
        steps = []
        for step_index in range(count):
            offset = base + step_index
            step = {
                "iteration": step_index,
                **{
                    name: values[offset].astype(float).tolist()
                    for name, values in vec_fields.items()
                },
                **{name: float(values[offset]) for name, values in scalar_fields.items()},
                "simplex_count_after": int(simplex_host[offset]),
                "exit_code": int(exit_host[offset]),
            }
            steps.append(step)
        result.append(
            {
                "face": int(face),
                "iteration_count": int(counts_host[face_index]),
                "truncated": bool(counts_host[face_index] > PAIR_TRACE_LIMIT),
                "steps": steps,
            }
        )
    return result


def manifold_sequence_probe(
    meshes: WarpArenaMeshes,
    faces: np.ndarray,
    body_origin_bt: np.ndarray,
    quaternion: np.ndarray,
    authority_input_basis: bool,
) -> list[dict[str, Any]]:
    if faces.size == 0:
        return []
    device = meshes.device
    face_array = wp.array(faces, dtype=wp.int32, device=device)
    valid = wp.zeros(faces.size, dtype=wp.int32, device=device)
    distance = wp.zeros(faces.size, dtype=wp.float32, device=device)
    local_a = wp.zeros(faces.size, dtype=wp.vec3, device=device)
    retained = wp.full(faces.size * 4, -1, dtype=wp.int32, device=device)
    wp.launch(
        probe_manifold_sequence,
        dim=1,
        inputs=[
            meshes.default.id,
            meshes.points_bt,
            meshes.indices,
            face_array,
            wp.vec3(*body_origin_bt.astype(float).tolist()),
            wp.quat(*quaternion.astype(float).tolist()),
            int(authority_input_basis),
            valid,
            distance,
            local_a,
            retained,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    valid_host = np.asarray(valid.numpy(), dtype=np.int32)
    distance_host = np.asarray(distance.numpy(), dtype=np.float32)
    local_host = np.asarray(local_a.numpy(), dtype=np.float32)
    retained_host = np.asarray(retained.numpy(), dtype=np.int32).reshape(-1, 4)
    return [
        {
            "face": int(face),
            "valid": bool(valid_host[index]),
            "distance_bt": float(distance_host[index]),
            "local_a": local_host[index].astype(float).tolist(),
            "retained_faces": [int(value) for value in retained_host[index] if value >= 0],
        }
        for index, face in enumerate(faces)
    ]


def body_transform_probe(
    meshes: WarpArenaMeshes,
    body_origin_bt: np.ndarray,
    quaternion: np.ndarray,
    authority_input_basis: bool,
) -> dict[str, Any]:
    basis_rows = wp.empty(3, dtype=wp.vec3, device=meshes.device)
    wp.launch(
        probe_body_transform,
        dim=1,
        inputs=[
            wp.quat(*quaternion.astype(float).tolist()),
            int(authority_input_basis),
            basis_rows,
        ],
        device=meshes.device,
    )
    wp.synchronize_device(meshes.device)
    basis = np.asarray(basis_rows.numpy(), dtype=np.float32)
    body_origin_bt = np.asarray(body_origin_bt, dtype=np.float32)
    child_offset_bt = np.asarray((0.277513981, 0.0, 0.415099978), dtype=np.float32)

    def rn(value: float | np.float32) -> np.float32:
        return np.float32(value)

    rotated_offset = np.empty(3, dtype=np.float32)
    for row in range(3):
        first = rn(rn(basis[row, 0] * child_offset_bt[0]) + rn(0.0))
        rotated_offset[row] = rn(first + rn(basis[row, 2] * child_offset_bt[2]))
    center_bt = np.asarray(
        [rn(body_origin_bt[index] + rotated_offset[index]) for index in range(3)],
        dtype=np.float32,
    )
    transform_origin_bt = np.asarray(
        [rn(center_bt[index] - rn(center_bt[index] * rn(0.5))) for index in range(3)],
        dtype=np.float32,
    )
    return {
        "body_origin_bt": body_origin_bt.astype(float).tolist(),
        "basis_rows": basis.astype(float).tolist(),
        "child_center_bt": center_bt.astype(float).tolist(),
        "position_offset_bt": (center_bt * np.float32(0.5)).astype(float).tolist(),
        "transform_a_origin_bt": transform_origin_bt.astype(float).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--oracle-cache-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ticks <= 0:
        raise ValueError("ticks must be positive")
    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    catalog = build_breadth_catalog(geometry)
    cases = generate_breadth_cases(catalog)
    selected = next(
        ((index, item) for index, item in enumerate(cases) if item.case_id == args.case_id),
        None,
    )
    if selected is None:
        raise ValueError(f"unknown generated case: {args.case_id}")
    case_index, case = selected
    controls = cases_to_controls([case])
    identity = build_authority_identity(geometry, cases)
    authority_cache = RocketSimAuthorityCache(
        args.oracle_cache_root,
        identity,
        cases,
    )
    cached = authority_cache.load((case_index,))
    authoritative = cached.authoritative_snapshot
    meshes = WarpArenaMeshes(geometry, args.device)
    sim = StaticWorldSim(
        1,
        args.collision_dir,
        variant="B3",
        device=args.device,
        initial=authoritative.copy(),
        geometry=geometry,
        meshes=meshes,
    )
    sim.set_controls(controls)
    trace: list[dict[str, Any]] = []
    for tick in range(1, args.ticks + 1):
        # Warp CPU arrays expose NumPy views, while CUDA snapshots are copies.
        # Keep the diagnostic pre-step state device-invariant.
        pre_state = sim.snapshot().copy()
        pre_vehicle = sim.vehicle_snapshot()
        if tick == 1:
            pre_body_origin_bt = np.asarray(
                pre_state.car_pos[0, 0] * np.float32(0.02),
                dtype=np.float32,
            )
        else:
            pre_body_origin_bt = np.asarray(
                pre_vehicle.rigid_position_bt[0],
                dtype=np.float32,
            )
        sim.step(1)
        state = sim.snapshot()
        vehicle = sim.vehicle_snapshot()
        mesh_candidate_faces = np.asarray(
            sim.vehicle.mesh_candidate_face.numpy(), dtype=np.int32
        ).reshape(sim.state.car_count, MAX_MESH_CANDIDATES_PER_CAR)
        reference = cached.frame(tick)
        count = int(vehicle.contact_count[0])
        has_world_contact = bool(
            np.linalg.norm(vehicle.world_contact_normal[0].astype(np.float64)) > 0.5
        )
        retained_face_count = int(vehicle.mesh_candidate_count[0])
        retained_faces = np.ascontiguousarray(
            mesh_candidate_faces[0, :retained_face_count], dtype=np.int32
        )
        trace.append(
            {
                "tick": tick,
                "rivalsim": {
                    "position": state.car_pos[0, 0].astype(float).tolist(),
                    "velocity": state.car_vel[0, 0].astype(float).tolist(),
                    "quaternion": state.car_quat[0, 0].astype(float).tolist(),
                    "angular_velocity": state.car_ang_vel[0, 0].astype(float).tolist(),
                    "solver_velocity_before_external_force": vehicle.solver_velocity[0]
                    .astype(float)
                    .tolist(),
                    "rigid_velocity_bt_before_external_force": vehicle.rigid_velocity_bt[0]
                    .astype(float)
                    .tolist(),
                    "solver_angular_velocity_before_external_force": (
                        vehicle.solver_angular_velocity[0].astype(float).tolist()
                    ),
                    "auto_roll_acceleration": vehicle.auto_roll_acceleration[0]
                    .astype(float)
                    .tolist(),
                    "auto_roll_angular_acceleration": vehicle.auto_roll_angular_acceleration[
                        0
                    ]
                    .astype(float)
                    .tolist(),
                    "total_force_bt": vehicle.total_force_bt[0].astype(float).tolist(),
                    "total_torque_bt": vehicle.total_torque_bt[0].astype(float).tolist(),
                    "inverse_inertia_world": vehicle.inverse_inertia_world[0]
                    .astype(float)
                    .tolist(),
                    "on_ground": bool(state.on_ground[0, 0]),
                    "world_contact": has_world_contact,
                    "world_contact_normal": vehicle.world_contact_normal[0].astype(float).tolist(),
                    "aabb_candidate_count": int(vehicle.candidate_count[0]),
                    "mesh_candidate_count": int(vehicle.mesh_candidate_count[0]),
                    "mesh_candidate_overflow": int(vehicle.mesh_candidate_overflow[0]),
                    "contact_overflow": int(vehicle.contact_overflow[0]),
                    "mesh_candidate_faces": mesh_candidate_faces[0, :retained_face_count]
                    .astype(int)
                    .tolist(),
                    "pre_step_body_transform": body_transform_probe(
                        meshes,
                        pre_body_origin_bt,
                        pre_state.car_quat[0, 0],
                        tick == 1,
                    ),
                    "pre_step_narrowphase": narrowphase_probe(
                        meshes,
                        retained_faces,
                        pre_state.car_pos[0, 0],
                        pre_body_origin_bt,
                        pre_state.car_quat[0, 0],
                        tick == 1,
                    ),
                    "pre_step_pair_iterations": pair_iteration_probe(
                        meshes,
                        retained_faces,
                        pre_body_origin_bt,
                        pre_state.car_quat[0, 0],
                        tick == 1,
                    ),
                    "pre_step_manifold_sequence": manifold_sequence_probe(
                        meshes,
                        retained_faces,
                        pre_body_origin_bt,
                        pre_state.car_quat[0, 0],
                        tick == 1,
                    ),
                    "contacts": [
                        {
                            "face": int(vehicle.contact_face[0, index]),
                            "mesh": int(vehicle.contact_mesh[0, index]),
                            "point": vehicle.contact_point[0, index].astype(float).tolist(),
                            "local_a": vehicle.contact_local_a[0, index].astype(float).tolist(),
                            "point_b": vehicle.contact_point_b[0, index].astype(float).tolist(),
                            "normal": vehicle.contact_normal[0, index].astype(float).tolist(),
                            "distance": float(vehicle.contact_distance[0, index]),
                            "distance_bt": float(
                                vehicle.contact_distance_bt[0, index]
                            ),
                            "lifetime": int(vehicle.contact_lifetime[0, index]),
                            "normal_impulse": float(vehicle.contact_normal_impulse[0, index]),
                            "tangent_impulse": float(vehicle.contact_tangent_impulse[0, index]),
                            "push_impulse": float(vehicle.contact_push_impulse[0, index]),
                            "normal_jacobian": float(
                                vehicle.contact_normal_jacobian[0, index]
                            ),
                            "tangent_jacobian": float(
                                vehicle.contact_tangent_jacobian[0, index]
                            ),
                            "normal_rhs": float(vehicle.contact_normal_rhs[0, index]),
                            "tangent_rhs": float(vehicle.contact_tangent_rhs[0, index]),
                            "push_rhs": float(vehicle.contact_push_rhs[0, index]),
                        }
                        for index in range(count)
                    ],
                    "wheels": [
                        {
                            "contact": bool(vehicle.wheel_contact[0, index]),
                            "world_contact": bool(vehicle.wheel_world_contact[0, index]),
                            "face": int(vehicle.wheel_hit_face[0, index]),
                            "ray_start": vehicle.wheel_ray_start[0, index]
                            .astype(float)
                            .tolist(),
                            "direction": vehicle.wheel_direction[0, index]
                            .astype(float)
                            .tolist(),
                            "point": vehicle.wheel_hit_point[0, index].astype(float).tolist(),
                            "point_bt": vehicle.wheel_hit_point_bt[0, index]
                            .astype(float)
                            .tolist(),
                            "normal": vehicle.wheel_hit_normal[0, index].astype(float).tolist(),
                            "distance": float(vehicle.wheel_hit_distance[0, index]),
                            "suspension_length": float(
                                vehicle.suspension_length[0, index]
                            ),
                            "suspension_velocity": float(
                                vehicle.suspension_velocity[0, index]
                            ),
                            "suspension_clipped_factor": float(
                                vehicle.suspension_clipped_factor[0, index]
                            ),
                            "suspension_force": float(vehicle.suspension_force[0, index]),
                            "suspension_pushback": float(
                                vehicle.suspension_pushback[0, index]
                            ),
                            "suspension_force_bt": float(
                                vehicle.suspension_force_bt[0, index]
                            ),
                            "suspension_pushback_bt": float(
                                vehicle.suspension_pushback_bt[0, index]
                            ),
                            "debug_ray_from_bt": (
                                vehicle.debug_wheel_ray_from_bt[0, index]
                                .astype(float)
                                .tolist()
                            ),
                            "debug_ray_to_bt": (
                                vehicle.debug_wheel_ray_to_bt[0, index]
                                .astype(float)
                                .tolist()
                            ),
                            "debug_ray_fraction": float(
                                vehicle.debug_wheel_ray_fraction[0, index]
                            ),
                            "debug_linear_after_suspension_bt": (
                                vehicle.debug_wheel_linear_bt[0, index]
                                .astype(float)
                                .tolist()
                            ),
                            "debug_angular_after_suspension": (
                                vehicle.debug_wheel_angular[0, index]
                                .astype(float)
                                .tolist()
                            ),
                            "axle": vehicle.wheel_axle[0, index].astype(float).tolist(),
                            "forward": vehicle.wheel_forward[0, index]
                            .astype(float)
                            .tolist(),
                            "friction_impulse": vehicle.wheel_friction_impulse[0, index]
                            .astype(float)
                            .tolist(),
                            "friction_impulse_bt": vehicle.wheel_friction_impulse_bt[0, index]
                            .astype(float)
                            .tolist(),
                            "friction_relative_bt": vehicle.wheel_friction_relative_bt[0, index]
                            .astype(float)
                            .tolist(),
                            "side_impulse": float(vehicle.side_impulse[0, index]),
                            "rolling_impulse": float(vehicle.rolling_impulse[0, index]),
                            "engine_acceleration": float(
                                vehicle.engine_acceleration[0, index]
                            ),
                            "brake_acceleration": float(
                                vehicle.brake_acceleration[0, index]
                            ),
                            "steer_angle": float(vehicle.steer_angle[0, index]),
                            "lateral_friction": float(
                                vehicle.lateral_friction[0, index]
                            ),
                            "longitudinal_friction": float(
                                vehicle.longitudinal_friction[0, index]
                            ),
                        }
                        for index in range(4)
                    ],
                },
                "rocketsim": {
                    "position": reference.car_pos[0].astype(float).tolist(),
                    "velocity": reference.car_vel[0].astype(float).tolist(),
                    "matrix": reference.car_matrix[0].astype(float).tolist(),
                    "angular_velocity": reference.car_ang_vel[0].astype(float).tolist(),
                    "on_ground": bool(reference.on_ground[0]),
                    "world_contact": bool(reference.has_world_contact[0]),
                    "world_contact_normal": reference.world_contact_normal[0]
                    .astype(float)
                    .tolist(),
                    "wheels": [bool(value) for value in reference.wheel_contacts[0]],
                },
                "deltas": {
                    "position_uu": float(
                        np.linalg.norm(state.car_pos[0, 0] - reference.car_pos[0])
                    ),
                    "linear_velocity_uu_per_s": float(
                        np.linalg.norm(state.car_vel[0, 0] - reference.car_vel[0])
                    ),
                    "angular_velocity_rad_per_s": float(
                        np.linalg.norm(state.car_ang_vel[0, 0] - reference.car_ang_vel[0])
                    ),
                },
            }
        )
    result = {
        "case": {
            "case_id": case.case_id,
            "case_kind": case.case_kind,
            "family": case.family,
            "mode": case.mode,
            "mesh_file": case.mesh_file,
            "target_face": case.target_face,
            "target_neighbor_face": case.target_neighbor_face,
            "target_edge": case.target_edge,
            "edge_class": case.edge_class,
            "position": authoritative.car_pos[0, 0].astype(float).tolist(),
            "velocity": authoritative.car_vel[0, 0].astype(float).tolist(),
            "quaternion": authoritative.car_quat[0, 0].astype(float).tolist(),
            "generated_quaternion": case.quaternion.astype(float).tolist(),
            "angular_velocity": authoritative.car_ang_vel[0, 0].astype(float).tolist(),
            "controls": list(case.controls),
        },
        "trace": trace,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
        print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
