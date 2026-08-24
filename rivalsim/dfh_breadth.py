"""Deterministic DFH static-world breadth corpus for RivalSim v0.2.2.

The generator enumerates source geometry, then assigns a bounded representative
state to every triangle/contact family and every directed shared edge.  It does
not contain reference outputs or runtime simulator exceptions.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from rivalsim.arena import ArenaGeometry, build_internal_edge_data
from rivalsim.controls import ControlBatch
from rivalsim.math import matrix_to_quat
from rivalsim.state import StateSnapshot

GENERATOR_SCHEMA_VERSION = 3
GENERATOR_SEED = 20260823
LOCAL_HORIZONS = (1, 4, 8, 12)
HITBOX_COLLISION_HALF = np.asarray(
    (60.18645668029785, 43.28265380859375, 19.26250457763672),
    dtype=np.float32,
)
HITBOX_OFFSET = np.asarray((13.8757, 0.0, 20.755), dtype=np.float32)
FRONT_RIGHT_CONNECTION = np.asarray((51.25, 25.9, 20.755), dtype=np.float32)
REAR_RIGHT_CONNECTION = np.asarray((-33.75, 29.5, 20.755), dtype=np.float32)

ContactPath = Literal["chassis", "wheel"]
EdgeClass = Literal["planar", "convex", "concave"]


@dataclass(frozen=True, slots=True)
class TriangleRecord:
    global_face: int
    mesh_index: int
    mesh_file: str
    local_face: int
    vertices: np.ndarray
    centroid: np.ndarray
    normal: np.ndarray
    forward: np.ndarray
    area_uu2: float
    minimum_altitude_uu: float
    region_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectedEdgeRecord:
    directed_index: int
    mesh_index: int
    mesh_file: str
    global_face: int
    local_face: int
    local_edge: int
    neighbor_face: int
    edge_class: EdgeClass
    stored_angle_rad: float
    normal_swap: bool
    start: np.ndarray
    end: np.ndarray
    midpoint: np.ndarray
    direction: np.ndarray
    face_normal: np.ndarray
    neighbor_normal: np.ndarray
    face_interior_direction: np.ndarray
    region_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BreadthCase:
    case_id: str
    case_kind: Literal["triangle_face", "shared_directed_edge", "analytic_plane"]
    family: str
    mode: str
    contact_path: ContactPath
    mesh_index: int | None
    mesh_file: str | None
    target_face: int | None
    target_neighbor_face: int | None
    target_edge: int | None
    edge_class: EdgeClass | None
    analytic_plane: str | None
    expected_plane_face: int | None
    target_point: np.ndarray
    target_normal: np.ndarray
    edge_start: np.ndarray | None
    edge_end: np.ndarray | None
    region_labels: tuple[str, ...]
    position: np.ndarray
    quaternion: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    controls: tuple[float, float, float, float, float, int, int, int]


@dataclass(frozen=True, slots=True)
class BreadthCatalog:
    triangles: tuple[TriangleRecord, ...]
    directed_edges: tuple[DirectedEdgeRecord, ...]
    topology_counts: dict[str, int]
    per_mesh_topology: tuple[dict[str, int | str], ...]


def build_breadth_catalog(geometry: ArenaGeometry) -> BreadthCatalog:
    """Classify every source triangle and reproduce every directed shared edge."""

    triangles: list[TriangleRecord] = []
    face_offset = 0
    for mesh_index, mesh in enumerate(geometry.meshes):
        vertices = np.asarray(mesh.vertices_uu, dtype=np.float32)
        for local_face, indices in enumerate(np.asarray(mesh.triangles, dtype=np.int32)):
            triangle = np.asarray(vertices[indices], dtype=np.float32)
            edge_vectors = np.asarray(
                (triangle[1] - triangle[0], triangle[2] - triangle[1], triangle[0] - triangle[2]),
                dtype=np.float32,
            )
            edge_lengths = np.linalg.norm(edge_vectors, axis=1)
            cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            double_area = float(np.linalg.norm(cross))
            if double_area <= 0.0:
                raise ValueError(f"degenerate source triangle {mesh.path.name}:{local_face}")
            normal = _normalize(cross)
            longest = int(np.argmax(edge_lengths))
            forward = _normalize(edge_vectors[longest])
            centroid = np.asarray(np.mean(triangle, axis=0), dtype=np.float32)
            minimum_altitude = min(
                double_area / max(float(length), 1e-20) for length in edge_lengths
            )
            triangles.append(
                TriangleRecord(
                    global_face=face_offset + local_face,
                    mesh_index=mesh_index,
                    mesh_file=mesh.path.name,
                    local_face=local_face,
                    vertices=triangle,
                    centroid=centroid,
                    normal=normal,
                    forward=forward,
                    area_uu2=0.5 * double_area,
                    minimum_altitude_uu=minimum_altitude,
                    region_labels=_region_labels(
                        centroid, normal, 0.5 * double_area, minimum_altitude
                    ),
                )
            )
        face_offset += mesh.triangle_count

    angles, flags = build_internal_edge_data(geometry)
    directed_edges: list[DirectedEdgeRecord] = []
    per_mesh_topology: list[dict[str, int | str]] = []
    topology = Counter[str]()
    face_offset = 0
    edge_vertices = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for mesh_index, mesh in enumerate(geometry.meshes):
        vertices_bt = np.asarray(mesh.vertices_bt, dtype=np.float32)
        vertices_uu = np.asarray(mesh.vertices_uu, dtype=np.float32)
        mesh_triangles = np.asarray(mesh.triangles, dtype=np.int32)
        vertex_keys = [tuple(vertex.view(np.uint32).tolist()) for vertex in vertices_bt]
        edge_map: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, int]]] = {}
        for local_face, triangle in enumerate(mesh_triangles):
            for local_edge, (start_index, end_index, _other_index) in enumerate(edge_vertices):
                key_a = vertex_keys[int(triangle[start_index])]
                key_b = vertex_keys[int(triangle[end_index])]
                key = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
                edge_map.setdefault(key, []).append((local_face, local_edge))

        mesh_counts = Counter[str]()
        for local_face, triangle in enumerate(mesh_triangles):
            for local_edge, (start_index, end_index, other_index) in enumerate(edge_vertices):
                angle = float(angles[face_offset + local_face, local_edge])
                if angle >= 2.0 * np.pi - 1e-5:
                    continue
                key_a = vertex_keys[int(triangle[start_index])]
                key_b = vertex_keys[int(triangle[end_index])]
                key = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
                neighbor_local = next(face for face, _edge in edge_map[key] if face != local_face)
                edge_class: EdgeClass
                if abs(angle) <= 1e-6:
                    edge_class = "planar"
                elif int(flags[face_offset + local_face]) & (1 << local_edge):
                    edge_class = "convex"
                else:
                    edge_class = "concave"
                start = np.asarray(vertices_uu[int(triangle[start_index])], dtype=np.float32)
                end = np.asarray(vertices_uu[int(triangle[end_index])], dtype=np.float32)
                midpoint = np.asarray((start + end) * np.float32(0.5), dtype=np.float32)
                direction = _normalize(end - start)
                face_record = triangles[face_offset + local_face]
                neighbor_record = triangles[face_offset + neighbor_local]
                interior = vertices_uu[int(triangle[other_index])] - midpoint
                interior = interior - direction * float(np.dot(interior, direction))
                interior = _normalize(interior)
                directed_edges.append(
                    DirectedEdgeRecord(
                        directed_index=len(directed_edges),
                        mesh_index=mesh_index,
                        mesh_file=mesh.path.name,
                        global_face=face_offset + local_face,
                        local_face=local_face,
                        local_edge=local_edge,
                        neighbor_face=face_offset + neighbor_local,
                        edge_class=edge_class,
                        stored_angle_rad=angle,
                        normal_swap=bool(
                            int(flags[face_offset + local_face]) & (1 << (local_edge + 3))
                        ),
                        start=start,
                        end=end,
                        midpoint=midpoint,
                        direction=direction,
                        face_normal=face_record.normal,
                        neighbor_normal=neighbor_record.normal,
                        face_interior_direction=interior,
                        region_labels=tuple(
                            sorted(set(face_record.region_labels + neighbor_record.region_labels))
                        ),
                    )
                )
                topology["shared_directed_edges"] += 1
                topology[f"{edge_class}_directed_edges"] += 1
                mesh_counts["shared_directed_edges"] += 1
                mesh_counts[f"{edge_class}_directed_edges"] += 1
        per_mesh_topology.append(
            {
                "file": mesh.path.name,
                "shared_directed_edges": mesh_counts["shared_directed_edges"],
                "planar_directed_edges": mesh_counts["planar_directed_edges"],
                "convex_directed_edges": mesh_counts["convex_directed_edges"],
                "concave_directed_edges": mesh_counts["concave_directed_edges"],
            }
        )
        face_offset += mesh.triangle_count

    return BreadthCatalog(
        triangles=tuple(triangles),
        directed_edges=tuple(directed_edges),
        topology_counts=dict(sorted(topology.items())),
        per_mesh_topology=tuple(per_mesh_topology),
    )


def generate_breadth_cases(catalog: BreadthCatalog) -> tuple[BreadthCase, ...]:
    """Generate the complete, stable v0.2.2 case order."""

    cases: list[BreadthCase] = []
    for triangle in catalog.triangles:
        cases.append(_triangle_chassis_case(triangle))
        cases.append(_triangle_wheel_case(triangle))
    for edge in catalog.directed_edges:
        cases.append(_edge_case(edge))
    cases.extend(_analytic_plane_cases())
    identifiers = [case.case_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("breadth generator emitted duplicate case identifiers")
    return tuple(cases)


def corpus_sha256(cases: tuple[BreadthCase, ...]) -> str:
    """Hash the complete generated corpus, including exact float32 payloads."""

    digest = hashlib.sha256()
    digest.update(f"v022-generator-{GENERATOR_SCHEMA_VERSION}-{GENERATOR_SEED}".encode())
    for case in cases:
        digest.update(case.case_id.encode())
        digest.update(case.family.encode())
        digest.update(case.mode.encode())
        for array in (
            case.target_point,
            case.target_normal,
            case.position,
            case.quaternion,
            case.velocity,
            case.angular_velocity,
        ):
            digest.update(np.asarray(array, dtype="<f4").tobytes())
        digest.update(np.asarray(case.controls, dtype="<f4").tobytes())
        digest.update(
            np.asarray(
                (
                    -1 if case.target_face is None else case.target_face,
                    -1 if case.target_neighbor_face is None else case.target_neighbor_face,
                    -1 if case.target_edge is None else case.target_edge,
                ),
                dtype="<i4",
            ).tobytes()
        )
    return digest.hexdigest().upper()


def selection_sha256(indices: tuple[int, ...], corpus_hash: str) -> str:
    """Bind an ordered corpus selection to the frozen complete-corpus hash."""

    digest = hashlib.sha256(bytes.fromhex(corpus_hash))
    digest.update(np.asarray(indices, dtype="<i4").tobytes())
    return digest.hexdigest().upper()


def generator_config() -> dict[str, object]:
    """Return the explicit non-source inputs that define the v0.2.2 corpus."""

    return {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "seed": GENERATOR_SEED,
        "local_horizons_ticks": list(LOCAL_HORIZONS),
        "hitbox_collision_half_uu": HITBOX_COLLISION_HALF.astype(float).tolist(),
        "hitbox_offset_uu": HITBOX_OFFSET.astype(float).tolist(),
        "front_right_connection_uu": FRONT_RIGHT_CONNECTION.astype(float).tolist(),
        "rear_right_connection_uu": REAR_RIGHT_CONNECTION.astype(float).tolist(),
        "case_order": [
            "two triangle cases per global face: chassis then wheel",
            "all directed shared-edge cases in catalog order",
            "twenty analytic-plane cases",
        ],
    }


def cases_to_state(cases: tuple[BreadthCase, ...] | list[BreadthCase]) -> StateSnapshot:
    state = StateSnapshot.empty(len(cases))
    for index, case in enumerate(cases):
        state.car_pos[index, 0] = case.position
        state.car_quat[index, 0] = case.quaternion
        state.car_vel[index, 0] = case.velocity
        state.car_ang_vel[index, 0] = case.angular_velocity
        state.boost[index, 0] = 100.0
        # The second RivalSim car is deliberately inert and well away from all
        # tested contacts. RocketSim's batch arena contains only the first car.
        state.car_pos[index, 1] = (0.0, 0.0, 1500.0)
        state.boost[index, 1] = 100.0
        throttle, steer, pitch, yaw, roll, jump, boost, handbrake = case.controls
        state.prev_throttle[index, 0] = throttle
        state.prev_steer[index, 0] = steer
        state.prev_pitch[index, 0] = pitch
        state.prev_yaw[index, 0] = yaw
        state.prev_roll[index, 0] = roll
        state.prev_jump[index, 0] = jump
        state.prev_boost[index, 0] = boost
        state.prev_handbrake[index, 0] = handbrake
    state.validate()
    return state


def cases_to_controls(cases: tuple[BreadthCase, ...] | list[BreadthCase]) -> ControlBatch:
    controls = ControlBatch.zeros(len(cases))
    for index, case in enumerate(cases):
        throttle, steer, pitch, yaw, roll, jump, boost, handbrake = case.controls
        controls.throttle[index, 0] = throttle
        controls.steer[index, 0] = steer
        controls.pitch[index, 0] = pitch
        controls.yaw[index, 0] = yaw
        controls.roll[index, 0] = roll
        controls.jump[index, 0] = jump
        controls.boost[index, 0] = boost
        controls.handbrake[index, 0] = handbrake
    controls.validate()
    return controls


def _triangle_chassis_case(triangle: TriangleRecord) -> BreadthCase:
    variant = triangle.global_face % 6
    modes = (
        "normal_impact",
        "shallow_near_margin",
        "deeper_overlap",
        "tangential_slide",
        "angled_chassis",
        "angular_contact",
    )
    penetration = (0.25, -0.02, 0.75, 0.25, 0.35, 0.35)[variant]
    tilt = np.deg2rad(8.0) if variant == 4 else 0.0
    position, quat, forward, right, up = _chassis_pose(
        triangle.centroid,
        triangle.normal,
        triangle.forward,
        penetration,
        tilt,
    )
    velocity = -triangle.normal * np.float32((80, 20, 40, 20, 50, 50)[variant])
    if variant == 3:
        velocity += forward * np.float32(150.0)
    elif variant == 5:
        velocity += right * np.float32(75.0)
    angular = np.zeros(3, dtype=np.float32)
    if variant in (4, 5):
        angular = np.asarray(forward * np.float32(0.15) + up * np.float32(0.1), dtype=np.float32)
    return BreadthCase(
        case_id=f"F{triangle.global_face:05d}-C",
        case_kind="triangle_face",
        family="triangle_chassis",
        mode=modes[variant],
        contact_path="chassis",
        mesh_index=triangle.mesh_index,
        mesh_file=triangle.mesh_file,
        target_face=triangle.global_face,
        target_neighbor_face=None,
        target_edge=None,
        edge_class=None,
        analytic_plane=None,
        expected_plane_face=None,
        target_point=triangle.centroid,
        target_normal=triangle.normal,
        edge_start=None,
        edge_end=None,
        region_labels=triangle.region_labels,
        position=position,
        quaternion=quat,
        velocity=np.asarray(velocity, dtype=np.float32),
        angular_velocity=angular,
        controls=_zero_controls(),
    )


def _triangle_wheel_case(triangle: TriangleRecord) -> BreadthCase:
    variant = triangle.global_face % 6
    modes = (
        "front_wheel_compressed",
        "rear_wheel_compressed",
        "wheel_tangential",
        "wheel_normal_approach",
        "wheel_angled",
        "wheel_angular",
    )
    tilt = np.deg2rad(6.0) if variant == 4 else 0.0
    up = _normalize(
        triangle.normal * np.float32(np.cos(tilt))
        + _surface_right(triangle.normal, triangle.forward) * np.float32(np.sin(tilt))
    )
    matrix = _basis(triangle.forward, up)
    quat = matrix_to_quat(matrix)
    connection = FRONT_RIGHT_CONNECTION if variant % 2 == 0 else REAR_RIGHT_CONNECTION
    source_distance = np.float32(42.0 + 1.0 * (variant % 3))
    source = triangle.centroid + up * source_distance
    position = source - matrix @ connection
    velocity = -up * np.float32(10.0 if variant != 3 else 40.0)
    if variant == 2:
        velocity += matrix[:, 0] * np.float32(100.0)
    angular = np.zeros(3, dtype=np.float32)
    if variant == 5:
        angular = np.asarray(matrix[:, 1] * np.float32(0.15), dtype=np.float32)
    return BreadthCase(
        case_id=f"F{triangle.global_face:05d}-W",
        case_kind="triangle_face",
        family="triangle_wheel",
        mode=modes[variant],
        contact_path="wheel",
        mesh_index=triangle.mesh_index,
        mesh_file=triangle.mesh_file,
        target_face=triangle.global_face,
        target_neighbor_face=None,
        target_edge=None,
        edge_class=None,
        analytic_plane=None,
        expected_plane_face=None,
        target_point=triangle.centroid,
        target_normal=triangle.normal,
        edge_start=None,
        edge_end=None,
        region_labels=triangle.region_labels,
        position=np.asarray(position, dtype=np.float32),
        quaternion=quat,
        velocity=np.asarray(velocity, dtype=np.float32),
        angular_velocity=angular,
        controls=_zero_controls(),
    )


def _edge_case(edge: DirectedEdgeRecord) -> BreadthCase:
    variant = edge.directed_index % 5
    modes = (
        "chassis_spanning_both_faces",
        "approach_from_directed_face",
        "wheel_ray_crossing_seam",
        "slide_along_seam",
        "shallow_breaking_threshold",
    )
    normal = _average_normal(edge.face_normal, edge.neighbor_normal)
    point = edge.midpoint
    # A concave seam has no free-space bisector that can support a chassis;
    # placing the car on the averaged normal drives the box into both solids.
    # Approach from the directed face instead, with a small in-face inset that
    # remains inside the five-unit edge-observation band. Convex and planar
    # seams retain their physically accessible shared normal.
    if edge.edge_class == "concave":
        normal = edge.face_normal
        point = np.asarray(
            edge.midpoint + edge.face_interior_direction * np.float32(4.0),
            dtype=np.float32,
        )
    contact_path: ContactPath = "wheel" if variant == 2 else "chassis"
    if variant == 1:
        point = np.asarray(
            edge.midpoint + edge.face_interior_direction * np.float32(2.0),
            dtype=np.float32,
        )
        normal = edge.face_normal
    if contact_path == "wheel":
        matrix = _basis(edge.direction, normal)
        quat = matrix_to_quat(matrix)
        source = point + normal * np.float32(42.0)
        position = source - matrix @ FRONT_RIGHT_CONNECTION
        velocity = -normal * np.float32(15.0)
        angular = edge.direction * np.float32(0.05)
    else:
        penetration = -0.02 if variant == 4 else 0.3
        position, quat, forward, right, up = _chassis_pose(
            point, normal, edge.direction, penetration, 0.0
        )
        velocity = -normal * np.float32(20.0)
        if variant == 1:
            velocity += -edge.face_interior_direction * np.float32(100.0)
        elif variant == 3:
            velocity += forward * np.float32(150.0)
        angular = np.asarray(right * np.float32(0.03) + up * np.float32(0.02), dtype=np.float32)
    return BreadthCase(
        case_id=f"E{edge.directed_index:05d}",
        case_kind="shared_directed_edge",
        family=f"edge_{edge.edge_class}_{contact_path}",
        mode=modes[variant],
        contact_path=contact_path,
        mesh_index=edge.mesh_index,
        mesh_file=edge.mesh_file,
        target_face=edge.global_face,
        target_neighbor_face=edge.neighbor_face,
        target_edge=edge.local_edge,
        edge_class=edge.edge_class,
        analytic_plane=None,
        expected_plane_face=None,
        target_point=point,
        target_normal=normal,
        edge_start=edge.start,
        edge_end=edge.end,
        region_labels=edge.region_labels,
        position=np.asarray(position, dtype=np.float32),
        quaternion=quat,
        velocity=np.asarray(velocity, dtype=np.float32),
        angular_velocity=np.asarray(angular, dtype=np.float32),
        controls=_zero_controls(),
    )


def _analytic_plane_cases() -> list[BreadthCase]:
    planes = (
        (
            "floor",
            np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            -10,
        ),
        (
            "ceiling",
            np.asarray((0.0, 0.0, 2048.0), dtype=np.float32),
            np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
            -11,
        ),
        (
            "left_wall",
            np.asarray((-4096.0, 0.0, 1000.0), dtype=np.float32),
            np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
            -12,
        ),
        (
            "right_wall",
            np.asarray((4096.0, 0.0, 1000.0), dtype=np.float32),
            np.asarray((-1.0, 0.0, 0.0), dtype=np.float32),
            -13,
        ),
    )
    modes = (
        "normal_impact",
        "tangential_slide",
        "wheel_compressed",
        "shallow_near_margin",
        "angled_overlap",
    )
    result: list[BreadthCase] = []
    for plane_index, (name, point, normal, plane_face) in enumerate(planes):
        seed_forward = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        if abs(float(np.dot(seed_forward, normal))) > 0.9:
            seed_forward = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        for mode_index, mode in enumerate(modes):
            contact_path: ContactPath = "wheel" if mode_index == 2 else "chassis"
            matrix = _basis(seed_forward, normal)
            if contact_path == "wheel":
                quat = matrix_to_quat(matrix)
                source = point + normal * np.float32(30.0)
                position = source - matrix @ FRONT_RIGHT_CONNECTION
                velocity = -normal * np.float32(30.0)
                angular = np.zeros(3, dtype=np.float32)
                expected_face = -2 - plane_index
            else:
                penetration = -0.02 if mode_index == 3 else 1.5 if mode_index == 4 else 0.5
                tilt = np.deg2rad(8.0) if mode_index == 4 else 0.0
                position, quat, forward, _right, _up = _chassis_pose(
                    point, normal, seed_forward, penetration, tilt
                )
                velocity = -normal * np.float32(100.0)
                if mode_index == 1:
                    velocity += forward * np.float32(400.0)
                angular = (
                    forward * np.float32(0.2) if mode_index == 4 else np.zeros(3, dtype=np.float32)
                )
                expected_face = plane_face
            result.append(
                BreadthCase(
                    case_id=f"P-{name}-{mode_index}",
                    case_kind="analytic_plane",
                    family=f"analytic_{name}_{contact_path}",
                    mode=mode,
                    contact_path=contact_path,
                    mesh_index=None,
                    mesh_file=None,
                    target_face=None,
                    target_neighbor_face=None,
                    target_edge=None,
                    edge_class=None,
                    analytic_plane=name,
                    expected_plane_face=expected_face,
                    target_point=point,
                    target_normal=normal,
                    edge_start=None,
                    edge_end=None,
                    region_labels=(name,),
                    position=np.asarray(position, dtype=np.float32),
                    quaternion=quat,
                    velocity=np.asarray(velocity, dtype=np.float32),
                    angular_velocity=np.asarray(angular, dtype=np.float32),
                    controls=_control_pattern(
                        plane_index * len(modes) + mode_index + 3,
                        allow_handbrake=mode_index == 1,
                    ),
                )
            )
    return result


def _chassis_pose(
    point: np.ndarray,
    surface_normal: np.ndarray,
    forward_hint: np.ndarray,
    penetration: float,
    tilt_radians: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    right_hint = _surface_right(surface_normal, forward_hint)
    up = _normalize(
        surface_normal * np.float32(np.cos(tilt_radians))
        + right_hint * np.float32(np.sin(tilt_radians))
    )
    matrix = _basis(forward_hint, up)
    forward = matrix[:, 0]
    right = matrix[:, 1]
    support = float(np.sum(HITBOX_COLLISION_HALF * np.abs(matrix.T @ surface_normal)))
    center = point + surface_normal * np.float32(support - penetration)
    position = center - matrix @ HITBOX_OFFSET
    return (
        np.asarray(position, dtype=np.float32),
        matrix_to_quat(matrix),
        forward,
        right,
        up,
    )


def _basis(forward_hint: np.ndarray, up: np.ndarray) -> np.ndarray:
    up = _normalize(up)
    forward = np.asarray(forward_hint, dtype=np.float32)
    forward = forward - up * float(np.dot(forward, up))
    if float(np.linalg.norm(forward)) < 1e-6:
        seed = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        if abs(float(np.dot(seed, up))) > 0.9:
            seed = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        forward = seed - up * float(np.dot(seed, up))
    forward = _normalize(forward)
    right = _normalize(np.cross(up, forward))
    forward = _normalize(np.cross(right, up))
    return np.ascontiguousarray(np.column_stack((forward, right, up)), dtype=np.float32)


def _surface_right(normal: np.ndarray, forward: np.ndarray) -> np.ndarray:
    return _normalize(np.cross(normal, forward))


def _average_normal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    combined = np.asarray(left + right, dtype=np.float32)
    if float(np.linalg.norm(combined)) < 1e-5:
        return np.asarray(left, dtype=np.float32)
    return _normalize(combined)


def _normalize(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(source))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return np.asarray(source / np.float32(norm), dtype=np.float32)


def _control_pattern(
    index: int, *, allow_handbrake: bool
) -> tuple[float, float, float, float, float, int, int, int]:
    # Controls remain constant for the complete 12-tick local transition. Boost
    # is always disabled and starting boost is full, isolating static geometry.
    throttle = (0.0, 0.35, -0.25, 0.0)[index % 4]
    steer = (0.0, 0.2, -0.2)[index % 3]
    handbrake = int(allow_handbrake and index % 2 == 0)
    return (throttle, steer, 0.0, 0.0, 0.0, 0, 0, handbrake)


def _zero_controls() -> tuple[float, float, float, float, float, int, int, int]:
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)


def _region_labels(
    centroid: np.ndarray,
    normal: np.ndarray,
    area_uu2: float,
    minimum_altitude_uu: float,
) -> tuple[str, ...]:
    x, y, z = (float(value) for value in centroid)
    normal_z = abs(float(normal[2]))
    labels: set[str] = set()
    if z < 180.0:
        labels.add("floor_transition")
    if abs(x) > 3800.0:
        labels.add("side_wall")
    if abs(y) > 4900.0:
        labels.add("back_wall")
    if abs(y) > 5150.0 and abs(x) < 3000.0:
        labels.add("goal_geometry")
    if abs(x) > 3500.0 and abs(y) > 4000.0:
        labels.add("corner")
    if 0.12 < normal_z < 0.95 and z < 900.0:
        labels.add("ramp")
        labels.add("curved_wall_floor_transition")
    if z > 1750.0:
        labels.add("ceiling_transition")
    if minimum_altitude_uu < 20.0 or area_uu2 < 250.0:
        labels.add("narrow_or_small_triangle")
    if not labels:
        labels.add("general_wall_geometry")
    return tuple(sorted(labels))
