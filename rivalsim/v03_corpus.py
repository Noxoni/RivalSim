"""Frozen deterministic v0.3 dynamic-contact acceptance corpora.

Phase A covers the Soccar sphere against every source triangle, all directed
shared edges, and every analytic plane family.  Later phases extend this
module without changing the already-frozen Phase A case stream.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from rivalsim.arena import ArenaGeometry
from rivalsim.dfh_breadth import BreadthCatalog, build_breadth_catalog
from rivalsim.state import StateSnapshot

V03_GENERATOR_SCHEMA_VERSION = 1
V03_GENERATOR_SEED = 20260824
V03_HARD_HORIZONS = (1, 4, 8, 12)
BALL_RADIUS_UU = np.float32(91.25)

BallCaseKind = Literal["triangle_face", "shared_directed_edge", "analytic_plane"]
CarBallRegion = Literal[
    "front",
    "rear",
    "left",
    "right",
    "roof",
    "underside",
    "edge",
    "corner",
]


@dataclass(frozen=True, slots=True)
class BallWorldCase:
    case_id: str
    case_kind: BallCaseKind
    family: str
    mode: str
    mesh_index: int | None
    mesh_file: str | None
    target_face: int | None
    target_neighbor_face: int | None
    target_edge: int | None
    edge_class: str | None
    analytic_plane: str | None
    region_labels: tuple[str, ...]
    target_point: np.ndarray
    target_normal: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray


@dataclass(frozen=True, slots=True)
class CarBallCase:
    """One isolated Octane/standard-ball Phase B authority state."""

    case_id: str
    family: str
    contact_region: CarBallRegion
    feature_index: int
    motion_mode: str
    orientation_mode: str
    static_context: str
    overlap_uu: float
    car_position: np.ndarray
    car_velocity: np.ndarray
    car_quaternion: np.ndarray
    car_angular_velocity: np.ndarray
    car_on_ground: bool
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    ball_quaternion: np.ndarray
    ball_angular_velocity: np.ndarray


def generate_phase_a_cases(
    geometry: ArenaGeometry,
) -> tuple[BreadthCatalog, tuple[BallWorldCase, ...]]:
    """Enumerate the immutable Phase A corpus in a geometry-derived order."""

    catalog = build_breadth_catalog(geometry)
    cases: list[BallWorldCase] = []
    for triangle in catalog.triangles:
        variant = triangle.global_face % 6
        modes = (
            "normal_impact",
            "shallow_breaking_threshold",
            "deep_overlap",
            "tangential_spin",
            "oblique_impact",
            "separating_overlap",
        )
        overlaps = (0.25, -0.25, 4.0, 0.5, 0.5, 1.0)
        normal_speeds = (-150.0, -20.0, -50.0, -40.0, -300.0, 200.0)
        tangent_speeds = (0.0, 0.0, 0.0, 800.0, 500.0, 0.0)
        normal = triangle.normal
        tangent = triangle.forward
        position = triangle.centroid + normal * np.float32(
            BALL_RADIUS_UU - overlaps[variant]
        )
        velocity = (
            normal * np.float32(normal_speeds[variant])
            + tangent * np.float32(tangent_speeds[variant])
        )
        angular = np.zeros(3, dtype=np.float32)
        if variant in (3, 4):
            angular = np.asarray(
                np.cross(normal, tangent) * np.float32(3.0 if variant == 3 else 1.5),
                dtype=np.float32,
            )
        cases.append(
            BallWorldCase(
                case_id=f"A-F{triangle.global_face:05d}",
                case_kind="triangle_face",
                family="ball_triangle",
                mode=modes[variant],
                mesh_index=triangle.mesh_index,
                mesh_file=triangle.mesh_file,
                target_face=triangle.global_face,
                target_neighbor_face=None,
                target_edge=None,
                edge_class=None,
                analytic_plane=None,
                region_labels=triangle.region_labels,
                target_point=triangle.centroid,
                target_normal=normal,
                position=np.asarray(position, dtype=np.float32),
                velocity=np.asarray(velocity, dtype=np.float32),
                quaternion=_identity_quaternion(),
                angular_velocity=angular,
            )
        )

    for edge in catalog.directed_edges:
        variant = edge.directed_index % 5
        modes = (
            "edge_normal_impact",
            "edge_tangential_spin",
            "edge_shallow_threshold",
            "edge_oblique_impact",
            "edge_deep_overlap",
        )
        overlaps = (0.5, 0.5, -0.2, 0.75, 3.0)
        normal_speeds = (-160.0, -40.0, -20.0, -250.0, -60.0)
        edge_speeds = (0.0, 700.0, 0.0, 350.0, 0.0)
        normal = _edge_contact_normal(edge.face_normal, edge.neighbor_normal, edge.edge_class)
        position = edge.midpoint + normal * np.float32(BALL_RADIUS_UU - overlaps[variant])
        velocity = (
            normal * np.float32(normal_speeds[variant])
            + edge.direction * np.float32(edge_speeds[variant])
        )
        angular = np.zeros(3, dtype=np.float32)
        if variant in (1, 3):
            angular = np.asarray(edge.direction * np.float32(3.5), dtype=np.float32)
        cases.append(
            BallWorldCase(
                case_id=f"A-E{edge.directed_index:05d}",
                case_kind="shared_directed_edge",
                family="ball_shared_edge",
                mode=modes[variant],
                mesh_index=edge.mesh_index,
                mesh_file=edge.mesh_file,
                target_face=edge.global_face,
                target_neighbor_face=edge.neighbor_face,
                target_edge=edge.local_edge,
                edge_class=edge.edge_class,
                analytic_plane=None,
                region_labels=edge.region_labels,
                target_point=edge.midpoint,
                target_normal=normal,
                position=np.asarray(position, dtype=np.float32),
                velocity=np.asarray(velocity, dtype=np.float32),
                quaternion=_identity_quaternion(),
                angular_velocity=angular,
            )
        )

    cases.extend(_analytic_plane_cases())
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("v0.3 Phase A generator emitted duplicate case identifiers")
    return catalog, tuple(cases)


def phase_a_corpus_sha256(cases: tuple[BallWorldCase, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"v03-phase-a-{V03_GENERATOR_SCHEMA_VERSION}-{V03_GENERATOR_SEED}".encode()
    )
    for case in cases:
        for text in (
            case.case_id,
            case.case_kind,
            case.family,
            case.mode,
            "" if case.edge_class is None else case.edge_class,
            "" if case.analytic_plane is None else case.analytic_plane,
        ):
            digest.update(text.encode())
        for array in (
            case.target_point,
            case.target_normal,
            case.position,
            case.velocity,
            case.quaternion,
            case.angular_velocity,
        ):
            digest.update(np.asarray(array, dtype="<f4").tobytes())
        digest.update(
            np.asarray(
                (
                    -1 if case.mesh_index is None else case.mesh_index,
                    -1 if case.target_face is None else case.target_face,
                    -1 if case.target_neighbor_face is None else case.target_neighbor_face,
                    -1 if case.target_edge is None else case.target_edge,
                ),
                dtype="<i4",
            ).tobytes()
        )
    return digest.hexdigest().upper()


def phase_a_generator_config() -> dict[str, object]:
    return {
        "schema_version": V03_GENERATOR_SCHEMA_VERSION,
        "seed": V03_GENERATOR_SEED,
        "hard_horizons_ticks": list(V03_HARD_HORIZONS),
        "cached_ticks": list(range(1, 13)),
        "ball_radius_uu": float(BALL_RADIUS_UU),
        "triangle_cases": "one source-derived case per all 8020 global faces",
        "directed_edge_cases": "one source-derived case per directed shared edge",
        "analytic_plane_cases": "five modes for each Soccar floor/ceiling/X-wall plane",
        "case_order": ["triangles", "directed_shared_edges", "analytic_planes"],
    }


def phase_a_representative_indices(
    cases: tuple[BallWorldCase, ...], sample_count: int = 1024
) -> tuple[int, ...]:
    """Stable breadth sample plus all small analytic families."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(sample_count, len(cases))
    selected = set(np.linspace(0, len(cases) - 1, count, dtype=np.int64).tolist())
    selected.update(
        index for index, case in enumerate(cases) if case.case_kind == "analytic_plane"
    )
    # Explicitly retain every mesh, triangle mode, edge class, and edge mode.
    required_keys: set[tuple[object, ...]] = set()
    for index, case in enumerate(cases):
        keys = (
            ("mesh", case.mesh_index),
            ("family_mode", case.family, case.mode),
            ("edge_class", case.edge_class),
        )
        for key in keys:
            if key[-1] is not None and key not in required_keys:
                selected.add(index)
                required_keys.add(key)
    return tuple(sorted(int(index) for index in selected))


def phase_a_selection_sha256(
    cases: tuple[BallWorldCase, ...], indices: tuple[int, ...]
) -> str:
    """Bind a Phase A selection to the immutable complete corpus."""

    selected = np.asarray(indices, dtype="<i8")
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(cases)):
        raise ValueError("Phase A selection contains an invalid corpus index")
    if len(set(int(index) for index in selected)) != len(selected):
        raise ValueError("Phase A selection contains duplicate corpus indices")
    digest = hashlib.sha256()
    digest.update(b"v03-phase-a-selection-v1")
    digest.update(phase_a_corpus_sha256(cases).encode("ascii"))
    digest.update(selected.tobytes())
    return digest.hexdigest().upper()


def phase_a_cases_to_state(cases: tuple[BallWorldCase, ...] | list[BallWorldCase]) -> StateSnapshot:
    result = StateSnapshot.empty(len(cases))
    # Cars remain inert and high enough that Phase A has no dynamic contacts.
    result.car_pos[:] = (0.0, 0.0, 1500.0)
    for index, case in enumerate(cases):
        result.ball_pos[index] = case.position
        result.ball_vel[index] = case.velocity
        result.ball_quat[index] = case.quaternion
        result.ball_ang_vel[index] = case.angular_velocity
    result.validate()
    return result


# The Octane values below are the exact pinned CarConfig source values.  The
# btBoxShape outer half extent is hitboxSize / 2; Bullet moves its ordinary
# convex margin inside that outer envelope.
OCTANE_HITBOX_HALF_UU = np.asarray(
    (120.507 / 2.0, 86.6994 / 2.0, 38.6591 / 2.0), dtype=np.float32
)
OCTANE_HITBOX_OFFSET_UU = np.asarray((13.8757, 0.0, 20.755), dtype=np.float32)
PHASE_B_CASE_COUNT = 8_192


def generate_phase_b_cases() -> tuple[CarBallCase, ...]:
    """Freeze the deterministic v0.3 car/ball relative-state breadth corpus.

    The stream deliberately cycles independently through contact feature,
    relative motion, orientation, overlap, spin, and static-world context.
    The prime strides keep all cross-products represented without relying on
    mutable random sampling.
    """

    features = _phase_b_features()
    motions = (
        ("central_closing_low", 350.0, 0.0, 0.0),
        ("central_closing_hard", 2300.0, 0.0, 0.0),
        ("mutual_closing", 1800.0, -900.0, 0.0),
        ("same_direction_overtake", 1700.0, 900.0, 0.0),
        ("ball_toward_car", 200.0, -1800.0, 0.0),
        ("glancing_positive", 1200.0, -200.0, 850.0),
        ("glancing_negative", 900.0, 100.0, -1100.0),
        ("near_rest_dribble", 80.0, 20.0, 35.0),
    )
    orientations = (
        ("identity", (0.0, 0.0, 0.0)),
        ("yaw_90", (0.0, 0.0, np.pi / 2.0)),
        ("yaw_oblique", (0.0, 0.0, 0.63)),
        ("pitched_aerial", (0.0, 0.58, -0.27)),
        ("rolled_aerial", (-0.72, 0.12, 0.41)),
        ("inverted", (np.pi, 0.0, -0.35)),
        ("compound_rotation", (0.44, -0.51, 1.07)),
        ("near_wall_rotation", (0.0, np.pi / 2.0, 0.18)),
    )
    overlaps = (-0.25, 0.05, 0.5, 2.0, 6.0, 14.0)
    spin_vectors = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 5.5),
        (4.0, -2.0, 1.0),
        (-1.5, 5.0, -2.5),
    )
    car_angular_vectors = (
        (0.0, 0.0, 0.0),
        (0.7, -0.4, 0.9),
        (-2.2, 1.1, -0.6),
        (4.5, -1.0, 2.0),
    )
    contexts = (
        "aerial_center",
        "aerial_high",
        "grounded",
        "car_floor_contact",
        "ball_floor_contact",
        "positive_x_wall",
        "negative_x_wall",
        "positive_corner",
    )

    cases: list[CarBallCase] = []
    for index in range(PHASE_B_CASE_COUNT):
        region, feature_index, feature_point, outward = features[index % len(features)]
        motion = motions[(index * 5 + index // len(features)) % len(motions)]
        orientation = orientations[(index * 3 + index // 17) % len(orientations)]
        overlap = overlaps[(index * 5 + index // 29) % len(overlaps)]
        spin = spin_vectors[(index * 3 + index // 31) % len(spin_vectors)]
        car_angular = car_angular_vectors[(index * 7 + index // 37) % len(car_angular_vectors)]
        context = contexts[(index * 5 + index // 43) % len(contexts)]

        quaternion = _quat_from_euler_xyz(*orientation[1])
        basis = _quat_matrix(quaternion)
        car_position, car_on_ground = _phase_b_context_pose(context, index)
        box_center = car_position + basis @ OCTANE_HITBOX_OFFSET_UU
        world_feature = box_center + basis @ feature_point
        world_normal = _normalized(basis @ outward)
        tangent = _phase_b_tangent(world_normal)

        ball_position = world_feature + world_normal * np.float32(
            BALL_RADIUS_UU - np.float32(overlap)
        )
        # Ball-floor contexts retain the selected horizontal feature but move
        # the ball to native Soccar rest height and recompute the along-normal
        # offset so the pair remains locally touching.
        if context == "ball_floor_contact":
            ball_position[2] = np.float32(93.15)
        elif context in {"positive_x_wall", "negative_x_wall", "positive_corner"}:
            wall_x = np.float32(3998.0 if context != "negative_x_wall" else -3998.0)
            translation = wall_x - ball_position[0]
            car_position[0] += translation
            box_center[0] += translation
            world_feature[0] += translation
            ball_position[0] += translation
            if context == "positive_corner":
                translation_y = np.float32(5005.0) - ball_position[1]
                car_position[1] += translation_y
                ball_position[1] += translation_y

        car_speed, ball_speed, tangent_speed = motion[1:]
        car_velocity = world_normal * np.float32(car_speed)
        ball_velocity = world_normal * np.float32(ball_speed) + tangent * np.float32(
            tangent_speed
        )
        if motion[0] == "mutual_closing":
            car_velocity = world_normal * np.float32(car_speed)
            ball_velocity = world_normal * np.float32(ball_speed)

        cases.append(
            CarBallCase(
                case_id=f"B-{index:05d}",
                family="octane_standard_ball",
                contact_region=region,
                feature_index=feature_index,
                motion_mode=motion[0],
                orientation_mode=orientation[0],
                static_context=context,
                overlap_uu=float(overlap),
                car_position=np.asarray(car_position, dtype=np.float32),
                car_velocity=np.asarray(car_velocity, dtype=np.float32),
                car_quaternion=quaternion,
                car_angular_velocity=np.asarray(car_angular, dtype=np.float32),
                car_on_ground=car_on_ground,
                ball_position=np.asarray(ball_position, dtype=np.float32),
                ball_velocity=np.asarray(ball_velocity, dtype=np.float32),
                ball_quaternion=_identity_quaternion(),
                ball_angular_velocity=np.asarray(spin, dtype=np.float32),
            )
        )
    return tuple(cases)


def phase_b_cases_to_state(cases: tuple[CarBallCase, ...] | list[CarBallCase]) -> StateSnapshot:
    result = StateSnapshot.empty(len(cases))
    # The second car is outside the active arena interaction region. Native
    # Phase B authority owns only the active Octane; RivalSim keeps its fixed
    # two-car world layout and leaves this lane inert.
    result.car_pos[:, 1] = (0.0, 0.0, 5000.0)
    for index, case in enumerate(cases):
        result.car_pos[index, 0] = case.car_position
        result.car_vel[index, 0] = case.car_velocity
        result.car_quat[index, 0] = case.car_quaternion
        result.car_ang_vel[index, 0] = case.car_angular_velocity
        result.on_ground[index, 0] = int(case.car_on_ground)
        result.ball_pos[index] = case.ball_position
        result.ball_vel[index] = case.ball_velocity
        result.ball_quat[index] = case.ball_quaternion
        result.ball_ang_vel[index] = case.ball_angular_velocity
    result.validate()
    return result


def phase_b_corpus_sha256(cases: tuple[CarBallCase, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"v03-phase-b-{V03_GENERATOR_SCHEMA_VERSION}-{V03_GENERATOR_SEED}".encode()
    )
    for case in cases:
        for text in (
            case.case_id,
            case.family,
            case.contact_region,
            case.motion_mode,
            case.orientation_mode,
            case.static_context,
        ):
            digest.update(text.encode())
        digest.update(np.asarray((case.feature_index,), dtype="<i4").tobytes())
        digest.update(np.asarray((case.overlap_uu,), dtype="<f4").tobytes())
        for value in (
            case.car_position,
            case.car_velocity,
            case.car_quaternion,
            case.car_angular_velocity,
            case.ball_position,
            case.ball_velocity,
            case.ball_quaternion,
            case.ball_angular_velocity,
        ):
            digest.update(np.asarray(value, dtype="<f4").tobytes())
        digest.update(bytes((int(case.car_on_ground),)))
    return digest.hexdigest().upper()


def phase_b_generator_config() -> dict[str, object]:
    return {
        "schema_version": V03_GENERATOR_SCHEMA_VERSION,
        "seed": V03_GENERATOR_SEED,
        "case_count": PHASE_B_CASE_COUNT,
        "hard_horizons_ticks": list(V03_HARD_HORIZONS),
        "cached_ticks": list(range(1, 13)),
        "car": "one Octane with exact pinned hitbox and offset",
        "ball": "standard Soccar sphere radius 91.25 UU",
        "strata": [
            "6 faces plus 12 edges plus 8 corners",
            "8 relative-motion modes",
            "8 body-orientation modes",
            "6 overlap depths",
            "4 car angular-velocity and 4 ball-spin modes",
            "8 static-world contexts",
        ],
    }


def phase_b_representative_indices(
    cases: tuple[CarBallCase, ...], sample_count: int = 1024
) -> tuple[int, ...]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    selected = set(
        np.linspace(0, len(cases) - 1, min(sample_count, len(cases)), dtype=np.int64).tolist()
    )
    observed: set[tuple[str, str]] = set()
    for index, case in enumerate(cases):
        for key in (
            ("region", case.contact_region),
            ("motion", case.motion_mode),
            ("orientation", case.orientation_mode),
            ("context", case.static_context),
        ):
            if key not in observed:
                selected.add(index)
                observed.add(key)
    return tuple(sorted(int(index) for index in selected))


def _phase_b_features() -> tuple[tuple[CarBallRegion, int, np.ndarray, np.ndarray], ...]:
    half = OCTANE_HITBOX_HALF_UU
    result: list[tuple[CarBallRegion, int, np.ndarray, np.ndarray]] = []
    face_data = (
        ("front", 0, 1.0),
        ("rear", 0, -1.0),
        ("left", 1, 1.0),
        ("right", 1, -1.0),
        ("roof", 2, 1.0),
        ("underside", 2, -1.0),
    )
    for feature_index, (name, axis, sign) in enumerate(face_data):
        point = np.zeros(3, dtype=np.float32)
        normal = np.zeros(3, dtype=np.float32)
        point[axis] = half[axis] * np.float32(sign)
        # Exercise off-center face witnesses as well as central hits.
        tangent_axis = (axis + 1) % 3
        point[tangent_axis] = half[tangent_axis] * np.float32(
            (-0.65, -0.25, 0.25, 0.65)[feature_index % 4]
        )
        normal[axis] = np.float32(sign)
        result.append((name, feature_index, point, normal))

    feature_index = len(result)
    for free_axis in range(3):
        fixed = [axis for axis in range(3) if axis != free_axis]
        for sign_a in (-1.0, 1.0):
            for sign_b in (-1.0, 1.0):
                point = np.zeros(3, dtype=np.float32)
                normal = np.zeros(3, dtype=np.float32)
                point[fixed[0]] = half[fixed[0]] * np.float32(sign_a)
                point[fixed[1]] = half[fixed[1]] * np.float32(sign_b)
                point[free_axis] = half[free_axis] * np.float32(
                    (-0.45, 0.45)[feature_index & 1]
                )
                normal[fixed[0]] = np.float32(sign_a)
                normal[fixed[1]] = np.float32(sign_b)
                result.append(("edge", feature_index, point, _normalized(normal)))
                feature_index += 1

    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                signs = np.asarray((sx, sy, sz), dtype=np.float32)
                result.append(
                    ("corner", feature_index, half * signs, _normalized(signs))
                )
                feature_index += 1
    return tuple(result)


def _phase_b_context_pose(context: str, index: int) -> tuple[np.ndarray, bool]:
    if context == "aerial_center":
        return np.asarray((0.0, 0.0, 900.0), dtype=np.float32), False
    if context == "aerial_high":
        return np.asarray((350.0, -700.0, 1650.0), dtype=np.float32), False
    if context in {"grounded", "car_floor_contact", "ball_floor_contact"}:
        z = 17.0 if context == "grounded" else (12.0 if context == "car_floor_contact" else 45.0)
        return np.asarray((250.0, -500.0, z), dtype=np.float32), True
    y = np.float32(-600.0 + (index % 13) * 100.0)
    return np.asarray((0.0, y, 350.0), dtype=np.float32), False


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.asarray(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dtype=np.float32,
    )


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _normalized(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    length = np.float32(np.linalg.norm(value))
    if length <= np.float32(1.0e-20):
        raise ValueError("cannot normalize a zero Phase B feature vector")
    return np.asarray(value / length, dtype=np.float32)


def _phase_b_tangent(normal: np.ndarray) -> np.ndarray:
    axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    if abs(float(normal[2])) > 0.85:
        axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    return _normalized(np.cross(normal, axis))


def _edge_contact_normal(
    face_normal: np.ndarray, neighbor_normal: np.ndarray, edge_class: str
) -> np.ndarray:
    if edge_class == "concave":
        return np.asarray(face_normal, dtype=np.float32)
    summed = np.asarray(face_normal + neighbor_normal, dtype=np.float32)
    length = float(np.linalg.norm(summed))
    if length <= 1.0e-20:
        return np.asarray(face_normal, dtype=np.float32)
    return np.asarray(summed / np.float32(length), dtype=np.float32)


def _analytic_plane_cases() -> list[BallWorldCase]:
    planes = (
        (
            "floor",
            np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
        ),
        (
            "ceiling",
            np.asarray((0.0, 0.0, 2048.0), dtype=np.float32),
            np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        ),
        (
            "negative_x_wall",
            np.asarray((-4096.0, 0.0, 1022.0), dtype=np.float32),
            np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
        ),
        (
            "positive_x_wall",
            np.asarray((4096.0, 0.0, 1022.0), dtype=np.float32),
            np.asarray((-1.0, 0.0, 0.0), dtype=np.float32),
        ),
    )
    modes = (
        ("normal_impact", 0.25, -200.0, 0.0, 0.0),
        ("shallow_breaking_threshold", -0.25, -20.0, 0.0, 0.0),
        ("deep_overlap", 4.0, -50.0, 0.0, 0.0),
        ("tangential_spin", 0.5, -40.0, 800.0, 3.0),
        ("separating_overlap", 1.0, 200.0, 0.0, 0.0),
    )
    result: list[BallWorldCase] = []
    for plane_index, (name, point, normal) in enumerate(planes):
        tangent = _plane_tangent(normal)
        for mode_index, (mode, overlap, normal_speed, tangent_speed, spin) in enumerate(modes):
            position = point + normal * np.float32(BALL_RADIUS_UU - overlap)
            velocity = normal * np.float32(normal_speed) + tangent * np.float32(tangent_speed)
            angular = np.asarray(tangent * np.float32(spin), dtype=np.float32)
            result.append(
                BallWorldCase(
                    case_id=f"A-P{plane_index}-{mode_index}",
                    case_kind="analytic_plane",
                    family="ball_analytic_plane",
                    mode=mode,
                    mesh_index=None,
                    mesh_file=None,
                    target_face=None,
                    target_neighbor_face=None,
                    target_edge=None,
                    edge_class=None,
                    analytic_plane=name,
                    region_labels=(name,),
                    target_point=point,
                    target_normal=normal,
                    position=np.asarray(position, dtype=np.float32),
                    velocity=np.asarray(velocity, dtype=np.float32),
                    quaternion=_identity_quaternion(),
                    angular_velocity=angular,
                )
            )
    return result


def _plane_tangent(normal: np.ndarray) -> np.ndarray:
    if abs(float(normal[2])) > 0.5:
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    return np.asarray((0.0, 1.0, 0.0), dtype=np.float32)


def _identity_quaternion() -> np.ndarray:
    return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
