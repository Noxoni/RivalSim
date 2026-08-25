"""Frozen deterministic v0.3 Phase C Octane/Octane authority corpus.

This module is deliberately separate from :mod:`rivalsim.v03_corpus` so the
already-frozen Phase A and Phase B authority identities remain immutable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from rivalsim.state import StateSnapshot
from rivalsim.v03_oracle_cache import canonical_json_bytes

PHASE_C_GENERATOR_SCHEMA_VERSION = 1
PHASE_C_GENERATOR_SEED = 20260824
PHASE_C_CASE_COUNT = 8192
PHASE_C_HARD_HORIZONS = (1, 4, 8, 12)

OCTANE_HALF_UU = np.asarray((60.2535, 43.3497, 19.32955), dtype=np.float32)
OCTANE_OFFSET_UU = np.asarray((13.8757, 0.0, 20.755), dtype=np.float32)

CONTACT_FEATURES = (
    "front_front",
    "nose_side",
    "side_side",
    "rear_overtake",
    "glancing_positive",
    "glancing_negative",
    "roof_underside",
    "front_corner_threshold",
)
ORIENTATION_MODES = (
    "identity",
    "yaw_90",
    "yaw_oblique",
    "pitched",
    "rolled",
    "compound",
    "inverted",
    "counter_rotated",
)
MOTION_MODES = (
    "low_closing",
    "medium_closing",
    "hard_closing",
    "high_head_on",
    "rear_overtake",
    "glancing",
    "supersonic_false",
    "supersonic_true",
)
STATIC_CONTEXTS = (
    "aerial_center",
    "aerial_high",
    "both_grounded",
    "car0_grounded",
    "car1_grounded",
    "positive_x_wall",
    "negative_x_wall",
    "positive_corner",
)
OVERLAPS_UU = (-0.25, 0.25, 1.0, 4.0, 8.0, 16.0, 0.5, 2.0)


@dataclass(frozen=True, slots=True)
class CarCarCase:
    case_id: str
    contact_feature: str
    orientation_mode: str
    motion_mode: str
    static_context: str
    overlap_uu: float
    car_positions: np.ndarray
    car_velocities: np.ndarray
    car_quaternions: np.ndarray
    car_angular_velocities: np.ndarray
    car_on_ground: np.ndarray
    car_is_supersonic: np.ndarray
    car_supersonic_time: np.ndarray


def phase_c_generator_config() -> dict[str, object]:
    return {
        "schema_version": PHASE_C_GENERATOR_SCHEMA_VERSION,
        "seed": PHASE_C_GENERATOR_SEED,
        "case_count": PHASE_C_CASE_COUNT,
        "hard_horizons": list(PHASE_C_HARD_HORIZONS),
        "contact_features": list(CONTACT_FEATURES),
        "orientation_modes": list(ORIENTATION_MODES),
        "motion_modes": list(MOTION_MODES),
        "static_contexts": list(STATIC_CONTEXTS),
        "overlaps_uu": list(OVERLAPS_UU),
        "octane_half_uu": OCTANE_HALF_UU.astype(float).tolist(),
        "octane_offset_uu": OCTANE_OFFSET_UU.astype(float).tolist(),
        "controls": "zero for both cars for ticks 1..12",
        "ball": "parked outside the local interaction and car-ball disabled in authority",
        "index_mixing": [1, 3, 5, 7],
    }


def generate_phase_c_cases() -> tuple[CarCarCase, ...]:
    """Return the immutable isolated two-Octane Phase C corpus."""

    cases: list[CarCarCase] = []
    for index in range(PHASE_C_CASE_COUNT):
        feature_index = index & 7
        orientation_index = (index // 8 + 3 * feature_index) & 7
        motion_index = (index // 64 + 5 * orientation_index) & 7
        context_index = (index // 512 + 7 * motion_index) & 7
        overlap_index = (index // 1024 + feature_index + orientation_index) & 7

        feature = CONTACT_FEATURES[feature_index]
        orientation = ORIENTATION_MODES[orientation_index]
        motion = MOTION_MODES[motion_index]
        context = STATIC_CONTEXTS[context_index]
        overlap = float(OVERLAPS_UU[overlap_index])

        quaternion_a, quaternion_b, direction = _orientations(feature, orientation)
        basis_a = _quat_matrix(quaternion_a)
        basis_b = _quat_matrix(quaternion_b)
        direction = _normalize(direction)
        support_a = float(np.sum(np.abs(basis_a.T @ direction) * OCTANE_HALF_UU))
        support_b = float(np.sum(np.abs(basis_b.T @ direction) * OCTANE_HALF_UU))

        midpoint = _context_midpoint(context, index)
        child_a = midpoint - direction * np.float32(
            0.5 * (support_a + support_b - overlap)
        )
        child_b = midpoint + direction * np.float32(
            0.5 * (support_a + support_b - overlap)
        )
        root_a = child_a - basis_a @ OCTANE_OFFSET_UU
        root_b = child_b - basis_b @ OCTANE_OFFSET_UU
        root_a, root_b = _apply_static_context(context, root_a, root_b)

        velocity_a, velocity_b, supersonic = _velocities(
            motion, direction, basis_a, index
        )
        angular_scale = np.float32(((index // 4096) * 2 - 1) * 0.35)
        angular_a = np.asarray(
            (angular_scale, -angular_scale * np.float32(0.5), angular_scale * np.float32(0.25)),
            dtype=np.float32,
        )
        angular_b = np.asarray(
            (-angular_scale * np.float32(0.75), angular_scale * np.float32(0.4), -angular_scale),
            dtype=np.float32,
        )
        grounded = np.asarray(
            (
                context in {"both_grounded", "car0_grounded"},
                context in {"both_grounded", "car1_grounded"},
            ),
            dtype=np.int32,
        )
        cases.append(
            CarCarCase(
                case_id=f"C-{index:05d}",
                contact_feature=feature,
                orientation_mode=orientation,
                motion_mode=motion,
                static_context=context,
                overlap_uu=overlap,
                car_positions=np.ascontiguousarray((root_a, root_b), dtype=np.float32),
                car_velocities=np.ascontiguousarray((velocity_a, velocity_b), dtype=np.float32),
                car_quaternions=np.ascontiguousarray(
                    (quaternion_a, quaternion_b), dtype=np.float32
                ),
                car_angular_velocities=np.ascontiguousarray(
                    (angular_a, angular_b), dtype=np.float32
                ),
                car_on_ground=grounded,
                car_is_supersonic=np.asarray(supersonic, dtype=np.int32),
                car_supersonic_time=np.zeros(2, dtype=np.float32),
            )
        )
    return tuple(cases)


def phase_c_cases_to_state(cases: tuple[CarCarCase, ...]) -> StateSnapshot:
    state = StateSnapshot.empty(len(cases))
    for index, case in enumerate(cases):
        state.car_pos[index] = case.car_positions
        state.car_vel[index] = case.car_velocities
        state.car_quat[index] = case.car_quaternions
        state.car_ang_vel[index] = case.car_angular_velocities
        state.on_ground[index] = case.car_on_ground
        state.is_supersonic[index] = case.car_is_supersonic
        state.supersonic_time[index] = case.car_supersonic_time
    state.ball_pos[:] = (-3000.0, -4000.0, 1500.0)
    state.ball_vel.fill(0.0)
    state.ball_ang_vel.fill(0.0)
    state.validate()
    return state


def phase_c_representative_indices(
    cases: tuple[CarCarCase, ...], base_count: int = 1024
) -> tuple[int, ...]:
    if base_count <= 0:
        raise ValueError("representative base count must be positive")
    selected = set(
        np.linspace(0, len(cases) - 1, min(base_count, len(cases)), dtype=np.int64).tolist()
    )
    buckets: dict[tuple[str, str], int] = {}
    for index, case in enumerate(cases):
        labels = (
            ("feature", case.contact_feature),
            ("orientation", case.orientation_mode),
            ("motion", case.motion_mode),
            ("context", case.static_context),
        )
        for key in labels:
            buckets.setdefault(key, index)
    selected.update(buckets.values())
    return tuple(sorted(selected))


def phase_c_corpus_sha256(cases: tuple[CarCarCase, ...]) -> str:
    payload = [_case_record(case) for case in cases]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest().upper()


def phase_c_selection_sha256(indices: tuple[int, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"indices": list(indices)})
    ).hexdigest().upper()


def _case_record(case: CarCarCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "contact_feature": case.contact_feature,
        "orientation_mode": case.orientation_mode,
        "motion_mode": case.motion_mode,
        "static_context": case.static_context,
        "overlap_uu": case.overlap_uu,
        "car_positions": case.car_positions.astype(float).tolist(),
        "car_velocities": case.car_velocities.astype(float).tolist(),
        "car_quaternions": case.car_quaternions.astype(float).tolist(),
        "car_angular_velocities": case.car_angular_velocities.astype(float).tolist(),
        "car_on_ground": case.car_on_ground.astype(int).tolist(),
        "car_is_supersonic": case.car_is_supersonic.astype(int).tolist(),
        "car_supersonic_time": case.car_supersonic_time.astype(float).tolist(),
    }


def case_record(case: CarCarCase) -> dict[str, object]:
    """Public canonical record used by the Phase C frozen cache."""

    return _case_record(case)


def _orientations(
    feature: str, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared = {
        "identity": (0.0, 0.0, 0.0),
        "yaw_90": (0.0, 0.0, np.pi / 2.0),
        "yaw_oblique": (0.0, 0.0, 0.37),
        "pitched": (0.0, 0.31, 0.0),
        "rolled": (0.42, 0.0, 0.0),
        "compound": (0.29, -0.24, 0.41),
        "inverted": (np.pi, 0.0, 0.18),
        "counter_rotated": (-0.33, 0.27, -0.46),
    }[mode]
    relative_b = {
        "front_front": (0.0, 0.0, np.pi),
        "nose_side": (0.0, 0.0, np.pi / 2.0),
        "side_side": (0.0, 0.0, 0.0),
        "rear_overtake": (0.0, 0.0, 0.0),
        "glancing_positive": (0.0, 0.0, 0.52),
        "glancing_negative": (0.0, 0.0, -0.52),
        "roof_underside": (np.pi, 0.0, 0.0),
        "front_corner_threshold": (0.0, 0.0, np.pi / 2.0),
    }[feature]
    quaternion_a = _quat_from_euler(*shared)
    quaternion_b = _quat_multiply(quaternion_a, _quat_from_euler(*relative_b))
    basis_a = _quat_matrix(quaternion_a)
    direction_local = {
        "front_front": (1.0, 0.0, 0.0),
        "nose_side": (1.0, 0.0, 0.0),
        "side_side": (0.0, 1.0, 0.0),
        "rear_overtake": (1.0, 0.0, 0.0),
        "glancing_positive": (1.0, 0.55, 0.0),
        "glancing_negative": (1.0, -0.55, 0.0),
        "roof_underside": (0.0, 0.0, 1.0),
        "front_corner_threshold": (0.35, 1.0, 0.0),
    }[feature]
    direction = basis_a @ np.asarray(direction_local, dtype=np.float32)
    return quaternion_a, quaternion_b, np.asarray(direction, dtype=np.float32)


def _velocities(
    mode: str, direction: np.ndarray, basis_a: np.ndarray, index: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    tangent = _normalize(np.cross(np.asarray((0.0, 0.0, 1.0), dtype=np.float32), direction))
    if np.linalg.norm(tangent) < 0.5:
        tangent = np.asarray(basis_a[:, 1], dtype=np.float32)
    values = {
        "low_closing": (50.0, -25.0, 0.0, (0, 0)),
        "medium_closing": (700.0, -350.0, 0.0, (0, 0)),
        "hard_closing": (1400.0, -900.0, 0.0, (0, 0)),
        "high_head_on": (2200.0, -2200.0, 0.0, (1, 1)),
        "rear_overtake": (1800.0, 400.0, 0.0, (0, 0)),
        "glancing": (1200.0, -500.0, 650.0, (0, 0)),
        "supersonic_false": (2199.0, 0.0, 0.0, (0, 0)),
        "supersonic_true": (2201.0, 0.0, 0.0, (1, 0)),
    }[mode]
    speed_a, speed_b, tangent_speed, supersonic = values
    sign = np.float32(-1.0 if (index & 4096) else 1.0)
    velocity_a = direction * np.float32(speed_a) + tangent * np.float32(tangent_speed) * sign
    velocity_b = direction * np.float32(speed_b) + tangent * np.float32(tangent_speed * 0.25) * sign
    return (
        np.asarray(velocity_a, dtype=np.float32),
        np.asarray(velocity_b, dtype=np.float32),
        supersonic,
    )


def _context_midpoint(context: str, index: int) -> np.ndarray:
    jitter = np.float32(((index * 2654435761) & 1023) / 1023.0 - 0.5)
    values = {
        "aerial_center": (0.0, 0.0, 650.0),
        "aerial_high": (750.0, -500.0, 1250.0),
        "both_grounded": (0.0, 0.0, 17.0),
        "car0_grounded": (-700.0, 400.0, 95.0),
        "car1_grounded": (700.0, -400.0, 95.0),
        "positive_x_wall": (4020.0, -700.0, 180.0),
        "negative_x_wall": (-4020.0, 700.0, 180.0),
        "positive_corner": (4015.0, 5030.0, 170.0),
    }[context]
    result = np.asarray(values, dtype=np.float32)
    result[1] += jitter * np.float32(80.0)
    return result


def _apply_static_context(
    context: str, root_a: np.ndarray, root_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(root_a, dtype=np.float32).copy()
    b = np.asarray(root_b, dtype=np.float32).copy()
    if context == "both_grounded":
        a[2] = np.float32(17.0)
        b[2] = np.float32(17.0)
    elif context == "car0_grounded":
        shift = np.float32(17.0) - a[2]
        a[2] += shift
        b[2] += shift
    elif context == "car1_grounded":
        shift = np.float32(17.0) - b[2]
        a[2] += shift
        b[2] += shift
    return a, b


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    result = np.asarray(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dtype=np.float32,
    )
    result /= np.linalg.norm(result)
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = np.asarray(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dtype=np.float32,
    )
    result /= np.linalg.norm(result)
    return result


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion.astype(np.float32)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        (
            (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
            (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
            (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
        ),
        dtype=np.float32,
    )


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    length = np.float32(np.linalg.norm(value))
    if length == 0.0:
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    return np.asarray(value / length, dtype=np.float32)
