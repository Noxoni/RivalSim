"""Frozen deterministic v0.3 Phase D integrated-contact corpus.

Phase D is intentionally smaller than the pairwise breadth corpora.  It joins
already-covered local shape pairs into bounded two-Octane/one-ball Soccar
transitions which exercise island, manifold, constraint, and integration
ordering.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rivalsim.controls import ControlBatch
from rivalsim.state import StateSnapshot
from rivalsim.v03_corpus import (
    OCTANE_HITBOX_HALF_UU,
    OCTANE_HITBOX_OFFSET_UU,
    V03_HARD_HORIZONS,
    generate_phase_b_cases,
    phase_b_cases_to_state,
)
from rivalsim.v03_oracle_cache import canonical_json_bytes
from rivalsim.v03_phase_c_corpus import (
    generate_phase_c_cases,
    phase_c_cases_to_state,
)

PHASE_D_GENERATOR_SCHEMA_VERSION = 2
PHASE_D_GENERATOR_SEED = 20260825
PHASE_D_CASE_COUNT = 512
PHASE_D_TICKS = 12
PHASE_D_HARD_HORIZONS = V03_HARD_HORIZONS

PHASE_D_FAMILIES = (
    "static_ball",
    "car_ball_car_static",
    "car_ball_ball_static",
    "car_ball_wall_edge",
    "car_car_static",
    "two_cars_ball",
    "wheel_car_ball",
    "three_body_multi_manifold",
)

CONTROL_NAMES = (
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "jump",
    "boost",
    "handbrake",
)


@dataclass(frozen=True, slots=True)
class IntegratedCase:
    case_id: str
    family: str
    mode: str
    car_positions: np.ndarray
    car_velocities: np.ndarray
    car_quaternions: np.ndarray
    car_angular_velocities: np.ndarray
    car_on_ground: np.ndarray
    car_boost: np.ndarray
    car_is_supersonic: np.ndarray
    car_supersonic_time: np.ndarray
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    ball_quaternion: np.ndarray
    ball_angular_velocity: np.ndarray
    controls: tuple[np.ndarray, ...]


def phase_d_generator_config() -> dict[str, object]:
    return {
        "schema_version": PHASE_D_GENERATOR_SCHEMA_VERSION,
        "seed": PHASE_D_GENERATOR_SEED,
        "case_count": PHASE_D_CASE_COUNT,
        "ticks": PHASE_D_TICKS,
        "hard_horizons": list(PHASE_D_HARD_HORIZONS),
        "families": list(PHASE_D_FAMILIES),
        "family_case_count": PHASE_D_CASE_COUNT // len(PHASE_D_FAMILIES),
        "world": "exactly two Octanes, one standard Soccar ball, static Soccar arena",
        "collisions": {
            "car_car": True,
            "car_ball": True,
            "ball_world": True,
            "car_world": True,
        },
        "controls": "per-case, per-car, per-tick frozen float32/int32 tapes",
        "native_branches": ["a_then_b", "b_then_a"],
        "branch_policy": "complete-trajectory relational matching only",
        "source_inputs": {
            "phase_b": "accepted Phase B generator states selected by static context",
            "phase_c": "accepted Phase C generator states selected by static context",
            "integrated": "bounded symmetric and three-body chain constructions",
        },
    }


def phase_d_generator_source_sha256() -> str:
    # Bind the cache to every helper used by this generator, not only the top-
    # level function body. This is deliberately a whole-module source hash.
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()


def generate_phase_d_cases() -> tuple[IntegratedCase, ...]:
    phase_b = generate_phase_b_cases()
    phase_c = generate_phase_c_cases()
    b_by_context = {
        context: tuple(case for case in phase_b if case.static_context == context)
        for context in (
            "grounded",
            "car_floor_contact",
            "ball_floor_contact",
            "positive_x_wall",
            "negative_x_wall",
            "positive_corner",
        )
    }
    c_by_context = {
        context: tuple(case for case in phase_c if case.static_context == context)
        for context in (
            "both_grounded",
            "car0_grounded",
            "car1_grounded",
            "positive_x_wall",
            "negative_x_wall",
            "positive_corner",
        )
    }
    cases: list[IntegratedCase] = []
    per_family = PHASE_D_CASE_COUNT // len(PHASE_D_FAMILIES)
    for index in range(PHASE_D_CASE_COUNT):
        family_index = index // per_family
        variant = index % per_family
        family = PHASE_D_FAMILIES[family_index]
        if family == "static_ball":
            choices = b_by_context["ball_floor_contact"]
            source = choices[variant % len(choices)]
            state = phase_b_cases_to_state((source,))
            state.car_pos[0, 0] = (-2600.0, -3200.0, 900.0)
            state.car_pos[0, 1] = (2600.0, 3200.0, 900.0)
            state.car_vel.fill(0.0)
            state.car_ang_vel.fill(0.0)
            mode = f"ball_floor_{source.motion_mode}"
            controls = _zero_controls()
        elif family == "car_ball_car_static":
            choices = b_by_context["car_floor_contact"]
            source = choices[variant % len(choices)]
            state = phase_b_cases_to_state((source,))
            _park_phase_b_nonparticipant(state)
            mode = f"car_floor_{source.contact_region}_{source.motion_mode}"
            controls = _drive_controls(variant, boost=False)
        elif family == "car_ball_ball_static":
            choices = b_by_context["ball_floor_contact"]
            source = choices[variant % len(choices)]
            state = phase_b_cases_to_state((source,))
            _park_phase_b_nonparticipant(state)
            mode = f"ball_floor_{source.contact_region}_{source.motion_mode}"
            controls = _drive_controls(variant, boost=(variant & 3) == 0)
        elif family == "car_ball_wall_edge":
            context = ("positive_x_wall", "negative_x_wall", "positive_corner")[variant % 3]
            source = b_by_context[context][variant % len(b_by_context[context])]
            state = phase_b_cases_to_state((source,))
            _park_phase_b_nonparticipant(state)
            mode = f"{context}_{source.contact_region}_{source.motion_mode}"
            controls = _air_controls(variant, boost=(variant & 1) == 0)
        elif family == "car_car_static":
            context = (
                "both_grounded",
                "car0_grounded",
                "car1_grounded",
                "positive_x_wall",
                "negative_x_wall",
                "positive_corner",
            )[variant % 6]
            source = c_by_context[context][variant % len(c_by_context[context])]
            state = phase_c_cases_to_state((source,))
            mode = f"{context}_{source.contact_feature}_{source.motion_mode}"
            controls = _drive_controls(variant, boost=(variant & 7) == 0)
        elif family == "two_cars_ball":
            state = _two_cars_ball_state(variant, grounded=False)
            mode = ("head_on", "glancing", "vertical_offset", "spinning")[variant % 4]
            controls = _air_controls(variant, boost=(variant & 1) == 0)
        elif family == "wheel_car_ball":
            source = b_by_context["grounded"][variant % len(b_by_context["grounded"])]
            state = phase_b_cases_to_state((source,))
            _park_phase_b_nonparticipant(state)
            state.on_ground[0, 0] = 1
            mode = f"active_suspension_{source.contact_region}_{source.motion_mode}"
            controls = _drive_controls(variant, boost=(variant & 3) == 1)
        else:
            state = _three_body_chain_state(variant)
            mode = ("ball_floor", "car_floor", "aerial_torque", "wall_chain")[variant % 4]
            controls = _air_controls(variant, boost=True)

        cases.append(_case_from_state(index, family, mode, state, controls))

    if len(cases) != PHASE_D_CASE_COUNT:
        raise RuntimeError("Phase D generator emitted the wrong number of cases")
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("Phase D generator emitted duplicate case IDs")
    return tuple(cases)


def _park_phase_b_nonparticipant(state: StateSnapshot) -> None:
    """Replace Phase B's out-of-arena one-car sentinel with a valid second car.

    Phase B intentionally put its unused second slot above the Soccar ceiling.
    Phase D enables every collision family, so retaining that sentinel creates
    an unintended ceiling manifold. Keep the second Octane isolated but inside
    the fixed arena so these cases test the family named by their definition.
    """

    state.car_pos[0, 1] = (-2600.0, 2800.0, 900.0)
    state.car_vel[0, 1] = (0.0, 0.0, 0.0)
    state.car_quat[0, 1] = (0.0, 0.0, 0.0, 1.0)
    state.car_ang_vel[0, 1] = (0.0, 0.0, 0.0)
    state.on_ground[0, 1] = 0


def phase_d_cases_to_state(
    cases: tuple[IntegratedCase, ...] | list[IntegratedCase],
) -> StateSnapshot:
    state = StateSnapshot.empty(len(cases))
    for index, case in enumerate(cases):
        state.car_pos[index] = case.car_positions
        state.car_vel[index] = case.car_velocities
        state.car_quat[index] = case.car_quaternions
        state.car_ang_vel[index] = case.car_angular_velocities
        state.on_ground[index] = case.car_on_ground
        state.boost[index] = case.car_boost
        state.is_supersonic[index] = case.car_is_supersonic
        state.supersonic_time[index] = case.car_supersonic_time
        state.ball_pos[index] = case.ball_position
        state.ball_vel[index] = case.ball_velocity
        state.ball_quat[index] = case.ball_quaternion
        state.ball_ang_vel[index] = case.ball_angular_velocity
        for control_index, name in enumerate(CONTROL_NAMES):
            previous = case.controls[control_index][0]
            getattr(state, f"prev_{name}")[index] = previous
    state.validate()
    return state


def phase_d_controls_at(
    cases: tuple[IntegratedCase, ...] | list[IntegratedCase], tick: int
) -> ControlBatch:
    if tick < 0 or tick >= PHASE_D_TICKS:
        raise ValueError("Phase D control tick is outside the frozen tape")
    values: list[np.ndarray] = []
    for control_index, _name in enumerate(CONTROL_NAMES):
        dtype = np.float32 if control_index < 5 else np.int32
        values.append(
            np.ascontiguousarray(
                [case.controls[control_index][tick] for case in cases], dtype=dtype
            )
        )
    result = ControlBatch(*values)
    result.validate()
    return result


def phase_d_corpus_sha256(cases: tuple[IntegratedCase, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes([phase_d_case_record(case) for case in cases])
    ).hexdigest().upper()


def phase_d_selection_sha256(indices: tuple[int, ...]) -> str:
    return hashlib.sha256(canonical_json_bytes({"indices": list(indices)})).hexdigest().upper()


def phase_d_representative_indices(
    cases: tuple[IntegratedCase, ...], base_count: int = 128
) -> tuple[int, ...]:
    if base_count <= 0:
        raise ValueError("representative base count must be positive")
    selected = set(
        np.linspace(0, len(cases) - 1, min(base_count, len(cases)), dtype=np.int64).tolist()
    )
    seen: set[tuple[str, str]] = set()
    for index, case in enumerate(cases):
        for key in (("family", case.family), ("mode", case.mode)):
            if key not in seen:
                selected.add(index)
                seen.add(key)
    return tuple(sorted(int(index) for index in selected))


def phase_d_case_record(case: IntegratedCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "mode": case.mode,
        "car_positions": case.car_positions.astype(float).tolist(),
        "car_velocities": case.car_velocities.astype(float).tolist(),
        "car_quaternions": case.car_quaternions.astype(float).tolist(),
        "car_angular_velocities": case.car_angular_velocities.astype(float).tolist(),
        "car_on_ground": case.car_on_ground.astype(int).tolist(),
        "car_boost": case.car_boost.astype(float).tolist(),
        "car_is_supersonic": case.car_is_supersonic.astype(int).tolist(),
        "car_supersonic_time": case.car_supersonic_time.astype(float).tolist(),
        "ball_position": case.ball_position.astype(float).tolist(),
        "ball_velocity": case.ball_velocity.astype(float).tolist(),
        "ball_quaternion": case.ball_quaternion.astype(float).tolist(),
        "ball_angular_velocity": case.ball_angular_velocity.astype(float).tolist(),
        "controls": {
            name: case.controls[index].astype(float if index < 5 else int).tolist()
            for index, name in enumerate(CONTROL_NAMES)
        },
    }


def _case_from_state(
    index: int,
    family: str,
    mode: str,
    state: StateSnapshot,
    controls: tuple[np.ndarray, ...],
) -> IntegratedCase:
    return IntegratedCase(
        case_id=f"D-{index:04d}",
        family=family,
        mode=mode,
        car_positions=np.ascontiguousarray(state.car_pos[0], dtype=np.float32),
        car_velocities=np.ascontiguousarray(state.car_vel[0], dtype=np.float32),
        car_quaternions=np.ascontiguousarray(state.car_quat[0], dtype=np.float32),
        car_angular_velocities=np.ascontiguousarray(state.car_ang_vel[0], dtype=np.float32),
        car_on_ground=np.ascontiguousarray(state.on_ground[0], dtype=np.int32),
        car_boost=np.ascontiguousarray(state.boost[0], dtype=np.float32),
        car_is_supersonic=np.ascontiguousarray(state.is_supersonic[0], dtype=np.int32),
        car_supersonic_time=np.ascontiguousarray(state.supersonic_time[0], dtype=np.float32),
        ball_position=np.ascontiguousarray(state.ball_pos[0], dtype=np.float32),
        ball_velocity=np.ascontiguousarray(state.ball_vel[0], dtype=np.float32),
        ball_quaternion=np.ascontiguousarray(state.ball_quat[0], dtype=np.float32),
        ball_angular_velocity=np.ascontiguousarray(state.ball_ang_vel[0], dtype=np.float32),
        controls=tuple(np.ascontiguousarray(value) for value in controls),
    )


def _zero_controls() -> tuple[np.ndarray, ...]:
    return tuple(
        np.zeros((PHASE_D_TICKS, 2), dtype=np.float32 if index < 5 else np.int32)
        for index in range(len(CONTROL_NAMES))
    )


def _drive_controls(variant: int, *, boost: bool) -> tuple[np.ndarray, ...]:
    result = list(_zero_controls())
    sign = np.float32(-1.0 if variant & 1 else 1.0)
    result[0][:, 0] = np.float32(0.65) * sign
    result[0][:, 1] = np.float32(-0.35) * sign
    result[1][:, 0] = np.float32(0.45) * sign
    result[1][:, 1] = np.float32(-0.25) * sign
    result[7][4:9, variant & 1] = 1
    if boost:
        result[6][1:7, 0] = 1
    return tuple(result)


def _air_controls(variant: int, *, boost: bool) -> tuple[np.ndarray, ...]:
    result = list(_zero_controls())
    sign = np.float32(-1.0 if variant & 1 else 1.0)
    result[0][:, 0] = np.float32(0.25)
    result[0][:, 1] = np.float32(-0.2)
    result[2][:, 0] = np.float32(0.72) * sign
    result[2][:, 1] = np.float32(-0.55) * sign
    result[3][:, 0] = np.float32(-0.48) * sign
    result[3][:, 1] = np.float32(0.64) * sign
    result[4][:, 0] = np.float32(0.31)
    result[4][:, 1] = np.float32(-0.27)
    if boost:
        result[6][0:8, :] = 1
    return tuple(result)


def _two_cars_ball_state(variant: int, *, grounded: bool) -> StateSnapshot:
    state = StateSnapshot.empty(1)
    yaw_bias = np.float32(((variant % 9) - 4) * 0.025)
    z = np.float32(91.0 if grounded else 800.0 + (variant % 5) * 35.0)
    overlap = np.float32((0.25, 1.0, 3.0, 6.0)[variant % 4])
    front = OCTANE_HITBOX_OFFSET_UU[0] + OCTANE_HITBOX_HALF_UU[0]
    separation = np.float32(front + 91.25 - overlap)
    state.car_pos[0, 0] = (-separation, 0.0, z - OCTANE_HITBOX_OFFSET_UU[2])
    state.car_pos[0, 1] = (separation, 0.0, z - OCTANE_HITBOX_OFFSET_UU[2])
    state.car_quat[0, 0] = _quat_z(yaw_bias)
    state.car_quat[0, 1] = _quat_z(np.float32(np.pi) - yaw_bias)
    tangent = np.float32(((variant % 7) - 3) * 45.0)
    speed = np.float32(300.0 + (variant % 8) * 180.0)
    state.car_vel[0, 0] = (speed, tangent, 0.0)
    state.car_vel[0, 1] = (-speed, -tangent * np.float32(0.75), 0.0)
    state.car_ang_vel[0, 0] = (0.2, -0.3, (variant % 4) * 0.4)
    state.car_ang_vel[0, 1] = (-0.25, 0.15, -(variant % 5) * 0.3)
    state.on_ground[0] = int(grounded)
    state.boost[0] = (100.0, 100.0)
    state.ball_pos[0] = (0.0, 0.0, z)
    state.ball_vel[0] = (0.0, tangent * np.float32(0.2), -25.0 if grounded else 0.0)
    state.ball_ang_vel[0] = (0.0, 0.0, (variant % 6) * 0.7)
    return state


def _three_body_chain_state(variant: int) -> StateSnapshot:
    state = _two_cars_ball_state(variant, grounded=(variant % 4) < 2)
    mode = variant % 4
    if mode == 0:
        state.ball_pos[0, 2] = 90.75
        state.car_pos[0, :, 2] = 70.0
        state.ball_vel[0, 2] = -80.0
    elif mode == 1:
        state.car_pos[0, :, 2] = 17.0
        state.on_ground[0] = 1
        state.ball_pos[0, 2] = 105.0
    elif mode == 2:
        state.on_ground[0] = 0
        state.ball_pos[0, 2] = 1100.0 + (variant % 6) * 25.0
        state.car_pos[0, :, 2] = state.ball_pos[0, 2] - OCTANE_HITBOX_OFFSET_UU[2]
    else:
        shift = np.float32(3990.0 - state.ball_pos[0, 0])
        state.ball_pos[0, 0] += shift
        state.car_pos[0, :, 0] += shift
        state.ball_vel[0, 0] = 120.0
    # Every fourth case also overlaps the two Octanes behind the ball, joining
    # all accepted pair families in one bounded island.
    if (variant // 4) & 1:
        midpoint = (state.car_pos[0, 0] + state.car_pos[0, 1]) * np.float32(0.5)
        state.car_pos[0, 0, 1] = midpoint[1] - np.float32(40.0)
        state.car_pos[0, 1, 1] = midpoint[1] + np.float32(40.0)
    return state


def _quat_z(angle: np.float32) -> np.ndarray:
    half = np.float32(angle * np.float32(0.5))
    return np.asarray((0.0, 0.0, np.sin(half), np.cos(half)), dtype=np.float32)


__all__ = [
    "CONTROL_NAMES",
    "PHASE_D_CASE_COUNT",
    "PHASE_D_FAMILIES",
    "PHASE_D_GENERATOR_SCHEMA_VERSION",
    "PHASE_D_GENERATOR_SEED",
    "PHASE_D_HARD_HORIZONS",
    "PHASE_D_TICKS",
    "IntegratedCase",
    "generate_phase_d_cases",
    "phase_d_case_record",
    "phase_d_cases_to_state",
    "phase_d_controls_at",
    "phase_d_corpus_sha256",
    "phase_d_generator_config",
    "phase_d_generator_source_sha256",
    "phase_d_representative_indices",
    "phase_d_selection_sha256",
]
