"""State-space-learning foundation reward and reset curriculum.

The reward surface in this module is deliberately small.  It contains four
bounded state potentials and no event or occupancy reward.  Scenario families
are reset-state diversity only; their identity is never included in the policy
observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp

from rivalsim.rival2_contracts import (
    BALL_LINEAR_SPEED_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    OBS_FIELD_NAMES,
    POSITION_SCALE,
)
from rivalsim.state import (
    CAR_FLOAT_FIELDS,
    CAR_INT_FIELDS,
    CAR_VEC3_FIELDS,
    GpuState,
    StateSnapshot,
)

SSL_FOUNDATION_GAMMA = 0.9987476493904754
SSL_FOUNDATION_WEIGHTS = {
    "field": 1.25,
    "access": 0.75,
    "control": 0.75,
    "defense": 0.50,
}
SCENARIO_NAMES = (
    "natural_ongoing",
    "loose_ball_access",
    "catch_control_possession",
    "shooting_finishing",
    "defensive_shadow_save",
    "contested_fifty",
    "wall_aerial",
    "recovery_scramble_low_boost",
)
SCENARIO_PROBABILITIES = (0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05)

_INDEX = {
    name: OBS_FIELD_NAMES.index(name)
    for name in (
        "ball.position.x",
        "ball.position.y",
        "ball.position.z",
        "ball.linear_velocity.x",
        "ball.linear_velocity.y",
        "ball.linear_velocity.z",
        "self.position.x",
        "self.position.y",
        "self.position.z",
        "self.linear_velocity.x",
        "self.linear_velocity.y",
        "self.linear_velocity.z",
        "opponent.position.x",
        "opponent.position.y",
        "opponent.position.z",
        "opponent.linear_velocity.x",
        "opponent.linear_velocity.y",
        "opponent.linear_velocity.z",
    )
}


@dataclass(frozen=True, slots=True)
class SslFoundationPotentials:
    field: torch.Tensor
    access: torch.Tensor
    control: torch.Tensor
    defense: torch.Tensor

    @property
    def weighted_total(self) -> torch.Tensor:
        return (
            self.field * SSL_FOUNDATION_WEIGHTS["field"]
            + self.access * SSL_FOUNDATION_WEIGHTS["access"]
            + self.control * SSL_FOUNDATION_WEIGHTS["control"]
            + self.defense * SSL_FOUNDATION_WEIGHTS["defense"]
        )


def _physical_vectors(observation: torch.Tensor) -> dict[str, torch.Tensor]:
    pscale = observation.new_tensor(POSITION_SCALE)
    ball_position = observation[..., [_INDEX[f"ball.position.{axis}"] for axis in "xyz"]] * pscale
    ball_velocity = (
        observation[..., [_INDEX[f"ball.linear_velocity.{axis}"] for axis in "xyz"]]
        * BALL_LINEAR_SPEED_SCALE
    )
    self_position = observation[..., [_INDEX[f"self.position.{axis}"] for axis in "xyz"]] * pscale
    self_velocity = (
        observation[..., [_INDEX[f"self.linear_velocity.{axis}"] for axis in "xyz"]]
        * CAR_LINEAR_SPEED_SCALE
    )
    opponent_position = (
        observation[..., [_INDEX[f"opponent.position.{axis}"] for axis in "xyz"]] * pscale
    )
    opponent_velocity = (
        observation[..., [_INDEX[f"opponent.linear_velocity.{axis}"] for axis in "xyz"]]
        * CAR_LINEAR_SPEED_SCALE
    )
    return {
        "ball_position": ball_position,
        "ball_velocity": ball_velocity,
        "self_position": self_position,
        "self_velocity": self_velocity,
        "opponent_position": opponent_position,
        "opponent_velocity": opponent_velocity,
    }


def _controllability(
    ball_position: torch.Tensor,
    ball_velocity: torch.Tensor,
    car_position: torch.Tensor,
    car_velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance = torch.linalg.vector_norm(ball_position - car_position, dim=-1)
    relative_speed = torch.linalg.vector_norm(ball_velocity - car_velocity, dim=-1)
    proximity = ((500.0 - distance) / 350.0).clamp(0.0, 1.0)
    velocity_match = (1.0 - relative_speed / 1200.0).clamp(0.0, 1.0)
    return proximity * velocity_match, distance


def _coverage(
    car_position: torch.Tensor,
    ball_position: torch.Tensor,
    own_goal_y: float,
) -> torch.Tensor:
    goal = torch.zeros_like(ball_position[..., :2])
    goal[..., 1] = own_goal_y
    goal_to_ball = ball_position[..., :2] - goal
    goal_to_car = car_position[..., :2] - goal
    denominator = goal_to_ball.square().sum(dim=-1).clamp_min(1.0)
    longitudinal = (goal_to_car * goal_to_ball).sum(dim=-1) / denominator
    projection = goal + longitudinal.unsqueeze(-1) * goal_to_ball
    cross_track = torch.linalg.vector_norm(car_position[..., :2] - projection, dim=-1)
    # Smoothly excludes positions behind the goal and beyond the ball while
    # retaining broad shadow coverage between them.
    between = (longitudinal / 0.10).clamp(0.0, 1.0) * ((1.0 - longitudinal) / 0.20).clamp(0.0, 1.0)
    lateral = (1.0 - cross_track / 1500.0).clamp(0.0, 1.0)
    return between * lateral


def ssl_foundation_potentials(observation: torch.Tensor) -> SslFoundationPotentials:
    """Compute the four normalized, agent-canonical state potentials."""

    if observation.shape[-1] != len(OBS_FIELD_NAMES):
        raise ValueError("SSL Foundation requires the complete 182-field observation")
    vectors = _physical_vectors(observation)
    ball_position = vectors["ball_position"]
    ball_velocity = vectors["ball_velocity"]
    self_control, self_distance = _controllability(
        ball_position,
        ball_velocity,
        vectors["self_position"],
        vectors["self_velocity"],
    )
    opponent_control, opponent_distance = _controllability(
        ball_position,
        ball_velocity,
        vectors["opponent_position"],
        vectors["opponent_velocity"],
    )
    field = (ball_position[..., 1] / 5120.0).clamp(-1.0, 1.0)
    free_ball = 1.0 - torch.maximum(self_control, opponent_control)
    access = free_ball * ((opponent_distance - self_distance) / 3000.0).clamp(-1.0, 1.0)
    control = self_control - opponent_control

    self_territory = ((-ball_position[..., 1] / 5120.0) + 1.0).mul(0.5).clamp(0.0, 1.0)
    opponent_territory = ((ball_position[..., 1] / 5120.0) + 1.0).mul(0.5).clamp(0.0, 1.0)
    self_goalward_velocity = (-ball_velocity[..., 1] / CAR_LINEAR_SPEED_SCALE).clamp(0.0, 1.0)
    opponent_goalward_velocity = (ball_velocity[..., 1] / CAR_LINEAR_SPEED_SCALE).clamp(0.0, 1.0)
    self_threat = (
        0.45 * self_territory + 0.30 * self_goalward_velocity + 0.25 * opponent_control
    ).clamp(0.0, 1.0)
    opponent_threat = (
        0.45 * opponent_territory + 0.30 * opponent_goalward_velocity + 0.25 * self_control
    ).clamp(0.0, 1.0)
    self_coverage = _coverage(vectors["self_position"], ball_position, -5120.0)
    opponent_coverage = _coverage(vectors["opponent_position"], ball_position, 5120.0)
    defense = self_threat * self_coverage - opponent_threat * opponent_coverage
    return SslFoundationPotentials(field, access, control, defense)


def ssl_foundation_shaping(
    before: torch.Tensor,
    after: torch.Tensor,
    terminated: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return policy-invariant shaping; goal successors are absorbing."""

    if before.shape != after.shape:
        raise ValueError("before/after observations must have identical shapes")
    if terminated.shape != before.shape[:-2]:
        raise ValueError("terminated must have one value per world")
    prior = ssl_foundation_potentials(before)
    successor = ssl_foundation_potentials(after)
    absorbing = terminated[..., None]
    result: dict[str, torch.Tensor] = {}
    for name in ("field", "access", "control", "defense"):
        before_value = getattr(prior, name)
        after_value = getattr(successor, name).masked_fill(absorbing, 0.0)
        result[name] = SSL_FOUNDATION_WEIGHTS[name] * (
            SSL_FOUNDATION_GAMMA * after_value - before_value
        )
    result["total"] = sum(result.values())
    return result


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray((0.0, 0.0, np.sin(yaw * 0.5), np.cos(yaw * 0.5)), dtype=np.float32)


def _euler_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
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


def _face(from_xy: np.ndarray, to_xy: np.ndarray) -> np.ndarray:
    delta = to_xy - from_xy
    return _yaw_quaternion(float(np.arctan2(delta[1], delta[0])))


def _scenario_counts(worlds: int) -> np.ndarray:
    expected = np.asarray(SCENARIO_PROBABILITIES, dtype=np.float64) * worlds
    counts = np.floor(expected).astype(np.int64)
    remainder = worlds - int(counts.sum())
    if remainder:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


@dataclass(frozen=True, slots=True)
class SslFoundationScenarioBatch:
    state: StateSnapshot
    family: np.ndarray
    focal_side: np.ndarray
    kickoff_indicator: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "worlds": int(self.family.size),
            "counts": {
                name: int((self.family == index).sum()) for index, name in enumerate(SCENARIO_NAMES)
            },
            "focal_side_counts": {
                str(side): int((self.focal_side == side).sum()) for side in (0, 1)
            },
            "task_or_scenario_id_in_observation": False,
            "scripted_solution_prefix_ticks": 0,
        }


def build_ssl_foundation_scenarios(worlds: int, *, seed: int) -> SslFoundationScenarioBatch:
    """Build a deterministic, role-balanced batch of physically valid starts."""

    if worlds < len(SCENARIO_NAMES) * 2:
        raise ValueError("SSL Foundation scenario batch needs at least 16 worlds")
    rng = np.random.default_rng(seed)
    counts = _scenario_counts(worlds)
    family = np.concatenate(
        [np.full(count, index, dtype=np.int8) for index, count in enumerate(counts)]
    )
    rng.shuffle(family)
    focal_side = rng.integers(0, 2, size=worlds, dtype=np.int8)
    # Force exact side balance whenever the world count permits it.
    focal_side[rng.permutation(worlds)[: worlds // 2]] = 0
    focal_side[np.flatnonzero(focal_side == 0)[worlds // 2 :]] = 1
    if int((focal_side == 0).sum()) != worlds // 2:
        order = rng.permutation(worlds)
        focal_side[:] = 1
        focal_side[order[: worlds // 2]] = 0

    state = StateSnapshot.empty(worlds)
    state.on_ground.fill(1)
    state.car_pos[..., 2] = 17.0
    state.ball_pos[..., 2] = 93.15
    state.boost[:] = rng.uniform(15.0, 100.0, size=(worlds, 2)).astype(np.float32)
    kickoff_indicator = np.zeros(worlds, dtype=np.int32)

    for row, kind in enumerate(family):
        ball = state.ball_pos[row]
        cars = state.car_pos[row]
        velocities = state.car_vel[row]
        if kind == 0:  # natural ongoing 1v1
            ball[:2] = (rng.uniform(-3000, 3000), rng.uniform(-3600, 3600))
            ball[2] = rng.uniform(93.15, 500.0)
            state.ball_vel[row] = rng.uniform((-900, -1200, -150), (900, 1200, 700))
            cars[0, :2] = (rng.uniform(-3000, 3000), rng.uniform(-4200, 1800))
            cars[1, :2] = (rng.uniform(-3000, 3000), rng.uniform(-1800, 4200))
            velocities[:] = rng.uniform(-1200, 1200, size=(2, 3))
            velocities[:, 2] = 0.0
        elif kind == 1:  # genuinely loose ball and two access routes
            ball[:2] = (rng.uniform(-2600, 2600), rng.uniform(-2600, 2600))
            state.ball_vel[row, :2] = rng.uniform(-800, 800, size=2)
            for car, angle in enumerate((rng.uniform(-2.7, -0.4), rng.uniform(0.4, 2.7))):
                distance = rng.uniform(900, 2600)
                cars[car, :2] = ball[:2] + distance * np.asarray((np.cos(angle), np.sin(angle)))
                velocities[car, :2] = (
                    rng.uniform(200, 900)
                    * (ball[:2] - cars[car, :2])
                    / np.linalg.norm(ball[:2] - cars[car, :2])
                )
        elif kind == 2:  # catch/control/possession
            cars[0, :2] = (rng.uniform(-2200, 2200), rng.uniform(-3200, 800))
            speed = rng.uniform(350, 1250)
            heading = rng.uniform(0.85, 2.30)
            direction = np.asarray((np.cos(heading), np.sin(heading)))
            velocities[0, :2] = speed * direction
            ball[:2] = cars[0, :2] + rng.uniform(170, 360) * direction
            state.ball_vel[row, :2] = velocities[0, :2] + rng.uniform(-180, 180, size=2)
            cars[1, :2] = ball[:2] + np.asarray((rng.uniform(-1500, 1500), rng.uniform(1000, 2400)))
        elif kind == 3:  # shooting and finishing
            ball[:2] = (rng.uniform(-2200, 2200), rng.uniform(300, 3600))
            cars[0, :2] = ball[:2] + np.asarray((rng.uniform(-500, 500), -rng.uniform(450, 1100)))
            cars[1, :2] = (rng.uniform(-1700, 1700), rng.uniform(ball[1] + 350, 4700))
            velocities[0, :2] = (
                rng.uniform(500, 1400)
                * (ball[:2] - cars[0, :2])
                / np.linalg.norm(ball[:2] - cars[0, :2])
            )
            state.ball_vel[row, 1] = rng.uniform(100, 900)
        elif kind == 4:  # defensive shadow/save
            ball[:2] = (rng.uniform(-2500, 2500), rng.uniform(-4000, -1200))
            state.ball_vel[row, :2] = (rng.uniform(-450, 450), -rng.uniform(600, 1900))
            cars[0, :2] = (rng.uniform(-1800, 1800), rng.uniform(-4750, ball[1] - 250))
            cars[1, :2] = ball[:2] + np.asarray((rng.uniform(-500, 500), rng.uniform(350, 1100)))
            velocities[0, :2] = (
                rng.uniform(250, 1100)
                * (ball[:2] - cars[0, :2])
                / max(1.0, np.linalg.norm(ball[:2] - cars[0, :2]))
            )
        elif kind == 5:  # contested ball / 50
            ball[:2] = (rng.uniform(-1800, 1800), rng.uniform(-2200, 2200))
            offset = np.asarray((rng.uniform(-450, 450), rng.uniform(550, 1050)))
            cars[0, :2] = ball[:2] - offset
            cars[1, :2] = ball[:2] + offset
            for car in (0, 1):
                direction = ball[:2] - cars[car, :2]
                velocities[car, :2] = rng.uniform(600, 1500) * direction / np.linalg.norm(direction)
        elif kind == 6:  # wall and aerial opportunities
            wall = rng.random() < 0.5
            ball[:2] = (
                rng.choice((-1.0, 1.0)) * rng.uniform(2800, 3700)
                if wall
                else rng.uniform(-2200, 2200),
                rng.uniform(-1800, 2600),
            )
            ball[2] = rng.uniform(450, 1700)
            state.ball_vel[row] = rng.uniform((-500, 100, -100), (500, 1300, 750))
            cars[0, :2] = ball[:2] + np.asarray((rng.uniform(-700, 700), -rng.uniform(700, 1600)))
            cars[1, :2] = ball[:2] + np.asarray((rng.uniform(-1300, 1300), rng.uniform(900, 2200)))
            velocities[0, :2] = (
                rng.uniform(500, 1400)
                * (ball[:2] - cars[0, :2])
                / max(1.0, np.linalg.norm(ball[:2] - cars[0, :2]))
            )
        else:  # awkward recovery/scramble/low boost
            ball[:2] = (rng.uniform(-2800, 2800), rng.uniform(-3400, 3400))
            state.ball_vel[row] = rng.uniform((-1000, -1000, -200), (1000, 1000, 650))
            cars[0] = (rng.uniform(-3000, 3000), rng.uniform(-3800, 3800), rng.uniform(180, 900))
            cars[1, :2] = (rng.uniform(-3000, 3000), rng.uniform(-3800, 3800))
            state.on_ground[row, 0] = 0
            state.has_jumped[row, 0] = 1
            state.air_time[row, 0] = rng.uniform(0.05, 0.8)
            state.air_time_since_jump[row, 0] = state.air_time[row, 0]
            state.car_ang_vel[row, 0] = rng.uniform(-3.5, 3.5, size=3)
            state.car_quat[row, 0] = _euler_quaternion(
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-1.2, 1.2),
                rng.uniform(-np.pi, np.pi),
            )
            state.boost[row, 0] = rng.uniform(0, 20)

        for car in (0, 1):
            if not (kind == 7 and car == 0):
                state.car_quat[row, car] = _face(cars[car, :2], ball[:2])

    state.car_pos[..., 0] = np.clip(state.car_pos[..., 0], -3800.0, 3800.0)
    state.car_pos[..., 1] = np.clip(state.car_pos[..., 1], -4900.0, 4900.0)
    state.ball_pos[..., 0] = np.clip(state.ball_pos[..., 0], -3900.0, 3900.0)
    state.ball_pos[..., 1] = np.clip(state.ball_pos[..., 1], -5000.0, 5000.0)

    # Assign the focal role to either physical side while preserving the same
    # canonical geometry.  A 180-degree team rotation keeps each focal player
    # attacking +Y in its own observation.
    for row in np.flatnonzero(focal_side == 1):
        for name in (*CAR_VEC3_FIELDS, *CAR_FLOAT_FIELDS, *CAR_INT_FIELDS, "car_quat"):
            value = getattr(state, name)
            value[row] = value[row, ::-1].copy()
        state.car_pos[row, :, :2] *= -1.0
        state.car_vel[row, :, :2] *= -1.0
        state.car_ang_vel[row, :, :2] *= -1.0
        state.flip_rel_torque[row, :, :2] *= -1.0
        state.ball_pos[row, :2] *= -1.0
        state.ball_vel[row, :2] *= -1.0
        state.ball_ang_vel[row, :2] *= -1.0
        x, y, z, w = np.moveaxis(state.car_quat[row].copy(), -1, 0)
        state.car_quat[row] = np.stack((-y, x, w, -z), axis=-1)
        bx, by, bz, bw = state.ball_quat[row].copy()
        state.ball_quat[row] = np.asarray((-by, bx, bw, -bz), dtype=np.float32)

    state.validate()
    return SslFoundationScenarioBatch(state, family, focal_side, kickoff_indicator)


@wp.kernel(enable_backward=False)
def _apply_reset_template(
    reset_mask: wp.array(dtype=wp.int32),
    kickoff_flag: wp.array(dtype=wp.int32),
    kickoff_indicator: wp.array(dtype=wp.int32),
    kickoff_reset: wp.array(dtype=wp.int32),
    source_car_pos: wp.array(dtype=wp.vec3),
    source_car_vel: wp.array(dtype=wp.vec3),
    source_car_quat: wp.array(dtype=wp.quat),
    source_car_ang_vel: wp.array(dtype=wp.vec3),
    source_boost: wp.array(dtype=wp.float32),
    source_boosting_time: wp.array(dtype=wp.float32),
    source_time_since_boosted: wp.array(dtype=wp.float32),
    source_on_ground: wp.array(dtype=wp.int32),
    source_has_jumped: wp.array(dtype=wp.int32),
    source_is_jumping: wp.array(dtype=wp.int32),
    source_has_double_jumped: wp.array(dtype=wp.int32),
    source_has_flipped: wp.array(dtype=wp.int32),
    source_is_flipping: wp.array(dtype=wp.int32),
    source_sticky_ticks: wp.array(dtype=wp.int32),
    source_jump_time: wp.array(dtype=wp.float32),
    source_air_time: wp.array(dtype=wp.float32),
    source_air_time_since_jump: wp.array(dtype=wp.float32),
    source_flip_time: wp.array(dtype=wp.float32),
    source_flip_rel_torque: wp.array(dtype=wp.vec3),
    source_auto_flip_timer: wp.array(dtype=wp.float32),
    source_auto_flip_torque_scale: wp.array(dtype=wp.float32),
    source_is_auto_flipping: wp.array(dtype=wp.int32),
    source_is_boosting: wp.array(dtype=wp.int32),
    source_is_supersonic: wp.array(dtype=wp.int32),
    source_supersonic_time: wp.array(dtype=wp.float32),
    source_prev_throttle: wp.array(dtype=wp.float32),
    source_prev_steer: wp.array(dtype=wp.float32),
    source_prev_pitch: wp.array(dtype=wp.float32),
    source_prev_yaw: wp.array(dtype=wp.float32),
    source_prev_roll: wp.array(dtype=wp.float32),
    source_prev_jump: wp.array(dtype=wp.int32),
    source_prev_boost: wp.array(dtype=wp.int32),
    source_prev_handbrake: wp.array(dtype=wp.int32),
    source_ball_pos: wp.array(dtype=wp.vec3),
    source_ball_vel: wp.array(dtype=wp.vec3),
    source_ball_quat: wp.array(dtype=wp.quat),
    source_ball_ang_vel: wp.array(dtype=wp.vec3),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    boost: wp.array(dtype=wp.float32),
    boosting_time: wp.array(dtype=wp.float32),
    time_since_boosted: wp.array(dtype=wp.float32),
    on_ground: wp.array(dtype=wp.int32),
    air_control_disabled: wp.array(dtype=wp.int32),
    has_jumped: wp.array(dtype=wp.int32),
    is_jumping: wp.array(dtype=wp.int32),
    has_double_jumped: wp.array(dtype=wp.int32),
    has_flipped: wp.array(dtype=wp.int32),
    is_flipping: wp.array(dtype=wp.int32),
    sticky_ticks: wp.array(dtype=wp.int32),
    jump_time: wp.array(dtype=wp.float32),
    air_time: wp.array(dtype=wp.float32),
    air_time_since_jump: wp.array(dtype=wp.float32),
    flip_time: wp.array(dtype=wp.float32),
    flip_rel_torque: wp.array(dtype=wp.vec3),
    auto_flip_timer: wp.array(dtype=wp.float32),
    auto_flip_torque_scale: wp.array(dtype=wp.float32),
    is_auto_flipping: wp.array(dtype=wp.int32),
    is_boosting: wp.array(dtype=wp.int32),
    is_supersonic: wp.array(dtype=wp.int32),
    supersonic_time: wp.array(dtype=wp.float32),
    prev_throttle: wp.array(dtype=wp.float32),
    prev_steer: wp.array(dtype=wp.float32),
    prev_pitch: wp.array(dtype=wp.float32),
    prev_yaw: wp.array(dtype=wp.float32),
    prev_roll: wp.array(dtype=wp.float32),
    prev_jump: wp.array(dtype=wp.int32),
    prev_boost: wp.array(dtype=wp.int32),
    prev_handbrake: wp.array(dtype=wp.int32),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_quat: wp.array(dtype=wp.quat),
    ball_ang_vel: wp.array(dtype=wp.vec3),
    ball_position_bt: wp.array(dtype=wp.vec3),
    ball_velocity_bt: wp.array(dtype=wp.vec3),
    rigid_position_bt: wp.array(dtype=wp.vec3),
    rigid_velocity_bt: wp.array(dtype=wp.vec3),
    solver_position: wp.array(dtype=wp.vec3),
    solver_orientation: wp.array(dtype=wp.quat),
    solver_velocity: wp.array(dtype=wp.vec3),
    solver_angular_velocity: wp.array(dtype=wp.vec3),
    wheel_contact: wp.array(dtype=wp.int32),
    wheel_world_contact: wp.array(dtype=wp.int32),
    wheels_with_contact: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    handbrake_value: wp.array(dtype=wp.float32),
):
    env = wp.tid()
    if reset_mask[env] == 0:
        return
    kickoff_indicator[env] = kickoff_flag[env]
    kickoff_reset[env] = kickoff_flag[env]
    for local_car in range(2):
        car = env * 2 + local_car
        car_pos[car] = source_car_pos[car]
        car_vel[car] = source_car_vel[car]
        car_quat[car] = source_car_quat[car]
        car_ang_vel[car] = source_car_ang_vel[car]
        boost[car] = source_boost[car]
        boosting_time[car] = source_boosting_time[car]
        time_since_boosted[car] = source_time_since_boosted[car]
        on_ground[car] = source_on_ground[car]
        air_control_disabled[car] = 0
        has_jumped[car] = source_has_jumped[car]
        is_jumping[car] = source_is_jumping[car]
        has_double_jumped[car] = source_has_double_jumped[car]
        has_flipped[car] = source_has_flipped[car]
        is_flipping[car] = source_is_flipping[car]
        sticky_ticks[car] = source_sticky_ticks[car]
        jump_time[car] = source_jump_time[car]
        air_time[car] = source_air_time[car]
        air_time_since_jump[car] = source_air_time_since_jump[car]
        flip_time[car] = source_flip_time[car]
        flip_rel_torque[car] = source_flip_rel_torque[car]
        auto_flip_timer[car] = source_auto_flip_timer[car]
        auto_flip_torque_scale[car] = source_auto_flip_torque_scale[car]
        is_auto_flipping[car] = source_is_auto_flipping[car]
        is_boosting[car] = source_is_boosting[car]
        is_supersonic[car] = source_is_supersonic[car]
        supersonic_time[car] = source_supersonic_time[car]
        prev_throttle[car] = source_prev_throttle[car]
        prev_steer[car] = source_prev_steer[car]
        prev_pitch[car] = source_prev_pitch[car]
        prev_yaw[car] = source_prev_yaw[car]
        prev_roll[car] = source_prev_roll[car]
        prev_jump[car] = source_prev_jump[car]
        prev_boost[car] = source_prev_boost[car]
        prev_handbrake[car] = source_prev_handbrake[car]
        rigid_position_bt[car] = source_car_pos[car] * 0.02
        rigid_velocity_bt[car] = source_car_vel[car] * 0.02
        solver_position[car] = source_car_pos[car]
        solver_orientation[car] = source_car_quat[car]
        solver_velocity[car] = source_car_vel[car]
        solver_angular_velocity[car] = source_car_ang_vel[car]
        wheels_with_contact[car] = 0
        contact_count[car] = 0
        handbrake_value[car] = 0.0
        for wheel in range(4):
            wheel_contact[car * 4 + wheel] = 0
            wheel_world_contact[car * 4 + wheel] = 0
    ball_pos[env] = source_ball_pos[env]
    ball_vel[env] = source_ball_vel[env]
    ball_quat[env] = source_ball_quat[env]
    ball_ang_vel[env] = source_ball_ang_vel[env]
    ball_position_bt[env] = source_ball_pos[env] * 0.02
    ball_velocity_bt[env] = source_ball_vel[env] * 0.02


class SslFoundationResetTemplate:
    """GPU-resident immutable reset-state bank, one template per world."""

    def __init__(self, batch: SslFoundationScenarioBatch, *, device: str):
        self.state = GpuState(batch.state, device)
        self.kickoff_indicator = wp.array(batch.kickoff_indicator, dtype=wp.int32, device=device)
        self.summary = batch.summary()
        self.logical_bytes = batch.state.nbytes + batch.kickoff_indicator.nbytes

    def apply(self, world: Any, reset_mask: wp.array) -> None:
        source = self.state
        state = world.state
        vehicle = world.vehicle
        wp.launch(
            _apply_reset_template,
            dim=world.num_envs,
            inputs=[
                reset_mask,
                self.kickoff_indicator,
                world.rival2.kickoff_indicator,
                world.lifecycle.kickoff_reset,
                source.car_pos,
                source.car_vel,
                source.car_quat,
                source.car_ang_vel,
                source.boost,
                source.boosting_time,
                source.time_since_boosted,
                source.on_ground,
                source.has_jumped,
                source.is_jumping,
                source.has_double_jumped,
                source.has_flipped,
                source.is_flipping,
                source.sticky_ticks,
                source.jump_time,
                source.air_time,
                source.air_time_since_jump,
                source.flip_time,
                source.flip_rel_torque,
                source.auto_flip_timer,
                source.auto_flip_torque_scale,
                source.is_auto_flipping,
                source.is_boosting,
                source.is_supersonic,
                source.supersonic_time,
                source.prev_throttle,
                source.prev_steer,
                source.prev_pitch,
                source.prev_yaw,
                source.prev_roll,
                source.prev_jump,
                source.prev_boost,
                source.prev_handbrake,
                source.ball_pos,
                source.ball_vel,
                source.ball_quat,
                source.ball_ang_vel,
                state.car_pos,
                state.car_vel,
                state.car_quat,
                state.car_ang_vel,
                state.boost,
                state.boosting_time,
                state.time_since_boosted,
                state.on_ground,
                state.air_control_disabled,
                state.has_jumped,
                state.is_jumping,
                state.has_double_jumped,
                state.has_flipped,
                state.is_flipping,
                state.sticky_ticks,
                state.jump_time,
                state.air_time,
                state.air_time_since_jump,
                state.flip_time,
                state.flip_rel_torque,
                state.auto_flip_timer,
                state.auto_flip_torque_scale,
                state.is_auto_flipping,
                state.is_boosting,
                state.is_supersonic,
                state.supersonic_time,
                state.prev_throttle,
                state.prev_steer,
                state.prev_pitch,
                state.prev_yaw,
                state.prev_roll,
                state.prev_jump,
                state.prev_boost,
                state.prev_handbrake,
                state.ball_pos,
                state.ball_vel,
                state.ball_quat,
                state.ball_ang_vel,
                world.ball_world.position_bt,
                world.ball_world.velocity_bt,
                vehicle.rigid_position_bt,
                vehicle.rigid_velocity_bt,
                vehicle.solver_position,
                vehicle.solver_orientation,
                vehicle.solver_velocity,
                vehicle.solver_angular_velocity,
                vehicle.wheel_contact,
                vehicle.wheel_world_contact,
                vehicle.wheels_with_contact,
                vehicle.contact_count,
                vehicle.handbrake_value,
            ],
            device=world.device,
        )


__all__ = [
    "SCENARIO_NAMES",
    "SCENARIO_PROBABILITIES",
    "SSL_FOUNDATION_GAMMA",
    "SSL_FOUNDATION_WEIGHTS",
    "SslFoundationPotentials",
    "SslFoundationResetTemplate",
    "SslFoundationScenarioBatch",
    "build_ssl_foundation_scenarios",
    "ssl_foundation_potentials",
    "ssl_foundation_shaping",
]
