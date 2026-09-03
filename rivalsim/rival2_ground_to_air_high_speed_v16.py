"""Measured high-speed ground-to-air continuation states for V3 expansion.

The physical ranges are derived from the exact V15 natural handoff diagnosis.
They intentionally cover the fast, low-boost rising-ball states that V23 visits
but the original V11 scorer curriculum did not. No action sequence, controller,
or reward is encoded in the state builder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rivalsim.state import StateSnapshot

GROUND_TO_AIR_HIGH_SPEED_V16_VERSION = "RIVAL2_GROUND_TO_AIR_HIGH_SPEED_V16"


def _yaw_quat(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.asarray((0.0, 0.0, math.sin(half), math.cos(half)), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class HighSpeedGroundToAirBatch:
    state: StateSnapshot
    attacker_side: int
    initial_planar_gap_uu: np.ndarray
    initial_ball_goalward_speed_uu_per_second: np.ndarray
    initial_car_goalward_speed_uu_per_second: np.ndarray
    initial_boost_fraction: np.ndarray
    initial_opponent_ball_distance_uu: np.ndarray


def build_high_speed_ground_to_air_scenarios(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
) -> HighSpeedGroundToAirBatch:
    """Create side-symmetric, action-free replicas of natural V23 entries."""

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid high-speed ground-to-air scenario request")
    rng = np.random.default_rng(seed)
    sign = 1.0 if attacker_side == 0 else -1.0
    other = 1 - attacker_side
    forward = _yaw_quat(sign * math.pi / 2.0)
    reverse = _yaw_quat(-sign * math.pi / 2.0)
    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0 / 3.0)

    planar_gap = np.zeros(worlds, dtype=np.float32)
    opponent_distance = np.zeros(worlds, dtype=np.float32)
    for world in range(worlds):
        ball_y = float(rng.uniform(1_650.0, 3_200.0))
        longitudinal_gap = float(rng.uniform(370.0, 445.0))
        lateral_gap = float(rng.uniform(-245.0, 245.0))
        # Keep every reset inside the measured broad router rather than
        # depending on rejection sampling at environment construction.
        maximum_lateral = math.sqrt(max(475.0**2 - longitudinal_gap**2, 0.0))
        lateral_gap = float(np.clip(lateral_gap, -maximum_lateral, maximum_lateral))
        ball_x = float(rng.uniform(-1_000.0, 1_000.0))
        car_speed = float(rng.uniform(1_450.0, 1_950.0))
        ball_speed = float(rng.uniform(-450.0, 1_300.0))
        boost = float(rng.uniform(22.0, 50.0))

        state.ball_pos[world] = (
            ball_x,
            sign * ball_y,
            float(rng.uniform(145.0, 225.0)),
        )
        state.ball_vel[world] = (
            float(rng.uniform(-90.0, 90.0)),
            sign * ball_speed,
            float(rng.uniform(235.0, 350.0)),
        )
        state.ball_ang_vel[world] = rng.uniform(-1.5, 1.5, size=3)
        state.car_pos[world, attacker_side] = (
            ball_x - lateral_gap,
            sign * (ball_y - longitudinal_gap),
            17.0,
        )
        state.car_vel[world, attacker_side] = (
            float(rng.uniform(-120.0, 120.0)),
            sign * car_speed,
            0.0,
        )
        state.car_quat[world, attacker_side] = forward
        state.boost[world, attacker_side] = boost

        defender_longitudinal = float(rng.uniform(1_350.0, 2_050.0))
        defender_lateral = float(rng.uniform(-450.0, 450.0))
        defender_y = min(ball_y + defender_longitudinal, 4_450.0)
        state.car_pos[world, other] = (
            ball_x + defender_lateral,
            sign * defender_y,
            17.0,
        )
        state.car_vel[world, other] = (
            float(rng.uniform(-200.0, 200.0)),
            -sign * float(rng.uniform(0.0, 650.0)),
            0.0,
        )
        state.car_quat[world, other] = reverse
        state.boost[world, other] = float(rng.uniform(20.0, 100.0))

        planar_gap[world] = math.hypot(longitudinal_gap, lateral_gap)
        opponent_distance[world] = math.hypot(
            defender_y - ball_y,
            defender_lateral,
        )

    state.validate()
    return HighSpeedGroundToAirBatch(
        state=state,
        attacker_side=attacker_side,
        initial_planar_gap_uu=planar_gap,
        initial_ball_goalward_speed_uu_per_second=(
            sign * state.ball_vel[:, 1]
        ).astype(np.float32),
        initial_car_goalward_speed_uu_per_second=(
            sign * state.car_vel[:, attacker_side, 1]
        ).astype(np.float32),
        initial_boost_fraction=(state.boost[:, attacker_side] / 100.0).astype(
            np.float32
        ),
        initial_opponent_ball_distance_uu=opponent_distance,
    )


__all__ = [
    "GROUND_TO_AIR_HIGH_SPEED_V16_VERSION",
    "HighSpeedGroundToAirBatch",
    "build_high_speed_ground_to_air_scenarios",
]
