"""Low-relative-speed ground-ball setups for a catchable pop-and-follow entry."""

from __future__ import annotations

import math

import numpy as np

from rivalsim.state import StateSnapshot

GROUND_BALL_SOFT_POP_VERSION = "RIVAL2_GROUND_BALL_SOFT_POP_V1"


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)),
        dtype=np.float32,
    )


def build_ground_ball_soft_pop_scenarios(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
) -> StateSnapshot:
    """Create possession-like approaches with matched car and ball momentum."""

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid soft-pop scenario request")
    rng = np.random.default_rng(seed)
    sign = 1.0 if attacker_side == 0 else -1.0
    other = 1 - attacker_side
    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0)
    forward = _yaw_quat(sign * math.pi / 2.0)
    reverse = _yaw_quat(-sign * math.pi / 2.0)
    for world in range(worlds):
        x = float(rng.uniform(-750.0, 750.0))
        y = float(rng.uniform(1_800.0, 2_800.0))
        ball_speed = float(rng.uniform(400.0, 900.0))
        approach_distance = float(rng.uniform(190.0, 320.0))
        relative_closing_speed = float(rng.uniform(80.0, 280.0))
        state.ball_pos[world] = (x, sign * y, 92.75)
        state.ball_vel[world] = (
            float(rng.uniform(-18.0, 18.0)),
            sign * ball_speed,
            0.0,
        )
        state.car_pos[world, attacker_side] = (
            x + float(rng.uniform(-15.0, 15.0)),
            sign * (y - approach_distance),
            17.0,
        )
        state.car_quat[world, attacker_side] = forward
        state.car_vel[world, attacker_side] = (
            float(rng.uniform(-15.0, 15.0)),
            sign * (ball_speed + relative_closing_speed),
            0.0,
        )
        state.car_pos[world, other] = (
            float(rng.uniform(-850.0, 850.0)),
            -sign * float(rng.uniform(3_900.0, 4_500.0)),
            17.0,
        )
        state.car_quat[world, other] = reverse
    state.validate()
    return state


__all__ = ["GROUND_BALL_SOFT_POP_VERSION", "build_ground_ball_soft_pop_scenarios"]
