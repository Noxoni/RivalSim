"""Natural low-bounce and incoming-chip ground-to-air training scenarios.

These scenarios model the setup families a player actually converts into a
ground-to-air play.  They deliberately do not ask the policy to launch a dead,
resting ball vertically.  Instead, the ball is already in a low bounce, arrives
toward the car for a chip, or sits slightly ahead of a matched-speed dribble so
that a light first touch and double-jump follow create separation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rivalsim.state import StateSnapshot

GROUND_TO_AIR_NATURAL_V4_VERSION = "RIVAL2_GROUND_TO_AIR_NATURAL_V4"
SETUP_LOW_BOUNCE = 0
SETUP_INCOMING_CHIP = 1
SETUP_MATCHED_DRIBBLE = 2
SETUP_NAMES = ("low_bounce", "incoming_chip", "matched_dribble")


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)),
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class NaturalGroundToAirScenarioBatch:
    state: StateSnapshot
    setup: np.ndarray
    attacker_side: int


def build_natural_ground_to_air_scenarios(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
    setup: int | None = None,
) -> NaturalGroundToAirScenarioBatch:
    """Build low-bounce, incoming-chip, and matched dribble opportunities.

    All three setup families begin in an attacking orientation with the ball
    close enough for a light physical setup touch.  The other car is parked far
    away so this stage measures setup/takeoff geometry before adding a live
    defender.
    """

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid natural ground-to-air scenario request")
    if setup is not None and setup not in range(len(SETUP_NAMES)):
        raise ValueError("unknown natural ground-to-air setup")
    rng = np.random.default_rng(seed)
    side_sign = 1.0 if attacker_side == 0 else -1.0
    other = 1 - attacker_side
    forward = _yaw_quat(side_sign * math.pi / 2.0)
    reverse = _yaw_quat(-side_sign * math.pi / 2.0)
    setup_id = (
        np.full(worlds, setup, dtype=np.int32)
        if setup is not None
        else rng.integers(0, len(SETUP_NAMES), size=worlds, dtype=np.int32)
    )

    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0)
    for world in range(worlds):
        family = int(setup_id[world])
        x = float(rng.uniform(-1_100.0, 1_100.0))
        canonical_y = float(rng.uniform(900.0, 3_300.0))
        lateral = float(rng.uniform(-28.0, 28.0))
        if family == SETUP_LOW_BOUNCE:
            gap = float(rng.uniform(65.0, 155.0))
            ball_speed = float(rng.uniform(250.0, 720.0))
            car_speed = float(ball_speed + rng.uniform(35.0, 180.0))
            ball_height = float(rng.uniform(112.0, 188.0))
            ball_vertical_speed = float(rng.uniform(-260.0, 220.0))
            ball_longitudinal_speed = ball_speed
        elif family == SETUP_INCOMING_CHIP:
            gap = float(rng.uniform(90.0, 165.0))
            incoming_speed = float(rng.uniform(220.0, 720.0))
            car_speed = float(rng.uniform(350.0, 900.0))
            ball_height = float(rng.uniform(93.0, 108.0))
            ball_vertical_speed = float(rng.uniform(-25.0, 45.0))
            ball_longitudinal_speed = -incoming_speed
        else:
            gap = float(rng.uniform(38.0, 92.0))
            ball_speed = float(rng.uniform(280.0, 700.0))
            car_speed = float(ball_speed + rng.uniform(-30.0, 65.0))
            ball_height = float(rng.uniform(105.0, 142.0))
            ball_vertical_speed = float(rng.uniform(-70.0, 70.0))
            ball_longitudinal_speed = ball_speed

        state.ball_pos[world] = (x, side_sign * canonical_y, ball_height)
        state.ball_vel[world] = (
            float(rng.uniform(-55.0, 55.0)),
            side_sign * ball_longitudinal_speed,
            ball_vertical_speed,
        )
        state.car_pos[world, attacker_side] = (
            x + lateral,
            side_sign * (canonical_y - gap),
            17.0,
        )
        state.car_quat[world, attacker_side] = forward
        state.car_vel[world, attacker_side] = (
            float(rng.uniform(-35.0, 35.0)),
            side_sign * car_speed,
            0.0,
        )
        state.boost[world, attacker_side] = float(rng.uniform(35.0, 100.0))

        state.car_pos[world, other] = (
            float(rng.uniform(-1_000.0, 1_000.0)),
            -side_sign * float(rng.uniform(3_700.0, 4_500.0)),
            17.0,
        )
        state.car_quat[world, other] = reverse
        state.boost[world, other] = float(rng.uniform(20.0, 100.0))

    state.validate()
    return NaturalGroundToAirScenarioBatch(
        state=state,
        setup=setup_id,
        attacker_side=attacker_side,
    )


__all__ = [
    "GROUND_TO_AIR_NATURAL_V4_VERSION",
    "SETUP_INCOMING_CHIP",
    "SETUP_LOW_BOUNCE",
    "SETUP_MATCHED_DRIBBLE",
    "SETUP_NAMES",
    "NaturalGroundToAirScenarioBatch",
    "build_natural_ground_to_air_scenarios",
]
