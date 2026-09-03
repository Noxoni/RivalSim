"""Human-calibrated natural entry feeds for the V11 aerial curriculum.

The entry feeds deliberately separate four physically different opportunities:
an assisted low bounce, a soft incoming-ball chip, a rising-bounce double-jump
contact, and a true roof carry.  No feed prescribes or rewards a controller
animation.  In particular, the roof carry permits either a plain double jump
or a partial tornado/front-corner solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rivalsim.rival2_ground_to_air_natural_v4 import (
    DEFENDER_LIVE,
    DEFENDER_MIXED,
    DEFENDER_MODES,
    DEFENDER_PARKED,
)
from rivalsim.state import StateSnapshot

GROUND_TO_AIR_ENTRY_V11_VERSION = "RIVAL2_GROUND_TO_AIR_ENTRY_V11"

SETUP_ASSISTED_LOW_BOUNCE = 0
SETUP_SOFT_INCOMING_CHIP = 1
SETUP_RISING_DOUBLE_JUMP = 2
SETUP_ROOF_CARRY = 3
SETUP_NAMES = (
    "assisted_low_bounce",
    "soft_incoming_chip",
    "rising_double_jump",
    "roof_carry",
)

# The default is intentionally dominated by the assisted opportunity shown in
# the supplied human clip.  A prospective campaign authority may bind another
# mixture explicitly without changing the scenario semantics.
DEFAULT_SETUP_WEIGHTS = (0.55, 0.20, 0.15, 0.10)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray(
        (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)),
        dtype=np.float32,
    )


def _lerp(easy: float, broad: float, difficulty: float) -> float:
    return easy + (broad - easy) * difficulty


def _range(
    rng: np.random.Generator,
    easy: tuple[float, float],
    broad: tuple[float, float],
    difficulty: float,
) -> float:
    return float(
        rng.uniform(
            _lerp(easy[0], broad[0], difficulty),
            _lerp(easy[1], broad[1], difficulty),
        )
    )


@dataclass(frozen=True, slots=True)
class GroundToAirEntryScenarioBatch:
    state: StateSnapshot
    setup: np.ndarray
    defender_active: np.ndarray
    attacker_side: int
    initial_goalward_gap_uu: np.ndarray
    initial_closing_speed_uu_per_second: np.ndarray
    ball_initially_rising: np.ndarray


def build_ground_to_air_entry_scenarios(
    worlds: int,
    *,
    seed: int,
    attacker_side: int,
    setup: int | None = None,
    setup_weights: tuple[float, ...] = DEFAULT_SETUP_WEIGHTS,
    difficulty: float = 0.0,
    defender_mode: str = DEFENDER_PARKED,
    live_defender_fraction: float = 0.5,
    attacker_boost_range: tuple[float, float] = (35.0, 100.0),
) -> GroundToAirEntryScenarioBatch:
    """Build deterministic, side-symmetric natural aerial-entry opportunities.

    ``difficulty=0`` supplies the bounded first-learning distribution.
    ``difficulty=1`` broadens the same physical families without changing their
    meaning.  Interpolation inside that interval supports a prospective staged
    authority; the live V10 campaign never imports this module.
    """

    if worlds <= 0 or attacker_side not in (0, 1):
        raise ValueError("invalid V11 ground-to-air entry request")
    if setup is not None and setup not in range(len(SETUP_NAMES)):
        raise ValueError("unknown V11 ground-to-air entry setup")
    weights = np.asarray(setup_weights, dtype=np.float64)
    if (
        weights.shape != (len(SETUP_NAMES),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError("setup weights must be finite nonnegative values")
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be finite and inside [0, 1]")
    if defender_mode not in DEFENDER_MODES:
        raise ValueError("unknown V11 defender mode")
    if not 0.0 <= live_defender_fraction <= 1.0:
        raise ValueError("live defender fraction must be in [0, 1]")
    boost_low, boost_high = attacker_boost_range
    if boost_low < 0.0 or boost_high > 100.0 or boost_low > boost_high:
        raise ValueError("attacker boost range must be ordered inside [0, 100]")

    rng = np.random.default_rng(seed)
    side_sign = 1.0 if attacker_side == 0 else -1.0
    other = 1 - attacker_side
    forward = _yaw_quat(side_sign * math.pi / 2.0)
    reverse = _yaw_quat(-side_sign * math.pi / 2.0)
    probabilities = weights / weights.sum()
    setup_id = (
        np.full(worlds, setup, dtype=np.int32)
        if setup is not None
        else rng.choice(
            len(SETUP_NAMES), size=worlds, replace=True, p=probabilities
        ).astype(np.int32)
    )
    if defender_mode == DEFENDER_LIVE:
        defender_active = np.ones(worlds, dtype=np.bool_)
    elif defender_mode == DEFENDER_PARKED:
        defender_active = np.zeros(worlds, dtype=np.bool_)
    else:
        defender_active = rng.random(worlds) < live_defender_fraction

    state = StateSnapshot.empty(worlds)
    state.car_pos[..., 2] = 17.0
    state.on_ground.fill(1)
    state.boost.fill(100.0)

    for world in range(worlds):
        family = int(setup_id[world])
        x = float(rng.uniform(-1_100.0, 1_100.0))
        canonical_y = float(rng.uniform(900.0, 3_250.0))
        lateral = _range(
            rng,
            (-18.0, 18.0),
            (-45.0, 45.0),
            difficulty,
        )

        if family == SETUP_ASSISTED_LOW_BOUNCE:
            gap = _range(rng, (175.0, 260.0), (150.0, 330.0), difficulty)
            ball_speed = _range(rng, (280.0, 550.0), (180.0, 760.0), difficulty)
            car_speed = ball_speed + _range(
                rng, (45.0, 120.0), (20.0, 220.0), difficulty
            )
            ball_height = _range(
                rng, (118.0, 165.0), (108.0, 205.0), difficulty
            )
            ball_vertical_speed = _range(
                rng, (120.0, 320.0), (60.0, 420.0), difficulty
            )
            ball_longitudinal_speed = ball_speed
            ball_lateral = _range(rng, (-30.0, 30.0), (-70.0, 70.0), difficulty)
        elif family == SETUP_SOFT_INCOMING_CHIP:
            gap = _range(rng, (220.0, 340.0), (170.0, 380.0), difficulty)
            incoming_speed = _range(
                rng, (80.0, 240.0), (60.0, 360.0), difficulty
            )
            car_speed = _range(rng, (80.0, 300.0), (40.0, 420.0), difficulty)
            ball_height = _range(rng, (94.0, 100.0), (93.0, 106.0), difficulty)
            ball_vertical_speed = _range(
                rng, (-10.0, 20.0), (-30.0, 60.0), difficulty
            )
            ball_longitudinal_speed = -incoming_speed
            ball_lateral = _range(rng, (-18.0, 18.0), (-45.0, 45.0), difficulty)
        elif family == SETUP_RISING_DOUBLE_JUMP:
            gap = _range(rng, (150.0, 225.0), (130.0, 300.0), difficulty)
            ball_speed = _range(rng, (280.0, 580.0), (180.0, 760.0), difficulty)
            car_speed = ball_speed + _range(
                rng, (-10.0, 90.0), (-80.0, 150.0), difficulty
            )
            ball_height = _range(
                rng, (150.0, 215.0), (125.0, 285.0), difficulty
            )
            ball_vertical_speed = _range(
                rng, (200.0, 400.0), (80.0, 580.0), difficulty
            )
            ball_longitudinal_speed = ball_speed
            ball_lateral = _range(rng, (-25.0, 25.0), (-70.0, 70.0), difficulty)
        else:
            # A true carry begins over the rear half of the roof, not several
            # car lengths in front.  No action sequence is encoded here.
            gap = _range(rng, (-35.0, -10.0), (-50.0, 10.0), difficulty)
            car_speed = _range(rng, (250.0, 500.0), (120.0, 750.0), difficulty)
            ball_longitudinal_speed = car_speed + _range(
                rng, (-15.0, 15.0), (-45.0, 45.0), difficulty
            )
            ball_height = _range(rng, (128.0, 133.0), (126.0, 136.0), difficulty)
            ball_vertical_speed = _range(
                rng, (-5.0, 15.0), (-25.0, 35.0), difficulty
            )
            ball_lateral = _range(rng, (-8.0, 8.0), (-22.0, 22.0), difficulty)

        state.car_pos[world, attacker_side] = (
            x + lateral,
            side_sign * (canonical_y - gap),
            17.0,
        )
        state.car_quat[world, attacker_side] = forward
        state.car_vel[world, attacker_side] = (
            _range(rng, (-25.0, 25.0), (-55.0, 55.0), difficulty),
            side_sign * car_speed,
            0.0,
        )
        state.boost[world, attacker_side] = float(rng.uniform(boost_low, boost_high))

        state.ball_pos[world] = (
            x + ball_lateral,
            side_sign * canonical_y,
            ball_height,
        )
        state.ball_vel[world] = (
            _range(rng, (-20.0, 20.0), (-55.0, 55.0), difficulty),
            side_sign * ball_longitudinal_speed,
            ball_vertical_speed,
        )
        state.ball_ang_vel[world] = (
            _range(rng, (-0.6, 0.6), (-2.0, 2.0), difficulty),
            _range(rng, (-0.6, 0.6), (-2.0, 2.0), difficulty),
            _range(rng, (-0.4, 0.4), (-1.2, 1.2), difficulty),
        )

        if defender_active[world]:
            defender_y = min(
                canonical_y + float(rng.uniform(750.0, 1_450.0)),
                4_450.0,
            )
            state.car_pos[world, other] = (
                x + float(rng.uniform(-700.0, 700.0)),
                side_sign * defender_y,
                17.0,
            )
            state.car_vel[world, other] = (
                float(rng.uniform(-150.0, 150.0)),
                -side_sign * float(rng.uniform(0.0, 450.0)),
                0.0,
            )
        else:
            state.car_pos[world, other] = (
                float(rng.uniform(-1_000.0, 1_000.0)),
                -side_sign * float(rng.uniform(3_700.0, 4_500.0)),
                17.0,
            )
        state.car_quat[world, other] = reverse
        state.boost[world, other] = float(rng.uniform(20.0, 100.0))

    state.validate()
    goalward_gap = side_sign * (
        state.ball_pos[:, 1] - state.car_pos[:, attacker_side, 1]
    )
    closing_speed = side_sign * (
        state.car_vel[:, attacker_side, 1] - state.ball_vel[:, 1]
    )
    return GroundToAirEntryScenarioBatch(
        state=state,
        setup=setup_id,
        defender_active=defender_active,
        attacker_side=attacker_side,
        initial_goalward_gap_uu=goalward_gap.astype(np.float32),
        initial_closing_speed_uu_per_second=closing_speed.astype(np.float32),
        ball_initially_rising=(state.ball_vel[:, 2] > 0.0),
    )


__all__ = [
    "DEFAULT_SETUP_WEIGHTS",
    "DEFENDER_LIVE",
    "DEFENDER_MIXED",
    "DEFENDER_PARKED",
    "GROUND_TO_AIR_ENTRY_V11_VERSION",
    "SETUP_ASSISTED_LOW_BOUNCE",
    "SETUP_NAMES",
    "SETUP_RISING_DOUBLE_JUMP",
    "SETUP_ROOF_CARRY",
    "SETUP_SOFT_INCOMING_CHIP",
    "GroundToAirEntryScenarioBatch",
    "build_ground_to_air_entry_scenarios",
]
