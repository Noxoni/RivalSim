from __future__ import annotations

import numpy as np
import pytest

from rivalsim.rival2_ground_to_air_natural_v4 import (
    SETUP_INCOMING_CHIP,
    SETUP_LOW_BOUNCE,
    SETUP_MATCHED_DRIBBLE,
    SETUP_NAMES,
    build_natural_ground_to_air_scenarios,
)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("setup", range(len(SETUP_NAMES)))
def test_natural_ground_to_air_scenarios_are_deterministic_and_side_aware(
    side: int, setup: int
) -> None:
    first = build_natural_ground_to_air_scenarios(
        32, seed=1234, attacker_side=side, setup=setup
    )
    second = build_natural_ground_to_air_scenarios(
        32, seed=1234, attacker_side=side, setup=setup
    )
    assert np.array_equal(first.setup, second.setup)
    assert np.array_equal(first.state.ball_pos, second.state.ball_pos)
    assert np.array_equal(first.state.ball_vel, second.state.ball_vel)
    assert np.array_equal(first.state.car_pos, second.state.car_pos)
    assert np.array_equal(first.state.car_vel, second.state.car_vel)
    assert np.all(first.setup == setup)
    sign = 1.0 if side == 0 else -1.0
    gap = sign * (
        first.state.ball_pos[:, 1] - first.state.car_pos[:, side, 1]
    )
    assert np.all(gap > 0.0)
    assert np.all(first.state.boost[:, side] >= 35.0)
    assert np.all(first.state.boost[:, side] <= 100.0)


def test_setup_families_encode_the_intended_ball_motion() -> None:
    bounce = build_natural_ground_to_air_scenarios(
        128, seed=7, attacker_side=0, setup=SETUP_LOW_BOUNCE
    )
    incoming = build_natural_ground_to_air_scenarios(
        128, seed=7, attacker_side=0, setup=SETUP_INCOMING_CHIP
    )
    dribble = build_natural_ground_to_air_scenarios(
        128, seed=7, attacker_side=0, setup=SETUP_MATCHED_DRIBBLE
    )
    assert np.all(bounce.state.ball_vel[:, 1] > 0.0)
    assert np.all(incoming.state.ball_vel[:, 1] < 0.0)
    assert np.all(dribble.state.ball_vel[:, 1] > 0.0)
    assert np.all(bounce.state.ball_pos[:, 2] > incoming.state.ball_pos[:, 2])


def test_random_mixture_contains_every_setup_family() -> None:
    batch = build_natural_ground_to_air_scenarios(
        3_000, seed=99, attacker_side=0
    )
    assert set(np.unique(batch.setup).tolist()) == set(range(len(SETUP_NAMES)))


def test_invalid_scenario_request_fails_closed() -> None:
    with pytest.raises(ValueError):
        build_natural_ground_to_air_scenarios(
            0, seed=1, attacker_side=0, setup=SETUP_LOW_BOUNCE
        )
    with pytest.raises(ValueError):
        build_natural_ground_to_air_scenarios(1, seed=1, attacker_side=2)
    with pytest.raises(ValueError):
        build_natural_ground_to_air_scenarios(1, seed=1, attacker_side=0, setup=99)
