from __future__ import annotations

import numpy as np
import pytest

from rivalsim.rival2_ground_to_air_entry_v11 import (
    DEFAULT_SETUP_WEIGHTS,
    DEFENDER_LIVE,
    DEFENDER_MIXED,
    SETUP_ASSISTED_LOW_BOUNCE,
    SETUP_NAMES,
    SETUP_RISING_DOUBLE_JUMP,
    SETUP_ROOF_CARRY,
    SETUP_SOFT_INCOMING_CHIP,
    build_ground_to_air_entry_scenarios,
)


@pytest.mark.parametrize("setup", range(len(SETUP_NAMES)))
def test_entry_feeds_are_deterministic_and_exactly_side_symmetric(setup: int) -> None:
    blue = build_ground_to_air_entry_scenarios(
        128,
        seed=20260903,
        attacker_side=0,
        setup=setup,
        difficulty=0.35,
    )
    blue_again = build_ground_to_air_entry_scenarios(
        128,
        seed=20260903,
        attacker_side=0,
        setup=setup,
        difficulty=0.35,
    )
    orange = build_ground_to_air_entry_scenarios(
        128,
        seed=20260903,
        attacker_side=1,
        setup=setup,
        difficulty=0.35,
    )

    assert np.array_equal(blue.state.ball_pos, blue_again.state.ball_pos)
    assert np.array_equal(blue.state.ball_vel, blue_again.state.ball_vel)
    assert np.array_equal(blue.state.car_pos, blue_again.state.car_pos)
    assert np.array_equal(blue.state.car_vel, blue_again.state.car_vel)
    assert np.array_equal(blue.setup, blue_again.setup)
    assert np.all(blue.setup == setup)

    assert np.array_equal(blue.state.ball_pos[:, 0], orange.state.ball_pos[:, 0])
    assert np.array_equal(blue.state.ball_pos[:, 1], -orange.state.ball_pos[:, 1])
    assert np.array_equal(blue.state.ball_pos[:, 2], orange.state.ball_pos[:, 2])
    assert np.array_equal(blue.state.ball_vel[:, 0], orange.state.ball_vel[:, 0])
    assert np.array_equal(blue.state.ball_vel[:, 1], -orange.state.ball_vel[:, 1])
    assert np.array_equal(blue.state.ball_vel[:, 2], orange.state.ball_vel[:, 2])
    assert np.array_equal(
        blue.state.car_pos[:, 0, 0], orange.state.car_pos[:, 1, 0]
    )
    assert np.array_equal(
        blue.state.car_pos[:, 0, 1], -orange.state.car_pos[:, 1, 1]
    )
    assert np.array_equal(
        blue.state.car_vel[:, 0, 1], -orange.state.car_vel[:, 1, 1]
    )
    assert np.array_equal(
        blue.initial_goalward_gap_uu, orange.initial_goalward_gap_uu
    )
    assert np.array_equal(
        blue.initial_closing_speed_uu_per_second,
        orange.initial_closing_speed_uu_per_second,
    )


def test_assisted_low_bounce_is_rising_and_requires_only_a_light_catchup() -> None:
    batch = build_ground_to_air_entry_scenarios(
        2_048,
        seed=11,
        attacker_side=0,
        setup=SETUP_ASSISTED_LOW_BOUNCE,
        difficulty=0.0,
    )
    assert np.all(batch.ball_initially_rising)
    assert np.all(batch.state.ball_pos[:, 2] >= 118.0)
    assert np.all(batch.state.ball_pos[:, 2] <= 165.0)
    assert np.all(batch.initial_goalward_gap_uu >= 175.0)
    assert np.all(batch.initial_goalward_gap_uu <= 260.0)
    assert np.all(batch.initial_closing_speed_uu_per_second >= 45.0)
    assert np.all(batch.initial_closing_speed_uu_per_second <= 120.0)


def test_soft_incoming_chip_removes_the_v10_extreme_closing_speed() -> None:
    easy = build_ground_to_air_entry_scenarios(
        4_096,
        seed=12,
        attacker_side=0,
        setup=SETUP_SOFT_INCOMING_CHIP,
        difficulty=0.0,
    )
    broad = build_ground_to_air_entry_scenarios(
        4_096,
        seed=12,
        attacker_side=0,
        setup=SETUP_SOFT_INCOMING_CHIP,
        difficulty=1.0,
    )
    assert np.all(easy.state.ball_vel[:, 1] < 0.0)
    assert np.all(easy.state.car_vel[:, 0, 1] > 0.0)
    assert float(easy.initial_closing_speed_uu_per_second.min()) >= 160.0
    assert float(easy.initial_closing_speed_uu_per_second.max()) <= 540.0
    assert float(broad.initial_closing_speed_uu_per_second.min()) >= 100.0
    assert float(broad.initial_closing_speed_uu_per_second.max()) <= 780.0


def test_rising_double_jump_feed_starts_above_and_climbing() -> None:
    batch = build_ground_to_air_entry_scenarios(
        1_024,
        seed=13,
        attacker_side=1,
        setup=SETUP_RISING_DOUBLE_JUMP,
        difficulty=0.0,
    )
    assert np.all(batch.ball_initially_rising)
    assert np.all(batch.state.ball_pos[:, 2] >= 150.0)
    assert np.all(batch.state.ball_vel[:, 2] >= 200.0)
    assert np.all(batch.initial_goalward_gap_uu > 0.0)


@pytest.mark.parametrize("side", [0, 1])
def test_roof_carry_uses_rear_half_geometry_and_matched_velocity(side: int) -> None:
    batch = build_ground_to_air_entry_scenarios(
        2_048,
        seed=14,
        attacker_side=side,
        setup=SETUP_ROOF_CARRY,
        difficulty=0.0,
    )
    sign = 1.0 if side == 0 else -1.0
    relative_forward_velocity = sign * (
        batch.state.ball_vel[:, 1] - batch.state.car_vel[:, side, 1]
    )
    lateral_offset = batch.state.ball_pos[:, 0] - batch.state.car_pos[:, side, 0]
    assert np.all(batch.initial_goalward_gap_uu >= -35.0)
    assert np.all(batch.initial_goalward_gap_uu <= -10.0)
    assert np.all(np.abs(lateral_offset) <= 26.0)
    assert np.all(batch.state.ball_pos[:, 2] >= 128.0)
    assert np.all(batch.state.ball_pos[:, 2] <= 133.0)
    assert np.all(relative_forward_velocity >= -15.0)
    assert np.all(relative_forward_velocity <= 15.0)


def test_default_mixture_is_low_bounce_dominant_but_contains_every_family() -> None:
    batch = build_ground_to_air_entry_scenarios(
        20_000,
        seed=15,
        attacker_side=0,
    )
    counts = np.bincount(batch.setup, minlength=len(SETUP_NAMES))
    fractions = counts / counts.sum()
    assert set(np.flatnonzero(counts).tolist()) == set(range(len(SETUP_NAMES)))
    assert fractions[SETUP_ASSISTED_LOW_BOUNCE] > 0.53
    assert fractions[SETUP_ASSISTED_LOW_BOUNCE] < 0.57
    assert DEFAULT_SETUP_WEIGHTS[SETUP_ASSISTED_LOW_BOUNCE] == max(
        DEFAULT_SETUP_WEIGHTS
    )


def test_live_and_mixed_defenders_preserve_goal_side_context() -> None:
    for mode in (DEFENDER_LIVE, DEFENDER_MIXED):
        batch = build_ground_to_air_entry_scenarios(
            2_000,
            seed=16,
            attacker_side=1,
            defender_mode=mode,
            live_defender_fraction=0.5,
        )
        if mode == DEFENDER_LIVE:
            assert np.all(batch.defender_active)
        else:
            assert np.any(batch.defender_active)
            assert np.any(~batch.defender_active)
        live = batch.defender_active
        goalward_gap = -1.0 * (
            batch.state.car_pos[:, 0, 1] - batch.state.ball_pos[:, 1]
        )
        assert np.all(goalward_gap[live] > 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"worlds": 0},
        {"attacker_side": 2},
        {"setup": 99},
        {"setup_weights": (1.0, 0.0)},
        {"setup_weights": (0.0, 0.0, 0.0, 0.0)},
        {"difficulty": -0.1},
        {"difficulty": 1.1},
        {"defender_mode": "unknown"},
        {"live_defender_fraction": 1.1},
        {"attacker_boost_range": (70.0, 20.0)},
    ],
)
def test_invalid_entry_request_fails_closed(kwargs: dict[str, object]) -> None:
    request: dict[str, object] = {
        "worlds": 1,
        "seed": 17,
        "attacker_side": 0,
    }
    request.update(kwargs)
    with pytest.raises(ValueError):
        build_ground_to_air_entry_scenarios(**request)  # type: ignore[arg-type]
