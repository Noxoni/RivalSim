from __future__ import annotations

import numpy as np

from rivalsim.rival2_ground_to_air_high_speed_v16 import (
    build_high_speed_ground_to_air_scenarios,
)


def test_high_speed_scenarios_are_deterministic_and_inside_measured_ranges() -> None:
    first = build_high_speed_ground_to_air_scenarios(
        128, seed=2026090320, attacker_side=0
    )
    second = build_high_speed_ground_to_air_scenarios(
        128, seed=2026090320, attacker_side=0
    )

    assert np.array_equal(first.state.car_pos, second.state.car_pos)
    assert np.array_equal(first.state.ball_vel, second.state.ball_vel)
    assert np.all((first.initial_planar_gap_uu >= 370.0) & (first.initial_planar_gap_uu <= 475.01))
    assert np.all(
        (first.initial_car_goalward_speed_uu_per_second >= 1_450.0)
        & (first.initial_car_goalward_speed_uu_per_second <= 1_950.0)
    )
    assert np.all(
        (first.initial_boost_fraction >= 0.22)
        & (first.initial_boost_fraction <= 0.50)
    )
    assert np.all(first.state.on_ground[:, 0] == 1)


def test_high_speed_scenarios_are_side_symmetric_in_canonical_axes() -> None:
    blue = build_high_speed_ground_to_air_scenarios(
        64, seed=2026090321, attacker_side=0
    )
    orange = build_high_speed_ground_to_air_scenarios(
        64, seed=2026090321, attacker_side=1
    )

    assert np.allclose(blue.state.ball_pos[:, 1], -orange.state.ball_pos[:, 1])
    assert np.allclose(blue.state.ball_vel[:, 1], -orange.state.ball_vel[:, 1])
    assert np.allclose(
        blue.initial_car_goalward_speed_uu_per_second,
        orange.initial_car_goalward_speed_uu_per_second,
    )
    assert np.array_equal(blue.initial_planar_gap_uu, orange.initial_planar_gap_uu)
