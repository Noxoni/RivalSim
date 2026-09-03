from __future__ import annotations

from benchmarks.run_rival2_ground_to_air_entry_probe_v11 import (
    human_envelope_config,
    summarize,
)
from rivalsim.rival2_ground_to_air_entry_v11 import SETUP_NAMES


def _row(setup: str, side: int, defender: str, value: float) -> dict:
    return {
        "setup": setup,
        "side": side,
        "defender_mode": defender,
        "difficulty": 0.0,
        "telemetry": {
            "fractions": {
                "first_contact": value,
                "entry_airborne_contact": value / 2.0,
                "human_envelope_reached": value / 3.0,
                "second_airborne_contact": value / 4.0,
                "goal_within_contact_budget": value / 5.0,
                "contact_budget_exceeded": 0.0,
                "ball_ground_failure": 1.0 - value,
            }
        },
    }


def test_summary_keeps_every_setup_and_averages_all_perspectives() -> None:
    rows = [
        _row(setup, side, defender, 0.2 + 0.2 * side)
        for setup in SETUP_NAMES
        for defender in ("parked", "live")
        for side in (0, 1)
    ]
    result = summarize(rows)["difficulty_0.000"]
    assert tuple(result) == SETUP_NAMES
    for setup in SETUP_NAMES:
        assert result[setup]["rows"] == 4
        assert abs(result[setup]["first_contact"] - 0.3) < 1.0e-7
        assert abs(result[setup]["entry_airborne_contact"] - 0.15) < 1.0e-7
        assert abs(result[setup]["ball_ground_failure"] - 0.7) < 1.0e-7


def test_summary_fails_closed_when_a_setup_is_missing() -> None:
    try:
        summarize([_row(SETUP_NAMES[0], 0, "parked", 0.5)])
    except ValueError as error:
        assert "missing setup" in str(error)
    else:
        raise AssertionError("incomplete calibration rows were accepted")


def test_human_envelope_is_bound_to_source_measured_ranges() -> None:
    config = human_envelope_config()
    config.validate()
    assert config.target_car_height_uu == 141.16314697265625
    assert config.target_ball_height_uu == 273.7956237792969
    assert config.target_car_vertical_speed_uu_per_second == 443.40289306640625
    assert config.target_distance_uu == 140.35697369650788
    assert config.maximum_event_distance_uu == 157.0
