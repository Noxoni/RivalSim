from __future__ import annotations

import copy
import json

from benchmarks import run_rival2_ground_to_air_timing_v8 as timing
from rivalsim.rival2_ground_to_air_timing_v8 import (
    TIMING_CANDIDATES,
    authority_with_timing,
)


def test_timing_v8_authority_is_no_learning_and_bound() -> None:
    authority = timing.load_authority()
    assert timing.capability.sha256_file(timing.AUTHORITY) == timing.AUTHORITY_SHA256
    for identity in authority["bound_inputs"].values():
        assert (
            timing.capability.sha256_file(timing.ROOT / identity["path"])
            == identity["sha256"]
        )
    assert authority["integrity"]["optimizer_steps"] == 0
    assert authority["integrity"]["policy_parameters_mutated"] is False
    assert authority["selection"]["no_checkpoint_promotion"]
    assert authority["selection"]["no_test_corpus"]


def test_timing_candidates_match_frozen_120hz_sweep() -> None:
    authority = json.loads(timing.AUTHORITY.read_text(encoding="utf-8"))
    assert authority["calibration"]["physics_hz"] == 120
    assert authority["calibration"]["timing_candidates"] == [
        candidate.record() for candidate in TIMING_CANDIDATES
    ]
    assert [candidate.first_jump_hold_ticks for candidate in TIMING_CANDIDATES] == [
        8,
        12,
        16,
        20,
        24,
    ]
    assert all(candidate.jump_release_ticks > 0 for candidate in TIMING_CANDIDATES)


def test_timing_override_changes_only_hold_and_release() -> None:
    base = json.loads(timing.V7_AUTHORITY.read_text(encoding="utf-8"))
    original = copy.deepcopy(base)
    candidate = TIMING_CANDIDATES[-1]
    updated = authority_with_timing(base, candidate)
    assert base == original
    assert updated["option_config"]["first_jump_hold_ticks"] == 24
    assert updated["option_config"]["jump_release_ticks"] == 4
    updated["option_config"]["first_jump_hold_ticks"] = original["option_config"][
        "first_jump_hold_ticks"
    ]
    updated["option_config"]["jump_release_ticks"] = original["option_config"][
        "jump_release_ticks"
    ]
    assert updated == original


def _row(*, height: float, close: float, takeoff: float) -> dict[str, object]:
    return {
        "defender_mode": "parked",
        "fractions": {
            "pop_touch": 1.0,
            "elevated_follow_touch": 0.0,
            "high_follow_touch": 0.0,
            "productive_continuation": 0.0,
            "goal_within_contact_budget": 0.0,
            "sustained_control": 0.0,
            "second_airborne_touch": 0.0,
            "goalward_velocity_contact": 0.0,
            "contact_budget_exceeded": 0.0,
            "unassisted_or_ground_goal": 0.0,
        },
        "finite": True,
        "analog_saturation_fraction": [0.0] * 5,
        "physical_probe": {
            "post_pop_within_160uu_fraction": close,
            "takeoff_after_pop_fraction": takeoff,
            "maximum_self_height_uu": {"p50": height},
            "maximum_self_vertical_speed_uu_per_second": {"p50": 500.0},
            "minimum_post_pop_ball_distance_uu": {"p50": 100.0},
        },
    }


def test_timing_selection_prefers_broad_close_takeoff_before_height() -> None:
    authority = json.loads(timing.V7_AUTHORITY.read_text(encoding="utf-8"))
    close = [_row(height=180.0, close=0.8, takeoff=1.0) for _ in range(2)]
    tall_but_far = [_row(height=400.0, close=0.2, takeoff=1.0) for _ in range(2)]
    assert timing.timing_selection_key(close, authority) > timing.timing_selection_key(
        tall_but_far, authority
    )


def test_timing_v8_forbids_reward_or_scenario_broadening() -> None:
    authority = json.loads(timing.AUTHORITY.read_text(encoding="utf-8"))
    integrity = authority["integrity"]
    assert integrity["production_reward_unchanged"]
    assert integrity["natural_scenario_geometry_unchanged"]
    assert integrity["dead_ball_vertical_launcher_used"] is False
    assert integrity["raw_airtime_reward_used"] is False
    assert integrity["named_mechanic_reward_or_detector_used"] is False


def test_timing_v8_result_follows_frozen_selection_and_preserves_parents() -> None:
    result_path = timing.RESULTS / "result.json"
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selected = max(
        result["candidates"], key=lambda item: tuple(item["selection_key"])
    )
    assert result["selected_candidate"] == selected["candidate"]
    assert result["selected_candidate"]["name"] == "hold_24_release_4"
    assert result["optimizer_steps"] == 0
    assert result["parent_unchanged"]
    assert result["protected_v23_unchanged"]
    assert result["untouched_test_opened"] is False
