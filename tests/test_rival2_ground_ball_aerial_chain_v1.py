from __future__ import annotations

import copy
import json

from benchmarks import run_rival2_ground_ball_aerial_chain_v1 as campaign


def _row() -> dict:
    return {
        "fractions": {
            "source_launch": 1.0,
            "pop_touch": 1.0,
            "ball_rise_180": 0.95,
            "ball_rise_250": 0.55,
            "elevated_follow_touch": 0.10,
            "high_follow_touch": 0.02,
            "second_airborne_touch": 0.015,
            "productive_continuation": 0.02,
            "goal_within_contact_budget": 0.015,
            "contact_budget_exceeded": 0.0,
            "unassisted_or_ground_goal": 0.20,
        },
        "analog_saturation_fraction": [0.2] * 5,
        "reward_per_attempt": 1.0,
        "finite": True,
    }


def test_authority_binds_calibrated_source_launch() -> None:
    authority = json.loads(campaign.AUTHORITY.read_text(encoding="utf-8"))
    config = campaign.source_config(authority)
    assert authority["source_launch"]["selected_candidate"] == 5
    assert config.trigger_distance_uu == 185.0
    assert config.pitch == 1.0
    assert config.use_approach_boost
    assert config.learned_start_tick == 15


def test_gate_requires_connected_aerial_outcome() -> None:
    authority = json.loads(campaign.AUTHORITY.read_text(encoding="utf-8"))
    rows = [_row(), _row()]
    assert campaign.passes_gate(rows, authority)
    failed = copy.deepcopy(rows)
    failed[1]["fractions"]["elevated_follow_touch"] = 0.0
    assert not campaign.passes_gate(failed, authority)


def test_score_prefers_aerial_goal_over_ground_goal() -> None:
    baseline = [_row(), _row()]
    improved = copy.deepcopy(baseline)
    for row in improved:
        row["fractions"]["goal_within_contact_budget"] += 0.01
        row["fractions"]["unassisted_or_ground_goal"] -= 0.01
    assert campaign.evaluation_score(improved) > campaign.evaluation_score(baseline)
