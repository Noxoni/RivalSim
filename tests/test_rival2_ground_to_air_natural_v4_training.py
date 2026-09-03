from __future__ import annotations

import copy
import json

from benchmarks import run_rival2_ground_to_air_natural_v4 as natural


def test_prospective_authority_and_bound_inputs_are_exact() -> None:
    authority = natural.load_authority()
    assert natural.capability.sha256_file(natural.AUTHORITY) == natural.AUTHORITY_SHA256
    for identity in authority["bound_inputs"].values():
        assert (
            natural.capability.sha256_file(natural.ROOT / identity["path"])
            == identity["sha256"]
        )
    assert authority["integrity"]["optimizer_steps_before_authority_commit"] == 0
    assert authority["integrity"]["test_seed_unopened_before_validation_pass"]


def test_authority_uses_natural_entries_live_defender_and_real_boost_floor() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    assert authority["scenario"]["setup_families"] == [
        "low_bounce",
        "incoming_chip",
        "matched_dribble",
    ]
    assert authority["scenario"]["training_live_defender_fraction"] == 0.5
    assert authority["scenario"]["training_boost_range"][0] == 20.0
    assert authority["option_config"]["minimum_boost_fraction"] == 0.2
    assert authority["mechanics_interpretation"]["dead_ball_vertical_launcher"] == (
        "rejected and excluded"
    )


def test_training_cannot_reward_airtime_or_mutate_general_policies() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    assert authority["reward"]["raw_airtime_reward"] == 0.0
    assert authority["integrity"]["production_reward_unchanged"]
    assert authority["integrity"]["protected_v23_unchanged"]
    assert authority["integrity"]["controlled_scorer_parent_unchanged"]
    assert authority["integrity"]["live_defender_is_inference_only"]


def _passing_rows(authority: dict) -> list[dict]:
    rows = []
    for defender in ("parked", "live"):
        gate = authority["acceptance"]["per_defender_mode"][defender]
        for setup in authority["scenario"]["setup_families"]:
            for side in (0, 1):
                fractions = {
                    "pop_touch": gate["pop_touch_fraction_min"],
                    "elevated_follow_touch": gate[
                        "elevated_follow_touch_fraction_min"
                    ],
                    "high_follow_touch": gate["high_follow_touch_fraction_min"],
                    "productive_continuation": gate[
                        "productive_continuation_fraction_min"
                    ],
                    "goal_within_contact_budget": gate[
                        "goal_within_contact_budget_fraction_min"
                    ],
                    "goalward_velocity_contact": 0.2,
                    "second_airborne_touch": 0.02,
                    "sustained_control": 0.02,
                    "contact_budget_exceeded": 0.0,
                    "unassisted_or_ground_goal": 0.0,
                }
                rows.append(
                    {
                        "setup": setup,
                        "defender_mode": defender,
                        "side": side,
                        "fractions": fractions,
                        "finite": True,
                        "analog_saturation_fraction": [0.0] * 5,
                    }
                )
    return rows


def test_promotion_gate_requires_every_setup_side_and_defender_mode() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    rows = _passing_rows(authority)
    assert natural.passes_gate(rows, authority)
    failed = copy.deepcopy(rows)
    failed[-1]["fractions"]["goal_within_contact_budget"] = 0.0
    assert not natural.passes_gate(failed, authority)


def test_live_defender_rows_have_extra_selection_weight() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    rows = _passing_rows(authority)
    baseline = natural.evaluation_score(rows)
    parked = copy.deepcopy(rows)
    live = copy.deepcopy(rows)
    for row in parked:
        if row["defender_mode"] == "parked":
            row["fractions"]["goal_within_contact_budget"] += 0.01
    for row in live:
        if row["defender_mode"] == "live":
            row["fractions"]["goal_within_contact_budget"] += 0.01
    assert natural.evaluation_score(parked) > baseline
    assert natural.evaluation_score(live) - baseline > (
        natural.evaluation_score(parked) - baseline
    )
