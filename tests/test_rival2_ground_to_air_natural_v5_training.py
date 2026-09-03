from __future__ import annotations

import json

from benchmarks import run_rival2_ground_to_air_natural_v5 as natural


def test_v5_authority_is_prospective_and_bound() -> None:
    authority = natural.load_authority()
    assert natural.capability.sha256_file(natural.AUTHORITY) == natural.AUTHORITY_SHA256
    for identity in authority["bound_inputs"].values():
        assert (
            natural.capability.sha256_file(natural.ROOT / identity["path"])
            == identity["sha256"]
        )
    assert authority["integrity"]["optimizer_steps_before_authority_commit"] == 0
    assert authority["integrity"]["test_seed_unopened_before_validation_pass"]


def test_every_setup_defender_stratum_has_equal_optimizer_weight() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    strata = natural.training_strata(authority)
    assert len(strata) == 6
    assert {row["setup"] for row in strata} == set(natural.SETUP_NAMES)
    assert {row["defender_mode"] for row in strata} == {"parked", "live"}
    assert {row["optimizer_weight"] for row in strata} == {1.0}
    assert authority["training"]["advantage_normalization"] == (
        "independent_within_each_setup_defender_side_rollout"
    )


def test_v5_matches_requested_entry_mechanics_and_contact_limit() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    mechanics = authority["mechanics_interpretation"]
    assert mechanics["dead_ball_vertical_launcher"] == "rejected and excluded"
    assert "light forward touch" in mechanics["low_bounce"]
    assert "rolling" in mechanics["incoming_chip"]
    assert "double jump" in mechanics["matched_dribble"]
    assert authority["episode"]["maximum_distinct_chain_contacts"] == 6
    assert authority["reward"]["over_contact_budget_failure"] == -12.0


def test_v5_does_not_reward_airtime_or_change_production_reward() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    assert authority["reward"]["raw_airtime_reward"] == 0.0
    assert authority["integrity"]["production_reward_unchanged"]
    assert authority["integrity"]["named_mechanic_classifier_used"] is False
    assert authority["integrity"]["controlled_scorer_parent_unchanged"]
    assert authority["integrity"]["protected_v23_unchanged"]


def test_v5_gate_remains_all_rows_and_not_average_only() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    rows = []
    for defender in ("parked", "live"):
        gate = authority["acceptance"]["per_defender_mode"][defender]
        for setup in authority["scenario"]["setup_families"]:
            for side in (0, 1):
                rows.append(
                    {
                        "setup": setup,
                        "defender_mode": defender,
                        "side": side,
                        "fractions": {
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
                            "contact_budget_exceeded": 0.0,
                            "unassisted_or_ground_goal": 0.0,
                        },
                        "finite": True,
                        "analog_saturation_fraction": [0.0] * 5,
                    }
                )
    assert natural.natural_v4.passes_gate(rows, authority)
    rows[0]["fractions"]["high_follow_touch"] = 0.0
    assert not natural.natural_v4.passes_gate(rows, authority)
