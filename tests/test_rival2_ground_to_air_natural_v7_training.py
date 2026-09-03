from __future__ import annotations

import json

from benchmarks import run_rival2_ground_to_air_natural_v7 as natural


def _authority() -> dict[str, object]:
    return json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))


def test_v7_authority_is_prospective_and_bound() -> None:
    authority = natural.load_authority()
    assert natural.capability.sha256_file(natural.AUTHORITY) == natural.AUTHORITY_SHA256
    for identity in authority["bound_inputs"].values():
        assert (
            natural.capability.sha256_file(natural.ROOT / identity["path"])
            == identity["sha256"]
        )
    assert authority["integrity"]["optimizer_steps_before_authority_commit"] == 0
    assert authority["integrity"]["test_seed_unopened_before_validation_pass"]


def test_v7_preserves_pitch_and_only_learns_narrow_pop_orientation() -> None:
    authority = _authority()
    orientation = authority["pop_orientation_control"]
    assert orientation == {
        "pitch_center": 0.5,
        "pitch_residual_scale": 0.0,
        "steer_scale": 0.25,
        "yaw_scale": 0.35,
        "roll_scale": 0.35,
    }
    assert authority["option_config"]["pop_pitch"] == 0.5
    assert authority["integrity"]["fixed_pop_pitch_preserved"]
    assert authority["integrity"]["only_active_pop_channels_in_likelihood"]


def test_v7_balances_every_setup_defender_side_stratum() -> None:
    authority = _authority()
    strata = natural.natural_v6.training_strata(authority)
    assert len(strata) == 12
    assert {
        (row["setup"], row["defender_mode"], row["side"]) for row in strata
    } == {
        (setup, defender, side)
        for setup in natural.SETUP_NAMES
        for defender in (natural.DEFENDER_PARKED, natural.DEFENDER_LIVE)
        for side in (0, 1)
    }
    assert authority["training"]["gradient_aggregation"] == (
        "equal mean of twelve stratum PPO losses before every optimizer step"
    )
    assert authority["training"]["success_volume_rehearsal"] is False


def test_v7_matches_researched_entry_families_without_airtime_reward() -> None:
    authority = _authority()
    mechanics = authority["mechanics_interpretation"]
    assert mechanics["dead_ball_vertical_launcher"] == "rejected and excluded"
    assert "light forward" in mechanics["low_bounce"]
    assert "rolling ball" in mechanics["incoming_chip"]
    assert "second jump" in mechanics["matched_dribble"]
    assert "front corner" in mechanics["partial_tornado_corner_touch"]
    assert authority["reward"]["raw_airtime_reward"] == 0.0
    assert authority["episode"]["maximum_distinct_chain_contacts"] == 6
    assert authority["reward"]["over_contact_budget_failure"] == -12.0
    assert authority["integrity"]["named_mechanic_classifier_used"] is False


def test_v7_keeps_worst_physical_row_as_primary_selection() -> None:
    authority = _authority()
    assert authority["selection"]["primary"] == (
        "maximum worst-row ratio to its frozen minimum gate"
    )
    assert authority["selection"]["test_not_used"]
    assert authority["acceptance"]["validation_and_untouched_test_must_both_pass"]
