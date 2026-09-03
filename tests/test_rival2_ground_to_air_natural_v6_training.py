from __future__ import annotations

import json

import torch

from benchmarks import run_rival2_ground_to_air_natural_v6 as natural


def test_v6_authority_is_prospective_and_bound() -> None:
    authority = natural.load_authority()
    assert natural.capability.sha256_file(natural.AUTHORITY) == natural.AUTHORITY_SHA256
    for identity in authority["bound_inputs"].values():
        assert (
            natural.capability.sha256_file(natural.ROOT / identity["path"])
            == identity["sha256"]
        )
    assert authority["integrity"]["optimizer_steps_before_authority_commit"] == 0
    assert authority["integrity"]["test_seed_unopened_before_validation_pass"]


def test_v6_has_exactly_twelve_balanced_physical_strata() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    assert len(natural.training_strata(authority)) == 12
    assert authority["training"]["success_volume_rehearsal"] is False
    assert authority["training"]["gradient_aggregation"].startswith("equal mean")


def test_channel_log_probability_sums_to_existing_hybrid_contract() -> None:
    torch.manual_seed(3)
    actor = torch.randn((11, 13))
    action = torch.cat(
        (torch.tanh(torch.randn((11, 5))), torch.randint(0, 2, (11, 3)).float()),
        dim=-1,
    )
    pre_tanh = torch.atanh(action[:, :5].clamp(-0.999999, 0.999999))
    config = natural.natural_v4.Rival2PolicyConfig()
    distribution = natural.HybridDistributionOverride(
        analog_log_std=-1.7,
        button_temperature=0.9,
    )
    channels = natural.hybrid_channel_log_probability(
        actor,
        action,
        pre_tanh=pre_tanh,
        config=config,
        distribution=distribution,
    )
    expected = natural.natural_v4.hybrid_log_probability(
        actor,
        action,
        pre_tanh=pre_tanh,
        config=config,
        distribution_override=distribution,
    )
    assert channels.shape == (11, 8)
    assert torch.allclose(channels.sum(dim=-1), expected, atol=1.0e-6, rtol=1.0e-6)


def test_selection_prioritizes_worst_physical_row() -> None:
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
                            "goalward_velocity_contact": 0.1,
                            "second_airborne_touch": 0.01,
                            "sustained_control": 0.01,
                            "contact_budget_exceeded": 0.0,
                            "unassisted_or_ground_goal": 0.0,
                        },
                    }
                )
    assert natural.minimum_gate_ratio(rows, authority) == 1.0
    rows[0]["fractions"]["high_follow_touch"] *= 0.5
    assert natural.minimum_gate_ratio(rows, authority) == 0.5


def test_v6_keeps_outcome_and_integrity_boundaries() -> None:
    authority = json.loads(natural.AUTHORITY.read_text(encoding="utf-8"))
    assert authority["reward"]["raw_airtime_reward"] == 0.0
    assert authority["episode"]["maximum_distinct_chain_contacts"] == 6
    assert authority["integrity"]["named_mechanic_classifier_used"] is False
    assert authority["integrity"]["production_reward_unchanged"]
