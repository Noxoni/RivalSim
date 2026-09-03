from __future__ import annotations

import json

from benchmarks import run_rival2_ground_to_air_natural_v7 as v7
from benchmarks import run_rival2_ground_to_air_natural_v9 as v9


def _authority() -> dict[str, object]:
    return json.loads(v9.AUTHORITY.read_text(encoding="utf-8"))


def test_v9_authority_is_prospective_bound_and_uses_original_parent() -> None:
    authority = v9.load_authority()
    assert v9.balanced.capability.sha256_file(v9.AUTHORITY) == v9.AUTHORITY_SHA256
    for identity in authority["bound_inputs"].values():
        assert (
            v9.balanced.capability.sha256_file(v9.balanced.ROOT / identity["path"])
            == identity["sha256"]
        )
    assert authority["integrity"]["optimizer_steps_before_authority_commit"] == 0
    assert authority["integrity"]["v7_diagnostic_descendant_not_used"]
    assert authority["integrity"]["test_seed_unopened_before_validation_pass"]


def test_v9_uses_selected_200ms_timing_and_natural_entries() -> None:
    authority = _authority()
    assert authority["option_config"]["first_jump_hold_ticks"] == 24
    assert authority["option_config"]["jump_release_ticks"] == 4
    assert authority["option_config"]["boost_during_pop"] is False
    assert authority["scenario"]["setup_families"] == [
        "low_bounce",
        "incoming_chip",
        "matched_dribble",
    ]
    assert authority["mechanics_interpretation"]["dead_ball_vertical_launcher"] == (
        "rejected and excluded"
    )


def test_v9_preserves_bounded_physical_learning_contract() -> None:
    authority = _authority()
    assert authority["reward"]["raw_airtime_reward"] == 0.0
    assert authority["episode"]["maximum_distinct_chain_contacts"] == 6
    assert authority["pop_orientation_control"] == {
        "pitch_center": 0.5,
        "pitch_residual_scale": 0.0,
        "steer_scale": 0.25,
        "yaw_scale": 0.35,
        "roll_scale": 0.35,
    }
    assert len(authority["scenario"]["training_strata"]) == 12
    assert authority["training"]["success_volume_rehearsal"] is False
    assert authority["integrity"]["named_mechanic_classifier_used"] is False
    assert v9._OVERRIDES["VALIDATION_PHYSICAL_PROBE"] is True


def test_v9_configuration_does_not_mutate_v7_module_defaults() -> None:
    original = {
        "VERSION": v7.VERSION,
        "AUTHORITY": v7.AUTHORITY,
        "RESULTS": v7.RESULTS,
        "CHECKPOINTS": v7.CHECKPOINTS,
    }
    v9.load_authority()
    assert {name: getattr(v7, name) for name in original} == original
