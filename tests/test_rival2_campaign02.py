from benchmarks.build_rival2_campaign02_evidence import classify_behavior
from benchmarks.run_rival2_campaign02 import (
    CAMPAIGN02_ENTROPY_COEFFICIENT,
    CAMPAIGN02_WORLDS,
    EXPECTED_INITIALIZATION_SHA256,
    campaign02_ppo_config,
    frozen_configuration,
    ppo_configuration_differences,
)


def test_campaign02_changes_only_the_entropy_coefficient() -> None:
    configuration = frozen_configuration()
    differences = ppo_configuration_differences(
        configuration["campaign01_ppo_config"], configuration["ppo_config"]
    )
    assert differences == {
        "entropy_coefficient": {"campaign01": 0.01, "campaign02": 0.0}
    }
    assert campaign02_ppo_config().entropy_coefficient == CAMPAIGN02_ENTROPY_COEFFICIENT
    assert configuration["worlds"] == CAMPAIGN02_WORLDS == 131072
    assert configuration["campaign_seed"] == 20260826
    assert configuration["evaluation"]["seed"] == 920260826
    assert configuration["expected_initialization_model_sha256"] == (
        EXPECTED_INITIALIZATION_SHA256
    )
    assert configuration["evaluation"]["protocol_sha256"] == (
        "964D7281C9BB8EF12C4A831B984015259A777D82285A02EAB329FBB6CC098CE7"
    )


def test_campaign02_prospective_improved_classification() -> None:
    initialization = {"touches": 1.0, "goals": 0.0, "touch_diff": 2.0}
    campaign01 = {"touches": 0.5, "goals": -2.0, "touch_diff": -3.0}
    campaign02 = {"touches": 1.1, "goals": 1.0, "touch_diff": 2.0}
    result = classify_behavior(initialization, campaign01, campaign02)
    assert result["behavioral_result"] == "IMPROVED"
    assert result["improved_vs_initialization_count"] == 2
    assert result["not_worse_than_campaign01_on_all_three"]


def test_campaign02_prospective_degraded_and_inconclusive_classification() -> None:
    initialization = {"touches": 1.0, "goals": 0.0, "touch_diff": 2.0}
    campaign01 = {"touches": 0.5, "goals": -2.0, "touch_diff": -3.0}
    degraded = classify_behavior(
        initialization,
        campaign01,
        {"touches": 0.9, "goals": -1.0, "touch_diff": 2.0},
    )
    assert degraded["behavioral_result"] == "DEGRADED"
    inconclusive = classify_behavior(
        initialization,
        campaign01,
        {"touches": 1.1, "goals": 0.0, "touch_diff": 2.0},
    )
    assert inconclusive["behavioral_result"] == "INCONCLUSIVE"
