from benchmarks.run_rival2_campaign01 import (
    CAPACITY_ORDER,
    EXPECTED_CONTRACT_HASHES,
    EXPECTED_POLICY_CONFIG_HASH,
    TARGET_SAMPLES,
    first_update_at_or_above,
    frozen_configuration,
    threshold_label_for_samples,
)


def test_campaign01_configuration_freezes_v05_authority() -> None:
    configuration = frozen_configuration()
    assert configuration["contract_hashes"] == EXPECTED_CONTRACT_HASHES
    assert configuration["policy_config_hash"] == EXPECTED_POLICY_CONFIG_HASH
    assert configuration["ppo_config"] == {
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "value_loss_coefficient": 0.50,
        "entropy_coefficient": 0.01,
        "max_gradient_norm": 0.50,
        "learning_rate": 3e-4,
        "epochs": 2,
        "rollout_horizon": 32,
        "minibatch_size": 65536,
    }
    assert configuration["capacity_order"] == list(CAPACITY_ORDER)
    assert configuration["target_agent_decision_samples"] == TARGET_SAMPLES
    assert len(configuration["evaluation"]["protocol_sha256"]) == 64


def test_campaign01_threshold_schedule_stops_at_first_crossing_update() -> None:
    assert first_update_at_or_above(131072, 10_000_000) == (2, 16_777_216)
    assert first_update_at_or_above(131072, 25_000_000) == (3, 25_165_824)
    assert first_update_at_or_above(131072, 50_000_000) == (6, 50_331_648)
    assert first_update_at_or_above(131072, 100_000_000) == (12, 100_663_296)
    assert first_update_at_or_above(65536, 100_000_000) == (24, 100_663_296)
    assert first_update_at_or_above(32768, 100_000_000) == (48, 100_663_296)


def test_campaign01_threshold_label_is_monotonic() -> None:
    assert threshold_label_for_samples(0) == "000m"
    assert threshold_label_for_samples(9_999_999) == "000m"
    assert threshold_label_for_samples(10_000_000) == "010m"
    assert threshold_label_for_samples(25_000_000) == "025m"
    assert threshold_label_for_samples(50_000_000) == "050m"
    assert threshold_label_for_samples(100_000_000) == "100m"
