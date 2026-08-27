from __future__ import annotations

import hashlib
from pathlib import Path

import torch

import benchmarks.run_rival2_gameplay_v1 as gameplay
from rivalsim.rival2_contracts import (
    EPISODE_CONTRACT_HASH,
    REWARD_CONTRACT_HASH,
    REWARD_GAMEPLAY_V1_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_ACQUISITION_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    contract_hashes_for_reward,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def test_gameplay_runner_is_pinned_to_acquisition_and_short_lifecycle() -> None:
    source = torch.load(gameplay.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    configuration = gameplay.frozen_configuration(source)

    assert _sha256(gameplay.SOURCE_CHECKPOINT) == gameplay.SOURCE_CHECKPOINT_SHA256
    assert source["iteration"] == source["policy_version"] == 120
    assert source["reward_version"] == RIVAL2_REWARD_ACQUISITION_V1_VERSION
    assert source["episode_version"] == RIVAL2_EPISODE_VERSION
    assert configuration["destination_reward_version"] == RIVAL2_REWARD_GAMEPLAY_V1_VERSION
    assert configuration["destination_episode_version"] == RIVAL2_EPISODE_VERSION
    assert configuration["destination_episode_contract_hash"] == EPISODE_CONTRACT_HASH
    assert configuration["five_minute_matches"] is False
    assert configuration["nexto_training"] is False


def test_gameplay_runner_arithmetic_contracts_and_guard_are_exact() -> None:
    source = torch.load(gameplay.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    configuration = gameplay.frozen_configuration(source)

    assert gameplay.ADDITIONAL_UPDATES == 239
    assert gameplay.SAMPLES_PER_UPDATE == 8_388_608
    assert gameplay.ADDITIONAL_SAMPLES == 2_004_877_312
    assert source["total_agent_samples"] + gameplay.ADDITIONAL_SAMPLES == 3_011_510_272
    assert gameplay.CHECKPOINT_OFFSETS == (60, 120, 180, 239)
    assert gameplay.KL_GUARD.minibatch_kl_limit == 0.10
    assert gameplay.KL_GUARD.completed_update_mean_kl_limit == 0.05
    assert REWARD_CONTRACT_HASH == gameplay.HISTORICAL_V1_HASH
    assert configuration["destination_reward_contract_hash"] == REWARD_GAMEPLAY_V1_CONTRACT_HASH
    hashes = contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_V1_VERSION, RIVAL2_EPISODE_VERSION)
    assert hashes[RIVAL2_REWARD_GAMEPLAY_V1_VERSION] == REWARD_GAMEPLAY_V1_CONTRACT_HASH


def test_gameplay_runner_preserves_frozen_ppo_configuration() -> None:
    source = torch.load(gameplay.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    ppo = gameplay.frozen_configuration(source)["ppo_config"]

    assert ppo == source["ppo_config"]
    assert ppo["entropy_coefficient"] == 0.0
    assert ppo["learning_rate"] == 0.0003
    assert ppo["clip_range"] == 0.2
    assert ppo["value_loss_coefficient"] == 0.5
    assert ppo["max_gradient_norm"] == 0.5
    assert ppo["rollout_horizon"] == 32
