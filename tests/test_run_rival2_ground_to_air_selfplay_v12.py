from __future__ import annotations

from dataclasses import asdict

from benchmarks import run_rival2_ground_to_air_selfplay_v12 as runner
from rivalsim.rival2_contracts import (
    REWARD_GAMEPLAY_120_V2_CONTRACT_HASH,
)
from rivalsim.rival2_ground_to_air_selfplay_v12 import (
    AerialOptionRouterConfig,
    AerialSelfPlayRewardConfig,
)
from rivalsim.rival2_ppo import RIVAL2_PPO_120HZ_CONTRACT_HASH


def test_v12_prospective_authority_is_frozen_and_exact() -> None:
    authority = runner.load_authority()
    assert runner.sha256_file(runner.AUTHORITY) == runner.AUTHORITY_SHA256
    assert authority["router"] == asdict(AerialOptionRouterConfig())
    assert authority["aerial_reward"] == asdict(AerialSelfPlayRewardConfig())
    assert authority["ppo"]["contract_sha256"] == RIVAL2_PPO_120HZ_CONTRACT_HASH
    assert (
        authority["sources"]["aerial_scorer"]["sha256"]
        == runner.OPTION_SHA256
    )


def test_v12_reward_is_physical_bounded_and_not_raw_airtime() -> None:
    reward = runner.reward_config()
    assert reward.raw_airtime_reward == 0.0
    assert reward.maximum_supplemental_reward_per_attempt == 40.0
    assert reward.goal_within_contact_budget_event > 10.0
    assert runner.load_authority()["integrity"]["mechanic_classifier_used"] is False


def test_v12_preserves_gameplay_contract_and_uses_pure_current_self_play() -> None:
    authority = runner.load_authority()
    assert authority["campaign"]["current_self_play_both_sides"] is True
    assert authority["campaign"]["nexto_training_probability"] == 0.0
    assert authority["integrity"]["production_gameplay_reward_changed"] is False
    assert authority["ppo"]["kl_telemetry_only"] is True
    assert REWARD_GAMEPLAY_120_V2_CONTRACT_HASH == (
        "E63920316F04ED66F02065D0DEDBEF500CDAF8F485BD2602E21AFECBA72EFF6C"
    )


def test_v12_sources_remain_byte_exact() -> None:
    assert runner.sha256_file(runner.BLUE) == runner.BLUE_SHA256
    assert runner.sha256_file(runner.ORANGE) == runner.ORANGE_SHA256
    assert runner.sha256_file(runner.OPTION) == runner.OPTION_SHA256

