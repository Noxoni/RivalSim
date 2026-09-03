from __future__ import annotations

from types import SimpleNamespace

import torch

from benchmarks import run_rival2_unified_ground_curriculum_ppo_v2 as campaign
from rivalsim.rival2_contracts import (
    RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
from rivalsim.rival2_unified_policy import Rival2UnifiedActorCritic


def test_authority_freezes_acquisition_then_gameplay_selfplay() -> None:
    authority = campaign.load_authority()
    assert authority["source"]["sha256"] == campaign.SOURCE_SHA256
    assert authority["integrity"]["abandoned_v1_update_loaded"] is False
    assert authority["integrity"]["value_loss_to_shared_trunk"] is False
    assert authority["opponents"]["current_selfplay"] == 1.0
    assert authority["opponents"]["nexto"] == 0.0
    assert authority["reward_design"]["mechanics_reward"] == 0.0
    assert authority["campaign"]["accepted_updates_total"] == 300


def test_phase_specs_are_frozen_and_use_slow_policy_updates() -> None:
    assert campaign.phase_spec("unified_ground_acquisition_v2") == (
        RIVAL2_REWARD_ACQUISITION_120_V1_VERSION,
        5.0e-7,
    )
    assert campaign.phase_spec("unified_ground_gameplay_v2") == (
        RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        1.0e-6,
    )


def test_optimizer_separates_policy_and_critic_learning_rates() -> None:
    trainer = SimpleNamespace(model=Rival2UnifiedActorCritic())
    campaign.configure_optimizer(trainer, campaign.PHASE_A_POLICY_LR)
    groups = campaign.optimizer_lrs(trainer)
    assert groups == {
        "policy": campaign.PHASE_A_POLICY_LR,
        "critic": campaign.CRITIC_LR,
    }
    critic_ids = {id(parameter) for parameter in trainer.model.critic.parameters()}
    policy_ids = {
        id(parameter) for parameter in trainer.optimizer.param_groups[0]["params"]
    }
    optimizer_critic_ids = {
        id(parameter) for parameter in trainer.optimizer.param_groups[1]["params"]
    }
    assert policy_ids.isdisjoint(critic_ids)
    assert optimizer_critic_ids == critic_ids


def test_fixed_horizon_touch_fraction_includes_live_successful_trials() -> None:
    completed, touched = campaign.include_live_fixed_horizon_episodes(
        4,
        3,
        torch.tensor([[True, True], [True, False]]),
    )
    assert completed == 8
    assert touched == 6
