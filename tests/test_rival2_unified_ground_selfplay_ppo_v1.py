from __future__ import annotations

import json
from itertools import pairwise

import torch

from benchmarks import run_rival2_unified_ground_selfplay_ppo_v1 as campaign
from rivalsim.rival2_policy import sample_hybrid_action
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_recurrent_ppo import (
    Rival2RecurrentRolloutBuffer,
    recurrent_ppo_update,
)
from rivalsim.rival2_unified_policy import (
    Rival2UnifiedActorCritic,
    Rival2UnifiedPolicyConfig,
)


def test_authority_freezes_source_reward_selfplay_and_exploration() -> None:
    authority = json.loads(campaign.AUTHORITY.read_text(encoding="utf-8"))
    assert authority["source"]["sha256"] == campaign.SOURCE_SHA256
    assert authority["opponents"] == {
        "both_current_sides_trainable": True,
        "current_selfplay": 1.0,
        "historical": 0.0,
        "nexto": 0.0,
        "wisp": 0.0,
    }
    assert authority["campaign"]["accepted_updates"] == 300
    assert authority["exploration"]["contract_sha256"] == (
        campaign.EXPLORATION_CONTRACT_HASH
    )
    assert campaign.load_authority() == authority


def test_exploration_schedule_is_bounded_and_monotonic() -> None:
    checkpoints = [1, 30, 60, 90, 120, 150, 300]
    values = [campaign.exploration_for_update(update) for update in checkpoints]
    assert values[0].analog_sigma == values[1].analog_sigma == 0.02
    assert values[0].button_temperature == values[1].button_temperature == 0.10
    assert values[-2].analog_sigma == values[-1].analog_sigma == 0.04
    assert values[-2].button_temperature == values[-1].button_temperature == 0.25
    assert all(
        left.analog_sigma <= right.analog_sigma
        for left, right in pairwise(values)
    )
    assert all(
        left.button_temperature <= right.button_temperature
        for left, right in pairwise(values)
    )


def test_unified_policy_uses_sequence_ppo_without_kl_rejection() -> None:
    torch.manual_seed(73)
    config = Rival2UnifiedPolicyConfig()
    model = Rival2UnifiedActorCritic(config)
    ppo = Rival2PPOConfig(
        learning_rate=3.0e-5,
        epochs=1,
        rollout_horizon=2,
        minibatch_size=2,
        entropy_coefficient=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo.learning_rate)
    exploration = campaign.exploration_for_update(1)
    action_generator = torch.Generator().manual_seed(79)
    hidden = model.initial_hidden(2).reshape(1, 1, 2, config.hidden_dim)
    rollout = Rival2RecurrentRolloutBuffer(2, 1, hidden, "cpu")
    flat_hidden = hidden.reshape(1, 2, config.hidden_dim)
    for tick in range(2):
        observation = torch.randn(1, 2, config.obs_dim)
        reset = torch.full((1, 2), tick == 0, dtype=torch.bool)
        with torch.no_grad():
            actor, value, flat_hidden = model(
                observation.reshape(2, config.obs_dim),
                flat_hidden,
                reset_before=reset.reshape(2),
            )
            sample = sample_hybrid_action(
                actor,
                generator=action_generator,
                config=config,
                distribution_override=exploration.distribution_override,
            )
        rollout.add(
            observation=observation,
            action=sample.action.reshape(1, 2, 8),
            pre_tanh=sample.pre_tanh.reshape(1, 2, 5),
            old_log_probability=sample.log_probability.reshape(1, 2),
            value=value.reshape(1, 2),
            reward=torch.randn(1, 2) * 0.01,
            terminated=torch.zeros(1, 2, dtype=torch.bool),
            truncated=torch.zeros(1, 2, dtype=torch.bool),
            next_value=torch.zeros(1, 2),
            train_mask=torch.ones(1, 2, dtype=torch.bool),
            reset_before=reset,
        )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = recurrent_ppo_update(
        model,
        optimizer,
        rollout,
        ppo,
        generator=torch.Generator().manual_seed(83),
        distribution_override=exploration.distribution_override,
    )
    assert int(metrics["optimizer_steps"].item()) == 2
    assert torch.isfinite(metrics["completed_update_mean_kl"])
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
