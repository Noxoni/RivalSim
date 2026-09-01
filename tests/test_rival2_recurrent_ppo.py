from __future__ import annotations

import torch

from rivalsim.rival2_exploration import fresh_human_seed_exploration
from rivalsim.rival2_policy import hybrid_log_probability, sample_hybrid_action
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_recurrent_policy import (
    Rival2RecurrentActorCritic,
    Rival2RecurrentPolicyConfig,
)
from rivalsim.rival2_recurrent_ppo import (
    Rival2RecurrentRolloutBuffer,
    recurrent_ppo_update,
)


def small_policy() -> Rival2RecurrentActorCritic:
    torch.manual_seed(7)
    return Rival2RecurrentActorCritic(
        Rival2RecurrentPolicyConfig(
            encoder_dim=16,
            hidden_dim=16,
            post_dim=16,
        )
    )


def test_recurrent_rollout_stores_complete_sequence_state() -> None:
    model = small_policy()
    hidden = model.initial_hidden(4).reshape(1, 2, 2, 16)
    rollout = Rival2RecurrentRolloutBuffer(4, 2, hidden, "cpu")
    layout = rollout.sequence_layout(8)
    assert layout.sequence_count == 4
    assert layout.sequences_per_minibatch == 2
    assert torch.equal(rollout.initial_hidden, hidden)
    assert rollout.logical_bytes > rollout.observations.numel() * 4


def test_rollout_sampling_and_ppo_recomputation_share_distribution() -> None:
    model = small_policy()
    observation = torch.randn(6, 182)
    actor, _value, _hidden = model(observation)
    exploration = fresh_human_seed_exploration(1)
    generator = torch.Generator().manual_seed(19)
    sample = sample_hybrid_action(
        actor,
        generator=generator,
        config=model.config,
        distribution_override=exploration.distribution_override,
    )
    recomputed = hybrid_log_probability(
        actor,
        sample.action,
        config=model.config,
        pre_tanh=sample.pre_tanh,
        distribution_override=exploration.distribution_override,
    )
    torch.testing.assert_close(recomputed, sample.log_probability, rtol=0.0, atol=0.0)


def test_recurrent_ppo_updates_sequences_without_kl_rejection() -> None:
    model = small_policy()
    config = Rival2PPOConfig(
        learning_rate=1.0e-4,
        epochs=1,
        rollout_horizon=4,
        minibatch_size=8,
        entropy_coefficient=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    exploration = fresh_human_seed_exploration(1)
    action_generator = torch.Generator().manual_seed(23)
    shuffle_generator = torch.Generator().manual_seed(29)
    hidden = model.initial_hidden(4)
    rollout = Rival2RecurrentRolloutBuffer(
        config.rollout_horizon,
        2,
        hidden.reshape(1, 2, 2, 16),
        "cpu",
    )
    for tick in range(config.rollout_horizon):
        observation = torch.randn(2, 2, 182)
        reset = torch.zeros(2, 2, dtype=torch.bool)
        if tick == 2:
            reset[0].fill_(True)
        with torch.no_grad():
            actor, value, next_hidden = model(
                observation.reshape(4, 182),
                hidden,
                reset_before=reset.reshape(4),
            )
            sample = sample_hybrid_action(
                actor,
                generator=action_generator,
                config=model.config,
                distribution_override=exploration.distribution_override,
            )
        terminated = torch.zeros(2, 2, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        if tick == config.rollout_horizon - 1:
            terminated.fill_(True)
        rollout.add(
            observation=observation,
            action=sample.action.reshape(2, 2, 8),
            pre_tanh=sample.pre_tanh.reshape(2, 2, 5),
            old_log_probability=sample.log_probability.reshape(2, 2),
            value=value.reshape(2, 2),
            reward=torch.randn(2, 2) * 0.01,
            terminated=terminated,
            truncated=truncated,
            next_value=torch.zeros(2, 2),
            train_mask=torch.ones(2, 2, dtype=torch.bool),
            reset_before=reset,
        )
        hidden = next_hidden
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = recurrent_ppo_update(
        model,
        optimizer,
        rollout,
        config,
        generator=shuffle_generator,
        distribution_override=exploration.distribution_override,
    )
    assert int(metrics["optimizer_steps"].item()) == 2
    assert torch.isfinite(metrics["total_loss"])
    assert torch.isfinite(metrics["completed_update_mean_kl"])
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
