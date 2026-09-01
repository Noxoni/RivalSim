from __future__ import annotations

import math

import torch

from rivalsim.rival2_exploration import fresh_human_seed_exploration
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    hybrid_distribution_parameters,
    hybrid_log_probability,
    sample_hybrid_action,
)
from rivalsim.rival2_ppo import (
    Rival2KLGuardConfig,
    Rival2PPOConfig,
    Rival2RolloutBuffer,
    evaluate_clipped_policy_objective,
    ppo_update,
)


def test_unified_schedule_exact_boundaries_and_smoothstep() -> None:
    for update in (0, 1, 60):
        value = fresh_human_seed_exploration(update)
        assert value.normalized_progress == 0.0
        assert value.analog_sigma == 0.01
        assert value.button_temperature == 0.02
    end = fresh_human_seed_exploration(300)
    assert end.normalized_progress == 1.0
    assert end.analog_sigma == 0.08
    assert end.button_temperature == 0.50
    assert fresh_human_seed_exploration(600) == end.__class__(
        accepted_update=600,
        normalized_progress=end.normalized_progress,
        analog_sigma=end.analog_sigma,
        analog_log_sigma=end.analog_log_sigma,
        button_temperature=end.button_temperature,
    )
    midpoint = fresh_human_seed_exploration(180)
    assert midpoint.normalized_progress == 0.5
    assert math.isclose(midpoint.analog_sigma, math.sqrt(0.01 * 0.08))
    assert midpoint.button_temperature == 0.26


def test_effective_distribution_bypasses_raw_log_std_and_preserves_button_boundary() -> None:
    actor = torch.tensor(
        [[0.2, -0.3, 0.4, -0.5, 0.6, -1.0, 0.5, -4.0, 1.0, -2.0, 2.0, -3.0, 0.0]],
        requires_grad=True,
    )
    scheduled = fresh_human_seed_exploration(1)
    mean, log_std, logits = hybrid_distribution_parameters(
        actor, distribution_override=scheduled.distribution_override
    )
    assert torch.equal(mean, actor[:, :5])
    assert torch.allclose(log_std, torch.full_like(log_std, math.log(0.01)))
    assert torch.equal(torch.sign(logits), torch.sign(actor[:, 10:13]))
    action = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, 1.0, 0.0, 1.0]])
    hybrid_log_probability(
        actor,
        action,
        distribution_override=scheduled.distribution_override,
    ).sum().backward()
    assert torch.count_nonzero(actor.grad[:, 5:10]) == 0


def test_rollout_sampling_and_ppo_recompute_use_identical_scheduled_distribution() -> None:
    torch.manual_seed(123)
    config = Rival2PolicyConfig()
    model = Rival2ActorCritic(config)
    observation = torch.randn(2, 2, config.obs_dim)
    actor, value = model(observation.reshape(-1, config.obs_dim))
    actor = actor.reshape(2, 2, 13)
    value = value.reshape(2, 2)
    scheduled = fresh_human_seed_exploration(61)
    sample = sample_hybrid_action(
        actor,
        generator=torch.Generator().manual_seed(456),
        config=config,
        distribution_override=scheduled.distribution_override,
    )
    recomputed = hybrid_log_probability(
        actor,
        sample.action,
        config=config,
        pre_tanh=sample.pre_tanh,
        distribution_override=scheduled.distribution_override,
    )
    assert torch.equal(sample.log_probability, recomputed)

    rollout = Rival2RolloutBuffer(1, 2, "cpu")
    rollout.add(
        observation=observation,
        action=sample.action.detach(),
        pre_tanh=sample.pre_tanh.detach(),
        old_log_probability=sample.log_probability.detach(),
        value=value.detach(),
        reward=torch.zeros(2, 2),
        terminated=torch.zeros(2, 2, dtype=torch.bool),
        truncated=torch.zeros(2, 2, dtype=torch.bool),
        next_value=value.detach(),
        policy_version=torch.zeros(2, 2, dtype=torch.int64),
        opponent_version=torch.zeros(2, 2, dtype=torch.int64),
        train_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    metrics = evaluate_clipped_policy_objective(
        model,
        rollout,
        Rival2PPOConfig(rollout_horizon=1, minibatch_size=4),
        distribution_override=scheduled.distribution_override,
    )
    assert abs(float(metrics["approx_kl"].item())) < 1.0e-8
    assert abs(float(metrics["mean_log_ratio"].item())) < 1.0e-8
    update_metrics = ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=0.0),
        rollout,
        Rival2PPOConfig(rollout_horizon=1, minibatch_size=4),
        generator=torch.Generator().manual_seed(789),
        gae_ready=True,
        kl_guard=Rival2KLGuardConfig(),
        distribution_override=scheduled.distribution_override,
    )
    assert abs(float(update_metrics["completed_update_mean_kl"].item())) < 1.0e-8


def test_kl_telemetry_only_mode_accepts_threshold_excess() -> None:
    torch.manual_seed(321)
    config = Rival2PolicyConfig()
    model = Rival2ActorCritic(config)
    observation = torch.randn(4, 2, config.obs_dim)
    with torch.no_grad():
        actor, value = model(observation.reshape(-1, config.obs_dim))
        actor = actor.reshape(4, 2, 13)
        value = value.reshape(4, 2)
        scheduled = fresh_human_seed_exploration(1)
        sample = sample_hybrid_action(
            actor,
            generator=torch.Generator().manual_seed(654),
            config=config,
            distribution_override=scheduled.distribution_override,
        )
    rollout = Rival2RolloutBuffer(1, 4, "cpu")
    rollout.add(
        observation=observation,
        action=sample.action,
        pre_tanh=sample.pre_tanh,
        old_log_probability=sample.log_probability,
        value=value,
        reward=torch.tensor(
            [[1.0, -1.0], [0.5, -0.5], [0.75, -0.75], [0.25, -0.25]]
        ),
        terminated=torch.zeros(4, 2, dtype=torch.bool),
        truncated=torch.zeros(4, 2, dtype=torch.bool),
        next_value=value,
        policy_version=torch.zeros(4, 2, dtype=torch.int64),
        opponent_version=torch.zeros(4, 2, dtype=torch.int64),
        train_mask=torch.ones(4, 2, dtype=torch.bool),
    )
    guard = Rival2KLGuardConfig(
        minibatch_kl_limit=1.0e-20,
        completed_update_mean_kl_limit=1.0e-20,
        reject_minibatch_kl=False,
        reject_completed_update_kl=False,
    )
    metrics = ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1.0e-4),
        rollout,
        Rival2PPOConfig(rollout_horizon=1, minibatch_size=8, epochs=1),
        generator=torch.Generator().manual_seed(987),
        gae_ready=True,
        kl_guard=guard,
        distribution_override=scheduled.distribution_override,
    )
    assert guard.kl_telemetry_only
    assert float(metrics["optimizer_post_step_approx_kl_max"].item()) > 1.0e-20
    assert float(metrics["completed_update_mean_kl"].item()) > 1.0e-20
