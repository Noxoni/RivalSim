from __future__ import annotations

import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim import StateSnapshot
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    ACTION_CONTRACT_HASH,
    CONTRACT_HASHES,
    EPISODE_CONTRACT_HASH,
    OBS_DIM,
    OBSERVATION_SCHEMA_HASH,
    ORANGE_PAD_REMAP,
    REWARD_CONTRACT_HASH,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    hybrid_log_probability,
    sample_hybrid_action,
)
from rivalsim.rival2_ppo import Rival2PPOConfig, compute_gae_gpu
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs, Warp CUDA, and PyTorch CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return root, geometry, WarpArenaMeshes(geometry)


def _env(
    assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    count: int,
    **kwargs: object,
) -> Rival2Env:
    root, geometry, meshes = assets
    return Rival2Env(
        count,
        root,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
        **kwargs,
    )


def test_contracts_are_frozen_and_content_addressed() -> None:
    assert OBS_DIM == 182
    assert len(ORANGE_PAD_REMAP) == 34
    assert all(ORANGE_PAD_REMAP[ORANGE_PAD_REMAP[index]] == index for index in range(34))
    assert CONTRACT_HASHES == {
        "RIVAL2_OBS_V1": OBSERVATION_SCHEMA_HASH,
        "RIVAL2_ACTION_V1": ACTION_CONTRACT_HASH,
        "RIVAL2_REWARD_V1": REWARD_CONTRACT_HASH,
        "RIVAL2_EPISODE_V1": EPISODE_CONTRACT_HASH,
    }
    assert all(len(value) == 64 for value in CONTRACT_HASHES.values())


def test_hybrid_log_probability_matches_independent_numpy_reference() -> None:
    rng = np.random.default_rng(20260825)
    output_np = rng.normal(0.0, 0.5, (7, 13)).astype(np.float32)
    pre_tanh_np = rng.normal(0.0, 0.7, (7, 5)).astype(np.float32)
    buttons_np = rng.integers(0, 2, (7, 3)).astype(np.float32)
    action_np = np.concatenate((np.tanh(pre_tanh_np), buttons_np), axis=-1)
    output = torch.tensor(output_np, device="cuda:0")
    pre_tanh = torch.tensor(pre_tanh_np, device="cuda:0")
    action = torch.tensor(action_np, device="cuda:0")
    actual = hybrid_log_probability(output, action, pre_tanh=pre_tanh).cpu().numpy()

    mean = output_np[:, :5].astype(np.float64)
    log_std = np.clip(output_np[:, 5:10], -5.0, 1.0).astype(np.float64)
    u = pre_tanh_np.astype(np.float64)
    gaussian = -0.5 * (
        ((u - mean) / np.exp(log_std)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
    )
    analog = np.sum(gaussian - np.log(1.0 - np.tanh(u) ** 2), axis=-1)
    logits = output_np[:, 10:13].astype(np.float64)
    buttons = buttons_np.astype(np.float64)
    bernoulli = np.sum(
        buttons * -np.logaddexp(0.0, -logits) + (1.0 - buttons) * -np.logaddexp(0.0, logits),
        axis=-1,
    )
    np.testing.assert_allclose(actual, analog + bernoulli, rtol=2e-6, atol=2e-6)


def test_sampling_and_deterministic_controller_contract() -> None:
    output = torch.linspace(-2.0, 2.0, 52, device="cuda:0").reshape(4, 13)
    generator_a = torch.Generator(device="cuda:0").manual_seed(42)
    generator_b = torch.Generator(device="cuda:0").manual_seed(42)
    sample_a = sample_hybrid_action(output, generator=generator_a)
    sample_b = sample_hybrid_action(output, generator=generator_b)
    torch.testing.assert_close(sample_a.action, sample_b.action, rtol=0, atol=0)
    torch.testing.assert_close(sample_a.log_probability, sample_b.log_probability, rtol=0, atol=0)
    assert torch.isfinite(sample_a.log_probability).all()
    assert torch.all((sample_a.action[:, :5] >= -1) & (sample_a.action[:, :5] <= 1))
    assert torch.all((sample_a.action[:, 5:] == 0) | (sample_a.action[:, 5:] == 1))
    deterministic = deterministic_hybrid_action(output)
    torch.testing.assert_close(deterministic[:, :5], torch.tanh(output[:, :5]))
    torch.testing.assert_close(deterministic[:, 5:], (output[:, 10:13] >= 0).to(torch.float32))


def test_gae_matches_independent_reference_with_terminal_and_truncation() -> None:
    rng = np.random.default_rng(77)
    shape = (6, 3, 2)
    rewards = rng.normal(size=shape).astype(np.float32)
    values = rng.normal(size=shape).astype(np.float32)
    next_values = rng.normal(size=shape).astype(np.float32)
    terminated = np.zeros(shape, dtype=bool)
    truncated = np.zeros(shape, dtype=bool)
    terminated[2, 0] = True
    truncated[4, 1] = True
    terminated[5, 2] = True
    gamma = 0.995
    gae_lambda = 0.95
    actual_advantage, actual_return = compute_gae_gpu(
        torch.tensor(rewards, device="cuda:0"),
        torch.tensor(values, device="cuda:0"),
        torch.tensor(next_values, device="cuda:0"),
        torch.tensor(terminated, device="cuda:0"),
        torch.tensor(truncated, device="cuda:0"),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    expected = np.empty_like(rewards)
    carry = np.zeros(shape[1:], dtype=np.float32)
    for time in range(shape[0] - 1, -1, -1):
        delta = rewards[time] + gamma * next_values[time] * (~terminated[time]) - values[time]
        carry = delta + gamma * gae_lambda * (~(terminated[time] | truncated[time])) * carry
        expected[time] = carry
    np.testing.assert_allclose(actual_advantage.cpu(), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_return.cpu(), expected + values, rtol=2e-6, atol=2e-6)


def test_zero_copy_observation_symmetry_and_mechanics4_cadence(arena_assets) -> None:
    env = _env(arena_assets, 2)
    report = env.bridge.alias_report()
    assert len(report) >= 40
    assert all(item["aliases"] for item in report.values())
    assert all(item["device"] == "cuda:0" for item in report.values())
    assert env.observation.shape == (2, 2, OBS_DIM)
    assert env.observation.dtype == torch.float32
    assert env.observation.is_cuda and env.observation.is_contiguous()
    torch.testing.assert_close(env.observation[:, 0], env.observation[:, 1], rtol=0, atol=1e-6)

    action = torch.zeros((2, 2, 8), device=env.device)
    action[..., 0] = 0.75
    action[..., 5] = 1.0
    env.reset_transfer_counters()
    result = env.step(action)
    torch.cuda.synchronize()
    assert env.world.tick_count == 4
    np.testing.assert_array_equal(env.world.rival2.interval_tick.numpy(), (4, 4))
    assert env.hot_path_transfer_bytes() == {"h2d": 0, "d2h": 0}
    torch.testing.assert_close(result.emitted_action, action)
    assert torch.isfinite(result.observation).all()


def test_goal_and_truncation_account_before_selective_kickoff(arena_assets) -> None:
    state = StateSnapshot.empty(1)
    state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    state.ball_pos[:] = np.asarray((0.0, 5300.0, 93.15), dtype=np.float32)
    goal_env = _env(arena_assets, 1, initial=state)
    goal_action = torch.full((1, 2, 8), 0.75, device=goal_env.device)
    goal_action[..., 5:] = 1.0
    result = goal_env.step(goal_action)
    torch.cuda.synchronize()
    torch.testing.assert_close(result.reward, torch.tensor([[10.0, -10.0]], device="cuda:0"))
    assert result.terminated.item() and not result.truncated.item()
    assert result.reset_mask.item()
    np.testing.assert_array_equal(
        goal_env.world.state.ball_pos.numpy(),
        np.asarray(((0.0, 0.0, np.float32(93.15)),), dtype=np.float32),
    )
    assert torch.all(result.observation[..., 175] == 1)
    assert torch.all(result.observation[..., 167:175] == 0)
    assert torch.all(result.observation[..., 176:180] == 0)
    assert torch.all(torch.abs(result.transition_observation[..., 1]) > 1.0)

    timeout_env = _env(arena_assets, 1)
    timeout_env.bridge.views["rival2.episode_ticks"].fill_(45 * 120 - 4)
    timeout = timeout_env.step(torch.zeros((1, 2, 8), device=timeout_env.device))
    torch.cuda.synchronize()
    assert timeout.truncated.item() and not timeout.terminated.item()
    assert timeout.reset_mask.item()
    assert timeout_env.bridge.views["rival2.episode_ticks"].item() == 0


def test_ppo_updates_actor_critic_and_checkpoint_restores_state(arena_assets) -> None:
    config = Rival2PPOConfig(rollout_horizon=2, minibatch_size=8, epochs=1)
    env = _env(arena_assets, 2)
    trainer = Rival2Trainer(env, ppo_config=config, seed=123)
    actor_before = trainer.model.actor.weight.detach().clone()
    critic_before = trainer.model.critic.weight.detach().clone()
    rollout, metrics = trainer.train_iteration()
    assert rollout.position == config.rollout_horizon
    assert rollout.observations.is_cuda and rollout.advantages.is_cuda
    assert all(torch.isfinite(value).item() for value in metrics.values())
    assert not torch.equal(actor_before, trainer.model.actor.weight)
    assert not torch.equal(critic_before, trainer.model.critic.weight)

    trainer.add_historical_snapshot()
    trainer.self_play_config = Rival2SelfPlayConfig(historical_chance=1.0)
    trainer.assign_opponents_at_reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    assert torch.all(trainer.opponent_assignment == trainer.policy_version)
    checkpoint = Path(".tools/v0.5/test_checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint)

    restored = Rival2Trainer(_env(arena_assets, 2), ppo_config=config, seed=999)
    restored.load_checkpoint(checkpoint)
    assert restored.policy_version == trainer.policy_version
    assert restored.iteration == trainer.iteration
    assert restored.total_agent_samples == trainer.total_agent_samples
    assert asdict(restored.ppo_config) == asdict(trainer.ppo_config)
    assert restored.opponent_pool.versions == trainer.opponent_pool.versions
    for expected, actual in zip(
        trainer.model.parameters(), restored.model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    expected_action, expected_value = trainer.deterministic_action_value(
        trainer.env.observation.reshape(-1, OBS_DIM)
    )
    actual_action, actual_value = restored.deterministic_action_value(
        trainer.env.observation.reshape(-1, OBS_DIM)
    )
    torch.testing.assert_close(expected_action, actual_action, rtol=0, atol=0)
    torch.testing.assert_close(expected_value, actual_value, rtol=0, atol=0)
    checkpoint.unlink()


def test_policy_architecture_is_frozen() -> None:
    config = Rival2PolicyConfig()
    model = Rival2ActorCritic(config)
    assert config.obs_dim == 182
    assert config.hidden_layers == 3 and config.hidden_dim == 512
    assert config.activation == "silu"
    assert model.actor.out_features == 13
    assert model.critic.out_features == 1
    assert model.parameter_count == 626190
