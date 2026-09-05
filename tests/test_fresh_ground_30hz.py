from dataclasses import replace
from pathlib import Path
import math

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim.fresh_ground_30hz import (
    GAMMA, PHYSICS_GAMMA, GAE_LAMBDA, WEIGHTS, FreshGroundEnv,
    authority, decision_reward, potentials, scenarios, scenario_hash, fresh_model,
)
from rivalsim.fresh_ground_30hz_training import FreshGroundTrainer
from rivalsim.rival2_contracts import OBS_FIELD_NAMES, POSITION_SCALE
from rivalsim.rival2_policy import sample_hybrid_action, hybrid_log_probability
from rivalsim.rival2_ppo import compute_gae_gpu


def observation():
    x = torch.zeros(4, 2, 182)
    for name, value in [("ball.position.z", .05), ("self.position.y", -.3),
                        ("opponent.position.y", .3), ("self.boost", .5)]:
        x[..., OBS_FIELD_NAMES.index(name)] = value
    return x


def test_physical_time_discount_and_trace():
    assert PHYSICS_GAMMA ** 4 == pytest.approx(GAMMA, abs=1e-14)
    assert (GAMMA * GAE_LAMBDA) ** 90 == pytest.approx(.5)
    assert authority()["maximum_updates"] is None and authority()["deadline"] is None
    assert authority()["direct_behavior_rewards"] == []


@pytest.mark.parametrize("goal_tick", [-1, 0, 1, 2, 3])
def test_four_tick_pbrs_telescopes_including_early_goal(goal_tick):
    sequence = [observation() + torch.rand(4, 2, 182)*.01 for _ in range(5)]
    phi = [sum(v * WEIGHTS[k] for k, v in potentials(obs).items()) for obs in sequence]
    expected = torch.zeros(4, 2)
    for tick in range(4):
        if goal_tick >= 0 and tick > goal_tick:
            continue
        successor = torch.zeros_like(phi[0]) if tick == goal_tick else phi[tick+1]
        reward = PHYSICS_GAMMA * successor - phi[tick]
        if tick == goal_tick:
            reward += torch.tensor([10., -10.])
        expected += PHYSICS_GAMMA**tick * reward
    actual, components = decision_reward(sequence[0], sequence[4],
                                         torch.full((4,), goal_tick), torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-6)
    if goal_tick >= 0:
        other, _ = decision_reward(sequence[0], torch.rand_like(sequence[4])*100,
                                    torch.full((4,), goal_tick), torch.zeros(4, dtype=torch.long))
        torch.testing.assert_close(actual, other, rtol=0, atol=0)


def test_access_stays_sensitive_far_behind_and_is_bounded():
    x = observation()
    x[..., OBS_FIELD_NAMES.index("self.position.y")] = -6000/POSITION_SCALE[1]
    x[..., OBS_FIELD_NAMES.index("opponent.position.y")] = 800/POSITION_SCALE[1]
    old = potentials(x)["access"]
    x[..., OBS_FIELD_NAMES.index("self.position.y")] = -5500/POSITION_SCALE[1]
    assert bool((potentials(x)["access"] > old).all())
    assert bool((old.abs() <= 1).all())


def test_velocity_potential_distinguishes_direction_without_touch_reward():
    x = observation()
    i = OBS_FIELD_NAMES.index("ball.linear_velocity.y")
    x[..., i] = .3
    forward = potentials(x)["goal_velocity"]
    x[..., i] = -.3
    backward = potentials(x)["goal_velocity"]
    assert bool((forward > 0).all()) and bool((backward < 0).all())
    x[..., OBS_FIELD_NAMES.index("lifecycle.self_touch_event")] = 1
    torch.testing.assert_close(backward, potentials(x)["goal_velocity"])


def test_scenario_reproducibility_validity_and_coherent_momentum():
    a, b = scenarios(1000), scenarios(1000)
    assert scenario_hash(a) == scenario_hash(b) != scenario_hash(scenarios(1000, 123))
    assert {int(f): int((a.family == f).sum()) for f in np.unique(a.family)} == {0:100, 1:150, 2:500, 4:250}
    assert set(a.kickoff_layout[a.kickoff_indicator != 0]) == set(range(5))
    assert np.all(np.linalg.norm(a.state.car_pos[:, 0]-a.state.car_pos[:, 1], axis=-1) >= 180)
    assert np.all(np.linalg.norm(a.state.ball_pos[:, None]-a.state.car_pos, axis=-1) > 150)
    assert np.allclose(np.linalg.norm(a.state.car_quat, axis=-1), 1, atol=1e-6)
    assert np.all(a.state.on_ground == 1)
    assert np.all(a.state.car_pos[..., 2] == 17)
    q = a.state.car_quat
    yaw = np.arctan2(2*q[..., 2]*q[..., 3], 1-2*q[..., 2]**2)
    speed = np.linalg.norm(a.state.car_vel[..., :2], axis=-1)
    forward_speed = (a.state.car_vel[..., :2] * np.stack((np.cos(yaw), np.sin(yaw)), -1)).sum(-1)
    assert np.all(forward_speed[speed > 1] / speed[speed > 1] > .97)


def test_fresh_initialization_reproducible_and_critic_is_independent():
    torch.set_num_threads(4)
    a, b = fresh_model(), fresh_model()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert not torch.equal(a.trunk[0].weight, a.critic.features[0].weight)
    obs = torch.randn(12, 182)
    a.isolated_value(obs).square().mean().backward()
    assert all(p.grad is None for n, p in a.named_parameters() if not n.startswith("critic."))
    assert all(p.grad is not None for p in a.critic.parameters())
    actor, _ = a.forward_actor(obs)
    sample = sample_hybrid_action(actor, generator=torch.Generator().manual_seed(1), config=a.config)
    recomputed = hybrid_log_probability(actor, sample.action, pre_tanh=sample.pre_tanh, config=a.config)
    torch.testing.assert_close(recomputed, sample.log_probability, rtol=0, atol=0)
    assert bool((sample.action[..., :5].abs() <= 1).all())
    assert set(sample.action[..., 5:].detach().flatten().tolist()) <= {0., 1.}


@pytest.fixture(scope="module")
def env():
    root = Path("G:/dev/RLBot-Rival/bot/collision_meshes")
    if not torch.cuda.is_available() or not root.exists():
        pytest.skip("requires native CUDA physics and local arena")
    return FreshGroundEnv(32, root, device="cuda:0", seed=42, ssl_foundation_scenarios=scenarios(32))


def force_goal(env, row, sign=1):
    p = torch.tensor([0., sign*5300., 93.15], device=env.device)
    env.bridge.views["ball_pos"].reshape(-1, 3)[row] = p
    wp.to_torch(env.world.ball_world.position_bt).reshape(-1, 3)[row] = p*.02


def test_native_goals_at_all_four_ticks_reset_exactly_once_and_repeat(env):
    zeros = torch.zeros(32, 2, 8, device=env.device)
    generation = wp.to_torch(env.world.ssl_foundation_reset.reset_generation)
    for repeat in range(2):
        prior = generation.clone()
        before = env.observation.clone()
        def provider(tick):
            force_goal(env, tick, 1 if tick % 2 == 0 else -1)
            return zeros
        tr = env.step_with_tick_actions(zeros, provider)
        assert tr.terminated[:4].all() and tr.reset_mask[:4].all()
        assert not tr.truncated[:4].any()
        assert torch.equal(env.last_native["first_goal_tick"][:4], torch.arange(4, device=env.device))
        assert torch.equal(generation[:4], prior[:4]+1)
        for tick in range(4):
            sign = 1 if tick % 2 == 0 else -1
            expected = torch.tensor([10., -10.], device=env.device) * sign * PHYSICS_GAMMA**tick
            torch.testing.assert_close(env.last_components["terminal_goal"][tick], expected)
        expected, _ = decision_reward(before, tr.transition_observation,
                                      env.last_native["first_goal_tick"], env.last_native["scoring_team"])
        torch.testing.assert_close(tr.reward, expected, rtol=0, atol=0)
        assert bool((env.bridge.views["rival2.episode_ticks"][:4] == 0).all())
        assert bool((wp.to_torch(env.world.lifecycle.reset_required)[:4] == 0).all())


def test_native_truncation_bootstrap_and_no_false_goal(env):
    env.bridge.views["rival2.episode_ticks"][5] = 3596
    env.bridge.views["rival2.no_touch_ticks"][6] = 1796
    before = env.observation.clone()
    tr = env.step(torch.zeros(32, 2, 8, device=env.device))
    assert tr.truncated[5:7].all() and tr.reset_mask[5:7].all()
    assert not tr.terminated[5:7].any()
    assert env.last_components["terminal_goal"][5:7].count_nonzero() == 0
    assert not torch.equal(tr.transition_observation[5], tr.observation[5])
    r = torch.tensor([[[1., 1.]]])
    terminal = torch.tensor([[[True, False]]])
    trunc = ~terminal
    a, returns = compute_gae_gpu(r, torch.zeros_like(r), torch.full_like(r, 2), terminal, trunc,
                                gamma=GAMMA, gae_lambda=GAE_LAMBDA)
    torch.testing.assert_close(returns, torch.tensor([[[1., 1.+GAMMA*2]]]))


def test_complete_small_rollout_logprob_and_safe_update_resume(env, tmp_path):
    trainer = FreshGroundTrainer(env)
    trainer.ppo_config = replace(trainer.ppo_config, rollout_horizon=5, minibatch_size=160)
    buffer = trainer.collect_rollout()
    assert buffer.train_mask.all() and buffer.opponent_family.count_nonzero() == 0
    assert trainer.nexto is None
    assert trainer.total_agent_samples == 5*32*2
    assert trainer.physical_physics_ticks_experienced == 5*32*4
    buffer.compute_gae(trainer.ppo_config)
    metrics = trainer.update(buffer)
    assert trainer.accepted_updates_total == 1
    assert all(torch.isfinite(v) for v in metrics.values())
    path = tmp_path / "own.pt"
    trainer.save_checkpoint(path)
    restored = FreshGroundTrainer(env)
    restored.ppo_config = trainer.ppo_config
    restored.load_checkpoint(path)
    assert restored.accepted_updates_total == 1 and restored.total_agent_samples == 320
    assert all(torch.equal(v, restored.model.state_dict()[k]) for k, v in trainer.model.state_dict().items())
    bad = trainer.checkpoint_payload()
    bad["source"]["parent"] = "V5"
    torch.save(bad, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="fresh random 30Hz lineage"):
        restored.load_checkpoint(tmp_path / "wrong.pt")
