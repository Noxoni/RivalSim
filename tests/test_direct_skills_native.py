from dataclasses import replace
from pathlib import Path

import pytest
import torch
import warp as wp

from rivalsim.direct_skills_training import DirectSkillCollector
from rivalsim.direct_skills_v1 import DirectSkillsEnv, scenarios
from rivalsim.fresh_ground_30hz import PHYSICS_GAMMA, decision_reward
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic
from tests.test_fresh_ground_30hz import force_goal


@pytest.fixture(scope="module")
def env():
    root = Path("G:/dev/RLBot-Rival/bot/collision_meshes")
    if not torch.cuda.is_available() or not root.exists():
        pytest.skip("requires actual CUDA simulator and arena")
    torch.set_num_threads(4)
    return DirectSkillsEnv(
        32, root, device="cuda:0", seed=42, ssl_foundation_scenarios=scenarios(32)
    )


def test_first_native_goal_each_tick_and_exact_reset_no_skill_leak(env):
    action = torch.zeros((32, 2, 8), device=env.device)
    generation = wp.to_torch(env.world.ssl_foundation_reset.reset_generation)
    for _ in range(2):
        before, prior, family = env.observation.clone(), generation.clone(), env.family.clone()

        def provider(tick):
            force_goal(env, tick, 1 if tick % 2 == 0 else -1)
            return action

        tr = env.step_with_tick_actions(action, provider)
        assert tr.terminated[:4].all() and not tr.truncated[:4].any()
        assert torch.equal(
            env.last_native["first_goal_tick"][:4], torch.arange(4, device=env.device)
        )
        assert torch.equal(generation[:4], prior[:4] + 1)
        natural, parts = decision_reward(
            before,
            tr.transition_observation,
            env.last_native["first_goal_tick"],
            env.last_native["scoring_team"],
        )
        expected = torch.where((family == 0)[:, None], natural, parts["terminal_goal"])
        torch.testing.assert_close(tr.reward[:4], expected[:4], rtol=0, atol=0)
        for tick in range(4):
            expected_goal = (
                torch.tensor([10.0, -10.0], device=env.device)
                * (1 if tick % 2 == 0 else -1)
                * PHYSICS_GAMMA**tick
            )
            torch.testing.assert_close(parts["terminal_goal"][tick], expected_goal)
        assert not bool(env.last_skill["events"][:4].any())
        assert not bool(env.events.paid[:4].any())
        assert (env.bridge.views["rival2.episode_ticks"][:4] == 0).all()


def test_skill_and_natural_time_limits_have_prereset_bootstrap(env):
    env.family[5:7] = torch.tensor([1, 0], device=env.device, dtype=env.family.dtype)
    env.bridge.views["rival2.episode_ticks"][5:7] = torch.tensor(
        [1436, 3596], device=env.device, dtype=torch.int32
    )
    tr = env.step(torch.zeros((32, 2, 8), device=env.device))
    assert tr.truncated[5:7].all() and not tr.terminated[5:7].any()
    assert not torch.equal(tr.transition_observation[5:7], tr.observation[5:7])
    assert env.last_skill["events"][5, :, 6].all()
    assert not env.last_skill["events"][6].any()


def test_fixed_nexto_share_mask_and_role_telemetry(env):
    model = EntityJointControlActorCritic().cuda()
    collector = DirectSkillCollector(env, model, seed=71)
    collector.config = replace(collector.config, rollout_horizon=4)
    teacher_before = {k: v.clone() for k, v in collector.nexto.actor.state_dict().items()}
    buffer = collector.collect()
    metrics = collector.last_metrics
    assert metrics["trainable_agent_samples"] == 32 * 4 * 3 // 2
    assert metrics["nexto_training_sample_count"] * 3 == metrics["trainable_agent_samples"]
    assert (
        sum(v["samples"] for v in metrics["by_reward_role_and_opponent"].values())
        == metrics["trainable_agent_samples"]
    )
    assert not bool((buffer.train_mask & (buffer.opponent_family == 1)).sum(2).gt(1).any())
    assert all(
        torch.equal(v, collector.nexto.actor.state_dict()[k]) for k, v in teacher_before.items()
    )
    assert torch.equal(
        model.action_table[buffer.action_indices][buffer.train_mask],
        buffer.actions[buffer.train_mask],
    )
    parts = env.last_components
    expected = sum(v for k, v in parts.items() if k != "total") + env.last_skill["bonus"]
    torch.testing.assert_close(parts["total"], expected)


def test_untouched_ground_shots_score_in_native_physics():
    root = Path("G:/dev/RLBot-Rival/bot/collision_meshes")
    bank = scenarios(16, seed=88, family_only=3)
    # Remove car obstruction for a calibration-only check of initial shot geometry.
    bank.state.car_pos[..., 0] = 3500
    bank.state.car_vel.fill(0)
    world = DirectSkillsEnv(16, root, device="cuda:0", seed=88, ssl_foundation_scenarios=bank)
    alive = torch.ones(16, device=world.device, dtype=torch.bool)
    scored = torch.zeros(16, device=world.device, dtype=torch.bool)
    side = torch.tensor(bank.focal_side, device=world.device)
    for _ in range(240):
        tr = world.step(torch.zeros((16, 2, 8), device=world.device))
        scored |= alive & tr.terminated & (world.last_native["scoring_team"] == 1 - side)
        alive &= ~tr.reset_mask
        if not bool(alive.any()):
            break
    assert scored.all(), scored
