from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    GAMEPLAY_STRICT_DOUBLE_DASH_REWARD,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_mixed_ppo import (
    Rival2MixedPPOSafetyConfig,
    mixed_optimizer_learning_rates,
)
from rivalsim.rival2_opponent_curriculum import (
    OPPONENT_CURRENT,
    OPPONENT_HISTORICAL,
    OPPONENT_NEXTO,
    OPPONENT_WISP,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_ppo import Rival2PPOConfig


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
    *,
    seed: int,
) -> Rival2Env:
    root, geometry, meshes = assets
    return Rival2Env(
        count,
        root,
        geometry=geometry,
        meshes=meshes,
        device="cuda:0",
        seed=seed,
        car_visitation_order="a_then_b",
        reward_version=RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    )


def _optimizer_steps_by_parameter(
    trainer: Rival2OpponentCurriculumTrainer,
) -> dict[str, int | None]:
    output: dict[str, int | None] = {}
    for name, parameter in trainer.model.named_parameters():
        step = trainer.optimizer.state.get(parameter, {}).get("step")
        output[name] = None if step is None else int(step.item())
    return output


def test_gameplay_v2_reward_adds_only_competitive_strict_double_dash(
    arena_assets,
) -> None:
    env = _env(arena_assets, 1, seed=2026082703)
    env.world.begin_decision()
    env.world.rival2.interval_tick.fill_(3)
    dash_count = env.bridge.views["rival2.strict_double_dash_count"]
    dash_count.zero_()
    dash_count[0] = 1
    env.world.step(1)
    torch.cuda.synchronize(env.device)

    expected = torch.tensor(
        [[GAMEPLAY_STRICT_DOUBLE_DASH_REWARD, -GAMEPLAY_STRICT_DOUBLE_DASH_REWARD]],
        device=env.device,
    )
    reward = env.bridge.views["rival2.reward"].reshape(1, 2)
    component = env.bridge.views["rival2.strict_double_dash_component"]
    torch.testing.assert_close(reward, expected, rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(
        component,
        torch.tensor([GAMEPLAY_STRICT_DOUBLE_DASH_REWARD], device=env.device),
        rtol=0.0,
        atol=1.0e-7,
    )
    torch.testing.assert_close(reward.sum(dim=1), torch.zeros(1, device=env.device))


def test_mixed_opponent_assignment_train_mask_reset_and_checkpoint_round_trip(
    arena_assets,
) -> None:
    count = 32
    ppo = Rival2PPOConfig(rollout_horizon=2, minibatch_size=32, epochs=1)
    config = Rival2OpponentCurriculumConfig()
    trainer = Rival2OpponentCurriculumTrainer(
        _env(arena_assets, count, seed=2026082703),
        ppo_config=ppo,
        opponent_curriculum=config,
        seed=2026082703,
    )
    trainer.add_historical_snapshot()
    trainer.initialize_curriculum_assignments()

    assert torch.all((trainer.opponent_family >= 0) & (trainer.opponent_family < 4))
    assert int(trainer.realized_family_assignments.sum()) == count
    assert torch.equal(
        trainer.realized_family_assignments,
        torch.bincount(trainer.opponent_family, minlength=4),
    )
    samples_before = trainer.total_agent_samples
    rollout = trainer.collect_rollout()
    curriculum_metrics = trainer.last_rollout_curriculum_metrics
    assert curriculum_metrics is not None
    expected_samples = int(rollout.train_mask.sum())
    assert trainer.total_agent_samples - samples_before == expected_samples
    assert sum(curriculum_metrics["trainable_agent_samples"].values()) == expected_samples
    assert sum(curriculum_metrics["world_decisions"].values()) == count * ppo.rollout_horizon

    pattern = torch.tensor(
        [OPPONENT_CURRENT, OPPONENT_HISTORICAL, OPPONENT_NEXTO, OPPONENT_WISP],
        dtype=torch.int64,
        device=trainer.device,
    ).repeat(count // 4)
    side = torch.arange(count, device=trainer.device, dtype=torch.int64) & 1
    trainer.opponent_family.copy_(pattern)
    trainer.rival_side.copy_(side)
    trainer.opponent_assignment.fill_(-1)
    trainer.opponent_assignment[pattern == OPPONENT_HISTORICAL] = trainer.opponent_pool.versions[0]
    _actor, _value, _version, train_mask = trainer._policy_outputs(trainer.env.observation)
    current = pattern == OPPONENT_CURRENT
    assert torch.all(train_mask[current])
    frozen = ~current
    rows = torch.arange(count, device=trainer.device)[frozen]
    frozen_side = side[frozen]
    assert torch.all(train_mask[rows, frozen_side])
    assert torch.all(~train_mask[rows, 1 - frozen_side])

    # Ordinary calls without an episode reset preserve both assignment and RNG.
    family_before = trainer.opponent_family.clone()
    side_before = trainer.rival_side.clone()
    rng_before = trainer.curriculum_generator.get_state().clone()
    trainer.assign_opponents_at_reset(torch.zeros(count, dtype=torch.bool, device=trainer.device))
    torch.testing.assert_close(trainer.opponent_family, family_before, rtol=0, atol=0)
    torch.testing.assert_close(trainer.rival_side, side_before, rtol=0, atol=0)
    assert torch.equal(trainer.curriculum_generator.get_state(), rng_before)

    # Every newly reset Wisp episode clears source-required temporal history,
    # including the case where the newly sampled family is Wisp again.
    trainer.wisp.old_action.fill_(1.0)
    trainer.wisp.new_action.fill_(2.0)
    trainer.wisp.previous_action.fill_(3.0)
    trainer.wisp.ticks.fill_(5)
    trainer.wisp.update_flag.zero_()
    trainer.wisp.eta_cache.fill(7.0)
    reset = torch.ones(count, dtype=torch.bool, device=trainer.device)
    trainer.assign_opponents_at_reset(reset)
    wisp = trainer.opponent_family == OPPONENT_WISP
    assert bool(wisp.any())
    assert torch.all(trainer.wisp.old_action[wisp] == 0)
    assert torch.all(trainer.wisp.new_action[wisp] == 0)
    assert torch.all(trainer.wisp.previous_action[wisp] == 0)
    assert torch.all(trainer.wisp.ticks[wisp] == -1)
    assert torch.all(trainer.wisp.update_flag[wisp])
    wisp_host = wisp.cpu().numpy()
    assert np.all(trainer.wisp.eta_cache[wisp_host] == 0.0)

    checkpoint = Path(".tools/opponent_curriculum_test_checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint)
    restored = Rival2OpponentCurriculumTrainer(
        _env(arena_assets, count, seed=999),
        ppo_config=ppo,
        opponent_curriculum=config,
        seed=999,
    )
    restored.load_checkpoint(checkpoint)

    for name in (
        "opponent_family",
        "rival_side",
        "realized_family_assignments",
        "opponent_assignment",
    ):
        torch.testing.assert_close(getattr(restored, name), getattr(trainer, name), rtol=0, atol=0)
    assert torch.equal(
        restored.curriculum_generator.get_state(), trainer.curriculum_generator.get_state()
    )
    for name in ("player_index", "previous_action", "neural_counter", "kickoff_index"):
        torch.testing.assert_close(
            getattr(restored.nexto, name), getattr(trainer.nexto, name), rtol=0, atol=0
        )
    assert restored.nexto._cadence_tick == trainer.nexto._cadence_tick
    for name in (
        "player_index",
        "old_action",
        "new_action",
        "previous_action",
        "ticks",
        "update_flag",
        "opponent_slot",
    ):
        torch.testing.assert_close(
            getattr(restored.wisp, name), getattr(trainer.wisp, name), rtol=0, atol=0
        )
    np.testing.assert_array_equal(restored.wisp.eta_cache, trainer.wisp.eta_cache)
    assert torch.equal(
        restored.wisp.observation_generator.get_state(),
        trainer.wisp.observation_generator.get_state(),
    )
    checkpoint.unlink()


def test_safe_mixed_optimizer_and_retention_checkpoint_round_trip(
    arena_assets,
) -> None:
    count = 16
    ppo = Rival2PPOConfig(rollout_horizon=2, minibatch_size=16, epochs=1)
    trainer = Rival2OpponentCurriculumTrainer(
        _env(arena_assets, count, seed=2026082704),
        ppo_config=ppo,
        seed=2026082704,
    )
    trainer.initialize_curriculum_assignments()
    rollout = trainer.collect_rollout()
    safety = Rival2MixedPPOSafetyConfig(retention_corpus_size=8)
    migration = trainer.enable_safe_mixed_ppo(safety)
    summary = trainer.initialize_retention_corpus_from_rollout(
        rollout,
        source_identity={"test_fixture": "healthy_source_policy"},
    )
    assert migration["verdict"] == "PASS_GREEN"
    assert summary["verdict"] == "PASS_GREEN"

    checkpoint = Path(".tools/opponent_curriculum_safe_transition_test_checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adaptive = payload["opponent_curriculum"]["adaptive_ppo"]
    assert adaptive["schema_version"] == 2
    assert adaptive["policy_learning_rate_scope"] == "ppo_update_local"
    assert adaptive["next_update_policy_learning_rate"] == 1.0e-4
    assert adaptive["optimizer_learning_rates"] == {
        "policy": 1.0e-4,
        "critic": 3.0e-4,
    }
    restored = Rival2OpponentCurriculumTrainer(
        _env(arena_assets, count, seed=999),
        ppo_config=ppo,
        seed=999,
    )
    restored.load_checkpoint(checkpoint)

    assert restored.mixed_ppo_safety == safety
    assert restored.optimizer_migration_proof == migration
    assert restored.retention_corpus_summary == summary
    torch.testing.assert_close(
        restored.retention_observations,
        trainer.retention_observations,
        rtol=0,
        atol=0,
    )
    assert mixed_optimizer_learning_rates(restored.optimizer) == {
        "policy": safety.initial_policy_learning_rate,
        "critic": safety.critic_learning_rate,
    }

    # A schema-v1 checkpoint could have captured an update-local backed-off LR.
    # Loading it must retain Adam state while arming the next update at the base LR.
    old_payload = payload
    old_adaptive = old_payload["opponent_curriculum"]["adaptive_ppo"]
    for key in (
        "schema_version",
        "policy_learning_rate_scope",
        "next_update_policy_learning_rate",
        "last_update_summary",
    ):
        old_adaptive.pop(key, None)
    for group in old_payload["optimizer"]["param_groups"]:
        if group.get("name") == "policy":
            group["lr"] = safety.minimum_policy_learning_rate
    old_checkpoint = Path(".tools/opponent_curriculum_safe_transition_v1_test_checkpoint.pt")
    torch.save(old_payload, old_checkpoint)
    restored_old = Rival2OpponentCurriculumTrainer(
        _env(arena_assets, count, seed=1000),
        ppo_config=ppo,
        seed=1000,
    )
    restored_old.load_checkpoint(old_checkpoint)
    assert mixed_optimizer_learning_rates(restored_old.optimizer) == {
        "policy": safety.initial_policy_learning_rate,
        "critic": safety.critic_learning_rate,
    }
    assert _optimizer_steps_by_parameter(restored_old) == _optimizer_steps_by_parameter(trainer)
    checkpoint.unlink()
    old_checkpoint.unlink()
