from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import warp as wp

from benchmarks.run_rival2_human_bc_ppo_v1 import advance_warmup_state
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_120hz_transition import tensor_tree_sha256
from rivalsim.rival2_contracts import (
    REWARD_GAMEPLAY_120_V1_CONTRACT,
    RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_mixed_ppo import (
    Rival2MixedPPOSafetyConfig,
    mixed_optimizer_learning_rates,
    probe_fresh_adam_first_minibatch,
)
from rivalsim.rival2_opponent_curriculum import (
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_training import Rival2SelfPlayConfig


def test_frozen_human_bc_ppo_campaign_contract() -> None:
    config = json.loads(
        Path("results/rival2/human_bc_ppo_v1/frozen_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["required_parent"] == "a8d0bfc98077b4c194ad3438640fd79ed2b5b788"
    assert config["human_bc_parent"]["accepted_step"] == 160
    assert config["human_bc_parent"]["sha256"] == (
        "560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874"
    )
    assert config["runtime"] == {
        "worlds": 32768,
        "physics_hz": 120,
        "policy_hz": 120,
        "physics_ticks_per_policy_action": 1,
        "rollout_horizon": 128,
    }
    assert config["opponents"]["current_probability"] == 0.8
    assert config["opponents"]["historical_probability"] == 0.2
    assert config["opponents"]["nexto_probability"] == 0.0
    assert config["opponents"]["wisp_probability"] == 0.0
    assert config["fresh_optimizer"]["restore_historical_adam"] is False
    assert config["campaign"]["mode"] == "bounded_wall_clock"
    assert config["campaign"]["training_duration_hours"] == 10.0
    assert config["campaign"]["checkpoint_every_accepted_updates"] == 30
    assert config["campaign"]["original_plus_120_boundary_is_not_a_stop"] is True


def test_clean_reward_contract_has_no_named_mechanic_signal_or_exemption() -> None:
    contract = REWARD_GAMEPLAY_120_V1_CONTRACT
    assert contract["unconditional_unique_touch"] == 0.0
    assert contract["named_mechanics_reward"] == 0.0
    assert contract["named_mechanics_hot_path"] is False
    assert contract["bad_flip_guard"]["active_exemptions_in_precedence_order"] == [
        "EXEMPT_CONTESTED_50",
        "EXEMPT_POWER_CONTACT",
    ]
    assert contract["bad_flip_guard"]["recognized_mechanic_exemption"] is False
    assert contract["bad_flip_guard"]["controlled_flick_exemption"] is False
    assert contract["bad_flip_guard"]["generic_jump_penalty"] == 0.0
    assert contract["bad_flip_guard"]["generic_flip_penalty"] == 0.0


def test_transition_warmup_doubles_only_after_clean_updates() -> None:
    state = {
        "active": True,
        "selected_initial_policy_lr": 1.25e-5,
        "next_update_starting_policy_lr": 1.25e-5,
        "warmup_updates_completed": 0,
        "consecutive_clean_updates_at_1e-4": 0,
        "normal_production_operation_start_offset": None,
    }
    clean = {
        "policy_learning_rate_backoffs": 0,
        "ppo_early_stop": False,
        "policy_learning_rate_end": 1.25e-5,
    }
    state = advance_warmup_state(state, clean, accepted_offset=1)
    assert state["next_update_starting_policy_lr"] == 2.5e-5
    clean["policy_learning_rate_end"] = 2.5e-5
    state = advance_warmup_state(state, clean, accepted_offset=2)
    assert state["next_update_starting_policy_lr"] == 5.0e-5
    clean["policy_learning_rate_end"] = 5.0e-5
    state = advance_warmup_state(state, clean, accepted_offset=3)
    assert state["next_update_starting_policy_lr"] == 1.0e-4
    clean["policy_learning_rate_end"] = 1.0e-4
    state = advance_warmup_state(state, clean, accepted_offset=4)
    assert state["active"] is True
    state = advance_warmup_state(state, clean, accepted_offset=5)
    assert state["active"] is False
    assert state["normal_production_operation_start_offset"] == 6

    backoff_state = {
        **state,
        "active": True,
        "next_update_starting_policy_lr": 2.5e-5,
        "consecutive_clean_updates_at_1e-4": 0,
    }
    backed_off = {
        "policy_learning_rate_backoffs": 1,
        "ppo_early_stop": False,
        "policy_learning_rate_end": 1.25e-5,
    }
    backoff_state = advance_warmup_state(
        backoff_state, backed_off, accepted_offset=6
    )
    assert backoff_state["next_update_starting_policy_lr"] == 1.25e-5


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    configured = os.environ.get("RIVALSIM_COLLISION_DIR")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path("G:/dev/RLBot-Rival/bot/collision_meshes"))
    root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "soccar" / "mesh_0.cmf").is_file()
        ),
        None,
    )
    if root is None or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs and CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return str(root), geometry, WarpArenaMeshes(geometry)


def test_clean_runtime_monitor_and_fresh_optimizer_gate(arena_assets) -> None:
    root, geometry, meshes = arena_assets
    env = Rival2Env(
        64,
        root,
        geometry=geometry,
        meshes=meshes,
        device="cuda:0",
        seed=2026082907,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        ppo_config=Rival2PPOConfig(
            entropy_coefficient=0.0,
            rollout_horizon=4,
            minibatch_size=64,
            epochs=1,
        ),
        self_play_config=Rival2SelfPlayConfig(
            historical_chance=0.2,
            historical_pool_bound=16,
        ),
        opponent_curriculum=Rival2OpponentCurriculumConfig(
            nexto_probability=0.0,
            wisp_probability=0.0,
            current_probability=0.8,
            historical_probability=0.2,
            seed=2026082908,
        ),
        seed=2026082907,
    )
    trainer.add_historical_snapshot()
    trainer.initialize_curriculum_assignments()
    assert len(trainer.optimizer.state) == 0
    trainer.enable_safe_mixed_ppo(Rival2MixedPPOSafetyConfig())
    assert len(trainer.optimizer.state) == 0
    assert mixed_optimizer_learning_rates(trainer.optimizer) == {
        "policy": 1.0e-4,
        "critic": 3.0e-4,
    }
    rollout = trainer.collect_rollout()
    metrics = trainer.last_rollout_gameplay_metrics
    curriculum = trainer.last_rollout_curriculum_metrics
    assert metrics is not None
    assert curriculum is not None
    assert int((trainer.opponent_family == 2).sum()) == 0
    assert int((trainer.opponent_family == 3).sum()) == 0
    assert curriculum["trainable_agent_samples"]["nexto"] == 0
    assert curriculum["trainable_agent_samples"]["wisp"] == 0
    assert metrics["named_mechanics_hot_path_absent"] is True
    assert metrics["named_mechanics_arrays"] == 0
    assert metrics["trusted_reward_component_absolute_sum"]["v1_touch_component"] == 0.0
    assert (
        metrics["trusted_reward_component_absolute_sum"][
            "strict_double_dash_component"
        ]
        == 0.0
    )
    assert torch.isfinite(rollout.observations).all()
    assert torch.isfinite(rollout.actions).all()
    assert torch.isfinite(rollout.rewards).all()
    assert len(trainer.optimizer.state) == 0


def test_retention_metadata_binds_the_active_observation_contract(arena_assets) -> None:
    root, geometry, meshes = arena_assets
    env = Rival2Env(
        256,
        root,
        geometry=geometry,
        meshes=meshes,
        device="cuda:0",
        seed=2026082910,
        reward_version=RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        ppo_config=Rival2PPOConfig(
            entropy_coefficient=0.0,
            rollout_horizon=2,
            minibatch_size=256,
            epochs=1,
        ),
        opponent_curriculum=Rival2OpponentCurriculumConfig(
            nexto_probability=0.0,
            wisp_probability=0.0,
            current_probability=1.0,
            historical_probability=0.0,
            seed=2026082911,
        ),
        seed=2026082910,
    )
    trainer.curriculum_transition = {"identity": "TEST"}
    trainer.initialize_curriculum_assignments()
    trainer.enable_safe_mixed_ppo(Rival2MixedPPOSafetyConfig())
    rollout = trainer.collect_rollout()
    summary = trainer.initialize_retention_corpus_from_rollout(
        rollout,
        source_identity={"identity": "TEST_120HZ_PARENT"},
    )
    assert summary["observation_contract"] == "RIVAL2_OBS_V2_120HZ"
    assert summary["observation_contract_sha256"] == env.contract_hashes[
        "RIVAL2_OBS_V2_120HZ"
    ]
    model_before = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_before = tensor_tree_sha256(trainer.optimizer.state_dict())
    generator_before = trainer.policy_generator.get_state().clone()
    probe = probe_fresh_adam_first_minibatch(
        trainer.model,
        rollout,
        trainer.ppo_config,
        retention_observations=trainer.retention_observations,
        family_names=("current", "historical", "nexto", "wisp"),
        generator=trainer.policy_generator,
        policy_learning_rate=1.5625e-6,
        critic_learning_rate=3.0e-4,
        policy_config=trainer.policy_config,
    )
    assert probe["rollback_complete"] is True
    assert probe["value_loss_isolated_from_policy_trunk_and_actor"] is True
    assert tensor_tree_sha256(trainer.model.state_dict()) == model_before
    assert tensor_tree_sha256(trainer.optimizer.state_dict()) == optimizer_before
    assert torch.equal(trainer.policy_generator.get_state(), generator_before)
