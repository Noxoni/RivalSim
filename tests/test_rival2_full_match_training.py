from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import warp as wp

from rivalsim import StateSnapshot
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.kernels.rival2_full_match import REGULATION_TICKS
from rivalsim.rival2_contracts import (
    RIVAL2_REWARD_GOAL_ONLY_VERSION,
    RIVAL2_REWARD_V2_VERSION,
)
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_training import Rival2Trainer


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs, Warp CUDA, and PyTorch CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return root, geometry, WarpArenaMeshes(geometry)


def _env(
    assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    count: int = 1,
    **kwargs: object,
) -> Rival2FullMatchEnv:
    root, geometry, meshes = assets
    return Rival2FullMatchEnv(
        count,
        root,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
        **kwargs,
    )


def test_goal_resets_kickoff_but_does_not_end_match(arena_assets) -> None:
    state = StateSnapshot.empty(1)
    state.car_pos[:] = (0.0, 0.0, 1500.0)
    state.ball_pos[:] = (0.0, 5300.0, 93.15)
    env = _env(arena_assets, initial=state)
    transition = env.step(torch.zeros((1, 2, 8), device=env.device))
    torch.cuda.synchronize()
    assert not transition.terminated.item()
    assert not transition.truncated.item()
    assert not transition.reset_mask.item()
    assert env.full_match_views["blue_score"].item() == 1
    assert env.full_match_views["orange_score"].item() == 0
    assert env.full_match_views["regulation_ticks_remaining"].item() == (
        REGULATION_TICKS - 4
    )
    np.testing.assert_allclose(
        env.world.state.ball_pos.numpy(), ((0.0, 0.0, 93.15),), atol=1.0e-5
    )
    assert env.full_match_views["kickoff_segment_active"].item() == 1


def test_regulation_and_overtime_are_match_terminal_only(arena_assets) -> None:
    lead = _env(arena_assets)
    lead.full_match_views["regulation_ticks_remaining"].fill_(4)
    lead.full_match_views["blue_score"].fill_(1)
    result = lead.step(torch.zeros((1, 2, 8), device=lead.device))
    torch.cuda.synchronize()
    assert result.terminated.item() and not result.truncated.item()
    assert result.reset_mask.item()
    assert lead.full_match_views["completed_matches"].item() == 1
    assert lead.full_match_views["completed_blue_wins"].item() == 1
    assert lead.full_match_views["regulation_ticks_remaining"].item() == REGULATION_TICKS
    assert lead.full_match_views["blue_score"].item() == 0

    tied = _env(arena_assets)
    tied.full_match_views["regulation_ticks_remaining"].fill_(4)
    regulation = tied.step(torch.zeros((1, 2, 8), device=tied.device))
    torch.cuda.synchronize()
    assert not regulation.terminated.item() and not regulation.truncated.item()
    assert tied.full_match_views["overtime"].item() == 1
    assert tied.full_match_views["regulation_ticks_remaining"].item() == 0
    overtime_state = StateSnapshot.empty(1)
    overtime_state.car_pos[:] = (0.0, 0.0, 1500.0)
    overtime_state.ball_pos[:] = (0.0, 5300.0, 93.15)
    overtime_env = _env(arena_assets, initial=overtime_state)
    overtime_env.full_match_views["regulation_ticks_remaining"].fill_(0)
    overtime_env.full_match_views["overtime"].fill_(1)
    overtime = overtime_env.step(
        torch.zeros((1, 2, 8), device=overtime_env.device)
    )
    torch.cuda.synchronize()
    assert overtime.terminated.item() and not overtime.truncated.item()
    assert overtime_env.full_match_views["completed_matches"].item() == 1


def test_no_touch_timeout_is_diagnostic_only(arena_assets) -> None:
    env = _env(arena_assets)
    env.full_match_views["kickoff_segment_ticks"].fill_(15 * 120 - 4)
    transition = env.step(torch.zeros((1, 2, 8), device=env.device))
    torch.cuda.synchronize()
    assert not transition.terminated.item()
    assert not transition.truncated.item()
    assert not transition.reset_mask.item()
    assert env.full_match_views["kickoff_segments_total"].item() == 1
    assert env.full_match_views["no_touch_segments_total"].item() == 1
    assert env.full_match_views["regulation_ticks_remaining"].item() == (
        REGULATION_TICKS - 4
    )


def test_goal_only_reward_removes_every_shaping_term(arena_assets) -> None:
    state = StateSnapshot.empty(1)
    state.car_pos[:] = (0.0, 0.0, 1500.0)
    state.ball_pos[:] = (0.0, 0.0, 1000.0)
    state.ball_vel[:] = (0.0, 1000.0, 0.0)
    env = _env(
        arena_assets,
        initial=state,
        reward_version=RIVAL2_REWARD_GOAL_ONLY_VERSION,
    )
    transition = env.step(torch.zeros((1, 2, 8), device=env.device))
    torch.cuda.synchronize()
    torch.testing.assert_close(transition.reward, torch.zeros_like(transition.reward))


def test_full_match_env_runs_real_ppo_and_reward_transition(arena_assets) -> None:
    env = _env(arena_assets, count=2, reward_version=RIVAL2_REWARD_V2_VERSION)
    trainer = Rival2Trainer(
        env,
        ppo_config=Rival2PPOConfig(rollout_horizon=2, minibatch_size=8, epochs=1),
        seed=20260826,
    )
    _rollout, metrics = trainer.train_iteration()
    assert all(torch.isfinite(value).item() for value in metrics.values())
    before = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
    trainer.transition_reward_curriculum(
        source_reward_version=RIVAL2_REWARD_V2_VERSION,
        destination_reward_version=RIVAL2_REWARD_GOAL_ONLY_VERSION,
        transition_record={"authority": "full-match curriculum unit test"},
    )
    assert env.reward_version == RIVAL2_REWARD_GOAL_ONLY_VERSION
    assert all(
        torch.equal(before[name], value)
        for name, value in trainer.model.state_dict().items()
    )


def test_masked_training_counts_only_first_match_samples(arena_assets) -> None:
    env = _env(arena_assets, count=2, reward_version=RIVAL2_REWARD_GOAL_ONLY_VERSION)
    trainer = Rival2Trainer(
        env,
        ppo_config=Rival2PPOConfig(rollout_horizon=2, minibatch_size=4, epochs=1),
        seed=20260827,
    )
    active = torch.tensor((True, False), device=env.device)
    rollout, metrics = trainer.train_iteration(active)
    assert trainer.total_agent_samples == 4
    assert not rollout.train_mask[:, 1].any().item()
    assert all(torch.isfinite(value).item() for value in metrics.values())


def test_explicit_phase_boundary_starts_fresh_complete_matches(arena_assets) -> None:
    env = _env(arena_assets, count=2)
    env.full_match_views["blue_score"].fill_(3)
    env.full_match_views["regulation_ticks_remaining"].fill_(17)
    env.full_match_views["completed_matches"].fill_(5)
    env.start_fresh_matches()
    torch.cuda.synchronize()
    assert torch.all(env.full_match_views["blue_score"] == 0).item()
    assert torch.all(
        env.full_match_views["regulation_ticks_remaining"] == REGULATION_TICKS
    ).item()
    assert torch.all(env.full_match_views["completed_matches"] == 5).item()
