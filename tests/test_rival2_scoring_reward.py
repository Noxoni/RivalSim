from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.kernels.rival2_full_match import REGULATION_TICKS
from rivalsim.rival2_contracts import (
    OBS_FIELD_NAMES,
    REWARD_SCORING_V1_CONTRACT,
    REWARD_SCORING_V1_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_FULL_MATCH_EPISODE_VERSION,
    RIVAL2_REWARD_ACQUISITION_V1_VERSION,
    RIVAL2_REWARD_SCORING_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_training import Rival2Trainer
from rivalsim.v03_corpus import generate_phase_b_cases, phase_b_cases_to_state


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs, Warp CUDA, and PyTorch CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return root, geometry, WarpArenaMeshes(geometry)


def _full_env(
    assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    count: int = 1,
    **kwargs: Any,
) -> Rival2FullMatchEnv:
    root, geometry, meshes = assets
    return Rival2FullMatchEnv(
        count,
        root,
        reward_version=RIVAL2_REWARD_SCORING_V1_VERSION,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
        **kwargs,
    )


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
            left.cpu(), right.cpu()
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def test_scoring_contract_is_explicit_and_content_addressed() -> None:
    assert REWARD_SCORING_V1_CONTRACT["goal"] == {
        "score": 10.0,
        "concede": -10.0,
        "zero_sum": True,
    }
    assert REWARD_SCORING_V1_CONTRACT["approach"]["coefficient"] == 0.10
    assert REWARD_SCORING_V1_CONTRACT["unique_touch_per_player"]["reward"] == 0.02
    assert (
        REWARD_SCORING_V1_CONTRACT["first_legitimate_touch_per_player_per_match"][
            "reward"
        ]
        == 0.0
    )
    assert REWARD_SCORING_V1_CONTRACT["no_touch_failure"] is None
    hashes = contract_hashes_for_reward(
        RIVAL2_REWARD_SCORING_V1_VERSION,
        RIVAL2_FULL_MATCH_EPISODE_VERSION,
    )
    assert hashes[RIVAL2_REWARD_SCORING_V1_VERSION] == (
        REWARD_SCORING_V1_CONTRACT_HASH
    )


def test_scoring_auxiliary_terms_are_event_based(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    env = _full_env(arena_assets)
    before = torch.zeros_like(env.observation)
    after = torch.zeros_like(before)
    action = torch.zeros((1, 2, 8), dtype=torch.float32, device=env.device)
    relative = OBS_FIELD_NAMES.index("relative.ball_position.x")
    previous_jump = OBS_FIELD_NAMES.index("previous_action.jump")
    has_flipped = OBS_FIELD_NAMES.index("self.has_flipped")
    before[..., relative] = 1.0
    after[..., relative] = 0.75
    action[..., 5] = 1.0
    before[..., previous_jump] = 0.0
    after[..., has_flipped] = 1.0

    auxiliary = env.bridge.scoring_auxiliary_reward(before, after, action)
    # 1024 UU of approach contributes +0.025, then one jump edge and one
    # actual flip onset contribute -0.002 and -0.01.
    torch.testing.assert_close(
        auxiliary,
        torch.full_like(auxiliary, 0.013),
        rtol=0.0,
        atol=2.0e-7,
    )

    # Holding jump and remaining in the same flip state do not incur another cost.
    held_before = after.clone()
    held_before[..., previous_jump] = 1.0
    held = env.bridge.scoring_auxiliary_reward(held_before, held_before, action)
    torch.testing.assert_close(held, torch.zeros_like(held), rtol=0.0, atol=1.0e-7)


def test_scoring_step_uses_progress_small_touch_and_auxiliary_terms(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    initial = phase_b_cases_to_state((generate_phase_b_cases()[0],))
    env = _full_env(arena_assets, initial=initial)
    before = env.observation.clone()
    action = torch.zeros((1, 2, 8), device=env.device)
    transition = env.step(action)
    after = transition.transition_observation
    ball_y = OBS_FIELD_NAMES.index("ball.position.y")
    touch = OBS_FIELD_NAMES.index("lifecycle.self_touch_event")
    progress = 0.5 * (after[..., ball_y] - before[..., ball_y])
    expected = progress + 0.02 * (after[..., touch] > 0.5).to(torch.float32)
    expected += env.bridge.scoring_auxiliary_reward(before, after, action)
    torch.cuda.synchronize()
    torch.testing.assert_close(transition.reward, expected, rtol=0.0, atol=2.0e-6)


def test_checkpoint_transition_preserves_compatible_state_and_freshens_match(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
    tmp_path: Path,
) -> None:
    root, geometry, meshes = arena_assets
    ppo = Rival2PPOConfig(rollout_horizon=2, minibatch_size=8, epochs=1)
    source_env = Rival2Env(
        2,
        root,
        reward_version=RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    source = Rival2Trainer(source_env, ppo_config=ppo, seed=20260827)
    source.add_historical_snapshot()
    source.train_iteration()
    checkpoint = tmp_path / "acquisition.pt"
    source.save_checkpoint(checkpoint)
    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    destination_env = _full_env(arena_assets, count=2)
    destination = Rival2Trainer(destination_env, ppo_config=ppo, seed=1)
    transition = destination.load_checkpoint_curriculum_transition(
        checkpoint,
        source_reward_version=RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        source_episode_version=RIVAL2_EPISODE_VERSION,
        transition_record={"authority": "scoring transition unit test"},
    )
    destination_payload = destination.checkpoint_payload()
    preserved = (
        "model",
        "optimizer",
        "policy_config",
        "ppo_config",
        "self_play_config",
        "policy_config_hash",
        "ppo_config_hash",
        "policy_version",
        "iteration",
        "total_agent_samples",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "policy_generator_state",
        "opponent_generator_state",
        "opponent_assignment",
        "historical_opponents",
    )
    assert all(
        _nested_exact(source_payload[name], destination_payload[name])
        for name in preserved
    )
    assert transition["destination_reward_version"] == (
        RIVAL2_REWARD_SCORING_V1_VERSION
    )
    assert transition["destination_episode_version"] == (
        RIVAL2_FULL_MATCH_EPISODE_VERSION
    )
    assert torch.all(destination_env.full_match_views["blue_score"] == 0).item()
    assert torch.all(destination_env.full_match_views["orange_score"] == 0).item()
    assert torch.all(
        destination_env.full_match_views["regulation_ticks_remaining"]
        == REGULATION_TICKS
    ).item()
    rollout, metrics = destination.train_iteration()
    assert not rollout.truncated.any().item()
    assert destination.iteration == int(source_payload["iteration"]) + 1
    assert all(torch.isfinite(value).item() for value in metrics.values())
