from __future__ import annotations

import copy
import os
from typing import Any

import pytest
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_ppo import (
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    Rival2PPOConfig,
)
from rivalsim.rival2_training import Rival2Trainer


@pytest.fixture(scope="module")
def arena_assets() -> tuple[str, ArenaGeometry, WarpArenaMeshes]:
    root = os.environ.get("RIVALSIM_COLLISION_DIR")
    if not root or not wp.is_cuda_available() or not torch.cuda.is_available():
        pytest.skip("exact local CMFs, Warp CUDA, and PyTorch CUDA are required")
    geometry = ArenaGeometry.load_soccar(root)
    return root, geometry, WarpArenaMeshes(geometry)


def _trainer(assets: tuple[str, ArenaGeometry, WarpArenaMeshes], seed: int) -> Rival2Trainer:
    root, geometry, meshes = assets
    env = Rival2Env(
        2,
        root,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    return Rival2Trainer(
        env,
        ppo_config=Rival2PPOConfig(
            rollout_horizon=2,
            minibatch_size=8,
            epochs=1,
            entropy_coefficient=0.0,
        ),
        seed=seed,
    )


def _exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_exact(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def test_kl_rejection_restores_model_optimizer_and_rng_exactly(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    trainer = _trainer(arena_assets, 20260827)
    rollout = trainer.collect_rollout()
    model_before = copy.deepcopy(trainer.model.state_dict())
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state(trainer.device).clone()
    policy_rng_before = trainer.policy_generator.get_state().clone()
    opponent_rng_before = trainer.opponent_generator.get_state().clone()
    iteration_before = trainer.iteration
    policy_version_before = trainer.policy_version
    samples_before = trainer.total_agent_samples

    with pytest.raises(Rival2PolicyDisplacementRejected) as caught:
        trainer.update(
            rollout,
            kl_guard=Rival2KLGuardConfig(
                minibatch_kl_limit=1.0e-12,
                completed_update_mean_kl_limit=1.0e-12,
            ),
        )

    assert caught.value.diagnostics["transactional_rollback_completed"] is True
    assert trainer.iteration == iteration_before
    assert trainer.policy_version == policy_version_before
    assert trainer.total_agent_samples == samples_before
    assert _exact(model_before, trainer.model.state_dict())
    assert _exact(optimizer_before, trainer.optimizer.state_dict())
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    assert torch.equal(torch.cuda.get_rng_state(trainer.device), cuda_rng_before)
    assert torch.equal(trainer.policy_generator.get_state(), policy_rng_before)
    assert torch.equal(trainer.opponent_generator.get_state(), opponent_rng_before)


def test_guarded_update_publishes_required_compact_diagnostics(
    arena_assets: tuple[str, ArenaGeometry, WarpArenaMeshes],
) -> None:
    trainer = _trainer(arena_assets, 20260828)
    rollout, metrics = trainer.train_iteration(
        kl_guard=Rival2KLGuardConfig(
            minibatch_kl_limit=1.0e6,
            completed_update_mean_kl_limit=1.0e6,
        )
    )
    required = {
        "approx_kl",
        "clip_fraction",
        "value_loss",
        "policy_loss",
        "gradient_norm",
        "post_clip_gradient_norm",
        "predicted_value_mean",
        "predicted_value_std",
        "predicted_value_max_abs",
        "return_mean",
        "return_std",
        "return_max_abs",
        "advantage_before_normalization_mean",
        "advantage_before_normalization_std",
        "advantage_before_normalization_max_abs",
        "actor_mean_mean_throttle",
        "actor_mean_abs_mean_throttle",
        "actor_mean_abs_max_throttle",
        "actor_log_std_mean_throttle",
        "emitted_action_saturation_fraction_throttle",
        "actor_button_probability_jump",
    }
    assert required <= metrics.keys()
    assert all(torch.isfinite(value).item() for value in metrics.values())
    assert rollout.position == trainer.ppo_config.rollout_horizon
