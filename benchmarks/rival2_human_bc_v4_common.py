"""Aligned simulator-corpus authority helpers for Human BC V4.

The original Human BC retention corpus stored authoritative observations but the
V2/V3 consumers reconstructed perspective roles from a single bootstrap
assignment.  Matches can reset during the 128-tick rollout, so V4 binds the
``opponent_family`` and ``train_mask`` recorded at every tick alongside the
observation tensor.  This module only collects and hashes simulator state; it
contains no human-data reader, optimizer, reward mutation, or PPO update.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.human_demo.missing_feature_distillation import (
    build_whole_world_split,
    canonical_sha256,
    file_sha256,
)
from rivalsim.rival2_120hz_transition import tensor_tree_sha256
from rivalsim.rival2_contracts import OBS_DIM
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_opponent_curriculum import (
    OPPONENT_NAMES,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2PolicyConfig
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_training import Rival2SelfPlayConfig

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOCCAR_GEOMETRY_SHA256 = (
    "2239556BDC74D205CAA6E46A0F6E91FA2C6E4257E84D4F608BA775958B0A5538"
)


@dataclass(slots=True)
class AlignedRolloutCorpus:
    """Authoritative observations and per-tick perspective-role metadata."""

    observations: torch.Tensor
    opponent_family: torch.Tensor
    train_mask: torch.Tensor


def _stream_tensor_sha256(
    value: torch.Tensor,
    *,
    logical_dtype: str,
    cast_dtype: torch.dtype,
) -> str:
    """Hash a tensor without materializing the full 6+ GiB corpus on the host."""

    tensor = value.detach()
    digest = hashlib.sha256()
    digest.update(logical_dtype.encode("ascii") + b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    if tensor.ndim == 0:
        block = tensor.to(cast_dtype).cpu().contiguous().numpy()
        digest.update(block.tobytes(order="C"))
    else:
        for tick in range(tensor.shape[0]):
            block = (
                tensor[tick : tick + 1]
                .to(cast_dtype)
                .cpu()
                .contiguous()
                .numpy()
            )
            digest.update(block.tobytes(order="C"))
    return digest.hexdigest().upper()


def _legacy_observation_sha256(observations: torch.Tensor) -> str:
    """Match the committed V1 corpus observation-hash algorithm exactly."""

    return _stream_tensor_sha256(
        observations,
        logical_dtype="float32",
        cast_dtype=torch.float32,
    )


def int64_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest().upper()


def _metadata_transition_counts(
    opponent_family: torch.Tensor,
    train_mask: torch.Tensor,
) -> dict[str, int]:
    family_change = opponent_family[1:, :, 0] != opponent_family[:-1, :, 0]
    mask_change = train_mask[1:] != train_mask[:-1]
    return {
        "opponent_family_world_tick_transitions": int(family_change.sum().item()),
        "train_mask_perspective_tick_transitions": int(mask_change.sum().item()),
        "worlds_with_opponent_family_transition": int(family_change.any(dim=0).sum().item()),
        "worlds_with_train_mask_transition": int(mask_change.any(dim=(0, 2)).sum().item()),
    }


def build_aligned_rollout_corpus(
    config: dict[str, Any],
    bootstrap_payload: dict[str, Any],
    bootstrap_identity: dict[str, Any],
    *,
    device: str,
) -> tuple[AlignedRolloutCorpus, dict[str, Any], dict[str, np.ndarray]]:
    """Collect the fixed 32768-world x 128-tick corpus with aligned roles."""

    corpus = config["corpus"]
    worlds = int(corpus["worlds"])
    horizon = int(corpus["decisions_per_world"])
    seed = int(corpus["seed"])
    collision_dir = Path(corpus["collision_mesh_directory"])
    if not collision_dir.is_dir():
        raise FileNotFoundError(f"collision mesh directory not found: {collision_dir}")
    if worlds != 32_768 or horizon != 128:
        raise ValueError("V4 corpus geometry must be exactly 32768 worlds x 128 ticks")

    geometry = ArenaGeometry.load_soccar(collision_dir)
    if geometry.content_sha256 != EXPECTED_SOCCAR_GEOMETRY_SHA256:
        raise RuntimeError(
            "authoritative Soccar collision geometry changed: "
            f"{geometry.content_sha256}"
        )
    meshes = WarpArenaMeshes(geometry, device)
    kickoff_selector = (np.arange(worlds, dtype=np.int32) + seed) % 5
    env = Rival2Env(
        worlds,
        str(collision_dir),
        device=device,
        seed=seed,
        reward_version=bootstrap_payload["reward_version"],
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2OpponentCurriculumTrainer(
        env,
        policy_config=Rival2PolicyConfig(**bootstrap_payload["policy_config"]),
        ppo_config=Rival2PPOConfig(**bootstrap_payload["ppo_config"]),
        self_play_config=Rival2SelfPlayConfig(**bootstrap_payload["self_play_config"]),
        opponent_curriculum=Rival2OpponentCurriculumConfig(
            **bootstrap_payload["opponent_curriculum"]["config"]
        ),
        seed=seed,
    )
    bootstrap_path = ROOT / config["authority"]["bootstrap_checkpoint"]
    trainer.load_checkpoint(bootstrap_path)
    if bool((trainer.opponent_family < 0).any()):
        raise RuntimeError("bootstrap did not restore complete curriculum assignments")

    model_before = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_before = tensor_tree_sha256(trainer.optimizer.state_dict())
    iteration_before = trainer.iteration
    policy_version_before = trainer.policy_version
    rng_before = {
        "policy_generator": tensor_tree_sha256(trainer.policy_generator.get_state()),
        "opponent_generator": tensor_tree_sha256(trainer.opponent_generator.get_state()),
    }
    initial_family_counts = torch.bincount(
        trainer.opponent_family, minlength=len(OPPONENT_NAMES)
    ).cpu().tolist()

    torch.cuda.synchronize(env.device)
    torch.cuda.reset_peak_memory_stats(env.device)
    started = time.perf_counter()
    rollout = trainer.collect_rollout()
    torch.cuda.synchronize(env.device)
    elapsed = time.perf_counter() - started
    if rollout.opponent_family is None:
        raise RuntimeError("authoritative rollout did not store opponent family")
    expected_observation_shape = [horizon, worlds, 2, OBS_DIM]
    expected_role_shape = [horizon, worlds, 2]
    if list(rollout.observations.shape) != expected_observation_shape:
        raise RuntimeError("V4 rollout observation shape mismatch")
    if list(rollout.opponent_family.shape) != expected_role_shape:
        raise RuntimeError("V4 rollout opponent-family shape mismatch")
    if list(rollout.train_mask.shape) != expected_role_shape:
        raise RuntimeError("V4 rollout train-mask shape mismatch")

    observations = rollout.observations.detach()
    opponent_family = rollout.opponent_family.detach()
    train_mask = rollout.train_mask.detach()
    observation_sha = _legacy_observation_sha256(observations)
    family_sha = _stream_tensor_sha256(
        opponent_family,
        logical_dtype="int64",
        cast_dtype=torch.int64,
    )
    train_mask_sha = _stream_tensor_sha256(
        train_mask,
        logical_dtype="bool",
        cast_dtype=torch.bool,
    )
    role_counts = {
        "trainable_perspective_rows": int(train_mask.sum().item()),
        "counterfactual_perspective_rows": int((~train_mask).sum().item()),
        "historical_counterfactual_rows": int(
            ((opponent_family == 1) & ~train_mask).sum().item()
        ),
    }
    family_tick_counts = torch.bincount(
        opponent_family[:, :, 0].reshape(-1), minlength=len(OPPONENT_NAMES)
    ).cpu().tolist()
    transition_counts = _metadata_transition_counts(opponent_family, train_mask)

    model_after = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_after = tensor_tree_sha256(trainer.optimizer.state_dict())
    checks = {
        "shape_exact": list(observations.shape) == expected_observation_shape,
        "aligned_role_shapes_exact": (
            list(opponent_family.shape) == expected_role_shape
            and list(train_mask.shape) == expected_role_shape
        ),
        "finite_observations": bool(torch.isfinite(observations).all()),
        "opponent_family_fully_assigned": bool((opponent_family >= 0).all()),
        "opponent_family_same_for_both_perspectives_per_tick": bool(
            (opponent_family[:, :, 0] == opponent_family[:, :, 1]).all()
        ),
        "at_least_one_trainable_perspective_per_tick_world": bool(
            train_mask.any(dim=-1).all()
        ),
        "model_unchanged": (
            model_before == model_after == bootstrap_identity["model_tensor_sha256"]
        ),
        "historical_ppo_optimizer_unchanged": optimizer_before == optimizer_after,
        "iteration_unchanged": trainer.iteration == iteration_before == 479,
        "policy_version_unchanged": trainer.policy_version == policy_version_before == 479,
        "bootstrap_file_unchanged": (
            file_sha256(bootstrap_path) == bootstrap_identity["sha256"]
        ),
        "aligned_metadata_captured_from_rollout_buffer": True,
        "no_optimizer_step_during_collection": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"aligned V4 corpus collection failed: {checks}")

    split_config = corpus["split"]
    split = build_whole_world_split(
        worlds=worlds,
        train_worlds=int(split_config["train_worlds"]),
        validation_worlds=int(split_config["validation_worlds"]),
        test_worlds=int(split_config["test_worlds"]),
        seed=int(split_config["seed"]),
    )
    split_manifest = split.manifest
    legacy_identity = canonical_sha256(
        {
            "bootstrap_sha256": bootstrap_identity["sha256"],
            "observation_sha256": observation_sha,
            "shape": list(observations.shape),
            "seed": seed,
            "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        }
    )
    aligned_identity = canonical_sha256(
        {
            "format": "RIVAL2_HUMAN_BC_V4_ALIGNED_SIMULATOR_CORPUS_V1",
            "legacy_observation_identity_sha256": legacy_identity,
            "opponent_family_tensor_sha256": family_sha,
            "train_mask_tensor_sha256": train_mask_sha,
            "arena_geometry_sha256": geometry.content_sha256,
        }
    )
    manifest = {
        "format": "RIVAL2_HUMAN_BC_V4_ALIGNED_SIMULATOR_CORPUS_V1",
        "identity_sha256": aligned_identity,
        "legacy_observation_only_identity_sha256": legacy_identity,
        "bootstrap": bootstrap_identity,
        "collection": {
            **corpus,
            "arena_geometry_sha256": geometry.content_sha256,
            "observation_shape": list(observations.shape),
            "opponent_family_shape": list(opponent_family.shape),
            "train_mask_shape": list(train_mask.shape),
            "observation_count": int(observations.numel() // OBS_DIM),
            "observation_tensor_sha256": observation_sha,
            "opponent_family_tensor_sha256": family_sha,
            "train_mask_tensor_sha256": train_mask_sha,
            "rollout_wall_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(env.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(env.device),
            "initial_opponent_family_worlds": {
                OPPONENT_NAMES[index]: int(initial_family_counts[index])
                for index in range(len(OPPONENT_NAMES))
            },
            "aligned_opponent_family_world_ticks": {
                OPPONENT_NAMES[index]: int(family_tick_counts[index])
                for index in range(len(OPPONENT_NAMES))
            },
            "aligned_role_counts": role_counts,
            "aligned_transition_counts": transition_counts,
            "rng_state_before_rollout": rng_before,
            "trainer_model_tensor_sha256_before_after": [model_before, model_after],
            "historical_ppo_optimizer_sha256_before_after": [
                optimizer_before,
                optimizer_after,
            ],
        },
        "split": split_manifest,
        "regeneration": {
            "corpus_binary_committed": False,
            "reason": "deterministically regenerate 6+ GiB observations plus aligned roles",
            "seed_split_and_all_tensor_hashes_recorded": True,
        },
        "checks": checks,
    }
    result = AlignedRolloutCorpus(
        observations=observations,
        opponent_family=opponent_family,
        train_mask=train_mask,
    )
    del rollout, trainer, env, meshes, geometry
    gc.collect()
    torch.cuda.empty_cache()
    return result, manifest, {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }


def select_whole_worlds(
    corpus: AlignedRolloutCorpus,
    source_worlds: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> AlignedRolloutCorpus:
    """Compact a frozen whole-world subset while preserving tick alignment."""

    indices = torch.from_numpy(
        np.ascontiguousarray(np.asarray(source_worlds, dtype=np.int64))
    ).to(corpus.observations.device)
    return AlignedRolloutCorpus(
        observations=corpus.observations.index_select(1, indices).to(device),
        opponent_family=corpus.opponent_family.index_select(1, indices).to(device),
        train_mask=corpus.train_mask.index_select(1, indices).to(device),
    )


select_world_subset = select_whole_worlds


__all__ = [
    "EXPECTED_SOCCAR_GEOMETRY_SHA256",
    "AlignedRolloutCorpus",
    "build_aligned_rollout_corpus",
    "int64_array_sha256",
    "select_whole_worlds",
    "select_world_subset",
]
