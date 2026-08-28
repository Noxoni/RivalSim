"""Run frozen missing-feature invariance distillation for Rival's 120 Hz bootstrap."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.human_demo.bc_observation_bridge import (  # noqa: E402
    BC_BRIDGE_VERSION,
    FIELD_QUALITY_CONTRACT_SHA256,
    BCBridgeTrajectoryAdapter,
    DegradationProfile,
    hybrid_actor_channel_kl,
)
from rivalsim.human_demo.missing_feature_distillation import (  # noqa: E402
    DISTILLATION_VERSION,
    DISTILLED_CHECKPOINT_FORMAT,
    MetricAccumulator,
    actor_output_statistics,
    build_whole_world_split,
    canonical_sha256,
    degrade_observations_torch,
    file_sha256,
    torch_profile_quality,
    world_observation_batch,
)
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_contracts import OBS_DIM  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    OPPONENT_NAMES,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig  # noqa: E402
from rivalsim.rival2_ppo import Rival2PPOConfig  # noqa: E402
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402

FROZEN_CONFIG = Path("results/rival2/missing_feature_distillation_v1/frozen_config.json")
FROZEN_CONFIG_SHA256 = "5F20CE9FDE854A99405D53864FB1FB72F9B28FA4EC882F8D4C675DF627A16955"
RESULT_ROOT = Path("results/rival2/missing_feature_distillation_v1")
CHECKPOINT = Path(
    "checkpoints/rival2/120hz_distilled/rival2_120hz_missing_feature_distilled.pt"
)
BRIDGE_MANIFEST = Path("results/rival2/human_demo_bc_bridge_v1/bridge_manifest.json")
WORK_ROOT = Path(".tools/rival2_missing_feature_distillation_v1")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    path = ROOT / FROZEN_CONFIG
    digest = file_sha256(path)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(f"frozen distillation config changed: {digest}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required_parent = config["authority"]["required_parent"]
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", required_parent, "HEAD"],
        check=True,
    )
    if config["authority"]["bridge_version"] != BC_BRIDGE_VERSION:
        raise ValueError("active BC bridge version differs from frozen authority")
    if (
        config["authority"]["bridge_quality_contract_sha256"]
        != FIELD_QUALITY_CONTRACT_SHA256
    ):
        raise ValueError("active BC bridge quality contract differs from frozen authority")
    return config, {
        "path": FROZEN_CONFIG.as_posix(),
        "sha256": digest,
        "git_blob_oid": _git("hash-object", str(FROZEN_CONFIG)),
        "required_parent": required_parent,
        "required_parent_is_ancestor": True,
    }


def _load_bootstrap(
    config: dict[str, Any],
) -> tuple[dict[str, Any], Rival2PolicyConfig, dict[str, Any]]:
    authority = config["authority"]
    path = ROOT / authority["bootstrap_checkpoint"]
    digest = file_sha256(path)
    if digest != authority["bootstrap_checkpoint_sha256"]:
        raise ValueError(f"bootstrap checkpoint SHA-256 mismatch: {digest}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = Rival2PolicyConfig(**payload["policy_config"])
    model_hash = tensor_tree_sha256(payload["model"])
    if model_hash != authority["bootstrap_model_tensor_sha256"]:
        raise ValueError(f"bootstrap model tensor SHA-256 mismatch: {model_hash}")
    checks = {
        "checkpoint_sha256_exact": True,
        "model_tensor_sha256_exact": True,
        "iteration_479": int(payload["iteration"]) == 479,
        "policy_version_479": int(payload["policy_version"]) == 479,
        "observation_v2_120hz": payload["observation_version"]
        == "RIVAL2_OBS_V2_120HZ",
        "action_v2_120hz": payload["action_version"] == "RIVAL2_ACTION_V2_120HZ",
        "policy_hz_120": int(payload["policy_hz"]) == 120,
        "physics_hz_120": int(payload["physics_hz"]) == 120,
    }
    if not all(checks.values()):
        raise ValueError(f"bootstrap identity failed: {checks}")
    return payload, policy, {
        "path": authority["bootstrap_checkpoint"],
        "bytes": path.stat().st_size,
        "sha256": digest,
        "model_tensor_sha256": model_hash,
        "iteration": int(payload["iteration"]),
        "policy_version": int(payload["policy_version"]),
        "historical_ppo_optimizer_tensor_sha256": tensor_tree_sha256(
            payload["optimizer"]
        ),
        "checks": checks,
    }


def _build_rollout_corpus(
    config: dict[str, Any],
    bootstrap_payload: dict[str, Any],
    bootstrap_identity: dict[str, Any],
    *,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    corpus = config["corpus"]
    worlds = int(corpus["worlds"])
    horizon = int(corpus["decisions_per_world"])
    seed = int(corpus["seed"])
    collision_dir = Path(corpus["collision_mesh_directory"])
    if not collision_dir.is_dir():
        raise FileNotFoundError(f"collision mesh directory not found: {collision_dir}")
    if worlds != 32_768 or horizon != 128:
        raise ValueError("frozen corpus geometry is not the authorized 32768 x 128")

    geometry = ArenaGeometry.load_soccar(collision_dir)
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
    family_counts = torch.bincount(trainer.opponent_family, minlength=4).cpu().tolist()
    torch.cuda.synchronize(env.device)
    torch.cuda.reset_peak_memory_stats(env.device)
    started = time.perf_counter()
    rollout = trainer.collect_rollout()
    torch.cuda.synchronize(env.device)
    elapsed = time.perf_counter() - started
    if list(rollout.observations.shape) != [horizon, worlds, 2, OBS_DIM]:
        raise RuntimeError("distillation rollout observation shape mismatch")
    observations = rollout.observations.detach()
    finite = bool(torch.isfinite(observations).all())

    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(json.dumps(list(observations.shape), separators=(",", ":")).encode())
    for tick in range(horizon):
        block = observations[tick : tick + 1].detach().cpu().contiguous().numpy()
        digest.update(block.tobytes(order="C"))
    observation_sha = digest.hexdigest().upper()
    model_after = tensor_tree_sha256(trainer.model.state_dict())
    optimizer_after = tensor_tree_sha256(trainer.optimizer.state_dict())
    checks = {
        "shape_exact": list(observations.shape) == [128, 32768, 2, 182],
        "finite_observations": finite,
        "model_unchanged": model_before == model_after == bootstrap_identity["model_tensor_sha256"],
        "historical_ppo_optimizer_unchanged": optimizer_before == optimizer_after,
        "iteration_unchanged": trainer.iteration == iteration_before == 479,
        "policy_version_unchanged": trainer.policy_version == policy_version_before == 479,
        "bootstrap_file_unchanged": file_sha256(bootstrap_path)
        == bootstrap_identity["sha256"],
        "no_optimizer_step_during_collection": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"authoritative corpus collection failed: {checks}")
    split_config = corpus["split"]
    split = build_whole_world_split(
        worlds=worlds,
        train_worlds=int(split_config["train_worlds"]),
        validation_worlds=int(split_config["validation_worlds"]),
        test_worlds=int(split_config["test_worlds"]),
        seed=int(split_config["seed"]),
    )
    split_manifest = split.manifest
    corpus_identity = canonical_sha256(
        {
            "bootstrap_sha256": bootstrap_identity["sha256"],
            "observation_sha256": observation_sha,
            "shape": list(observations.shape),
            "seed": seed,
            "split_manifest_sha256": split_manifest["split_manifest_sha256"],
        }
    )
    manifest = {
        "format": "RIVAL2_MISSING_FEATURE_DISTILLATION_CORPUS_V1",
        "identity_sha256": corpus_identity,
        "bootstrap": bootstrap_identity,
        "collection": {
            **corpus,
            "observation_shape": list(observations.shape),
            "observation_count": int(observations.numel() // OBS_DIM),
            "observation_tensor_sha256": observation_sha,
            "rollout_wall_seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(env.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(env.device),
            "opponent_family_worlds": {
                OPPONENT_NAMES[index]: int(family_counts[index])
                for index in range(len(OPPONENT_NAMES))
            },
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
            "reason": "deterministically regenerated authoritative 6+ GiB observation tensor",
            "required_seed_and_hashes_recorded": True,
        },
        "checks": checks,
    }
    del rollout, trainer, env, meshes, geometry
    gc.collect()
    torch.cuda.empty_cache()
    return observations, manifest, {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }


class _ActorOutputAccumulator:
    def __init__(self) -> None:
        self.rows: list[torch.Tensor] = []

    def update(self, actor: torch.Tensor) -> None:
        self.rows.append(actor.detach().cpu())

    def result(self) -> dict[str, Any]:
        return actor_output_statistics(torch.cat(self.rows, dim=0))


@torch.no_grad()
def _evaluate_worlds(
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    observations: torch.Tensor,
    worlds: np.ndarray,
    *,
    worlds_per_batch: int,
    gameplay_quality: torch.Tensor,
    freeplay_quality: torch.Tensor,
    policy_config: Rival2PolicyConfig,
) -> dict[str, Any]:
    metrics = {
        "full": MetricAccumulator.create(),
        "gameplay_degraded": MetricAccumulator.create(),
        "freeplay_degraded": MetricAccumulator.create(),
    }
    outputs = {name: _ActorOutputAccumulator() for name in metrics}
    for start in range(0, len(worlds), worlds_per_batch):
        full = world_observation_batch(
            observations,
            worlds[start : start + worlds_per_batch],
        )
        gameplay = degrade_observations_torch(full, gameplay_quality)
        freeplay = degrade_observations_torch(full, freeplay_quality)
        teacher_actor, teacher_value = teacher(full)
        combined_actor, combined_value = student(torch.cat((full, gameplay, freeplay)))
        count = full.shape[0]
        student_rows = {
            "full": (combined_actor[:count], combined_value[:count]),
            "gameplay_degraded": (
                combined_actor[count : 2 * count],
                combined_value[count : 2 * count],
            ),
            "freeplay_degraded": (
                combined_actor[2 * count :],
                combined_value[2 * count :],
            ),
        }
        for name, (student_actor, student_value) in student_rows.items():
            channel = hybrid_actor_channel_kl(
                teacher_actor,
                student_actor,
                policy_config=policy_config,
            )
            metrics[name].update(channel, teacher_value, student_value)
            outputs[name].update(student_actor)
    return {
        name: {
            **metrics[name].result(),
            "student_actor_output": outputs[name].result(),
        }
        for name in metrics
    }


@torch.no_grad()
def _retention_guard(
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    observations: torch.Tensor,
    policy_config: Rival2PolicyConfig,
    guard: dict[str, Any],
) -> dict[str, Any]:
    teacher_actor, teacher_value = teacher(observations)
    student_actor, student_value = student(observations)
    channel = hybrid_actor_channel_kl(
        teacher_actor,
        student_actor,
        policy_config=policy_config,
    )
    accumulator = MetricAccumulator.create()
    accumulator.update(channel, teacher_value, student_value)
    result = accumulator.result()
    checks = {
        "actor_mean_kl": result["actor_mean_kl"] <= guard["full_actor_mean_kl"],
        "actor_max_sample_kl": result["actor_max_sample_kl"]
        <= guard["full_actor_max_sample_kl"],
        "actor_max_channel_kl": max(result["actor_channel_kl"].values())
        <= guard["full_actor_max_channel_kl"],
        "value_rmse": result["value_rmse"] <= guard["full_value_rmse"],
        "value_max_absolute_drift": result["value_max_absolute_drift"]
        <= guard["full_value_max_absolute_drift"],
        "finite": result["actor_finite"] and result["value_finite"],
        "parameters_finite": all(
            bool(torch.isfinite(parameter).all()) for parameter in student.parameters()
        ),
    }
    return {**result, "checks": checks, "accepted": all(checks.values())}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _save_interval_state(
    path: Path,
    student: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student": _cpu_tree(student.state_dict()),
            "optimizer": _cpu_tree(optimizer.state_dict()),
        },
        path,
    )


def _load_interval_state(
    path: Path,
    student: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    student.load_state_dict(state["student"])
    optimizer.load_state_dict(state["optimizer"])


def _checkpoint_payload(
    *,
    config: dict[str, Any],
    config_identity: dict[str, Any],
    corpus_manifest: dict[str, Any],
    bootstrap_identity: dict[str, Any],
    policy_config: Rival2PolicyConfig,
    student: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    accepted_steps: int,
    proposed_steps: int,
    validation: dict[str, Any],
    training_state: dict[str, Any],
    final_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": DISTILLED_CHECKPOINT_FORMAT,
        "distillation_version": DISTILLATION_VERSION,
        "model": _cpu_tree(student.state_dict()),
        "supervised_optimizer": _cpu_tree(optimizer.state_dict()),
        "policy_config": asdict(policy_config),
        "teacher": bootstrap_identity,
        "bootstrap_parent": bootstrap_identity["sha256"],
        "bridge": {
            "version": BC_BRIDGE_VERSION,
            "field_quality_contract_sha256": FIELD_QUALITY_CONTRACT_SHA256,
        },
        "frozen_config": config_identity,
        "distillation_corpus": {
            "identity_sha256": corpus_manifest["identity_sha256"],
            "observation_tensor_sha256": corpus_manifest["collection"][
                "observation_tensor_sha256"
            ],
            "split_manifest_sha256": corpus_manifest["split"][
                "split_manifest_sha256"
            ],
        },
        "student_identity": {
            "model_tensor_sha256": tensor_tree_sha256(student.state_dict()),
            "architecture_unchanged": True,
            "initialized_byte_identically_from_teacher": True,
        },
        "supervised_training": {
            "accepted_optimizer_steps": accepted_steps,
            "proposed_optimizer_steps": proposed_steps,
            "optimizer_type": config["training"]["optimizer"],
            "training_state": training_state,
        },
        "validation_at_checkpoint": validation,
        "final_evaluation": final_evaluation,
        "ppo_resume_semantics": {
            "historical_iteration_479_optimizer_mutated": False,
            "historical_iteration_479_optimizer_included": False,
            "historical_iteration_479_optimizer_valid_for_this_student": False,
            "future_ppo_requires_fresh_optimizer_transition": True,
            "not_a_ppo_resumable_iteration_479_checkpoint": True,
        },
        "human_behavior_cloning_performed": False,
        "human_optimizer_steps": 0,
        "ppo_steps": 0,
        "reward_changed": False,
        "mechanic_detector_changed": False,
    }


def _train(
    config: dict[str, Any],
    config_identity: dict[str, Any],
    corpus_manifest: dict[str, Any],
    split: dict[str, np.ndarray],
    observations: torch.Tensor,
    bootstrap_payload: dict[str, Any],
    bootstrap_identity: dict[str, Any],
    policy_config: Rival2PolicyConfig,
    *,
    device: torch.device,
) -> tuple[Rival2ActorCritic, torch.optim.Optimizer, dict[str, Any]]:
    training = config["training"]
    guard_config = config["retention_guard"]
    teacher = Rival2ActorCritic(policy_config).to(device)
    teacher.load_state_dict(bootstrap_payload["model"])
    teacher.eval()
    teacher.requires_grad_(False)
    student = Rival2ActorCritic(policy_config).to(device)
    student.load_state_dict(bootstrap_payload["model"])
    student.train()
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=float(training["initial_learning_rate"]),
        betas=tuple(float(value) for value in training["optimizer_betas"]),
        eps=float(training["optimizer_epsilon"]),
    )
    if tensor_tree_sha256(student.state_dict()) != bootstrap_identity["model_tensor_sha256"]:
        raise RuntimeError("student initialization is not byte-identical to teacher")
    teacher_hash_before = tensor_tree_sha256(teacher.state_dict())
    bootstrap_hash_before = file_sha256(ROOT / bootstrap_identity["path"])
    ppo_optimizer_hash_before = tensor_tree_sha256(bootstrap_payload["optimizer"])

    retention_path = ROOT / guard_config["source"]
    if file_sha256(retention_path) != guard_config["source_sha256"]:
        raise ValueError("fixed full-observation retention corpus SHA-256 changed")
    retention_payload = torch.load(retention_path, map_location="cpu", weights_only=False)
    retention = retention_payload["observations"].to(device=device, dtype=torch.float32)
    gameplay_quality = torch_profile_quality(DegradationProfile.GAMEPLAY, device=device)
    freeplay_quality = torch_profile_quality(DegradationProfile.FREEPLAY, device=device)
    worlds_per_batch = int(training["worlds_per_minibatch"])

    teacher.eval()
    student.eval()
    baseline_validation = _evaluate_worlds(
        teacher,
        student,
        observations,
        split["validation"],
        worlds_per_batch=worlds_per_batch,
        gameplay_quality=gameplay_quality,
        freeplay_quality=freeplay_quality,
        policy_config=policy_config,
    )
    baseline_test = _evaluate_worlds(
        teacher,
        student,
        observations,
        split["test"],
        worlds_per_batch=worlds_per_batch,
        gameplay_quality=gameplay_quality,
        freeplay_quality=freeplay_quality,
        policy_config=policy_config,
    )
    initial_guard = _retention_guard(
        teacher,
        student,
        retention,
        policy_config,
        guard_config,
    )
    if not initial_guard["accepted"]:
        raise RuntimeError("byte-identical student failed initial full-observation guard")

    chunks: list[tuple[int, np.ndarray]] = []
    for epoch in range(int(training["max_epochs"])):
        order = np.random.default_rng(int(training["seed"]) + epoch).permutation(
            split["train"]
        )
        for start in range(0, len(order), worlds_per_batch):
            chunks.append((epoch, np.ascontiguousarray(order[start : start + worlds_per_batch])))
    max_steps = min(int(training["max_accepted_optimizer_steps"]), len(chunks))
    chunks = chunks[:max_steps]
    validation_interval = int(training["validation_interval_optimizer_steps"])
    curve: list[dict[str, Any]] = []
    guard_attempts: list[dict[str, Any]] = []
    accepted_steps = 0
    proposed_steps = 0
    cursor = 0
    best_monitor = (
        baseline_validation["gameplay_degraded"]["actor_mean_kl"]
        + baseline_validation["freeplay_degraded"]["actor_mean_kl"]
    ) / 2.0
    best_step = 0
    best_validation = baseline_validation
    no_material_improvement = 0
    current_lr = float(training["initial_learning_rate"])
    stop_reason = "maximum_frozen_training_budget_reached"
    interval_state = ROOT / WORK_ROOT / "interval_start.pt"
    checkpoint_path = ROOT / CHECKPOINT
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    while cursor < len(chunks):
        interval = chunks[cursor : cursor + validation_interval]
        _save_interval_state(interval_state, student, optimizer)
        retries = 0
        accepted_interval = False
        while not accepted_interval:
            student.train()
            interval_loss_sum = 0.0
            interval_grad_max = 0.0
            nonfinite_training = False
            for _epoch, world_batch in interval:
                full = world_observation_batch(observations, world_batch)
                gameplay = degrade_observations_torch(full, gameplay_quality)
                freeplay = degrade_observations_torch(full, freeplay_quality)
                with torch.no_grad():
                    teacher_actor, teacher_value = teacher(full)
                combined_actor, combined_value = student(
                    torch.cat((gameplay, freeplay, full), dim=0)
                )
                count = full.shape[0]
                gameplay_actor = combined_actor[:count]
                freeplay_actor = combined_actor[count : 2 * count]
                full_actor = combined_actor[2 * count :]
                full_value = combined_value[2 * count :]
                gameplay_kl = hybrid_actor_channel_kl(
                    teacher_actor,
                    gameplay_actor,
                    policy_config=policy_config,
                ).sum(dim=-1).mean()
                freeplay_kl = hybrid_actor_channel_kl(
                    teacher_actor,
                    freeplay_actor,
                    policy_config=policy_config,
                ).sum(dim=-1).mean()
                full_kl = hybrid_actor_channel_kl(
                    teacher_actor,
                    full_actor,
                    policy_config=policy_config,
                ).sum(dim=-1).mean()
                critic_loss = (full_value - teacher_value).square().mean()
                loss = (
                    float(training["gameplay_degraded_actor_kl_weight"])
                    * gameplay_kl
                    + float(training["freeplay_degraded_actor_kl_weight"])
                    * freeplay_kl
                    + float(training["full_actor_retention_weight"]) * full_kl
                    + float(training["critic_full_retention_weight"]) * critic_loss
                )
                optimizer.zero_grad(set_to_none=True)
                if not bool(torch.isfinite(loss)):
                    nonfinite_training = True
                    break
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    float(training["gradient_clip_norm"]),
                    error_if_nonfinite=False,
                )
                if not bool(torch.isfinite(gradient_norm)):
                    nonfinite_training = True
                    break
                optimizer.step()
                proposed_steps += 1
                interval_loss_sum += float(loss.detach().item())
                interval_grad_max = max(interval_grad_max, float(gradient_norm.item()))

            student.eval()
            guard_result = _retention_guard(
                teacher,
                student,
                retention,
                policy_config,
                guard_config,
            )
            if nonfinite_training:
                guard_result["checks"]["finite_training"] = False
                guard_result["accepted"] = False
            guard_attempts.append(
                {
                    "interval_start_cursor": cursor,
                    "retry": retries,
                    "learning_rate": current_lr,
                    "proposed_optimizer_steps": proposed_steps,
                    "interval_steps_completed": (
                        0 if nonfinite_training else len(interval)
                    ),
                    "retention_guard": guard_result,
                }
            )
            _write_json(
                ROOT / RESULT_ROOT / "guard_attempts_live.json",
                {
                    "format": "RIVAL2_MISSING_FEATURE_DISTILLATION_GUARD_ATTEMPTS_V1",
                    "attempts": guard_attempts,
                },
            )
            if not guard_result["accepted"]:
                _load_interval_state(interval_state, student, optimizer)
                retries += 1
                current_lr *= float(training["lr_backoff_factor"])
                if (
                    retries > int(training["max_guard_retries_per_interval"])
                    or current_lr < float(training["minimum_learning_rate"])
                ):
                    stop_reason = "full_observation_retention_guard_limiting"
                    cursor = len(chunks)
                    break
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                continue

            accepted_interval = True
            accepted_steps += len(interval)
            cursor += len(interval)
            validation = _evaluate_worlds(
                teacher,
                student,
                observations,
                split["validation"],
                worlds_per_batch=worlds_per_batch,
                gameplay_quality=gameplay_quality,
                freeplay_quality=freeplay_quality,
                policy_config=policy_config,
            )
            monitor = (
                validation["gameplay_degraded"]["actor_mean_kl"]
                + validation["freeplay_degraded"]["actor_mean_kl"]
            ) / 2.0
            relative_improvement = (best_monitor - monitor) / max(best_monitor, 1.0e-12)
            material = relative_improvement >= float(
                training["early_stopping_material_relative_improvement"]
            )
            curve_row = {
                "accepted_optimizer_steps": accepted_steps,
                "proposed_optimizer_steps": proposed_steps,
                "epoch": interval[-1][0],
                "learning_rate": current_lr,
                "interval_steps": len(interval),
                "interval_mean_training_loss": interval_loss_sum / len(interval),
                "interval_max_preclip_gradient_norm": interval_grad_max,
                "guard_retries": retries,
                "retention_guard": guard_result,
                "validation": validation,
                "combined_degraded_validation_kl": monitor,
                "relative_improvement_from_previous_best": relative_improvement,
                "material_improvement": material,
            }
            if material:
                best_monitor = monitor
                best_step = accepted_steps
                best_validation = validation
                no_material_improvement = 0
                payload = _checkpoint_payload(
                    config=config,
                    config_identity=config_identity,
                    corpus_manifest=corpus_manifest,
                    bootstrap_identity=bootstrap_identity,
                    policy_config=policy_config,
                    student=student,
                    optimizer=optimizer,
                    accepted_steps=accepted_steps,
                    proposed_steps=proposed_steps,
                    validation=validation,
                    training_state={
                        "best_step": best_step,
                        "current_learning_rate": current_lr,
                        "early_stop_counter": no_material_improvement,
                        "resumable": True,
                    },
                )
                torch.save(payload, checkpoint_path)
                curve_row["checkpoint"] = {
                    "path": CHECKPOINT.as_posix(),
                    "sha256": file_sha256(checkpoint_path),
                    "student_model_tensor_sha256": tensor_tree_sha256(
                        student.state_dict()
                    ),
                }
            else:
                no_material_improvement += 1
                if (
                    no_material_improvement
                    % int(training["lr_plateau_patience_validations"])
                    == 0
                    and current_lr > float(training["minimum_learning_rate"])
                ):
                    current_lr = max(
                        float(training["minimum_learning_rate"]),
                        current_lr * float(training["lr_backoff_factor"]),
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = current_lr
                    curve_row["plateau_lr_backoff"] = current_lr
            curve.append(curve_row)
            if no_material_improvement >= int(
                training["early_stopping_patience_validations"]
            ):
                stop_reason = "degraded_validation_kl_plateau"
                cursor = len(chunks)
                break

    if best_step <= 0 or not checkpoint_path.is_file():
        _write_json(
            ROOT / RESULT_ROOT / "failed_training_evidence.json",
            {
                "format": "RIVAL2_MISSING_FEATURE_DISTILLATION_FAILED_TRAINING_V1",
                "frozen_config_sha256": config_identity["sha256"],
                "best_step": best_step,
                "accepted_optimizer_steps": accepted_steps,
                "proposed_optimizer_steps": proposed_steps,
                "stop_reason": stop_reason,
                "guard_attempts": guard_attempts,
                "bootstrap_unchanged": file_sha256(ROOT / bootstrap_identity["path"])
                == bootstrap_hash_before,
                "teacher_unchanged": tensor_tree_sha256(teacher.state_dict())
                == teacher_hash_before,
                "historical_ppo_optimizer_unchanged": tensor_tree_sha256(
                    bootstrap_payload["optimizer"]
                )
                == ppo_optimizer_hash_before,
            },
        )
        raise RuntimeError("distillation produced no material guard-accepted checkpoint")
    best_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    student.load_state_dict(best_payload["model"])
    optimizer.load_state_dict(best_payload["supervised_optimizer"])
    student.eval()
    final_guard = _retention_guard(
        teacher,
        student,
        retention,
        policy_config,
        guard_config,
    )
    if not final_guard["accepted"]:
        raise RuntimeError("restored best checkpoint fails frozen retention guard")
    integrity = {
        "teacher_model_tensor_sha256_before_after": [
            teacher_hash_before,
            tensor_tree_sha256(teacher.state_dict()),
        ],
        "bootstrap_checkpoint_sha256_before_after": [
            bootstrap_hash_before,
            file_sha256(ROOT / bootstrap_identity["path"]),
        ],
        "historical_ppo_optimizer_tensor_sha256_before_after": [
            ppo_optimizer_hash_before,
            tensor_tree_sha256(bootstrap_payload["optimizer"]),
        ],
        "teacher_gradients_absent": all(
            parameter.grad is None for parameter in teacher.parameters()
        ),
        "student_parameters_finite": all(
            bool(torch.isfinite(parameter).all()) for parameter in student.parameters()
        ),
    }
    integrity["valid"] = (
        len(set(integrity["teacher_model_tensor_sha256_before_after"])) == 1
        and len(set(integrity["bootstrap_checkpoint_sha256_before_after"])) == 1
        and len(set(integrity["historical_ppo_optimizer_tensor_sha256_before_after"]))
        == 1
        and integrity["teacher_gradients_absent"]
        and integrity["student_parameters_finite"]
    )
    if not integrity["valid"]:
        raise RuntimeError(f"immutable teacher/bootstrap integrity failed: {integrity}")
    return student, optimizer, {
        "format": "RIVAL2_MISSING_FEATURE_DISTILLATION_TRAINING_CURVE_V1",
        "baseline_validation": baseline_validation,
        "baseline_test": baseline_test,
        "initial_retention_guard": initial_guard,
        "curve": curve,
        "guard_attempts": guard_attempts,
        "best_step": best_step,
        "best_validation": best_validation,
        "best_combined_degraded_validation_kl": best_monitor,
        "accepted_optimizer_steps_executed_before_best_restore": accepted_steps,
        "proposed_optimizer_steps": proposed_steps,
        "stop_reason": stop_reason,
        "final_retention_guard": final_guard,
        "integrity": integrity,
    }


def _verify_human_sources(source_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((ROOT / BRIDGE_MANIFEST).read_text(encoding="utf-8"))
    verified = []
    for session in manifest["human_source_verification"]:
        session_dir = source_root / session["session_uuid"]
        rows = []
        digest = hashlib.sha256()
        for expected in session["files"]:
            path = session_dir / expected["path"]
            actual = file_sha256(path)
            if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"human source changed: {path}")
            digest.update(f"{expected['path']}:{actual}\n".encode())
            rows.append({"path": expected["path"], "sha256": actual})
        file_set = digest.hexdigest().upper()
        if file_set != session["source_file_set_sha256"]:
            raise ValueError(f"human source file set changed: {session['session_uuid']}")
        verified.append(
            {
                "session_uuid": session["session_uuid"],
                "source_file_set_sha256": file_set,
                "file_count": len(rows),
                "unchanged": True,
            }
        )
    return verified


@torch.no_grad()
def _evaluate_human_corpus(
    student: Rival2ActorCritic,
    config: dict[str, Any],
    *,
    source_root: Path,
) -> dict[str, Any]:
    verification_before = _verify_human_sources(source_root)
    dataset = json.loads(
        (ROOT / config["human_post_distillation_evaluation"]["source_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    spans: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    kinds: dict[str, str] = {}
    for row in dataset["mechanic_positive_attempts"]:
        identity = str(row["attempt_id"])
        kinds[identity] = "mechanic"
        spans[str(row["session_uuid"])].append(
            (identity, int(row["start_sequence"]), int(row["end_sequence"]))
        )
    gameplay = dataset["general_gameplay"]
    for row in gameplay["regions"]:
        identity = str(row["region_id"])
        kinds[identity] = "gameplay"
        spans[str(gameplay["session_uuid"])].append(
            (identity, int(row["start_sequence"]), int(row["end_sequence"]))
        )
    batch_size = int(config["human_post_distillation_evaluation"]["inference_batch_size"])
    device = next(student.parameters()).device
    buffers: dict[str, list[np.ndarray]] = {"gameplay": [], "mechanic": []}
    outputs: dict[str, list[torch.Tensor]] = {"gameplay": [], "mechanic": []}
    counts = {"gameplay": 0, "mechanic": 0}

    def flush(kind: str) -> None:
        if not buffers[kind]:
            return
        value = torch.from_numpy(np.stack(buffers[kind])).to(device=device)
        actor, _critic = student(value)
        outputs[kind].append(actor.detach().cpu())
        counts[kind] += len(buffers[kind])
        buffers[kind].clear()

    for session_uuid in sorted(spans):
        adapter = BCBridgeTrajectoryAdapter(source_root / session_uuid)
        for identity, sample in adapter.iter_spans(spans[session_uuid]):
            if not sample.bc_usable:
                raise RuntimeError(f"accepted human frame became unusable: {identity}")
            kind = kinds[identity]
            buffers[kind].append(np.asarray(sample.observation).copy())
            if len(buffers[kind]) >= batch_size:
                flush(kind)
    flush("gameplay")
    flush("mechanic")
    expected = int(config["human_post_distillation_evaluation"]["expected_bc_usable_frames"])
    total = sum(counts.values())
    verification_after = _verify_human_sources(source_root)
    result = {
        "format": "RIVAL2_MISSING_FEATURE_DISTILLATION_HUMAN_PRE_BC_BASELINE_V1",
        "frame_count": total,
        "expected_frame_count": expected,
        "cohorts": {
            kind: actor_output_statistics(torch.cat(outputs[kind], dim=0))
            for kind in ("gameplay", "mechanic")
        },
        "source_verification_before": verification_before,
        "source_verification_after": verification_after,
        "source_hashes_unchanged": verification_before == verification_after,
        "all_outputs_finite": all(
            bool(torch.isfinite(torch.cat(outputs[kind], dim=0)).all())
            for kind in outputs
        ),
        "human_actions_used_as_targets": False,
        "human_optimizer_steps": 0,
        "behavior_cloning_performed": False,
    }
    result["valid"] = (
        total == expected
        and result["source_hashes_unchanged"]
        and result["all_outputs_finite"]
        and not result["human_actions_used_as_targets"]
    )
    return result


def _profile_acceptance(
    before: dict[str, Any],
    after: dict[str, Any],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    gameplay_before = before["gameplay_degraded"]["actor_mean_kl"]
    gameplay_after = after["gameplay_degraded"]["actor_mean_kl"]
    freeplay_before = before["freeplay_degraded"]["actor_mean_kl"]
    freeplay_after = after["freeplay_degraded"]["actor_mean_kl"]
    gameplay_reduction = (gameplay_before - gameplay_after) / gameplay_before
    freeplay_reduction = (freeplay_before - freeplay_after) / freeplay_before
    output_profiles = [after[name]["student_actor_output"] for name in after]
    max_analog = max(
        abs(channel[key])
        for output in output_profiles
        for channel in output["analog_mean"].values()
        for key in ("min", "max")
    )
    max_log_std_clamp = max(
        max(channel["at_min_fraction"], channel["at_max_fraction"])
        for output in output_profiles
        for channel in output["log_std"].values()
    )
    max_button_saturation = max(
        channel["saturation_fraction"]
        for output in output_profiles
        for channel in output["button_probability"].values()
    )
    checks = {
        "gameplay_relative_reduction": gameplay_reduction
        >= acceptance["gameplay_degraded_minimum_relative_kl_reduction"],
        "gameplay_absolute_kl": gameplay_after
        <= acceptance["gameplay_degraded_mean_kl_max"],
        "freeplay_relative_reduction": freeplay_reduction
        >= acceptance["freeplay_degraded_minimum_relative_kl_reduction"],
        "full_actor_retention": after["full"]["actor_mean_kl"]
        <= acceptance["full_observation_actor_mean_kl_max"],
        "full_critic_retention": after["full"]["value_rmse"]
        <= acceptance["full_observation_value_rmse_max"],
        "analog_not_saturated": max_analog <= acceptance["maximum_analog_mean_absolute"],
        "log_std_not_collapsed": max_log_std_clamp
        <= acceptance["maximum_log_std_clamp_fraction"],
        "buttons_not_collapsed": max_button_saturation
        <= acceptance["maximum_button_probability_saturation_fraction"],
        "all_finite": all(
            after[name]["actor_finite"]
            and after[name]["value_finite"]
            and after[name]["student_actor_output"]["finite"]
            for name in after
        ),
    }
    return {
        "gameplay_degraded_kl_before": gameplay_before,
        "gameplay_degraded_kl_after": gameplay_after,
        "gameplay_relative_reduction": gameplay_reduction,
        "freeplay_degraded_kl_before": freeplay_before,
        "freeplay_degraded_kl_after": freeplay_after,
        "freeplay_relative_reduction": freeplay_reduction,
        "maximum_analog_mean_absolute": max_analog,
        "maximum_log_std_clamp_fraction": max_log_std_clamp,
        "maximum_button_saturation_fraction": max_button_saturation,
        "checks": checks,
        "accepted": all(checks.values()),
    }


def _artifact_manifest(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for relative in sorted(paths, key=lambda value: value.as_posix()):
        path = ROOT / relative
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "format": "RIVAL2_MISSING_FEATURE_DISTILLATION_ARTIFACT_MANIFEST_V1",
        "files": rows,
        "file_count": len(rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_identity = _load_config()
    bootstrap_payload, policy_config, bootstrap_identity = _load_bootstrap(config)
    bootstrap_hash_before = bootstrap_identity["sha256"]
    human_source_before = _verify_human_sources(args.human_source_root)
    observations, corpus_manifest, split = _build_rollout_corpus(
        config,
        bootstrap_payload,
        bootstrap_identity,
        device=args.device,
    )
    _write_json(ROOT / RESULT_ROOT / "corpus_manifest.json", corpus_manifest)
    student, optimizer, training = _train(
        config,
        config_identity,
        corpus_manifest,
        split,
        observations,
        bootstrap_payload,
        bootstrap_identity,
        policy_config,
        device=torch.device(args.device),
    )
    _write_json(ROOT / RESULT_ROOT / "training_curve.json", training)
    gameplay_quality = torch_profile_quality(
        DegradationProfile.GAMEPLAY, device=args.device
    )
    freeplay_quality = torch_profile_quality(
        DegradationProfile.FREEPLAY, device=args.device
    )
    teacher = Rival2ActorCritic(policy_config).to(args.device)
    teacher.load_state_dict(bootstrap_payload["model"])
    teacher.eval().requires_grad_(False)
    final_test = _evaluate_worlds(
        teacher,
        student.eval(),
        observations,
        split["test"],
        worlds_per_batch=int(config["training"]["worlds_per_minibatch"]),
        gameplay_quality=gameplay_quality,
        freeplay_quality=freeplay_quality,
        policy_config=policy_config,
    )
    acceptance = _profile_acceptance(
        training["baseline_test"],
        final_test,
        config["acceptance"],
    )
    human = _evaluate_human_corpus(
        student,
        config,
        source_root=args.human_source_root,
    )
    _write_json(ROOT / RESULT_ROOT / "test_evaluation.json", final_test)
    _write_json(ROOT / RESULT_ROOT / "acceptance.json", acceptance)
    _write_json(ROOT / RESULT_ROOT / "human_pre_bc_baseline.json", human)
    human_source_after = _verify_human_sources(args.human_source_root)
    final_payload = _checkpoint_payload(
        config=config,
        config_identity=config_identity,
        corpus_manifest=corpus_manifest,
        bootstrap_identity=bootstrap_identity,
        policy_config=policy_config,
        student=student,
        optimizer=optimizer,
        accepted_steps=int(training["best_step"]),
        proposed_steps=int(training["proposed_optimizer_steps"]),
        validation=training["best_validation"],
        training_state={
            "best_step": training["best_step"],
            "stop_reason": training["stop_reason"],
            "resumable": True,
        },
        final_evaluation={
            "test": final_test,
            "acceptance": acceptance,
            "human_pre_bc_baseline": {
                "frame_count": human["frame_count"],
                "valid": human["valid"],
            },
        },
    )
    checkpoint_path = ROOT / CHECKPOINT
    torch.save(final_payload, checkpoint_path)
    checkpoint_identity = {
        "path": CHECKPOINT.as_posix(),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": file_sha256(checkpoint_path),
        "student_model_tensor_sha256": tensor_tree_sha256(student.state_dict()),
        "accepted_supervised_optimizer_steps": int(training["best_step"]),
        "resumable_supervised_optimizer_present": True,
        "ppo_resumable": False,
    }
    checks = {
        "acceptance": acceptance["accepted"],
        "human_pre_bc_baseline_valid": human["valid"],
        "human_sources_unchanged": human_source_before
        == human_source_after
        == human["source_verification_before"]
        == human["source_verification_after"],
        "bootstrap_byte_identical": file_sha256(
            ROOT / config["authority"]["bootstrap_checkpoint"]
        )
        == bootstrap_hash_before,
        "teacher_model_byte_identical": tensor_tree_sha256(teacher.state_dict())
        == bootstrap_identity["model_tensor_sha256"],
        "historical_ppo_optimizer_untouched": tensor_tree_sha256(
            bootstrap_payload["optimizer"]
        )
        == bootstrap_identity["historical_ppo_optimizer_tensor_sha256"],
        "checkpoint_marks_ppo_optimizer_stale": final_payload["ppo_resume_semantics"][
            "historical_iteration_479_optimizer_valid_for_this_student"
        ]
        is False,
        "human_behavior_cloning_absent": not human["behavior_cloning_performed"],
        "human_optimizer_steps_zero": human["human_optimizer_steps"] == 0,
        "ppo_steps_zero": final_payload["ppo_steps"] == 0,
        "reward_unchanged": not final_payload["reward_changed"],
        "mechanic_detector_unchanged": not final_payload["mechanic_detector_changed"],
    }
    verdict = "PASS" if all(checks.values()) else "BLOCKED"
    summary = {
        "format": "RIVAL2_MISSING_FEATURE_DISTILLATION_EVIDENCE_V1",
        "created_utc": datetime.now(UTC).isoformat(),
        "distillation_version": DISTILLATION_VERSION,
        "frozen_config": config_identity,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "warp": wp.__version__,
            "cuda_device": torch.cuda.get_device_name(torch.device(args.device)),
        },
        "bootstrap": bootstrap_identity,
        "corpus_identity_sha256": corpus_manifest["identity_sha256"],
        "checkpoint": checkpoint_identity,
        "training": {
            "best_step": training["best_step"],
            "proposed_optimizer_steps": training["proposed_optimizer_steps"],
            "stop_reason": training["stop_reason"],
        },
        "acceptance": acceptance,
        "full_observation_retention": final_test["full"],
        "human_pre_bc_baseline": {
            "frame_count": human["frame_count"],
            "valid": human["valid"],
        },
        "checks": checks,
        "verdict": verdict,
    }
    _write_json(ROOT / RESULT_ROOT / "evidence.json", summary)
    artifacts = _artifact_manifest(
        [
            FROZEN_CONFIG,
            RESULT_ROOT / "acceptance.json",
            RESULT_ROOT / "corpus_manifest.json",
            RESULT_ROOT / "evidence.json",
            RESULT_ROOT / "human_pre_bc_baseline.json",
            RESULT_ROOT / "test_evaluation.json",
            RESULT_ROOT / "training_curve.json",
            CHECKPOINT,
        ]
    )
    _write_json(ROOT / RESULT_ROOT / "artifact_manifest.json", artifacts)
    if verdict != "PASS":
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"distillation acceptance failed: {failed}")
    return summary


def verify() -> dict[str, Any]:
    manifest = json.loads(
        (ROOT / RESULT_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    errors = []
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            errors.append(f"missing artifact: {row['path']}")
        elif path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            errors.append(f"artifact hash mismatch: {row['path']}")
    evidence = json.loads((ROOT / RESULT_ROOT / "evidence.json").read_text())
    if evidence["verdict"] == "PASS":
        if not all(evidence["checks"].values()):
            errors.append("PASS evidence contains a failed check")
    elif evidence["verdict"] == "BLOCKED":
        required_true = (
            "all_failed_intervals_rolled_back",
            "frozen_guard_not_weakened",
            "historical_ppo_optimizer_untouched",
            "human_demo_frozen_split_manifest_unchanged",
            "human_demo_sources_unchanged",
        )
        if not all(evidence["checks"][name] for name in required_true):
            errors.append("BLOCKED evidence lacks required preservation checks")
        if evidence["checks"]["acceptance_claimed"]:
            errors.append("BLOCKED evidence incorrectly claims acceptance")
        if evidence["checks"]["distilled_checkpoint_emitted"]:
            errors.append("BLOCKED evidence incorrectly claims a checkpoint")
        if (ROOT / CHECKPOINT).exists():
            errors.append("blocked run unexpectedly left a distilled checkpoint")
    else:
        errors.append("unknown evidence verdict")
    return {
        "valid": not errors,
        "errors": errors,
        "artifact_count": len(manifest["files"]),
        "verdict": evidence["verdict"],
    }


def finalize_blocked() -> dict[str, Any]:
    evidence_path = ROOT / RESULT_ROOT / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence["verdict"] != "BLOCKED":
        raise ValueError("refusing blocked finalization for non-blocked evidence")
    if (ROOT / CHECKPOINT).exists():
        raise RuntimeError("refusing blocked finalization with a distilled checkpoint present")
    artifacts = _artifact_manifest(
        [
            FROZEN_CONFIG,
            RESULT_ROOT / "blocked_baseline_profiles.json",
            RESULT_ROOT / "corpus_manifest.json",
            RESULT_ROOT / "evidence.json",
            RESULT_ROOT / "failed_training_evidence.json",
            RESULT_ROOT / "guard_attempts_live.json",
            RESULT_ROOT / "pre_step_freeze_evidence.json",
            RESULT_ROOT / "preflight_correction_evidence.json",
            RESULT_ROOT / "training_curve.json",
            RESULT_ROOT / "verification_evidence.json",
        ]
    )
    _write_json(ROOT / RESULT_ROOT / "artifact_manifest.json", artifacts)
    return verify()


def baseline_only(args: argparse.Namespace) -> dict[str, Any]:
    config, config_identity = _load_config()
    bootstrap_payload, policy_config, bootstrap_identity = _load_bootstrap(config)
    observations, regenerated, split = _build_rollout_corpus(
        config,
        bootstrap_payload,
        bootstrap_identity,
        device=args.device,
    )
    committed = json.loads(
        (ROOT / RESULT_ROOT / "corpus_manifest.json").read_text(encoding="utf-8")
    )
    if regenerated["identity_sha256"] != committed["identity_sha256"]:
        raise RuntimeError("read-only baseline regeneration changed corpus identity")
    teacher = Rival2ActorCritic(policy_config).to(args.device)
    teacher.load_state_dict(bootstrap_payload["model"])
    teacher.eval().requires_grad_(False)
    student = Rival2ActorCritic(policy_config).to(args.device)
    student.load_state_dict(bootstrap_payload["model"])
    student.eval().requires_grad_(False)
    gameplay_quality = torch_profile_quality(
        DegradationProfile.GAMEPLAY, device=args.device
    )
    freeplay_quality = torch_profile_quality(
        DegradationProfile.FREEPLAY, device=args.device
    )
    worlds_per_batch = int(config["training"]["worlds_per_minibatch"])
    validation = _evaluate_worlds(
        teacher,
        student,
        observations,
        split["validation"],
        worlds_per_batch=worlds_per_batch,
        gameplay_quality=gameplay_quality,
        freeplay_quality=freeplay_quality,
        policy_config=policy_config,
    )
    test = _evaluate_worlds(
        teacher,
        student,
        observations,
        split["test"],
        worlds_per_batch=worlds_per_batch,
        gameplay_quality=gameplay_quality,
        freeplay_quality=freeplay_quality,
        policy_config=policy_config,
    )
    result = {
        "format": "RIVAL2_MISSING_FEATURE_DISTILLATION_BLOCKED_BASELINES_V1",
        "frozen_config": config_identity,
        "corpus_identity_sha256": regenerated["identity_sha256"],
        "deterministic_regeneration_matches_failed_run": True,
        "validation": validation,
        "test": test,
        "student_model_tensor_sha256": tensor_tree_sha256(student.state_dict()),
        "student_byte_identical_to_teacher": tensor_tree_sha256(student.state_dict())
        == tensor_tree_sha256(teacher.state_dict())
        == bootstrap_identity["model_tensor_sha256"],
        "supervised_optimizer_constructed": False,
        "optimizer_steps": 0,
        "human_behavior_cloning_performed": False,
        "ppo_performed": False,
        "bootstrap_unchanged": file_sha256(ROOT / bootstrap_identity["path"])
        == bootstrap_identity["sha256"],
    }
    _write_json(ROOT / RESULT_ROOT / "blocked_baseline_profiles.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--human-source-root",
        type=Path,
        default=(
            Path(os.environ["APPDATA"])
            / "bakkesmod"
            / "bakkesmod"
            / "data"
            / "rival2"
            / "human_demos"
        ),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--finalize-blocked", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def _configure_cuda(device: str) -> None:
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        raise RuntimeError("CUDA PyTorch and Warp are required")
    torch.cuda.set_device(device)
    torch.manual_seed(2026082823)
    torch.cuda.manual_seed_all(2026082823)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main() -> int:
    args = parse_args()
    if args.verify_only:
        result = verify()
    elif args.finalize_blocked:
        result = finalize_blocked()
    elif args.baseline_only:
        _configure_cuda(args.device)
        result = baseline_only(args)
    else:
        _configure_cuda(args.device)
        result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if "valid" in result:
        return 0 if result["valid"] else 1
    if "verdict" in result:
        return 0 if result["verdict"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
