"""Run the prospectively frozen first Rival human behavior-cloning campaign."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_missing_feature_distillation import (  # noqa: E402
    _build_rollout_corpus,
    _load_bootstrap,
    _verify_human_sources,
)
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes  # noqa: E402
from rivalsim.human_demo.bc_observation_bridge import (  # noqa: E402
    BCBridgeTrajectoryAdapter,
    FieldQuality,
    hybrid_actor_channel_kl,
)
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    HUMAN_BC_CHECKPOINT_FORMAT,
    HUMAN_BC_VERSION,
    MechanicHierarchySampler,
    action_metric_summary,
    human_behavior_cloning_objective,
    simulator_retention_objective,
)
from rivalsim.human_demo.missing_feature_distillation import (  # noqa: E402
    MetricAccumulator,
    actor_output_statistics,
    canonical_sha256,
    file_sha256,
    world_observation_batch,
)
from rivalsim.human_demo.observation_adapter_v2 import (  # noqa: E402
    OBSERVATION_ADAPTER_CHECKPOINT_FORMAT,
    OBSERVATION_ADAPTER_VERSION,
    AdapterProfile,
    HumanDemoObservationAdapterV2,
    ObservationAdapterConfig,
    apply_native_pad_overlay,
    native_pad_overlay,
)
from rivalsim.human_demo.reader import SessionReader  # noqa: E402
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_contracts import ACTION_NAMES, OBS_DIM, OBS_FIELD_NAMES  # noqa: E402
from rivalsim.rival2_env import Rival2Env  # noqa: E402
from rivalsim.rival2_opponent_curriculum import (  # noqa: E402
    OPPONENT_NAMES,
    Rival2OpponentCurriculumConfig,
    Rival2OpponentCurriculumTrainer,
)
from rivalsim.rival2_policy import (  # noqa: E402
    Rival2ActorCritic,
    Rival2PolicyConfig,
    sample_hybrid_action,
)
from rivalsim.rival2_ppo import Rival2PPOConfig  # noqa: E402
from rivalsim.rival2_training import Rival2SelfPlayConfig  # noqa: E402

FROZEN_CONFIG = Path("results/rival2/human_behavior_cloning_v1/frozen_config.json")
FROZEN_CONFIG_SHA256 = "CC7159838230492D90F64F11E473E61142D6879B740419BDA984D558444D51C4"
RESULT_ROOT = Path("results/rival2/human_behavior_cloning_v1")
WORK_ROOT = Path(".tools/rival2_human_behavior_cloning_v1")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


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


def _load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    path = ROOT / FROZEN_CONFIG
    digest = file_sha256(path)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(f"frozen human-BC config changed: {digest}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = config["authority"]["required_parent"]
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", required, "HEAD"],
        check=True,
    )
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    if head != origin:
        raise ValueError(f"training requires remotely persisted pre-step HEAD: {head} != {origin}")
    blob = _git("rev-parse", f"HEAD:{FROZEN_CONFIG.as_posix()}")
    if blob != _git("hash-object", str(FROZEN_CONFIG)):
        raise ValueError("working frozen config differs from the committed pre-step blob")
    return config, {
        "path": FROZEN_CONFIG.as_posix(),
        "sha256": digest,
        "git_blob_oid": blob,
        "pre_step_git_commit": head,
        "origin_main": origin,
        "required_parent": required,
        "required_parent_is_ancestor": True,
    }


def _verify_authority_files(config: dict[str, Any]) -> dict[str, Any]:
    authority = config["authority"]
    checks: dict[str, Any] = {}
    for key in (
        "bootstrap_checkpoint",
        "adapter_checkpoint",
        "human_dataset_manifest",
        "human_sampling_metadata",
        "mechanic_adjudication",
    ):
        path = ROOT / authority[key]
        actual = file_sha256(path)
        expected = authority[f"{key}_sha256"]
        checks[key] = {
            "path": authority[key],
            "sha256": actual,
            "expected_sha256": expected,
            "exact": actual == expected,
            "bytes": path.stat().st_size,
        }
    if not all(row["exact"] for row in checks.values()):
        raise ValueError(f"immutable authority hash mismatch: {checks}")
    return checks


def _dataset_split_audit(config: dict[str, Any]) -> dict[str, Any]:
    dataset = json.loads(
        (ROOT / config["authority"]["human_dataset_manifest"]).read_text(encoding="utf-8")
    )
    expected = config["human_data"]
    audit: dict[str, Any] = {"splits": {}, "disjoint_attempts": True}
    attempt_splits: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        mechanics = [row for row in dataset["mechanic_positive_attempts"] if row["split"] == split]
        gameplay = [row for row in dataset["general_gameplay"]["regions"] if row["split"] == split]
        for row in mechanics:
            prior = attempt_splits.setdefault(row["attempt_id"], split)
            audit["disjoint_attempts"] &= prior == split
        audit["splits"][split] = {
            "mechanic_attempts": len(mechanics),
            "mechanic_frames": sum(int(row["source_frame_count"]) for row in mechanics),
            "gameplay_regions": len(gameplay),
            "gameplay_frames": sum(int(row["source_frame_count"]) for row in gameplay),
        }
        row = audit["splits"][split]
        row["matches_frozen_counts"] = all(
            (
                row["mechanic_attempts"] == expected["expected_mechanic_attempts"][split],
                row["mechanic_frames"] == expected["expected_mechanic_frames"][split],
                row["gameplay_frames"] == expected["expected_gameplay_frames"][split],
            )
        )
    audit["failed_or_ambiguous_in_positive_cohort"] = any(
        not bool(row.get("initial_bc_positive_cohort"))
        for row in dataset["mechanic_positive_attempts"]
    )
    audit["valid"] = bool(
        audit["disjoint_attempts"]
        and not audit["failed_or_ambiguous_in_positive_cohort"]
        and all(row["matches_frozen_counts"] for row in audit["splits"].values())
    )
    if not audit["valid"]:
        raise ValueError(f"frozen human split audit failed: {audit}")
    return audit


def _load_adapter(
    config: dict[str, Any], device: str
) -> tuple[HumanDemoObservationAdapterV2, dict[str, Any], dict[str, Any]]:
    path = ROOT / config["authority"]["adapter_checkpoint"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != OBSERVATION_ADAPTER_CHECKPOINT_FORMAT:
        raise ValueError("frozen observation-adapter checkpoint format changed")
    if payload.get("adapter_version") != OBSERVATION_ADAPTER_VERSION:
        raise ValueError("frozen observation-adapter version changed")
    adapter_config = ObservationAdapterConfig(**payload["adapter_config"])
    adapter = HumanDemoObservationAdapterV2(adapter_config).to(device)
    adapter.load_state_dict(payload["adapter"])
    adapter.eval().requires_grad_(False)
    identity = {
        "path": config["authority"]["adapter_checkpoint"],
        "sha256": file_sha256(path),
        "tensor_sha256": tensor_tree_sha256(adapter.state_dict()),
        "version": OBSERVATION_ADAPTER_VERSION,
        "external_frozen": True,
        "requires_grad_false": all(
            not parameter.requires_grad for parameter in adapter.parameters()
        ),
    }
    if identity["tensor_sha256"] != config["authority"]["adapter_tensor_sha256"]:
        raise ValueError("frozen adapter tensor identity changed")
    return adapter, payload, identity


@dataclass(slots=True)
class HumanSplit:
    split: str
    gameplay_observation: torch.Tensor
    gameplay_action: torch.Tensor
    gameplay_identity: list[str]
    mechanic_observation: torch.Tensor
    mechanic_action: torch.Tensor
    mechanic_label: list[str]
    mechanic_attempt: list[str]
    mechanic_session: list[str]
    action_sha256: str
    source_sequences_sha256: str
    quality_counts: dict[str, int]


@torch.no_grad()
def _load_human_split(
    split: str,
    *,
    config: dict[str, Any],
    adapter: HumanDemoObservationAdapterV2,
    source_root: Path,
    device: str,
) -> HumanSplit:
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown human split: {split}")
    dataset = json.loads(
        (ROOT / config["authority"]["human_dataset_manifest"]).read_text(encoding="utf-8")
    )
    spans: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    metadata: dict[str, tuple[str, str, str]] = {}
    for row in dataset["mechanic_positive_attempts"]:
        if row["split"] != split:
            continue
        identity = str(row["attempt_id"])
        metadata[identity] = ("mechanic", str(row["declared_label"]), str(row["session_uuid"]))
        spans[str(row["session_uuid"])].append(
            (identity, int(row["start_sequence"]), int(row["end_sequence"]))
        )
    gameplay = dataset["general_gameplay"]
    for row in gameplay["regions"]:
        if row["split"] != split:
            continue
        identity = str(row["region_id"])
        metadata[identity] = (
            "gameplay",
            str(gameplay["declared_label"]),
            str(gameplay["session_uuid"]),
        )
        spans[str(gameplay["session_uuid"])].append(
            (identity, int(row["start_sequence"]), int(row["end_sequence"]))
        )

    observations: dict[str, list[torch.Tensor]] = {"gameplay": [], "mechanic": []}
    actions: dict[str, list[torch.Tensor]] = {"gameplay": [], "mechanic": []}
    identities: dict[str, list[str]] = {"gameplay": [], "mechanic": []}
    labels: list[str] = []
    mechanic_sessions: list[str] = []
    quality_count = np.zeros(4, dtype=np.int64)
    action_digest = hashlib.sha256()
    sequence_digest = hashlib.sha256()
    buffer: dict[str, list[tuple[str, Any, np.ndarray, np.ndarray]]] = {
        "gameplay": [],
        "mechanic": [],
    }

    def flush(kind: str) -> None:
        if not buffer[kind]:
            return
        profile = AdapterProfile.GAMEPLAY if kind == "gameplay" else AdapterProfile.FREEPLAY
        source = torch.from_numpy(
            np.stack([np.asarray(row[1].observation) for row in buffer[kind]])
        ).to(device)
        quality = torch.from_numpy(
            np.stack([np.asarray(row[1].quality) for row in buffer[kind]])
        ).to(device)
        repaired = adapter(source, quality, profile=profile)
        if kind == "gameplay":
            pad_values = torch.from_numpy(np.stack([row[2] for row in buffer[kind]])).to(device)
            pad_support = torch.from_numpy(np.stack([row[3] for row in buffer[kind]])).to(device)
            repaired = apply_native_pad_overlay(repaired, pad_values, pad_support)
        observations[kind].append(repaired.cpu())
        actions[kind].append(
            torch.from_numpy(np.stack([np.asarray(row[1].action) for row in buffer[kind]])).to(
                torch.float32
            )
        )
        identities[kind].extend(row[0] for row in buffer[kind])
        if kind == "mechanic":
            for identity, _sample, _pad, _support in buffer[kind]:
                labels.append(metadata[identity][1])
                mechanic_sessions.append(metadata[identity][2])
        buffer[kind].clear()

    for session_uuid in sorted(spans):
        trajectory = BCBridgeTrajectoryAdapter(source_root / session_uuid)
        contains_gameplay = any(metadata[row[0]][0] == "gameplay" for row in spans[session_uuid])
        native_iterator = (
            iter(SessionReader(source_root / session_uuid).iter_frames())
            if contains_gameplay
            else None
        )
        native_frame = next(native_iterator, None) if native_iterator is not None else None
        for identity, sample in trajectory.iter_spans(spans[session_uuid]):
            if not sample.bc_usable or not sample.action_unchanged_from_exact_adapter:
                raise RuntimeError(f"accepted human sample became unusable or changed: {identity}")
            kind = metadata[identity][0]
            if kind == "gameplay":
                while native_frame is not None and int(native_frame["sequence"]) < int(
                    sample.sequence
                ):
                    native_frame = next(native_iterator, None)
                if native_frame is None or int(native_frame["sequence"]) != int(sample.sequence):
                    raise RuntimeError("native gameplay pad-overlay sequence alignment failed")
                overlay = native_pad_overlay(native_frame)
                pad = np.asarray(overlay.values).copy()
                support = np.asarray(overlay.supported).copy()
            else:
                pad = np.zeros(OBS_DIM, dtype=np.float32)
                support = np.zeros(OBS_DIM, dtype=np.bool_)
            quality = np.asarray(sample.quality)
            quality_count += np.bincount(quality, minlength=4)
            action = np.asarray(sample.action, dtype=np.float32)
            if (
                not np.isfinite(action).all()
                or np.any(action[:5] < -1.0)
                or np.any(action[:5] > 1.0)
            ):
                raise ValueError("human action violates analog contract")
            if not np.isin(action[5:], (0.0, 1.0)).all():
                raise ValueError("human action violates exact button contract")
            action_digest.update(action.tobytes(order="C"))
            sequence_digest.update(f"{session_uuid}:{sample.sequence}\n".encode())
            buffer[kind].append((identity, sample, pad, support))
            if len(buffer[kind]) >= 8192:
                flush(kind)
    flush("gameplay")
    flush("mechanic")

    def merged(rows: list[torch.Tensor], width: int) -> torch.Tensor:
        return torch.cat(rows, dim=0) if rows else torch.empty((0, width), dtype=torch.float32)

    result = HumanSplit(
        split=split,
        gameplay_observation=merged(observations["gameplay"], OBS_DIM),
        gameplay_action=merged(actions["gameplay"], 8),
        gameplay_identity=identities["gameplay"],
        mechanic_observation=merged(observations["mechanic"], OBS_DIM),
        mechanic_action=merged(actions["mechanic"], 8),
        mechanic_label=labels,
        mechanic_attempt=identities["mechanic"],
        mechanic_session=mechanic_sessions,
        action_sha256=action_digest.hexdigest().upper(),
        source_sequences_sha256=sequence_digest.hexdigest().upper(),
        quality_counts={
            "unavailable": int(quality_count[int(FieldQuality.UNAVAILABLE)]),
            "approximate": int(quality_count[int(FieldQuality.APPROXIMATE)]),
            "exactly_derived": int(quality_count[int(FieldQuality.EXACT_DERIVED)]),
            "exact_direct": int(quality_count[int(FieldQuality.EXACT_DIRECT)]),
        },
    )
    expected = config["human_data"]
    if result.gameplay_action.shape[0] != expected["expected_gameplay_frames"][split]:
        raise RuntimeError(f"{split} gameplay frame count changed")
    if result.mechanic_action.shape[0] != expected["expected_mechanic_frames"][split]:
        raise RuntimeError(f"{split} mechanic frame count changed")
    if len(set(result.mechanic_attempt)) != expected["expected_mechanic_attempts"][split]:
        raise RuntimeError(f"{split} whole-attempt count changed")
    if not bool(torch.isfinite(result.gameplay_observation).all()):
        raise RuntimeError(f"{split} gameplay repaired observations are nonfinite")
    if not bool(torch.isfinite(result.mechanic_observation).all()):
        raise RuntimeError(f"{split} mechanic repaired observations are nonfinite")
    return result


@torch.no_grad()
def _evaluate_human(
    model: Rival2ActorCritic,
    teacher: Rival2ActorCritic,
    data: HumanSplit,
    *,
    device: str,
    batch_size: int = 8192,
) -> dict[str, Any]:
    result: dict[str, Any] = {"split": data.split, "families": {}, "per_mechanic_label": {}}
    for family, observation, target in (
        ("gameplay", data.gameplay_observation, data.gameplay_action),
        ("mechanic", data.mechanic_observation, data.mechanic_action),
    ):
        actors: list[torch.Tensor] = []
        teacher_log_std: list[torch.Tensor] = []
        for start in range(0, observation.shape[0], batch_size):
            value = observation[start : start + batch_size].to(device)
            actor, _critic = model(value)
            teacher_actor, _teacher_critic = teacher(value)
            actors.append(actor.cpu())
            teacher_log_std.append(teacher_actor[:, 5:10].cpu())
        actor = torch.cat(actors)
        frozen_log_std = torch.cat(teacher_log_std).clamp(-5.0, 1.0)
        metrics = action_metric_summary(actor, target)
        student_log_std = actor[:, 5:10].clamp(-5.0, 1.0)
        drift = student_log_std - frozen_log_std
        metrics["log_std_teacher_drift"] = {
            "mae": float(drift.abs().mean().item()),
            "rmse": float(drift.square().mean().sqrt().item()),
            "per_channel_mae": {
                ACTION_NAMES[index]: float(drift[:, index].abs().mean().item())
                for index in range(5)
            },
        }
        metrics["actor_output_statistics"] = actor_output_statistics(actor)
        metrics["finite"] = bool(torch.isfinite(actor).all())
        result["families"][family] = metrics
        if family == "mechanic":
            labels = np.asarray(data.mechanic_label)
            for label in sorted(set(data.mechanic_label)):
                indices = torch.from_numpy(np.flatnonzero(labels == label))
                result["per_mechanic_label"][label] = action_metric_summary(
                    actor.index_select(0, indices), target.index_select(0, indices)
                )
    result["finite"] = all(row["finite"] for row in result["families"].values())
    return result


@torch.no_grad()
def _evaluate_retention(
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    adapter: HumanDemoObservationAdapterV2,
    observations: torch.Tensor,
    worlds: np.ndarray,
    *,
    worlds_per_batch: int,
    policy_config: Rival2PolicyConfig,
) -> dict[str, Any]:
    accumulator = MetricAccumulator.create()
    for start in range(0, len(worlds), worlds_per_batch):
        full = world_observation_batch(observations, worlds[start : start + worlds_per_batch])
        structural = adapter(full, None, profile=AdapterProfile.FULL)
        if structural.data_ptr() != full.data_ptr():
            raise RuntimeError("full authoritative adapter path stopped being a structural bypass")
        teacher_actor, teacher_value = teacher(full)
        student_actor, student_value = student(structural)
        channel = hybrid_actor_channel_kl(teacher_actor, student_actor, policy_config=policy_config)
        accumulator.update(channel, teacher_value, student_value)
    return accumulator.result()


def _guard(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    limit = config["retention"]["hard_guard"]
    checks = {
        "actor_mean_kl": metrics["actor_mean_kl"] <= limit["actor_mean_kl"],
        "actor_max_sample_kl": metrics["actor_max_sample_kl"] <= limit["actor_max_sample_kl"],
        "actor_max_channel_kl": max(metrics["actor_channel_kl"].values())
        <= limit["actor_max_channel_kl"],
        "critic_rmse": metrics["value_rmse"] <= limit["critic_rmse"],
        "critic_max_absolute_drift": metrics["value_max_absolute_drift"]
        <= limit["critic_max_absolute_drift"],
        "finite": metrics["actor_finite"] and metrics["value_finite"],
    }
    return {"checks": checks, "accepted": all(checks.values())}


def _candidate_summary(
    *,
    step: int,
    human: dict[str, Any],
    baseline: dict[str, Any],
    retention: dict[str, Any],
    guard: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gameplay_ratio = (
        human["families"]["gameplay"]["complete_action_rmse"]
        / baseline["families"]["gameplay"]["complete_action_rmse"]
    )
    mechanic_ratio = (
        human["families"]["mechanic"]["complete_action_rmse"]
        / baseline["families"]["mechanic"]["complete_action_rmse"]
    )
    label_ratios = {
        label: row["complete_action_rmse"]
        / baseline["per_mechanic_label"][label]["complete_action_rmse"]
        for label, row in human["per_mechanic_label"].items()
    }
    score = (
        0.5 * (gameplay_ratio + mechanic_ratio)
        + 0.10 * retention["actor_mean_kl"] / config["retention"]["soft_actor_mean_kl"]
    )
    return {
        "accepted_step": step,
        "human_validation": human,
        "simulator_retention": retention,
        "retention_guard": guard,
        "gameplay_rmse_ratio": gameplay_ratio,
        "mechanic_rmse_ratio": mechanic_ratio,
        "mechanic_label_rmse_ratios": label_ratios,
        "mechanic_labels_improved_fraction": sum(value < 1.0 for value in label_ratios.values())
        / len(label_ratios),
        "mechanic_labels_nonregressed_fraction": sum(
            value <= 1.0 + config["acceptance"]["mechanic_label_nonregression_relative_tolerance"]
            for value in label_ratios.values()
        )
        / len(label_ratios),
        "selection_score": score,
        "eligible_for_selection": bool(
            guard["accepted"] and human["finite"] and gameplay_ratio < 1.0 and mechanic_ratio < 1.0
        ),
    }


def _save_work_checkpoint(
    path: Path,
    *,
    student: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    accepted_steps: int,
    proposed_steps: int,
    candidate: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _cpu_tree(student.state_dict()),
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "accepted_steps": accepted_steps,
            "proposed_steps": proposed_steps,
            "candidate": candidate,
        },
        path,
    )


def _restore_work_checkpoint(
    path: Path,
    student: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    student.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload


def _train(
    *,
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    adapter: HumanDemoObservationAdapterV2,
    policy_config: Rival2PolicyConfig,
    train_data: HumanSplit,
    validation_data: HumanSplit,
    corpus: torch.Tensor,
    train_worlds: np.ndarray,
    validation_worlds: np.ndarray,
    config: dict[str, Any],
    device: str,
) -> tuple[torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    settings = config["training"]
    objective = config["objective"]
    sampling = config["sampling"]
    generator = torch.Generator(device="cpu").manual_seed(int(settings["seed"]))
    simulator_generator = torch.Generator(device="cpu").manual_seed(int(settings["seed"]) + 1)
    mechanic_sampler = MechanicHierarchySampler(
        train_data.mechanic_label,
        train_data.mechanic_attempt,
        uniform_label_fraction=float(sampling["mechanic_uniform_label_fraction"]),
        maximum_oversampling_ratio=float(sampling["maximum_mechanic_frame_oversampling_ratio"]),
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(settings["initial_learning_rate"]),
        betas=tuple(settings["optimizer_betas"]),
        eps=float(settings["optimizer_epsilon"]),
        weight_decay=float(settings["weight_decay"]),
    )
    baseline_human = _evaluate_human(student, teacher, validation_data, device=device)
    selected_validation_worlds = validation_worlds[
        : int(config["retention"]["validation_subset"]["worlds"])
    ]
    baseline_retention = _evaluate_retention(
        teacher,
        student,
        adapter,
        corpus,
        selected_validation_worlds,
        worlds_per_batch=int(config["retention"]["worlds_per_validation_batch"]),
        policy_config=policy_config,
    )
    baseline_guard = _guard(baseline_retention, config)
    if not baseline_guard["accepted"] or baseline_retention["actor_mean_kl"] != 0.0:
        raise RuntimeError("byte-identical student failed zero-drift baseline retention")
    baseline = {
        "accepted_step": 0,
        "human_validation": baseline_human,
        "simulator_retention": baseline_retention,
        "retention_guard": baseline_guard,
        "selection_score": 1.0,
    }
    _write_json(ROOT / WORK_ROOT / "baseline_validation.json", baseline)

    accepted_steps = 0
    proposed_steps = 0
    interval = int(settings["validation_interval_optimizer_steps"])
    best_score = 1.0
    best_candidate: dict[str, Any] | None = None
    best_path = ROOT / WORK_ROOT / "best.pt"
    curve: list[dict[str, Any]] = []
    interval_attempts: list[dict[str, Any]] = []
    no_improvement = 0
    stop_reason = "maximum accepted optimizer steps"
    initial_student_hash = tensor_tree_sha256(student.state_dict())

    while accepted_steps < int(settings["max_accepted_optimizer_steps"]):
        block = min(interval, int(settings["max_accepted_optimizer_steps"]) - accepted_steps)
        rollback_path = ROOT / WORK_ROOT / "rollback.pt"
        _save_work_checkpoint(
            rollback_path,
            student=student,
            optimizer=optimizer,
            accepted_steps=accepted_steps,
            proposed_steps=proposed_steps,
            candidate={},
        )
        generator_state = generator.get_state()
        simulator_generator_state = simulator_generator.get_state()
        retries = 0
        while True:
            loss_sums = defaultdict(float)
            grad_norm_max = 0.0
            for _ in range(block):
                proposed_steps += 1
                gameplay_indices = torch.randint(
                    train_data.gameplay_observation.shape[0],
                    (int(sampling["gameplay_frames_per_step"]),),
                    generator=generator,
                )
                mechanic_indices = mechanic_sampler.sample(
                    int(sampling["mechanic_frames_per_step"])
                )
                human_observation = torch.cat(
                    (
                        train_data.gameplay_observation.index_select(0, gameplay_indices),
                        train_data.mechanic_observation.index_select(0, mechanic_indices),
                    )
                ).to(device)
                human_action = torch.cat(
                    (
                        train_data.gameplay_action.index_select(0, gameplay_indices),
                        train_data.mechanic_action.index_select(0, mechanic_indices),
                    )
                ).to(device)
                sim_positions = torch.randint(
                    len(train_worlds),
                    (int(config["retention"]["worlds_per_training_minibatch"]),),
                    generator=simulator_generator,
                ).numpy()
                sim_worlds = train_worlds[sim_positions]
                sim_observation = world_observation_batch(corpus, sim_worlds)
                full_student_observation = adapter(
                    sim_observation, None, profile=AdapterProfile.FULL
                )
                with torch.no_grad():
                    teacher_human_actor, _ = teacher(human_observation)
                    teacher_sim_actor, teacher_sim_value = teacher(sim_observation)
                student_human_actor, _ = student(human_observation)
                student_sim_actor, student_sim_value = student(full_student_observation)
                human_loss = human_behavior_cloning_objective(
                    student_human_actor,
                    teacher_human_actor,
                    human_action,
                    smooth_l1_beta=float(objective["smooth_l1_beta"]),
                    analog_weight=float(objective["analog_weight"]),
                    button_weight=float(objective["button_weight"]),
                    log_std_weight=float(objective["human_log_std_retention_weight"]),
                    policy_config=policy_config,
                )
                retention_loss = simulator_retention_objective(
                    student_sim_actor,
                    student_sim_value,
                    teacher_sim_actor,
                    teacher_sim_value,
                    actor_weight=float(objective["simulator_actor_retention_weight"]),
                    critic_weight=float(objective["simulator_critic_retention_weight"]),
                    policy_config=policy_config,
                )
                loss = human_loss.loss + retention_loss.loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite human-BC objective before optimizer step")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), float(settings["gradient_clip_norm"])
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise RuntimeError("nonfinite human-BC gradient norm")
                optimizer.step()
                if not all(
                    bool(torch.isfinite(parameter).all()) for parameter in student.parameters()
                ):
                    raise RuntimeError("nonfinite human-BC parameter after optimizer step")
                grad_norm_max = max(grad_norm_max, float(grad_norm.item()))
                for key, value in (
                    ("total", loss),
                    ("analog", human_loss.analog_smooth_l1),
                    ("buttons", human_loss.button_bce),
                    ("human_log_std", human_loss.log_std_retention),
                    ("sim_actor_kl", retention_loss.actor_kl),
                    ("sim_critic_mse", retention_loss.critic_mse),
                ):
                    loss_sums[key] += float(value.detach().item())
                for index, name in enumerate(ACTION_NAMES[:5]):
                    loss_sums[f"human_analog_{name}"] += float(
                        human_loss.analog_per_channel[index].detach().item()
                    )
                    loss_sums[f"human_log_std_{name}"] += float(
                        human_loss.log_std_per_channel[index].detach().item()
                    )
                for index, name in enumerate(ACTION_NAMES[5:]):
                    loss_sums[f"human_button_{name}"] += float(
                        human_loss.button_per_channel[index].detach().item()
                    )
                for index, name in enumerate(ACTION_NAMES):
                    loss_sums[f"simulator_actor_kl_{name}"] += float(
                        retention_loss.per_channel_kl[index].detach().item()
                    )

            candidate_human = _evaluate_human(student, teacher, validation_data, device=device)
            candidate_retention = _evaluate_retention(
                teacher,
                student,
                adapter,
                corpus,
                selected_validation_worlds,
                worlds_per_batch=int(config["retention"]["worlds_per_validation_batch"]),
                policy_config=policy_config,
            )
            candidate_guard = _guard(candidate_retention, config)
            attempted_step = accepted_steps + block
            attempt = {
                "attempted_accepted_step": attempted_step,
                "retry": retries,
                "proposed_optimizer_steps": proposed_steps,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "mean_training_loss": {key: value / block for key, value in loss_sums.items()},
                "max_preclip_gradient_norm": grad_norm_max,
                "simulator_retention": candidate_retention,
                "retention_guard": candidate_guard,
            }
            interval_attempts.append(attempt)
            if candidate_guard["accepted"]:
                accepted_steps = attempted_step
                candidate = _candidate_summary(
                    step=accepted_steps,
                    human=candidate_human,
                    baseline=baseline_human,
                    retention=candidate_retention,
                    guard=candidate_guard,
                    config=config,
                )
                candidate["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                candidate["training_loss"] = attempt["mean_training_loss"]
                candidate["max_preclip_gradient_norm"] = grad_norm_max
                candidate["soft_retention_limit_exceeded"] = (
                    candidate_retention["actor_mean_kl"] > config["retention"]["soft_actor_mean_kl"]
                )
                curve.append(candidate)
                boundary_path = ROOT / WORK_ROOT / "accepted" / f"step-{accepted_steps:04d}.pt"
                _save_work_checkpoint(
                    boundary_path,
                    student=student,
                    optimizer=optimizer,
                    accepted_steps=accepted_steps,
                    proposed_steps=proposed_steps,
                    candidate=candidate,
                )
                material = candidate["eligible_for_selection"] and candidate[
                    "selection_score"
                ] < best_score - float(
                    config["selection"]["early_stopping_material_score_improvement"]
                )
                if material:
                    best_score = float(candidate["selection_score"])
                    best_candidate = copy.deepcopy(candidate)
                    _save_work_checkpoint(
                        best_path,
                        student=student,
                        optimizer=optimizer,
                        accepted_steps=accepted_steps,
                        proposed_steps=proposed_steps,
                        candidate=candidate,
                    )
                    no_improvement = 0
                else:
                    no_improvement += 1
                break

            _restore_work_checkpoint(rollback_path, student, optimizer)
            generator.set_state(generator_state)
            simulator_generator.set_state(simulator_generator_state)
            retries += 1
            current_lr = float(optimizer.param_groups[0]["lr"])
            next_lr = current_lr * float(settings["lr_backoff_factor"])
            if retries > int(settings["max_guard_retries_per_interval"]) or next_lr < float(
                settings["minimum_learning_rate"]
            ):
                stop_reason = "hard retention guard exhausted transactional retries"
                break
            for group in optimizer.param_groups:
                group["lr"] = next_lr
        if stop_reason == "hard retention guard exhausted transactional retries":
            break
        if no_improvement >= int(config["selection"]["early_stopping_patience_validations"]):
            stop_reason = "validation early stopping patience reached"
            break

    if best_candidate is None or not best_path.is_file():
        raise RuntimeError("bounded campaign selected no human-improving retention-safe checkpoint")
    final_training_state = _restore_work_checkpoint(best_path, student, optimizer)
    result = {
        "version": HUMAN_BC_VERSION,
        "fresh_optimizer": True,
        "historical_ppo_optimizer_loaded": False,
        "initial_student_model_tensor_sha256": initial_student_hash,
        "baseline": baseline,
        "curve": curve,
        "interval_attempts": interval_attempts,
        "accepted_optimizer_steps_executed": accepted_steps,
        "proposed_optimizer_steps": proposed_steps,
        "selected_accepted_step": int(final_training_state["accepted_steps"]),
        "selected_candidate": best_candidate,
        "stop_reason": stop_reason,
        "mechanic_sampling": {
            "uniform_label_fraction": mechanic_sampler.uniform_label_fraction,
            "maximum_realized_oversampling_ratio": (
                mechanic_sampler.maximum_realized_oversampling_ratio
            ),
            "frozen_cap": mechanic_sampler.maximum_oversampling_ratio,
            "label_count": len(mechanic_sampler.labels),
        },
    }
    return optimizer, result, best_candidate


@torch.no_grad()
def _mixed_opponent_sanity(
    *,
    label: str,
    model_state: dict[str, torch.Tensor],
    bootstrap_payload: dict[str, Any],
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    settings = config["closed_loop"]
    worlds = int(settings["sanity_worlds"])
    decisions = int(settings["sanity_decisions_per_world"])
    seed = int(settings["sanity_seed"])
    collision_dir = config["retention"]["corpus"]["collision_mesh_directory"]
    geometry = ArenaGeometry.load_soccar(collision_dir)
    meshes = WarpArenaMeshes(geometry, device)
    kickoff_selector = (np.arange(worlds, dtype=np.int32) + seed) % 5
    env = Rival2Env(
        worlds,
        collision_dir,
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
    trainer.load_checkpoint(ROOT / config["authority"]["bootstrap_checkpoint"])
    trainer.model.load_state_dict(model_state)
    trainer.model.eval()
    generator = trainer.policy_generator
    observation = env.observation
    touches = goals = concedes = truncations = no_touch = 0
    movement_speed_sum = action_rows = 0.0
    analog_saturated = button_sum = 0.0
    family_counts = torch.bincount(trainer.opponent_family, minlength=4).cpu().tolist()
    self_velocity_indices = [
        OBS_FIELD_NAMES.index(f"self.linear_velocity.{axis}") for axis in "xyz"
    ]
    for _ in range(decisions):
        actor, _value, _version, train_mask = trainer._policy_outputs(observation)
        sample = sample_hybrid_action(actor, generator=generator, config=trainer.policy_config)
        action = trainer._apply_historical_policy_cadence(sample.action)
        selected_action = action[train_mask]
        selected_obs = observation[train_mask]
        action_rows += float(selected_action.shape[0])
        analog_saturated += float((selected_action[:, :5].abs() >= 0.999).sum().item())
        button_sum += float(selected_action[:, 5:8].sum().item())
        movement_speed_sum += float(
            torch.linalg.vector_norm(selected_obs[:, self_velocity_indices], dim=1).sum().item()
        )
        transition = trainer._step_with_frozen_opponents(action)
        terminal = transition.terminated
        truncated = transition.truncated
        terminal_observation = transition.transition_observation
        touches += int(((terminal_observation[..., 176] > 0.5) & train_mask).sum().item())
        truncations += int(truncated.sum().item())
        no_touch += int((truncated & (terminal_observation[:, 0, 181] >= 1.0)).sum().item())
        scoring_team = env.bridge.views["rival2.scoring_team_latched"].to(torch.int64)
        rival_scored = terminal & (scoring_team == trainer.rival_side)
        opponent_scored = terminal & (scoring_team == (1 - trainer.rival_side))
        goals += int(rival_scored.sum().item())
        concedes += int(opponent_scored.sum().item())
        trainer.assign_opponents_at_reset(transition.reset_mask)
        observation = transition.observation
    simulated_minutes = decisions * worlds / 120.0 / 60.0
    result = {
        "label": label,
        "worlds": worlds,
        "decisions_per_world": decisions,
        "simulated_minutes": simulated_minutes,
        "opponent_family_worlds_at_start": {
            OPPONENT_NAMES[index]: int(value) for index, value in enumerate(family_counts)
        },
        "touches": touches,
        "touches_per_simulated_minute": touches / simulated_minutes,
        "goals": goals,
        "goals_per_simulated_minute": goals / simulated_minutes,
        "concedes": concedes,
        "concedes_per_simulated_minute": concedes / simulated_minutes,
        "truncations": truncations,
        "no_touch_truncations": no_touch,
        "mean_normalized_self_speed": movement_speed_sum / action_rows,
        "analog_saturation_fraction": analog_saturated / (action_rows * 5.0),
        "button_activation_fraction": button_sum / (action_rows * 3.0),
        "all_finite": all(
            math.isfinite(value)
            for value in (
                movement_speed_sum,
                action_rows,
                analog_saturated,
                button_sum,
            )
        ),
    }
    del trainer, env, meshes, geometry
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _mechanic_closed_loop_representability(config: dict[str, Any]) -> dict[str, Any]:
    dataset = json.loads(
        (ROOT / config["authority"]["human_dataset_manifest"]).read_text(encoding="utf-8")
    )
    labels = sorted({str(row["declared_label"]) for row in dataset["mechanic_positive_attempts"]})
    rows = {
        label: {
            "verdict": "NOT_EVALUABLE_EXACTLY",
            "source_attempt_count": sum(
                row["declared_label"] == label for row in dataset["mechanic_positive_attempts"]
            ),
            "reasons": [
                "no native frame is exact-adapter usable for all 182 simulator observation fields",
                "RivalSim lifecycle and contact-history state is not source-exact in the "
                "native recording",
                "Rocket League native contact manifold and engine integrator state cannot "
                "be imported exactly into RivalSim",
            ],
            "source_bound_adjudication_run": False,
            "learned_capability_claimed": False,
        }
        for label in labels
    }
    return {
        "format": "RIVAL2_HUMAN_BC_CLOSED_LOOP_MECHANIC_REPRESENTABILITY_V1",
        "exact_start_required": True,
        "criteria_path": config["authority"]["mechanic_adjudication"],
        "criteria_sha256": config["authority"]["mechanic_adjudication_sha256"],
        "criteria_used_as_detector_or_reward": False,
        "labels": rows,
        "exactly_evaluable_label_count": 0,
        "not_evaluable_label_count": len(rows),
        "capability_observed": False,
        "interpretation": (
            "No mechanic is scored from an approximate cross-engine start; open-loop "
            "imitation is not called learned closed-loop capability."
        ),
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
        "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_ARTIFACT_MANIFEST_V1",
        "files": rows,
        "file_set_sha256": canonical_sha256(rows),
    }


def _preflight(
    config: dict[str, Any],
    config_identity: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    authority = _verify_authority_files(config)
    splits = _dataset_split_audit(config)
    sources = _verify_human_sources(source_root)
    result = {
        "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_PRE_STEP_PREFLIGHT_V1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_identity,
        "authority": authority,
        "split_audit": splits,
        "source_verification": sources,
        "optimizer_steps": 0,
        "ppo_updates": 0,
        "valid": True,
    }
    _write_json(ROOT / WORK_ROOT / "pre_step_preflight.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_identity = _load_config()
    preflight = _preflight(config, config_identity, args.human_source_root)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return {"verdict": "PREFLIGHT_PASS"}

    started = time.perf_counter()
    torch.manual_seed(int(config["training"]["seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["seed"]))
    np.random.seed(int(config["training"]["seed"]))
    bootstrap_bytes_before = (ROOT / config["authority"]["bootstrap_checkpoint"]).read_bytes()
    adapter_bytes_before = (ROOT / config["authority"]["adapter_checkpoint"]).read_bytes()
    source_before = _verify_human_sources(args.human_source_root)
    compatibility = {"authority": config["authority"], "corpus": config["retention"]["corpus"]}
    bootstrap_payload, policy_config, bootstrap_identity = _load_bootstrap(compatibility)
    historical_optimizer_before = tensor_tree_sha256(bootstrap_payload["optimizer"])
    adapter, _adapter_payload, adapter_identity = _load_adapter(config, args.device)
    train_data = _load_human_split(
        "train",
        config=config,
        adapter=adapter,
        source_root=args.human_source_root,
        device=args.device,
    )
    validation_data = _load_human_split(
        "validation",
        config=config,
        adapter=adapter,
        source_root=args.human_source_root,
        device=args.device,
    )
    human_split_identity = {
        "train": {
            "gameplay_frames": int(train_data.gameplay_action.shape[0]),
            "mechanic_frames": int(train_data.mechanic_action.shape[0]),
            "mechanic_attempts": len(set(train_data.mechanic_attempt)),
            "action_sha256": train_data.action_sha256,
            "source_sequences_sha256": train_data.source_sequences_sha256,
            "quality_counts": train_data.quality_counts,
        },
        "validation": {
            "gameplay_frames": int(validation_data.gameplay_action.shape[0]),
            "mechanic_frames": int(validation_data.mechanic_action.shape[0]),
            "mechanic_attempts": len(set(validation_data.mechanic_attempt)),
            "action_sha256": validation_data.action_sha256,
            "source_sequences_sha256": validation_data.source_sequences_sha256,
            "quality_counts": validation_data.quality_counts,
        },
    }
    test_access_before_selection = False
    corpus, corpus_manifest, splits = _build_rollout_corpus(
        compatibility,
        bootstrap_payload,
        bootstrap_identity,
        device=args.device,
    )
    corpus_checks = {
        "identity_exact": corpus_manifest["identity_sha256"]
        == config["retention"]["corpus"]["expected_identity_sha256"],
        "observation_hash_exact": corpus_manifest["collection"]["observation_tensor_sha256"]
        == config["retention"]["corpus"]["expected_observation_tensor_sha256"],
        "test_worlds_not_used_for_training_or_selection": True,
    }
    if not all(corpus_checks.values()):
        raise RuntimeError(f"authoritative simulator corpus changed: {corpus_checks}")
    torch.use_deterministic_algorithms(True)
    teacher = Rival2ActorCritic(policy_config).to(args.device)
    teacher.load_state_dict(bootstrap_payload["model"])
    teacher.eval().requires_grad_(False)
    student = Rival2ActorCritic(policy_config).to(args.device)
    student.load_state_dict(bootstrap_payload["model"])
    student.train()
    initial_parity = tensor_tree_sha256(student.state_dict()) == tensor_tree_sha256(
        teacher.state_dict()
    )
    if not initial_parity:
        raise RuntimeError("student did not initialize byte-identically from bootstrap")

    optimizer, training, selected = _train(
        teacher=teacher,
        student=student,
        adapter=adapter,
        policy_config=policy_config,
        train_data=train_data,
        validation_data=validation_data,
        corpus=corpus,
        train_worlds=splits["train"],
        validation_worlds=splits["validation"],
        config=config,
        device=args.device,
    )

    # The test boundaries are opened exactly once and only after one validation
    # checkpoint has been selected and restored.
    test_access_utc = datetime.now(UTC).isoformat()
    test_data = _load_human_split(
        "test",
        config=config,
        adapter=adapter,
        source_root=args.human_source_root,
        device=args.device,
    )
    human_split_identity["test"] = {
        "gameplay_frames": int(test_data.gameplay_action.shape[0]),
        "mechanic_frames": int(test_data.mechanic_action.shape[0]),
        "mechanic_attempts": len(set(test_data.mechanic_attempt)),
        "action_sha256": test_data.action_sha256,
        "source_sequences_sha256": test_data.source_sequences_sha256,
        "quality_counts": test_data.quality_counts,
    }
    test_baseline = _evaluate_human(teacher, teacher, test_data, device=args.device)
    test_final = _evaluate_human(student, teacher, test_data, device=args.device)
    simulator_test = _evaluate_retention(
        teacher,
        student,
        adapter,
        corpus,
        splits["test"],
        worlds_per_batch=int(config["retention"]["worlds_per_validation_batch"]),
        policy_config=policy_config,
    )
    simulator_test_guard = _guard(simulator_test, config)

    selected_state = _cpu_tree(student.state_dict())
    del corpus, train_data, validation_data, test_data
    gc.collect()
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)
    gameplay_baseline = _mixed_opponent_sanity(
        label="bootstrap",
        model_state=bootstrap_payload["model"],
        bootstrap_payload=bootstrap_payload,
        config=config,
        device=args.device,
    )
    gameplay_final = _mixed_opponent_sanity(
        label="human_bc_v1",
        model_state=selected_state,
        bootstrap_payload=bootstrap_payload,
        config=config,
        device=args.device,
    )
    sanity_comparison = {
        "bootstrap": gameplay_baseline,
        "selected": gameplay_final,
        "touch_rate_ratio": gameplay_final["touches_per_simulated_minute"]
        / max(gameplay_baseline["touches_per_simulated_minute"], 1e-12),
        "mean_speed_ratio": gameplay_final["mean_normalized_self_speed"]
        / max(gameplay_baseline["mean_normalized_self_speed"], 1e-12),
        "analog_saturation_absolute_change": gameplay_final["analog_saturation_fraction"]
        - gameplay_baseline["analog_saturation_fraction"],
        "goal_rate_change": gameplay_final["goals_per_simulated_minute"]
        - gameplay_baseline["goals_per_simulated_minute"],
        "concede_rate_change": gameplay_final["concedes_per_simulated_minute"]
        - gameplay_baseline["concedes_per_simulated_minute"],
    }
    acceptance = config["acceptance"]
    validation = selected
    checks = {
        "gameplay_imitation_material": 1.0 - validation["gameplay_rmse_ratio"]
        >= acceptance["gameplay_complete_action_rmse_relative_improvement_min"],
        "mechanic_imitation_material": 1.0 - validation["mechanic_rmse_ratio"]
        >= acceptance["mechanic_complete_action_rmse_relative_improvement_min"],
        "mechanic_labels_broadly_improved": validation["mechanic_labels_improved_fraction"]
        >= acceptance["mechanic_labels_improved_fraction_min"],
        "mechanic_labels_broadly_nonregressed": validation["mechanic_labels_nonregressed_fraction"]
        >= acceptance["mechanic_labels_nonregressed_fraction_min"],
        "validation_retention_safe": validation["retention_guard"]["accepted"],
        "test_retention_safe": simulator_test_guard["accepted"],
        "gameplay_touch_rate_not_catastrophic": sanity_comparison["touch_rate_ratio"]
        >= acceptance["gameplay_sanity_touch_rate_relative_floor"],
        "gameplay_movement_not_catastrophic": sanity_comparison["mean_speed_ratio"]
        >= acceptance["gameplay_sanity_mean_speed_relative_floor"],
        "gameplay_action_saturation_not_catastrophic": sanity_comparison[
            "analog_saturation_absolute_change"
        ]
        <= acceptance["gameplay_sanity_action_saturation_absolute_increase_max"],
        "all_finite": bool(
            test_baseline["finite"]
            and test_final["finite"]
            and gameplay_baseline["all_finite"]
            and gameplay_final["all_finite"]
        ),
    }
    representability = _mechanic_closed_loop_representability(config)
    checks["closed_loop_requirement_respected"] = bool(
        representability["exactly_evaluable_label_count"] == 0
        or representability["capability_observed"]
    )
    accepted = all(checks.values())

    checkpoint_path = ROOT / config["checkpoint"]["path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "format": HUMAN_BC_CHECKPOINT_FORMAT,
        "version": HUMAN_BC_VERSION,
        "model": selected_state,
        "fresh_supervised_optimizer": _cpu_tree(optimizer.state_dict()),
        "optimizer_provenance": {
            "fresh_for_human_bc": True,
            "historical_ppo_optimizer_loaded": False,
            "historical_ppo_optimizer_resumable": False,
            "selected_accepted_step": training["selected_accepted_step"],
        },
        "policy_config": asdict(policy_config),
        "observation_version": bootstrap_payload["observation_version"],
        "action_version": bootstrap_payload["action_version"],
        "physics_hz": 120,
        "policy_hz": 120,
        "authority": {
            "frozen_config": config_identity,
            "bootstrap": bootstrap_identity,
            "adapter": adapter_identity,
            "human_dataset_manifest_sha256": config["authority"]["human_dataset_manifest_sha256"],
            "human_split_identity": human_split_identity,
            "simulator_corpus_identity_sha256": corpus_manifest["identity_sha256"],
        },
        "counters": {
            "accepted_optimizer_steps": training["selected_accepted_step"],
            "proposed_optimizer_steps": training["proposed_optimizer_steps"],
            "source_iteration": bootstrap_identity["iteration"],
            "source_policy_version": bootstrap_identity["policy_version"],
        },
        "selected_validation": selected,
        "final_test": {"baseline": test_baseline, "selected": test_final},
        "simulator_test_retention": simulator_test,
        "resumability": {
            "human_bc_resumable": True,
            "ppo_resumable": False,
            "ppo_requires_explicit_new_transition_authority": True,
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_identity = {
        "path": config["checkpoint"]["path"],
        "bytes": checkpoint_path.stat().st_size,
        "sha256": file_sha256(checkpoint_path),
        "model_tensor_sha256": tensor_tree_sha256(selected_state),
    }

    source_after = _verify_human_sources(args.human_source_root)
    integrity = {
        "bootstrap_checkpoint_sha256_before_after": [
            _sha256_bytes(bootstrap_bytes_before),
            file_sha256(ROOT / config["authority"]["bootstrap_checkpoint"]),
        ],
        "adapter_checkpoint_sha256_before_after": [
            _sha256_bytes(adapter_bytes_before),
            file_sha256(ROOT / config["authority"]["adapter_checkpoint"]),
        ],
        "adapter_tensor_sha256_before_after": [
            config["authority"]["adapter_tensor_sha256"],
            tensor_tree_sha256(adapter.state_dict()),
        ],
        "historical_ppo_optimizer_sha256_before_after": [
            historical_optimizer_before,
            tensor_tree_sha256(bootstrap_payload["optimizer"]),
        ],
        "source_hashes_unchanged": source_before == source_after,
        "student_initialized_byte_identically": initial_parity,
        "teacher_requires_grad_false": all(
            not parameter.requires_grad for parameter in teacher.parameters()
        ),
        "teacher_gradients_absent": all(
            parameter.grad is None for parameter in teacher.parameters()
        ),
        "adapter_requires_grad_false": all(
            not parameter.requires_grad for parameter in adapter.parameters()
        ),
        "adapter_gradients_absent": all(
            parameter.grad is None for parameter in adapter.parameters()
        ),
        "test_access_before_selection": test_access_before_selection,
        "test_access_count": 1,
        "test_access_utc": test_access_utc,
        "ppo_updates": 0,
        "reward_changes": 0,
        "mechanic_detector_changes": 0,
        "native_recording_mutations": 0,
    }
    integrity["valid"] = bool(
        integrity["bootstrap_checkpoint_sha256_before_after"][0]
        == integrity["bootstrap_checkpoint_sha256_before_after"][1]
        == config["authority"]["bootstrap_checkpoint_sha256"]
        and integrity["adapter_checkpoint_sha256_before_after"][0]
        == integrity["adapter_checkpoint_sha256_before_after"][1]
        == config["authority"]["adapter_checkpoint_sha256"]
        and integrity["adapter_tensor_sha256_before_after"][0]
        == integrity["adapter_tensor_sha256_before_after"][1]
        and integrity["historical_ppo_optimizer_sha256_before_after"][0]
        == integrity["historical_ppo_optimizer_sha256_before_after"][1]
        and integrity["source_hashes_unchanged"]
        and not integrity["test_access_before_selection"]
        and integrity["test_access_count"] == 1
        and integrity["ppo_updates"] == 0
        and integrity["reward_changes"] == 0
        and integrity["mechanic_detector_changes"] == 0
        and integrity["native_recording_mutations"] == 0
    )
    accepted = accepted and integrity["valid"]
    evidence = {
        "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_EVIDENCE_V1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "verdict": "PASS" if accepted else "BLOCKED",
        "config": config_identity,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
        "preflight": preflight,
        "bootstrap": bootstrap_identity,
        "adapter": adapter_identity,
        "corpus": corpus_manifest,
        "corpus_checks": corpus_checks,
        "human_split_identity": human_split_identity,
        "training": training,
        "final_test": {"baseline": test_baseline, "selected": test_final},
        "simulator_test_retention": simulator_test,
        "simulator_test_guard": simulator_test_guard,
        "closed_loop_mechanic": representability,
        "gameplay_sanity": sanity_comparison,
        "acceptance": {"checks": checks, "accepted": accepted},
        "checkpoint": checkpoint_identity,
        "integrity": integrity,
    }
    _write_json(ROOT / RESULT_ROOT / "pre_step_preflight.json", preflight)
    _write_json(ROOT / RESULT_ROOT / "corpus_manifest.json", corpus_manifest)
    _write_json(ROOT / RESULT_ROOT / "training_curve.json", training)
    _write_json(ROOT / RESULT_ROOT / "final_test_metrics.json", evidence["final_test"])
    _write_json(ROOT / RESULT_ROOT / "simulator_retention_test.json", simulator_test)
    _write_json(ROOT / RESULT_ROOT / "closed_loop_mechanic_evaluation.json", representability)
    _write_json(ROOT / RESULT_ROOT / "gameplay_sanity.json", sanity_comparison)
    _write_json(ROOT / RESULT_ROOT / "evidence.json", evidence)
    artifact_paths = [
        Path(config["checkpoint"]["path"]),
        RESULT_ROOT / "REVIEW.md",
        RESULT_ROOT / "closed_loop_mechanic_evaluation.json",
        RESULT_ROOT / "corpus_manifest.json",
        RESULT_ROOT / "evidence.json",
        RESULT_ROOT / "final_test_metrics.json",
        RESULT_ROOT / "frozen_config.json",
        RESULT_ROOT / "gameplay_sanity.json",
        RESULT_ROOT / "pre_step_authority.json",
        RESULT_ROOT / "pre_step_preflight.json",
        RESULT_ROOT / "simulator_retention_test.json",
        RESULT_ROOT / "training_curve.json",
        RESULT_ROOT / "verification_evidence.json",
    ]
    _write_json(ROOT / RESULT_ROOT / "artifact_manifest.json", _artifact_manifest(artifact_paths))
    print(
        json.dumps(
            {
                "verdict": evidence["verdict"],
                "checkpoint": checkpoint_identity,
                "accepted_steps": training["selected_accepted_step"],
                "validation_gameplay_rmse_ratio": selected["gameplay_rmse_ratio"],
                "validation_mechanic_rmse_ratio": selected["mechanic_rmse_ratio"],
                "simulator_test_actor_mean_kl": simulator_test["actor_mean_kl"],
                "closed_loop_exactly_evaluable_mechanics": 0,
            },
            indent=2,
        )
    )
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--human-source-root",
        type=Path,
        default=Path(os.environ["APPDATA"]) / "bakkesmod/bakkesmod/data/rival2/human_demos",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = run(args)
    return 0 if evidence["verdict"] in ("PASS", "PREFLIGHT_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
