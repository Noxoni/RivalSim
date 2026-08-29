"""Continue Rival human BC from the selected V1 checkpoint under frozen guards."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_human_behavior_cloning_v1 import (  # noqa: E402
    _artifact_manifest,
    _candidate_summary,
    _cpu_tree,
    _dataset_split_audit,
    _evaluate_human,
    _evaluate_retention,
    _guard,
    _load_adapter,
    _load_human_split,
    _verify_authority_files,
    _write_json,
)
from benchmarks.run_rival2_missing_feature_distillation import (  # noqa: E402
    _build_rollout_corpus,
    _load_bootstrap,
    _verify_human_sources,
)
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    HUMAN_BC_VERSION,
    MechanicHierarchySampler,
    human_behavior_cloning_objective,
    simulator_retention_objective,
)
from rivalsim.human_demo.missing_feature_distillation import (  # noqa: E402
    file_sha256,
    world_observation_batch,
)
from rivalsim.human_demo.observation_adapter_v2 import (  # noqa: E402
    AdapterProfile,
    HumanDemoObservationAdapterV2,
)
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_contracts import ACTION_NAMES  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig  # noqa: E402

CONTINUATION_VERSION = "RIVAL2_HUMAN_BC_CONTINUATION_V1"
CONTINUATION_CHECKPOINT_FORMAT = "RIVAL2_HUMAN_BC_CONTINUATION_CHECKPOINT_V1"
FROZEN_CONFIG = Path("results/rival2/human_bc_continuation_v1/frozen_config.json")
FROZEN_CONFIG_SHA256 = "B2A6EFF143D71609E5D42F021A6F5D1AE848C27F5FC3588EFD129F6AF81FA411"
RESULT_ROOT = Path("results/rival2/human_bc_continuation_v1")
WORK_ROOT = Path(".tools/rival2_human_bc_continuation_v1")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _load_continuation_config() -> tuple[dict[str, Any], dict[str, Any]]:
    path = ROOT / FROZEN_CONFIG
    digest = file_sha256(path)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(f"frozen continuation config changed: {digest}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = config["authority"]["required_parent"]
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", required, "HEAD"],
        check=True,
    )
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    if head != origin:
        raise ValueError(f"continuation requires remotely persisted HEAD: {head} != {origin}")
    committed_blob = _git("rev-parse", f"HEAD:{FROZEN_CONFIG.as_posix()}")
    working_blob = _git("hash-object", str(FROZEN_CONFIG))
    if committed_blob != working_blob:
        raise ValueError("working continuation config differs from committed authority")
    return config, {
        "path": FROZEN_CONFIG.as_posix(),
        "sha256": digest,
        "git_blob_oid": committed_blob,
        "pre_step_git_commit": head,
        "origin_main": origin,
        "required_parent": required,
        "required_parent_is_ancestor": True,
    }


def _load_base_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = config["authority"]
    path = ROOT / authority["base_bc_config"]
    evidence_path = ROOT / authority["base_bc_evidence"]
    checks = {
        "base_config_sha256_exact": file_sha256(path) == authority["base_bc_config_sha256"],
        "base_evidence_sha256_exact": file_sha256(evidence_path)
        == authority["base_bc_evidence_sha256"],
        "source_checkpoint_sha256_exact": file_sha256(ROOT / authority["source_checkpoint"])
        == authority["source_checkpoint_sha256"],
        "adapter_checkpoint_sha256_exact": file_sha256(
            ROOT / authority["observation_adapter_checkpoint"]
        )
        == authority["observation_adapter_checkpoint_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"continuation parent authority failed: {checks}")
    return (
        json.loads(path.read_text(encoding="utf-8")),
        json.loads(evidence_path.read_text(encoding="utf-8")),
    )


def _load_source_checkpoint(
    config: dict[str, Any], policy_config: Rival2PolicyConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = config["authority"]
    path = ROOT / authority["source_checkpoint"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "format_exact": payload.get("format") == "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V1",
        "human_bc_resumable": bool(payload["resumability"]["human_bc_resumable"]),
        "ppo_not_resumable": not bool(payload["resumability"]["ppo_resumable"]),
        "policy_config_exact": payload["policy_config"] == asdict(policy_config),
        "source_step_exact": int(payload["counters"]["accepted_optimizer_steps"])
        == int(authority["source_selected_accepted_step"]),
        "model_tensor_exact": tensor_tree_sha256(payload["model"])
        == authority["source_model_tensor_sha256"],
        "fresh_supervised_optimizer_present": "fresh_supervised_optimizer" in payload,
        "historical_ppo_optimizer_absent": not bool(
            payload["optimizer_provenance"]["historical_ppo_optimizer_loaded"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"source human-BC checkpoint failed: {checks}")
    return payload, {
        "path": authority["source_checkpoint"],
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "model_tensor_sha256": tensor_tree_sha256(payload["model"]),
        "accepted_optimizer_steps": int(payload["counters"]["accepted_optimizer_steps"]),
        "proposed_optimizer_steps": int(payload["counters"]["proposed_optimizer_steps"]),
        "checks": checks,
    }


def _distribution_guard(human: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    limits = config["distribution_guard"]
    checks: dict[str, bool] = {"finite": bool(human["finite"])}
    for family, metrics in human["families"].items():
        stats = metrics["actor_output_statistics"]
        for name, row in stats["analog_mean"].items():
            checks[f"{family}.analog.{name}.finite"] = all(
                np.isfinite(float(row[key])) for key in ("mean", "std", "min", "max")
            )
            checks[f"{family}.analog.{name}.not_extreme"] = (
                float(row["absolute_ge_5_fraction"])
                <= limits["maximum_analog_actor_absolute_ge_5_fraction"]
            )
            checks[f"{family}.analog.{name}.nonconstant"] = (
                float(row["std"]) >= limits["minimum_nonconstant_channel_std"]
            )
        for name, row in stats["button_probability"].items():
            checks[f"{family}.button.{name}.not_saturated"] = (
                float(row["saturation_fraction"])
                <= limits["maximum_button_probability_saturation_fraction"]
            )
            checks[f"{family}.button.{name}.nonconstant"] = (
                float(row["std"]) >= limits["minimum_nonconstant_channel_std"]
            )
        for name, row in stats["log_std"].items():
            checks[f"{family}.log_std.{name}.not_clamped"] = (
                max(float(row["at_min_fraction"]), float(row["at_max_fraction"]))
                <= limits["maximum_log_std_clamp_fraction"]
            )
    return {"checks": checks, "accepted": all(checks.values())}


def _transactional_retry_learning_rates(
    interval_start_learning_rate: float,
    *,
    backoff_factor: float,
    retry_count: int,
) -> tuple[float, ...]:
    if interval_start_learning_rate <= 0.0:
        raise ValueError("interval-start learning rate must be positive")
    if not 0.0 < backoff_factor < 1.0:
        raise ValueError("backoff factor must be in (0, 1)")
    if retry_count < 0:
        raise ValueError("retry count must be nonnegative")
    return tuple(
        interval_start_learning_rate * (backoff_factor**retry) for retry in range(retry_count + 1)
    )


def _save_state(
    path: Path,
    *,
    student: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    human_generator: torch.Generator,
    simulator_generator: torch.Generator,
    cumulative_step: int,
    proposed_steps: int,
    candidate: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _cpu_tree(student.state_dict()),
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "human_generator_state": human_generator.get_state().clone(),
            "simulator_generator_state": simulator_generator.get_state().clone(),
            "cumulative_step": cumulative_step,
            "proposed_steps": proposed_steps,
            "candidate": candidate,
        },
        path,
    )


def _restore_state(
    path: Path,
    *,
    student: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    human_generator: torch.Generator,
    simulator_generator: torch.Generator,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    student.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    human_generator.set_state(payload["human_generator_state"])
    simulator_generator.set_state(payload["simulator_generator_state"])
    return payload


def _train_continuation(
    *,
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    adapter: HumanDemoObservationAdapterV2,
    policy_config: Rival2PolicyConfig,
    source_payload: dict[str, Any],
    base_evidence: dict[str, Any],
    train_data: Any,
    validation_data: Any,
    corpus: torch.Tensor,
    train_worlds: np.ndarray,
    validation_worlds: np.ndarray,
    continuation_config: dict[str, Any],
    base_config: dict[str, Any],
    device: str,
) -> tuple[torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    settings = base_config["training"]
    objective = base_config["objective"]
    sampling = base_config["sampling"]
    source_step = int(source_payload["counters"]["accepted_optimizer_steps"])
    seed = int(continuation_config["optimizer"]["continuation_seed"])
    human_generator = torch.Generator(device="cpu").manual_seed(seed)
    simulator_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    mechanic_sampler = MechanicHierarchySampler(
        train_data.mechanic_label,
        train_data.mechanic_attempt,
        uniform_label_fraction=float(sampling["mechanic_uniform_label_fraction"]),
        maximum_oversampling_ratio=float(sampling["maximum_mechanic_frame_oversampling_ratio"]),
        generator=human_generator,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(settings["initial_learning_rate"]),
        betas=tuple(settings["optimizer_betas"]),
        eps=float(settings["optimizer_epsilon"]),
        weight_decay=float(settings["weight_decay"]),
    )
    optimizer.load_state_dict(source_payload["fresh_supervised_optimizer"])
    if len(optimizer.param_groups) != 1:
        raise ValueError("source supervised optimizer parameter groups changed")
    source_optimizer_lr = float(optimizer.param_groups[0]["lr"])
    parent_human = _evaluate_human(student, teacher, validation_data, device=device)
    selected_validation_worlds = validation_worlds[
        : int(base_config["retention"]["validation_subset"]["worlds"])
    ]
    parent_retention = _evaluate_retention(
        teacher,
        student,
        adapter,
        corpus,
        selected_validation_worlds,
        worlds_per_batch=int(base_config["retention"]["worlds_per_validation_batch"]),
        policy_config=policy_config,
    )
    parent_guard = _guard(parent_retention, base_config)
    parent_distribution = _distribution_guard(parent_human, continuation_config)
    frozen_bootstrap_human = base_evidence["training"]["baseline"]["human_validation"]
    parent_candidate = _candidate_summary(
        step=source_step,
        human=parent_human,
        baseline=frozen_bootstrap_human,
        retention=parent_retention,
        guard=parent_guard,
        config=base_config,
    )
    parent_candidate["distribution_guard"] = parent_distribution
    parent_candidate["eligible_for_selection"] = bool(
        parent_candidate["eligible_for_selection"] and parent_distribution["accepted"]
    )
    recorded_parent = source_payload["selected_validation"]
    parent_reproduction = {
        "gameplay_complete_action_rmse_absolute_error": abs(
            parent_human["families"]["gameplay"]["complete_action_rmse"]
            - recorded_parent["human_validation"]["families"]["gameplay"]["complete_action_rmse"]
        ),
        "mechanic_complete_action_rmse_absolute_error": abs(
            parent_human["families"]["mechanic"]["complete_action_rmse"]
            - recorded_parent["human_validation"]["families"]["mechanic"]["complete_action_rmse"]
        ),
        "simulator_actor_mean_kl_absolute_error": abs(
            parent_retention["actor_mean_kl"]
            - recorded_parent["simulator_retention"]["actor_mean_kl"]
        ),
    }
    parent_reproduction["exact_within_tolerance"] = max(parent_reproduction.values()) <= 1e-7
    if (
        not parent_candidate["eligible_for_selection"]
        or not parent_reproduction["exact_within_tolerance"]
    ):
        raise RuntimeError(
            "source checkpoint failed continuation baseline reproduction: "
            f"candidate={parent_candidate}, reproduction={parent_reproduction}"
        )

    best_score = float(parent_candidate["selection_score"])
    best_candidate = copy.deepcopy(parent_candidate)
    best_path = ROOT / WORK_ROOT / "best.pt"
    proposed_steps = 0
    cumulative_step = source_step
    additional_accepted_steps = 0
    no_improvement = 0
    curve: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    stop_reason = ""
    _save_state(
        best_path,
        student=student,
        optimizer=optimizer,
        human_generator=human_generator,
        simulator_generator=simulator_generator,
        cumulative_step=cumulative_step,
        proposed_steps=proposed_steps,
        candidate=parent_candidate,
    )
    interval = int(settings["validation_interval_optimizer_steps"])
    emergency = int(continuation_config["continuation"]["emergency_max_additional_accepted_steps"])
    retry_limit = int(continuation_config["optimizer"]["transactional_retries_per_interval"])
    minimum_lr = float(continuation_config["optimizer"]["minimum_learning_rate"])
    backoff = float(continuation_config["optimizer"]["transactional_backoff_factor"])
    patience = int(continuation_config["selection"]["early_stopping_patience_validations"])
    material_delta = float(
        continuation_config["selection"]["early_stopping_material_score_improvement"]
    )

    while additional_accepted_steps < emergency:
        rollback_path = ROOT / WORK_ROOT / "rollback.pt"
        _save_state(
            rollback_path,
            student=student,
            optimizer=optimizer,
            human_generator=human_generator,
            simulator_generator=simulator_generator,
            cumulative_step=cumulative_step,
            proposed_steps=proposed_steps,
            candidate={},
        )
        interval_start_lr = float(optimizer.param_groups[0]["lr"])
        accepted_boundary = False
        retry_learning_rates = _transactional_retry_learning_rates(
            interval_start_lr,
            backoff_factor=backoff,
            retry_count=retry_limit,
        )
        for retry, retry_lr in enumerate(retry_learning_rates):
            if retry:
                _restore_state(
                    rollback_path,
                    student=student,
                    optimizer=optimizer,
                    human_generator=human_generator,
                    simulator_generator=simulator_generator,
                )
            if retry_lr < minimum_lr - 1e-15:
                attempts.append(
                    {
                        "attempted_cumulative_step": cumulative_step + interval,
                        "retry": retry,
                        "learning_rate": retry_lr,
                        "executed": False,
                        "reason": "retry learning rate below frozen minimum",
                    }
                )
                break
            for group in optimizer.param_groups:
                group["lr"] = retry_lr
            loss_sums: defaultdict[str, float] = defaultdict(float)
            grad_norm_max = 0.0
            for _ in range(interval):
                proposed_steps += 1
                gameplay_indices = torch.randint(
                    train_data.gameplay_observation.shape[0],
                    (int(sampling["gameplay_frames_per_step"]),),
                    generator=human_generator,
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
                positions = torch.randint(
                    len(train_worlds),
                    (int(base_config["retention"]["worlds_per_training_minibatch"]),),
                    generator=simulator_generator,
                ).numpy()
                sim_observation = world_observation_batch(corpus, train_worlds[positions])
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
                    raise RuntimeError("nonfinite continuation objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), float(settings["gradient_clip_norm"])
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise RuntimeError("nonfinite continuation gradient norm")
                optimizer.step()
                if not all(
                    bool(torch.isfinite(parameter).all()) for parameter in student.parameters()
                ):
                    raise RuntimeError("nonfinite continuation parameter")
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
                worlds_per_batch=int(base_config["retention"]["worlds_per_validation_batch"]),
                policy_config=policy_config,
            )
            retention_guard = _guard(candidate_retention, base_config)
            distribution_guard = _distribution_guard(candidate_human, continuation_config)
            attempted_step = cumulative_step + interval
            attempt = {
                "attempted_cumulative_step": attempted_step,
                "retry": retry,
                "learning_rate": retry_lr,
                "executed": True,
                "proposed_optimizer_steps": proposed_steps,
                "mean_training_loss": {key: value / interval for key, value in loss_sums.items()},
                "max_preclip_gradient_norm": grad_norm_max,
                "simulator_retention": candidate_retention,
                "retention_guard": retention_guard,
                "distribution_guard": distribution_guard,
            }
            attempts.append(attempt)
            if retention_guard["accepted"] and distribution_guard["accepted"]:
                cumulative_step = attempted_step
                additional_accepted_steps += interval
                candidate = _candidate_summary(
                    step=cumulative_step,
                    human=candidate_human,
                    baseline=frozen_bootstrap_human,
                    retention=candidate_retention,
                    guard=retention_guard,
                    config=base_config,
                )
                parent_gameplay = parent_human["families"]["gameplay"]["complete_action_rmse"]
                parent_mechanic = parent_human["families"]["mechanic"]["complete_action_rmse"]
                candidate["distribution_guard"] = distribution_guard
                candidate["gameplay_rmse_ratio_to_parent"] = (
                    candidate_human["families"]["gameplay"]["complete_action_rmse"]
                    / parent_gameplay
                )
                candidate["mechanic_rmse_ratio_to_parent"] = (
                    candidate_human["families"]["mechanic"]["complete_action_rmse"]
                    / parent_mechanic
                )
                candidate["eligible_for_selection"] = bool(
                    candidate["eligible_for_selection"]
                    and distribution_guard["accepted"]
                    and candidate["gameplay_rmse_ratio_to_parent"] < 1.0
                    and candidate["mechanic_rmse_ratio_to_parent"] < 1.0
                )
                candidate["learning_rate"] = retry_lr
                candidate["training_loss"] = attempt["mean_training_loss"]
                candidate["max_preclip_gradient_norm"] = grad_norm_max
                candidate["soft_retention_limit_exceeded"] = (
                    candidate_retention["actor_mean_kl"]
                    > base_config["retention"]["soft_actor_mean_kl"]
                )
                curve.append(candidate)
                boundary_path = ROOT / WORK_ROOT / "accepted" / f"step-{cumulative_step:04d}.pt"
                _save_state(
                    boundary_path,
                    student=student,
                    optimizer=optimizer,
                    human_generator=human_generator,
                    simulator_generator=simulator_generator,
                    cumulative_step=cumulative_step,
                    proposed_steps=proposed_steps,
                    candidate=candidate,
                )
                if (
                    candidate["eligible_for_selection"]
                    and candidate["selection_score"] < best_score - material_delta
                ):
                    best_score = float(candidate["selection_score"])
                    best_candidate = copy.deepcopy(candidate)
                    _save_state(
                        best_path,
                        student=student,
                        optimizer=optimizer,
                        human_generator=human_generator,
                        simulator_generator=simulator_generator,
                        cumulative_step=cumulative_step,
                        proposed_steps=proposed_steps,
                        candidate=candidate,
                    )
                    no_improvement = 0
                else:
                    no_improvement += 1
                accepted_boundary = True
                break
        if not accepted_boundary:
            stop_reason = "frozen retention/distribution guard rejected further progress"
            break
        if no_improvement >= patience:
            stop_reason = "held-out validation improvement plateaued"
            break
    if not stop_reason:
        stop_reason = "emergency accepted-step ceiling reached while not converged"

    selected_state = _restore_state(
        best_path,
        student=student,
        optimizer=optimizer,
        human_generator=human_generator,
        simulator_generator=simulator_generator,
    )
    selected_new_checkpoint = int(selected_state["cumulative_step"]) > source_step
    result = {
        "version": CONTINUATION_VERSION,
        "source_checkpoint_step": source_step,
        "source_optimizer_learning_rate": source_optimizer_lr,
        "source_checkpoint_rng_state_available": False,
        "continuation_seed": seed,
        "fresh_supervised_optimizer_resumed": True,
        "historical_ppo_optimizer_loaded": False,
        "parent_validation": parent_candidate,
        "parent_reproduction": parent_reproduction,
        "curve": curve,
        "interval_attempts": attempts,
        "additional_accepted_optimizer_steps_executed": additional_accepted_steps,
        "additional_proposed_optimizer_steps": proposed_steps,
        "selected_cumulative_accepted_step": int(selected_state["cumulative_step"]),
        "selected_additional_accepted_steps": int(selected_state["cumulative_step"]) - source_step,
        "selected_candidate": best_candidate,
        "selected_new_checkpoint": selected_new_checkpoint,
        "stop_reason": stop_reason,
        "plateaued": stop_reason == "held-out validation improvement plateaued",
        "guard_stopped": stop_reason.startswith("frozen retention"),
        "emergency_ceiling_reached": stop_reason.startswith("emergency"),
        "mechanic_sampling": {
            "uniform_label_fraction": mechanic_sampler.uniform_label_fraction,
            "maximum_realized_oversampling_ratio": (
                mechanic_sampler.maximum_realized_oversampling_ratio
            ),
            "frozen_cap": mechanic_sampler.maximum_oversampling_ratio,
            "label_count": len(mechanic_sampler.labels),
        },
        "selected_rng_state": {
            "human_generator_sha256": tensor_tree_sha256(selected_state["human_generator_state"]),
            "simulator_generator_sha256": tensor_tree_sha256(
                selected_state["simulator_generator_state"]
            ),
        },
    }
    return optimizer, result, selected_state


def _preflight(
    continuation_config: dict[str, Any],
    config_identity: dict[str, Any],
    base_config: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    base_authority = _verify_authority_files(base_config)
    split_audit = _dataset_split_audit(base_config)
    sources = _verify_human_sources(source_root)
    result = {
        "format": "RIVAL2_HUMAN_BC_CONTINUATION_PRE_STEP_PREFLIGHT_V1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_identity,
        "continuation_parent": {
            key: continuation_config["authority"][key]
            for key in (
                "source_checkpoint",
                "source_checkpoint_sha256",
                "source_model_tensor_sha256",
                "source_selected_accepted_step",
            )
        },
        "base_authority": base_authority,
        "split_audit": split_audit,
        "source_verification": sources,
        "optimizer_steps": 0,
        "ppo_updates": 0,
        "valid": True,
    }
    _write_json(ROOT / WORK_ROOT / "pre_step_preflight.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    continuation_config, config_identity = _load_continuation_config()
    base_config, base_evidence = _load_base_config(continuation_config)
    preflight = _preflight(
        continuation_config, config_identity, base_config, args.human_source_root
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return {"verdict": "PREFLIGHT_PASS"}

    started = time.perf_counter()
    seed = int(continuation_config["optimizer"]["continuation_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    source_checkpoint_path = ROOT / continuation_config["authority"]["source_checkpoint"]
    source_checkpoint_bytes_before = source_checkpoint_path.read_bytes()
    adapter_path = ROOT / continuation_config["authority"]["observation_adapter_checkpoint"]
    adapter_bytes_before = adapter_path.read_bytes()
    source_before = _verify_human_sources(args.human_source_root)
    compatibility = {
        "authority": base_config["authority"],
        "corpus": base_config["retention"]["corpus"],
    }
    bootstrap_payload, policy_config, bootstrap_identity = _load_bootstrap(compatibility)
    historical_optimizer_before = tensor_tree_sha256(bootstrap_payload["optimizer"])
    source_payload, source_identity = _load_source_checkpoint(continuation_config, policy_config)
    adapter, _adapter_payload, adapter_identity = _load_adapter(base_config, args.device)
    train_data = _load_human_split(
        "train",
        config=base_config,
        adapter=adapter,
        source_root=args.human_source_root,
        device=args.device,
    )
    validation_data = _load_human_split(
        "validation",
        config=base_config,
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
        },
        "validation": {
            "gameplay_frames": int(validation_data.gameplay_action.shape[0]),
            "mechanic_frames": int(validation_data.mechanic_action.shape[0]),
            "mechanic_attempts": len(set(validation_data.mechanic_attempt)),
            "action_sha256": validation_data.action_sha256,
            "source_sequences_sha256": validation_data.source_sequences_sha256,
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
        == base_config["retention"]["corpus"]["expected_identity_sha256"],
        "observation_hash_exact": corpus_manifest["collection"]["observation_tensor_sha256"]
        == base_config["retention"]["corpus"]["expected_observation_tensor_sha256"],
        "test_worlds_not_used_for_training_or_selection": True,
    }
    if not all(corpus_checks.values()):
        raise RuntimeError(f"authoritative simulator corpus changed: {corpus_checks}")
    torch.use_deterministic_algorithms(True)
    teacher = Rival2ActorCritic(policy_config).to(args.device)
    teacher.load_state_dict(bootstrap_payload["model"])
    teacher.eval().requires_grad_(False)
    parent = Rival2ActorCritic(policy_config).to(args.device)
    parent.load_state_dict(source_payload["model"])
    parent.eval().requires_grad_(False)
    student = Rival2ActorCritic(policy_config).to(args.device)
    student.load_state_dict(source_payload["model"])
    student.train()
    if tensor_tree_sha256(student.state_dict()) != source_identity["model_tensor_sha256"]:
        raise RuntimeError("continuation student is not byte-identical to selected V1 parent")

    optimizer, training, selected_state = _train_continuation(
        teacher=teacher,
        student=student,
        adapter=adapter,
        policy_config=policy_config,
        source_payload=source_payload,
        base_evidence=base_evidence,
        train_data=train_data,
        validation_data=validation_data,
        corpus=corpus,
        train_worlds=splits["train"],
        validation_worlds=splits["validation"],
        continuation_config=continuation_config,
        base_config=base_config,
        device=args.device,
    )

    test_access_utc = datetime.now(UTC).isoformat()
    test_data = _load_human_split(
        "test",
        config=base_config,
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
    }
    test_parent = _evaluate_human(parent, teacher, test_data, device=args.device)
    test_selected = _evaluate_human(student, teacher, test_data, device=args.device)
    simulator_test = _evaluate_retention(
        teacher,
        student,
        adapter,
        corpus,
        splits["test"],
        worlds_per_batch=int(base_config["retention"]["worlds_per_validation_batch"]),
        policy_config=policy_config,
    )
    simulator_test_guard = _guard(simulator_test, base_config)
    selected_distribution = _distribution_guard(
        training["selected_candidate"]["human_validation"], continuation_config
    )

    selected_model = _cpu_tree(student.state_dict())
    selected_optimizer = _cpu_tree(optimizer.state_dict())
    selected_additional = int(training["selected_additional_accepted_steps"])
    checks = {
        "selected_new_material_checkpoint": bool(training["selected_new_checkpoint"]),
        "selected_additional_steps_positive": selected_additional > 0,
        "selected_validation_retention_safe": training["selected_candidate"]["retention_guard"][
            "accepted"
        ],
        "selected_distribution_healthy": selected_distribution["accepted"],
        "simulator_test_retention_safe": simulator_test_guard["accepted"],
        "gameplay_validation_improved_over_parent": training["selected_candidate"].get(
            "gameplay_rmse_ratio_to_parent", 1.0
        )
        < 1.0,
        "mechanic_validation_improved_over_parent": training["selected_candidate"].get(
            "mechanic_rmse_ratio_to_parent", 1.0
        )
        < 1.0,
        "not_emergency_ceiling": not training["emergency_ceiling_reached"],
        "all_outputs_finite": bool(test_parent["finite"] and test_selected["finite"]),
    }
    accepted = all(checks.values())
    checkpoint_path = ROOT / continuation_config["checkpoint"]["path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "format": CONTINUATION_CHECKPOINT_FORMAT,
        "version": CONTINUATION_VERSION,
        "base_human_bc_version": HUMAN_BC_VERSION,
        "model": selected_model,
        "fresh_supervised_optimizer": selected_optimizer,
        "optimizer_provenance": {
            "fresh_for_human_bc": True,
            "resumed_from_source_checkpoint": source_identity,
            "historical_ppo_optimizer_loaded": False,
            "historical_ppo_optimizer_resumable": False,
        },
        "policy_config": asdict(policy_config),
        "observation_version": source_payload["observation_version"],
        "action_version": source_payload["action_version"],
        "physics_hz": 120,
        "policy_hz": 120,
        "authority": {
            "continuation_config": config_identity,
            "source_checkpoint": source_identity,
            "bootstrap": bootstrap_identity,
            "adapter": adapter_identity,
            "human_split_identity": human_split_identity,
            "simulator_corpus_identity_sha256": corpus_manifest["identity_sha256"],
        },
        "counters": {
            "source_accepted_optimizer_steps": source_identity["accepted_optimizer_steps"],
            "additional_accepted_optimizer_steps": selected_additional,
            "cumulative_accepted_optimizer_steps": training["selected_cumulative_accepted_step"],
            "continuation_proposed_optimizer_steps": training[
                "additional_proposed_optimizer_steps"
            ],
        },
        "rng_state": {
            "human_generator": selected_state["human_generator_state"],
            "simulator_generator": selected_state["simulator_generator_state"],
        },
        "selected_validation": training["selected_candidate"],
        "final_test": {"parent": test_parent, "selected": test_selected},
        "simulator_test_retention": simulator_test,
        "stop": {
            "reason": training["stop_reason"],
            "plateaued": training["plateaued"],
            "guard_stopped": training["guard_stopped"],
        },
        "resumability": {
            "human_bc_resumable": True,
            "ppo_resumable": False,
            "ppo_requires_explicit_new_transition_authority": True,
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_identity = {
        "path": continuation_config["checkpoint"]["path"],
        "bytes": checkpoint_path.stat().st_size,
        "sha256": file_sha256(checkpoint_path),
        "model_tensor_sha256": tensor_tree_sha256(selected_model),
    }
    source_after = _verify_human_sources(args.human_source_root)
    integrity = {
        "source_checkpoint_sha256_before_after": [
            hashlib.sha256(source_checkpoint_bytes_before).hexdigest().upper(),
            file_sha256(source_checkpoint_path),
        ],
        "adapter_checkpoint_sha256_before_after": [
            hashlib.sha256(adapter_bytes_before).hexdigest().upper(),
            file_sha256(adapter_path),
        ],
        "historical_ppo_optimizer_sha256_before_after": [
            historical_optimizer_before,
            tensor_tree_sha256(bootstrap_payload["optimizer"]),
        ],
        "native_source_hashes_unchanged": source_before == source_after,
        "teacher_frozen": all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in teacher.parameters()
        ),
        "parent_frozen": all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in parent.parameters()
        ),
        "adapter_frozen": all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in adapter.parameters()
        ),
        "test_access_before_selection": test_access_before_selection,
        "test_access_count_this_task": 1,
        "test_access_utc": test_access_utc,
        "historical_v1_test_access_acknowledged": True,
        "closed_loop_mechanic_evaluation": False,
        "rocket_league_to_rivalsim_reconstruction": False,
        "ppo_updates": 0,
        "reward_changes": 0,
        "mechanic_definition_changes": 0,
        "raw_recording_mutations": 0,
    }
    integrity["valid"] = bool(
        integrity["source_checkpoint_sha256_before_after"][0]
        == integrity["source_checkpoint_sha256_before_after"][1]
        == continuation_config["authority"]["source_checkpoint_sha256"]
        and integrity["adapter_checkpoint_sha256_before_after"][0]
        == integrity["adapter_checkpoint_sha256_before_after"][1]
        == continuation_config["authority"]["observation_adapter_checkpoint_sha256"]
        and integrity["historical_ppo_optimizer_sha256_before_after"][0]
        == integrity["historical_ppo_optimizer_sha256_before_after"][1]
        and integrity["native_source_hashes_unchanged"]
        and integrity["teacher_frozen"]
        and integrity["parent_frozen"]
        and integrity["adapter_frozen"]
        and not integrity["test_access_before_selection"]
        and integrity["test_access_count_this_task"] == 1
        and not integrity["closed_loop_mechanic_evaluation"]
        and not integrity["rocket_league_to_rivalsim_reconstruction"]
        and integrity["ppo_updates"] == 0
        and integrity["reward_changes"] == 0
        and integrity["mechanic_definition_changes"] == 0
        and integrity["raw_recording_mutations"] == 0
    )
    accepted = accepted and integrity["valid"]
    evidence = {
        "format": "RIVAL2_HUMAN_BC_CONTINUATION_EVIDENCE_V1",
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
        "source_checkpoint": source_identity,
        "bootstrap": bootstrap_identity,
        "adapter": adapter_identity,
        "human_split_identity": human_split_identity,
        "corpus": corpus_manifest,
        "corpus_checks": corpus_checks,
        "training": training,
        "final_test": {"parent": test_parent, "selected": test_selected},
        "simulator_test_retention": simulator_test,
        "simulator_test_guard": simulator_test_guard,
        "acceptance": {"checks": checks, "accepted": accepted},
        "checkpoint": checkpoint_identity,
        "integrity": integrity,
        "prohibited_work": {
            "closed_loop_mechanic_framework": False,
            "rocket_league_to_rivalsim_mechanic_reconstruction": False,
            "ppo": False,
            "reward_change": False,
            "mechanic_definition_change": False,
            "raw_recording_mutation": False,
            "observation_action_contract_change": False,
        },
    }
    _write_json(ROOT / RESULT_ROOT / "pre_step_preflight.json", preflight)
    _write_json(ROOT / RESULT_ROOT / "corpus_manifest.json", corpus_manifest)
    _write_json(ROOT / RESULT_ROOT / "training_curve.json", training)
    _write_json(ROOT / RESULT_ROOT / "final_test_metrics.json", evidence["final_test"])
    _write_json(ROOT / RESULT_ROOT / "simulator_retention_test.json", simulator_test)
    _write_json(ROOT / RESULT_ROOT / "evidence.json", evidence)
    artifact_paths = [
        Path(continuation_config["checkpoint"]["path"]),
        RESULT_ROOT / "README.md",
        RESULT_ROOT / "corpus_manifest.json",
        RESULT_ROOT / "evidence.json",
        RESULT_ROOT / "final_test_metrics.json",
        RESULT_ROOT / "frozen_config.json",
        RESULT_ROOT / "pre_step_authority.json",
        RESULT_ROOT / "pre_step_preflight.json",
        RESULT_ROOT / "simulator_retention_test.json",
        RESULT_ROOT / "training_curve.json",
    ]
    _write_json(ROOT / RESULT_ROOT / "artifact_manifest.json", _artifact_manifest(artifact_paths))
    print(
        json.dumps(
            {
                "verdict": evidence["verdict"],
                "checkpoint": checkpoint_identity,
                "selected_additional_steps": selected_additional,
                "selected_cumulative_step": training["selected_cumulative_accepted_step"],
                "gameplay_rmse_ratio_to_parent": training["selected_candidate"].get(
                    "gameplay_rmse_ratio_to_parent"
                ),
                "mechanic_rmse_ratio_to_parent": training["selected_candidate"].get(
                    "mechanic_rmse_ratio_to_parent"
                ),
                "simulator_test_actor_mean_kl": simulator_test["actor_mean_kl"],
                "stop_reason": training["stop_reason"],
            },
            indent=2,
        )
    )
    del corpus, train_data, validation_data, test_data
    gc.collect()
    torch.cuda.empty_cache()
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
