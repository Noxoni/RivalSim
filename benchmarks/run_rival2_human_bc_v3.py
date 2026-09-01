"""Train actor-only Rival Human BC V3 under prospective tail-safe retention."""

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

import benchmarks.run_rival2_human_bc_continuation_v1 as continuation  # noqa: E402
from benchmarks.run_rival2_human_behavior_cloning_v1 import (  # noqa: E402
    _artifact_manifest,
    _cpu_tree,
    _dataset_split_audit,
    _evaluate_human,
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
from rivalsim.human_demo.bc_v3_retention import (  # noqa: E402
    build_retention_pools,
    detailed_retention_guard,
    evaluate_detailed_retention,
    gather_encoded_rows,
    int64_sha256,
    sample_retention_rows,
    tail_aware_actor_retention_loss,
    verify_retention_pools,
)
from rivalsim.human_demo.behavior_cloning import (  # noqa: E402
    HUMAN_BC_VERSION,
    MechanicHierarchySampler,
    human_behavior_cloning_objective,
)
from rivalsim.human_demo.missing_feature_distillation import file_sha256  # noqa: E402
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_contracts import ACTION_NAMES  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig  # noqa: E402

V3_VERSION = "RIVAL2_HUMAN_BEHAVIOR_CLONING_V3"
V3_CHECKPOINT_FORMAT = "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V3"
FROZEN_CONFIG = Path("results/rival2/human_bc_v3/frozen_config.json")
FROZEN_CONFIG_SHA256 = "E82AFD2E8C34CB43792FD4DE01066692720C38E96A03BA6DB71953FAD7504450"
RESULT_ROOT = Path("results/rival2/human_bc_v3")
WORK_ROOT = Path(".tools/rival2_human_bc_v3")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    path = ROOT / FROZEN_CONFIG
    digest = file_sha256(path)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(f"frozen V3 config changed: {digest}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = config["authority"]["required_parent"]
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", required, "HEAD"],
        check=True,
    )
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    if head != origin:
        raise ValueError(f"V3 training requires remotely persisted HEAD: {head} != {origin}")
    committed = _git("rev-parse", f"HEAD:{FROZEN_CONFIG.as_posix()}")
    working = _git("hash-object", str(FROZEN_CONFIG))
    if committed != working:
        raise ValueError("working V3 config differs from committed authority")
    return config, {
        "path": FROZEN_CONFIG.as_posix(),
        "sha256": digest,
        "git_blob_oid": committed,
        "pre_step_git_commit": head,
        "origin_main": origin,
        "required_parent": required,
        "required_parent_is_ancestor": True,
    }


def _load_manifest(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = Path(path)
    absolute = ROOT / relative
    value = json.loads(absolute.read_text(encoding="utf-8"))
    return value, {
        "path": relative.as_posix(),
        "sha256": file_sha256(absolute),
        "git_blob_oid": _git("hash-object", str(relative)),
    }


def _human_identity(data: Any) -> dict[str, Any]:
    return {
        "gameplay_frames": int(data.gameplay_action.shape[0]),
        "mechanic_frames": int(data.mechanic_action.shape[0]),
        "mechanic_attempts": len(set(data.mechanic_attempt)),
        "action_sha256": data.action_sha256,
        "source_sequences_sha256": data.source_sequences_sha256,
    }


def _selection_score(
    human: dict[str, Any],
    parent_human: dict[str, Any],
    label_comparison: dict[str, Any],
    retention: dict[str, Any],
    config: dict[str, Any],
) -> float:
    weights = config["selection"]["score_weights"]
    gameplay_ratio = (
        human["families"]["gameplay"]["complete_action_rmse"]
        / parent_human["families"]["gameplay"]["complete_action_rmse"]
    )
    mechanic_ratio = (
        human["families"]["mechanic"]["complete_action_rmse"]
        / parent_human["families"]["mechanic"]["complete_action_rmse"]
    )
    all_rows = retention["all_perspectives"]
    return float(
        weights["gameplay_rmse_ratio"] * gameplay_ratio
        + weights["mechanic_rmse_ratio"] * mechanic_ratio
        + weights["mean_per_label_rmse_ratio"] * label_comparison["mean_rmse_ratio_to_parent"]
        + weights["simulator_actor_kl_soft_limit_ratio"]
        * all_rows["mean_kl"]
        / config["validation"]["soft_actor_mean_kl"]
        + weights["simulator_max_kl_hard_limit_ratio"]
        * all_rows["max_sample_kl"]
        / config["validation"]["hard_guard"]["actor_max_sample_kl"]
    )


def _candidate(
    *,
    additional_step: int,
    cumulative_step: int,
    human: dict[str, Any],
    parent_human: dict[str, Any],
    retention: dict[str, Any],
    guard: dict[str, Any],
    distribution: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    labels = continuation._mechanic_label_comparison(
        parent_human,
        human,
        nonregression_relative_tolerance=float(
            config["selection"]["mechanic_label_nonregression_relative_tolerance"]
        ),
    )
    gameplay_ratio = (
        human["families"]["gameplay"]["complete_action_rmse"]
        / parent_human["families"]["gameplay"]["complete_action_rmse"]
    )
    mechanic_ratio = (
        human["families"]["mechanic"]["complete_action_rmse"]
        / parent_human["families"]["mechanic"]["complete_action_rmse"]
    )
    minimum = float(config["selection"]["minimum_human_family_relative_improvement"])
    eligible = bool(
        additional_step > 0
        and gameplay_ratio <= 1.0 - minimum
        and mechanic_ratio <= 1.0 - minimum
        and labels["improved_fraction"]
        >= config["selection"]["minimum_mechanic_labels_improved_fraction"]
        and labels["nonregressed_fraction"]
        >= config["selection"]["minimum_mechanic_labels_nonregressed_fraction"]
        and guard["accepted"]
        and distribution["accepted"]
    )
    return {
        "additional_accepted_step": additional_step,
        "cumulative_accepted_step": cumulative_step,
        "human_validation": human,
        "gameplay_rmse_ratio_to_parent": gameplay_ratio,
        "mechanic_rmse_ratio_to_parent": mechanic_ratio,
        "mechanic_label_comparison_to_parent": labels,
        "simulator_retention": retention,
        "retention_guard": guard,
        "distribution_guard": distribution,
        "selection_score": _selection_score(human, parent_human, labels, retention, config),
        "eligible_for_selection": eligible,
    }


def _preflight(
    config: dict[str, Any],
    config_identity: dict[str, Any],
    base_config: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    strata, strata_identity = _load_manifest(
        config["simulator_authority"]["retention_strata_manifest"]
    )
    new_test, new_test_identity = _load_manifest(
        config["simulator_authority"]["new_untouched_test_authority"]
    )
    checks = {
        "source_checkpoint_exact": file_sha256(ROOT / config["authority"]["source_checkpoint"])
        == config["authority"]["source_checkpoint_sha256"],
        "adapter_exact": file_sha256(ROOT / config["authority"]["observation_adapter_checkpoint"])
        == config["authority"]["observation_adapter_checkpoint_sha256"],
        "v2_diagnostic_only": bool(config["authority"]["v2_checkpoint_training_prohibited"]),
        "strata_teacher_only": not bool(strata["student_outputs_used_for_membership"]),
        "new_test_unopened": int(new_test["student_evaluations_before_final_selection"]) == 0,
        "new_test_seed_disjoint": bool(new_test["disjointness"]["seed_namespace_distinct"]),
        "complete_validation": int(config["validation"]["worlds"]) == 3277,
        "hard_max_unchanged": float(config["validation"]["hard_guard"]["actor_max_sample_kl"])
        == 2.0,
    }
    authority_files = _verify_authority_files(base_config)
    split_audit = _dataset_split_audit(base_config)
    human_sources = _verify_human_sources(source_root)
    if not all(checks.values()):
        raise RuntimeError(f"V3 preflight failed: {checks}")
    return {
        "format": "RIVAL2_HUMAN_BC_V3_RUNTIME_PRE_STEP_PREFLIGHT_V1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": config_identity,
        "checks": checks,
        "strata_manifest": strata_identity,
        "new_test_authority": new_test_identity,
        "base_authority_files": authority_files,
        "human_split_audit": split_audit,
        "human_source_verification": human_sources,
        "optimizer_steps": 0,
        "human_test_accesses": 0,
        "new_simulator_test_student_evaluations": 0,
        "old_opened_test_candidate_evaluations": 0,
        "ppo_updates": 0,
        "reward_changes": 0,
        "valid": True,
    }


def _train(
    *,
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    source_payload: dict[str, Any],
    policy_config: Rival2PolicyConfig,
    train_data: Any,
    validation_data: Any,
    corpus: torch.Tensor,
    train_worlds: np.ndarray,
    validation_worlds: np.ndarray,
    opponent_family: torch.Tensor,
    rival_side: torch.Tensor,
    pools: Any,
    config: dict[str, Any],
    base_config: dict[str, Any],
    device: str,
) -> tuple[torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    settings = config["training"]
    base_objective = base_config["objective"]
    sampling = base_config["sampling"]
    seed = int(config["optimizer"]["continuation_seed"])
    human_generator = torch.Generator(device="cpu").manual_seed(seed)
    simulator_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    mechanic_sampler = MechanicHierarchySampler(
        train_data.mechanic_label,
        train_data.mechanic_attempt,
        uniform_label_fraction=float(sampling["mechanic_uniform_label_fraction"]),
        maximum_oversampling_ratio=float(sampling["maximum_mechanic_frame_oversampling_ratio"]),
        generator=human_generator,
    )
    trainable, names = continuation._configure_trainable_parameters(student, config)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["initial_learning_rate"]),
        betas=tuple(settings["optimizer_betas"]),
        eps=float(settings["optimizer_epsilon"]),
        weight_decay=float(settings["weight_decay"]),
    )
    initial_partition = continuation._model_partition_hashes(student)
    parent_human = _evaluate_human(student, teacher, validation_data, device=device)
    parent_retention = evaluate_detailed_retention(
        teacher,
        student,
        corpus,
        validation_worlds,
        opponent_family,
        rival_side,
        low_variance_threshold_log_std=pools.low_variance_threshold_log_std,
        policy_config=policy_config,
        worlds_per_batch=int(config["validation"]["worlds_per_batch"]),
    )
    parent_guard = detailed_retention_guard(parent_retention, config["validation"]["hard_guard"])
    parent_distribution = continuation._distribution_guard(parent_human, config)
    if not parent_guard["accepted"] or not parent_distribution["accepted"]:
        raise RuntimeError("BC V1 parent failed V3 validation baseline")

    source_step = int(source_payload["counters"]["accepted_optimizer_steps"])
    parent_candidate = _candidate(
        additional_step=0,
        cumulative_step=source_step,
        human=parent_human,
        parent_human=parent_human,
        retention=parent_retention,
        guard=parent_guard,
        distribution=parent_distribution,
        config=config,
    )
    best_path = ROOT / WORK_ROOT / "best.pt"
    rollback_path = ROOT / WORK_ROOT / "rollback.pt"
    continuation._save_state(
        best_path,
        student=student,
        optimizer=optimizer,
        human_generator=human_generator,
        simulator_generator=simulator_generator,
        cumulative_step=source_step,
        proposed_steps=0,
        candidate=parent_candidate,
    )
    best_score = float("inf")
    material_best_score = float("inf")
    best_candidate: dict[str, Any] | None = None
    curve: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    proposed_steps = 0
    additional_steps = 0
    cumulative_step = source_step
    no_material_improvement = 0
    guard_stopped = False
    stop_reason = ""
    interval = int(settings["validation_interval_optimizer_steps"])
    ceiling = int(settings["maximum_accepted_supervised_steps"])
    patience = int(config["selection"]["early_stopping_patience_validations"])
    minimum_plateau_steps = int(config["selection"]["minimum_accepted_steps_before_plateau"])
    material_delta = float(config["selection"]["early_stopping_material_score_improvement"])
    retry_limit = int(config["optimizer"]["transactional_retries_per_interval"])
    backoff = float(config["optimizer"]["transactional_backoff_factor"])
    minimum_lr = float(config["optimizer"]["minimum_learning_rate"])
    mixture_counts = config["retention_sampling"]["rows_per_step"]
    tail = config["tail_aware_retention"]

    while additional_steps < ceiling:
        steps_this_boundary = min(interval, ceiling - additional_steps)
        continuation._save_state(
            rollback_path,
            student=student,
            optimizer=optimizer,
            human_generator=human_generator,
            simulator_generator=simulator_generator,
            cumulative_step=cumulative_step,
            proposed_steps=proposed_steps,
            candidate={},
        )
        interval_lr = float(optimizer.param_groups[0]["lr"])
        accepted_boundary = False
        retries = continuation._transactional_retry_learning_rates(
            interval_lr,
            backoff_factor=backoff,
            retry_count=retry_limit,
        )
        for retry, retry_lr in enumerate(retries):
            if retry:
                continuation._restore_state(
                    rollback_path,
                    student=student,
                    optimizer=optimizer,
                    human_generator=human_generator,
                    simulator_generator=simulator_generator,
                )
            if retry_lr < minimum_lr - 1e-15:
                attempts.append(
                    {
                        "attempted_additional_step": additional_steps + steps_this_boundary,
                        "retry": retry,
                        "learning_rate": retry_lr,
                        "executed": False,
                        "reason": "below frozen minimum learning rate",
                    }
                )
                break
            optimizer.param_groups[0]["lr"] = retry_lr
            loss_sums: defaultdict[str, float] = defaultdict(float)
            grad_max = 0.0
            for _step in range(steps_this_boundary):
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
                encoded, realized = sample_retention_rows(
                    pools, mixture_counts, generator=simulator_generator
                )
                if realized != {name: int(mixture_counts[name]) for name in realized}:
                    raise RuntimeError("stratified retention batch composition changed")
                simulator_observation = gather_encoded_rows(corpus, encoded)
                with torch.no_grad():
                    teacher_human_actor, _ = teacher(human_observation)
                    teacher_sim_actor, _ = teacher(simulator_observation)
                student_human_actor, _ = student(human_observation)
                student_sim_actor, _ = student(simulator_observation)
                human_loss = human_behavior_cloning_objective(
                    student_human_actor,
                    teacher_human_actor,
                    human_action,
                    smooth_l1_beta=float(base_objective["smooth_l1_beta"]),
                    analog_weight=float(base_objective["analog_weight"]),
                    button_weight=float(base_objective["button_weight"]),
                    log_std_weight=float(base_objective["human_log_std_retention_weight"]),
                    policy_config=policy_config,
                )
                retention_loss = tail_aware_actor_retention_loss(
                    teacher_sim_actor,
                    student_sim_actor,
                    policy_config=policy_config,
                    mean_kl_coefficient=float(
                        tail["ordinary_mean_teacher_to_student_kl_coefficient"]
                    ),
                    barrier_threshold=float(tail["activation_threshold_sample_kl"]),
                    barrier_temperature=float(tail["temperature"]),
                    barrier_coefficient=float(tail["barrier_coefficient"]),
                )
                loss = human_loss.loss + retention_loss.loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite V3 objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    trainable, float(settings["gradient_clip_norm"])
                )
                if not bool(torch.isfinite(gradient)):
                    raise RuntimeError("nonfinite V3 gradient")
                optimizer.step()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
                    raise RuntimeError("nonfinite V3 actor parameter")
                if any(
                    parameter.grad is not None
                    for name, parameter in student.named_parameters()
                    if name not in names
                ):
                    raise RuntimeError("frozen trunk/critic received a V3 gradient")
                grad_max = max(grad_max, float(gradient.item()))
                loss_sums["total"] += float(loss.detach().item())
                loss_sums["human_analog"] += float(human_loss.analog_smooth_l1.detach().item())
                loss_sums["human_buttons"] += float(human_loss.button_bce.detach().item())
                loss_sums["human_log_std"] += float(human_loss.log_std_retention.detach().item())
                loss_sums["simulator_mean_kl"] += float(retention_loss.mean_kl.detach().item())
                loss_sums["simulator_tail_barrier"] += float(retention_loss.barrier.detach().item())
                loss_sums["simulator_batch_max_kl"] += float(
                    retention_loss.maximum_sample_kl.detach().item()
                )
                for index, name in enumerate(ACTION_NAMES):
                    loss_sums[f"simulator_mean_kl_{name}"] += float(
                        retention_loss.per_channel_mean_kl[index].detach().item()
                    )

            candidate_human = _evaluate_human(student, teacher, validation_data, device=device)
            candidate_retention = evaluate_detailed_retention(
                teacher,
                student,
                corpus,
                validation_worlds,
                opponent_family,
                rival_side,
                low_variance_threshold_log_std=pools.low_variance_threshold_log_std,
                policy_config=policy_config,
                worlds_per_batch=int(config["validation"]["worlds_per_batch"]),
            )
            retention_guard = detailed_retention_guard(
                candidate_retention, config["validation"]["hard_guard"]
            )
            distribution = continuation._distribution_guard(candidate_human, config)
            partition = continuation._model_partition_hashes(student)
            frozen_exact = (
                partition["frozen_trunk_and_critic"] == initial_partition["frozen_trunk_and_critic"]
            )
            if not frozen_exact:
                raise RuntimeError("V3 frozen trunk/critic tensor identity changed")
            attempt = {
                "attempted_additional_step": additional_steps + steps_this_boundary,
                "retry": retry,
                "learning_rate": retry_lr,
                "executed": True,
                "proposed_optimizer_steps": proposed_steps,
                "mean_training_loss": {
                    key: value / steps_this_boundary for key, value in loss_sums.items()
                },
                "max_preclip_gradient_norm": grad_max,
                "simulator_retention": candidate_retention,
                "retention_guard": retention_guard,
                "distribution_guard": distribution,
                "frozen_trunk_and_critic_exact": frozen_exact,
            }
            attempts.append(attempt)
            if retention_guard["accepted"] and distribution["accepted"]:
                additional_steps += steps_this_boundary
                cumulative_step += steps_this_boundary
                candidate = _candidate(
                    additional_step=additional_steps,
                    cumulative_step=cumulative_step,
                    human=candidate_human,
                    parent_human=parent_human,
                    retention=candidate_retention,
                    guard=retention_guard,
                    distribution=distribution,
                    config=config,
                )
                candidate["learning_rate"] = retry_lr
                candidate["mean_training_loss"] = attempt["mean_training_loss"]
                candidate["max_preclip_gradient_norm"] = grad_max
                candidate["model_partition_hashes"] = partition
                improved = bool(
                    candidate["eligible_for_selection"]
                    and candidate["selection_score"] < best_score
                )
                material = bool(
                    candidate["eligible_for_selection"]
                    and candidate["selection_score"] <= material_best_score - material_delta
                )
                candidate["selected_as_new_best"] = improved
                candidate["material_plateau_improvement"] = material
                if improved:
                    best_score = float(candidate["selection_score"])
                    best_candidate = copy.deepcopy(candidate)
                    continuation._save_state(
                        best_path,
                        student=student,
                        optimizer=optimizer,
                        human_generator=human_generator,
                        simulator_generator=simulator_generator,
                        cumulative_step=cumulative_step,
                        proposed_steps=proposed_steps,
                        candidate=candidate,
                    )
                if material:
                    material_best_score = float(candidate["selection_score"])
                    no_material_improvement = 0
                else:
                    no_material_improvement += 1
                curve.append(candidate)
                boundary_path = ROOT / WORK_ROOT / "accepted" / f"step-{additional_steps:05d}.pt"
                continuation._save_state(
                    boundary_path,
                    student=student,
                    optimizer=optimizer,
                    human_generator=human_generator,
                    simulator_generator=simulator_generator,
                    cumulative_step=cumulative_step,
                    proposed_steps=proposed_steps,
                    candidate=candidate,
                )
                accepted_boundary = True
                print(
                    json.dumps(
                        {
                            "accepted_step": additional_steps,
                            "learning_rate": retry_lr,
                            "gameplay_rmse": candidate_human["families"]["gameplay"][
                                "complete_action_rmse"
                            ],
                            "mechanic_rmse": candidate_human["families"]["mechanic"][
                                "complete_action_rmse"
                            ],
                            "mean_kl": candidate_retention["all_perspectives"]["mean_kl"],
                            "max_kl": candidate_retention["all_perspectives"]["max_sample_kl"],
                            "eligible": candidate["eligible_for_selection"],
                            "new_best": improved,
                            "no_material_improvement": no_material_improvement,
                        }
                    ),
                    flush=True,
                )
                break
            attempt["rollback_required"] = True

        if not accepted_boundary:
            continuation._restore_state(
                rollback_path,
                student=student,
                optimizer=optimizer,
                human_generator=human_generator,
                simulator_generator=simulator_generator,
            )
            guard_stopped = True
            stop_reason = "complete-validation retention/distribution guard exhausted retries"
            break
        if additional_steps >= minimum_plateau_steps and no_material_improvement >= patience:
            stop_reason = "held-out validation improvement plateaued"
            break
    if not stop_reason:
        stop_reason = "10000 accepted-step ceiling reached"
    if best_candidate is None:
        stop_reason = f"{stop_reason}; no validation-eligible material V3 checkpoint"
    selected_state = continuation._restore_state(
        best_path,
        student=student,
        optimizer=optimizer,
        human_generator=human_generator,
        simulator_generator=simulator_generator,
    )
    selected_candidate = selected_state["candidate"]
    selected_additional = int(selected_candidate.get("additional_accepted_step", 0))
    selected_partition = continuation._model_partition_hashes(student)
    return (
        optimizer,
        {
            "parent_human_validation": parent_human,
            "parent_simulator_retention": parent_retention,
            "parent_retention_guard": parent_guard,
            "selected_candidate": selected_candidate,
            "selected_additional_accepted_steps": selected_additional,
            "selected_cumulative_accepted_step": int(selected_state["cumulative_step"]),
            "accepted_steps_executed": additional_steps,
            "proposed_optimizer_steps": proposed_steps,
            "training_curve": curve,
            "transactional_attempts": attempts,
            "stop_reason": stop_reason,
            "plateaued": stop_reason == "held-out validation improvement plateaued",
            "guard_stopped": guard_stopped,
            "ceiling_reached": additional_steps >= ceiling,
            "trainable_parameter_names": list(names),
            "initial_model_partition_hashes": initial_partition,
            "selected_model_partition_hashes": selected_partition,
            "tests_accessed_during_selection": 0,
        },
        selected_state,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_identity = _load_config()
    base_config, _base_evidence = continuation._load_base_config(config)
    preflight = _preflight(config, config_identity, base_config, args.human_source_root)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return {"verdict": "PREFLIGHT_PASS"}
    started = time.perf_counter()
    seed = int(config["optimizer"]["continuation_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    source_path = ROOT / config["authority"]["source_checkpoint"]
    adapter_path = ROOT / config["authority"]["observation_adapter_checkpoint"]
    source_bytes = source_path.read_bytes()
    adapter_bytes = adapter_path.read_bytes()
    source_human_before = _verify_human_sources(args.human_source_root)

    compatibility = {
        "authority": base_config["authority"],
        "corpus": base_config["retention"]["corpus"],
    }
    bootstrap_payload, policy_config, bootstrap_identity = _load_bootstrap(compatibility)
    historical_optimizer_before = tensor_tree_sha256(bootstrap_payload["optimizer"])
    source_payload, source_identity = continuation._load_source_checkpoint(config, policy_config)
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
    human_identity = {
        "train": _human_identity(train_data),
        "validation": _human_identity(validation_data),
    }
    corpus, corpus_manifest, splits = _build_rollout_corpus(
        compatibility,
        bootstrap_payload,
        bootstrap_identity,
        device=args.device,
    )
    torch.use_deterministic_algorithms(True)
    simulator_authority = config["simulator_authority"]
    corpus_checks = {
        "identity_exact": corpus_manifest["identity_sha256"]
        == simulator_authority["training_validation_corpus_identity_sha256"],
        "observation_exact": corpus_manifest["collection"]["observation_tensor_sha256"]
        == simulator_authority["training_validation_observation_sha256"],
        "train_worlds_exact": corpus_manifest["split"]["world_index_sha256"]["train"]
        == simulator_authority["training_world_indices_sha256"],
        "validation_worlds_exact": corpus_manifest["split"]["world_index_sha256"]["validation"]
        == simulator_authority["validation_world_indices_sha256"],
        "old_opened_test_not_used": True,
    }
    if not all(corpus_checks.values()):
        raise RuntimeError(f"V3 training/validation corpus changed: {corpus_checks}")

    teacher = Rival2ActorCritic(policy_config).to(args.device)
    teacher.load_state_dict(source_payload["model"])
    teacher.eval().requires_grad_(False)
    parent = Rival2ActorCritic(policy_config).to(args.device)
    parent.load_state_dict(source_payload["model"])
    parent.eval().requires_grad_(False)
    student = Rival2ActorCritic(policy_config).to(args.device)
    student.load_state_dict(source_payload["model"])
    student.train()
    if tensor_tree_sha256(student.state_dict()) != source_identity["model_tensor_sha256"]:
        raise RuntimeError("V3 student is not byte-identical to BC V1 parent")
    opponent_family = bootstrap_payload["opponent_curriculum"]["family"]
    rival_side = bootstrap_payload["opponent_curriculum"]["rival_side"]
    pools = build_retention_pools(
        teacher,
        corpus,
        splits["train"],
        opponent_family,
        rival_side,
        low_variance_quantile=float(config["retention_sampling"]["low_variance_quantile"]),
        policy_config=policy_config,
    )
    strata_manifest, strata_identity = _load_manifest(
        simulator_authority["retention_strata_manifest"]
    )
    verify_retention_pools(pools, strata_manifest)

    optimizer, training, selected_state = _train(
        teacher=teacher,
        student=student,
        source_payload=source_payload,
        policy_config=policy_config,
        train_data=train_data,
        validation_data=validation_data,
        corpus=corpus,
        train_worlds=splits["train"],
        validation_worlds=splits["validation"],
        opponent_family=opponent_family,
        rival_side=rival_side,
        pools=pools,
        config=config,
        base_config=base_config,
        device=args.device,
    )
    final_selection_utc = datetime.now(UTC).isoformat()
    selected_candidate = training["selected_candidate"]

    test_data = _load_human_split(
        "test",
        config=base_config,
        adapter=adapter,
        source_root=args.human_source_root,
        device=args.device,
    )
    human_identity["test"] = _human_identity(test_data)
    human_test_parent = _evaluate_human(parent, teacher, test_data, device=args.device)
    human_test_selected = _evaluate_human(student, teacher, test_data, device=args.device)
    human_test_access_utc = datetime.now(UTC).isoformat()

    del corpus, pools
    gc.collect()
    torch.cuda.empty_cache()
    new_corpus_config = copy.deepcopy(base_config["retention"]["corpus"])
    new_corpus_config["seed"] = int(simulator_authority["new_untouched_test_corpus_seed"])
    new_corpus_config["split"]["seed"] = int(simulator_authority["new_untouched_test_split_seed"])
    new_corpus_config.pop("expected_identity_sha256", None)
    new_corpus_config.pop("expected_observation_tensor_sha256", None)
    new_compatibility = {
        "authority": base_config["authority"],
        "corpus": new_corpus_config,
    }
    torch.use_deterministic_algorithms(False)
    try:
        new_corpus, new_manifest, new_splits = _build_rollout_corpus(
            new_compatibility,
            bootstrap_payload,
            bootstrap_identity,
            device=args.device,
        )
    finally:
        torch.use_deterministic_algorithms(True)
    new_test_checks = {
        "corpus_identity_exact": new_manifest["identity_sha256"]
        == simulator_authority["new_untouched_test_corpus_identity_sha256"],
        "observation_hash_exact": new_manifest["collection"]["observation_tensor_sha256"]
        == simulator_authority["new_untouched_test_observation_sha256"],
        "test_worlds_hash_exact": int64_sha256(new_splits["test"])
        == simulator_authority["new_untouched_test_world_indices_sha256"],
        "world_count_exact": len(new_splits["test"])
        == simulator_authority["new_untouched_test_worlds"],
        "evaluated_after_final_selection": True,
    }
    if not all(new_test_checks.values()):
        raise RuntimeError(f"new untouched simulator test authority changed: {new_test_checks}")
    simulator_test = evaluate_detailed_retention(
        teacher,
        student,
        new_corpus,
        new_splits["test"],
        opponent_family,
        rival_side,
        low_variance_threshold_log_std=float(
            config["retention_sampling"]["low_variance_threshold_log_std"]
        ),
        policy_config=policy_config,
        worlds_per_batch=int(config["validation"]["worlds_per_batch"]),
    )
    simulator_test_guard = detailed_retention_guard(
        simulator_test, config["validation"]["hard_guard"]
    )
    simulator_test_access_utc = datetime.now(UTC).isoformat()

    selected_additional = int(training["selected_additional_accepted_steps"])
    selected_model = _cpu_tree(student.state_dict())
    selected_optimizer = _cpu_tree(optimizer.state_dict())
    labels = selected_candidate.get("mechanic_label_comparison_to_parent", {})
    acceptance_checks = {
        "selected_material_checkpoint": selected_additional > 0,
        "gameplay_validation_material_improvement": selected_candidate.get(
            "gameplay_rmse_ratio_to_parent", 1.0
        )
        <= 1.0 - config["selection"]["minimum_human_family_relative_improvement"],
        "mechanic_validation_material_improvement": selected_candidate.get(
            "mechanic_rmse_ratio_to_parent", 1.0
        )
        <= 1.0 - config["selection"]["minimum_human_family_relative_improvement"],
        "broad_mechanic_label_improvement": labels.get("improved_fraction", 0.0)
        >= config["selection"]["minimum_mechanic_labels_improved_fraction"],
        "broad_mechanic_label_nonregression": labels.get("nonregressed_fraction", 0.0)
        >= config["selection"]["minimum_mechanic_labels_nonregressed_fraction"],
        "complete_validation_guard": bool(
            selected_candidate.get("retention_guard", {}).get("accepted", False)
        ),
        "new_untouched_simulator_test_guard": simulator_test_guard["accepted"],
        "selected_distribution_healthy": bool(
            selected_candidate.get("distribution_guard", {}).get("accepted", False)
        ),
        "human_test_outputs_finite": bool(
            human_test_parent["finite"] and human_test_selected["finite"]
        ),
        "frozen_trunk_and_critic_exact": training["initial_model_partition_hashes"][
            "frozen_trunk_and_critic"
        ]
        == training["selected_model_partition_hashes"]["frozen_trunk_and_critic"],
        "test_selection_not_reopened": True,
    }
    accepted = all(acceptance_checks.values())
    checkpoint_path = ROOT / config["checkpoint"]["path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "format": V3_CHECKPOINT_FORMAT,
        "version": V3_VERSION,
        "base_human_bc_version": HUMAN_BC_VERSION,
        "model": selected_model,
        "actor_only_optimizer": selected_optimizer,
        "optimizer_provenance": {
            "fresh_actor_only_adamw": True,
            "source_bc_optimizer_loaded": False,
            "historical_ppo_optimizer_loaded": False,
            "trainable_parameter_names": training["trainable_parameter_names"],
        },
        "policy_config": asdict(policy_config),
        "observation_version": source_payload["observation_version"],
        "action_version": source_payload["action_version"],
        "physics_hz": 120,
        "policy_hz": 120,
        "authority": {
            "v3_config": config_identity,
            "source_checkpoint": source_identity,
            "bootstrap": bootstrap_identity,
            "observation_adapter": adapter_identity,
            "human_split_identity": human_identity,
            "training_validation_corpus_identity_sha256": corpus_manifest["identity_sha256"],
            "retention_strata": strata_identity,
            "new_untouched_simulator_test_corpus_identity_sha256": new_manifest["identity_sha256"],
        },
        "counters": {
            "source_accepted_optimizer_steps": source_identity["accepted_optimizer_steps"],
            "accepted_actor_only_supervised_steps": selected_additional,
            "cumulative_human_bc_steps": training["selected_cumulative_accepted_step"],
            "accepted_steps_executed": training["accepted_steps_executed"],
            "proposed_optimizer_steps": training["proposed_optimizer_steps"],
        },
        "rng_state": {
            "human_generator": selected_state["human_generator_state"],
            "simulator_generator": selected_state["simulator_generator_state"],
        },
        "selected_validation": selected_candidate,
        "human_test": {"parent": human_test_parent, "selected": human_test_selected},
        "new_simulator_test_retention": simulator_test,
        "stop": {
            "reason": training["stop_reason"],
            "plateaued": training["plateaued"],
            "guard_stopped": training["guard_stopped"],
            "ceiling_reached": training["ceiling_reached"],
        },
        "resumability": {
            "human_bc_resumable": True,
            "ppo_resumable": False,
            "ppo_requires_explicit_transition_authority": True,
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    checkpoint_identity = {
        "path": config["checkpoint"]["path"],
        "bytes": checkpoint_path.stat().st_size,
        "sha256": file_sha256(checkpoint_path),
        "model_tensor_sha256": tensor_tree_sha256(selected_model),
        "model_partition_hashes": continuation._model_partition_hashes(student),
    }
    source_human_after = _verify_human_sources(args.human_source_root)
    integrity = {
        "source_checkpoint_sha256_before_after": [
            hashlib.sha256(source_bytes).hexdigest().upper(),
            file_sha256(source_path),
        ],
        "adapter_checkpoint_sha256_before_after": [
            hashlib.sha256(adapter_bytes).hexdigest().upper(),
            file_sha256(adapter_path),
        ],
        "historical_ppo_optimizer_sha256_before_after": [
            historical_optimizer_before,
            tensor_tree_sha256(bootstrap_payload["optimizer"]),
        ],
        "native_human_sources_unchanged": source_human_before == source_human_after,
        "frozen_trunk_and_critic_sha256_before_after": [
            training["initial_model_partition_hashes"]["frozen_trunk_and_critic"],
            training["selected_model_partition_hashes"]["frozen_trunk_and_critic"],
        ],
        "human_test_access_count": 1,
        "new_simulator_test_access_count": 1,
        "old_opened_simulator_test_candidate_evaluations": 0,
        "final_selection_utc": final_selection_utc,
        "human_test_access_utc": human_test_access_utc,
        "simulator_test_access_utc": simulator_test_access_utc,
        "selection_reopened_after_tests": False,
        "ppo_updates": 0,
        "reward_changes": 0,
        "mechanic_definition_changes": 0,
        "dataset_split_changes": 0,
        "observation_adapter_changes": 0,
    }
    integrity["valid"] = bool(
        integrity["source_checkpoint_sha256_before_after"][0]
        == integrity["source_checkpoint_sha256_before_after"][1]
        == config["authority"]["source_checkpoint_sha256"]
        and integrity["adapter_checkpoint_sha256_before_after"][0]
        == integrity["adapter_checkpoint_sha256_before_after"][1]
        == config["authority"]["observation_adapter_checkpoint_sha256"]
        and integrity["historical_ppo_optimizer_sha256_before_after"][0]
        == integrity["historical_ppo_optimizer_sha256_before_after"][1]
        and integrity["native_human_sources_unchanged"]
        and integrity["frozen_trunk_and_critic_sha256_before_after"][0]
        == integrity["frozen_trunk_and_critic_sha256_before_after"][1]
        and integrity["human_test_access_count"] == 1
        and integrity["new_simulator_test_access_count"] == 1
        and integrity["old_opened_simulator_test_candidate_evaluations"] == 0
        and not integrity["selection_reopened_after_tests"]
        and integrity["ppo_updates"] == 0
        and integrity["reward_changes"] == 0
        and integrity["mechanic_definition_changes"] == 0
        and integrity["dataset_split_changes"] == 0
        and integrity["observation_adapter_changes"] == 0
    )
    accepted = accepted and integrity["valid"]
    evidence = {
        "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_EVIDENCE_V3",
        "generated_utc": datetime.now(UTC).isoformat(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "verdict": "PASS" if accepted else "BLOCKED",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
        "config": config_identity,
        "preflight": preflight,
        "source_checkpoint": source_identity,
        "bootstrap": bootstrap_identity,
        "observation_adapter": adapter_identity,
        "human_split_identity": human_identity,
        "training_validation_corpus": corpus_manifest,
        "training_validation_corpus_checks": corpus_checks,
        "retention_strata": strata_manifest,
        "training": training,
        "human_test": {"parent": human_test_parent, "selected": human_test_selected},
        "new_simulator_test_corpus": new_manifest,
        "new_simulator_test_checks": new_test_checks,
        "new_simulator_test_retention": simulator_test,
        "new_simulator_test_guard": simulator_test_guard,
        "acceptance": {"checks": acceptance_checks, "accepted": accepted},
        "checkpoint": checkpoint_identity,
        "integrity": integrity,
        "old_opened_simulator_test": {
            "role": "diagnostic_only",
            "candidate_evaluations": 0,
            "used_for_acceptance": False,
        },
        "prohibited_work": {
            "ppo": False,
            "reward_change": False,
            "mechanic_definition_change": False,
            "demonstration_change": False,
            "dataset_split_change": False,
            "observation_adapter_change": False,
            "observation_action_contract_change": False,
        },
    }
    _write_json(ROOT / RESULT_ROOT / "runtime_pre_step_preflight.json", preflight)
    _write_json(ROOT / RESULT_ROOT / "training_curve.json", training)
    _write_json(ROOT / RESULT_ROOT / "human_test_metrics.json", evidence["human_test"])
    _write_json(
        ROOT / RESULT_ROOT / "new_simulator_test_results.json",
        {
            "corpus_checks": new_test_checks,
            "retention": simulator_test,
            "guard": simulator_test_guard,
        },
    )
    _write_json(ROOT / RESULT_ROOT / "evidence.json", evidence)
    artifacts = [
        Path(config["checkpoint"]["path"]),
        RESULT_ROOT / "README.md",
        RESULT_ROOT / "evidence.json",
        RESULT_ROOT / "frozen_config.json",
        RESULT_ROOT / "human_test_metrics.json",
        RESULT_ROOT / "new_simulator_test_authority.json",
        RESULT_ROOT / "new_simulator_test_results.json",
        RESULT_ROOT / "pre_step_authority.json",
        RESULT_ROOT / "pre_step_preflight.json",
        RESULT_ROOT / "retention_strata_manifest.json",
        RESULT_ROOT / "runtime_pre_step_preflight.json",
        RESULT_ROOT / "training_curve.json",
    ]
    _write_json(ROOT / RESULT_ROOT / "artifact_manifest.json", _artifact_manifest(artifacts))
    print(
        json.dumps(
            {
                "verdict": evidence["verdict"],
                "checkpoint": checkpoint_identity,
                "accepted_steps": selected_additional,
                "gameplay_validation": {
                    "parent": training["parent_human_validation"]["families"]["gameplay"][
                        "complete_action_rmse"
                    ],
                    "v3": selected_candidate.get("human_validation", {})
                    .get("families", {})
                    .get("gameplay", {})
                    .get("complete_action_rmse"),
                },
                "mechanic_validation": {
                    "parent": training["parent_human_validation"]["families"]["mechanic"][
                        "complete_action_rmse"
                    ],
                    "v3": selected_candidate.get("human_validation", {})
                    .get("families", {})
                    .get("mechanic", {})
                    .get("complete_action_rmse"),
                },
                "complete_validation_max_kl": selected_candidate.get("simulator_retention", {})
                .get("all_perspectives", {})
                .get("max_sample_kl"),
                "new_test_max_kl": simulator_test["all_perspectives"]["max_sample_kl"],
                "new_test_mean_kl": simulator_test["all_perspectives"]["mean_kl"],
                "stop_reason": training["stop_reason"],
            },
            indent=2,
        )
    )
    del new_corpus, train_data, validation_data, test_data
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
