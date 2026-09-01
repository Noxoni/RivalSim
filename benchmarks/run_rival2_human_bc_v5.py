"""Train actor-only Rival Human BC V5 under prospective tail-safe retention.

This runner is intentionally fail closed.  Its optimizer cannot be constructed until
the V5 authority, strata, stress-validation, and untouched-test manifests are present
in the current remotely persisted commit.  Human and simulator test data are not
opened until one validation-only checkpoint has been selected.
"""

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
from collections.abc import Mapping
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
from benchmarks.rival2_human_bc_v4_common import (  # noqa: E402
    build_aligned_rollout_corpus,
    select_world_subset,
)
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
    _load_bootstrap,
    _verify_human_sources,
)
from rivalsim.human_demo.bc_v3_retention import (  # noqa: E402
    detailed_retention_guard,
    int64_sha256,
)
from rivalsim.human_demo.bc_v4_retention import (  # noqa: E402
    HardTailReplayState,
    build_v4_retention_pools,
    evaluate_v4_retention,
    gather_encoded_rows,
    initialize_hard_tail_replay,
    mine_training_hard_tail,
    sample_v4_retention_rows,
    v4_tail_aware_actor_retention_loss,
    verify_v4_retention_pools,
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

V5_VERSION = "RIVAL2_HUMAN_BEHAVIOR_CLONING_V5"
V5_CHECKPOINT_FORMAT = "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V5"
FROZEN_CONFIG = Path("results/rival2/human_bc_v5/frozen_config.json")
# Replaced exactly once, in the prospective authority commit, before any optimizer step.
FROZEN_CONFIG_SHA256 = "793DD36ACE16F4D87FB846E2329D0A7FDB530C75677B83A3103A186F7E8A9F79"
RESULT_ROOT = Path("results/rival2/human_bc_v5")
WORK_ROOT = Path(".tools/rival2_human_bc_v5")
TEST_ACCESS_LEDGER = RESULT_ROOT / "test_access_ledger.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _atomic_write_json(path: Path, value: Any) -> None:
    """Persist one-shot test access state before advancing to the next phase."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _advance_test_ledger(
    ledger: dict[str, Any], path: Path, *, phase: str, **updates: Any
) -> None:
    ledger["phase"] = phase
    ledger["last_updated_utc"] = datetime.now(UTC).isoformat()
    ledger.update(updates)
    _atomic_write_json(path, ledger)


def _committed_json(path_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = Path(path_text)
    absolute = ROOT / relative
    payload = json.loads(absolute.read_text(encoding="utf-8"))
    committed = _git("rev-parse", f"HEAD:{relative.as_posix()}")
    working = _git("hash-object", str(relative))
    if committed != working:
        raise ValueError(f"working authority differs from committed blob: {relative}")
    return payload, {
        "path": relative.as_posix(),
        "sha256": file_sha256(absolute),
        "git_blob_oid": committed,
    }


def _load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    path = ROOT / FROZEN_CONFIG
    digest = file_sha256(path)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "frozen V5 config has not been prospectively bound in the runner "
            f"or changed after binding: {digest} != {FROZEN_CONFIG_SHA256}"
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    required = config["authority"]["required_parent"]
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", required, "HEAD"],
        check=True,
    )
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    if head != origin:
        raise ValueError(f"V5 training requires remotely persisted HEAD: {head} != {origin}")
    worktree_status = _git("status", "--porcelain")
    if worktree_status:
        raise ValueError(
            "V5 training requires a clean committed worktree before any optimizer is "
            f"constructed; found:\n{worktree_status}"
        )
    committed = _git("rev-parse", f"HEAD:{FROZEN_CONFIG.as_posix()}")
    working = _git("hash-object", str(FROZEN_CONFIG))
    if committed != working:
        raise ValueError("working V5 config differs from committed authority")
    return config, {
        "path": FROZEN_CONFIG.as_posix(),
        "sha256": digest,
        "git_blob_oid": committed,
        "pre_step_git_commit": head,
        "origin_main": origin,
        "required_parent": required,
        "required_parent_is_ancestor": True,
    }


def _human_identity(data: Any) -> dict[str, Any]:
    return {
        "gameplay_frames": int(data.gameplay_action.shape[0]),
        "mechanic_frames": int(data.mechanic_action.shape[0]),
        "mechanic_attempts": len(set(data.mechanic_attempt)),
        "action_sha256": data.action_sha256,
        "source_sequences_sha256": data.source_sequences_sha256,
    }


def _corpus_value(corpus: Any, name: str) -> Any:
    if hasattr(corpus, name):
        return getattr(corpus, name)
    if isinstance(corpus, Mapping):
        return corpus[name]
    raise TypeError(f"aligned corpus has no {name!r} member")


def _retention_evaluation(
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    corpus: Any,
    worlds: np.ndarray,
    pools: Any,
    config: Mapping[str, Any],
    policy_config: Rival2PolicyConfig,
) -> dict[str, Any]:
    return evaluate_v4_retention(
        teacher,
        student,
        _corpus_value(corpus, "observations"),
        worlds,
        _corpus_value(corpus, "opponent_family"),
        _corpus_value(corpus, "train_mask"),
        low_variance_threshold_log_std=pools.low_variance_threshold_log_std,
        orientation_authority=pools.orientation_authority,
        policy_config=policy_config,
        worlds_per_batch=int(config["validation"]["worlds_per_batch"]),
    )


def _selection_score(
    human: Mapping[str, Any],
    parent_human: Mapping[str, Any],
    labels: Mapping[str, Any],
    complete: Mapping[str, Any],
    stress: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    weights = config["selection"]["score_weights"]
    gameplay_ratio = (
        float(human["families"]["gameplay"]["complete_action_rmse"])
        / float(parent_human["families"]["gameplay"]["complete_action_rmse"])
    )
    mechanic_ratio = (
        float(human["families"]["mechanic"]["complete_action_rmse"])
        / float(parent_human["families"]["mechanic"]["complete_action_rmse"])
    )
    complete_all = complete["all_perspectives"]
    stress_all = stress["all_perspectives"]
    return float(
        float(weights["gameplay_rmse_ratio"]) * gameplay_ratio
        + float(weights["mechanic_rmse_ratio"]) * mechanic_ratio
        + float(weights["mean_per_label_rmse_ratio"])
        * float(labels["mean_rmse_ratio_to_parent"])
        + float(weights["complete_mean_kl_soft_limit_ratio"])
        * float(complete_all["mean_kl"])
        / float(config["validation"]["soft_actor_mean_kl"])
        + float(weights["complete_max_kl_selection_ratio"])
        * float(complete_all["max_sample_kl"])
        / float(config["validation"]["hard_guard"]["actor_max_sample_kl"])
        + float(weights["stress_mean_kl_soft_limit_ratio"])
        * float(stress_all["mean_kl"])
        / float(config["validation"]["soft_actor_mean_kl"])
        + float(weights["stress_max_kl_selection_ratio"])
        * float(stress_all["max_sample_kl"])
        / float(config["validation"]["hard_guard"]["actor_max_sample_kl"])
    )


def _candidate(
    *,
    additional_step: int,
    cumulative_step: int,
    human: dict[str, Any],
    parent_human: dict[str, Any],
    complete: dict[str, Any],
    stress: dict[str, Any],
    complete_guard: dict[str, Any],
    stress_guard: dict[str, Any],
    human_distribution: dict[str, Any],
    complete_distribution: dict[str, Any],
    stress_distribution: dict[str, Any],
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
    eligible = bool(
        additional_step > 0
        and gameplay_ratio < 1.0
        and mechanic_ratio < 1.0
        and labels["improved_fraction"]
        >= float(config["selection"]["minimum_mechanic_labels_improved_fraction"])
        and labels["nonregressed_fraction"]
        >= float(config["selection"]["minimum_mechanic_labels_nonregressed_fraction"])
        and complete_guard["accepted"]
        and stress_guard["accepted"]
        and human_distribution["accepted"]
        and complete_distribution["accepted"]
        and stress_distribution["accepted"]
    )
    return {
        "additional_accepted_step": additional_step,
        "cumulative_accepted_step": cumulative_step,
        "human_validation": human,
        "gameplay_rmse_ratio_to_parent": gameplay_ratio,
        "mechanic_rmse_ratio_to_parent": mechanic_ratio,
        "mechanic_label_comparison_to_parent": labels,
        "complete_simulator_validation": complete,
        "stress_simulator_validation": stress,
        "complete_retention_guard": complete_guard,
        "stress_retention_guard": stress_guard,
        "distribution_guard": human_distribution,
        "complete_simulator_distribution_guard": complete_distribution,
        "stress_simulator_distribution_guard": stress_distribution,
        "selection_score": _selection_score(
            human, parent_human, labels, complete, stress, config
        ),
        "eligible_for_selection": eligible,
    }


def _simulator_distribution_guard(
    retention_metrics: Mapping[str, Any],
    parent_metrics: Mapping[str, Any],
    config: dict[str, Any],
    *,
    authority_phase: str,
) -> dict[str, Any]:
    """Apply all distribution checks plus the sole parent-relative boost exception."""

    group_names = tuple(config["validation"]["groups"])
    wrapped = {
        "finite": all(
            bool(retention_metrics[name]["actor_distribution"]["finite"])
            for name in group_names
        ),
        "families": {
            name: {
                "actor_output_statistics": retention_metrics[name][
                    "actor_distribution"
                ]
            }
            for name in group_names
        },
    }
    result = continuation._distribution_guard(wrapped, config)
    rule = config["parent_relative_boost_saturation"]
    relative: dict[str, Any] = {}
    authority_exact = True
    for name in group_names:
        parent = float(
            parent_metrics[name]["actor_distribution"]["button_probability"]["boost"]
            ["saturation_fraction"]
        )
        candidate = float(
            retention_metrics[name]["actor_distribution"]["button_probability"]
            ["boost"]["saturation_fraction"]
        )
        limit = max(
            float(rule["absolute_baseline_limit"]),
            parent + float(rule["worsening_allowance"]),
        )
        if authority_phase in ("complete_validation", "stress_validation"):
            frozen = rule[authority_phase][name]
            parent_exact = abs(
                parent - float(frozen["bc_v1_parent_saturation_fraction"])
            ) <= 1e-12
            limit_exact = abs(limit - float(frozen["candidate_limit"])) <= 1e-12
            authority_exact = authority_exact and parent_exact and limit_exact
        else:
            parent_exact = True
            limit_exact = True
        check_name = f"{name}.button.boost.not_saturated"
        result["checks"][check_name] = candidate <= limit
        relative[name] = {
            "bc_v1_parent_saturation_fraction": parent,
            "candidate_saturation_fraction": candidate,
            "candidate_limit": limit,
            "accepted": candidate <= limit,
            "parent_authority_exact": parent_exact,
            "limit_authority_exact": limit_exact,
        }
    result["checks"]["parent_relative_boost_authority_exact"] = authority_exact
    result["accepted"] = all(result["checks"].values())
    result["parent_relative_boost_saturation"] = relative
    result["authority_phase"] = authority_phase
    return result


def _v5_validation_guard(
    metrics: Mapping[str, Any], hard_guard: Mapping[str, Any]
) -> dict[str, Any]:
    """Use the real frozen retention contract without V4's extra selection walls."""

    contract = detailed_retention_guard(metrics, hard_guard)
    return {
        "contract": contract,
        "contract_accepted": bool(contract["accepted"]),
        "extra_max_sample_selection_margin_removed": True,
        "extra_orientation_channel_selection_margin_removed": True,
        "accepted": bool(contract["accepted"]),
    }


def _parent_pretraining_baseline_guard(
    *,
    complete_retention: Mapping[str, Any],
    stress_retention: Mapping[str, Any],
    human_distribution: Mapping[str, Any],
    complete_distribution: Mapping[str, Any],
    stress_distribution: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate the unchanged BC-V1 parent before any optimizer step.

    Simulator distribution-health checks are candidate acceptance checks. They are
    still recorded for the parent, but requiring the unchanged parent to satisfy
    them would make an inherited teacher characteristic an impossible prerequisite
    for beginning the actor-only correction. Every trained candidate, selection,
    and untouched-test result remains subject to those unchanged checks.
    """

    checks = {
        "complete_retention": bool(complete_retention["accepted"]),
        "stress_retention": bool(stress_retention["accepted"]),
        "human_distribution": bool(human_distribution["accepted"]),
        "complete_distribution": bool(complete_distribution["accepted"]),
        "stress_distribution": bool(stress_distribution["accepted"]),
    }
    return {"checks": checks, "accepted": all(checks.values())}


def _snapshot_rng() -> dict[str, Any]:
    return {
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        "numpy": copy.deepcopy(np.random.get_state()),
    }


def _restore_rng(payload: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(payload["torch_cpu"])
    torch.cuda.set_rng_state_all(payload["torch_cuda"])
    np.random.set_state(payload["numpy"])


def _save_state(
    path: Path,
    *,
    student: Rival2ActorCritic,
    optimizer: torch.optim.Optimizer,
    human_generator: torch.Generator,
    simulator_generator: torch.Generator,
    replay_state: HardTailReplayState,
    cumulative_step: int,
    proposed_steps: int,
    mining_round: int,
    candidate: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _cpu_tree(student.state_dict()),
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "human_generator_state": human_generator.get_state().clone(),
            "simulator_generator_state": simulator_generator.get_state().clone(),
            "global_rng": _snapshot_rng(),
            "replay_state": copy.deepcopy(replay_state),
            "cumulative_step": cumulative_step,
            "proposed_steps": proposed_steps,
            "mining_round": mining_round,
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
    _restore_rng(payload["global_rng"])
    return payload


def _preflight(
    config: dict[str, Any],
    config_identity: dict[str, Any],
    base_config: dict[str, Any],
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    strata, strata_identity = _committed_json(
        config["simulator_authority"]["retention_strata_manifest"]
    )
    stress, stress_identity = _committed_json(
        config["simulator_authority"]["stress_validation_authority"]
    )
    untouched, untouched_identity = _committed_json(
        config["simulator_authority"]["untouched_test_authority"]
    )
    build_preflight, build_preflight_identity = _committed_json(
        "results/rival2/human_bc_v5/pre_step_preflight.json"
    )
    reuse, reuse_identity = _committed_json(
        "results/rival2/human_bc_v5/v4_authority_reuse_manifest.json"
    )
    pre_step, pre_step_identity = _committed_json(
        config["authority"]["pre_step_authority"]
    )
    disjointness, disjointness_identity = _committed_json(
        "results/rival2/human_bc_v4/simulator_disjointness_proof.json"
    )
    authority_commit = str(pre_step["authority_commit"])
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", authority_commit, "HEAD"],
        check=True,
    )
    bound_file_checks: dict[str, bool] = {}
    for path_text, identity in pre_step["bound_files"].items():
        relative = Path(path_text)
        bound_file_checks[path_text] = bool(
            file_sha256(ROOT / relative) == identity["sha256"]
            and _git("rev-parse", f"{authority_commit}:{relative.as_posix()}")
            == identity["git_blob_oid"]
            and _git("hash-object", str(relative)) == identity["git_blob_oid"]
        )
    checks = {
        "source_checkpoint_exact": file_sha256(ROOT / config["authority"]["source_checkpoint"])
        == config["authority"]["source_checkpoint_sha256"],
        "adapter_exact": file_sha256(
            ROOT / config["authority"]["observation_adapter_checkpoint"]
        )
        == config["authority"]["observation_adapter_checkpoint_sha256"],
        "v2_v3_v4_training_prohibited": bool(
            config["authority"]["v2_v3_v4_checkpoint_training_prohibited"]
        ),
        "strata_no_student_or_test_membership": bool(
            strata["membership_source"]
            == "teacher_and_training_state_only_no_student"
            and not strata["orientation_sensitive_definition"]["student_outputs_used"]
            and not strata["orientation_sensitive_definition"]["opened_test_rows_used"]
        ),
        "hard_tail_training_only": bool(
            strata["mining_candidate_pool"]["source"]
            == "frozen simulator training rows only"
        ),
        "stress_is_selection_not_test": bool(
            stress["role"] == "stress_validation"
        ),
        "new_test_unopened": int(untouched["student_evaluations_before_final_selection"])
        == 0,
        "new_test_disjoint": bool(
            disjointness["all_seed_namespaces_unique"]
            and all(
                int(value) == 0
                for value in disjointness["pairwise_trajectory_overlap"].values()
            )
        ),
        "hard_max_unchanged": float(
            config["validation"]["hard_guard"]["actor_max_sample_kl"]
        )
        == 2.0,
        "extra_selection_margins_removed": bool(
            config["selection"]["extra_kl_selection_margins_removed"]
            and "selection_margin" not in config["validation"]
        ),
        "parent_relative_boost_exact": bool(
            float(config["parent_relative_boost_saturation"]["worsening_allowance"])
            == 0.005
            and float(
                config["parent_relative_boost_saturation"]["absolute_baseline_limit"]
            )
            == 0.95
        ),
        "v4_authority_reused": bool(
            config["simulator_authority"]["v4_assets_reused_exactly"]
            and not config["simulator_authority"]["new_v5_corpora_generated"]
            and int(reuse["new_simulator_corpora_generated"]) == 0
        ),
        "reused_v4_test_still_sealed": bool(
            not (ROOT / "results/rival2/human_bc_v4/test_access_ledger.json").exists()
            and not (
                ROOT / "results/rival2/human_bc_v4/untouched_test_results.json"
            ).exists()
        ),
        "actor_only": config["trainable_parameters"]["mode"] == "actor_head_only",
        "maximum_steps_exact": int(
            config["training"]["maximum_accepted_supervised_steps"]
        )
        == 10_000,
        "authority_commit_is_ancestor": True,
        "authority_was_remotely_persisted": pre_step["origin_main_at_freeze"]
        == authority_commit,
        "authority_bound_files_exact": all(bound_file_checks.values()),
        "authority_pre_optimizer": int(pre_step["optimizer_steps"]) == 0,
        "authority_no_test_access": int(
            build_preflight["reused_v4_test_student_evaluations"]
        )
        == 0,
    }
    authority_files = _verify_authority_files(base_config)
    split_audit = _dataset_split_audit(base_config)
    human_sources = _verify_human_sources(source_root)
    if not all(checks.values()):
        raise RuntimeError(f"V5 pre-step preflight failed: {checks}")
    return (
        {
            "format": "RIVAL2_HUMAN_BC_V5_RUNTIME_PRE_STEP_PREFLIGHT_V1",
            "generated_utc": datetime.now(UTC).isoformat(),
            "config": config_identity,
            "checks": checks,
            "manifests": {
                "retention_strata": strata_identity,
                "stress_validation": stress_identity,
                "untouched_test": untouched_identity,
                "build_preflight": build_preflight_identity,
                "v4_authority_reuse": reuse_identity,
                "pre_step_authority": pre_step_identity,
                "simulator_disjointness": disjointness_identity,
            },
            "base_authority_files": authority_files,
            "human_split_audit": split_audit,
            "human_source_verification": human_sources,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "human_test_accesses": 0,
            "v5_test_student_evaluations": 0,
            "opened_v2_v3_test_candidate_evaluations": 0,
            "ppo_updates": 0,
            "reward_changes": 0,
            "valid": True,
            "authority_bound_file_checks": bound_file_checks,
        },
        {
            "strata": strata,
            "strata_identity": strata_identity,
            "stress": stress,
            "stress_identity": stress_identity,
            "untouched": untouched,
            "untouched_identity": untouched_identity,
            "pre_step_identity": pre_step_identity,
            "build_preflight_identity": build_preflight_identity,
            "reuse_identity": reuse_identity,
            "disjointness_identity": disjointness_identity,
        },
    )


def _training_loss_telemetry(loss: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in (
        "mean_kl",
        "total_barrier",
        "total_mean_barrier",
        "total_cvar_barrier",
        "orientation_barrier",
        "orientation_cvar_barrier",
        "maximum_sample_kl",
        "maximum_orientation_kl",
        "maximum_orientation_channel_kl",
        "maximum_individual_orientation_channel_kl",
    ):
        value = getattr(loss, name, None)
        if value is not None:
            values[name] = float(value.detach().item())
    for index, action_name in enumerate(ACTION_NAMES):
        values[f"mean_kl_{action_name}"] = float(
            loss.per_channel_mean_kl[index].detach().item()
        )
    return values


def _corpus_hash_checks(
    corpus: Any,
    manifest: Mapping[str, Any],
    splits: Mapping[str, np.ndarray],
    authority: Mapping[str, Any],
    *,
    prefix: str,
    reserved_split: str | None = None,
) -> dict[str, bool]:
    collection = manifest["collection"]
    checks = {
        "identity_exact": manifest["identity_sha256"]
        == authority[f"{prefix}_corpus_identity_sha256"],
        "observation_exact": collection["observation_tensor_sha256"]
        == authority[f"{prefix}_observation_sha256"],
        "opponent_family_exact": collection["opponent_family_tensor_sha256"]
        == authority[f"{prefix}_opponent_family_sha256"],
        "train_mask_exact": collection["train_mask_tensor_sha256"]
        == authority[f"{prefix}_train_mask_sha256"],
        "aligned_shapes": (
            _corpus_value(corpus, "opponent_family").shape
            == _corpus_value(corpus, "train_mask").shape
            == _corpus_value(corpus, "observations").shape[:3]
        ),
    }
    if prefix == "training_validation":
        checks.update(
            {
                "train_worlds_exact": manifest["split"]["world_index_sha256"]["train"]
                == authority["training_world_indices_sha256"],
                "validation_worlds_exact": manifest["split"]["world_index_sha256"]
                ["validation"]
                == authority["validation_world_indices_sha256"],
                "validation_count_exact": len(splits["validation"])
                == int(authority["complete_validation_worlds"]),
            }
        )
    else:
        if reserved_split is None:
            raise ValueError("reserved split required for independent V5 corpus")
        reserved = np.asarray(splits[reserved_split], dtype=np.int64)
        checks.update(
            {
                "reserved_worlds_hash_exact": int64_sha256(reserved)
                == authority[f"{prefix}_world_indices_sha256"],
                "reserved_world_count_exact": len(reserved)
                == int(authority[f"{prefix}_worlds"]),
            }
        )
    return checks


def _independent_corpus_config(
    base_config: Mapping[str, Any],
    *,
    corpus_seed: int,
    split_seed: int,
) -> dict[str, Any]:
    corpus = copy.deepcopy(base_config["retention"]["corpus"])
    corpus["seed"] = corpus_seed
    corpus["split"] = {
        "algorithm": "numpy PCG64 permutation of whole world indices",
        "seed": split_seed,
        "train_worlds": 16_384,
        "validation_worlds": 8_192,
        "test_worlds": 8_192,
    }
    corpus.pop("expected_identity_sha256", None)
    corpus.pop("expected_observation_tensor_sha256", None)
    return {"authority": base_config["authority"], "corpus": corpus}


def _train(
    *,
    teacher: Rival2ActorCritic,
    student: Rival2ActorCritic,
    source_payload: dict[str, Any],
    policy_config: Rival2PolicyConfig,
    train_data: Any,
    validation_data: Any,
    complete_corpus: Any,
    complete_train_worlds: np.ndarray,
    complete_validation_worlds: np.ndarray,
    stress_corpus: Any,
    stress_validation_worlds: np.ndarray,
    pools: Any,
    config: dict[str, Any],
    base_config: dict[str, Any],
    device: str,
) -> tuple[torch.optim.Optimizer, dict[str, Any], dict[str, Any]]:
    """Run validation-bounded actor-only BC without touching either test split."""

    settings = config["training"]
    base_objective = base_config["objective"]
    sampling = base_config["sampling"]
    human_generator = torch.Generator(device="cpu").manual_seed(
        int(config["optimizer"]["training_seed"])
    )
    simulator_generator = torch.Generator(device="cpu").manual_seed(
        int(config["optimizer"]["retention_sampling_seed"])
    )
    mechanic_sampler = MechanicHierarchySampler(
        train_data.mechanic_label,
        train_data.mechanic_attempt,
        uniform_label_fraction=float(sampling["mechanic_uniform_label_fraction"]),
        maximum_oversampling_ratio=float(
            sampling["maximum_mechanic_frame_oversampling_ratio"]
        ),
        generator=human_generator,
    )

    # This is deliberately below preflight and corpus/strata verification.  Reaching
    # this line is the first point at which optimizer construction is permitted.
    trainable, trainable_names = continuation._configure_trainable_parameters(
        student, config
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["initial_learning_rate"]),
        betas=tuple(settings["optimizer_betas"]),
        eps=float(settings["optimizer_epsilon"]),
        weight_decay=float(settings["weight_decay"]),
    )
    initial_partition = continuation._model_partition_hashes(student)
    mining = config["hard_tail_mining"]
    replay_state = initialize_hard_tail_replay(
        pools.initial_hard_tail_replay,
        generation=0,
        provenance="teacher_state_bootstrap",
    )

    parent_human = _evaluate_human(student, teacher, validation_data, device=device)
    parent_complete = _retention_evaluation(
        teacher,
        student,
        complete_corpus,
        complete_validation_worlds,
        pools,
        config,
        policy_config,
    )
    parent_stress = _retention_evaluation(
        teacher,
        student,
        stress_corpus,
        stress_validation_worlds,
        pools,
        config,
        policy_config,
    )
    hard_guard = config["validation"]["hard_guard"]
    parent_complete_guard = _v5_validation_guard(parent_complete, hard_guard)
    parent_stress_guard = _v5_validation_guard(parent_stress, hard_guard)
    parent_distribution = continuation._distribution_guard(parent_human, config)
    parent_complete_distribution = _simulator_distribution_guard(
        parent_complete,
        parent_complete,
        config,
        authority_phase="complete_validation",
    )
    parent_stress_distribution = _simulator_distribution_guard(
        parent_stress,
        parent_stress,
        config,
        authority_phase="stress_validation",
    )
    parent_baseline_guard = _parent_pretraining_baseline_guard(
        complete_retention=parent_complete_guard,
        stress_retention=parent_stress_guard,
        human_distribution=parent_distribution,
        complete_distribution=parent_complete_distribution,
        stress_distribution=parent_stress_distribution,
    )
    if not parent_baseline_guard["accepted"]:
        raise RuntimeError("BC V1 parent failed V5 complete/stress validation baseline")

    source_step = int(source_payload["counters"]["accepted_optimizer_steps"])
    parent_candidate = _candidate(
        additional_step=0,
        cumulative_step=source_step,
        human=parent_human,
        parent_human=parent_human,
        complete=parent_complete,
        stress=parent_stress,
        complete_guard=parent_complete_guard,
        stress_guard=parent_stress_guard,
        human_distribution=parent_distribution,
        complete_distribution=parent_complete_distribution,
        stress_distribution=parent_stress_distribution,
        config=config,
    )
    best_path = ROOT / WORK_ROOT / "best.pt"
    rollback_path = ROOT / WORK_ROOT / "rollback.pt"
    _save_state(
        best_path,
        student=student,
        optimizer=optimizer,
        human_generator=human_generator,
        simulator_generator=simulator_generator,
        replay_state=replay_state,
        cumulative_step=source_step,
        proposed_steps=0,
        mining_round=0,
        candidate=parent_candidate,
    )

    best_score = float("inf")
    material_best_score = float("inf")
    best_candidate: dict[str, Any] | None = None
    curve: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    mining_telemetry: list[dict[str, Any]] = []
    proposed_steps = 0
    additional_steps = 0
    cumulative_step = source_step
    mining_round = 0
    no_material_improvement = 0
    guard_stopped = False
    nonfinite_stopped = False
    stop_reason = ""
    interval = int(settings["validation_interval_optimizer_steps"])
    ceiling = int(settings["maximum_accepted_supervised_steps"])
    patience = int(config["selection"]["early_stopping_patience_validations"])
    minimum_plateau_steps = int(
        config["selection"]["minimum_accepted_steps_before_plateau"]
    )
    material_delta = float(
        config["selection"]["early_stopping_material_score_improvement"]
    )
    retry_learning_rates = tuple(
        float(value) for value in config["optimizer"]["exact_retry_learning_rates"]
    )
    if retry_learning_rates != (0.00005, 0.000025, 0.0000125):
        raise RuntimeError("V5 transactional LR schedule changed")
    mixture_counts = config["retention_sampling"]["rows_per_step"]
    expected_mixture = {name: int(value) for name, value in mixture_counts.items()}
    tail = config["tail_aware_retention"]
    tail_arguments = {
        "mean_kl_coefficient": float(tail["ordinary_mean_kl_coefficient"]),
        "total_barrier_threshold": float(
            tail["total_sample_activation_threshold"]
        ),
        "total_barrier_temperature": float(tail["total_sample_temperature"]),
        "total_barrier_coefficient": float(
            tail["total_mean_barrier_coefficient"]
        ),
        "total_cvar_fraction": float(tail["cvar_top_quantile_fraction"]),
        "total_cvar_coefficient": float(tail["total_cvar_barrier_coefficient"]),
        "orientation_tail_threshold": float(
            tail["orientation_activation_threshold"]
        ),
        "orientation_tail_temperature": float(tail["orientation_temperature"]),
        "orientation_cvar_fraction": float(tail["cvar_top_quantile_fraction"]),
        "orientation_cvar_coefficient": float(
            tail["orientation_cvar_barrier_coefficient"]
        ),
    }
    mine_frequency = int(mining["mining_frequency_accepted_steps"])

    while additional_steps < ceiling:
        steps_this_boundary = min(interval, ceiling - additional_steps)
        _save_state(
            rollback_path,
            student=student,
            optimizer=optimizer,
            human_generator=human_generator,
            simulator_generator=simulator_generator,
            replay_state=replay_state,
            cumulative_step=cumulative_step,
            proposed_steps=proposed_steps,
            mining_round=mining_round,
            candidate={},
        )
        accepted_boundary = False
        for retry, retry_lr in enumerate(retry_learning_rates):
            if retry:
                restored = _restore_state(
                    rollback_path,
                    student=student,
                    optimizer=optimizer,
                    human_generator=human_generator,
                    simulator_generator=simulator_generator,
                )
                replay_state = restored["replay_state"]
                mining_round = int(restored["mining_round"])
            optimizer.param_groups[0]["lr"] = retry_lr
            loss_sums: defaultdict[str, float] = defaultdict(float)
            grad_max = 0.0
            boundary_nonfinite = False
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
                encoded, realized = sample_v4_retention_rows(
                    pools,
                    replay_state.rows,
                    mixture_counts,
                    generator=simulator_generator,
                )
                if realized != expected_mixture:
                    raise RuntimeError("V5 retention batch composition changed")
                simulator_observation = gather_encoded_rows(
                    _corpus_value(complete_corpus, "observations"), encoded
                )
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
                    log_std_weight=float(
                        base_objective["human_log_std_retention_weight"]
                    ),
                    policy_config=policy_config,
                )
                retention_loss = v4_tail_aware_actor_retention_loss(
                    teacher_sim_actor,
                    student_sim_actor,
                    policy_config=policy_config,
                    **tail_arguments,
                )
                loss = human_loss.loss + retention_loss.loss
                if not bool(torch.isfinite(loss)):
                    boundary_nonfinite = True
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    trainable, float(settings["gradient_clip_norm"])
                )
                if not bool(torch.isfinite(gradient)):
                    boundary_nonfinite = True
                    break
                optimizer.step()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
                    boundary_nonfinite = True
                    break
                if any(
                    parameter.grad is not None
                    for name, parameter in student.named_parameters()
                    if name not in trainable_names
                ):
                    raise RuntimeError("frozen trunk/critic received a V5 gradient")
                grad_max = max(grad_max, float(gradient.item()))
                loss_sums["total"] += float(loss.detach().item())
                loss_sums["human_analog"] += float(
                    human_loss.analog_smooth_l1.detach().item()
                )
                loss_sums["human_buttons"] += float(
                    human_loss.button_bce.detach().item()
                )
                loss_sums["human_log_std"] += float(
                    human_loss.log_std_retention.detach().item()
                )
                for name, value in _training_loss_telemetry(retention_loss).items():
                    loss_sums[f"simulator_{name}"] += value

            if boundary_nonfinite:
                attempts.append(
                    {
                        "attempted_additional_step": additional_steps + steps_this_boundary,
                        "retry": retry,
                        "learning_rate": retry_lr,
                        "executed": True,
                        "accepted": False,
                        "reason": "nonfinite objective, gradient, or actor parameter",
                    }
                )
                restored = _restore_state(
                    rollback_path,
                    student=student,
                    optimizer=optimizer,
                    human_generator=human_generator,
                    simulator_generator=simulator_generator,
                )
                replay_state = restored["replay_state"]
                nonfinite_stopped = True
                stop_reason = "nonfinite/collapse guard stopped training"
                break

            candidate_human = _evaluate_human(
                student, teacher, validation_data, device=device
            )
            candidate_complete = _retention_evaluation(
                teacher,
                student,
                complete_corpus,
                complete_validation_worlds,
                pools,
                config,
                policy_config,
            )
            candidate_stress = _retention_evaluation(
                teacher,
                student,
                stress_corpus,
                stress_validation_worlds,
                pools,
                config,
                policy_config,
            )
            complete_guard = _v5_validation_guard(candidate_complete, hard_guard)
            stress_guard = _v5_validation_guard(candidate_stress, hard_guard)
            distribution = continuation._distribution_guard(candidate_human, config)
            complete_distribution = _simulator_distribution_guard(
                candidate_complete,
                parent_complete,
                config,
                authority_phase="complete_validation",
            )
            stress_distribution = _simulator_distribution_guard(
                candidate_stress,
                parent_stress,
                config,
                authority_phase="stress_validation",
            )
            partition = continuation._model_partition_hashes(student)
            frozen_exact = (
                partition["frozen_trunk_and_critic"]
                == initial_partition["frozen_trunk_and_critic"]
            )
            if not frozen_exact:
                raise RuntimeError("V5 frozen trunk/critic tensor identity changed")
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
                "human_validation": candidate_human,
                "complete_simulator_validation": candidate_complete,
                "stress_simulator_validation": candidate_stress,
                "complete_retention_guard": complete_guard,
                "stress_retention_guard": stress_guard,
                "distribution_guard": distribution,
                "complete_simulator_distribution_guard": complete_distribution,
                "stress_simulator_distribution_guard": stress_distribution,
                "frozen_trunk_and_critic_exact": frozen_exact,
            }
            attempts.append(attempt)
            if (
                complete_guard["accepted"]
                and stress_guard["accepted"]
                and distribution["accepted"]
                and complete_distribution["accepted"]
                and stress_distribution["accepted"]
            ):
                proposed_additional_steps = additional_steps + steps_this_boundary
                proposed_cumulative_step = cumulative_step + steps_this_boundary
                candidate = _candidate(
                    additional_step=proposed_additional_steps,
                    cumulative_step=proposed_cumulative_step,
                    human=candidate_human,
                    parent_human=parent_human,
                    complete=candidate_complete,
                    stress=candidate_stress,
                    complete_guard=complete_guard,
                    stress_guard=stress_guard,
                    human_distribution=distribution,
                    complete_distribution=complete_distribution,
                    stress_distribution=stress_distribution,
                    config=config,
                )
                candidate["learning_rate"] = retry_lr
                candidate["mean_training_loss"] = attempt["mean_training_loss"]
                candidate["max_preclip_gradient_norm"] = grad_max
                candidate["model_partition_hashes"] = partition
                # Hard-tail mining is allowed only after both validation authorities
                # accepted the boundary.  Validation/test rows are never candidates.
                mined: dict[str, Any] | None = None
                if proposed_additional_steps % mine_frequency == 0:
                    proposed_mining_round = mining_round + 1
                    try:
                        mining_result = mine_training_hard_tail(
                            teacher,
                            student,
                            _corpus_value(complete_corpus, "observations"),
                            pools.mining_candidate_pool,
                            replay_state,
                            top_k=int(mining["top_k_per_generation"]),
                            max_replay_rows=int(mining["maximum_replay_rows"]),
                            replay_lifetime_generations=int(
                                mining["maximum_generation_age_inclusive"]
                            )
                            + 1,
                            policy_config=policy_config,
                            rows_per_batch=int(
                                mining["candidate_evaluation_rows_per_batch"]
                            ),
                            mining_round=proposed_mining_round,
                        )
                    except RuntimeError as error:
                        if "nonfinite" not in str(error).lower():
                            raise
                        attempt["accepted"] = False
                        attempt["rollback_required"] = True
                        attempt["hard_tail_mining_guard"] = {
                            "accepted": False,
                            "reason": str(error),
                        }
                        restored = _restore_state(
                            rollback_path,
                            student=student,
                            optimizer=optimizer,
                            human_generator=human_generator,
                            simulator_generator=simulator_generator,
                        )
                        replay_state = restored["replay_state"]
                        mining_round = int(restored["mining_round"])
                        nonfinite_stopped = True
                        stop_reason = (
                            "nonfinite training-candidate hard-tail guard stopped training"
                        )
                        break
                    replay_state = mining_result.replay_state
                    mining_round = proposed_mining_round
                    mined = mining_result.telemetry
                    mining_telemetry.append(mined)
                additional_steps = proposed_additional_steps
                cumulative_step = proposed_cumulative_step
                candidate["post_acceptance_hard_tail_mining"] = mined
                candidate["hard_tail_replay_rows"] = int(replay_state.rows.numel())
                candidate["hard_tail_replay_sha256"] = int64_sha256(replay_state.rows)
                improved = bool(
                    candidate["eligible_for_selection"]
                    and candidate["selection_score"] < best_score
                )
                material = bool(
                    candidate["eligible_for_selection"]
                    and candidate["selection_score"]
                    <= material_best_score - material_delta
                )
                candidate["selected_as_new_best"] = improved
                candidate["material_plateau_improvement"] = material

                if improved:
                    best_score = float(candidate["selection_score"])
                    best_candidate = copy.deepcopy(candidate)
                    _save_state(
                        best_path,
                        student=student,
                        optimizer=optimizer,
                        human_generator=human_generator,
                        simulator_generator=simulator_generator,
                        replay_state=replay_state,
                        cumulative_step=cumulative_step,
                        proposed_steps=proposed_steps,
                        mining_round=mining_round,
                        candidate=candidate,
                    )
                if material:
                    material_best_score = float(candidate["selection_score"])
                    no_material_improvement = 0
                else:
                    no_material_improvement += 1
                curve.append(candidate)
                boundary_path = (
                    ROOT
                    / WORK_ROOT
                    / "accepted"
                    / f"step-{additional_steps:05d}.pt"
                )
                _save_state(
                    boundary_path,
                    student=student,
                    optimizer=optimizer,
                    human_generator=human_generator,
                    simulator_generator=simulator_generator,
                    replay_state=replay_state,
                    cumulative_step=cumulative_step,
                    proposed_steps=proposed_steps,
                    mining_round=mining_round,
                    candidate=candidate,
                )
                accepted_boundary = True
                print(
                    json.dumps(
                        {
                            "accepted_step": additional_steps,
                            "learning_rate": retry_lr,
                            "gameplay_rmse": candidate_human["families"]["gameplay"]
                            ["complete_action_rmse"],
                            "mechanic_rmse": candidate_human["families"]["mechanic"]
                            ["complete_action_rmse"],
                            "complete_max_kl": candidate_complete["all_perspectives"]
                            ["max_sample_kl"],
                            "stress_max_kl": candidate_stress["all_perspectives"]
                            ["max_sample_kl"],
                            "eligible": candidate["eligible_for_selection"],
                            "new_best": improved,
                            "mining_round": mining_round,
                            "no_material_improvement": no_material_improvement,
                        }
                    ),
                    flush=True,
                )
                break
            attempt["rollback_required"] = True

        if nonfinite_stopped:
            break
        if not accepted_boundary:
            restored = _restore_state(
                rollback_path,
                student=student,
                optimizer=optimizer,
                human_generator=human_generator,
                simulator_generator=simulator_generator,
            )
            replay_state = restored["replay_state"]
            guard_stopped = True
            stop_reason = (
                "complete/stress validation retention or distribution guard exhausted retries"
            )
            break
        if additional_steps >= minimum_plateau_steps and no_material_improvement >= patience:
            stop_reason = "held-out validation improvement plateaued"
            break

    if not stop_reason:
        stop_reason = "10000 accepted-step ceiling reached"
    if best_candidate is None:
        stop_reason = f"{stop_reason}; no validation-eligible material V5 checkpoint"
    selected_state = _restore_state(
        best_path,
        student=student,
        optimizer=optimizer,
        human_generator=human_generator,
        simulator_generator=simulator_generator,
    )
    selected_candidate = selected_state["candidate"]
    selected_partition = continuation._model_partition_hashes(student)
    return (
        optimizer,
        {
            "parent_human_validation": parent_human,
            "parent_complete_simulator_validation": parent_complete,
            "parent_stress_simulator_validation": parent_stress,
            "parent_complete_guard": parent_complete_guard,
            "parent_stress_guard": parent_stress_guard,
            "parent_human_distribution_guard": parent_distribution,
            "parent_pretraining_baseline_guard": parent_baseline_guard,
            "parent_complete_simulator_distribution_guard": parent_complete_distribution,
            "parent_stress_simulator_distribution_guard": parent_stress_distribution,
            "selected_candidate": selected_candidate,
            "selected_additional_accepted_steps": int(
                selected_candidate.get("additional_accepted_step", 0)
            ),
            "selected_cumulative_accepted_step": int(selected_state["cumulative_step"]),
            "accepted_steps_executed": additional_steps,
            "proposed_optimizer_steps": proposed_steps,
            "training_curve": curve,
            "transactional_attempts": attempts,
            "hard_tail_mining_telemetry": mining_telemetry,
            "stop_reason": stop_reason,
            "plateaued": stop_reason == "held-out validation improvement plateaued",
            "guard_stopped": guard_stopped,
            "nonfinite_stopped": nonfinite_stopped,
            "ceiling_reached": additional_steps >= ceiling,
            "trainable_parameter_names": list(trainable_names),
            "initial_model_partition_hashes": initial_partition,
            "selected_model_partition_hashes": selected_partition,
            "tests_accessed_during_selection": 0,
            "opened_v2_v3_test_candidate_evaluations": 0,
        },
        selected_state,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, config_identity = _load_config()
    base_config, _base_evidence = continuation._load_base_config(config)
    preflight, authority_manifests = _preflight(
        config, config_identity, base_config, args.human_source_root
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return {"verdict": "PREFLIGHT_PASS"}
    test_ledger_path = ROOT / TEST_ACCESS_LEDGER
    if test_ledger_path.exists():
        raise RuntimeError(
            "V5 one-shot test ledger already exists; refusing to retrain or reopen any "
            "test split. Audit/finalize the persisted ledger without rerunning tests."
        )

    started = time.perf_counter()
    training_seed = int(config["optimizer"]["training_seed"])
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    np.random.seed(training_seed)
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
    source_payload, source_identity = continuation._load_source_checkpoint(
        config, policy_config
    )
    adapter, _adapter_payload, adapter_identity = _load_adapter(
        base_config, args.device
    )
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
    simulator_authority = config["simulator_authority"]

    # Stress validation is generated and verified before the optimizer exists.  Only
    # its prospectively reserved 8192 whole worlds remain resident for selection.
    stress_compatibility = _independent_corpus_config(
        base_config,
        corpus_seed=int(simulator_authority["stress_validation_corpus_seed"]),
        split_seed=int(simulator_authority["stress_validation_split_seed"]),
    )
    torch.use_deterministic_algorithms(False)
    try:
        stress_full, stress_manifest, stress_splits = build_aligned_rollout_corpus(
            stress_compatibility,
            bootstrap_payload,
            bootstrap_identity,
            device=args.device,
        )
    finally:
        torch.use_deterministic_algorithms(True)
    stress_checks = _corpus_hash_checks(
        stress_full,
        stress_manifest,
        stress_splits,
        simulator_authority,
        prefix="stress_validation",
        reserved_split="validation",
    )
    if not all(stress_checks.values()):
        raise RuntimeError(f"V5 stress-validation corpus changed: {stress_checks}")
    stress_source_worlds = np.asarray(stress_splits["validation"], dtype=np.int64)
    stress_corpus = select_world_subset(
        stress_full, stress_source_worlds, device="cpu"
    )
    stress_validation_worlds = np.arange(
        stress_source_worlds.size, dtype=np.int64
    )
    del stress_full
    gc.collect()
    torch.cuda.empty_cache()

    # Recreate the established corpus with aligned tick-local role metadata.  This
    # is both the sole hard-tail source and the complete validation authority.
    torch.use_deterministic_algorithms(False)
    try:
        complete_corpus, complete_manifest, complete_splits = (
            build_aligned_rollout_corpus(
                compatibility,
                bootstrap_payload,
                bootstrap_identity,
                device=args.device,
            )
        )
    finally:
        torch.use_deterministic_algorithms(True)
    complete_checks = _corpus_hash_checks(
        complete_corpus,
        complete_manifest,
        complete_splits,
        simulator_authority,
        prefix="training_validation",
    )
    if not all(complete_checks.values()):
        raise RuntimeError(f"V5 established corpus changed: {complete_checks}")

    teacher = Rival2ActorCritic(policy_config).to(args.device)
    teacher.load_state_dict(source_payload["model"], strict=True)
    teacher.eval().requires_grad_(False)
    student = Rival2ActorCritic(policy_config).to(args.device)
    student.load_state_dict(source_payload["model"], strict=True)
    student.train()
    if tensor_tree_sha256(student.state_dict()) != source_identity["model_tensor_sha256"]:
        raise RuntimeError("V5 student is not byte-identical to BC V1 parent")

    orientation = config["orientation_sensitive_stratum"]
    mining = config["hard_tail_mining"]
    pools = build_v4_retention_pools(
        teacher,
        _corpus_value(complete_corpus, "observations"),
        complete_splits["train"],
        _corpus_value(complete_corpus, "opponent_family"),
        _corpus_value(complete_corpus, "train_mask"),
        low_variance_quantile=float(
            config["retention_sampling"]["low_variance_quantile"]
        ),
        orientation_core_quantile=float(orientation["core_feature_quantile"]),
        recovery_position_quantile=float(
            orientation["context_quantiles"]["recovery_position"]
        ),
        recovery_velocity_quantile=float(
            orientation["context_quantiles"]["recovery_velocity"]
        ),
        recovery_dynamics_quantile=float(
            orientation["context_quantiles"]["recovery_dynamics"]
        ),
        contact_absolute_up_quantile=float(
            orientation["context_quantiles"]["contact_absolute_up_z"]
        ),
        contact_up_quantile=float(
            orientation["context_quantiles"]["contact_up_z"]
        ),
        candidate_pool_rows=int(mining["candidate_pool_rows"]),
        candidate_pool_fractions=mining["candidate_pool_fractions"],
        candidate_pool_seed=int(mining["candidate_pool_seed"]),
        initial_replay_rows=int(mining["initial_replay_rows"]),
        policy_config=policy_config,
        rows_per_batch=int(mining["candidate_evaluation_rows_per_batch"]),
    )
    verify_v4_retention_pools(pools, authority_manifests["strata"])
    if int64_sha256(pools.mining_candidate_pool) != mining["candidate_pool_sha256"]:
        raise RuntimeError("V5 mining candidate pool changed after authority freeze")

    optimizer, training, selected_state = _train(
        teacher=teacher,
        student=student,
        source_payload=source_payload,
        policy_config=policy_config,
        train_data=train_data,
        validation_data=validation_data,
        complete_corpus=complete_corpus,
        complete_train_worlds=complete_splits["train"],
        complete_validation_worlds=complete_splits["validation"],
        stress_corpus=stress_corpus,
        stress_validation_worlds=stress_validation_worlds,
        pools=pools,
        config=config,
        base_config=base_config,
        device=args.device,
    )
    final_selection_utc = datetime.now(UTC).isoformat()
    selected_candidate = training["selected_candidate"]
    selected_additional = int(training["selected_additional_accepted_steps"])
    if not (
        selected_additional > 0
        and bool(selected_candidate.get("eligible_for_selection", False))
    ):
        blocked = {
            "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_EVIDENCE_V5",
            "generated_utc": datetime.now(UTC).isoformat(),
            "verdict": "BLOCKED",
            "blocker": (
                "no validation-eligible material V5 checkpoint was selected; all human "
                "and simulator test authorities remain untouched"
            ),
            "config": config_identity,
            "preflight": preflight,
            "source_checkpoint": source_identity,
            "observation_adapter": adapter_identity,
            "human_split_identity": human_identity,
            "training_validation_corpus": complete_manifest,
            "training_validation_corpus_checks": complete_checks,
            "stress_validation_corpus": stress_manifest,
            "stress_validation_corpus_checks": stress_checks,
            "training": training,
            "test_discipline": {
                "human_test_access_count": 0,
                "untouched_simulator_test_student_evaluation_count": 0,
                "pre_v4_opened_simulator_test_candidate_evaluations": 0,
                "test_authorities_preserved": True,
            },
            "final_selection_utc": final_selection_utc,
            "ppo_updates": 0,
            "reward_changes": 0,
        }
        _write_json(ROOT / RESULT_ROOT / "runtime_pre_step_preflight.json", preflight)
        _write_json(ROOT / RESULT_ROOT / "training_curve.json", training)
        _write_json(ROOT / RESULT_ROOT / "evidence.json", blocked)
        blocked_artifacts = [
            RESULT_ROOT / "README.md",
            RESULT_ROOT / "evidence.json",
            RESULT_ROOT / "frozen_config.json",
            RESULT_ROOT / "pre_step_authority.json",
            RESULT_ROOT / "pre_step_preflight.json",
            RESULT_ROOT / "runtime_pre_step_preflight.json",
            RESULT_ROOT / "training_curve.json",
            RESULT_ROOT / "v4_authority_reuse_manifest.json",
            Path(config["simulator_authority"]["retention_strata_manifest"]),
            Path(config["simulator_authority"]["stress_validation_authority"]),
            Path(config["simulator_authority"]["untouched_test_authority"]),
            Path("results/rival2/human_bc_v4/simulator_disjointness_proof.json"),
        ]
        _write_json(
            ROOT / RESULT_ROOT / "artifact_manifest.json",
            _artifact_manifest(blocked_artifacts),
        )
        print(json.dumps({"verdict": "BLOCKED", "blocker": blocked["blocker"]}))
        return blocked
    test_ledger = {
        "format": "RIVAL2_HUMAN_BC_V5_ONE_SHOT_TEST_ACCESS_LEDGER_V1",
        "created_utc": datetime.now(UTC).isoformat(),
        "phase": "selection_frozen_no_test_access",
        "config_sha256": config_identity["sha256"],
        "selected_model_tensor_sha256": tensor_tree_sha256(student.state_dict()),
        "selected_additional_accepted_steps": int(
            training["selected_additional_accepted_steps"]
        ),
        "selection_reopen_allowed": False,
        "human_test_access_count": 0,
        "simulator_test_student_evaluation_count": 0,
        "pre_v4_opened_test_candidate_evaluations": 0,
    }
    _atomic_write_json(test_ledger_path, test_ledger)

    # Both tests are opened exactly once and only after the selected validation state
    # has already been restored.  No subsequent code can alter model selection.
    _advance_test_ledger(
        test_ledger,
        test_ledger_path,
        phase="human_test_access_started",
        human_test_access_count=1,
        human_test_access_started_utc=datetime.now(UTC).isoformat(),
    )
    test_data = _load_human_split(
        "test",
        config=base_config,
        adapter=adapter,
        source_root=args.human_source_root,
        device=args.device,
    )
    human_identity["test"] = _human_identity(test_data)
    human_test_selected = _evaluate_human(student, teacher, test_data, device=args.device)
    human_test_access_utc = datetime.now(UTC).isoformat()
    _advance_test_ledger(
        test_ledger,
        test_ledger_path,
        phase="human_test_complete",
        human_test_completed_utc=human_test_access_utc,
        human_test_identity=human_identity["test"],
        human_test_selected=human_test_selected,
    )

    del complete_corpus, stress_corpus
    gc.collect()
    torch.cuda.empty_cache()
    test_compatibility = _independent_corpus_config(
        base_config,
        corpus_seed=int(simulator_authority["untouched_test_corpus_seed"]),
        split_seed=int(simulator_authority["untouched_test_split_seed"]),
    )
    _advance_test_ledger(
        test_ledger,
        test_ledger_path,
        phase="simulator_test_regeneration_started_no_student_evaluation",
        simulator_test_regeneration_started_utc=datetime.now(UTC).isoformat(),
    )
    torch.use_deterministic_algorithms(False)
    try:
        test_full, test_manifest, test_splits = build_aligned_rollout_corpus(
            test_compatibility,
            bootstrap_payload,
            bootstrap_identity,
            device=args.device,
        )
    finally:
        torch.use_deterministic_algorithms(True)
    test_checks = _corpus_hash_checks(
        test_full,
        test_manifest,
        test_splits,
        simulator_authority,
        prefix="untouched_test",
        reserved_split="test",
    )
    if not all(test_checks.values()):
        raise RuntimeError(f"V5 untouched test corpus changed: {test_checks}")
    test_source_worlds = np.asarray(test_splits["test"], dtype=np.int64)
    test_corpus = select_world_subset(test_full, test_source_worlds, device="cpu")
    test_worlds = np.arange(test_source_worlds.size, dtype=np.int64)
    del test_full
    gc.collect()
    torch.cuda.empty_cache()
    _advance_test_ledger(
        test_ledger,
        test_ledger_path,
        phase="simulator_test_student_evaluation_started",
        simulator_test_parent_evaluation_count=1,
        simulator_test_student_evaluation_count=1,
        simulator_test_student_evaluation_started_utc=datetime.now(UTC).isoformat(),
        simulator_test_corpus_checks=test_checks,
    )
    parent_simulator_test = _retention_evaluation(
        teacher,
        teacher,
        test_corpus,
        test_worlds,
        pools,
        config,
        policy_config,
    )
    simulator_test = _retention_evaluation(
        teacher,
        student,
        test_corpus,
        test_worlds,
        pools,
        config,
        policy_config,
    )
    simulator_test_contract = detailed_retention_guard(
        simulator_test, config["validation"]["hard_guard"]
    )
    simulator_test_distribution = _simulator_distribution_guard(
        simulator_test,
        parent_simulator_test,
        config,
        authority_phase="untouched_test",
    )
    simulator_test_access_utc = datetime.now(UTC).isoformat()
    _advance_test_ledger(
        test_ledger,
        test_ledger_path,
        phase="all_one_shot_tests_complete",
        simulator_test_completed_utc=simulator_test_access_utc,
        simulator_test_parent_retention=parent_simulator_test,
        simulator_test_retention=simulator_test,
        simulator_test_contract=simulator_test_contract,
        simulator_test_distribution_guard=simulator_test_distribution,
    )

    selected_model = _cpu_tree(student.state_dict())
    selected_optimizer = _cpu_tree(optimizer.state_dict())
    labels = selected_candidate.get("mechanic_label_comparison_to_parent", {})
    acceptance_checks = {
        "selected_material_checkpoint": selected_additional > 0,
        "gameplay_validation_material_improvement": selected_candidate.get(
            "gameplay_rmse_ratio_to_parent", 1.0
        )
        < 1.0,
        "mechanic_validation_material_improvement": selected_candidate.get(
            "mechanic_rmse_ratio_to_parent", 1.0
        )
        < 1.0,
        "broad_mechanic_label_improvement": labels.get("improved_fraction", 0.0)
        >= float(config["selection"]["minimum_mechanic_labels_improved_fraction"]),
        "broad_mechanic_label_nonregression": labels.get("nonregressed_fraction", 0.0)
        >= float(config["selection"]["minimum_mechanic_labels_nonregressed_fraction"]),
        "complete_validation_guard": bool(
            selected_candidate.get("complete_retention_guard", {}).get("accepted", False)
        ),
        "stress_validation_guard": bool(
            selected_candidate.get("stress_retention_guard", {}).get("accepted", False)
        ),
        "untouched_simulator_test_contract": bool(simulator_test_contract["accepted"]),
        "untouched_simulator_test_max_kl_le_2": float(
            simulator_test["all_perspectives"]["max_sample_kl"]
        )
        <= 2.0,
        "selected_distribution_healthy": bool(
            selected_candidate.get("distribution_guard", {}).get("accepted", False)
        ),
        "complete_simulator_distribution_healthy": bool(
            selected_candidate.get("complete_simulator_distribution_guard", {}).get(
                "accepted", False
            )
        ),
        "stress_simulator_distribution_healthy": bool(
            selected_candidate.get("stress_simulator_distribution_guard", {}).get(
                "accepted", False
            )
        ),
        "untouched_simulator_test_distribution_healthy": bool(
            simulator_test_distribution["accepted"]
        ),
        "human_test_outputs_finite": bool(human_test_selected["finite"]),
        "frozen_trunk_and_critic_exact": training["initial_model_partition_hashes"]
        ["frozen_trunk_and_critic"]
        == training["selected_model_partition_hashes"]["frozen_trunk_and_critic"],
        "test_selection_not_reopened": True,
    }
    accepted = all(acceptance_checks.values())

    checkpoint_path = ROOT / config["checkpoint"]["path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "format": V5_CHECKPOINT_FORMAT,
        "version": V5_VERSION,
        "base_human_bc_version": HUMAN_BC_VERSION,
        "model": selected_model,
        "actor_only_optimizer": selected_optimizer,
        "optimizer_provenance": {
            "fresh_actor_only_adamw": True,
            "source_bc_optimizer_loaded": False,
            "historical_ppo_optimizer_loaded": False,
            "v2_v3_v4_optimizer_loaded": False,
            "trainable_parameter_names": training["trainable_parameter_names"],
        },
        "policy_config": asdict(policy_config),
        "observation_version": source_payload["observation_version"],
        "action_version": source_payload["action_version"],
        "physics_hz": 120,
        "policy_hz": 120,
        "authority": {
            "v5_config": config_identity,
            "source_checkpoint": source_identity,
            "bootstrap": bootstrap_identity,
            "observation_adapter": adapter_identity,
            "human_split_identity": human_identity,
            "training_validation_corpus_identity_sha256": complete_manifest[
                "identity_sha256"
            ],
            "orientation_sensitive_stratum": pools.orientation_authority,
            "hard_tail_mining_authority": config["hard_tail_mining"],
            "complete_validation_authority": {
                "world_indices_sha256": simulator_authority[
                    "validation_world_indices_sha256"
                ],
                "worlds": simulator_authority["complete_validation_worlds"],
            },
            "stress_validation": authority_manifests["stress_identity"],
            "untouched_test": authority_manifests["untouched_identity"],
        },
        "counters": {
            "source_accepted_optimizer_steps": source_identity[
                "accepted_optimizer_steps"
            ],
            "accepted_actor_only_supervised_steps": selected_additional,
            "cumulative_human_bc_steps": training["selected_cumulative_accepted_step"],
            "accepted_steps_executed": training["accepted_steps_executed"],
            "proposed_optimizer_steps": training["proposed_optimizer_steps"],
        },
        "rng_state": {
            "human_generator": selected_state["human_generator_state"],
            "simulator_generator": selected_state["simulator_generator_state"],
            "global": selected_state["global_rng"],
        },
        "hard_tail_replay_state": selected_state["replay_state"],
        "selected_validation": selected_candidate,
        "human_test": {"selected": human_test_selected},
        "untouched_simulator_test_retention": simulator_test,
        "untouched_simulator_test_bc_v1_parent": parent_simulator_test,
        "untouched_simulator_test_distribution_guard": simulator_test_distribution,
        "stop": {
            "reason": training["stop_reason"],
            "plateaued": training["plateaued"],
            "guard_stopped": training["guard_stopped"],
            "nonfinite_stopped": training["nonfinite_stopped"],
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
        "critic_drift_exact_zero_complete": float(
            selected_candidate["complete_simulator_validation"]["critic"]
            ["max_absolute_drift"]
        )
        == 0.0,
        "critic_drift_exact_zero_stress": float(
            selected_candidate["stress_simulator_validation"]["critic"]
            ["max_absolute_drift"]
        )
        == 0.0,
        "critic_drift_exact_zero_test": float(
            simulator_test["critic"]["max_absolute_drift"]
        )
        == 0.0,
        "human_test_access_count": int(test_ledger["human_test_access_count"]),
        "untouched_simulator_test_access_count": int(
            test_ledger["simulator_test_student_evaluation_count"]
        ),
        "opened_v2_v3_test_candidate_evaluations": 0,
        "final_selection_utc": final_selection_utc,
        "human_test_access_utc": human_test_access_utc,
        "simulator_test_access_utc": simulator_test_access_utc,
        "selection_reopened_after_tests": False,
        "ppo_updates": 0,
        "reward_changes": 0,
        "mechanic_definition_or_adjudication_changes": 0,
        "dataset_split_changes": 0,
        "observation_adapter_changes": 0,
        "observation_action_contract_changes": 0,
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
        and integrity["critic_drift_exact_zero_complete"]
        and integrity["critic_drift_exact_zero_stress"]
        and integrity["critic_drift_exact_zero_test"]
        and integrity["human_test_access_count"] == 1
        and integrity["untouched_simulator_test_access_count"] == 1
        and integrity["opened_v2_v3_test_candidate_evaluations"] == 0
        and not integrity["selection_reopened_after_tests"]
        and integrity["ppo_updates"] == 0
        and integrity["reward_changes"] == 0
        and integrity["mechanic_definition_or_adjudication_changes"] == 0
        and integrity["dataset_split_changes"] == 0
        and integrity["observation_adapter_changes"] == 0
        and integrity["observation_action_contract_changes"] == 0
    )
    accepted = bool(accepted and integrity["valid"])

    evidence = {
        "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_EVIDENCE_V5",
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
        "training_validation_corpus": complete_manifest,
        "training_validation_corpus_checks": complete_checks,
        "retention_strata": authority_manifests["strata"],
        "stress_validation_corpus": stress_manifest,
        "stress_validation_corpus_checks": stress_checks,
        "training": training,
        "human_test": {"selected": human_test_selected},
        "untouched_simulator_test_corpus": test_manifest,
        "untouched_simulator_test_checks": test_checks,
        "untouched_simulator_test_retention": simulator_test,
        "untouched_simulator_test_bc_v1_parent": parent_simulator_test,
        "untouched_simulator_test_contract": simulator_test_contract,
        "untouched_simulator_test_distribution_guard": simulator_test_distribution,
        "acceptance": {"checks": acceptance_checks, "accepted": accepted},
        "checkpoint": checkpoint_identity,
        "integrity": integrity,
        "one_shot_test_access_ledger": {
            "path": TEST_ACCESS_LEDGER.as_posix(),
            "phase_at_evidence_construction": test_ledger["phase"],
            "human_test_access_count": test_ledger["human_test_access_count"],
            "simulator_test_student_evaluation_count": test_ledger[
                "simulator_test_student_evaluation_count"
            ],
            "selection_reopen_allowed": False,
        },
        "pre_v4_opened_simulator_tests": {
            "role": "diagnostic_only",
            "candidate_evaluations": 0,
            "used_for_acceptance": False,
        },
        "prohibited_work": {
            "ppo": False,
            "reward_change": False,
            "mechanic_definition_or_adjudication_change": False,
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
        ROOT / RESULT_ROOT / "untouched_test_results.json",
        {
            "corpus_checks": test_checks,
            "retention": simulator_test,
            "bc_v1_parent": parent_simulator_test,
            "guard": simulator_test_contract,
            "distribution_guard": simulator_test_distribution,
        },
    )
    _write_json(ROOT / RESULT_ROOT / "evidence.json", evidence)
    _advance_test_ledger(
        test_ledger,
        test_ledger_path,
        phase="final_evidence_persisted",
        final_verdict=evidence["verdict"],
        checkpoint=checkpoint_identity,
        evidence_sha256=file_sha256(ROOT / RESULT_ROOT / "evidence.json"),
    )
    artifacts = [
        Path(config["checkpoint"]["path"]),
        RESULT_ROOT / "README.md",
        RESULT_ROOT / "evidence.json",
        RESULT_ROOT / "frozen_config.json",
        RESULT_ROOT / "human_test_metrics.json",
        RESULT_ROOT / "pre_step_authority.json",
        RESULT_ROOT / "pre_step_preflight.json",
        RESULT_ROOT / "runtime_pre_step_preflight.json",
        TEST_ACCESS_LEDGER,
        RESULT_ROOT / "training_curve.json",
        RESULT_ROOT / "untouched_test_results.json",
        RESULT_ROOT / "v4_authority_reuse_manifest.json",
        Path(config["simulator_authority"]["retention_strata_manifest"]),
        Path(config["simulator_authority"]["stress_validation_authority"]),
        Path(config["simulator_authority"]["untouched_test_authority"]),
        Path("results/rival2/human_bc_v4/simulator_disjointness_proof.json"),
    ]
    _write_json(ROOT / RESULT_ROOT / "artifact_manifest.json", _artifact_manifest(artifacts))
    print(
        json.dumps(
            {
                "verdict": evidence["verdict"],
                "checkpoint": checkpoint_identity,
                "accepted_steps": selected_additional,
                "gameplay_validation": {
                    "parent": training["parent_human_validation"]["families"]
                    ["gameplay"]["complete_action_rmse"],
                    "v5": selected_candidate.get("human_validation", {})
                    .get("families", {})
                    .get("gameplay", {})
                    .get("complete_action_rmse"),
                },
                "mechanic_validation": {
                    "parent": training["parent_human_validation"]["families"]
                    ["mechanic"]["complete_action_rmse"],
                    "v5": selected_candidate.get("human_validation", {})
                    .get("families", {})
                    .get("mechanic", {})
                    .get("complete_action_rmse"),
                },
                "complete_validation_max_kl": selected_candidate.get(
                    "complete_simulator_validation", {}
                )
                .get("all_perspectives", {})
                .get("max_sample_kl"),
                "stress_validation_max_kl": selected_candidate.get(
                    "stress_simulator_validation", {}
                )
                .get("all_perspectives", {})
                .get("max_sample_kl"),
                "untouched_test_max_kl": simulator_test["all_perspectives"]
                ["max_sample_kl"],
                "untouched_test_mean_kl": simulator_test["all_perspectives"]
                ["mean_kl"],
                "boost_saturation_all_perspectives": {
                    "complete": selected_candidate[
                        "complete_simulator_distribution_guard"
                    ]["parent_relative_boost_saturation"]["all_perspectives"],
                    "stress": selected_candidate[
                        "stress_simulator_distribution_guard"
                    ]["parent_relative_boost_saturation"]["all_perspectives"],
                    "untouched_test": simulator_test_distribution[
                        "parent_relative_boost_saturation"
                    ]["all_perspectives"],
                },
                "stop_reason": training["stop_reason"],
            },
            indent=2,
        )
    )
    del test_corpus, train_data, validation_data, test_data
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
        default=Path(os.environ["APPDATA"])
        / "bakkesmod/bakkesmod/data/rival2/human_demos",
    )
    return parser.parse_args()


def main() -> int:
    evidence = run(parse_args())
    return 0 if evidence["verdict"] in ("PASS", "PREFLIGHT_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
