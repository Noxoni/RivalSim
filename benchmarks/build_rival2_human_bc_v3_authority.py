"""Build the no-optimizer-step prospective authority for Human BC V3."""

from __future__ import annotations

import copy
import gc
import json
import subprocess
import sys
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
)
from rivalsim.human_demo.bc_v3_retention import (  # noqa: E402
    build_retention_pools,
    int64_sha256,
)
from rivalsim.human_demo.missing_feature_distillation import file_sha256  # noqa: E402
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig  # noqa: E402

RESULT_ROOT = ROOT / "results/rival2/human_bc_v3"
BASE_CONFIG_PATH = ROOT / "results/rival2/human_behavior_cloning_v1/frozen_config.json"
BASE_EVIDENCE_PATH = ROOT / "results/rival2/human_behavior_cloning_v1/evidence.json"
SOURCE_CHECKPOINT = ROOT / "checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt"
ADAPTER_CHECKPOINT = (
    ROOT / "checkpoints/rival2/observation_adapter_v2/rival2_human_demo_observation_adapter_v2.pt"
)
V2_EVIDENCE = ROOT / "results/rival2/human_bc_v2/evidence.json"
V2_CHECKPOINT = ROOT / "checkpoints/rival2/human_bc_v2/rival2_human_bc_v2.pt"
REQUIRED_START = "b9140d96f73a78a539df7ebd019a8f9670bc34e7"
SOURCE_SHA = "560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874"
ADAPTER_SHA = "EDEDC9CCDE3269B393FB4C944F641CF4D34A78AB5944662F9019009BBA914C99"
NEW_TEST_CORPUS_SEED = 2026090107
NEW_TEST_SPLIT_SEED = 2026090108
LOW_VARIANCE_QUANTILE = 0.10


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _sha(path: Path) -> str:
    return file_sha256(path)


def _source_teacher(device: str) -> tuple[dict[str, Any], Rival2ActorCritic]:
    if _sha(SOURCE_CHECKPOINT) != SOURCE_SHA:
        raise RuntimeError("Human BC V1 source checkpoint changed")
    payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    policy = Rival2PolicyConfig(**payload["policy_config"])
    model = Rival2ActorCritic(policy).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return payload, model


def _new_corpus_config(base_config: dict[str, Any]) -> dict[str, Any]:
    corpus = copy.deepcopy(base_config["retention"]["corpus"])
    corpus["seed"] = NEW_TEST_CORPUS_SEED
    corpus["split"]["seed"] = NEW_TEST_SPLIT_SEED
    corpus.pop("expected_identity_sha256", None)
    corpus.pop("expected_observation_tensor_sha256", None)
    return corpus


def main() -> int:
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", REQUIRED_START, "HEAD"],
        check=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for authoritative simulator corpus generation")
    device = "cuda:0"
    if _sha(ADAPTER_CHECKPOINT) != ADAPTER_SHA:
        raise RuntimeError("frozen observation adapter changed")
    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    source_payload, teacher = _source_teacher(device)
    compatibility = {
        "authority": base_config["authority"],
        "corpus": base_config["retention"]["corpus"],
    }
    bootstrap_payload, policy_config, bootstrap_identity = _load_bootstrap(compatibility)

    old_corpus, old_manifest, old_splits = _build_rollout_corpus(
        compatibility,
        bootstrap_payload,
        bootstrap_identity,
        device=device,
    )
    old_expected = base_config["retention"]["corpus"]
    if (
        old_manifest["identity_sha256"] != old_expected["expected_identity_sha256"]
        or old_manifest["collection"]["observation_tensor_sha256"]
        != old_expected["expected_observation_tensor_sha256"]
    ):
        raise RuntimeError("existing training/validation simulator corpus changed")
    family = bootstrap_payload["opponent_curriculum"]["family"]
    rival_side = bootstrap_payload["opponent_curriculum"]["rival_side"]
    pools = build_retention_pools(
        teacher,
        old_corpus,
        old_splits["train"],
        family,
        rival_side,
        low_variance_quantile=LOW_VARIANCE_QUANTILE,
        policy_config=policy_config,
    )
    strata_manifest = {
        **pools.manifest,
        "source_simulator_corpus_identity_sha256": old_manifest["identity_sha256"],
        "source_train_world_indices_sha256": old_manifest["split"]["world_index_sha256"]["train"],
        "opponent_family_source": "bootstrap checkpoint frozen assignments",
        "student_outputs_used_for_membership": False,
        "optimizer_steps": 0,
    }
    del old_corpus, pools
    gc.collect()
    torch.cuda.empty_cache()

    new_corpus_config = _new_corpus_config(base_config)
    new_compatibility = {
        "authority": base_config["authority"],
        "corpus": new_corpus_config,
    }
    new_corpus, new_manifest, new_splits = _build_rollout_corpus(
        new_compatibility,
        bootstrap_payload,
        bootstrap_identity,
        device=device,
    )
    new_test_worlds = np.asarray(new_splits["test"], dtype=np.int64)
    old_seed = int(base_config["retention"]["corpus"]["seed"])
    new_test_authority = {
        "format": "RIVAL2_HUMAN_BC_V3_UNTOUCHED_SIMULATOR_TEST_AUTHORITY_V1",
        "created_utc": datetime.now(UTC).isoformat(),
        "corpus": new_manifest,
        "reserved_split": "test",
        "reserved_worlds": int(new_test_worlds.size),
        "reserved_world_indices_int64_sha256": int64_sha256(new_test_worlds),
        "trajectory_identity": "sha256(corpus_seed || world_index || observation trajectory)",
        "disjointness": {
            "old_corpus_seed": old_seed,
            "new_corpus_seed": NEW_TEST_CORPUS_SEED,
            "seed_namespace_distinct": old_seed != NEW_TEST_CORPUS_SEED,
            "old_train_validation_and_opened_test_trajectory_overlap": 0,
            "reason": (
                "all prior worlds are namespaced by the old corpus seed; the new corpus "
                "uses a distinct deterministic seed and independently hashed trajectories"
            ),
        },
        "student_evaluations_before_final_selection": 0,
        "optimizer_steps_before_binding": 0,
        "observation_tensor_generated_only_for_hash_binding": True,
    }
    del new_corpus
    gc.collect()
    torch.cuda.empty_cache()

    config = {
        "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_FROZEN_CONFIG_V3",
        "authority": {
            "required_parent": REQUIRED_START,
            "base_bc_config": BASE_CONFIG_PATH.relative_to(ROOT).as_posix(),
            "base_bc_config_sha256": _sha(BASE_CONFIG_PATH),
            "base_bc_evidence": BASE_EVIDENCE_PATH.relative_to(ROOT).as_posix(),
            "base_bc_evidence_sha256": _sha(BASE_EVIDENCE_PATH),
            "source_checkpoint": SOURCE_CHECKPOINT.relative_to(ROOT).as_posix(),
            "source_checkpoint_sha256": SOURCE_SHA,
            "source_model_tensor_sha256": tensor_tree_sha256(source_payload["model"]),
            "source_selected_accepted_step": int(
                source_payload["counters"]["accepted_optimizer_steps"]
            ),
            "observation_adapter_checkpoint": ADAPTER_CHECKPOINT.relative_to(ROOT).as_posix(),
            "observation_adapter_checkpoint_sha256": ADAPTER_SHA,
            "v2_diagnostic_checkpoint": V2_CHECKPOINT.relative_to(ROOT).as_posix(),
            "v2_diagnostic_checkpoint_sha256": _sha(V2_CHECKPOINT),
            "v2_diagnostic_evidence": V2_EVIDENCE.relative_to(ROOT).as_posix(),
            "v2_diagnostic_evidence_sha256": _sha(V2_EVIDENCE),
            "v2_checkpoint_training_prohibited": True,
            "old_opened_simulator_test_role": (
                "diagnostic_only_never_selection_stopping_tuning_or_acceptance"
            ),
        },
        "checkpoint": {
            "path": "checkpoints/rival2/human_bc_v3/rival2_human_bc_v3.pt",
            "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V3",
            "ppo_resumable": False,
            "historical_ppo_optimizer_resumable": False,
            "save_every_accepted_validation_boundary": True,
        },
        "trainable_parameters": {
            "mode": "actor_head_only",
            "names": ["actor.weight", "actor.bias"],
            "shared_trunk": "frozen byte-identical to Human BC V1",
            "critic_head": "frozen byte-identical to Human BC V1",
        },
        "inherit_exactly_from_base_bc_v1": {
            "human_action_objective": True,
            "human_dataset_and_source_hashes": True,
            "human_train_validation_test_splits": True,
            "mechanic_aware_sampling": True,
            "observation_action_contracts": True,
            "observation_adapter": True,
            "sampling_family_proportions": True,
            "simulator_training_and_validation_corpus": True,
            "simulator_retention_hard_limits": True,
        },
        "simulator_authority": {
            "training_validation_corpus_identity_sha256": old_manifest["identity_sha256"],
            "training_validation_observation_sha256": old_manifest["collection"][
                "observation_tensor_sha256"
            ],
            "training_world_indices_sha256": old_manifest["split"]["world_index_sha256"]["train"],
            "validation_world_indices_sha256": old_manifest["split"]["world_index_sha256"][
                "validation"
            ],
            "complete_validation_worlds": len(old_splits["validation"]),
            "old_test_worlds_excluded": True,
            "retention_strata_manifest": (
                "results/rival2/human_bc_v3/retention_strata_manifest.json"
            ),
            "new_untouched_test_authority": (
                "results/rival2/human_bc_v3/new_simulator_test_authority.json"
            ),
            "new_untouched_test_corpus_seed": NEW_TEST_CORPUS_SEED,
            "new_untouched_test_split_seed": NEW_TEST_SPLIT_SEED,
            "new_untouched_test_corpus_identity_sha256": new_manifest["identity_sha256"],
            "new_untouched_test_observation_sha256": new_manifest["collection"][
                "observation_tensor_sha256"
            ],
            "new_untouched_test_world_indices_sha256": int64_sha256(new_test_worlds),
            "new_untouched_test_worlds": int(new_test_worlds.size),
            "test_access_before_final_validation_selection": 0,
        },
        "retention_sampling": {
            "membership_source": "frozen BC-V1 teacher only",
            "with_replacement": True,
            "ordered_strata": [
                "natural",
                "current_policy_applicable",
                "historical_opponent",
                "low_teacher_variance",
            ],
            "rows_per_step": {
                "natural": 2048,
                "current_policy_applicable": 1024,
                "historical_opponent": 1024,
                "low_teacher_variance": 1024,
            },
            "low_variance_quantile": LOW_VARIANCE_QUANTILE,
            "low_variance_threshold_log_std": strata_manifest["low_variance_definition"][
                "threshold_log_std"
            ],
            "pool_hashes": {
                name: row["ordered_int64_sha256"] for name, row in strata_manifest["pools"].items()
            },
        },
        "tail_aware_retention": {
            "ordinary_mean_teacher_to_student_kl_coefficient": 2.0,
            "formulation": (
                "mean_KL + coefficient * mean((temperature * softplus((sample_KL - "
                "threshold) / temperature))^2)"
            ),
            "activation_threshold_sample_kl": 0.5,
            "temperature": 0.05,
            "barrier_coefficient": 4.0,
            "cvar_top_quantile_fraction": None,
            "hard_all_perspective_max_sample_kl": 2.0,
            "existing_mean_channel_and_critic_guards_unchanged": True,
        },
        "optimizer": {
            "type": "fresh torch.optim.AdamW over actor.weight and actor.bias only",
            "continuation_seed": 2026090111,
            "historical_ppo_optimizer_loaded": False,
            "source_full_model_bc_optimizer_loaded": False,
            "transactional_retries_per_interval": 3,
            "transactional_backoff_factor": 0.5,
            "minimum_learning_rate": 0.00000375,
        },
        "training": {
            "initial_learning_rate": 0.00003,
            "optimizer_betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-8,
            "weight_decay": 0.00001,
            "gradient_clip_norm": 1.0,
            "validation_interval_optimizer_steps": 64,
            "maximum_accepted_supervised_steps": 10000,
        },
        "selection": {
            "requires_complete_validation_hard_guard": True,
            "requires_both_human_families_improve_over_parent": True,
            "minimum_human_family_relative_improvement": 0.05,
            "mechanic_label_nonregression_relative_tolerance": 0.02,
            "minimum_mechanic_labels_improved_fraction": 0.6,
            "minimum_mechanic_labels_nonregressed_fraction": 0.8,
            "requires_action_distribution_health": True,
            "candidate_score": (
                "0.32 gameplay ratio + 0.32 mechanic ratio + 0.18 mean label ratio + "
                "0.10 mean-KL/soft-limit + 0.08 max-KL/hard-limit"
            ),
            "score_weights": {
                "gameplay_rmse_ratio": 0.32,
                "mechanic_rmse_ratio": 0.32,
                "mean_per_label_rmse_ratio": 0.18,
                "simulator_actor_kl_soft_limit_ratio": 0.10,
                "simulator_max_kl_hard_limit_ratio": 0.08,
            },
            "best_checkpoint_update": "strict combined score improvement among eligible candidates",
            "early_stopping_material_score_improvement": 0.0005,
            "early_stopping_patience_validations": 20,
            "minimum_accepted_steps_before_plateau": 2048,
            "test_used_for_selection": False,
        },
        "validation": {
            "worlds": len(old_splits["validation"]),
            "worlds_per_batch": int(base_config["retention"]["worlds_per_validation_batch"]),
            "groups": [
                "all_perspectives",
                "current_policy_applicable",
                "counterfactual_opponent",
                "historical_opponent",
                "low_teacher_variance",
            ],
            "threshold_counts": [0.5, 1.0, 2.0],
            "per_channel_tail_contribution": True,
            "hard_guard": base_config["retention"]["hard_guard"],
            "soft_actor_mean_kl": base_config["retention"]["soft_actor_mean_kl"],
        },
        "distribution_guard": {
            "maximum_analog_actor_absolute_ge_5_fraction": 0.95,
            "maximum_button_probability_saturation_fraction": 0.95,
            "maximum_log_std_clamp_fraction": 0.95,
            "minimum_nonconstant_channel_std": 0.01,
            "nonfinite_allowed": False,
        },
        "test_discipline": {
            "human_test_access": "exactly once after final validation selection",
            "new_simulator_test_access": "exactly once after final validation selection",
            "old_opened_simulator_test": "diagnostic only and not reevaluated for selection",
            "selection_reopened_after_test": False,
        },
        "prohibited": {
            "ppo": True,
            "reward_change": True,
            "additional_demonstrations": True,
            "dataset_or_split_change": True,
            "mechanic_definition_change": True,
            "observation_adapter_change": True,
            "observation_action_contract_change": True,
            "raw_recording_or_review_mutation": True,
        },
    }
    preflight = {
        "format": "RIVAL2_HUMAN_BC_V3_PRE_STEP_PREFLIGHT_V1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "required_starting_commit_is_ancestor": True,
        "source_checkpoint_sha256_exact": _sha(SOURCE_CHECKPOINT) == SOURCE_SHA,
        "adapter_checkpoint_sha256_exact": _sha(ADAPTER_CHECKPOINT) == ADAPTER_SHA,
        "old_training_validation_corpus_exact": True,
        "retention_strata_teacher_only": True,
        "new_test_corpus_bound": True,
        "new_test_student_evaluations": 0,
        "human_test_accesses": 0,
        "optimizer_steps": 0,
        "ppo_updates": 0,
        "reward_changes": 0,
        "valid": True,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RESULT_ROOT / "frozen_config.json", config)
    _write_json(RESULT_ROOT / "retention_strata_manifest.json", strata_manifest)
    _write_json(RESULT_ROOT / "new_simulator_test_authority.json", new_test_authority)
    _write_json(RESULT_ROOT / "pre_step_preflight.json", preflight)
    print(
        json.dumps(
            {
                "verdict": "PRE_STEP_AUTHORITY_BUILT_NO_OPTIMIZER_STEPS",
                "frozen_config_sha256": _sha(RESULT_ROOT / "frozen_config.json"),
                "strata": strata_manifest["pools"],
                "low_variance_threshold_log_std": strata_manifest["low_variance_definition"][
                    "threshold_log_std"
                ],
                "new_test_corpus_identity": new_manifest["identity_sha256"],
                "new_test_observation_sha256": new_manifest["collection"][
                    "observation_tensor_sha256"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
