"""Build the prospective, no-optimizer-step authority for Human BC V4."""

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

from benchmarks.rival2_human_bc_v4_common import (  # noqa: E402
    build_aligned_rollout_corpus,
)
from benchmarks.run_rival2_missing_feature_distillation import (  # noqa: E402
    _load_bootstrap,
)
from rivalsim.human_demo.bc_v3_retention import int64_sha256  # noqa: E402
from rivalsim.human_demo.bc_v4_retention import (  # noqa: E402
    build_v4_retention_pools,
)
from rivalsim.human_demo.missing_feature_distillation import (  # noqa: E402
    canonical_sha256,
    file_sha256,
)
from rivalsim.rival2_120hz_transition import tensor_tree_sha256  # noqa: E402
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig  # noqa: E402

RESULT_ROOT = ROOT / "results/rival2/human_bc_v4"
BASE_CONFIG_PATH = ROOT / "results/rival2/human_behavior_cloning_v1/frozen_config.json"
BASE_EVIDENCE_PATH = ROOT / "results/rival2/human_behavior_cloning_v1/evidence.json"
SOURCE_CHECKPOINT = ROOT / "checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt"
ADAPTER_CHECKPOINT = (
    ROOT / "checkpoints/rival2/observation_adapter_v2/"
    "rival2_human_demo_observation_adapter_v2.pt"
)
V2_CHECKPOINT = ROOT / "checkpoints/rival2/human_bc_v2/rival2_human_bc_v2.pt"
V2_EVIDENCE = ROOT / "results/rival2/human_bc_v2/evidence.json"
V3_CHECKPOINT = ROOT / "checkpoints/rival2/human_bc_v3/rival2_human_bc_v3.pt"
V3_EVIDENCE = ROOT / "results/rival2/human_bc_v3/evidence.json"
V3_TEST_AUTHORITY = ROOT / "results/rival2/human_bc_v3/new_simulator_test_authority.json"

REQUIRED_START = "da185d219127968d6612d596552f90ee18e02cca"
SOURCE_SHA = "560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874"
ADAPTER_SHA = "EDEDC9CCDE3269B393FB4C944F641CF4D34A78AB5944662F9019009BBA914C99"
ESTABLISHED_CORPUS_SEED = 2026082821
ESTABLISHED_SPLIT_SEED = 2026082822
STRESS_CORPUS_SEED = 2026090117
STRESS_SPLIT_SEED = 2026090118
TEST_CORPUS_SEED = 2026090119
TEST_SPLIT_SEED = 2026090120
CANDIDATE_POOL_SEED = 2026090121
RETENTION_SAMPLE_SEED = 2026090122
TRAINING_SEED = 2026090123
RESERVED_WORLDS = 8192

LOW_VARIANCE_QUANTILE = 0.10
ORIENTATION_CORE_QUANTILE = 0.90
RECOVERY_POSITION_QUANTILE = 0.35
RECOVERY_VELOCITY_QUANTILE = 0.35
RECOVERY_DYNAMICS_QUANTILE = 0.65
CONTACT_ABSOLUTE_UP_QUANTILE = 0.20
CONTACT_UP_QUANTILE = 0.05
CANDIDATE_POOL_ROWS = 524_288
INITIAL_REPLAY_ROWS = 4_096


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_readme() -> None:
    (RESULT_ROOT / "README.md").write_text(
        """# Human BC V4 prospective authority

This directory binds the complete Human BC V4 simulator-retention authority
before any optimizer step. The established training/validation corpus now
includes rollout-aligned opponent-family and train-mask hashes. The separate
stress-validation and untouched-test corpora use distinct deterministic seed
namespaces and reserve exactly 8,192 whole worlds each.

Orientation-sensitive membership is derived only from the frozen Human BC V1
teacher and authoritative simulator training state. Dynamic hard-tail mining
may inspect only the frozen training candidate pool. Neither human nor simulator
test data may affect training, selection, stopping, thresholds, or acceptance
until the single final post-selection evaluation.

The authority build performs no optimizer step, PPO update, reward change,
human-data mutation, or test-candidate evaluation.
""",
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return file_sha256(path)


def _source_teacher(device: str) -> tuple[dict[str, Any], Rival2ActorCritic]:
    if _sha(SOURCE_CHECKPOINT) != SOURCE_SHA:
        raise RuntimeError("Human BC V1 source checkpoint changed")
    payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    policy = Rival2PolicyConfig(**payload["policy_config"])
    teacher = Rival2ActorCritic(policy).to(device)
    teacher.load_state_dict(payload["model"], strict=True)
    teacher.eval().requires_grad_(False)
    return payload, teacher


def _corpus_config(
    base_config: dict[str, Any],
    *,
    seed: int,
    split_seed: int,
) -> dict[str, Any]:
    corpus = copy.deepcopy(base_config["retention"]["corpus"])
    corpus["seed"] = seed
    corpus["split"] = {
        "algorithm": "numpy PCG64 permutation of whole world indices",
        "seed": split_seed,
        "train_worlds": 16_384,
        "validation_worlds": RESERVED_WORLDS,
        "test_worlds": RESERVED_WORLDS,
    }
    corpus.pop("expected_identity_sha256", None)
    corpus.pop("expected_observation_tensor_sha256", None)
    return corpus


def _compatibility(
    base_config: dict[str, Any], corpus: dict[str, Any]
) -> dict[str, Any]:
    return {"authority": base_config["authority"], "corpus": corpus}


def _reserved_authority(
    *,
    role: str,
    manifest: dict[str, Any],
    worlds: np.ndarray,
) -> dict[str, Any]:
    return {
        "format": f"RIVAL2_HUMAN_BC_V4_{role.upper()}_SIMULATOR_AUTHORITY_V1",
        "created_utc": datetime.now(UTC).isoformat(),
        "role": role,
        "corpus": manifest,
        "reserved_split": "validation" if role == "stress_validation" else "test",
        "reserved_worlds": int(worlds.size),
        "reserved_world_indices_int64_sha256": int64_sha256(worlds),
        "reserved_trajectory_binding": canonical_sha256(
            {
                "aligned_corpus_identity_sha256": manifest["identity_sha256"],
                "world_indices_int64_sha256": int64_sha256(worlds),
                "role": role,
            }
        ),
        "whole_world_reservation": True,
        "student_evaluations_before_final_selection": 0,
        "bc_v1_teacher_evaluations_before_final_selection": 0,
        "optimizer_steps_before_binding": 0,
        "observation_and_aligned_role_tensors_generated_only_for_hash_binding": True,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def main() -> int:
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", REQUIRED_START, "HEAD"],
        check=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for authoritative simulator corpus generation")
    if _sha(ADAPTER_CHECKPOINT) != ADAPTER_SHA:
        raise RuntimeError("frozen observation adapter changed")
    if _sha(SOURCE_CHECKPOINT) != SOURCE_SHA:
        raise RuntimeError("frozen Human BC V1 parent changed")
    if not all(
        path.is_file()
        for path in (V2_CHECKPOINT, V2_EVIDENCE, V3_CHECKPOINT, V3_EVIDENCE)
    ):
        raise FileNotFoundError("V2/V3 diagnostic evidence is incomplete")

    # Warp simulation kernels are not PyTorch deterministic-algorithm kernels. Corpus
    # identity is established by fixed simulator seeds plus exact output hashes.
    torch.use_deterministic_algorithms(False)
    device = "cuda:0"
    base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    source_payload, teacher = _source_teacher(device)
    established_corpus_config = copy.deepcopy(base_config["retention"]["corpus"])
    if (
        int(established_corpus_config["seed"]) != ESTABLISHED_CORPUS_SEED
        or int(established_corpus_config["split"]["seed"]) != ESTABLISHED_SPLIT_SEED
    ):
        raise RuntimeError("established simulator corpus seeds changed")
    established_compatibility = _compatibility(base_config, established_corpus_config)
    bootstrap_payload, policy_config, bootstrap_identity = _load_bootstrap(
        established_compatibility
    )

    established, established_manifest, established_splits = (
        build_aligned_rollout_corpus(
            established_compatibility,
            bootstrap_payload,
            bootstrap_identity,
            device=device,
        )
    )
    if (
        established_manifest["legacy_observation_only_identity_sha256"]
        != established_corpus_config["expected_identity_sha256"]
        or established_manifest["collection"]["observation_tensor_sha256"]
        != established_corpus_config["expected_observation_tensor_sha256"]
    ):
        raise RuntimeError("established simulator observation corpus changed")
    pools = build_v4_retention_pools(
        teacher,
        established.observations,
        established_splits["train"],
        established.opponent_family,
        established.train_mask,
        low_variance_quantile=LOW_VARIANCE_QUANTILE,
        orientation_core_quantile=ORIENTATION_CORE_QUANTILE,
        recovery_position_quantile=RECOVERY_POSITION_QUANTILE,
        recovery_velocity_quantile=RECOVERY_VELOCITY_QUANTILE,
        recovery_dynamics_quantile=RECOVERY_DYNAMICS_QUANTILE,
        contact_absolute_up_quantile=CONTACT_ABSOLUTE_UP_QUANTILE,
        contact_up_quantile=CONTACT_UP_QUANTILE,
        candidate_pool_rows=CANDIDATE_POOL_ROWS,
        candidate_pool_fractions={
            "orientation_sensitive": 0.50,
            "current_policy_applicable": 0.25,
            "historical_opponent": 0.125,
            "natural": 0.125,
        },
        candidate_pool_seed=CANDIDATE_POOL_SEED,
        initial_replay_rows=INITIAL_REPLAY_ROWS,
        policy_config=policy_config,
    )
    strata_manifest = {
        **pools.manifest,
        "source_aligned_simulator_corpus_identity_sha256": established_manifest[
            "identity_sha256"
        ],
        "source_observation_tensor_sha256": established_manifest["collection"][
            "observation_tensor_sha256"
        ],
        "source_opponent_family_tensor_sha256": established_manifest["collection"][
            "opponent_family_tensor_sha256"
        ],
        "source_train_mask_tensor_sha256": established_manifest["collection"][
            "train_mask_tensor_sha256"
        ],
        "source_train_world_indices_sha256": established_manifest["split"][
            "world_index_sha256"
        ]["train"],
        "role_membership_source": "rollout-aligned opponent_family and train_mask",
        "student_outputs_used_for_static_membership": False,
        "opened_test_rows_used": False,
        "optimizer_steps": 0,
    }
    del pools, established, teacher
    gc.collect()
    torch.cuda.empty_cache()

    stress_config = _corpus_config(
        base_config,
        seed=STRESS_CORPUS_SEED,
        split_seed=STRESS_SPLIT_SEED,
    )
    stress_compatibility = _compatibility(base_config, stress_config)
    stress, stress_manifest, stress_splits = build_aligned_rollout_corpus(
        stress_compatibility,
        bootstrap_payload,
        bootstrap_identity,
        device=device,
    )
    stress_worlds = np.asarray(stress_splits["validation"], dtype=np.int64)
    if stress_worlds.size != RESERVED_WORLDS:
        raise RuntimeError("stress authority did not reserve exactly 8192 worlds")
    stress_authority = _reserved_authority(
        role="stress_validation",
        manifest=stress_manifest,
        worlds=stress_worlds,
    )
    del stress
    gc.collect()
    torch.cuda.empty_cache()

    test_config = _corpus_config(
        base_config,
        seed=TEST_CORPUS_SEED,
        split_seed=TEST_SPLIT_SEED,
    )
    test_compatibility = _compatibility(base_config, test_config)
    untouched, untouched_manifest, untouched_splits = build_aligned_rollout_corpus(
        test_compatibility,
        bootstrap_payload,
        bootstrap_identity,
        device=device,
    )
    untouched_worlds = np.asarray(untouched_splits["test"], dtype=np.int64)
    if untouched_worlds.size != RESERVED_WORLDS:
        raise RuntimeError("test authority did not reserve exactly 8192 worlds")
    untouched_authority = _reserved_authority(
        role="untouched_test",
        manifest=untouched_manifest,
        worlds=untouched_worlds,
    )
    # Collection necessarily performs frozen bootstrap policy inference to generate
    # natural states. No BC-V1 teacher or candidate V4 student is run on this corpus.
    del untouched
    gc.collect()
    torch.cuda.empty_cache()

    seed_namespaces = {
        "established_original_and_v2": ESTABLISHED_CORPUS_SEED,
        "opened_v3_test": 2026090107,
        "v4_stress_validation": STRESS_CORPUS_SEED,
        "v4_untouched_test": TEST_CORPUS_SEED,
    }
    disjointness = {
        "format": "RIVAL2_HUMAN_BC_V4_SIMULATOR_DISJOINTNESS_PROOF_V1",
        "trajectory_identity": "(corpus_seed, world_index, complete 128-tick trajectory)",
        "seed_namespaces": seed_namespaces,
        "all_seed_namespaces_unique": len(set(seed_namespaces.values()))
        == len(seed_namespaces),
        "pairwise_trajectory_overlap": {
            "v4_test_vs_established_train_validation_opened_test": 0,
            "v4_test_vs_opened_v3_test": 0,
            "v4_test_vs_v4_stress_validation": 0,
            "v4_stress_vs_established_train_validation_opened_test": 0,
        },
        "proof": (
            "trajectory identity includes corpus seed; all relevant corpora use unique "
            "fixed seed namespaces. Whole-world splits are deterministic permutations "
            "and are additionally bound by exact index and tensor hashes."
        ),
        "established_train_validation_same_corpus_overlap": int(
            np.intersect1d(
                established_splits["train"], established_splits["validation"]
            ).size
        ),
        "established_train_opened_test_same_corpus_overlap": int(
            np.intersect1d(
                established_splits["train"], established_splits["test"]
            ).size
        ),
        "established_validation_opened_test_same_corpus_overlap": int(
            np.intersect1d(
                established_splits["validation"], established_splits["test"]
            ).size
        ),
        "stress_reserved_world_indices_sha256": int64_sha256(stress_worlds),
        "test_reserved_world_indices_sha256": int64_sha256(untouched_worlds),
        "student_outcome_used_for_membership": False,
        "opened_test_outcome_used_for_membership": False,
    }
    if not disjointness["all_seed_namespaces_unique"] or any(
        value != 0 for value in disjointness["pairwise_trajectory_overlap"].values()
    ) or any(
        disjointness[name] != 0
        for name in (
            "established_train_validation_same_corpus_overlap",
            "established_train_opened_test_same_corpus_overlap",
            "established_validation_opened_test_same_corpus_overlap",
        )
    ):
        raise RuntimeError("V4 simulator corpus disjointness failed")

    config = {
        "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_FROZEN_CONFIG_V4",
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
            "observation_adapter_checkpoint": ADAPTER_CHECKPOINT.relative_to(
                ROOT
            ).as_posix(),
            "observation_adapter_checkpoint_sha256": ADAPTER_SHA,
            "v2_diagnostic_checkpoint_sha256": _sha(V2_CHECKPOINT),
            "v2_diagnostic_evidence_sha256": _sha(V2_EVIDENCE),
            "v3_diagnostic_checkpoint_sha256": _sha(V3_CHECKPOINT),
            "v3_diagnostic_evidence_sha256": _sha(V3_EVIDENCE),
            "v3_opened_test_authority_sha256": _sha(V3_TEST_AUTHORITY),
            "v2_v3_checkpoint_training_prohibited": True,
            "all_pre_v4_simulator_tests_diagnostic_only": True,
            "pre_step_authority": (
                "results/rival2/human_bc_v4/pre_step_authority.json"
            ),
        },
        "checkpoint": {
            "path": "checkpoints/rival2/human_bc_v4/rival2_human_bc_v4.pt",
            "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V4",
            "ppo_resumable": False,
            "historical_ppo_optimizer_resumable": False,
            "save_every_accepted_validation_boundary": True,
        },
        "trainable_parameters": {
            "mode": "actor_head_only",
            "names": ["actor.weight", "actor.bias"],
            "shared_trunk": "frozen byte-identical to Human BC V1",
            "critic_head": "frozen byte-identical to Human BC V1",
            "optimizer": "fresh actor-only torch.optim.AdamW",
        },
        "inherit_exactly_from_base_bc_v1": {
            "human_action_objective": True,
            "human_dataset_and_source_hashes": True,
            "human_train_validation_test_splits": True,
            "mechanic_aware_sampling": True,
            "observation_action_contracts": True,
            "observation_adapter": True,
            "sampling_family_proportions": True,
            "simulator_retention_hard_limits": True,
        },
        "simulator_authority": {
            "training_validation_corpus_identity_sha256": established_manifest[
                "identity_sha256"
            ],
            "training_validation_legacy_identity_sha256": established_manifest[
                "legacy_observation_only_identity_sha256"
            ],
            "training_validation_observation_sha256": established_manifest[
                "collection"
            ]["observation_tensor_sha256"],
            "training_validation_opponent_family_sha256": established_manifest[
                "collection"
            ]["opponent_family_tensor_sha256"],
            "training_validation_train_mask_sha256": established_manifest["collection"][
                "train_mask_tensor_sha256"
            ],
            "training_world_indices_sha256": established_manifest["split"][
                "world_index_sha256"
            ]["train"],
            "validation_world_indices_sha256": established_manifest["split"][
                "world_index_sha256"
            ]["validation"],
            "complete_validation_worlds": len(established_splits["validation"]),
            "old_test_worlds_excluded": True,
            "retention_strata_manifest": (
                "results/rival2/human_bc_v4/retention_strata_manifest.json"
            ),
            "stress_validation_authority": (
                "results/rival2/human_bc_v4/stress_validation_authority.json"
            ),
            "stress_validation_corpus_identity_sha256": stress_manifest[
                "identity_sha256"
            ],
            "stress_validation_corpus_seed": STRESS_CORPUS_SEED,
            "stress_validation_split_seed": STRESS_SPLIT_SEED,
            "stress_validation_observation_sha256": stress_manifest["collection"][
                "observation_tensor_sha256"
            ],
            "stress_validation_opponent_family_sha256": stress_manifest["collection"][
                "opponent_family_tensor_sha256"
            ],
            "stress_validation_train_mask_sha256": stress_manifest["collection"][
                "train_mask_tensor_sha256"
            ],
            "stress_validation_world_indices_sha256": int64_sha256(
                stress_worlds
            ),
            "stress_validation_worlds": RESERVED_WORLDS,
            "untouched_test_authority": (
                "results/rival2/human_bc_v4/untouched_test_authority.json"
            ),
            "untouched_test_corpus_identity_sha256": untouched_manifest[
                "identity_sha256"
            ],
            "untouched_test_corpus_seed": TEST_CORPUS_SEED,
            "untouched_test_split_seed": TEST_SPLIT_SEED,
            "untouched_test_observation_sha256": untouched_manifest["collection"][
                "observation_tensor_sha256"
            ],
            "untouched_test_opponent_family_sha256": untouched_manifest["collection"][
                "opponent_family_tensor_sha256"
            ],
            "untouched_test_train_mask_sha256": untouched_manifest["collection"][
                "train_mask_tensor_sha256"
            ],
            "untouched_test_world_indices_sha256": int64_sha256(
                untouched_worlds
            ),
            "untouched_test_worlds": RESERVED_WORLDS,
            "test_access_before_final_validation_selection": 0,
        },
        "retention_sampling": {
            "membership_source": (
                "frozen BC-V1 teacher, authoritative training state, and rollout-aligned roles"
            ),
            "with_replacement": True,
            "ordered_strata": [
                "natural",
                "current_policy_applicable",
                "historical_opponent",
                "low_teacher_variance",
                "orientation_sensitive",
                "hard_tail_replay",
            ],
            "rows_per_step": {
                "natural": 2048,
                "current_policy_applicable": 1024,
                "historical_opponent": 1024,
                "low_teacher_variance": 1024,
                "orientation_sensitive": 2048,
                "hard_tail_replay": 1024,
            },
            "total_rows_per_step": 8192,
            "low_variance_quantile": LOW_VARIANCE_QUANTILE,
            "low_variance_threshold_log_std": strata_manifest[
                "low_variance_definition"
            ]["threshold_log_std"],
            "pool_hashes": {
                name: row["ordered_int64_sha256"]
                for name, row in strata_manifest["pools"].items()
            },
        },
        "orientation_sensitive_stratum": strata_manifest[
            "orientation_sensitive_definition"
        ],
        "hard_tail_mining": {
            "source": "prospectively frozen simulator training candidate pool only",
            "candidate_pool_rows": CANDIDATE_POOL_ROWS,
            "candidate_pool_sha256": strata_manifest["mining_candidate_pool"][
                "ordered_int64_sha256"
            ],
            "candidate_pool_seed": CANDIDATE_POOL_SEED,
            "candidate_pool_fractions": {
                "orientation_sensitive": 0.50,
                "current_policy_applicable": 0.25,
                "historical_opponent": 0.125,
                "natural": 0.125,
            },
            "mining_frequency_accepted_steps": 64,
            "ranking": "hybrid actor sample KL descending then encoded row ascending",
            "top_k_per_generation": 4096,
            "maximum_replay_rows": 16384,
            "maximum_generation_age_inclusive": 3,
            "replay_fraction_per_retention_batch": 0.125,
            "replacement_aging_policy": (
                "newest generation first; refreshed rows replace older identity; retain "
                "at most current plus three preceding generations; truncate deterministically"
            ),
            "initial_replay_rows": INITIAL_REPLAY_ROWS,
            "initial_replay_sha256": strata_manifest["initial_hard_tail_replay"][
                "ordered_int64_sha256"
            ],
            "candidate_evaluation_rows_per_batch": 65_536,
            "validation_or_test_inspection_prohibited": True,
        },
        "tail_aware_retention": {
            "sample_kl": "sum of all eight hybrid actor channel KL contributions",
            "orientation_tail": (
                "maximum single steer/pitch/yaw/roll channel KL contribution per sample"
            ),
            "formulation": (
                "2*mean(K) + 4*mean(phi(K;0.5,0.05)) + "
                "4*CVaR_top1pct(phi(K;0.5,0.05)) + "
                "4*CVaR_top1pct(phi(M_orientation;0.125,0.0125))"
            ),
            "barrier_phi": "(temperature * softplus((value-threshold)/temperature))^2",
            "ordinary_mean_kl_coefficient": 2.0,
            "total_sample_activation_threshold": 0.5,
            "total_sample_temperature": 0.05,
            "total_mean_barrier_coefficient": 4.0,
            "total_cvar_barrier_coefficient": 4.0,
            "orientation_activation_threshold": 0.125,
            "orientation_temperature": 0.0125,
            "orientation_cvar_barrier_coefficient": 4.0,
            "cvar_top_quantile_fraction": 0.01,
            "hard_all_perspective_max_sample_kl": 2.0,
            "inherited_mean_channel_and_critic_guards_unchanged": True,
        },
        "optimizer": {
            "type": "fresh torch.optim.AdamW over actor.weight and actor.bias only",
            "training_seed": TRAINING_SEED,
            "retention_sampling_seed": RETENTION_SAMPLE_SEED,
            "historical_ppo_optimizer_loaded": False,
            "source_full_model_bc_optimizer_loaded": False,
            "v2_v3_optimizer_loaded": False,
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
            "normal_complete_validation_required": True,
            "stress_validation_required": True,
            "requires_both_human_families_improve_over_parent": True,
            "minimum_human_family_relative_improvement": 0.05,
            "mechanic_label_nonregression_relative_tolerance": 0.02,
            "minimum_mechanic_labels_improved_fraction": 0.6,
            "minimum_mechanic_labels_nonregressed_fraction": 0.8,
            "requires_action_distribution_health": True,
            "prospective_validation_safety_margin": {
                "maximum_sample_kl": 1.0,
                "maximum_single_orientation_channel_kl": 0.5,
                "acceptance_hard_sample_kl_unchanged": 2.0,
            },
            "candidate_score": (
                "0.29 gameplay ratio + 0.29 mechanic ratio + 0.16 mean label ratio + "
                "0.08 complete mean-KL/soft + 0.06 complete max-KL/margin + "
                "0.06 stress mean-KL/soft + 0.06 stress max-KL/margin"
            ),
            "score_weights": {
                "gameplay_rmse_ratio": 0.29,
                "mechanic_rmse_ratio": 0.29,
                "mean_per_label_rmse_ratio": 0.16,
                "complete_mean_kl_soft_limit_ratio": 0.08,
                "complete_max_kl_selection_ratio": 0.06,
                "stress_mean_kl_soft_limit_ratio": 0.06,
                "stress_max_kl_selection_ratio": 0.06,
            },
            "early_stopping_material_score_improvement": 0.0005,
            "early_stopping_patience_validations": 20,
            "minimum_accepted_steps_before_plateau": 2048,
            "test_used_for_selection": False,
        },
        "validation": {
            "complete_worlds": len(established_splits["validation"]),
            "stress_worlds": RESERVED_WORLDS,
            "worlds_per_batch": int(
                base_config["retention"]["worlds_per_validation_batch"]
            ),
            "groups": [
                "all_perspectives",
                "current_policy_applicable",
                "counterfactual_opponent",
                "historical_opponent",
                "low_teacher_variance",
                "orientation_sensitive",
            ],
            "threshold_counts": [0.5, 1.0, 2.0],
            "orientation_channels": ["steer", "pitch", "yaw", "roll"],
            "hard_guard": base_config["retention"]["hard_guard"],
            "soft_actor_mean_kl": base_config["retention"]["soft_actor_mean_kl"],
            "selection_margin": {
                "maximum_sample_kl": 1.0,
                "maximum_individual_orientation_channel_kl": 0.5,
            },
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
            "v4_simulator_test_access": "exactly once after final validation selection",
            "all_pre_v4_tests": "diagnostic only and never reevaluated for selection",
            "selection_reopened_after_test": False,
        },
        "prohibited": {
            "ppo": True,
            "reward_change": True,
            "additional_demonstrations": True,
            "dataset_or_split_change": True,
            "mechanic_definition_or_adjudication_change": True,
            "observation_adapter_change": True,
            "observation_action_contract_change": True,
            "raw_recording_or_review_mutation": True,
        },
    }
    preflight = {
        "format": "RIVAL2_HUMAN_BC_V4_PRE_STEP_PREFLIGHT_V1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "required_starting_commit_is_ancestor": True,
        "source_checkpoint_sha256_exact": _sha(SOURCE_CHECKPOINT) == SOURCE_SHA,
        "adapter_checkpoint_sha256_exact": _sha(ADAPTER_CHECKPOINT) == ADAPTER_SHA,
        "established_observation_corpus_exact": True,
        "aligned_role_metadata_bound": True,
        "orientation_stratum_teacher_training_state_only": True,
        "hard_tail_candidate_training_rows_only": True,
        "stress_validation_corpus_bound": True,
        "untouched_test_corpus_bound": True,
        "untouched_test_student_evaluations": 0,
        "untouched_test_bc_v1_teacher_evaluations": 0,
        "human_test_accesses": 0,
        "optimizer_steps": 0,
        "ppo_updates": 0,
        "reward_changes": 0,
        "mechanic_changes": 0,
        "valid": True,
    }

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RESULT_ROOT / "frozen_config.json", config)
    _write_json(RESULT_ROOT / "retention_strata_manifest.json", strata_manifest)
    _write_json(RESULT_ROOT / "stress_validation_authority.json", stress_authority)
    _write_json(RESULT_ROOT / "untouched_test_authority.json", untouched_authority)
    _write_json(RESULT_ROOT / "simulator_disjointness_proof.json", disjointness)
    _write_json(RESULT_ROOT / "pre_step_preflight.json", preflight)
    _write_readme()
    artifacts = {
        path.name: _artifact_record(path)
        for path in (
            RESULT_ROOT / "frozen_config.json",
            RESULT_ROOT / "retention_strata_manifest.json",
            RESULT_ROOT / "stress_validation_authority.json",
            RESULT_ROOT / "untouched_test_authority.json",
            RESULT_ROOT / "simulator_disjointness_proof.json",
            RESULT_ROOT / "pre_step_preflight.json",
            RESULT_ROOT / "README.md",
        )
    }
    artifact_manifest = {
        "format": "RIVAL2_HUMAN_BC_V4_PROSPECTIVE_ARTIFACT_MANIFEST_V1",
        "created_utc": datetime.now(UTC).isoformat(),
        "artifacts": artifacts,
        "authority_identity_sha256": canonical_sha256(artifacts),
        "optimizer_steps": 0,
        "test_student_evaluations": 0,
    }
    _write_json(RESULT_ROOT / "prospective_artifact_manifest.json", artifact_manifest)

    print(
        json.dumps(
            {
                "verdict": "V4_PROSPECTIVE_AUTHORITY_BUILT_NO_OPTIMIZER_STEPS",
                "frozen_config_sha256": _sha(RESULT_ROOT / "frozen_config.json"),
                "established_aligned_identity": established_manifest["identity_sha256"],
                "orientation_rows": strata_manifest["pools"]["orientation_sensitive"][
                    "rows"
                ],
                "stress_corpus_identity": stress_manifest["identity_sha256"],
                "stress_reserved_worlds": int(stress_worlds.size),
                "untouched_test_corpus_identity": untouched_manifest["identity_sha256"],
                "untouched_test_reserved_worlds": int(untouched_worlds.size),
                "untouched_test_student_evaluations": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
