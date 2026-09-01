"""Freeze the minimal Human BC V5 authority without generating new corpora."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_rival2_human_behavior_cloning_v1 import _write_json  # noqa: E402
from rivalsim.human_demo.missing_feature_distillation import file_sha256  # noqa: E402

REQUIRED_START = "7b885d1211ddeb56fce6858b90ea9666ee03914b"
V4_BLOCKED_RESULT = "9bb0dd0be8beedfeed5fe2c4e24938b166a88646"
SOURCE_CHECKPOINT = ROOT / "checkpoints/rival2/human_bc_v1/rival2_human_bc_v1.pt"
SOURCE_SHA256 = "560C2414C17039DC920126EA148BF73FE6CC4677EE440F043599A7E1C76D2874"
ADAPTER_CHECKPOINT = (
    ROOT
    / "checkpoints/rival2/observation_adapter_v2/"
    "rival2_human_demo_observation_adapter_v2.pt"
)
ADAPTER_SHA256 = "EDEDC9CCDE3269B393FB4C944F641CF4D34A78AB5944662F9019009BBA914C99"
PACKAGE_PROMPT = ROOT / "handoff/rival2-human-bc-v5/CODEX_START_PROMPT.md"
V4_ROOT = ROOT / "results/rival2/human_bc_v4"
V4_CONFIG = V4_ROOT / "frozen_config.json"
V4_EVIDENCE = V4_ROOT / "evidence.json"
V4_STRATA = V4_ROOT / "retention_strata_manifest.json"
V4_STRESS = V4_ROOT / "stress_validation_authority.json"
V4_TEST = V4_ROOT / "untouched_test_authority.json"
V4_DISJOINTNESS = V4_ROOT / "simulator_disjointness_proof.json"
RESULT_ROOT = ROOT / "results/rival2/human_bc_v5"
GROUPS = (
    "all_perspectives",
    "current_policy_applicable",
    "counterfactual_opponent",
    "historical_opponent",
    "low_teacher_variance",
    "orientation_sensitive",
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "git_blob_oid": _git("rev-parse", f"HEAD:{path.relative_to(ROOT).as_posix()}"),
    }


def _parent_boost_rows(evidence: dict[str, Any], family: str) -> dict[str, Any]:
    metrics = evidence["training"][family]
    rows: dict[str, Any] = {}
    for group in GROUPS:
        parent = float(
            metrics[group]["actor_distribution"]["button_probability"]["boost"]
            ["saturation_fraction"]
        )
        rows[group] = {
            "bc_v1_parent_saturation_fraction": parent,
            "candidate_limit": max(0.95, parent + 0.005),
            "formula": "max(0.95, BC_V1_parent_saturation + 0.005)",
        }
    return rows


def main() -> int:
    for commit in (REQUIRED_START, V4_BLOCKED_RESULT):
        subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=True,
        )
    if file_sha256(SOURCE_CHECKPOINT) != SOURCE_SHA256:
        raise RuntimeError("Human BC V1 parent changed")
    if file_sha256(ADAPTER_CHECKPOINT) != ADAPTER_SHA256:
        raise RuntimeError("observation adapter V2 changed")

    v4_config = json.loads(V4_CONFIG.read_text(encoding="utf-8"))
    v4_evidence = json.loads(V4_EVIDENCE.read_text(encoding="utf-8"))
    v4_test_authority = json.loads(V4_TEST.read_text(encoding="utf-8"))
    sealed_checks = {
        "v4_verdict_blocked": v4_evidence["verdict"] == "BLOCKED",
        "v4_accepted_zero_steps": int(
            v4_evidence["training"]["accepted_steps_executed"]
        )
        == 0,
        "v4_human_test_access_zero": int(
            v4_evidence["test_discipline"]["human_test_access_count"]
        )
        == 0,
        "v4_simulator_test_student_evaluations_zero": int(
            v4_evidence["test_discipline"]
            ["untouched_simulator_test_student_evaluation_count"]
        )
        == 0,
        "v4_test_authority_declares_zero_evaluations": int(
            v4_test_authority["student_evaluations_before_final_selection"]
        )
        == 0,
        "v4_test_access_ledger_absent": not (V4_ROOT / "test_access_ledger.json").exists(),
        "v4_human_test_results_absent": not (V4_ROOT / "human_test_metrics.json").exists(),
        "v4_simulator_test_results_absent": not (
            V4_ROOT / "untouched_test_results.json"
        ).exists(),
    }
    if not all(sealed_checks.values()):
        raise RuntimeError(f"V4 untouched-test authority is not sealed: {sealed_checks}")

    config = copy.deepcopy(v4_config)
    config["format"] = "RIVAL2_HUMAN_BEHAVIOR_CLONING_FROZEN_CONFIG_V5"
    authority = config["authority"]
    authority.update(
        {
            "required_parent": REQUIRED_START,
            "v4_blocked_result_commit": V4_BLOCKED_RESULT,
            "v4_blocked_evidence": V4_EVIDENCE.relative_to(ROOT).as_posix(),
            "v4_blocked_evidence_sha256": file_sha256(V4_EVIDENCE),
            "v4_authority_reused_without_new_corpus": True,
            "v2_v3_v4_checkpoint_training_prohibited": True,
            "pre_step_authority": (
                "results/rival2/human_bc_v5/pre_step_authority.json"
            ),
            "package_prompt": PACKAGE_PROMPT.relative_to(ROOT).as_posix(),
            "package_prompt_sha256": file_sha256(PACKAGE_PROMPT),
        }
    )
    config["checkpoint"] = {
        "format": "RIVAL2_HUMAN_BEHAVIOR_CLONING_CHECKPOINT_V5",
        "path": "checkpoints/rival2/human_bc_v5/rival2_human_bc_v5.pt",
        "save_every_accepted_validation_boundary": True,
        "ppo_resumable": False,
        "historical_ppo_optimizer_resumable": False,
    }
    config["tail_aware_retention"].update(
        {
            "formulation": (
                "0.5*mean(K) + 1.5*mean(phi(K;0.5,0.05)) + "
                "2*CVaR_top1pct(phi(K;0.5,0.05)) + "
                "2*CVaR_top1pct(phi(M_orientation;0.125,0.0125))"
            ),
            "ordinary_mean_kl_coefficient": 0.5,
            "total_mean_barrier_coefficient": 1.5,
            "total_cvar_barrier_coefficient": 2.0,
            "orientation_cvar_barrier_coefficient": 2.0,
        }
    )
    config["optimizer"].update(
        {
            "transactional_retries_per_interval": 2,
            "transactional_backoff_factor": 0.5,
            "minimum_learning_rate": 0.0000125,
            "exact_retry_learning_rates": [0.00005, 0.000025, 0.0000125],
        }
    )
    config["training"].update(
        {
            "initial_learning_rate": 0.00005,
            "validation_interval_optimizer_steps": 128,
            "maximum_accepted_supervised_steps": 10000,
            "weight_decay": 0.00001,
            "optimizer_betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-8,
            "gradient_clip_norm": 1.0,
        }
    )
    config["selection"].update(
        {
            "candidate_score": (
                "0.36 gameplay ratio + 0.36 mechanic ratio + 0.18 mean label "
                "ratio + 0.03 complete mean-KL/soft + 0.02 complete max-KL/hard "
                "+ 0.03 stress mean-KL/soft + 0.02 stress max-KL/hard"
            ),
            "score_weights": {
                "gameplay_rmse_ratio": 0.36,
                "mechanic_rmse_ratio": 0.36,
                "mean_per_label_rmse_ratio": 0.18,
                "complete_mean_kl_soft_limit_ratio": 0.03,
                "complete_max_kl_selection_ratio": 0.02,
                "stress_mean_kl_soft_limit_ratio": 0.03,
                "stress_max_kl_selection_ratio": 0.02,
            },
            "human_imitation_weight": 0.90,
            "retention_ranking_weight": 0.10,
            "minimum_human_family_relative_improvement": 0.0,
            "strict_human_family_improvement": True,
            "mechanic_label_nonregression_relative_tolerance": 0.03,
            "minimum_mechanic_labels_improved_fraction": 0.60,
            "minimum_mechanic_labels_nonregressed_fraction": 0.80,
            "minimum_accepted_steps_before_plateau": 3072,
            "early_stopping_patience_validations": 16,
            "early_stopping_material_score_improvement": 0.0005,
            "extra_kl_selection_margins_removed": True,
        }
    )
    config["selection"].pop("prospective_validation_safety_margin", None)
    config["validation"].pop("selection_margin", None)
    config["validation"]["hard_max_sample_kl_is_only_sample_max_rejection"] = True
    config["parent_relative_boost_saturation"] = {
        "absolute_baseline_limit": 0.95,
        "worsening_allowance": 0.005,
        "formula": "max(0.95, BC_V1_parent_saturation + 0.005)",
        "exception_scope": "boost probability saturation only",
        "all_unrelated_distribution_health_checks_unchanged": True,
        "complete_validation": _parent_boost_rows(
            v4_evidence, "parent_complete_simulator_validation"
        ),
        "stress_validation": _parent_boost_rows(
            v4_evidence, "parent_stress_simulator_validation"
        ),
        "untouched_test": {
            "parent_and_selected_candidate_evaluated_in_same_final_pass": True,
            "parent_not_evaluated_before_final_selection": True,
            "limit_computed_from_final_parent_measurement": True,
        },
    }
    config["test_discipline"] = {
        "human_test_access": "exactly once after final validation selection",
        "reused_v4_simulator_test_access": (
            "exactly once after final validation selection"
        ),
        "bc_v1_parent_and_v5_candidate_same_final_simulator_pass": True,
        "selection_reopened_after_test": False,
        "all_pre_v4_opened_tests": "diagnostic only and never evaluated by V5",
    }
    config["simulator_authority"]["v4_assets_reused_exactly"] = True
    config["simulator_authority"]["new_v5_corpora_generated"] = False
    config["prohibited"]["v4_test_replacement"] = True

    preflight = {
        "format": "RIVAL2_HUMAN_BC_V5_PRE_STEP_PREFLIGHT_V1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "required_starting_commit_is_ancestor": True,
        "v4_blocked_result_is_ancestor": True,
        "source_checkpoint_sha256_exact": True,
        "adapter_checkpoint_sha256_exact": True,
        "v4_authority_reused_without_new_corpus": True,
        "v4_sealed_test_checks": sealed_checks,
        "parent_relative_boost_rule_exact": True,
        "hard_max_sample_kl_exact": 2.0,
        "extra_1_0_and_0_5_selection_margins_absent": True,
        "optimizer_steps": 0,
        "human_test_accesses": 0,
        "reused_v4_test_student_evaluations": 0,
        "ppo_updates": 0,
        "reward_changes": 0,
        "valid": True,
    }

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RESULT_ROOT / "frozen_config.json", config)
    _write_json(RESULT_ROOT / "pre_step_preflight.json", preflight)
    reuse = {
        "format": "RIVAL2_HUMAN_BC_V5_V4_AUTHORITY_REUSE_MANIFEST_V1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "new_simulator_corpora_generated": 0,
        "v4_blocked_evidence": _record(V4_EVIDENCE),
        "v4_frozen_config": _record(V4_CONFIG),
        "retention_strata": _record(V4_STRATA),
        "stress_validation": _record(V4_STRESS),
        "untouched_test": _record(V4_TEST),
        "disjointness": _record(V4_DISJOINTNESS),
        "untouched_test_sealed_checks": sealed_checks,
    }
    _write_json(RESULT_ROOT / "v4_authority_reuse_manifest.json", reuse)
    (RESULT_ROOT / "README.md").write_text(
        "# Rival Human BC V5\n\n"
        "Minimal actor-only retry from accepted Human BC V1. V5 reuses the V4 "
        "simulator strata, complete/stress authorities, and still-sealed untouched "
        "test. It changes only the handoff-authorized boost-saturation rule, KL "
        "selection margins, retention weights, optimizer cadence, and selection/"
        "plateau settings. Detailed runtime evidence is generated by "
        "`benchmarks/run_rival2_human_bc_v5.py`.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "verdict": "V5_PROSPECTIVE_AUTHORITY_BUILT_NO_OPTIMIZER_STEPS",
                "frozen_config_sha256": file_sha256(
                    RESULT_ROOT / "frozen_config.json"
                ),
                "v4_untouched_test_reused_and_sealed": True,
                "optimizer_steps": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
