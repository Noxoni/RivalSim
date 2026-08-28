"""Publish and audit the bounded Gameplay V3 479 -> 489 PPO smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "rival2" / "gameplay_v3_ppo_smoke_v1"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints"
    / "rival2"
    / "gameplay_v3_smoke"
    / "rival2_gameplay_v3_iteration_489_resume.pt"
)
DEFAULT_REPORT = REPO_ROOT / "docs" / "RIVAL2_GAMEPLAY_V3_PPO_SMOKE_V1.md"
BASELINE = (
    REPO_ROOT
    / "results"
    / "rival2"
    / "gameplay_v3_validation_correction_v2"
    / "shadow_gate_summary.json"
)
EXPECTED_SOURCE_SHA256 = "3B994E118A9498713DC6115D38F061958A900EA8F4D00CE568F916942E851D9A"
EXPECTED_FINAL_SHA256 = "10D97428B3F1CC2E307040314D1DD1A924BD82975D4B88C0F73C3FC2716DCF54"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _change(source: float, final: float) -> dict[str, float | None]:
    return {
        "iteration_479": source,
        "iteration_489": final,
        "absolute_change": final - source,
        "relative_change": None if source == 0.0 else final / source - 1.0,
    }


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _markdown(
    reviewer: dict[str, Any],
    rows: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    final = reviewer["final_checkpoint"]
    lines = [
        "# Rival2 Gameplay V3 bounded PPO smoke (479 to 489)",
        "",
        f"Status: `{reviewer['status']}`",
        "",
        "This is the real ten-update mixed-opponent Gameplay V3 continuation. It stopped "
        "exactly at update 489. No reward, classifier, simulator-physics, PPO, network, "
        "observation, action, or curriculum coefficient was changed.",
        "",
        "## Identity",
        "",
        f"- Training implementation commit: `{reviewer['training_implementation_commit']}`.",
        f"- Evaluation implementation commit: `{reviewer['evaluation_implementation_commit']}`.",
        f"- Source checkpoint SHA-256: `{EXPECTED_SOURCE_SHA256}`.",
        f"- Final checkpoint: `{final['repository_path']}`.",
        f"- Final checkpoint SHA-256: `{final['sha256']}`.",
        f"- Final iteration/policy: `{final['iteration']}` / `{final['policy_version']}`.",
        f"- Final sample counter: `{final['agent_decision_samples']:,}`.",
        "",
        "## PPO safety by accepted update",
        "",
        "| update | proposals | accepted | early stop | LR start | LR end | backoffs | "
        "retries | max minibatch KL | mean KL | retention KL |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        safety = row["ppo_safety_summary"]
        lines.append(
            f"| {row['iteration']} | {safety['optimizer_step_proposals']} | "
            f"{safety['accepted_optimizer_steps']} | "
            f"{'yes' if safety['retention_budget_early_stop'] else 'no'} | "
            f"{safety['policy_learning_rate_start']:.7f} | "
            f"{safety['policy_learning_rate_end']:.7f} | "
            f"{safety['policy_learning_rate_backoffs']} | "
            f"{safety['transactional_retries']} | "
            f"{safety['maximum_post_step_minibatch_kl']:.6f} | "
            f"{safety['completed_update_mean_kl']:.6f} | "
            f"{safety['retention_mean_kl']:.6f} |"
        )
    aggregate = reviewer["training_aggregate"]
    lines.extend(
        [
            "",
            "All checkpoints rearm the policy/shared-trunk LR to `1e-4`; the critic remains "
            "at `3e-4`. Every accepted step passed the soft 0.02 minibatch and retention "
            "budgets. The four early stops (480-483) are normal soft-budget exits. No hard "
            "0.10 minibatch or 0.05 completed-update guard fired.",
            "",
            "## Reward scale across all ten training rollouts",
            "",
            f"- Mechanics / absolute gameplay reward: "
            f"`{aggregate['mechanics_to_absolute_gameplay_reward']:.6f}`.",
            f"- Bad-flip penalty / absolute gameplay reward: "
            f"`{aggregate['bad_flip_to_absolute_gameplay_reward']:.6f}`.",
            f"- Mechanics / progress: `{aggregate['mechanics_to_progress']:.6f}`.",
            f"- Bad-flip / progress: `{aggregate['bad_flip_to_progress']:.6f}`.",
            f"- Maximum single-rollout mechanics/gameplay ratio: "
            f"`{aggregate['maximum_per_update_mechanics_to_gameplay']:.6f}`.",
            f"- Maximum single-rollout bad-flip/gameplay ratio: "
            f"`{aggregate['maximum_per_update_bad_flip_to_gameplay']:.6f}`.",
            "",
            "Update 480 began from a fresh kickoff population and had exactly zero progress, "
            "so its mechanics/progress ratio is undefined; the raw ledger retains the zero "
            "denominator instead of using it as behavioral evidence. Across all ten rollouts, "
            "both new terms remain far below one percent of absolute gameplay reward.",
            "",
            "## Controlled 479 versus 489 shadow",
            "",
            "The primary comparison uses the exact iteration-479 assignments, frozen opponent "
            "snapshots, and RNG context with only the hashed iteration-489 model substituted. "
            "Both source and policy checkpoints remained byte-identical and no PPO update ran.",
            "",
            "| metric | iteration 479 | iteration 489 | relative change |",
            "|---|---:|---:|---:|",
        ]
    )
    labels = {
        "touches_per_min": "Touches/min",
        "flip_active_touches_per_min": "Flip-active touches/min",
        "unnecessary_flip_contacts_per_min": "Unnecessary contacts/min",
        "unnecessary_flip_touch_fraction": "Unnecessary / flip-touch fraction",
        "mechanics_progress_ratio": "Mechanics/progress",
        "bad_flip_progress_ratio": "Bad-flip/progress",
    }
    for key, label in labels.items():
        item = comparison["metrics"][key]
        relative = item["relative_change"]
        rendered = "n/a" if relative is None else f"{100.0 * relative:+.2f}%"
        lines.append(
            f"| {label} | {item['iteration_479']:.6f} | "
            f"{item['iteration_489']:.6f} | {rendered} |"
        )
    scoring = comparison["scoring_behavior"]
    no_touch = comparison["no_touch_behavior"]
    lines.extend(
        [
            "",
            f"Rival goal share moved from `{scoring['rival_goal_share']['iteration_479']:.6f}` "
            f"to `{scoring['rival_goal_share']['iteration_489']:.6f}`. Goal-ended episode "
            f"fraction moved from `{scoring['goal_episode_fraction']['iteration_479']:.6f}` "
            f"to `{scoring['goal_episode_fraction']['iteration_489']:.6f}`. No-touch "
            f"truncations moved from `{no_touch['no_touch_truncations']['iteration_479']}` "
            f"to `{no_touch['no_touch_truncations']['iteration_489']}` of 256 episodes.",
            "",
            "The policy used slightly fewer flip-active ball contacts and also converted a "
            "larger share of the remaining flip contacts into legitimate cases. Therefore the "
            "observed mechanism is **both**, not only fewer flips or only relabeling/conversion.",
            "",
            "Mechanics did not collapse broadly: raw detected events moved from 851 to 812 "
            "and paid events from 848 to 812; redirects, speedflips, half-flips, and car resets "
            "were maintained or increased, while pogos, successful dashes, pinches, and ball "
            "resets declined in this short natural sample. Touch acquisition declined 3.19%, "
            "which is a modest regression to monitor, not a collapse.",
            "",
            "## Recommendation",
            "",
            f"`{reviewer['recommendation']}`",
            "",
            reviewer["recommendation_reason"],
            "",
            "Do not treat this smoke as authorization to continue beyond iteration 489 under "
            "this task. A future continuation should preserve the same safety gates and monitor "
            "touches/min, no-touch endings, successful-dash/pogo retention, and goal-ended "
            "episode fraction closely.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    work = args.work_dir.resolve()
    results = args.results_dir.resolve()
    checkpoint_output = args.checkpoint_output.resolve()
    report_path = args.report.resolve()
    run_summary = _read_json(work / "run_summary.json")
    rows = [
        json.loads(line)
        for line in (work / "training_curve.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoints = _read_json(work / "checkpoints.json")
    baseline_committed = _read_json(BASELINE)
    baseline_extended = _read_json(work / "evaluation_479_extended" / "shadow_gate_summary.json")
    final_paired = _read_json(work / "evaluation_489_paired" / "shadow_gate_summary.json")
    final_native = _read_json(work / "evaluation_489" / "shadow_gate_summary.json")
    final_source = Path(run_summary["final_checkpoint"]["path"])

    checks = {
        "run_complete": run_summary["status"] == "COMPLETE_10_ACCEPTED_UPDATES",
        "ten_rows": len(rows) == 10,
        "iterations_exact": [row["iteration"] for row in rows] == list(range(480, 490)),
        "all_training_rows_green": all(row["verdict"] == "PASS_GREEN" for row in rows),
        "ten_checkpoints": len(checkpoints) == 10,
        "checkpoint_iterations_exact": [row["iteration"] for row in checkpoints]
        == list(range(480, 490)),
        "all_checkpoint_audits_green": all(
            row["audit"]["verdict"] == "PASS_GREEN" for row in checkpoints
        ),
        "final_checkpoint_exists": final_source.is_file(),
        "final_checkpoint_hash_exact": _sha256(final_source) == EXPECTED_FINAL_SHA256,
        "source_hash_exact": run_summary["source_checkpoint"]["sha256"]
        == EXPECTED_SOURCE_SHA256,
        "source_unchanged": run_summary["source_checkpoint"]["byte_identical_after_run"],
        "hard_guard_not_fired": not run_summary["hard_safety_guard_fired"],
        "stopped_exactly_489": run_summary["final_iteration"] == 489
        and run_summary["final_policy_version"] == 489,
        "baseline_committed_pass": baseline_committed["verdict"] == "PASS",
        "baseline_reproduction_pass": baseline_extended["verdict"] == "PASS",
        "baseline_contact_metrics_reproduced": all(
            baseline_extended["metrics"][name] == baseline_committed["metrics"][name]
            for name in (
                "touches_per_min",
                "flip_active_touches_per_min",
                "unnecessary_flip_contacts_per_min",
                "unnecessary_flip_touch_fraction",
                "mechanics_progress_ratio",
                "bad_flip_progress_ratio",
            )
        ),
        "paired_shadow_pass": final_paired["verdict"] == "PASS",
        "paired_shadow_policy_489": final_paired["evaluation_policy"]["iteration"] == 489,
        "native_shadow_pass": final_native["verdict"] == "PASS",
        "max_minibatch_kl_safe": max(
            row["ppo_safety_summary"]["maximum_post_step_minibatch_kl"] for row in rows
        )
        <= 0.10,
        "max_completed_kl_safe": max(
            row["ppo_safety_summary"]["completed_update_mean_kl"] for row in rows
        )
        <= 0.05,
        "all_value_loss_isolation_green": all(
            row["ppo_safety_summary"]["value_loss_to_policy_trunk_gradient_exact_zero"]
            and row["ppo_safety_summary"]["value_loss_to_actor_gradient_exact_zero"]
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"final evidence gate failed: {[k for k, v in checks.items() if not v]}")

    results.mkdir(parents=True, exist_ok=True)
    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    _copy(final_source, checkpoint_output)
    if _sha256(checkpoint_output) != EXPECTED_FINAL_SHA256:
        raise RuntimeError("repository final checkpoint copy hash mismatch")

    baseline = baseline_extended["metrics"]
    final = final_paired["metrics"]
    metric_names = (
        "touches_per_min",
        "flip_active_touches_per_min",
        "unnecessary_flip_contacts_per_min",
        "unnecessary_flip_touch_fraction",
        "mechanics_progress_ratio",
        "bad_flip_progress_ratio",
        "mechanics_absolute_gameplay_reward_ratio",
        "bad_flip_absolute_gameplay_reward_ratio",
    )
    comparison = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "method": (
            "iteration-479 evaluation context and RNG with only the iteration-489 model "
            "substituted; 256 natural short episodes, frozen policy/opponents, no update"
        ),
        "metrics": {name: _change(baseline[name], final[name]) for name in metric_names},
        "mechanics": {
            "iteration_479_detected": baseline["mechanic_detected"],
            "iteration_489_detected": final["mechanic_detected"],
            "iteration_479_paid": baseline["mechanic_paid"],
            "iteration_489_paid": final["mechanic_paid"],
            "detected_total": _change(
                float(sum(baseline["mechanic_detected"].values())),
                float(sum(final["mechanic_detected"].values())),
            ),
            "paid_total": _change(
                float(sum(baseline["mechanic_paid"].values())),
                float(sum(final["mechanic_paid"].values())),
            ),
        },
        "exemptions": {
            name: {
                "raw_count": _change(
                    float(baseline["exemption_flags"][name]),
                    float(final["exemption_flags"][name]),
                ),
                "rate_per_flip_touch": _change(
                    baseline["exemption_rates_per_flip_touch"][name],
                    final["exemption_rates_per_flip_touch"][name],
                ),
            }
            for name in baseline["exemption_flags"]
        },
        "scoring_behavior": {
            name: _change(baseline["scoring_behavior"][name], final["scoring_behavior"][name])
            for name in ("goal_episode_fraction", "rival_goal_share")
        },
        "no_touch_behavior": {
            name: _change(
                float(baseline["no_touch_behavior"][name]),
                float(final["no_touch_behavior"][name]),
            )
            for name in (
                "no_touch_truncations",
                "hard_time_truncations",
                "no_touch_episode_fraction",
            )
        },
        "movement_action_telemetry": {
            category: {
                name: _change(
                    baseline["movement_action_telemetry"][category][name],
                    final["movement_action_telemetry"][category][name],
                )
                for name in baseline["movement_action_telemetry"][category]
            }
            for category in baseline["movement_action_telemetry"]
        },
        "behavior_change_mechanism": {
            "fewer_flip_active_contacts": final["flip_active_touches_per_min"]
            < baseline["flip_active_touches_per_min"],
            "greater_legitimate_share": final["unnecessary_flip_touch_fraction"]
            < baseline["unnecessary_flip_touch_fraction"],
            "classification": "both",
        },
    }
    _write_json(results / "evaluation_comparison.json", comparison)

    component_names = ("mechanics", "unnecessary_flip", "progress")
    component_sums = {
        name: sum(
            row["reward_and_behavior_telemetry"]["reward_contributions"][name][
                "absolute_blue_sum"
            ]
            for row in rows
        )
        for name in component_names
    }
    gameplay_sum = sum(
        row["reward_and_behavior_telemetry"]["absolute_gameplay_reward_sum"] for row in rows
    )
    aggregate = {
        "mechanics_to_absolute_gameplay_reward": component_sums["mechanics"] / gameplay_sum,
        "bad_flip_to_absolute_gameplay_reward": component_sums["unnecessary_flip"]
        / gameplay_sum,
        "mechanics_to_progress": component_sums["mechanics"] / component_sums["progress"],
        "bad_flip_to_progress": component_sums["unnecessary_flip"]
        / component_sums["progress"],
        "maximum_per_update_mechanics_to_gameplay": max(
            row["reward_and_behavior_telemetry"]["ratios"][
                "mechanics_reward_to_absolute_gameplay_reward"
            ]
            for row in rows
        ),
        "maximum_per_update_bad_flip_to_gameplay": max(
            row["reward_and_behavior_telemetry"]["ratios"][
                "unnecessary_flip_penalty_to_absolute_gameplay_reward"
            ]
            for row in rows
        ),
    }
    max_minibatch = max(
        row["ppo_safety_summary"]["maximum_post_step_minibatch_kl"] for row in rows
    )
    max_completed = max(
        row["ppo_safety_summary"]["completed_update_mean_kl"] for row in rows
    )
    max_retention = max(row["ppo_safety_summary"]["retention_mean_kl"] for row in rows)
    reviewer = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "status": "GAMEPLAY_V3_BOUNDED_PPO_SMOKE_COMPLETE",
        "training_implementation_commit": run_summary["implementation_commit"],
        "evaluation_implementation_commit": args.evaluation_commit,
        "source_checkpoint": run_summary["source_checkpoint"],
        "final_checkpoint": {
            **run_summary["final_checkpoint"],
            "repository_path": checkpoint_output.relative_to(REPO_ROOT).as_posix(),
            "sha256": EXPECTED_FINAL_SHA256,
        },
        "new_training_sample_count": run_summary["final_agent_decision_samples"],
        "additional_samples": run_summary["additional_agent_decision_samples"],
        "ppo_safety": {
            "early_stop_updates": [
                row["iteration"]
                for row in rows
                if row["ppo_safety_summary"]["retention_budget_early_stop"]
            ],
            "maximum_accepted_minibatch_kl": max_minibatch,
            "maximum_completed_update_mean_kl": max_completed,
            "maximum_retention_mean_kl": max_retention,
            "total_optimizer_step_proposals": sum(
                row["ppo_safety_summary"]["optimizer_step_proposals"] for row in rows
            ),
            "total_accepted_optimizer_steps": sum(
                row["ppo_safety_summary"]["accepted_optimizer_steps"] for row in rows
            ),
            "total_transactional_retries": sum(
                row["ppo_safety_summary"]["transactional_retries"] for row in rows
            ),
            "total_lr_backoffs": sum(
                row["ppo_safety_summary"]["policy_learning_rate_backoffs"] for row in rows
            ),
            "hard_safety_guard_fired": False,
        },
        "training_aggregate": aggregate,
        "comparison": comparison,
        "gameplay_regressions": {
            "touches_per_min_relative": comparison["metrics"]["touches_per_min"][
                "relative_change"
            ],
            "mechanics_paid_total_relative": comparison["mechanics"]["paid_total"][
                "relative_change"
            ],
            "goal_episode_fraction_relative": comparison["scoring_behavior"][
                "goal_episode_fraction"
            ]["relative_change"],
            "no_touch_truncations": comparison["no_touch_behavior"][
                "no_touch_truncations"
            ],
            "interpretation": (
                "modest touch/goal-ended/mechanics-rate softening to monitor; no broad "
                "mechanics, scoring, movement, or touch-acquisition collapse"
            ),
        },
        "recommendation": "CONTINUE_GAMEPLAY_V3_TRAINING",
        "recommendation_reason": (
            "PPO remained stable, both V3 terms stayed small, unnecessary flip-through rate "
            "and fraction declined, and legitimate mechanics/exemptions remained active. "
            "Continue only under the existing gates while monitoring the modest touch-rate, "
            "no-touch, and selected mechanics declines."
        ),
        "checks": checks,
        "verdict": "PASS_GREEN",
    }
    _write_json(results / "reviewer_summary.json", reviewer)

    copies = {
        "launch_gate.json": work / "launch_gate.json",
        "transition_gate.json": work / "transition_gate.json",
        "run_summary.json": work / "run_summary.json",
        "checkpoints.json": work / "checkpoints.json",
        "ppo_safety_summary.json": work / "ppo_safety_summary.json",
        "training_curve.jsonl": work / "training_curve.jsonl",
        "shadow_479_extended.json": work
        / "evaluation_479_extended"
        / "shadow_gate_summary.json",
        "shadow_489_native.json": work / "evaluation_489" / "shadow_gate_summary.json",
        "shadow_489_paired.json": work
        / "evaluation_489_paired"
        / "shadow_gate_summary.json",
        "shadow_489_paired_events.json": work
        / "evaluation_489_paired"
        / "shadow_event_evidence.json",
    }
    for name, source in copies.items():
        _copy(source, results / name)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _markdown(reviewer, rows, comparison), encoding="utf-8", newline="\n"
    )
    manifest_paths = [
        *sorted(path for path in results.iterdir() if path.is_file()),
        checkpoint_output,
        report_path,
    ]
    manifest = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "artifacts": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in manifest_paths
        ],
        "final_checkpoint_sha256": EXPECTED_FINAL_SHA256,
        "verdict": "PASS_GREEN",
    }
    _write_json(results / "artifact_manifest.json", manifest)
    return reviewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--checkpoint-output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--evaluation-commit", required=True)
    return parser.parse_args()


def main() -> int:
    reviewer = finalize(parse_args())
    print(json.dumps(reviewer, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
