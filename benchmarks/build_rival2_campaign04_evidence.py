"""Publish the bounded Rival 2.0 Campaign 04 continuation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

import benchmarks.run_rival2_campaign04 as campaign04

SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "touches_per_simulated_minute": float(result["touches_per_simulated_minute"]),
        "goals_per_simulated_minute": float(result["goals_per_simulated_minute"]),
        "goal_terminated_fraction": float(result["goal_terminated_fraction"]),
        "no_touch_truncated_fraction": float(result["no_touch_truncated_fraction"]),
        "mean_episode_duration_seconds": float(result["mean_episode_duration_seconds"]),
        "mean_absolute_analog_action": result["mean_absolute_analog_action"],
        "button_activation_rate": result["button_activation_rate"],
        "mean_analog_policy_std": result["mean_analog_policy_std"],
        "mean_button_probability": result["mean_button_probability"],
        "mean_button_entropy": result["mean_button_entropy"],
    }


def _numeric_metrics(value: dict[str, Any]) -> dict[str, float]:
    return {name: number for name, number in value.items() if isinstance(number, float)}


def _delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_numeric = _numeric_metrics(left)
    right_numeric = _numeric_metrics(right)
    return {name: right_numeric[name] - left_numeric[name] for name in left_numeric}


def _trend_classification(points: list[dict[str, Any]]) -> dict[str, Any]:
    previous = points[-2]["metrics"]
    final = points[-1]["metrics"]
    touches_improved = (
        final["touches_per_simulated_minute"]
        > previous["touches_per_simulated_minute"]
    )
    no_touch_improved = (
        final["no_touch_truncated_fraction"]
        < previous["no_touch_truncated_fraction"]
    )
    touches_worsened = (
        final["touches_per_simulated_minute"]
        < previous["touches_per_simulated_minute"]
    )
    no_touch_worsened = (
        final["no_touch_truncated_fraction"]
        > previous["no_touch_truncated_fraction"]
    )
    if touches_improved and no_touch_improved:
        classification = "CONTINUING"
    elif touches_worsened and no_touch_worsened:
        classification = "DEGRADING"
    else:
        classification = "FLATTENING"
    return {
        "classification": classification,
        "rule": "frozen before continuation in Campaign 04 config",
        "comparison": "1B versus 750M",
        "touches_axis_improved": touches_improved,
        "no_touch_axis_improved": no_touch_improved,
        "touches_axis_worsened": touches_worsened,
        "no_touch_axis_worsened": no_touch_worsened,
        "no_numeric_success_threshold": True,
    }


def _ppo_summary(curve: list[dict[str, Any]]) -> dict[str, Any]:
    def maximum(name: str) -> dict[str, Any]:
        point = max(curve, key=lambda item: item["ppo_stability"][name])
        return {"value": point["ppo_stability"][name], "iteration": point["iteration"]}

    return {
        "continuation_update_count": len(curve),
        "first_iteration": curve[0]["iteration"],
        "last_iteration": curve[-1]["iteration"],
        "all_updates_integrity_green": all(
            point["integrity"]["verdict"] == "PASS_GREEN" for point in curve
        ),
        "maximum_approximate_kl": maximum("approximate_kl"),
        "maximum_clip_fraction": maximum("clip_fraction"),
        "maximum_gradient_norm": maximum("gradient_norm"),
        "maximum_post_clip_gradient_norm": maximum("post_clip_gradient_norm"),
        "mean_agent_decisions_per_second": sum(
            point["agent_decisions_per_second"] for point in curve
        )
        / len(curve),
        "minimum_agent_decisions_per_second": min(
            point["agent_decisions_per_second"] for point in curve
        ),
        "entropy_coefficient": 0.0,
        "entropy_optimization_contribution": 0.0,
    }


def _report_text(
    *,
    summary: dict[str, Any],
    comparison: dict[str, Any],
    ppo: dict[str, Any],
) -> str:
    points = comparison["points"]

    def row(point: dict[str, Any]) -> str:
        metrics = point["metrics"]
        return (
            f"| {point['label']} | {point['iteration']} | "
            f"{point['agent_decision_samples']:,} | "
            f"{metrics['touches_per_simulated_minute']:.6f} | "
            f"{metrics['goals_per_simulated_minute']:.6f} | "
            f"{metrics['goal_terminated_fraction']:.6f} | "
            f"{metrics['no_touch_truncated_fraction']:.6f} | "
            f"{metrics['mean_episode_duration_seconds']:.6f} |"
        )

    rows = "\n".join(row(point) for point in points)
    final = points[-1]["metrics"]
    previous = points[-2]["metrics"]
    baseline = points[0]["metrics"]
    checkpoint_size = summary["published_checkpoint_size_bytes"]
    return f"""# Rival 2.0 Campaign 04 Results

Campaign 04 completed the exact long-run continuation of the Campaign 03 Reward V2 policy.
It loaded checkpoint SHA-256 `{summary['resume_checkpoint_sha256']}` with optimizer, Torch and
CUDA RNGs, policy/opponent generator states, counters, opponent assignments, and historical
policies intact. Training resumed at update 12 / 100,663,296 cumulative samples and stopped at
update 120 / 1,006,632,960 samples. Update 121 did not run.

No preflight, reward smoke, baseline rerun, world-count sweep, inherited parity/regression suite,
post-run test/lint/compile ceremony, viewer work, or v0.6 work was performed.

## Training integrity

All {summary['continuation_update_count']} continuation updates passed finite-state, optimizer,
policy-version, sample-count, action-bound, historical-policy, Reward V2 identity, and zero
tracked hot-path transfer checks. The final checkpoint passed exact reload and continuation.

- maximum approximate KL: `{ppo['maximum_approximate_kl']['value']:.6f}` at update
  `{ppo['maximum_approximate_kl']['iteration']}`;
- maximum clip fraction: `{ppo['maximum_clip_fraction']['value']:.6f}` at update
  `{ppo['maximum_clip_fraction']['iteration']}`;
- maximum gradient norm: `{ppo['maximum_gradient_norm']['value']:.6f}`;
- mean continuation throughput: `{ppo['mean_agent_decisions_per_second']:,.2f}` agent
  decisions/second;
- total continuation wall time including the four evaluations:
  `{summary['campaign_wall_seconds_including_evaluations']:.3f}` seconds.

## Authorized behavioral curve

Each new point is exactly one 4,096-world ordinary stochastic self-play evaluation at seed
`920260826`. The 100M Campaign 03 point is reused from its published evidence and was not rerun.

| Point | Update | Cumulative samples | Touches/min | Goals/min | Goal fraction | No-touch fraction | Mean duration, s |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

The prospectively frozen two-axis classification is
**`{summary['behavioral_trend']}`**: from 750M to 1B, touches/minute increased from
`{previous['touches_per_simulated_minute']:.6f}` to
`{final['touches_per_simulated_minute']:.6f}`, while no-touch truncation fell from
`{previous['no_touch_truncated_fraction']:.6f}` to
`{final['no_touch_truncated_fraction']:.6f}`. Relative to the 100M baseline, the final policy
increased touches/minute by
`{final['touches_per_simulated_minute'] - baseline['touches_per_simulated_minute']:+.6f}` and
reduced no-touch truncation by
`{baseline['no_touch_truncated_fraction'] - final['no_touch_truncated_fraction']:.6f}` absolute.

Secondary goal metrics were not monotonic: the 1B goal rate was
`{final['goals_per_simulated_minute']:.6f}`, down from the 750M value
`{previous['goals_per_simulated_minute']:.6f}` even while the two frozen approach-learning axes
continued improving. No setting was changed in response.

## Final checkpoint

The exact final resumable checkpoint is
`checkpoints/rival2/campaign04/rival2_campaign04_1b_resume.pt`. It is
`{checkpoint_size:,}` bytes with SHA-256 `{summary['published_checkpoint_sha256']}` and contains
policy/update version 120, 1,006,632,960 cumulative samples, optimizer/RNG/assignment state, and
historical policy versions `[0, 2, 3, 6, 12, 30, 60, 90, 120]`.

Campaign 04 is closed at the 1B boundary. The result does not itself authorize a viewer,
additional training, reward/PPO changes, or v0.6 RocketSim/RLBot transfer work.
"""


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    configuration = _read_json(work_dir / "config_frozen_before_training.json")
    resume_authority = _read_json(work_dir / "resume_authority.json")
    loaded_state = _read_json(work_dir / "loaded_state.json")
    run = _read_json(work_dir / "run_summary.json")
    curve = _read_json(work_dir / "training_curve.json")
    checkpoint_data = _read_json(work_dir / "checkpoints.json")

    if configuration != campaign04.frozen_configuration():
        raise RuntimeError("Campaign 04 frozen configuration differs from runner authority")
    if resume_authority["verdict"] != "PASS_GREEN" or not all(
        resume_authority["checks"].values()
    ):
        raise RuntimeError("Campaign 04 resume checkpoint authority failed")
    if loaded_state["verdict"] != "PASS_GREEN" or not all(loaded_state["checks"].values()):
        raise RuntimeError("Campaign 04 loaded state failed")
    if run["execution_status"] != "COMPLETE":
        raise RuntimeError("Campaign 04 execution did not complete")
    if (
        run["final_iteration"] != campaign04.TARGET_ITERATION
        or run["final_policy_version"] != campaign04.TARGET_ITERATION
        or run["final_agent_decision_samples"] != campaign04.TARGET_SAMPLES
    ):
        raise RuntimeError("Campaign 04 final boundary is not exact")
    if len(curve) != 108 or [point["iteration"] for point in curve] != list(range(13, 121)):
        raise RuntimeError("Campaign 04 continuation update sequence differs")
    if curve[-1]["agent_decision_samples"] != campaign04.TARGET_SAMPLES:
        raise RuntimeError("Campaign 04 final curve sample count differs")
    if any(point["integrity"]["verdict"] != "PASS_GREEN" for point in curve):
        raise RuntimeError("Campaign 04 contains a failed training update")
    if run["final_checkpoint_reload"]["verdict"] != "PASS_GREEN":
        raise RuntimeError("Campaign 04 final checkpoint reload failed")

    checkpoints = checkpoint_data["checkpoints"]
    if len(checkpoints) != len(campaign04.CHECKPOINT_THRESHOLDS):
        raise RuntimeError("Campaign 04 checkpoint count differs")
    evaluations: list[dict[str, Any]] = []
    for checkpoint, expected in zip(
        checkpoints, campaign04.CHECKPOINT_THRESHOLDS, strict=True
    ):
        label, iteration, samples = expected
        path = Path(checkpoint["path"])
        if (
            checkpoint["label"] != label
            or checkpoint["iteration"] != iteration
            or checkpoint["policy_version"] != iteration
            or checkpoint["agent_decision_samples"] != samples
            or not path.is_file()
            or _sha256_file(path) != checkpoint["sha256"]
        ):
            raise RuntimeError(f"Campaign 04 {label} checkpoint custody failed")
        evaluation = _read_json(work_dir / f"evaluation_{label}.json")
        if (
            evaluation["verdict"] != "PASS_GREEN"
            or evaluation["iteration"] != iteration
            or evaluation["agent_decision_samples"] != samples
            or evaluation["evaluation_seed"] != campaign04.EVALUATION_SEED
            or evaluation["evaluation_worlds"] != campaign04.EVALUATION_WORLDS
        ):
            raise RuntimeError(f"Campaign 04 {label} evaluation failed")
        evaluations.append(evaluation)

    expected_evaluation_files = {
        f"evaluation_{label}.json" for label, _, _ in campaign04.CHECKPOINT_THRESHOLDS
    }
    actual_evaluation_files = {
        path.name for path in work_dir.glob("evaluation_*.json")
    }
    if actual_evaluation_files != expected_evaluation_files:
        raise RuntimeError("Campaign 04 evaluation set contains missing or extra runs")

    baseline_evaluation = _read_json(
        Path("results/rival2/campaign03/final_evaluation.json")
    )
    points = [
        {
            "label": "100m",
            "iteration": 12,
            "agent_decision_samples": 100_663_296,
            "source": "published Campaign 03 final; not rerun",
            "metrics": _metrics(baseline_evaluation["result"]),
        }
    ]
    points.extend(
        {
            "label": evaluation["checkpoint_label"],
            "iteration": evaluation["iteration"],
            "agent_decision_samples": evaluation["agent_decision_samples"],
            "source": "Campaign 04 authorized evaluation",
            "metrics": _metrics(evaluation["result"]),
        }
        for evaluation in evaluations
    )
    for index, point in enumerate(points):
        point["delta_from_100m"] = _delta(points[0]["metrics"], point["metrics"])
        point["delta_from_previous"] = (
            None
            if index == 0
            else _delta(points[index - 1]["metrics"], point["metrics"])
        )
    trend = _trend_classification(points)
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": configuration["evaluation"],
        "frozen_trend_rule": configuration[
            "behavioral_trend_rule_frozen_before_continuation"
        ],
        "trend": trend,
        "points": points,
    }
    ppo = _ppo_summary(curve)

    final_local = Path(checkpoints[-1]["path"])
    final_payload = torch.load(final_local, map_location="cpu", weights_only=False)
    if (
        final_payload["iteration"] != campaign04.TARGET_ITERATION
        or final_payload["policy_version"] != campaign04.TARGET_ITERATION
        or final_payload["total_agent_samples"] != campaign04.TARGET_SAMPLES
        or final_payload["reward_version"] != campaign04.RIVAL2_REWARD_V2_VERSION
        or final_payload["contract_hashes"] != configuration["active_contract_hashes"]
        or [entry["version"] for entry in final_payload["historical_opponents"]]
        != [0, 2, 3, 6, 12, 30, 60, 90, 120]
    ):
        raise RuntimeError("Campaign 04 final checkpoint payload differs")

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    published_checkpoint = args.checkpoints_dir / "rival2_campaign04_1b_resume.pt"
    shutil.copyfile(final_local, published_checkpoint)
    if _sha256_file(published_checkpoint) != checkpoints[-1]["sha256"]:
        raise RuntimeError("published Campaign 04 checkpoint differs from local custody")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.results_dir / "config.json", configuration)
    _write_json(args.results_dir / "resume_authority.json", resume_authority)
    _write_json(args.results_dir / "loaded_state.json", loaded_state)
    _write_json(args.results_dir / "training_curve.json", curve)
    for evaluation in evaluations:
        _write_json(
            args.results_dir / f"evaluation_{evaluation['checkpoint_label']}.json",
            evaluation,
        )
    _write_json(args.results_dir / "behavioral_curve.json", comparison)
    _write_json(args.results_dir / "ppo_stability.json", ppo)
    _write_json(
        args.results_dir / "checkpoints.json",
        {
            "schema_version": SCHEMA_VERSION,
            "local_checkpoint_custody": checkpoints,
            "published_final_checkpoint": {
                "path": published_checkpoint.as_posix(),
                "sha256": _sha256_file(published_checkpoint),
                "size_bytes": published_checkpoint.stat().st_size,
                "format": "RIVAL2_CHECKPOINT_V1",
                "reward_version": campaign04.RIVAL2_REWARD_V2_VERSION,
                "artifact_kind": "full_resumable_training_checkpoint",
            },
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "campaign": "Rival 2.0 Campaign 04 Reward V2 long-run continuation",
        "execution_status": "COMPLETE",
        "behavioral_trend": trend["classification"],
        "resume_checkpoint_sha256": resume_authority["checkpoint_sha256"],
        "resume_authority_verdict": resume_authority["verdict"],
        "start_iteration": campaign04.START_ITERATION,
        "start_agent_decision_samples": campaign04.START_SAMPLES,
        "final_iteration": campaign04.TARGET_ITERATION,
        "final_agent_decision_samples": campaign04.TARGET_SAMPLES,
        "continuation_update_count": len(curve),
        "update_integrity_pass_count": len(curve),
        "authorized_evaluation_count": len(evaluations),
        "evaluation_pass_count": len(evaluations),
        "campaign_wall_seconds_including_evaluations": run[
            "campaign_wall_seconds_including_evaluations"
        ],
        "final_checkpoint_reload_verdict": run["final_checkpoint_reload"]["verdict"],
        "published_checkpoint_sha256": _sha256_file(published_checkpoint),
        "published_checkpoint_size_bytes": published_checkpoint.stat().st_size,
        "behavioral_points": {
            point["label"]: _numeric_metrics(point["metrics"]) for point in points
        },
        "maximum_approximate_kl": ppo["maximum_approximate_kl"],
        "maximum_clip_fraction": ppo["maximum_clip_fraction"],
        "update_121_run": False,
        "extra_evaluation_run": False,
        "preflight_regression_lint_ceremony_run": False,
        "viewer_built": False,
        "v06_begun": False,
    }
    _write_json(args.results_dir / "summary.json", summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _report_text(summary=summary, comparison=comparison, ppo=ppo),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
