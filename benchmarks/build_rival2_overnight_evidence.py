"""Validate and publish the compact Rival 2.0 overnight curriculum evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

import benchmarks.run_rival2_overnight as overnight
from rivalsim.rival2_contracts import (
    RIVAL2_REWARD_V2_VERSION,
    RIVAL2_REWARD_VERSION,
    contract_hashes_for_reward,
)

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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _checkpoint_custody(record: dict[str, Any], reward_version: str) -> Path:
    path = Path(record["path"])
    if (
        not path.is_file()
        or _sha256_file(path) != record["sha256"]
        or path.stat().st_size != record["size_bytes"]
        or record["reward_version"] != reward_version
        or record["contract_hashes"] != contract_hashes_for_reward(reward_version)
        or record["iteration"] != record["policy_version"]
    ):
        raise RuntimeError(f"checkpoint custody failed for {record['label']}")
    return path


def _evaluation_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    result = evaluation["result"]
    return {
        "touches_per_simulated_minute": result["touches_per_simulated_minute"],
        "goals_per_simulated_minute": result["goals_per_simulated_minute"],
        "goal_terminated_fraction": result["goal_terminated_fraction"],
        "no_touch_truncated_fraction": result["no_touch_truncated_fraction"],
        "hard_truncated_fraction": result["hard_truncated_fraction"],
        "mean_episode_duration_seconds": result["mean_episode_duration_seconds"],
        "mean_absolute_analog_action": result["mean_absolute_analog_action"],
        "button_activation_rate": result["button_activation_rate"],
        "mean_analog_policy_std": result["mean_analog_policy_std"],
        "mean_button_probability": result["mean_button_probability"],
        "mean_button_entropy": result["mean_button_entropy"],
    }


def _evaluation_point(
    evaluation: dict[str, Any], *, update_offset: int | None = None
) -> dict[str, Any]:
    point = {
        "label": evaluation["checkpoint_label"],
        "phase": evaluation["phase"],
        "reward_version": evaluation["reward_version"],
        "iteration": evaluation["iteration"],
        "policy_version": evaluation["policy_version"],
        "agent_decision_samples": evaluation["agent_decision_samples"],
        "evaluation_seed": evaluation["evaluation_seed"],
        "evaluation_worlds": evaluation["evaluation_worlds"],
        "evaluation_wall_seconds": evaluation["wall_seconds"],
        "metrics": _evaluation_metrics(evaluation),
        "verdict": evaluation["verdict"],
    }
    if update_offset is not None:
        point["phase_update_offset"] = update_offset
    for name in (
        "acquisition_threshold",
        "acquisition_threshold_passed",
        "consecutive_passing_evaluations",
        "phase_c_threshold_seconds",
        "phase_c_elapsed_seconds_at_update_completion",
        "phase_c_elapsed_seconds_after_evaluation",
    ):
        if name in evaluation:
            point[name] = evaluation[name]
    return point


def _load_evaluation(work_dir: Path, label: str) -> dict[str, Any]:
    evaluation = _read_json(work_dir / f"evaluation_{label}.json")
    if (
        evaluation["verdict"] != "PASS_GREEN"
        or evaluation["evaluation_seed"] != overnight.EVALUATION_SEED
        or evaluation["evaluation_worlds"] != overnight.EVALUATION_WORLDS
        or evaluation["mode"] != "ordinary stochastic self-play"
        or evaluation["checkpoint_label"] != label
    ):
        raise RuntimeError(f"evaluation protocol failed for {label}")
    return evaluation


def _phase_ppo_summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    def maximum(name: str) -> dict[str, Any]:
        point = max(points, key=lambda item: item["ppo_stability"][name])
        return {"value": point["ppo_stability"][name], "iteration": point["iteration"]}

    return {
        "update_count": len(points),
        "first_iteration": points[0]["iteration"],
        "last_iteration": points[-1]["iteration"],
        "first_agent_decision_samples": points[0]["agent_decision_samples"],
        "last_agent_decision_samples": points[-1]["agent_decision_samples"],
        "all_updates_integrity_green": all(
            point["integrity"]["verdict"] == "PASS_GREEN" for point in points
        ),
        "maximum_approximate_kl": maximum("approximate_kl"),
        "maximum_clip_fraction": maximum("clip_fraction"),
        "maximum_gradient_norm": maximum("gradient_norm"),
        "maximum_post_clip_gradient_norm": maximum("post_clip_gradient_norm"),
        "mean_agent_decisions_per_second": sum(
            point["agent_decisions_per_second"] for point in points
        )
        / len(points),
        "minimum_agent_decisions_per_second": min(
            point["agent_decisions_per_second"] for point in points
        ),
        "maximum_agent_decisions_per_second": max(
            point["agent_decisions_per_second"] for point in points
        ),
        "entropy_coefficient": 0.0,
        "entropy_optimization_contribution": 0.0,
    }


def _update_ledger(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for point in points:
        stability = point["ppo_stability"]
        entry = {
            "phase": point["phase"],
            "reward_version": point["reward_version"],
            "iteration": point["iteration"],
            "policy_version": point["policy_version"],
            "phase_update_offset": point["phase_update_offset"],
            "agent_decision_samples": point["agent_decision_samples"],
            "iteration_agent_decision_samples": point["iteration_agent_decision_samples"],
            "wall_seconds": point["wall_seconds"],
            "agent_decisions_per_second": point["agent_decisions_per_second"],
            "approximate_kl": stability["approximate_kl"],
            "clip_fraction": stability["clip_fraction"],
            "gradient_norm": stability["gradient_norm"],
            "post_clip_gradient_norm": stability["post_clip_gradient_norm"],
            "integrity_verdict": point["integrity"]["verdict"],
        }
        if "phase_c_elapsed_seconds_at_update_completion" in point:
            entry["phase_c_elapsed_seconds_at_update_completion"] = point[
                "phase_c_elapsed_seconds_at_update_completion"
            ]
        ledger.append(entry)
    return ledger


def _report_table(points: list[dict[str, Any]], first_column: str) -> str:
    rows: list[str] = []
    for point in points:
        metrics = point["metrics"]
        if first_column == "Phase A update":
            name = str(point["iteration"])
        elif first_column == "Phase B offset":
            name = f"+{point['phase_update_offset']}"
        else:
            name = point["label"].removeprefix("phase_c_")
        rows.append(
            f"| {name} | {point['iteration']} | {point['agent_decision_samples']:,} | "
            f"{metrics['touches_per_simulated_minute']:.6f} | "
            f"{metrics['goals_per_simulated_minute']:.6f} | "
            f"{metrics['goal_terminated_fraction']:.6f} | "
            f"{metrics['no_touch_truncated_fraction']:.6f} | "
            f"{metrics['mean_episode_duration_seconds']:.6f} |"
        )
    return (
        f"| {first_column} | Update | Cumulative samples | Touches/min | Goals/min | "
        "Goal fraction | No-touch fraction | Mean duration, s |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
    )


def _report_text(
    *,
    summary: dict[str, Any],
    phase_a_curve: dict[str, Any],
    transition: dict[str, Any],
    phase_b_curve: dict[str, Any],
    phase_c_curve: dict[str, Any],
    ppo: dict[str, Any],
) -> str:
    a_points = phase_a_curve["points"]
    b_points = phase_b_curve["points"]
    c_points = phase_c_curve["points"]
    final = c_points[-1]
    return f"""# Rival 2.0 Overnight Curriculum Results

The authorized uninterrupted curriculum completed from the exact Campaign 04 checkpoint.
Reward V2 acquisition training continued until two consecutive 4,096-world held-out
evaluations met the no-touch threshold, the reward alone was explicitly migrated to the
preserved base `RIVAL2_REWARD_V1`, exactly 239 additional PPO updates were completed, and the
same V1 policy then trained until the first completed update at or after six real elapsed
hours. The viewer was not built and v0.6 was not begun.

## Phase A — Reward V2 acquisition completion

{_report_table(a_points, 'Phase A update')}

The first qualifying consecutive pair was updates
`{a_points[-2]['iteration']}` and `{a_points[-1]['iteration']}`, with no-touch fractions
`{a_points[-2]['metrics']['no_touch_truncated_fraction']:.6f}` and
`{a_points[-1]['metrics']['no_touch_truncated_fraction']:.6f}`. The confirming checkpoint is
`{summary['published_acquisition_checkpoint']['path']}` with SHA-256
`{summary['published_acquisition_checkpoint']['sha256']}`.

## Explicit Reward V2 -> Reward V1 transition

The transition occurred at update `{transition['transition']['transition_iteration']}` /
`{transition['transition']['transition_agent_decision_samples']:,}` cumulative samples. The
source checkpoint SHA-256 was `{transition['transition']['parent_checkpoint_sha256']}`. Every
model, optimizer, RNG, counter, assignment, historical-policy, and live runtime identity check
was exact; only the reward version/contracts changed, and the transition record was embedded in
the post-transition and all descendant checkpoints.

## Phase B — 2B additional base-reward samples

{_report_table(b_points, 'Phase B offset')}

Phase B completed exactly `{summary['phase_b_additional_updates']}` updates /
`{summary['phase_b_additional_agent_decision_samples']:,}` additional samples. The durable 2B
base-reward checkpoint is `{summary['published_phase_b_checkpoint']['path']}` with SHA-256
`{summary['published_phase_b_checkpoint']['sha256']}` and passed exact reload.

## Phase C — six real elapsed hours

{_report_table(c_points, 'Elapsed point')}

The hourly trigger details were:

{chr(10).join(f"- {point['label'].removeprefix('phase_c_')}: update {point['iteration']}, " f"{point['agent_decision_samples']:,} samples, " f"{point['phase_c_elapsed_seconds_at_update_completion']:.3f} elapsed seconds at update completion, " f"{point['phase_c_elapsed_seconds_after_evaluation']:.3f} seconds after evaluation." for point in c_points)}

The final training update completed at
`{summary['phase_c_stop_elapsed_seconds_at_update_completion']:.3f}` elapsed seconds. Its held-out
evaluation recorded `{final['metrics']['touches_per_simulated_minute']:.6f}` touches/minute,
`{final['metrics']['goals_per_simulated_minute']:.6f}` goals/minute, and
`{final['metrics']['no_touch_truncated_fraction']:.6f}` no-touch truncation.

## Training integrity and final checkpoint

All `{summary['total_continuation_updates']}` continuation updates passed the trainer's finite,
optimizer, policy/sample counter, action-bound, active-reward-contract, entropy-zero, and
zero-hot-transfer checks. Mean throughput across the full continuation was
`{ppo['overall']['mean_agent_decisions_per_second']:,.2f}` agent decisions/second. Maximum
approximate KL was `{ppo['overall']['maximum_approximate_kl']['value']:.6f}` at update
`{ppo['overall']['maximum_approximate_kl']['iteration']}`; maximum clip fraction was
`{ppo['overall']['maximum_clip_fraction']['value']:.6f}` at update
`{ppo['overall']['maximum_clip_fraction']['iteration']}`.

The final full resumable checkpoint is `{summary['published_final_checkpoint']['path']}`. It is
`{summary['published_final_checkpoint']['size_bytes']:,}` bytes with SHA-256
`{summary['published_final_checkpoint']['sha256']}`, binds to `RIVAL2_REWARD_V1`, retains the
authorized curriculum-transition record, and passed exact reload.

No preflight, smoke, parity/regression, pytest, Ruff, compileall, viewer, or v0.6 work was run.
The overnight curriculum is closed at this final six-hour boundary.
"""


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    configuration = _read_json(work_dir / "config_frozen_before_training.json")
    resume_authority = _read_json(work_dir / "resume_authority.json")
    loaded_state = _read_json(work_dir / "loaded_state.json")
    phase_a = _read_json(work_dir / "phase_a_summary.json")
    phase_a_progress = _read_json(work_dir / "phase_a_progress.json")
    transition = _read_json(work_dir / "reward_transition.json")
    phase_b = _read_json(work_dir / "phase_b_summary.json")
    phase_c = _read_json(work_dir / "phase_c_summary.json")
    run = _read_json(work_dir / "run_summary.json")
    training_points = _read_jsonl(work_dir / "training_curve.jsonl")

    if configuration != overnight.frozen_configuration():
        raise RuntimeError("overnight frozen configuration differs from runner authority")
    if resume_authority["verdict"] != "PASS_GREEN" or not all(
        resume_authority["checks"].values()
    ):
        raise RuntimeError("overnight resume authority failed")
    if loaded_state["verdict"] != "PASS_GREEN" or not all(loaded_state["checks"].values()):
        raise RuntimeError("overnight loaded state failed")
    if run["execution_status"] != "COMPLETE":
        raise RuntimeError("overnight curriculum did not complete")
    if not training_points:
        raise RuntimeError("overnight training curve is empty")
    final_iteration = run["final_iteration"]
    expected_iterations = list(range(overnight.START_ITERATION + 1, final_iteration + 1))
    if [point["iteration"] for point in training_points] != expected_iterations:
        raise RuntimeError("overnight update sequence is not contiguous")
    if any(point["policy_version"] != point["iteration"] for point in training_points):
        raise RuntimeError("overnight policy/update version sequence differs")
    if any(
        point["iteration_agent_decision_samples"] != overnight.ROLLOUT_AGENT_SAMPLES
        for point in training_points
    ):
        raise RuntimeError("overnight per-update sample count differs")
    if any(point["integrity"]["verdict"] != "PASS_GREEN" for point in training_points):
        raise RuntimeError("overnight training contains a failed update")
    if training_points[-1]["agent_decision_samples"] != run["final_agent_decision_samples"]:
        raise RuntimeError("overnight final curve sample count differs")
    if run["final_agent_decision_samples"] != (
        overnight.START_SAMPLES
        + len(training_points) * overnight.ROLLOUT_AGENT_SAMPLES
    ):
        raise RuntimeError("overnight cumulative sample accounting differs")

    # Phase A: exact 30-update cadence and first two-consecutive <=1% crossing.
    if phase_a["execution_status"] != "COMPLETE" or phase_a["reward_version"] != RIVAL2_REWARD_V2_VERSION:
        raise RuntimeError("Phase A summary differs")
    phase_a_checkpoint_records = phase_a_progress["checkpoints"]
    phase_a_eval_records = phase_a_progress["evaluations"]
    expected_phase_a_iterations = list(
        range(
            overnight.START_ITERATION + overnight.PHASE_A_EVALUATION_INTERVAL,
            phase_a["final_iteration"] + 1,
            overnight.PHASE_A_EVALUATION_INTERVAL,
        )
    )
    if [item["iteration"] for item in phase_a_checkpoint_records] != expected_phase_a_iterations:
        raise RuntimeError("Phase A checkpoint cadence differs")
    if len(phase_a_checkpoint_records) != len(phase_a_eval_records) or len(phase_a_eval_records) < 2:
        raise RuntimeError("Phase A checkpoint/evaluation count differs")
    phase_a_points: list[dict[str, Any]] = []
    for checkpoint, recorded in zip(
        phase_a_checkpoint_records, phase_a_eval_records, strict=True
    ):
        _checkpoint_custody(checkpoint, RIVAL2_REWARD_V2_VERSION)
        evaluation = _load_evaluation(work_dir, checkpoint["label"])
        if evaluation != recorded or evaluation["reward_version"] != RIVAL2_REWARD_V2_VERSION:
            raise RuntimeError(f"Phase A evaluation custody failed at {checkpoint['label']}")
        passed = evaluation["result"]["no_touch_truncated_fraction"] <= overnight.PHASE_A_THRESHOLD
        if passed != evaluation["acquisition_threshold_passed"]:
            raise RuntimeError("Phase A threshold classification differs")
        phase_a_points.append(_evaluation_point(evaluation))
    pass_flags = [point["acquisition_threshold_passed"] for point in phase_a_points]
    if pass_flags[-2:] != [True, True]:
        raise RuntimeError("Phase A does not end in two qualifying evaluations")
    if any(left and right for left, right in zip(pass_flags[:-2], pass_flags[1:-1])):
        raise RuntimeError("Phase A continued after an earlier qualifying pair")
    if phase_a["acquisition_complete_checkpoint"] != phase_a_checkpoint_records[-1]:
        raise RuntimeError("Phase A confirming checkpoint differs")
    acquisition_path = _checkpoint_custody(
        phase_a["acquisition_complete_checkpoint"], RIVAL2_REWARD_V2_VERSION
    )

    # Explicit transition custody and exact state preservation.
    if transition["verdict"] != "PASS_GREEN" or not all(transition["checks"].values()):
        raise RuntimeError("reward transition proof failed")
    if transition["source_checkpoint"] != phase_a["acquisition_complete_checkpoint"]:
        raise RuntimeError("reward transition source differs from acquisition checkpoint")
    post_transition_path = _checkpoint_custody(
        transition["post_transition_checkpoint"], RIVAL2_REWARD_VERSION
    )
    source_payload = torch.load(acquisition_path, map_location="cpu", weights_only=False)
    transition_payload = torch.load(post_transition_path, map_location="cpu", weights_only=False)
    if (
        transition_payload["curriculum_transition"] != transition["transition"]
        or transition_payload["curriculum_transition"]["parent_checkpoint_sha256"]
        != phase_a["acquisition_complete_checkpoint"]["sha256"]
        or source_payload["reward_version"] != RIVAL2_REWARD_V2_VERSION
        or transition_payload["reward_version"] != RIVAL2_REWARD_VERSION
    ):
        raise RuntimeError("reward transition checkpoint metadata differs")
    del source_payload, transition_payload

    # Phase B: exact offsets and exact 239-update / 2,004,877,312-sample boundary.
    if (
        phase_b["execution_status"] != "COMPLETE"
        or phase_b["reward_version"] != RIVAL2_REWARD_VERSION
        or phase_b["start_iteration"] != phase_a["final_iteration"]
        or phase_b["start_agent_decision_samples"] != phase_a["final_agent_decision_samples"]
        or phase_b["additional_updates"] != overnight.PHASE_B_UPDATES
        or phase_b["additional_agent_decision_samples"]
        != overnight.PHASE_B_ADDITIONAL_SAMPLES
        or phase_b["final_iteration"]
        != phase_b["start_iteration"] + overnight.PHASE_B_UPDATES
        or phase_b["final_agent_decision_samples"]
        != phase_b["start_agent_decision_samples"] + overnight.PHASE_B_ADDITIONAL_SAMPLES
    ):
        raise RuntimeError("Phase B exact boundary differs")
    if len(phase_b["checkpoints"]) != len(overnight.PHASE_B_EVALUATION_OFFSETS):
        raise RuntimeError("Phase B checkpoint count differs")
    phase_b_points: list[dict[str, Any]] = []
    for checkpoint, offset in zip(
        phase_b["checkpoints"], overnight.PHASE_B_EVALUATION_OFFSETS, strict=True
    ):
        _checkpoint_custody(checkpoint, RIVAL2_REWARD_VERSION)
        if checkpoint["iteration"] != phase_b["start_iteration"] + offset:
            raise RuntimeError(f"Phase B +{offset} checkpoint iteration differs")
        evaluation = _load_evaluation(work_dir, checkpoint["label"])
        if (
            evaluation["iteration"] != checkpoint["iteration"]
            or evaluation["agent_decision_samples"] != checkpoint["agent_decision_samples"]
            or evaluation["reward_version"] != RIVAL2_REWARD_VERSION
        ):
            raise RuntimeError(f"Phase B +{offset} evaluation differs")
        phase_b_points.append(_evaluation_point(evaluation, update_offset=offset))
    if (
        phase_b["base_reward_2b_checkpoint"] != phase_b["checkpoints"][-1]
        or phase_b["base_reward_2b_checkpoint_reload"]["verdict"] != "PASS_GREEN"
    ):
        raise RuntimeError("Phase B durable checkpoint differs")
    phase_b_path = _checkpoint_custody(
        phase_b["base_reward_2b_checkpoint"], RIVAL2_REWARD_VERSION
    )

    # Phase C: real monotonic timing, first completed update crossing each threshold.
    if (
        phase_c["execution_status"] != "COMPLETE"
        or phase_c["reward_version"] != RIVAL2_REWARD_VERSION
        or phase_c["start_iteration"] != phase_b["final_iteration"]
        or phase_c["start_agent_decision_samples"] != phase_b["final_agent_decision_samples"]
        or len(phase_c["checkpoints"]) != len(overnight.PHASE_C_THRESHOLDS)
        or phase_c["final_checkpoint_reload"]["verdict"] != "PASS_GREEN"
    ):
        raise RuntimeError("Phase C summary differs")
    phase_c_training_points = [
        point for point in training_points if point["phase"] == "C_REWARD_V1_SIX_HOURS"
    ]
    phase_c_by_iteration = {point["iteration"]: point for point in phase_c_training_points}
    phase_c_points: list[dict[str, Any]] = []
    for checkpoint, (hour_label, threshold_seconds) in zip(
        phase_c["checkpoints"], overnight.PHASE_C_THRESHOLDS, strict=True
    ):
        _checkpoint_custody(checkpoint, RIVAL2_REWARD_VERSION)
        if (
            checkpoint["label"] != f"phase_c_{hour_label}"
            or checkpoint["phase_c_threshold_seconds"] != threshold_seconds
            or checkpoint["phase_c_elapsed_seconds_at_update_completion"] < threshold_seconds
        ):
            raise RuntimeError(f"Phase C {hour_label} checkpoint timing differs")
        update_point = phase_c_by_iteration[checkpoint["iteration"]]
        if update_point["phase_c_elapsed_seconds_at_update_completion"] != checkpoint[
            "phase_c_elapsed_seconds_at_update_completion"
        ]:
            raise RuntimeError(f"Phase C {hour_label} trigger evidence differs")
        previous_point = phase_c_by_iteration[checkpoint["iteration"] - 1]
        if previous_point["phase_c_elapsed_seconds_at_update_completion"] >= threshold_seconds:
            raise RuntimeError(f"Phase C {hour_label} was not the first crossing update")
        evaluation = _load_evaluation(work_dir, checkpoint["label"])
        if (
            evaluation["iteration"] != checkpoint["iteration"]
            or evaluation["agent_decision_samples"] != checkpoint["agent_decision_samples"]
            or evaluation["reward_version"] != RIVAL2_REWARD_VERSION
            or evaluation["phase_c_threshold_seconds"] != threshold_seconds
        ):
            raise RuntimeError(f"Phase C {hour_label} evaluation differs")
        phase_c_points.append(_evaluation_point(evaluation))
    if (
        phase_c["final_checkpoint"] != phase_c["checkpoints"][-1]
        or phase_c["stop_elapsed_seconds_at_update_completion"]
        < overnight.PHASE_C_DURATION_SECONDS
        or phase_c_training_points[-2]["phase_c_elapsed_seconds_at_update_completion"]
        >= overnight.PHASE_C_DURATION_SECONDS
    ):
        raise RuntimeError("Phase C final stop boundary differs")
    final_path = _checkpoint_custody(phase_c["final_checkpoint"], RIVAL2_REWARD_VERSION)

    expected_evaluation_files = {
        f"evaluation_{record['label']}.json" for record in phase_a_checkpoint_records
    } | {
        f"evaluation_{record['label']}.json" for record in phase_b["checkpoints"]
    } | {
        f"evaluation_{record['label']}.json" for record in phase_c["checkpoints"]
    }
    actual_evaluation_files = {path.name for path in work_dir.glob("evaluation_*.json")}
    if actual_evaluation_files != expected_evaluation_files:
        raise RuntimeError("overnight evaluation set contains missing or extra runs")

    phase_a_training = [
        point for point in training_points if point["phase"] == "A_REWARD_V2_ACQUISITION"
    ]
    phase_b_training = [
        point for point in training_points if point["phase"] == "B_REWARD_V1_2B"
    ]
    if (
        len(phase_a_training) != phase_a["continuation_updates"]
        or len(phase_b_training) != overnight.PHASE_B_UPDATES
        or any(point["reward_version"] != RIVAL2_REWARD_V2_VERSION for point in phase_a_training)
        or any(point["reward_version"] != RIVAL2_REWARD_VERSION for point in phase_b_training)
        or any(point["reward_version"] != RIVAL2_REWARD_VERSION for point in phase_c_training_points)
    ):
        raise RuntimeError("overnight phase/reward training partition differs")

    ppo = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "overall": _phase_ppo_summary(training_points),
        "phase_a": _phase_ppo_summary(phase_a_training),
        "phase_b": _phase_ppo_summary(phase_b_training),
        "phase_c": _phase_ppo_summary(phase_c_training_points),
    }
    update_ledger = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "compact per-update PPO/integrity ledger; full local rollout details are not published",
        "updates": _update_ledger(training_points),
    }
    phase_a_curve = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "reward_version": RIVAL2_REWARD_V2_VERSION,
        "threshold": overnight.PHASE_A_THRESHOLD,
        "required_consecutive": overnight.PHASE_A_REQUIRED_CONSECUTIVE,
        "first_qualifying_pair": [phase_a_points[-2]["label"], phase_a_points[-1]["label"]],
        "points": phase_a_points,
    }
    phase_b_curve = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "reward_version": RIVAL2_REWARD_VERSION,
        "start_iteration": phase_b["start_iteration"],
        "start_agent_decision_samples": phase_b["start_agent_decision_samples"],
        "points": phase_b_points,
    }
    phase_c_curve = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "reward_version": RIVAL2_REWARD_VERSION,
        "timer_started_utc": phase_c["timer_started_utc"],
        "start_iteration": phase_c["start_iteration"],
        "start_agent_decision_samples": phase_c["start_agent_decision_samples"],
        "points": phase_c_points,
    }

    final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
    phase_b_payload = torch.load(phase_b_path, map_location="cpu", weights_only=False)
    acquisition_payload = torch.load(acquisition_path, map_location="cpu", weights_only=False)
    expected_transition = transition["transition"]
    if (
        acquisition_payload["reward_version"] != RIVAL2_REWARD_V2_VERSION
        or "curriculum_transition" in acquisition_payload
        or phase_b_payload["reward_version"] != RIVAL2_REWARD_VERSION
        or phase_b_payload["curriculum_transition"] != expected_transition
        or final_payload["reward_version"] != RIVAL2_REWARD_VERSION
        or final_payload["contract_hashes"] != contract_hashes_for_reward(RIVAL2_REWARD_VERSION)
        or final_payload["curriculum_transition"] != expected_transition
        or final_payload["iteration"] != run["final_iteration"]
        or final_payload["policy_version"] != run["final_policy_version"]
        or final_payload["total_agent_samples"] != run["final_agent_decision_samples"]
    ):
        raise RuntimeError("published-boundary checkpoint payload differs")
    del acquisition_payload, phase_b_payload, final_payload

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    published_records: dict[str, dict[str, Any]] = {}
    for artifact_label, source, filename, reward_version in (
        (
            "acquisition_complete",
            acquisition_path,
            "rival2_overnight_acquisition_complete_resume.pt",
            RIVAL2_REWARD_V2_VERSION,
        ),
        (
            "phase_b_2b_base_reward",
            phase_b_path,
            "rival2_overnight_2b_base_reward_resume.pt",
            RIVAL2_REWARD_VERSION,
        ),
        (
            "final_6h",
            final_path,
            "rival2_overnight_final_6h_resume.pt",
            RIVAL2_REWARD_VERSION,
        ),
    ):
        destination = args.checkpoints_dir / filename
        shutil.copyfile(source, destination)
        source_sha = _sha256_file(source)
        if _sha256_file(destination) != source_sha:
            raise RuntimeError(f"published {artifact_label} checkpoint differs")
        published_records[artifact_label] = {
            "path": destination.as_posix(),
            "sha256": source_sha,
            "size_bytes": destination.stat().st_size,
            "format": "RIVAL2_CHECKPOINT_V1",
            "reward_version": reward_version,
            "artifact_kind": "full_resumable_training_checkpoint",
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "campaign": "Rival 2.0 uninterrupted overnight acquisition-to-base-reward curriculum",
        "execution_status": "COMPLETE",
        "resume_checkpoint_sha256": resume_authority["checkpoint_sha256"],
        "start_iteration": overnight.START_ITERATION,
        "start_agent_decision_samples": overnight.START_SAMPLES,
        "phase_a_final_iteration": phase_a["final_iteration"],
        "phase_a_final_agent_decision_samples": phase_a["final_agent_decision_samples"],
        "phase_a_continuation_updates": phase_a["continuation_updates"],
        "phase_a_evaluation_count": len(phase_a_points),
        "phase_a_confirming_no_touch_fractions": phase_a["confirming_no_touch_fractions"],
        "reward_transition_verdict": transition["verdict"],
        "phase_b_additional_updates": phase_b["additional_updates"],
        "phase_b_additional_agent_decision_samples": phase_b[
            "additional_agent_decision_samples"
        ],
        "phase_c_additional_updates": phase_c["additional_updates"],
        "phase_c_additional_agent_decision_samples": phase_c[
            "additional_agent_decision_samples"
        ],
        "phase_c_stop_elapsed_seconds_at_update_completion": phase_c[
            "stop_elapsed_seconds_at_update_completion"
        ],
        "final_iteration": run["final_iteration"],
        "final_policy_version": run["final_policy_version"],
        "final_agent_decision_samples": run["final_agent_decision_samples"],
        "total_continuation_updates": len(training_points),
        "total_additional_agent_decision_samples": run[
            "total_additional_agent_decision_samples"
        ],
        "curriculum_wall_seconds_including_all_evaluations": run[
            "curriculum_wall_seconds_including_all_evaluations"
        ],
        "update_integrity_pass_count": len(training_points),
        "evaluation_pass_count": len(phase_a_points) + len(phase_b_points) + len(phase_c_points),
        "published_acquisition_checkpoint": published_records["acquisition_complete"],
        "published_phase_b_checkpoint": published_records["phase_b_2b_base_reward"],
        "published_final_checkpoint": published_records["final_6h"],
        "final_checkpoint_reload_verdict": run["final_checkpoint_reload"]["verdict"],
        "final_reward_version": run["final_reward_version"],
        "final_historical_policy_versions": run["final_historical_policy_versions"],
        "preflight_regression_lint_ceremony_run": False,
        "viewer_built": False,
        "v06_begun": False,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.results_dir / "config.json", configuration)
    _write_json(args.results_dir / "resume_authority.json", resume_authority)
    _write_json(args.results_dir / "loaded_state.json", loaded_state)
    _write_json(args.results_dir / "phase_a_curve.json", phase_a_curve)
    _write_json(args.results_dir / "reward_transition.json", transition)
    _write_json(args.results_dir / "phase_b_curve.json", phase_b_curve)
    _write_json(args.results_dir / "phase_c_curve.json", phase_c_curve)
    _write_json(args.results_dir / "ppo_stability.json", ppo)
    _write_json(args.results_dir / "update_ledger.json", update_ledger)
    _write_json(args.results_dir / "checkpoints.json", published_records)
    _write_json(args.results_dir / "run_summary.json", run)
    _write_json(args.results_dir / "summary.json", summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _report_text(
            summary=summary,
            phase_a_curve=phase_a_curve,
            transition=transition,
            phase_b_curve=phase_b_curve,
            phase_c_curve=phase_c_curve,
            ppo=ppo,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
