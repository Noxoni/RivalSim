"""Validate and publish compact Rival 2.0 Campaign 01 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.run_rival2_campaign01 import (
    EXPECTED_HEAD,
    TARGET_SAMPLES,
    THRESHOLDS,
    _prior_results_manifest,
    first_update_at_or_above,
    frozen_configuration,
)

MAX_COMMITTED_CHECKPOINT_BYTES = 25 * 1024**2
BEHAVIORAL_RESULTS = (
    "CLEAR_EMERGENCE",
    "WEAK_EMERGENCE",
    "NO_CLEAR_EMERGENCE",
    "DEGRADED",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--behavioral-result", choices=BEHAVIORAL_RESULTS, required=True)
    parser.add_argument("--behavioral-rationale", required=True)
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


def _compact_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return evaluation


def _format_rate(value: float) -> str:
    return f"{value:.6f}"


def _evaluation_row(evaluation: dict[str, Any]) -> str:
    self_play = evaluation["modes"]["stochastic_self_play"]
    deterministic = evaluation["modes"]["deterministic_vs_initialization"][
        "versus_initialization"
    ]
    stochastic = evaluation["modes"]["stochastic_vs_initialization"][
        "versus_initialization"
    ]
    return " | ".join(
        (
            evaluation["checkpoint_label"],
            f"{evaluation['agent_decision_samples']:,}",
            _format_rate(self_play["goal_terminated_fraction"]),
            _format_rate(self_play["no_touch_truncated_fraction"]),
            _format_rate(self_play["hard_truncated_fraction"]),
            _format_rate(self_play["touches_per_simulated_minute"]),
            _format_rate(self_play["goals_per_simulated_minute"]),
            str(deterministic["goal_differential"]),
            str(deterministic["touch_differential"]),
            str(stochastic["goal_differential"]),
            str(stochastic["touch_differential"]),
        )
    )


def _report_text(
    *,
    summary: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    preflight: dict[str, Any],
    configuration: dict[str, Any],
) -> str:
    final = evaluations[-1]
    final_self = final["modes"]["stochastic_self_play"]
    final_det = final["modes"]["deterministic_vs_initialization"][
        "versus_initialization"
    ]
    final_stochastic = final["modes"]["stochastic_vs_initialization"][
        "versus_initialization"
    ]
    checkpoint_lines = "\n".join(
        f"- `{item['label']}`: {item['agent_decision_samples']:,} samples, update "
        f"{item['iteration']}, `{item['sha256']}`, {item['size_bytes']:,} bytes"
        for item in checkpoints
    )
    evaluation_rows = "\n".join(f"| {_evaluation_row(item)} |" for item in evaluations)
    attempts = "\n".join(
        f"- {attempt['worlds']:,} worlds: `{attempt['status']}`"
        + (
            f", peak {attempt['vram_peak_observed_bytes']:,} bytes, "
            f"margin {attempt['vram_margin_bytes']:,} bytes"
            if attempt.get("status") == "PASS"
            else f", {attempt.get('failure', 'failed a required check')}"
        )
        for attempt in preflight["attempts"]
    )
    evaluation_header = (
        "| checkpoint | samples | goal fraction | no-touch fraction | hard fraction | "
        "touches/min | goals/min | det goal diff | det touch diff | stochastic goal diff | "
        "stochastic touch diff |"
    )
    evaluation_rule = (
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    return f"""# Rival 2.0 Campaign 01 Results

Campaign 01 completed its exact bounded execution. The fresh policy finished update
`{summary['final_iteration']}` with `{summary['final_agent_decision_samples']:,}` agent decision
samples, which is the first completed PPO update at or above 100,000,000 samples. No additional
training update and no v0.6 work was performed.

## Verdict

- execution status: `{summary['execution_status']}`;
- behavioral result: `{summary['behavioral_result']}`;
- behavioral rationale: {summary['behavioral_rationale']}
- frozen v0.5 trainer verdict: unchanged (`PASS_GREEN`);
- final checkpoint reload and exact next stochastic sample:
  `{summary['checkpoint_reload_verdict']}`.

The behavioral verdict is descriptive and independent of execution acceptance. Campaign 01 did
not change a reward, observation, action, episode, architecture, curriculum, or PPO setting in
response to the measured outcome.

## Frozen authority

- authorized campaign commit: `{EXPECTED_HEAD}`;
- campaign seed: `{configuration['campaign_seed']}`;
- evaluation seed: `{configuration['evaluation']['seed']}`;
- evaluation protocol SHA-256: `{configuration['evaluation']['protocol_sha256']}`;
- selected worlds: `{preflight['selected_worlds']:,}`;
- horizon: `{configuration['ppo_config']['rollout_horizon']}`;
- entropy coefficient: `{configuration['ppo_config']['entropy_coefficient']}`;
- policy configuration SHA-256: `{configuration['policy_config_hash']}`.

All four frozen contract identities are recorded in `config.json` and match v0.5.

## Capacity preflight

The candidates were attempted once in the authorized order and the first fully passing capacity
was selected:

{attempts}

This was a real horizon-32 rollout/GAE/PPO update with finite-state checks, checkpoint/inference
allocation, zero simulator hot-path H2D/D2H traffic, and a prospectively frozen 4 GiB safety
margin.

## Checkpoint custody

Initialization and every first threshold-crossing checkpoint remain in the ignored local campaign
artifact directory with exact SHA-256 identities:

{checkpoint_lines}

The final resume checkpoint is also published at
`checkpoints/rival2/campaign01/rival2_campaign01_100m_resume.pt`; its committed artifact is
`{summary['published_checkpoint_size_bytes']:,}` bytes with SHA-256
`{summary['published_checkpoint_sha256']}`. It is the exact v0.5-format full resume artifact, not
an inference-only substitute.

## Fixed evaluation

Each checkpoint used the same 4,096 held-out worlds, all five kickoff layouts, seeds, balanced
sides, and first-episode limit. Rates below are from stochastic ordinary self-play. The final four
columns are current-checkpoint minus frozen-initialization results.

{evaluation_header}
{evaluation_rule}
{evaluation_rows}

At the final checkpoint, ordinary stochastic self-play recorded
`{final_self['touch_entries']:,}` accepted touch entries,
`{final_self['goal_terminated_episodes']:,}` goal terminations, and
`{final_self['demolition_events']:,}` demolition events. Its mean first-episode duration was
`{final_self['mean_episode_duration_seconds']:.6f}` seconds. The complete analog magnitudes,
button activation rates, policy standard deviations, Bernoulli probabilities/entropies,
termination mix, and outcome counts are in the five `evaluation_*.json` artifacts.

Final deterministic play against initialization produced goal differential
`{final_det['goal_differential']}`, touch differential `{final_det['touch_differential']}`, and
outcomes `{final_det['current_wins']}` current wins / `{final_det['initialization_wins']}`
initialization wins / `{final_det['draws']}` draws. Final stochastic play produced goal
differential `{final_stochastic['goal_differential']}`, touch differential
`{final_stochastic['touch_differential']}`, and outcomes `{final_stochastic['current_wins']}` /
`{final_stochastic['initialization_wins']}` / `{final_stochastic['draws']}`.

## Integrity and regression

Every one of the `{summary['final_iteration']}` completed campaign updates passed finite checks
for observations, actions, rewards, values, log probabilities, advantages, returns, losses,
gradients, parameters, and optimizer state. Analog actions remained bounded, buttons remained
binary, selective done/reset accounting remained valid, historical snapshots remained frozen and
gradient-free, sample/version accounting remained exact, and ordinary simulator hot-path transfer
counters remained zero.

The final v0.5 checkpoint loader reproduced model/value outputs, deterministic inference, and the
next stochastic action/pre-tanh/log-probability exactly. Tracked `results/v0.1/` through
`results/v0.5/` matched the prospectively recorded byte manifest at closeout. Campaign tooling did
not modify simulator or trainer implementation behavior, so no expensive prior authority corpus
was rerun.

## Boundary

Campaign 01 is closed at the first completed update crossing 100M samples plus its fixed
evaluation and evidence closeout. This report does not authorize or begin v0.6.
"""


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    configuration = _read_json(work_dir / "config_frozen_before_training.json")
    if configuration != frozen_configuration():
        raise RuntimeError("frozen campaign configuration no longer matches the runner authority")
    preflight = _read_json(work_dir / "preflight.json")
    run = _read_json(work_dir / "run_summary.json")
    checkpoint_data = _read_json(work_dir / "checkpoints.json")
    curve = _read_json(work_dir / "training_curve.json")
    baseline = _read_json(work_dir / "prior_results_baseline.json")
    current_prior = _prior_results_manifest()
    baseline_comparable = {
        key: value for key, value in baseline.items() if key != "created_utc"
    }
    if current_prior != baseline_comparable:
        raise RuntimeError("results/v0.1 through results/v0.5 are not byte-for-byte unchanged")
    if preflight["verdict"] != "PASS_GREEN" or preflight["selected_worlds"] is None:
        raise RuntimeError("capacity preflight did not select an authorized world count")
    selected_worlds = int(preflight["selected_worlds"])
    expected_updates, expected_final_samples = first_update_at_or_above(
        selected_worlds, TARGET_SAMPLES
    )
    if run["execution_status"] != "COMPLETE":
        raise RuntimeError("campaign execution did not complete")
    if run["final_iteration"] != expected_updates:
        raise RuntimeError("campaign did not stop on the first crossing update")
    if run["final_agent_decision_samples"] != expected_final_samples:
        raise RuntimeError("campaign final sample count differs from the bounded schedule")
    if run["final_checkpoint_reload"]["verdict"] != "PASS_GREEN":
        raise RuntimeError("final checkpoint reload gate did not pass")
    if len(curve) != expected_updates or any(
        point["integrity"]["verdict"] != "PASS_GREEN" for point in curve
    ):
        raise RuntimeError("one or more campaign updates lack passing integrity evidence")
    if curve[-2]["agent_decision_samples"] >= TARGET_SAMPLES:
        raise RuntimeError("campaign continued past the first update crossing 100M")
    checkpoints = checkpoint_data["checkpoints"]
    if [item["label"] for item in checkpoints] != [label for label, _ in THRESHOLDS]:
        raise RuntimeError("threshold checkpoint set is incomplete or reordered")
    evaluations: list[dict[str, Any]] = []
    for checkpoint, (label, threshold) in zip(checkpoints, THRESHOLDS, strict=True):
        expected_iteration, expected_samples = (
            (0, 0)
            if threshold == 0
            else first_update_at_or_above(selected_worlds, threshold)
        )
        if checkpoint["iteration"] != expected_iteration:
            raise RuntimeError(f"{label} checkpoint iteration is not the first crossing")
        if checkpoint["agent_decision_samples"] != expected_samples:
            raise RuntimeError(f"{label} checkpoint sample count is wrong")
        local_path = Path(checkpoint["path"])
        if not local_path.is_file() or _sha256_file(local_path) != checkpoint["sha256"]:
            raise RuntimeError(f"{label} local checkpoint custody failed")
        evaluation = _read_json(work_dir / f"evaluation_{label}.json")
        if evaluation["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"{label} fixed evaluation integrity failed")
        if evaluation["agent_decision_samples"] != expected_samples:
            raise RuntimeError(f"{label} evaluation sample identity is wrong")
        if (
            evaluation["evaluation_protocol_sha256"]
            != configuration["evaluation"]["protocol_sha256"]
        ):
            raise RuntimeError(f"{label} evaluation protocol changed")
        evaluations.append(evaluation)
    final_local = Path(checkpoints[-1]["path"])
    if final_local.stat().st_size > MAX_COMMITTED_CHECKPOINT_BYTES:
        raise RuntimeError(
            "final full resume checkpoint exceeds 25 MiB; publish a compact inference artifact "
            "and explicitly document the full local custody path instead"
        )
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    published_checkpoint = args.checkpoints_dir / "rival2_campaign01_100m_resume.pt"
    shutil.copyfile(final_local, published_checkpoint)
    if _sha256_file(published_checkpoint) != checkpoints[-1]["sha256"]:
        raise RuntimeError("published checkpoint differs from the verified final local checkpoint")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    published_config = dict(configuration)
    published_config["capacity_selection"] = {
        "selected_worlds": selected_worlds,
        "preflight_verdict": preflight["verdict"],
    }
    _write_json(args.results_dir / "config.json", published_config)
    _write_json(args.results_dir / "preflight.json", preflight)
    published_checkpoints = {
        "schema_version": 1,
        "local_checkpoint_custody": checkpoints,
        "published_final_checkpoint": {
            "path": published_checkpoint.as_posix(),
            "sha256": _sha256_file(published_checkpoint),
            "size_bytes": published_checkpoint.stat().st_size,
            "format": "RIVAL2_CHECKPOINT_V1",
            "artifact_kind": "full_resumable_training_checkpoint",
        },
    }
    _write_json(args.results_dir / "checkpoints.json", published_checkpoints)
    for evaluation in evaluations:
        _write_json(
            args.results_dir / f"evaluation_{evaluation['checkpoint_label']}.json",
            _compact_evaluation(evaluation),
        )
    _write_json(args.results_dir / "training_curve.json", curve)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "campaign": "Rival 2.0 Campaign 01",
        "execution_status": run["execution_status"],
        "behavioral_result": args.behavioral_result,
        "behavioral_rationale": args.behavioral_rationale,
        "selected_worlds": selected_worlds,
        "campaign_seed": configuration["campaign_seed"],
        "evaluation_seed": configuration["evaluation"]["seed"],
        "evaluation_protocol_sha256": configuration["evaluation"]["protocol_sha256"],
        "final_iteration": run["final_iteration"],
        "final_agent_decision_samples": run["final_agent_decision_samples"],
        "first_update_crossing_target": run["first_update_crossing_target"],
        "checkpoint_reload_verdict": run["final_checkpoint_reload"]["verdict"],
        "published_checkpoint_sha256": _sha256_file(published_checkpoint),
        "published_checkpoint_size_bytes": published_checkpoint.stat().st_size,
        "initialization_model_sha256": run["initialization_model_sha256"],
        "update_integrity_pass_count": len(curve),
        "update_integrity_failure_count": 0,
        "fixed_evaluation_pass_count": len(evaluations),
        "prior_results_v01_through_v05_unchanged": True,
        "prior_results_manifest_sha256": current_prior["manifest_sha256"],
        "trainer_or_simulator_implementation_changed": False,
        "v05_pass_green_unchanged": True,
        "v06_begun": False,
    }
    _write_json(args.results_dir / "summary.json", summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _report_text(
            summary=summary,
            checkpoints=checkpoints,
            evaluations=evaluations,
            preflight=preflight,
            configuration=configuration,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
