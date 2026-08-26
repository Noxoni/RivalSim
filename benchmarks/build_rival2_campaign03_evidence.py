"""Validate and publish compact Rival 2.0 Campaign 03 closeout evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

import benchmarks.run_rival2_campaign03 as campaign03

MAX_COMMITTED_CHECKPOINT_BYTES = 25 * 1024**2
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


def _metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    result = evaluation["result"]
    return {
        "touches_per_simulated_minute": float(result["touches_per_simulated_minute"]),
        "goals_per_simulated_minute": float(result["goals_per_simulated_minute"]),
        "goal_terminated_fraction": float(result["goal_terminated_fraction"]),
        "no_touch_truncated_fraction": float(result["no_touch_truncated_fraction"]),
        "mean_episode_duration_seconds": float(result["mean_episode_duration_seconds"]),
    }


def _campaign02_metrics() -> dict[str, float]:
    evaluation = _read_json(Path("results/rival2/campaign02/evaluation_100m.json"))
    result = evaluation["modes"]["stochastic_self_play"]
    return {
        "touches_per_simulated_minute": float(result["touches_per_simulated_minute"]),
        "goals_per_simulated_minute": float(result["goals_per_simulated_minute"]),
        "goal_terminated_fraction": float(result["goal_terminated_fraction"]),
        "no_touch_truncated_fraction": float(result["no_touch_truncated_fraction"]),
        "mean_episode_duration_seconds": float(result["mean_episode_duration_seconds"]),
    }


def _optimizer_summary(curve: list[dict[str, Any]]) -> dict[str, Any]:
    def extremum(name: str, *, maximum: bool = True) -> dict[str, Any]:
        selected = (max if maximum else min)(curve, key=lambda point: point["ppo_stability"][name])
        return {
            "value": selected["ppo_stability"][name],
            "iteration": selected["iteration"],
        }

    return {
        "entropy_coefficient": 0.0,
        "entropy_optimization_contribution": 0.0,
        "maximum_approximate_kl": extremum("approximate_kl"),
        "maximum_clip_fraction": extremum("clip_fraction"),
        "maximum_gradient_norm": extremum("gradient_norm"),
        "maximum_post_clip_gradient_norm": extremum("post_clip_gradient_norm"),
        "mean_iteration_agent_decisions_per_second": sum(
            point["agent_decisions_per_second"] for point in curve
        )
        / len(curve),
        "all_updates_integrity_green": all(
            point["integrity"]["verdict"] == "PASS_GREEN" for point in curve
        ),
    }


def _report_text(
    *,
    summary: dict[str, Any],
    comparison: dict[str, Any],
    optimizer: dict[str, Any],
    smoke: dict[str, Any],
) -> str:
    c02 = comparison["campaign02_final"]
    c03 = comparison["campaign03_final"]
    delta = comparison["campaign03_minus_campaign02"]
    post_reset_contaminated = smoke["reset_case"]["post_reset_contaminated_delta"][0][0]
    checkpoint_size = summary["published_checkpoint_size_bytes"]

    def row(label: str, key: str) -> str:
        return (
            f"| {label} | {c02[key]:.6f} | {c03[key]:.6f} | "
            f"{delta[key]:+.6f} |"
        )

    table_rows = "\n".join(
        (
            row("Touches / simulated minute", "touches_per_simulated_minute"),
            row("Goals / simulated minute", "goals_per_simulated_minute"),
            row("Goal-terminated fraction", "goal_terminated_fraction"),
            row("No-touch truncation fraction", "no_touch_truncated_fraction"),
            row("Mean episode duration, seconds", "mean_episode_duration_seconds"),
        )
    )
    return f"""# Rival 2.0 Campaign 03 Results

Campaign 03 is complete. It implemented `RIVAL2_REWARD_V2`, ran only the authorized
targeted GPU reward smoke, immediately trained from scratch at 131,072 worlds / horizon 32,
and stopped at update {summary['final_iteration']} / {summary['final_agent_decision_samples']:,}
agent decision samples, the first completed update crossing 100M. It did not run capacity
preflight, initialization evaluation, inherited parity/regression gates, a world-count sweep,
or intermediate held-out evaluations.

## Reward V2 custody

`RIVAL2_REWARD_V1` remains unchanged. Reward V2 adds exactly one per-agent term:

`(car_ball_distance_before - car_ball_distance_after) / 4096.0`

The true 3D distances are reconstructed on CUDA from frozen observation relative-position
fields at decision start and the final pre-reset transition state. Reward V2's deterministic
content SHA-256 is `{summary['reward_v2_contract_hash']}`.

The one targeted smoke was `{smoke['verdict']}`. Its closing, opening, and unchanged cases
produced `{smoke['synthetic_approach'][0][0]:.9f}`, `{smoke['synthetic_approach'][1][0]:.9f}`,
and `{smoke['synthetic_approach'][2][0]:.9f}`. The forced reset case's integrated approach
matched the pre-reset value `{smoke['reset_case']['pre_reset_expected'][0][0]:.12f}`; the
post-reset-contaminated alternative was `{post_reset_contaminated:.12f}`.
All smoke tensors were finite and device-resident.

## Bounded training execution

The Campaign 02 PPO/model/observation/action/episode/self-play baseline was otherwise unchanged,
including entropy coefficient `0.0`. Checkpoints were saved at the first updates crossing 25M,
50M, and 100M. All {summary['update_integrity_pass_count']} updates passed numerical and
device-transfer integrity, and the final checkpoint passed exact reload/continuation checks.

- maximum approximate KL: `{optimizer['maximum_approximate_kl']['value']:.6f}` at update
  `{optimizer['maximum_approximate_kl']['iteration']}`;
- maximum clip fraction: `{optimizer['maximum_clip_fraction']['value']:.6f}` at update
  `{optimizer['maximum_clip_fraction']['iteration']}`;
- maximum gradient norm: `{optimizer['maximum_gradient_norm']['value']:.6f}`;
- mean training throughput: `{optimizer['mean_iteration_agent_decisions_per_second']:,.2f}`
  agent decisions/s;
- training wall time: `{summary['campaign_training_wall_seconds']:.3f}` seconds.

The final resumable checkpoint is
`checkpoints/rival2/campaign03/rival2_campaign03_100m_resume.pt`, with SHA-256
`{summary['published_checkpoint_sha256']}` and size `{checkpoint_size:,}` bytes.

## Single final evaluation

Exactly one 4,096-world ordinary stochastic self-play evaluation was run after the final
checkpoint, using seed `920260826`.

| Metric | Campaign 02 final | Campaign 03 final | C03 - C02 |
|---|---:|---:|---:|
{table_rows}

The dense approach term coincided with a large increase in touch and goal frequency and a
5.3467 percentage-point reduction in no-touch truncation on this single frozen final protocol.
This is the requested direct Campaign 02 comparison, not a claim of external Rocket League
competence or v0.6 transfer readiness.

Campaign 03 is closed. No curriculum, extra reward term, action mask, hyperparameter tuning,
v0.6 RocketSim/RLBot work, or post-boundary training was begun.
"""


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    configuration = _read_json(work_dir / "config_frozen_before_training.json")
    smoke = _read_json(work_dir / "reward_smoke.json")
    run = _read_json(work_dir / "run_summary.json")
    curve = _read_json(work_dir / "training_curve.json")
    checkpoint_data = _read_json(work_dir / "checkpoints.json")
    evaluation = _read_json(work_dir / "final_evaluation.json")

    if configuration != campaign03.frozen_configuration():
        raise RuntimeError("Campaign 03 frozen configuration differs from runner authority")
    if smoke["verdict"] != "PASS_GREEN" or not all(smoke["checks"].values()):
        raise RuntimeError("Campaign 03 targeted reward smoke did not pass")
    if run["execution_status"] != "COMPLETE":
        raise RuntimeError("Campaign 03 training did not complete")
    expected_iteration, expected_samples = campaign03.campaign01.first_update_at_or_above(
        campaign03.CAMPAIGN03_WORLDS, campaign03.TARGET_SAMPLES
    )
    if (
        run["final_iteration"] != expected_iteration
        or run["final_agent_decision_samples"] != expected_samples
    ):
        raise RuntimeError("Campaign 03 did not stop at the first 100M-crossing update")
    if curve[-2]["agent_decision_samples"] >= campaign03.TARGET_SAMPLES:
        raise RuntimeError("Campaign 03 continued beyond the authorized stop")
    if len(curve) != expected_iteration or any(
        point["integrity"]["verdict"] != "PASS_GREEN" for point in curve
    ):
        raise RuntimeError("one or more Campaign 03 updates lack green integrity evidence")
    if run["final_checkpoint_reload"]["verdict"] != "PASS_GREEN":
        raise RuntimeError("Campaign 03 final checkpoint reload failed")
    if evaluation["verdict"] != "PASS_GREEN" or evaluation["evaluation_worlds"] != 4096:
        raise RuntimeError("Campaign 03 single final evaluation failed")
    evaluation_files = list(work_dir.glob("*evaluation*.json"))
    if [path.name for path in evaluation_files] != ["final_evaluation.json"]:
        raise RuntimeError("Campaign 03 contains an unauthorized extra evaluation")

    checkpoints = checkpoint_data["checkpoints"]
    expected = (("025m", 3, 25165824), ("050m", 6, 50331648), ("100m", 12, 100663296))
    if len(checkpoints) != len(expected):
        raise RuntimeError("Campaign 03 checkpoint count differs")
    for checkpoint, (label, iteration, samples) in zip(checkpoints, expected, strict=True):
        path = Path(checkpoint["path"])
        if (
            checkpoint["label"] != label
            or checkpoint["iteration"] != iteration
            or checkpoint["agent_decision_samples"] != samples
            or not path.is_file()
            or _sha256_file(path) != checkpoint["sha256"]
        ):
            raise RuntimeError(f"Campaign 03 {label} checkpoint custody failed")

    final_local = Path(checkpoints[-1]["path"])
    if final_local.stat().st_size > MAX_COMMITTED_CHECKPOINT_BYTES:
        raise RuntimeError("Campaign 03 final checkpoint exceeds the publication bound")
    payload = torch.load(final_local, map_location="cpu", weights_only=False)
    if (
        payload["reward_version"] != campaign03.RIVAL2_REWARD_V2_VERSION
        or payload["contract_hashes"] != configuration["active_contract_hashes"]
        or payload["ppo_config"]["entropy_coefficient"] != 0.0
    ):
        raise RuntimeError("Campaign 03 final checkpoint identity is incompatible")

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    published = args.checkpoints_dir / "rival2_campaign03_100m_resume.pt"
    shutil.copyfile(final_local, published)
    if _sha256_file(published) != checkpoints[-1]["sha256"]:
        raise RuntimeError("published Campaign 03 checkpoint differs from local custody")

    campaign02_metrics = _campaign02_metrics()
    campaign03_metrics = _metrics(evaluation)
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "mode": "ordinary stochastic self-play",
            "worlds": 4096,
            "seed": 920260826,
            "checkpoint": "first completed update crossing 100M samples",
        },
        "campaign02_final": campaign02_metrics,
        "campaign03_final": campaign03_metrics,
        "campaign03_minus_campaign02": {
            name: campaign03_metrics[name] - campaign02_metrics[name]
            for name in campaign02_metrics
        },
        "no_post_hoc_success_threshold": True,
    }
    optimizer = _optimizer_summary(curve)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.results_dir / "config.json", configuration)
    _write_json(args.results_dir / "reward_smoke.json", smoke)
    _write_json(args.results_dir / "training_curve.json", curve)
    _write_json(args.results_dir / "final_evaluation.json", evaluation)
    _write_json(args.results_dir / "comparison_campaign02.json", comparison)
    _write_json(args.results_dir / "ppo_stability.json", optimizer)
    _write_json(
        args.results_dir / "checkpoints.json",
        {
            "schema_version": SCHEMA_VERSION,
            "local_checkpoint_custody": checkpoints,
            "published_final_checkpoint": {
                "path": published.as_posix(),
                "sha256": _sha256_file(published),
                "size_bytes": published.stat().st_size,
                "format": "RIVAL2_CHECKPOINT_V1",
                "reward_version": campaign03.RIVAL2_REWARD_V2_VERSION,
                "artifact_kind": "full_resumable_training_checkpoint",
            },
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "campaign": "Rival 2.0 Campaign 03 direct reward-density training",
        "execution_status": "COMPLETE",
        "selected_worlds": campaign03.CAMPAIGN03_WORLDS,
        "campaign_seed": campaign03.CAMPAIGN03_SEED,
        "evaluation_seed": campaign03.EVALUATION_SEED,
        "reward_version": campaign03.RIVAL2_REWARD_V2_VERSION,
        "reward_v2_contract_hash": campaign03.REWARD_V2_CONTRACT_HASH,
        "reward_smoke_verdict": smoke["verdict"],
        "final_iteration": run["final_iteration"],
        "final_agent_decision_samples": run["final_agent_decision_samples"],
        "campaign_training_wall_seconds": run["campaign_training_wall_seconds"],
        "checkpoint_reload_verdict": run["final_checkpoint_reload"]["verdict"],
        "published_checkpoint_sha256": _sha256_file(published),
        "published_checkpoint_size_bytes": published.stat().st_size,
        "update_integrity_pass_count": len(curve),
        "held_out_evaluation_count": 1,
        "final_evaluation_verdict": evaluation["verdict"],
        "campaign02_final": campaign02_metrics,
        "campaign03_final": campaign03_metrics,
        "campaign03_minus_campaign02": comparison["campaign03_minus_campaign02"],
        "maximum_approximate_kl": optimizer["maximum_approximate_kl"],
        "maximum_clip_fraction": optimizer["maximum_clip_fraction"],
        "old_preflight_regression_evaluation_ceremony_run": False,
        "v06_begun": False,
    }
    _write_json(args.results_dir / "summary.json", summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _report_text(summary=summary, comparison=comparison, optimizer=optimizer, smoke=smoke),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
