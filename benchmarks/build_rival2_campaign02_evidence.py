"""Validate and publish the controlled Rival 2.0 Campaign 02 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import benchmarks.run_rival2_campaign01 as campaign01
import benchmarks.run_rival2_campaign02 as campaign02

MAX_COMMITTED_CHECKPOINT_BYTES = 25 * 1024**2
BEHAVIORAL_RESULTS = ("IMPROVED", "DEGRADED", "INCONCLUSIVE")


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


def primary_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    self_play = evaluation["modes"]["stochastic_self_play"]
    versus = evaluation["modes"]["stochastic_vs_initialization"][
        "versus_initialization"
    ]
    return {
        "ordinary_self_play_touches_per_simulated_minute": float(
            self_play["touches_per_simulated_minute"]
        ),
        "stochastic_vs_initialization_goal_differential": float(
            versus["goal_differential"]
        ),
        "stochastic_vs_initialization_touch_differential": float(
            versus["touch_differential"]
        ),
    }


def classify_behavior(
    initialization: dict[str, float],
    campaign01_final: dict[str, float],
    campaign02_final: dict[str, float],
) -> dict[str, Any]:
    improved_vs_initialization = {
        name: campaign02_final[name] > initialization[name] for name in initialization
    }
    worse_vs_initialization = {
        name: campaign02_final[name] < initialization[name] for name in initialization
    }
    not_worse_than_campaign01 = {
        name: campaign02_final[name] >= campaign01_final[name] for name in initialization
    }
    improved_count = sum(improved_vs_initialization.values())
    worse_count = sum(worse_vs_initialization.values())
    if improved_count >= 2 and all(not_worse_than_campaign01.values()):
        result = "IMPROVED"
    elif worse_count >= 2:
        result = "DEGRADED"
    else:
        result = "INCONCLUSIVE"
    return {
        "behavioral_result": result,
        "primary_metrics": {
            name: {
                "initialization": initialization[name],
                "campaign01_final": campaign01_final[name],
                "campaign02_final": campaign02_final[name],
                "campaign02_minus_initialization": campaign02_final[name]
                - initialization[name],
                "campaign02_minus_campaign01": campaign02_final[name]
                - campaign01_final[name],
                "improved_vs_initialization": improved_vs_initialization[name],
                "worse_vs_initialization": worse_vs_initialization[name],
                "not_worse_than_campaign01": not_worse_than_campaign01[name],
            }
            for name in initialization
        },
        "improved_vs_initialization_count": improved_count,
        "worse_vs_initialization_count": worse_count,
        "not_worse_than_campaign01_on_all_three": all(
            not_worse_than_campaign01.values()
        ),
        "prospective_rule": (
            "IMPROVED iff at least two primary final metrics improve vs initialization and none "
            "is worse than Campaign 01; DEGRADED iff at least two are worse vs initialization; "
            "otherwise INCONCLUSIVE"
        ),
    }


def _evaluation_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    self_play = evaluation["modes"]["stochastic_self_play"]
    deterministic = evaluation["modes"]["deterministic_vs_initialization"]
    stochastic = evaluation["modes"]["stochastic_vs_initialization"]
    return {
        "ordinary_stochastic_self_play": {
            name: self_play[name]
            for name in (
                "goal_terminated_fraction",
                "no_touch_truncated_fraction",
                "hard_truncated_fraction",
                "touches_per_simulated_minute",
                "goals_per_simulated_minute",
                "demolitions_per_simulated_minute",
                "mean_episode_duration_seconds",
                "mean_absolute_analog_action",
                "button_activation_rate",
                "mean_analog_policy_std",
                "mean_button_probability",
                "mean_button_entropy",
            )
        },
        "deterministic_vs_initialization": {
            "metrics": {
                name: deterministic[name]
                for name in (
                    "goal_terminated_fraction",
                    "no_touch_truncated_fraction",
                    "touches_per_simulated_minute",
                    "goals_per_simulated_minute",
                    "mean_episode_duration_seconds",
                    "mean_absolute_analog_action",
                    "button_activation_rate",
                    "mean_analog_policy_std",
                    "mean_button_probability",
                    "mean_button_entropy",
                )
            },
            "outcome": deterministic["versus_initialization"],
        },
        "stochastic_vs_initialization": {
            "metrics": {
                name: stochastic[name]
                for name in (
                    "goal_terminated_fraction",
                    "no_touch_truncated_fraction",
                    "touches_per_simulated_minute",
                    "goals_per_simulated_minute",
                    "mean_episode_duration_seconds",
                    "mean_absolute_analog_action",
                    "button_activation_rate",
                    "mean_analog_policy_std",
                    "mean_button_probability",
                    "mean_button_entropy",
                )
            },
            "outcome": stochastic["versus_initialization"],
        },
    }


def _numeric_delta(left: Any, right: Any) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        return {
            key: _numeric_delta(left[key], right[key])
            for key in sorted(set(left) & set(right))
        }
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return float(right) - float(left)
    return None


def build_optimizer_diagnosis(
    campaign01_curve: list[dict[str, Any]], campaign02_curve: list[dict[str, Any]]
) -> dict[str, Any]:
    c02_metrics = [point["integrity"]["metrics"] for point in campaign02_curve]
    max_kl_index = max(range(len(c02_metrics)), key=lambda index: c02_metrics[index]["approx_kl"])
    max_clip_index = max(
        range(len(c02_metrics)), key=lambda index: c02_metrics[index]["clip_fraction"]
    )
    flagged = [
        {
            "iteration": point["iteration"],
            "approx_kl": point["integrity"]["metrics"]["approx_kl"],
            "clip_fraction": point["integrity"]["metrics"]["clip_fraction"],
            "approximate_kl_ge_0_1": point["optimizer_diagnosis"]["flags"][
                "approximate_kl_ge_0_1"
            ],
            "clip_fraction_ge_0_3": point["optimizer_diagnosis"]["flags"][
                "clip_fraction_ge_0_3"
            ],
        }
        for point in campaign02_curve
        if any(point["optimizer_diagnosis"]["flags"].values())
    ]
    per_update = []
    for c01_point, c02_point in zip(campaign01_curve, campaign02_curve, strict=True):
        c01_metrics = c01_point["integrity"]["metrics"]
        c02_values = c02_point["integrity"]["metrics"]
        per_update.append(
            {
                "iteration": c02_point["iteration"],
                "campaign01": c01_metrics,
                "campaign02": c02_values,
                "campaign02_minus_campaign01": {
                    name: c02_values[name] - c01_metrics[name]
                    for name in c01_metrics
                },
                "campaign02_policy_distribution": c02_point[
                    "policy_distribution_on_frozen_observations"
                ],
                "campaign02_optimizer_diagnosis": c02_point["optimizer_diagnosis"],
            }
        )
    initial_std = campaign02_curve[0]["policy_distribution_on_frozen_observations"][
        "mean_analog_policy_std"
    ]
    final_std = campaign02_curve[-1]["policy_distribution_on_frozen_observations"][
        "mean_analog_policy_std"
    ]
    initial_std_mean = sum(initial_std.values()) / len(initial_std)
    final_std_mean = sum(final_std.values()) / len(final_std)
    return {
        "entropy_coefficient": 0.0,
        "entropy_optimization_contribution": 0.0,
        "diagnostic_entropy_logged": True,
        "maximum_approximate_kl": {
            "value": c02_metrics[max_kl_index]["approx_kl"],
            "iteration": campaign02_curve[max_kl_index]["iteration"],
        },
        "maximum_clip_fraction": {
            "value": c02_metrics[max_clip_index]["clip_fraction"],
            "iteration": campaign02_curve[max_clip_index]["iteration"],
        },
        "flagged_updates": flagged,
        "campaign01_update4_style_instability_recurred": bool(flagged),
        "instability_recurrence_definition": (
            "any Campaign 02 update with approximate KL >=0.1 or clip fraction >=0.3"
        ),
        "analog_standard_deviation": {
            "first_post_update_mean": initial_std_mean,
            "final_mean": final_std_mean,
            "configured_ceiling": math.exp(1.0),
            "final_fraction_of_ceiling": final_std_mean / math.exp(1.0),
            "trended_toward_ceiling": final_std_mean > initial_std_mean
            and final_std_mean >= 0.9 * math.exp(1.0),
        },
        "per_update_campaign_comparison": per_update,
    }


def _checkpoint_rows(checkpoints: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- `{item['label']}`: update {item['iteration']}, "
        f"{item['agent_decision_samples']:,} samples, `{item['sha256']}`, "
        f"{item['size_bytes']:,} bytes"
        for item in checkpoints
    )


def _comparison_table(comparison: dict[str, Any]) -> str:
    rows = []
    for label, value in comparison["evaluation_checkpoints"].items():
        c01 = value["campaign01"]["ordinary_stochastic_self_play"]
        c02 = value["campaign02"]["ordinary_stochastic_self_play"]
        c01_vs = value["campaign01"]["stochastic_vs_initialization"]["outcome"]
        c02_vs = value["campaign02"]["stochastic_vs_initialization"]["outcome"]
        c01_std = sum(c01["mean_analog_policy_std"].values()) / 5.0
        c02_std = sum(c02["mean_analog_policy_std"].values()) / 5.0
        rows.append(
            "| "
            + " | ".join(
                (
                    label,
                    f"{c01['touches_per_simulated_minute']:.6f}",
                    f"{c02['touches_per_simulated_minute']:.6f}",
                    str(c01_vs["goal_differential"]),
                    str(c02_vs["goal_differential"]),
                    str(c01_vs["touch_differential"]),
                    str(c02_vs["touch_differential"]),
                    f"{c01_std:.6f}",
                    f"{c02_std:.6f}",
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _optimizer_table(diagnosis: dict[str, Any]) -> str:
    rows = []
    for point in diagnosis["per_update_campaign_comparison"]:
        c01 = point["campaign01"]
        c02 = point["campaign02"]
        distribution = point["campaign02_policy_distribution"]["mean_analog_policy_std"]
        mean_std = sum(distribution.values()) / len(distribution)
        rows.append(
            f"| {point['iteration']} | {c01['entropy']:.6f} | {c02['entropy']:.6f} | "
            f"{c01['approx_kl']:.6f} | {c02['approx_kl']:.6f} | "
            f"{c01['clip_fraction']:.6f} | {c02['clip_fraction']:.6f} | "
            f"{mean_std:.6f} |"
        )
    return "\n".join(rows)


def _report_text(
    *,
    summary: dict[str, Any],
    configuration: dict[str, Any],
    initialization_control: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    comparison: dict[str, Any],
    optimizer: dict[str, Any],
) -> str:
    classification = comparison["behavioral_classification"]
    final_c02 = comparison["evaluation_checkpoints"]["100m"]["campaign02"]
    final_self = final_c02["ordinary_stochastic_self_play"]
    final_det = final_c02["deterministic_vs_initialization"]["outcome"]
    final_stochastic = final_c02["stochastic_vs_initialization"]["outcome"]
    flagged = optimizer["flagged_updates"]
    flagged_text = (
        ", ".join(
            f"update {item['iteration']} (KL {item['approx_kl']:.6f}, "
            f"clip {item['clip_fraction']:.6f})"
            for item in flagged
        )
        if flagged
        else "none"
    )
    primary_lines = "\n".join(
        f"- `{name}`: init `{value['initialization']:.6f}`, C01 final "
        f"`{value['campaign01_final']:.6f}`, C02 final `{value['campaign02_final']:.6f}`; "
        f"C02-init `{value['campaign02_minus_initialization']:+.6f}`, "
        f"C02-C01 `{value['campaign02_minus_campaign01']:+.6f}`"
        for name, value in classification["primary_metrics"].items()
    )
    comparison_header = (
        "| checkpoint | C01 touches/min | C02 touches/min | C01 stochastic goal diff | "
        "C02 stochastic goal diff | C01 stochastic touch diff | C02 stochastic touch diff | "
        "C01 std | C02 std |"
    )
    optimizer_header = (
        "| update | C01 entropy | C02 diagnostic entropy | C01 KL | C02 KL | C01 clip | "
        "C02 clip | C02 frozen-observation std |"
    )
    return f"""# Rival 2.0 Campaign 02 Results

Campaign 02 completed the controlled entropy-off rerun. It stopped at update
`{summary['final_iteration']}` with `{summary['final_agent_decision_samples']:,}` agent decision
samples, the first completed update crossing 100M. No later update and no v0.6 work ran.

## Independent verdicts

- execution status: `{summary['execution_status']}`;
- behavioral result: `{summary['behavioral_result']}`;
- initialization control: `{initialization_control['verdict']}`;
- final checkpoint continuation: `{summary['checkpoint_reload_verdict']}`;
- frozen v0.5 trainer: unchanged (`PASS_GREEN`).

The behavioral classification is the prospectively fixed Campaign 02 rule; it is independent of
execution correctness and was not adjusted after results were observed.

## Controlled-variable proof

- authorized Campaign 02 commit: `{campaign02.AUTHORIZED_HEAD}`;
- Campaign 01 closeout parent: `{campaign02.CAMPAIGN01_CLOSEOUT}`;
- campaign seed: `{configuration['campaign_seed']}`;
- evaluation seed: `{configuration['evaluation']['seed']}`;
- worlds / horizon: `{configuration['worlds']:,}` /
  `{configuration['ppo_config']['rollout_horizon']}`;
- Campaign 01 entropy coefficient: `0.01`;
- Campaign 02 entropy coefficient: `0.0`;
- all other PPO fields: exact match;
- model/contract/self-play/seed/evaluation fields: exact match;
- initialization model SHA-256: `{initialization_control['actual_model_sha256']}`;
- Campaign 01 initialization SHA-256 match: exact;
- initialization evaluation semantic metrics: exact;
- evaluation protocol SHA-256: `{configuration['evaluation']['protocol_sha256']}`.

Only non-semantic initialization-evaluation timestamps and wall-clock durations differ. The
diagnostic entropy metric remained logged, but its optimization contribution was exactly zero in
every Campaign 02 update.

## Checkpoint custody

{_checkpoint_rows(checkpoints)}

The final full resumable v0.5-format artifact is committed at
`checkpoints/rival2/campaign02/rival2_campaign02_100m_resume.pt`. Its exact size is
`{summary['published_checkpoint_size_bytes']:,}` bytes and SHA-256 is
`{summary['published_checkpoint_sha256']}`.

## Direct Campaign 01 versus Campaign 02 evaluation

All values use the same 4,096-world, five-layout, first-episode protocol. `std` is the mean of the
five ordinary-self-play analog policy standard deviations.

{comparison_header}
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{_comparison_table(comparison)}

Final Campaign 02 ordinary self-play had `{final_self['touches_per_simulated_minute']:.6f}`
touches/minute, `{final_self['goals_per_simulated_minute']:.6f}` goals/minute,
`{final_self['no_touch_truncated_fraction']:.6f}` no-touch truncation fraction, and
`{final_self['mean_episode_duration_seconds']:.6f}` seconds mean episode duration.

Final deterministic play against initialization produced goal differential
`{final_det['goal_differential']}`, touch differential `{final_det['touch_differential']}`, and
outcomes `{final_det['current_wins']}` current wins / `{final_det['initialization_wins']}`
initialization wins / `{final_det['draws']}` draws. Final stochastic play produced goal
differential `{final_stochastic['goal_differential']}`, touch differential
`{final_stochastic['touch_differential']}`, and outcomes `{final_stochastic['current_wins']}` /
`{final_stochastic['initialization_wins']}` / `{final_stochastic['draws']}`.

## Prospective behavioral classification

{primary_lines}

The rule counted `{classification['improved_vs_initialization_count']}` improvements and
`{classification['worse_vs_initialization_count']}` regressions relative to initialization.
Campaign 02 was not worse than Campaign 01 on all three primary metrics:
`{str(classification['not_worse_than_campaign01_on_all_three']).lower()}`. The resulting
classification is **`{classification['behavioral_result']}`**.

## Optimizer diagnosis

{optimizer_header}
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{_optimizer_table(optimizer)}

- maximum Campaign 02 approximate KL: `{optimizer['maximum_approximate_kl']['value']:.6f}` at
  update `{optimizer['maximum_approximate_kl']['iteration']}`;
- maximum Campaign 02 clip fraction: `{optimizer['maximum_clip_fraction']['value']:.6f}` at
  update `{optimizer['maximum_clip_fraction']['iteration']}`;
- threshold-flagged updates: {flagged_text};
- Campaign 01 update-4 style instability recurred:
  `{str(optimizer['campaign01_update4_style_instability_recurred']).lower()}`;
- final representative analog standard deviation:
  `{optimizer['analog_standard_deviation']['final_mean']:.6f}` / `exp(1)` ceiling
  `{optimizer['analog_standard_deviation']['configured_ceiling']:.6f}`;
- standard deviation trended toward the ceiling:
  `{str(optimizer['analog_standard_deviation']['trended_toward_ceiling']).lower()}`.

The repository artifacts contain every policy/value loss, diagnostic entropy, total loss,
approximate KL, clip fraction, pre/post-clip gradient norm, fixed-observation standard deviation,
integrity check, transfer count, and policy/sample age for all 12 updates.

## Integrity, immutability, and boundary

All 12 updates passed finite rollout/loss/gradient/parameter/optimizer checks, action bounds,
binary buttons, done/reset accounting, frozen historical-opponent custody, version/sample age, and
zero hot-path H2D/D2H traffic. The final checkpoint reproduced deterministic outputs and the next
stochastic action/pre-tanh/log-probability exactly under the entropy-zero config identity.

Tracked v0.1-v0.5 results, Campaign 01 artifacts, and the frozen v0.5 training implementation
matched their prospectively frozen byte manifests at closeout. No inherited expensive simulator
authority rerun was required because no shared simulator/trainer implementation file changed.

Campaign 02 is closed. This result does not authorize a reward change, another hyperparameter
trial, curriculum work, or v0.6 RocketSim/RLBot transfer.
"""


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    configuration = _read_json(work_dir / "config_frozen_before_training.json")
    if configuration != campaign02.frozen_configuration():
        raise RuntimeError("frozen Campaign 02 configuration differs from runner authority")
    immutable = _read_json(work_dir / "immutable_parent_manifest.json")
    current_immutable = campaign02.immutable_parent_manifest()
    immutable_comparable = {
        key: value for key, value in immutable.items() if key != "created_utc"
    }
    if immutable_comparable != current_immutable:
        raise RuntimeError("a frozen v0.5 or Campaign 01 artifact changed")
    initialization = _read_json(work_dir / "initialization_control.json")
    run = _read_json(work_dir / "run_summary.json")
    curve = _read_json(work_dir / "training_curve.json")
    checkpoint_data = _read_json(work_dir / "checkpoints.json")
    if initialization["verdict"] != "PASS_GREEN":
        raise RuntimeError("initialization control failed")
    if run["execution_status"] != "COMPLETE":
        raise RuntimeError("Campaign 02 execution did not complete")
    expected_updates, expected_samples = campaign01.first_update_at_or_above(
        campaign02.CAMPAIGN02_WORLDS, campaign01.TARGET_SAMPLES
    )
    if (
        run["final_iteration"] != expected_updates
        or run["final_agent_decision_samples"] != expected_samples
    ):
        raise RuntimeError("Campaign 02 did not stop on the first 100M-crossing update")
    if run["final_checkpoint_reload"]["verdict"] != "PASS_GREEN":
        raise RuntimeError("Campaign 02 final checkpoint continuation failed")
    if len(curve) != expected_updates or any(
        point["integrity"]["verdict"] != "PASS_GREEN" for point in curve
    ):
        raise RuntimeError("one or more Campaign 02 updates lack green integrity evidence")
    if curve[-2]["agent_decision_samples"] >= campaign01.TARGET_SAMPLES:
        raise RuntimeError("Campaign 02 continued beyond its authorized stop")
    if any(
        point["optimizer_diagnosis"]["entropy_optimization_contribution"] != 0.0
        or point["optimizer_diagnosis"]["entropy_coefficient"] != 0.0
        for point in curve
    ):
        raise RuntimeError("Campaign 02 entropy entered the optimization objective")
    checkpoints = checkpoint_data["checkpoints"]
    labels = [label for label, _ in campaign01.THRESHOLDS]
    if [item["label"] for item in checkpoints] != labels:
        raise RuntimeError("Campaign 02 threshold checkpoint set is incomplete")
    evaluations: dict[str, dict[str, Any]] = {}
    for checkpoint, (label, threshold) in zip(
        checkpoints, campaign01.THRESHOLDS, strict=True
    ):
        expected_iteration, threshold_samples = (
            (0, 0)
            if threshold == 0
            else campaign01.first_update_at_or_above(
                campaign02.CAMPAIGN02_WORLDS, threshold
            )
        )
        if checkpoint["iteration"] != expected_iteration:
            raise RuntimeError(f"{label} checkpoint is not the first threshold crossing")
        if checkpoint["agent_decision_samples"] != threshold_samples:
            raise RuntimeError(f"{label} checkpoint sample count differs")
        path = Path(checkpoint["path"])
        if not path.is_file() or _sha256_file(path) != checkpoint["sha256"]:
            raise RuntimeError(f"{label} local checkpoint custody failed")
        evaluation = _read_json(work_dir / f"evaluation_{label}.json")
        if evaluation["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"{label} evaluation integrity failed")
        if (
            evaluation["evaluation_protocol_sha256"]
            != configuration["evaluation"]["protocol_sha256"]
        ):
            raise RuntimeError(f"{label} evaluation protocol differs from Campaign 01")
        evaluations[label] = evaluation
    campaign01_evaluations = {
        label: _read_json(Path(f"results/rival2/campaign01/evaluation_{label}.json"))
        for label in labels
    }
    evaluation_comparison = {
        label: {
            "agent_decision_samples": evaluations[label]["agent_decision_samples"],
            "campaign01": _evaluation_metrics(campaign01_evaluations[label]),
            "campaign02": _evaluation_metrics(evaluations[label]),
            "campaign02_minus_campaign01": _numeric_delta(
                _evaluation_metrics(campaign01_evaluations[label]),
                _evaluation_metrics(evaluations[label]),
            ),
        }
        for label in labels
    }
    classification = classify_behavior(
        primary_metrics(evaluations["000m"]),
        primary_metrics(campaign01_evaluations["100m"]),
        primary_metrics(evaluations["100m"]),
    )
    if classification["behavioral_result"] not in BEHAVIORAL_RESULTS:
        raise RuntimeError("Campaign 02 classification is invalid")
    campaign01_curve = _read_json(Path("results/rival2/campaign01/training_curve.json"))
    optimizer = build_optimizer_diagnosis(campaign01_curve, curve)
    comparison = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "controlled_variable": configuration["controlled_variable"],
        "initialization_model_sha256_exact": initialization["model_sha256_exact"],
        "initialization_evaluation_semantic_metrics_exact": initialization[
            "evaluation_control"
        ]["semantic_metrics_exact"],
        "evaluation_checkpoints": evaluation_comparison,
        "behavioral_classification": classification,
    }
    final_local = Path(checkpoints[-1]["path"])
    if final_local.stat().st_size > MAX_COMMITTED_CHECKPOINT_BYTES:
        raise RuntimeError("final full Campaign 02 resume checkpoint exceeds 25 MiB")
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    published_checkpoint = args.checkpoints_dir / "rival2_campaign02_100m_resume.pt"
    shutil.copyfile(final_local, published_checkpoint)
    if _sha256_file(published_checkpoint) != checkpoints[-1]["sha256"]:
        raise RuntimeError("published Campaign 02 checkpoint differs from local custody")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.results_dir / "config.json", configuration)
    _write_json(args.results_dir / "initialization_control.json", initialization)
    _write_json(
        args.results_dir / "checkpoints.json",
        {
            "schema_version": 1,
            "local_checkpoint_custody": checkpoints,
            "published_final_checkpoint": {
                "path": published_checkpoint.as_posix(),
                "sha256": _sha256_file(published_checkpoint),
                "size_bytes": published_checkpoint.stat().st_size,
                "format": "RIVAL2_CHECKPOINT_V1",
                "artifact_kind": "full_resumable_training_checkpoint",
            },
        },
    )
    for label, evaluation in evaluations.items():
        _write_json(args.results_dir / f"evaluation_{label}.json", evaluation)
    _write_json(args.results_dir / "training_curve.json", curve)
    _write_json(args.results_dir / "comparison_campaign01.json", comparison)
    _write_json(args.results_dir / "optimizer_diagnosis.json", optimizer)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "campaign": "Rival 2.0 Campaign 02 entropy-off controlled rerun",
        "execution_status": run["execution_status"],
        "behavioral_result": classification["behavioral_result"],
        "selected_worlds": campaign02.CAMPAIGN02_WORLDS,
        "campaign_seed": configuration["campaign_seed"],
        "evaluation_seed": configuration["evaluation"]["seed"],
        "entropy_coefficient": 0.0,
        "entropy_optimization_contribution": 0.0,
        "only_learning_change_from_campaign01": "entropy_coefficient: 0.01 -> 0.0",
        "initialization_model_sha256": run["initialization_model_sha256"],
        "initialization_control_verdict": initialization["verdict"],
        "initialization_evaluation_exact": initialization["evaluation_control"][
            "semantic_metrics_exact"
        ],
        "final_iteration": run["final_iteration"],
        "final_agent_decision_samples": run["final_agent_decision_samples"],
        "checkpoint_reload_verdict": run["final_checkpoint_reload"]["verdict"],
        "published_checkpoint_sha256": _sha256_file(published_checkpoint),
        "published_checkpoint_size_bytes": published_checkpoint.stat().st_size,
        "update_integrity_pass_count": len(curve),
        "fixed_evaluation_pass_count": len(evaluations),
        "optimizer_maximum_approximate_kl": optimizer["maximum_approximate_kl"],
        "optimizer_maximum_clip_fraction": optimizer["maximum_clip_fraction"],
        "optimizer_flagged_update_count": len(optimizer["flagged_updates"]),
        "prior_v05_and_campaign01_artifacts_unchanged": True,
        "immutable_parent_manifest_sha256": current_immutable["manifest_sha256"],
        "trainer_or_simulator_implementation_changed": False,
        "v05_pass_green_unchanged": True,
        "v06_begun": False,
    }
    _write_json(args.results_dir / "summary.json", summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        _report_text(
            summary=summary,
            configuration=configuration,
            initialization_control=initialization,
            checkpoints=checkpoints,
            comparison=comparison,
            optimizer=optimizer,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
