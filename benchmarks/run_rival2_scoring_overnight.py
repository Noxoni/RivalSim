"""Continue unchanged Rival 2.0 Scoring V1 training until 07:00 ET."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmarks.run_rival2_campaign01 as campaign01
import benchmarks.run_rival2_full_match_curriculum as full_match
import benchmarks.run_rival2_scoring_v1 as scoring
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    RIVAL2_FULL_MATCH_EPISODE_VERSION,
    RIVAL2_REWARD_SCORING_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.rival2_policy import Rival2PolicyConfig
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer

SCHEMA_VERSION = 1
PHASE = "SCORING_V1_OVERNIGHT"
AUTHORIZED_DEADLINE_LOCAL = "2026-08-27T07:00:00-04:00"
EXPECTED_START_ITERATION = 240
EXPECTED_START_SAMPLES = EXPECTED_START_ITERATION * scoring.ROLLOUT_AGENT_SAMPLES
SNAPSHOT_CADENCE = 60
DEFAULT_RESULTS = Path("results/rival2/scoring_overnight_20260827")
DEFAULT_PUBLISHED_CHECKPOINT = Path(
    "checkpoints/rival2/scoring_v1/rival2_scoring_overnight_20260827_resume.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the update-240 Scoring V1 checkpoint, train unchanged until "
            "the first completed PPO update crossing 07:00 ET, then evaluate once."
        )
    )
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-sha256", required=True)
    parser.add_argument("--parent-progress", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--deadline-local",
        default=AUTHORIZED_DEADLINE_LOCAL,
        help="Fixed ISO-8601 local deadline authorized for this run.",
    )
    return parser.parse_args()


def parse_authorized_deadline(value: str) -> datetime:
    deadline = datetime.fromisoformat(value)
    if deadline.tzinfo is None:
        raise ValueError("deadline-local must include an explicit UTC offset")
    if deadline.isoformat() != AUTHORIZED_DEADLINE_LOCAL:
        raise ValueError(
            f"authorized deadline is fixed to {AUTHORIZED_DEADLINE_LOCAL}"
        )
    return deadline


def is_snapshot_iteration(iteration: int) -> bool:
    return iteration > EXPECTED_START_ITERATION and iteration % SNAPSHOT_CADENCE == 0


def crosses_deadline(
    previous_completion: datetime,
    current_completion: datetime,
    deadline: datetime,
) -> bool:
    return previous_completion < deadline <= current_completion


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    scoring._write_json(path, value)


def _append_jsonl(path: Path, value: object) -> None:
    scoring._append_jsonl(path, value)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _configuration(
    payload: dict[str, Any],
    resume_checkpoint: Path,
    resume_sha256: str,
    deadline: datetime,
) -> dict[str, Any]:
    policy = Rival2PolicyConfig(**payload["policy_config"])
    ppo = Rival2PPOConfig(**payload["ppo_config"])
    self_play = Rival2SelfPlayConfig(**payload["self_play_config"])
    return {
        "schema_version": SCHEMA_VERSION,
        "steering_authority": (
            "User-authorized continuous continuation of the unchanged Scoring V1 "
            "curriculum until 07:00 AM Eastern on 2026-08-27."
        ),
        "resume_checkpoint": resume_checkpoint.resolve().as_posix(),
        "resume_checkpoint_sha256": resume_sha256.upper(),
        "start_iteration": int(payload["iteration"]),
        "start_policy_version": int(payload["policy_version"]),
        "start_agent_decision_samples": int(payload["total_agent_samples"]),
        "deadline_local": deadline.isoformat(),
        "stop_rule": (
            "Stop after the first completed PPO update whose completion timestamp "
            "is at or after the deadline; evaluate only after training stops."
        ),
        "worlds": scoring.WORLDS,
        "campaign_seed": scoring.CAMPAIGN_SEED,
        "rollout_agent_decision_samples": scoring.ROLLOUT_AGENT_SAMPLES,
        "snapshot_cadence_updates": SNAPSHOT_CADENCE,
        "reward_version": RIVAL2_REWARD_SCORING_V1_VERSION,
        "episode_version": RIVAL2_FULL_MATCH_EPISODE_VERSION,
        "contract_hashes": contract_hashes_for_reward(
            RIVAL2_REWARD_SCORING_V1_VERSION,
            RIVAL2_FULL_MATCH_EPISODE_VERSION,
        ),
        "policy_config": asdict(policy),
        "policy_config_hash": policy.content_hash,
        "ppo_config": asdict(ppo),
        "ppo_config_hash": ppo.content_hash,
        "self_play_config": asdict(self_play),
        "continuation_semantics": {
            "model": "exact",
            "optimizer": "exact",
            "torch_cpu_rng": "exact",
            "torch_cuda_rng": "exact",
            "policy_rng": "exact",
            "opponent_rng": "exact",
            "counters": "exact",
            "opponent_assignments": "exact",
            "historical_pool": "exact",
            "reward": "unchanged",
            "episode": "unchanged",
            "simulator": "unchanged",
            "world_match_state": "fresh at process-resume boundary; not checkpointed",
        },
        "evaluation": {
            "during_training": False,
            "after_deadline": True,
            "self_play_worlds": scoring.EVALUATION_WORLDS,
            "frozen_acquisition_worlds": scoring.REFERENCE_WORLDS,
            "stochastic": True,
            "complete_full_matches": True,
        },
    }


def _launch_gate(
    configuration: dict[str, Any],
    payload: dict[str, Any],
    resume_checkpoint: Path,
    parent_progress_path: Path,
    baseline_evaluation_path: Path,
    deadline: datetime,
) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    actual_sha = scoring._sha256(resume_checkpoint)
    parent_progress = _read_json(parent_progress_path)
    parent_checkpoint = parent_progress["checkpoints"][-1]
    parent_evaluation = parent_progress["evaluations"][-1]
    parent_comparison = parent_progress["frozen_comparisons"][-1]
    expected_contracts = contract_hashes_for_reward(
        RIVAL2_REWARD_SCORING_V1_VERSION,
        RIVAL2_FULL_MATCH_EPISODE_VERSION,
    )
    now = datetime.now(deadline.tzinfo)
    checks = {
        "head_pushed_to_origin_main": head == origin,
        "tracked_worktree_clean": subprocess.run(
            ["git", "diff", "--quiet"]
        ).returncode
        == 0,
        "index_clean": subprocess.run(
            ["git", "diff", "--cached", "--quiet"]
        ).returncode
        == 0,
        "resume_checkpoint_present": resume_checkpoint.is_file(),
        "resume_checkpoint_sha256_exact": actual_sha
        == configuration["resume_checkpoint_sha256"],
        "resume_format_exact": payload.get("format") == "RIVAL2_CHECKPOINT_V1",
        "resume_iteration_exact": int(payload["iteration"])
        == EXPECTED_START_ITERATION,
        "resume_policy_version_exact": int(payload["policy_version"])
        == EXPECTED_START_ITERATION,
        "resume_samples_exact": int(payload["total_agent_samples"])
        == EXPECTED_START_SAMPLES,
        "resume_reward_exact": payload.get("reward_version")
        == RIVAL2_REWARD_SCORING_V1_VERSION,
        "resume_episode_exact": payload.get("episode_version")
        == RIVAL2_FULL_MATCH_EPISODE_VERSION,
        "resume_contracts_exact": payload.get("contract_hashes")
        == expected_contracts,
        "parent_checkpoint_hash_exact": parent_checkpoint["sha256"] == actual_sha,
        "parent_checkpoint_iteration_exact": parent_checkpoint["iteration"]
        == EXPECTED_START_ITERATION,
        "parent_checkpoint_audit_green": parent_checkpoint["audit"]["verdict"]
        == "PASS_GREEN",
        "parent_evaluation_complete": parent_evaluation["checkpoint_label"]
        == "plus_120",
        "parent_comparison_complete": parent_comparison["checkpoint_label"]
        == "plus_120",
        "baseline_evaluation_present": baseline_evaluation_path.is_file(),
        "deadline_exact": deadline.isoformat() == AUTHORIZED_DEADLINE_LOCAL,
        "deadline_in_future_at_launch": now < deadline,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "head": head,
        "origin_main": origin,
        "launch_local": now.isoformat(),
        "deadline_local": deadline.isoformat(),
        "resume_checkpoint_sha256": actual_sha,
        "parent_progress": parent_progress_path.resolve().as_posix(),
        "baseline_evaluation": baseline_evaluation_path.resolve().as_posix(),
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"scoring overnight launch gate failed: {checks}")
    return result


def _loaded_state_audit(
    source_payload: dict[str, Any], trainer: Rival2Trainer
) -> dict[str, Any]:
    loaded_payload = trainer.checkpoint_payload()
    preserved_fields = (
        "model",
        "optimizer",
        "policy_config",
        "ppo_config",
        "self_play_config",
        "contract_hashes",
        "reward_version",
        "episode_version",
        "policy_config_hash",
        "ppo_config_hash",
        "policy_version",
        "iteration",
        "total_agent_samples",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "policy_generator_state",
        "opponent_generator_state",
        "opponent_assignment",
        "historical_opponents",
        "curriculum_transition",
    )
    checks = {
        f"{name}_exact": scoring._nested_exact(
            source_payload.get(name), loaded_payload.get(name)
        )
        for name in preserved_fields
    }
    checks.update(
        {
            "fresh_blue_score_zero": bool(
                torch.all(trainer.env.full_match_views["blue_score"] == 0).item()
            ),
            "fresh_orange_score_zero": bool(
                torch.all(trainer.env.full_match_views["orange_score"] == 0).item()
            ),
            "reward_unchanged": trainer.env.reward_version
            == RIVAL2_REWARD_SCORING_V1_VERSION,
            "episode_unchanged": trainer.env.episode_version
            == RIVAL2_FULL_MATCH_EPISODE_VERSION,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "preserved_fields": list(preserved_fields),
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _save_snapshot(
    trainer: Rival2Trainer,
    work_dir: Path,
    *,
    final: bool,
) -> dict[str, Any]:
    trainer.add_historical_snapshot()
    label = (
        f"overnight_final_update_{trainer.iteration:05d}"
        if final
        else f"overnight_update_{trainer.iteration:05d}"
    )
    record = scoring._checkpoint_record(
        work_dir / "checkpoints" / f"rival2_scoring_{label}_resume.pt",
        label,
        trainer.iteration - 120,
        trainer,
    )
    audit = full_match._checkpoint_audit(record, trainer)
    if audit["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"checkpoint audit failed at {label}: {audit['checks']}")
    record["audit"] = audit
    record["final_deadline_snapshot"] = final
    return record


def run(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    launch_gate: dict[str, Any],
    source_payload: dict[str, Any],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    deadline: datetime,
) -> dict[str, Any]:
    policy_config = Rival2PolicyConfig(**source_payload["policy_config"])
    ppo_config = Rival2PPOConfig(**source_payload["ppo_config"])
    self_play_config = Rival2SelfPlayConfig(**source_payload["self_play_config"])
    kickoff_selector = (
        np.arange(scoring.WORLDS, dtype=np.int32) + scoring.CAMPAIGN_SEED
    ) % 5
    env = Rival2FullMatchEnv(
        scoring.WORLDS,
        args.collision_dir,
        device=args.device,
        seed=scoring.CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_SCORING_V1_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2Trainer(
        env,
        policy_config=policy_config,
        ppo_config=ppo_config,
        self_play_config=self_play_config,
        seed=1,
    )
    trainer.load_checkpoint(args.resume_checkpoint)
    loaded_audit = _loaded_state_audit(source_payload, trainer)
    _write_json(args.work_dir / "loaded_state_audit.json", loaded_audit)
    if loaded_audit["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"loaded-state audit failed: {loaded_audit['checks']}")

    start_iteration = trainer.iteration
    start_samples = trainer.total_agent_samples
    training_ledger = args.work_dir / "training_curve.jsonl"
    timing_ledger = args.work_dir / "wall_clock_curve.jsonl"
    if training_ledger.exists() or timing_ledger.exists():
        raise RuntimeError("work directory already contains a training ledger")
    checkpoints: list[dict[str, Any]] = []
    started = time.perf_counter()
    previous_completion = datetime.now(deadline.tzinfo)
    if previous_completion >= deadline:
        raise RuntimeError("deadline was reached before training could start")

    while True:
        update_started = datetime.now(deadline.tzinfo)
        full_match._train_one_update(
            phase=PHASE,
            phase_start_iteration=start_iteration,
            trainer=trainer,
            device=args.device,
            ledger=training_ledger,
        )
        update_completed = datetime.now(deadline.tzinfo)
        crossed = crosses_deadline(previous_completion, update_completed, deadline)
        _append_jsonl(
            timing_ledger,
            {
                "iteration": trainer.iteration,
                "agent_decision_samples": trainer.total_agent_samples,
                "update_started_local": update_started.isoformat(),
                "update_completed_local": update_completed.isoformat(),
                "deadline_local": deadline.isoformat(),
                "crossed_deadline": crossed,
            },
        )

        cadence_snapshot = is_snapshot_iteration(trainer.iteration)
        if cadence_snapshot or crossed:
            record = _save_snapshot(
                trainer,
                args.work_dir,
                final=crossed,
            )
            checkpoints.append(record)
            _write_json(
                args.work_dir / "progress.json",
                {
                    "deadline_local": deadline.isoformat(),
                    "checkpoints": checkpoints,
                    "latest_completed_update_local": update_completed.isoformat(),
                },
            )
            print(
                f"scoring overnight checkpoint={record['label']} "
                f"samples={trainer.total_agent_samples} sha256={record['sha256']}",
                flush=True,
            )
        if crossed:
            boundary = {
                "previous_completed_update_local": previous_completion.isoformat(),
                "crossing_update_started_local": update_started.isoformat(),
                "crossing_update_completed_local": update_completed.isoformat(),
                "deadline_local": deadline.isoformat(),
                "first_completed_update_crossing_deadline": True,
                "iteration": trainer.iteration,
                "agent_decision_samples": trainer.total_agent_samples,
            }
            _write_json(args.work_dir / "deadline_boundary.json", boundary)
            break
        previous_completion = update_completed

    acquisition_payload = torch.load(
        scoring.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    frozen_model = scoring._load_frozen_model(acquisition_payload, trainer)
    del acquisition_payload
    final_label = f"overnight_final_update_{trainer.iteration:05d}"
    final_offset = trainer.iteration - 120
    final_evaluation = scoring.evaluate_self_play(
        trainer=trainer,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        label=final_label,
        offset=final_offset,
    )
    final_comparison = scoring.evaluate_frozen_reference(
        trainer=trainer,
        frozen_model=frozen_model,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        label=final_label,
        offset=final_offset,
    )
    _write_json(args.work_dir / "final_evaluation.json", final_evaluation)
    _write_json(args.work_dir / "final_frozen_comparison.json", final_comparison)
    baseline = _read_json(args.baseline_evaluation)
    assessment = scoring.behavioral_assessment(
        baseline, final_evaluation, final_comparison
    )
    expected_sample_delta = (
        (trainer.iteration - start_iteration) * scoring.ROLLOUT_AGENT_SAMPLES
    )
    checks = {
        "loaded_state_green": loaded_audit["verdict"] == "PASS_GREEN",
        "at_least_one_update": trainer.iteration > start_iteration,
        "sample_delta_exact": trainer.total_agent_samples - start_samples
        == expected_sample_delta,
        "first_completed_update_crossed_deadline": crosses_deadline(
            datetime.fromisoformat(boundary["previous_completed_update_local"]),
            datetime.fromisoformat(boundary["crossing_update_completed_local"]),
            deadline,
        ),
        "final_checkpoint_exact": checkpoints[-1]["iteration"]
        == trainer.iteration,
        "final_checkpoint_marked": checkpoints[-1]["final_deadline_snapshot"],
        "final_evaluation_green": final_evaluation["verdict"] == "PASS_GREEN",
        "final_comparison_green": final_comparison["verdict"] == "PASS_GREEN",
        "reward_unchanged": trainer.env.reward_version
        == RIVAL2_REWARD_SCORING_V1_VERSION,
        "episode_unchanged": trainer.env.episode_version
        == RIVAL2_FULL_MATCH_EPISODE_VERSION,
        "contract_hashes_exact": trainer.env.contract_hashes
        == configuration["contract_hashes"],
        "no_nexto_training": True,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": "COMPLETE" if all(checks.values()) else "FAIL_RED",
        "source_head": launch_gate["head"],
        "configuration": configuration,
        "launch_gate": launch_gate,
        "loaded_state_audit": loaded_audit,
        "deadline_boundary": boundary,
        "start_iteration": start_iteration,
        "start_policy_version": int(source_payload["policy_version"]),
        "start_agent_decision_samples": start_samples,
        "final_iteration": trainer.iteration,
        "final_policy_version": trainer.policy_version,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "additional_updates": trainer.iteration - start_iteration,
        "additional_agent_decision_samples": trainer.total_agent_samples
        - start_samples,
        "checkpoints": checkpoints,
        "final_checkpoint": checkpoints[-1],
        "final_evaluation": final_evaluation,
        "final_frozen_acquisition_comparison": final_comparison,
        "behavioral_assessment": assessment,
        "wall_seconds_including_final_evaluations": time.perf_counter() - started,
        "checks": checks,
    }
    _write_json(args.work_dir / "run_summary.json", summary)
    if summary["execution_status"] != "COMPLETE":
        raise RuntimeError(f"scoring overnight boundary failed: {checks}")
    return summary


def publish(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    launch_gate: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    results = args.results_dir
    if results.exists() and any(results.iterdir()):
        raise RuntimeError(f"results directory is not empty: {results}")
    results.mkdir(parents=True, exist_ok=True)
    _write_json(results / "config.json", configuration)
    _write_json(results / "launch_gate.json", launch_gate)
    for name in (
        "loaded_state_audit.json",
        "training_curve.jsonl",
        "wall_clock_curve.jsonl",
        "progress.json",
        "deadline_boundary.json",
        "final_evaluation.json",
        "final_frozen_comparison.json",
        "run_summary.json",
    ):
        shutil.copy2(args.work_dir / name, results / name)

    final_source = Path(summary["final_checkpoint"]["path"])
    destination = DEFAULT_PUBLISHED_CHECKPOINT
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_source, destination)
    published_final = {
        **copy.deepcopy(summary["final_checkpoint"]),
        "path": destination.as_posix(),
        "sha256": scoring._sha256(destination),
        "size_bytes": destination.stat().st_size,
    }
    _write_json(
        results / "checkpoints.json",
        {
            "schema_version": SCHEMA_VERSION,
            "published_final_checkpoint": published_final,
            "intermediate_checkpoints": [
                {
                    "label": record["label"],
                    "iteration": record["iteration"],
                    "agent_decision_samples": record["agent_decision_samples"],
                    "sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                    "work_filename": Path(record["path"]).name,
                    "published_to_git": record is summary["final_checkpoint"],
                }
                for record in summary["checkpoints"]
            ],
        },
    )
    _write_json(
        results / "viewer_commands.json",
        {
            "schema_version": SCHEMA_VERSION,
            "viewer_commands": [
                {
                    "label": published_final["label"],
                    "checkpoint": destination.as_posix(),
                    "command": (
                        ".\\.venv\\Scripts\\python.exe -m rivalsim.viewer "
                        f"--checkpoint {str(destination).replace('/', chr(92))} "
                        "--stochastic --seed 20260827"
                    ),
                }
            ],
        },
    )

    metric = summary["final_evaluation"]["result"]
    duel = summary["final_frozen_acquisition_comparison"]["result"]["overall"]
    lines = [
        "# Rival 2.0 Scoring V1 overnight continuation",
        "",
        "The run resumed the completed update-240 checkpoint with learned state,",
        "optimizer, RNG streams, counters, opponent assignments, and historical pool",
        "exact. It trained under unchanged Reward Scoring V1/full-match contracts until",
        "the first completed PPO update crossing the authorized 07:00 ET deadline.",
        "",
        "## Final snapshot",
        "",
        f"- Policy update: `{summary['final_iteration']}`.",
        f"- Cumulative agent decisions: `{summary['final_agent_decision_samples']:,}`.",
        f"- Additional updates: `{summary['additional_updates']}`.",
        f"- Deadline: `{summary['deadline_boundary']['deadline_local']}`.",
        "- Crossing update completed: "
        f"`{summary['deadline_boundary']['crossing_update_completed_local']}`.",
        f"- Goals per simulated minute: `{metric['goals_per_simulated_minute']:.6f}`.",
        f"- Touches per simulated minute: `{metric['touches_per_simulated_minute']:.6f}`.",
        "- Counterfactual no-touch kickoff-segment fraction: "
        f"`{metric['counterfactual_no_touch_kickoff_segment_fraction']:.6%}`.",
        f"- Frozen acquisition comparison: `{duel['wins']}-{duel['losses']}`.",
        f"- Frozen acquisition goal differential: `{duel['goal_differential']:+d}`.",
        f"- Checkpoint: `{published_final['path']}`.",
        f"- Checkpoint SHA-256: `{published_final['sha256']}`.",
        "",
        "Intermediate resumable checkpoints remain in the external run directory; their",
        "hashes and counters are committed. Only the final checkpoint binary is published",
        "to Git to keep the repository evidence compact.",
        "",
    ]
    report = Path("docs/RIVAL2_SCORING_OVERNIGHT_20260827_RESULTS.md")
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    manifest_paths = [
        *sorted(path for path in results.iterdir() if path.name != "manifest.json"),
        destination,
        report,
    ]
    _write_json(
        results / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "artifacts": [
                {
                    "path": path.as_posix(),
                    "sha256": scoring._sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in manifest_paths
            ],
        },
    )


def main() -> int:
    args = parse_args()
    deadline = parse_authorized_deadline(args.deadline_local)
    args.work_dir = args.work_dir.resolve()
    args.resume_checkpoint = args.resume_checkpoint.resolve()
    args.parent_progress = args.parent_progress.resolve()
    args.baseline_evaluation = args.baseline_evaluation.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise RuntimeError(f"work directory is not empty: {args.work_dir}")
    if not args.resume_checkpoint.is_file():
        raise FileNotFoundError(args.resume_checkpoint)
    if not args.parent_progress.is_file():
        raise FileNotFoundError(args.parent_progress)
    if not args.baseline_evaluation.is_file():
        raise FileNotFoundError(args.baseline_evaluation)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source_payload = torch.load(
        args.resume_checkpoint, map_location="cpu", weights_only=False
    )
    configuration = _configuration(
        source_payload,
        args.resume_checkpoint,
        args.resume_sha256,
        deadline,
    )
    launch_gate = _launch_gate(
        configuration,
        source_payload,
        args.resume_checkpoint,
        args.parent_progress,
        args.baseline_evaluation,
        deadline,
    )
    _write_json(args.work_dir / "config.json", configuration)
    _write_json(args.work_dir / "launch_gate.json", launch_gate)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    summary = run(
        args,
        configuration,
        launch_gate,
        source_payload,
        geometry,
        meshes,
        deadline,
    )
    publish(args, configuration, launch_gate, summary)
    print(
        "RIVAL2_SCORING_OVERNIGHT COMPLETE "
        f"iteration={summary['final_iteration']} "
        f"samples={summary['final_agent_decision_samples']} "
        f"checkpoint_sha256={summary['final_checkpoint']['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
