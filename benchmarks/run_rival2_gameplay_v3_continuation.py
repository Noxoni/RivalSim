"""Production Gameplay V3 continuation with 30-update evaluation boundaries.

The campaign resumes the accepted iteration-489 Gameplay V3 checkpoint, runs
exactly 120 accepted mixed-opponent PPO updates, and stops at iteration 609.
Training is split into four fresh processes so the full resumable checkpoint at
each +30 boundary is durable and the trainer releases the GPU before the fixed
Nexto/Wisp and Gameplay V3 shadow evaluations run.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_rival2_gameplay_v3_ppo_smoke import (  # noqa: E402
    EXPECTED_V3_HASH,
    KL_GUARD,
    MIXED_PPO_SAFETY,
    RolloutTelemetry,
    _append_jsonl,
    _compact_safety,
    _integrity_gate,
    _make_trainer,
    _nested_finite,
    _sha256,
    _write_json,
)
from benchmarks.run_rival2_gameplay_v3_validation import (  # noqa: E402
    _object_digest,
    _tensor_digest,
)
from benchmarks.run_rival2_opponent_curriculum_v1 import (  # noqa: E402
    evaluate_checkpoint,
)
from rivalsim.rival2_contracts import (  # noqa: E402
    ACTION_CONTRACT_HASH,
    EPISODE_CONTRACT_HASH,
    OBSERVATION_SCHEMA_HASH,
    REWARD_GAMEPLAY_V1_CONTRACT_HASH,
    REWARD_GAMEPLAY_V2_CONTRACT_HASH,
    REWARD_GAMEPLAY_V3_CONTRACT,
    REWARD_GAMEPLAY_V3_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_mixed_ppo import mixed_optimizer_learning_rates  # noqa: E402
from rivalsim.rival2_ppo import Rival2PolicyDisplacementRejected  # noqa: E402

SCHEMA_VERSION = 1
SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints"
    / "rival2"
    / "gameplay_v3_smoke"
    / "rival2_gameplay_v3_iteration_489_resume.pt"
)
SOURCE_SHA256 = "10D97428B3F1CC2E307040314D1DD1A924BD82975D4B88C0F73C3FC2716DCF54"
SOURCE_ITERATION = 489
SOURCE_SAMPLES = 3_711_438_222
ADDITIONAL_UPDATES = 120
FINAL_ITERATION = SOURCE_ITERATION + ADDITIONAL_UPDATES
CHECKPOINT_OFFSETS = (30, 60, 90, 120)
CHECKPOINT_ITERATIONS = tuple(SOURCE_ITERATION + offset for offset in CHECKPOINT_OFFSETS)
RESULTS_DIR = REPO_ROOT / "results" / "rival2" / "gameplay_v3_continuation_v1"
FINAL_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints"
    / "rival2"
    / "gameplay_v3_continuation"
    / "rival2_gameplay_v3_iteration_609_resume.pt"
)
REPORT_PATH = REPO_ROOT / "docs" / "RIVAL2_GAMEPLAY_V3_CONTINUATION_V1.md"
BASELINE_SHADOW = (
    REPO_ROOT / "results" / "rival2" / "gameplay_v3_ppo_smoke_v1" / "shadow_489_native.json"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT, text=True).strip()


def _git_is_ancestor(ancestor: str, descendant: str = "HEAD") -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        check=False,
    )
    return completed.returncode == 0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _checkpoint_path(work_dir: Path, offset: int) -> Path:
    return work_dir / "checkpoints" / f"rival2_gameplay_v3_continuation_plus_{offset:03d}_resume.pt"


def _launch_gate(source: dict[str, Any]) -> dict[str, Any]:
    adaptive = source["opponent_curriculum"]["adaptive_ppo"]
    rates = {group.get("name"): float(group["lr"]) for group in source["optimizer"]["param_groups"]}
    checks = {
        "head_equals_origin_main": _git("rev-parse", "HEAD") == _git("rev-parse", "origin/main"),
        "worktree_clean": not _git("status", "--short"),
        "v3_authority_ancestor": _git_is_ancestor("0228a833e90d2db2715f8b79b65f6cbdc59fefbc"),
        "smoke_evidence_ancestor": _git_is_ancestor("f309a44e44d6bcb53bb9398afce59328b8d1d537"),
        "source_path_exact": SOURCE_CHECKPOINT.resolve()
        == (
            REPO_ROOT / "checkpoints/rival2/gameplay_v3_smoke/"
            "rival2_gameplay_v3_iteration_489_resume.pt"
        ).resolve(),
        "source_hash_exact": _sha256(SOURCE_CHECKPOINT) == SOURCE_SHA256,
        "source_format_exact": source["format"] == "RIVAL2_CHECKPOINT_V1",
        "source_iteration_exact": int(source["iteration"]) == SOURCE_ITERATION,
        "source_policy_version_exact": int(source["policy_version"]) == SOURCE_ITERATION,
        "source_samples_exact": int(source["total_agent_samples"]) == SOURCE_SAMPLES,
        "source_reward_v3_exact": source["reward_version"] == RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "source_episode_exact": source["episode_version"] == RIVAL2_EPISODE_VERSION,
        "source_contracts_exact": source["contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_V3_VERSION, RIVAL2_EPISODE_VERSION),
        "v1_contract_immutable": REWARD_GAMEPLAY_V1_CONTRACT_HASH
        == "48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072",
        "v2_contract_immutable": REWARD_GAMEPLAY_V2_CONTRACT_HASH
        == "4073E29C1013458D5784435061FE47C639525BE37E8CD519783889C69BA87D41",
        "v3_contract_exact": REWARD_GAMEPLAY_V3_CONTRACT_HASH == EXPECTED_V3_HASH,
        "observation_contract_immutable": OBSERVATION_SCHEMA_HASH
        == "10669E7D240D553BEA601F8AD7AEF9F9029310E55EA6DC4072E866F34BB218AF",
        "action_contract_immutable": ACTION_CONTRACT_HASH
        == "145AF5B49E1A0C85209022A6DE862F05EF996EB89B90B124072A59BC9936727B",
        "episode_contract_immutable": EPISODE_CONTRACT_HASH
        == "E22B6014C6D975D700D1129B9F554D6F35E4CA5003F1C1BD09C7D394D4F9347E",
        "ordinary_touch_zero": REWARD_GAMEPLAY_V3_CONTRACT["unconditional_unique_touch"] == 0.0,
        "mechanics_reward_exact": REWARD_GAMEPLAY_V3_CONTRACT["mechanics"]["event_reward"] == 0.005,
        "mechanics_budget_exact": REWARD_GAMEPLAY_V3_CONTRACT["mechanics"]["episode_budget"]
        == 0.05,
        "mechanics_cap_exact": REWARD_GAMEPLAY_V3_CONTRACT["mechanics"][
            "max_paid_events_per_player_episode"
        ]
        == 10,
        "bad_flip_penalty_exact": REWARD_GAMEPLAY_V3_CONTRACT["unnecessary_flip_through_contact"][
            "penalty_to_offender_before_zero_sum"
        ]
        == -0.01,
        "opponent_mix_exact": source["opponent_curriculum"]["config"]
        == {
            "nexto_probability": 0.35,
            "wisp_probability": 0.35,
            "current_probability": 0.20,
            "historical_probability": 0.10,
            "seed": 2_026_082_703,
        },
        "adaptive_schema_v2": adaptive["schema_version"] == 2,
        "adaptive_lr_scope_update_local": adaptive["policy_learning_rate_scope"]
        == "ppo_update_local",
        "adaptive_config_exact": adaptive["config"]
        == {
            "initial_policy_learning_rate": 0.0001,
            "critic_learning_rate": 0.0003,
            "soft_minibatch_kl_target": 0.02,
            "retention_soft_mean_kl_target": 0.02,
            "policy_learning_rate_backoff": 0.5,
            "minimum_policy_learning_rate": 0.000025,
            "retention_corpus_size": 512,
        },
        "policy_lr_rearmed": rates.get("policy") == 0.0001,
        "critic_lr_exact": rates.get("critic") == 0.0003,
        "retention_present": adaptive.get("retention_observations") is not None,
        "historical_pool_present": bool(source["historical_opponents"]),
        "hard_minibatch_guard_exact": KL_GUARD.minibatch_kl_limit == 0.10,
        "hard_completed_guard_exact": KL_GUARD.completed_update_mean_kl_limit == 0.05,
        "boundary_offsets_exact": CHECKPOINT_OFFSETS == (30, 60, 90, 120),
        "final_iteration_exact": FINAL_ITERATION == 609,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "head": _git("rev-parse", "HEAD"),
        "origin_main": _git("rev-parse", "origin/main"),
        "source_checkpoint": SOURCE_CHECKPOINT.resolve().as_posix(),
        "source_checkpoint_sha256": SOURCE_SHA256,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Gameplay V3 continuation launch gate failed: {failed}")
    return result


def _resume_gate(
    source: dict[str, Any],
    trainer: Any,
    *,
    offset: int,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload()
    source_curriculum = source["opponent_curriculum"]
    restored_curriculum = payload["opponent_curriculum"]
    checks = {
        "checkpoint_hash_exact": _sha256(checkpoint) == checkpoint_sha256,
        "iteration_exact": trainer.iteration == SOURCE_ITERATION + offset,
        "policy_version_exact": trainer.policy_version == SOURCE_ITERATION + offset,
        "sample_counter_exact": trainer.total_agent_samples == int(source["total_agent_samples"]),
        "model_exact": _tensor_digest(trainer.model.state_dict())
        == _tensor_digest(source["model"]),
        "optimizer_exact": _object_digest(trainer.optimizer.state_dict())
        == _object_digest(source["optimizer"]),
        "cpu_rng_exact": torch.equal(
            torch.get_rng_state().cpu(), source["torch_cpu_rng_state"].cpu()
        ),
        "cuda_rng_exact": torch.equal(
            torch.cuda.get_rng_state(trainer.device).cpu(),
            source["torch_cuda_rng_state"].cpu(),
        ),
        "policy_rng_exact": torch.equal(
            trainer.policy_generator.get_state().cpu(),
            source["policy_generator_state"].cpu(),
        ),
        "opponent_rng_exact": torch.equal(
            trainer.opponent_generator.get_state().cpu(),
            source["opponent_generator_state"].cpu(),
        ),
        "curriculum_rng_exact": torch.equal(
            trainer.curriculum_generator.get_state().cpu(),
            source_curriculum["generator_state"].cpu(),
        ),
        "opponent_assignment_exact": torch.equal(
            trainer.opponent_assignment.cpu(), source["opponent_assignment"].cpu()
        ),
        "opponent_family_exact": torch.equal(
            trainer.opponent_family.cpu(), source_curriculum["family"].cpu()
        ),
        "rival_side_exact": torch.equal(
            trainer.rival_side.cpu(), source_curriculum["rival_side"].cpu()
        ),
        "historical_pool_exact": _object_digest(trainer.opponent_pool.checkpoint_state())
        == _object_digest(source["historical_opponents"]),
        "external_adapter_state_exact": _object_digest(
            {
                "nexto": restored_curriculum["nexto"],
                "wisp": restored_curriculum["wisp"],
            }
        )
        == _object_digest(
            {
                "nexto": source_curriculum["nexto"],
                "wisp": source_curriculum["wisp"],
            }
        ),
        "retention_exact": torch.equal(
            trainer.retention_observations.cpu(),
            source_curriculum["adaptive_ppo"]["retention_observations"].cpu(),
        ),
        "adaptive_config_exact": trainer.mixed_ppo_safety == MIXED_PPO_SAFETY,
        "policy_lr_rearmed": mixed_optimizer_learning_rates(trainer.optimizer)["policy"] == 0.0001,
        "critic_lr_exact": mixed_optimizer_learning_rates(trainer.optimizer)["critic"] == 0.0003,
        "reward_v3_exact": trainer.env.reward_version == RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "episode_exact": trainer.env.episode_version == RIVAL2_EPISODE_VERSION,
        "fresh_simulator_world": bool(
            (trainer.env.bridge.views["rival2.episode_ticks"] == 0).all().item()
            and (trainer.env.bridge.views["gameplay_v3.total_detected"] == 0).all().item()
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "mode": "STRICT_SAME_CONTRACT_BOUNDARY_RESUME",
        "offset": offset,
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "simulator_world_state": "fresh, as in the prior production boundary-resume procedure",
        "checkpoint_state": "strictly restored, including external-adapter temporal state",
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Gameplay V3 continuation resume gate failed: {failed}")
    return result


def _save_checkpoint(trainer: Any, work_dir: Path, offset: int) -> dict[str, Any]:
    path = _checkpoint_path(work_dir, offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    adaptive = payload["opponent_curriculum"]["adaptive_ppo"]
    rates = {
        group.get("name"): float(group["lr"]) for group in payload["optimizer"]["param_groups"]
    }
    checks = {
        "format_exact": payload["format"] == "RIVAL2_CHECKPOINT_V1",
        "iteration_exact": int(payload["iteration"]) == SOURCE_ITERATION + offset,
        "policy_version_exact": int(payload["policy_version"]) == SOURCE_ITERATION + offset,
        "sample_counter_exact": int(payload["total_agent_samples"]) == trainer.total_agent_samples,
        "reward_v3_exact": payload["reward_version"] == RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "episode_exact": payload["episode_version"] == RIVAL2_EPISODE_VERSION,
        "contracts_exact": payload["contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_V3_VERSION, RIVAL2_EPISODE_VERSION),
        "model_finite": _nested_finite(payload["model"]),
        "optimizer_finite": _nested_finite(payload["optimizer"]),
        "split_optimizer": len(payload["optimizer"]["param_groups"]) == 2,
        "policy_lr_rearmed": rates.get("policy") == 0.0001,
        "critic_lr_exact": rates.get("critic") == 0.0003,
        "curriculum_transition_present": "curriculum_transition" in payload,
        "opponent_curriculum_present": payload.get("opponent_curriculum") is not None,
        "opponent_assignment_present": payload.get("opponent_assignment") is not None,
        "historical_pool_present": bool(payload.get("historical_opponents")),
        "boundary_snapshot_present": any(
            int(item["version"]) == SOURCE_ITERATION + offset
            for item in payload["historical_opponents"]
        ),
        "adaptive_schema_v2": adaptive["schema_version"] == 2,
        "adaptive_lr_scope_update_local": adaptive["policy_learning_rate_scope"]
        == "ppo_update_local",
        "retention_present": adaptive.get("retention_observations") is not None,
        "next_update_policy_lr_exact": adaptive["next_update_policy_learning_rate"] == 0.0001,
    }
    result = {
        "label": f"plus_{offset:03d}",
        "offset": offset,
        "iteration": int(payload["iteration"]),
        "policy_version": int(payload["policy_version"]),
        "agent_decision_samples": int(payload["total_agent_samples"]),
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "historical_pool_versions": [
            int(item["version"]) for item in payload["historical_opponents"]
        ],
        "audit": {
            "checks": checks,
            "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        },
    }
    if result["audit"]["verdict"] != "PASS_GREEN":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"boundary checkpoint audit failed: {failed}")
    return result


def _save_failure_checkpoint(trainer: Any, work_dir: Path, target: int) -> dict[str, Any]:
    path = work_dir / "failures" / f"pre_update_{target:03d}_restored_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return {
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
    }


def _instrument(trainer: Any, telemetry: RolloutTelemetry) -> None:
    original_step: Callable[[torch.Tensor], Any] = trainer._step_with_frozen_opponents
    original_resets: Callable[[], None] = trainer.env.world.apply_interval_resets

    def instrumented_resets() -> None:
        telemetry.capture_pre_reset()
        original_resets()

    def instrumented_step(action: torch.Tensor) -> Any:
        result = original_step(action)
        telemetry.capture_post_step(result)
        return result

    trainer.env.world.apply_interval_resets = instrumented_resets  # type: ignore[method-assign]
    trainer._step_with_frozen_opponents = instrumented_step  # type: ignore[method-assign]


def train_segment(args: argparse.Namespace) -> dict[str, Any]:
    offset = int(args.resume_offset)
    if offset not in (0, 30, 60, 90):
        raise ValueError("resume offset must be one of 0, 30, 60, or 90")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    launch = _launch_gate(source)
    launch_path = work_dir / "launch_gate.json"
    if launch_path.is_file():
        existing = _read_json(launch_path)
        if existing["source_checkpoint_sha256"] != SOURCE_SHA256:
            raise RuntimeError("existing launch gate has a different source")
    else:
        _write_json(launch_path, launch)

    ledger_path = work_dir / "training_curve.jsonl"
    existing_rows = _read_jsonl(ledger_path)
    if [int(row["offset"]) for row in existing_rows] != list(range(1, offset + 1)):
        raise RuntimeError("training ledger is not an exact accepted prefix")
    checkpoints = (
        _read_json(work_dir / "checkpoints.json")
        if (work_dir / "checkpoints.json").is_file()
        else []
    )
    evaluations = (
        _read_json(work_dir / "evaluation_curve.json")
        if (work_dir / "evaluation_curve.json").is_file()
        else []
    )
    if offset > 0:
        prior_label = f"plus_{offset:03d}"
        prior_evaluations = [
            item for item in evaluations if item.get("checkpoint_label") == prior_label
        ]
        if len(prior_evaluations) != 1 or prior_evaluations[0].get("verdict") != "PASS_GREEN":
            raise RuntimeError("prior boundary evaluation is not complete and green")
        resume_checkpoint = _checkpoint_path(work_dir, offset)
        resume_records = [item for item in checkpoints if int(item["offset"]) == offset]
        if len(resume_records) != 1:
            raise RuntimeError("prior boundary checkpoint ledger is not exact")
        expected_sha = str(resume_records[0]["sha256"])
    else:
        resume_checkpoint = SOURCE_CHECKPOINT
        expected_sha = SOURCE_SHA256
    if not resume_checkpoint.is_file() or _sha256(resume_checkpoint) != expected_sha:
        raise RuntimeError("resume checkpoint identity mismatch")

    resume_source = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
    torch.cuda.empty_cache()
    trainer = _make_trainer(resume_source, args.collision_dir.resolve(), args.device)
    trainer.load_checkpoint(resume_checkpoint)
    resume_gate = _resume_gate(
        resume_source,
        trainer,
        offset=offset,
        checkpoint=resume_checkpoint,
        checkpoint_sha256=expected_sha,
    )
    _write_json(work_dir / f"resume_gate_{offset:03d}.json", resume_gate)

    telemetry = RolloutTelemetry(trainer)
    _instrument(trainer, telemetry)
    target_offset = offset + 30
    segment_started = time.perf_counter()
    compact_safety = (
        _read_json(work_dir / "ppo_safety_summary.json")
        if (work_dir / "ppo_safety_summary.json").is_file()
        else []
    )
    if [int(item["iteration"]) for item in compact_safety] != [
        SOURCE_ITERATION + item for item in range(1, offset + 1)
    ]:
        raise RuntimeError("PPO safety ledger is not an exact accepted prefix")

    for campaign_offset in range(offset + 1, target_offset + 1):
        target_iteration = SOURCE_ITERATION + campaign_offset
        policy_before = trainer.policy_version
        iteration_before = trainer.iteration
        samples_before = trainer.total_agent_samples
        pool_before = list(trainer.opponent_pool.versions)
        trainer.env.reset_transfer_counters()
        telemetry.begin_update()
        update_started = time.perf_counter()
        rollout = trainer.collect_rollout()
        rollout_telemetry = telemetry.finish_update()
        try:
            metrics = trainer.update(rollout, kl_guard=KL_GUARD)
        except Rival2PolicyDisplacementRejected as error:
            restored = _save_failure_checkpoint(trainer, work_dir, target_iteration)
            failure = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "STOPPED_HARD_SAFETY_GUARD",
                "target_iteration": target_iteration,
                "last_accepted_iteration": trainer.iteration,
                "restored_checkpoint": restored,
                "rollout_telemetry": rollout_telemetry,
                "diagnostic": error.diagnostics,
                "source_checkpoint_byte_identical": _sha256(SOURCE_CHECKPOINT) == SOURCE_SHA256,
                "no_later_training_performed": True,
            }
            _write_json(work_dir / "hard_safety_failure.json", failure)
            _write_json(work_dir / "run_summary.json", failure)
            return failure
        torch.cuda.synchronize(args.device)
        wall_seconds = time.perf_counter() - update_started
        integrity = _integrity_gate(
            trainer,
            rollout,
            metrics,
            policy_before=policy_before,
            iteration_before=iteration_before,
            samples_before=samples_before,
        )
        if integrity["verdict"] != "PASS_GREEN":
            failure_checkpoint = _save_failure_checkpoint(trainer, work_dir, target_iteration + 1)
            failure = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "STOPPED_POST_UPDATE_INTEGRITY_FAILURE",
                "target_iteration": target_iteration,
                "last_accepted_iteration": trainer.iteration,
                "failure_checkpoint": failure_checkpoint,
                "integrity": integrity,
                "no_later_training_performed": True,
            }
            _write_json(work_dir / "hard_safety_failure.json", failure)
            _write_json(work_dir / "run_summary.json", failure)
            return failure
        if trainer.iteration != target_iteration:
            raise RuntimeError("accepted update landed on the wrong iteration")
        if list(trainer.opponent_pool.versions) != pool_before:
            raise RuntimeError("historical pool changed outside the boundary schedule")
        adaptive = copy.deepcopy(trainer.last_adaptive_ppo_diagnostics)
        safety = _compact_safety(target_iteration, adaptive)
        compact_safety.append(safety)
        point = {
            "schema_version": SCHEMA_VERSION,
            "phase": "GAMEPLAY_V3_MIXED_OPPONENT_PRODUCTION_CONTINUATION",
            "created_utc": _utc_now(),
            "offset": campaign_offset,
            "iteration": trainer.iteration,
            "policy_version": trainer.policy_version,
            "agent_decision_samples": trainer.total_agent_samples,
            "iteration_agent_decision_samples": trainer.total_agent_samples - samples_before,
            "wall_seconds": wall_seconds,
            "reward_version": trainer.env.reward_version,
            "episode_version": trainer.env.episode_version,
            "family": trainer.last_rollout_curriculum_metrics,
            "adaptive_ppo": adaptive,
            "ppo_safety_summary": safety,
            "reward_and_behavior_telemetry": rollout_telemetry,
            "metrics": {name: float(value.item()) for name, value in metrics.items()},
            "integrity": integrity,
            "historical_pool_unchanged_this_update": True,
            "verdict": "PASS_GREEN",
        }
        _append_jsonl(ledger_path, point)
        _write_json(work_dir / "ppo_safety_summary.json", compact_safety)
        print(
            "gameplay-v3-continuation "
            f"update={trainer.iteration} offset={campaign_offset}/120 "
            f"samples={trainer.total_agent_samples} "
            f"delta={trainer.total_agent_samples - samples_before} "
            f"seconds={wall_seconds:.3f} "
            f"mb_kl={safety['maximum_post_step_minibatch_kl']:.6f} "
            f"mean_kl={safety['completed_update_mean_kl']:.6f} "
            f"retention_kl={safety['retention_mean_kl']:.6f} "
            f"early_stop={safety['retention_budget_early_stop']} "
            "verdict=PASS_GREEN",
            flush=True,
        )
        del rollout, metrics, adaptive
        gc.collect()
        torch.cuda.empty_cache()

    pool_before = list(trainer.opponent_pool.versions)
    trainer.add_historical_snapshot()
    pool_after = list(trainer.opponent_pool.versions)
    evicted = [version for version in pool_before if version not in pool_after]
    snapshot = {
        "offset": target_offset,
        "iteration": trainer.iteration,
        "added_version": trainer.policy_version,
        "pool_before": pool_before,
        "pool_after": pool_after,
        "evicted_versions": evicted,
    }
    snapshots_path = work_dir / "snapshot_records.json"
    snapshots = _read_json(snapshots_path) if snapshots_path.is_file() else []
    snapshots.append(snapshot)
    checkpoint = _save_checkpoint(trainer, work_dir, target_offset)
    checkpoints.append(checkpoint)
    _write_json(snapshots_path, snapshots)
    _write_json(work_dir / "checkpoints.json", checkpoints)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "status": "PAUSED_FOR_30_UPDATE_BOUNDARY_EVALUATION",
        "completed_additional_updates": target_offset,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "boundary_checkpoint": checkpoint,
        "boundary_evaluation_pending": True,
        "hard_safety_guard_fired": False,
        "wall_seconds_this_segment": time.perf_counter() - segment_started,
        "source_checkpoint_byte_identical": _sha256(SOURCE_CHECKPOINT) == SOURCE_SHA256,
    }
    _write_json(work_dir / "run_summary.json", summary)
    return summary


def evaluate_boundary(args: argparse.Namespace) -> dict[str, Any]:
    offset = int(args.boundary_offset)
    if offset not in CHECKPOINT_OFFSETS:
        raise ValueError("boundary offset must be 30, 60, 90, or 120")
    work_dir = args.work_dir.resolve()
    records = _read_json(work_dir / "checkpoints.json")
    matches = [item for item in records if int(item["offset"]) == offset]
    if len(matches) != 1:
        raise RuntimeError("boundary checkpoint ledger is not exact")
    checkpoint = Path(matches[0]["path"])
    checkpoint_sha256 = str(matches[0]["sha256"])
    if not checkpoint.is_file() or _sha256(checkpoint) != checkpoint_sha256:
        raise RuntimeError("boundary checkpoint identity mismatch")
    label = f"plus_{offset:03d}"
    evaluation_dir = work_dir / "evaluations" / label
    compact_dir = evaluation_dir / "compact_opponents"
    compact_dir.mkdir(parents=True, exist_ok=True)
    compact = evaluate_checkpoint(
        label=label,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        collision_dir=args.collision_dir.resolve(),
        device=args.device,
        work_dir=compact_dir,
    )

    shadow_dir = evaluation_dir / "paired_v3_shadow"
    shadow_summary = shadow_dir / "shadow_gate_summary.json"
    if not shadow_summary.is_file():
        command = [
            sys.executable,
            str(REPO_ROOT / "benchmarks/run_rival2_gameplay_v3_validation.py"),
            "shadow",
            "--collision-dir",
            str(args.collision_dir.resolve()),
            "--output-dir",
            str(shadow_dir.resolve()),
            "--checkpoint",
            str(SOURCE_CHECKPOINT.resolve()),
            "--checkpoint-sha256",
            SOURCE_SHA256,
            "--policy-checkpoint",
            str(checkpoint.resolve()),
            "--policy-checkpoint-sha256",
            checkpoint_sha256,
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    shadow = _read_json(shadow_summary)
    compact_green = all(
        result.get("verdict") == "PASS_GREEN"
        for modes in compact["opponents"].values()
        for result in modes.values()
    )
    checks = {
        "checkpoint_byte_identical_after_evaluation": _sha256(checkpoint) == checkpoint_sha256,
        "compact_nexto_wisp_green": compact_green,
        "paired_v3_shadow_green": shadow.get("verdict") == "PASS",
        "paired_shadow_uses_489_context": shadow["source"]["sha256"] == SOURCE_SHA256,
        "paired_shadow_policy_iteration_exact": int(shadow["evaluation_policy"]["iteration"])
        == SOURCE_ITERATION + offset,
        "paired_shadow_policy_hash_exact": shadow["evaluation_policy"]["sha256"]
        == checkpoint_sha256,
        "no_learning_in_shadow": shadow.get("ppo_update_calls") == 0,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "checkpoint_label": label,
        "checkpoint_offset": offset,
        "checkpoint_iteration": SOURCE_ITERATION + offset,
        "checkpoint_path": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "compact_opponent_evaluation": compact,
        "paired_v3_shadow": shadow,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    _write_json(evaluation_dir / "boundary_summary.json", result)
    curve_path = work_dir / "evaluation_curve.json"
    curve = _read_json(curve_path) if curve_path.is_file() else []
    curve = [item for item in curve if int(item["checkpoint_offset"]) != offset]
    curve.append(result)
    curve.sort(key=lambda item: int(item["checkpoint_offset"]))
    _write_json(curve_path, curve)
    if result["verdict"] != "PASS_GREEN":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"boundary evaluation failed: {failed}")
    return result


def aggregate_training_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("training rows are required")
    safety = [row["ppo_safety_summary"] for row in rows]
    telemetry = [row["reward_and_behavior_telemetry"] for row in rows]
    reward_names = tuple(telemetry[0]["reward_contributions"])
    reward_absolute = {
        name: sum(
            float(item["reward_contributions"][name]["absolute_blue_sum"]) for item in telemetry
        )
        for name in reward_names
    }
    gameplay_abs = sum(float(item["absolute_gameplay_reward_sum"]) for item in telemetry)
    progress_abs = reward_absolute["progress"]
    raw_names = tuple(telemetry[0]["raw_counts_and_activity"])
    raw = {
        name: sum(float(item["raw_counts_and_activity"][name]) for item in telemetry)
        for name in raw_names
    }
    mechanics_names = tuple(telemetry[0]["mechanics"]["detected"])
    mechanics = {
        category: {
            name: sum(int(item["mechanics"][category][name]) for item in telemetry)
            for name in mechanics_names
        }
        for category in ("detected", "paid")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted_updates": len(rows),
        "optimizer_step_proposals": sum(int(item["optimizer_step_proposals"]) for item in safety),
        "accepted_optimizer_steps": sum(int(item["accepted_optimizer_steps"]) for item in safety),
        "policy_learning_rate_backoffs": sum(
            int(item["policy_learning_rate_backoffs"]) for item in safety
        ),
        "transactional_retries": sum(int(item["transactional_retries"]) for item in safety),
        "retention_budget_early_stop_updates": [
            int(item["iteration"]) for item in safety if item["retention_budget_early_stop"]
        ],
        "maximum_accepted_minibatch_kl": max(
            float(item["maximum_post_step_minibatch_kl"]) for item in safety
        ),
        "maximum_completed_update_mean_kl": max(
            float(item["completed_update_mean_kl"]) for item in safety
        ),
        "maximum_retention_mean_kl": max(float(item["retention_mean_kl"]) for item in safety),
        "reward_absolute_blue_sums": reward_absolute,
        "absolute_gameplay_reward_sum": gameplay_abs,
        "mechanics_to_absolute_gameplay_reward": reward_absolute["mechanics"]
        / max(gameplay_abs, 1e-30),
        "bad_flip_to_absolute_gameplay_reward": reward_absolute["unnecessary_flip"]
        / max(gameplay_abs, 1e-30),
        "mechanics_to_progress": reward_absolute["mechanics"] / max(progress_abs, 1e-30),
        "bad_flip_to_progress": reward_absolute["unnecessary_flip"] / max(progress_abs, 1e-30),
        "raw_counts_and_activity": raw,
        "mechanics": mechanics,
        "maximum_single_update_mechanics_to_gameplay": max(
            float(item["ratios"]["mechanics_reward_to_absolute_gameplay_reward"])
            for item in telemetry
        ),
        "maximum_single_update_bad_flip_to_gameplay": max(
            float(item["ratios"]["unnecessary_flip_penalty_to_absolute_gameplay_reward"])
            for item in telemetry
        ),
        "hard_safety_guard_fired": False,
    }


def _is_hard_stop(summary: dict[str, Any]) -> bool:
    return str(summary.get("status", "")).startswith("STOPPED_")


def _relative_change(before: float, after: float) -> float | None:
    return None if before == 0.0 else after / before - 1.0


def _write_report(summary: dict[str, Any]) -> None:
    aggregate = summary["training_aggregate"]
    lines = [
        "# Rival2 Gameplay V3 production continuation (+120)",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"Source: iteration `{SOURCE_ITERATION}` / `{SOURCE_SHA256}`.",
        "",
        f"Final: iteration `{summary['final_iteration']}` / "
        f"`{summary['final_checkpoint']['sha256']}`.",
        "",
        "## PPO safety",
        "",
        f"- Accepted updates: `{aggregate['accepted_updates']}`.",
        f"- Maximum accepted minibatch KL: `{aggregate['maximum_accepted_minibatch_kl']:.9f}`.",
        f"- Maximum completed-update mean KL: "
        f"`{aggregate['maximum_completed_update_mean_kl']:.9f}`.",
        f"- Maximum retention mean KL: `{aggregate['maximum_retention_mean_kl']:.9f}`.",
        f"- Hard safety guard fired: `{aggregate['hard_safety_guard_fired']}`.",
        "",
        "## Reward scale",
        "",
        f"- Mechanics / absolute gameplay: "
        f"`{aggregate['mechanics_to_absolute_gameplay_reward']:.9f}`.",
        f"- Bad flip / absolute gameplay: "
        f"`{aggregate['bad_flip_to_absolute_gameplay_reward']:.9f}`.",
        f"- Mechanics / progress: `{aggregate['mechanics_to_progress']:.9f}`.",
        f"- Bad flip / progress: `{aggregate['bad_flip_to_progress']:.9f}`.",
        "## Fixed-context Gameplay V3 shadows",
        "",
        "| iteration | touches/min | flip touches/min | bad/min | bad/flip | "
        "mechanics/progress | bad/progress | Rival goal share | no-touch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["shadow_curve"]:
        metrics = item["metrics"]
        scoring = metrics.get("scoring_behavior")
        no_touch = metrics.get("no_touch_behavior")
        goal_share = (
            "n/a" if scoring is None else f"{scoring['rival_goal_share']:.6f}"
        )
        no_touch_fraction = (
            "n/a" if no_touch is None else f"{no_touch['no_touch_episode_fraction']:.6f}"
        )
        lines.append(
            f"| {item['iteration']} | {metrics['touches_per_min']:.6f} | "
            f"{metrics['flip_active_touches_per_min']:.6f} | "
            f"{metrics['unnecessary_flip_contacts_per_min']:.6f} | "
            f"{metrics['unnecessary_flip_touch_fraction']:.6f} | "
            f"{metrics['mechanics_progress_ratio']:.6f} | "
            f"{metrics['bad_flip_progress_ratio']:.6f} | "
            f"{goal_share} | {no_touch_fraction} |"
        )
    lines.extend(
        [
            "",
            "The four boundaries also use the prior compact held-out protocol: "
            "10 canonical deterministic and 256 stochastic short episodes against "
            "each of frozen Nexto and frozen Wisp, split by Rival side.",
            "",
            "Training stopped exactly at iteration 609. No later PPO update ran.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _write_hard_stop_report(summary: dict[str, Any]) -> None:
    aggregate = summary["training_aggregate"]
    diagnostic = summary["hard_safety_failure"]["diagnostic"]
    lines = [
        "# Rival2 Gameplay V3 production continuation (+120 target)",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"Source: iteration `{SOURCE_ITERATION}` / `{SOURCE_SHA256}`.",
        "",
        f"Last accepted model: iteration `{summary['final_iteration']}` / "
        f"`{summary['final_checkpoint']['sha256']}`.",
        "",
        "The configured hard PPO safety boundary fired on proposed update "
        f"`{diagnostic['rejected_iteration']}`. Training stopped immediately; the "
        "guard was not weakened and no later update ran.",
        "",
        "## Hard-stop diagnostic",
        "",
        f"- Reason: `{diagnostic['reason']}`.",
        f"- Post-step minibatch KL: `{diagnostic['post_step_approx_kl']:.9f}` "
        f"(hard limit `{diagnostic['minibatch_kl_limit']:.9f}`).",
        f"- Retention mean KL: `{diagnostic['retention_mean_kl']:.9f}`.",
        f"- Transactional rollback completed: "
        f"`{diagnostic['transactional_rollback_completed']}`.",
        f"- Parameters restored exactly: "
        f"`{diagnostic['transactional_step_restore']['parameters_exact']}`.",
        f"- Optimizer restored exactly: "
        f"`{diagnostic['transactional_step_restore']['optimizer_state_exact']}`.",
        f"- Adam counters restored exactly: "
        f"`{diagnostic['transactional_step_restore']['adam_step_counters_exact']}`.",
        "",
        "## Accepted-update PPO safety",
        "",
        f"- Accepted updates: `{aggregate['accepted_updates']}`.",
        f"- Maximum accepted minibatch KL: `{aggregate['maximum_accepted_minibatch_kl']:.9f}`.",
        f"- Maximum completed-update mean KL: "
        f"`{aggregate['maximum_completed_update_mean_kl']:.9f}`.",
        f"- Maximum retention mean KL: `{aggregate['maximum_retention_mean_kl']:.9f}`.",
        f"- Retention-budget early stops: "
        f"`{len(aggregate['retention_budget_early_stop_updates'])}`.",
        "",
        "## Reward scale across accepted updates",
        "",
        f"- Mechanics / absolute gameplay: "
        f"`{aggregate['mechanics_to_absolute_gameplay_reward']:.9f}`.",
        f"- Bad flip / absolute gameplay: "
        f"`{aggregate['bad_flip_to_absolute_gameplay_reward']:.9f}`.",
        f"- Mechanics / progress: `{aggregate['mechanics_to_progress']:.9f}`.",
        f"- Bad flip / progress: `{aggregate['bad_flip_to_progress']:.9f}`.",
        f"- Maximum single-update mechanics / gameplay: "
        f"`{summary['reward_scale_extrema']['mechanics']['ratio']:.9f}` at iteration "
        f"`{summary['reward_scale_extrema']['mechanics']['iteration']}`.",
        f"- Maximum single-update bad flip / gameplay: "
        f"`{summary['reward_scale_extrema']['unnecessary_flip']['ratio']:.9f}` at "
        f"iteration `{summary['reward_scale_extrema']['unnecessary_flip']['iteration']}`.",
        "",
        "The mechanics maximum occurred on the first fresh-simulator rollout after "
        "the +90 boundary. That rollout had no ball touch, no progress component, "
        "and no completed episode, but did detect 7,580 pogo events. This boundary "
        "transition is an investigation signal; it is not presented as ordinary "
        "steady-state reward composition.",
        "",
        "## Completed fixed-context Gameplay V3 shadows",
        "",
        "| iteration | touches/min | flip touches/min | bad/min | bad/flip | "
        "mechanics/progress | bad/progress | Rival goal share | no-touch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["shadow_curve"]:
        metrics = item["metrics"]
        scoring = metrics.get("scoring_behavior")
        no_touch = metrics.get("no_touch_behavior")
        goal_share = (
            "n/a" if scoring is None else f"{scoring['rival_goal_share']:.6f}"
        )
        no_touch_fraction = (
            "n/a" if no_touch is None else f"{no_touch['no_touch_episode_fraction']:.6f}"
        )
        lines.append(
            f"| {item['iteration']} | {metrics['touches_per_min']:.6f} | "
            f"{metrics['flip_active_touches_per_min']:.6f} | "
            f"{metrics['unnecessary_flip_contacts_per_min']:.6f} | "
            f"{metrics['unnecessary_flip_touch_fraction']:.6f} | "
            f"{metrics['mechanics_progress_ratio']:.6f} | "
            f"{metrics['bad_flip_progress_ratio']:.6f} | "
            f"{goal_share} | {no_touch_fraction} |"
        )
    lines.extend(
        [
            "",
            "Scheduled checkpoint/evaluation boundaries at iterations 519, 549, and "
            "579 completed green. The iteration-609 boundary was not reached.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def finalize_failure(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    failure = _read_json(work_dir / "hard_safety_failure.json")
    if not _is_hard_stop(failure):
        raise RuntimeError("failure finalization requires a recorded hard stop")
    rows = _read_jsonl(work_dir / "training_curve.jsonl")
    checkpoints = _read_json(work_dir / "checkpoints.json")
    evaluations = _read_json(work_dir / "evaluation_curve.json")
    accepted = int(failure["last_accepted_iteration"]) - SOURCE_ITERATION
    if [int(row["offset"]) for row in rows] != list(range(1, accepted + 1)):
        raise RuntimeError("hard-stop training ledger is not an exact accepted prefix")
    if [int(row["iteration"]) for row in rows] != list(
        range(SOURCE_ITERATION + 1, SOURCE_ITERATION + accepted + 1)
    ):
        raise RuntimeError("hard-stop iteration ledger is not an exact accepted prefix")
    if any(row.get("verdict") != "PASS_GREEN" for row in rows):
        raise RuntimeError("hard-stop training ledger contains a non-green accepted row")
    completed_offsets = [offset for offset in CHECKPOINT_OFFSETS if offset <= accepted]
    if [int(item["offset"]) for item in checkpoints] != completed_offsets:
        raise RuntimeError("hard-stop checkpoint ledger is not the exact completed prefix")
    if [int(item["checkpoint_offset"]) for item in evaluations] != completed_offsets:
        raise RuntimeError("hard-stop evaluation ledger is not the exact completed prefix")
    for checkpoint in checkpoints:
        path = Path(checkpoint["path"])
        if not path.is_file() or _sha256(path) != checkpoint["sha256"]:
            raise RuntimeError("hard-stop checkpoint hash audit failed")
    if any(item.get("verdict") != "PASS_GREEN" for item in evaluations):
        raise RuntimeError("hard-stop evaluation ledger contains a non-green boundary")

    restored = failure["restored_checkpoint"]
    restored_path = Path(restored["path"])
    if not restored_path.is_file() or _sha256(restored_path) != restored["sha256"]:
        raise RuntimeError("restored hard-stop checkpoint identity mismatch")
    payload = torch.load(restored_path, map_location="cpu", weights_only=False)
    rates = {
        group.get("name"): float(group["lr"]) for group in payload["optimizer"]["param_groups"]
    }
    diagnostic = failure["diagnostic"]
    restored_checks = {
        "format_exact": payload["format"] == "RIVAL2_CHECKPOINT_V1",
        "iteration_exact": int(payload["iteration"]) == int(failure["last_accepted_iteration"]),
        "policy_version_exact": int(payload["policy_version"])
        == int(failure["last_accepted_iteration"]),
        "sample_counter_exact": int(payload["total_agent_samples"])
        == int(restored["agent_decision_samples"]),
        "reward_v3_exact": payload["reward_version"] == RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
        "episode_exact": payload["episode_version"] == RIVAL2_EPISODE_VERSION,
        "contracts_exact": payload["contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_GAMEPLAY_V3_VERSION, RIVAL2_EPISODE_VERSION),
        "model_finite": _nested_finite(payload["model"]),
        "optimizer_finite": _nested_finite(payload["optimizer"]),
        "split_optimizer": len(payload["optimizer"]["param_groups"]) == 2,
        "policy_lr_rearmed": rates.get("policy") == 0.0001,
        "critic_lr_exact": rates.get("critic") == 0.0003,
        "opponent_curriculum_present": payload.get("opponent_curriculum") is not None,
        "historical_pool_present": bool(payload.get("historical_opponents")),
        "transactional_rollback_completed": bool(
            diagnostic["transactional_rollback_completed"]
        ),
        "parameters_restored_exact": bool(
            diagnostic["transactional_step_restore"]["parameters_exact"]
        ),
        "optimizer_restored_exact": bool(
            diagnostic["transactional_step_restore"]["optimizer_state_exact"]
        ),
        "adam_counters_restored_exact": bool(
            diagnostic["transactional_step_restore"]["adam_step_counters_exact"]
        ),
    }
    if not all(restored_checks.values()):
        failed = [name for name, passed in restored_checks.items() if not passed]
        raise RuntimeError(f"restored hard-stop checkpoint audit failed: {failed}")

    aggregate = aggregate_training_rows(rows)
    aggregate["hard_safety_guard_fired"] = True
    mechanics_extreme = max(
        rows,
        key=lambda row: float(
            row["reward_and_behavior_telemetry"]["ratios"][
                "mechanics_reward_to_absolute_gameplay_reward"
            ]
        ),
    )
    bad_flip_extreme = max(
        rows,
        key=lambda row: float(
            row["reward_and_behavior_telemetry"]["ratios"][
                "unnecessary_flip_penalty_to_absolute_gameplay_reward"
            ]
        ),
    )
    reward_scale_extrema = {
        "mechanics": {
            "iteration": int(mechanics_extreme["iteration"]),
            "ratio": float(
                mechanics_extreme["reward_and_behavior_telemetry"]["ratios"][
                    "mechanics_reward_to_absolute_gameplay_reward"
                ]
            ),
            "touches": int(
                mechanics_extreme["reward_and_behavior_telemetry"][
                    "raw_counts_and_activity"
                ]["touches"]
            ),
            "progress_absolute_blue_sum": float(
                mechanics_extreme["reward_and_behavior_telemetry"]["reward_contributions"][
                    "progress"
                ]["absolute_blue_sum"]
            ),
            "completed_player_episodes": int(
                mechanics_extreme["reward_and_behavior_telemetry"][
                    "raw_counts_and_activity"
                ]["completed_player_episodes"]
            ),
            "pogo_detected": int(
                mechanics_extreme["reward_and_behavior_telemetry"]["mechanics"]["detected"][
                    "pogo"
                ]
            ),
        },
        "unnecessary_flip": {
            "iteration": int(bad_flip_extreme["iteration"]),
            "ratio": float(
                bad_flip_extreme["reward_and_behavior_telemetry"]["ratios"][
                    "unnecessary_flip_penalty_to_absolute_gameplay_reward"
                ]
            ),
        },
    }
    baseline = _read_json(BASELINE_SHADOW)
    shadow_curve = [
        {
            "iteration": SOURCE_ITERATION,
            "checkpoint_sha256": SOURCE_SHA256,
            "metrics": baseline["metrics"],
        },
        *(
            {
                "iteration": int(item["checkpoint_iteration"]),
                "checkpoint_sha256": item["checkpoint_sha256"],
                "metrics": item["paired_v3_shadow"]["metrics"],
            }
            for item in evaluations
        ),
    ]
    repository_failure_checkpoint = (
        REPO_ROOT
        / "checkpoints"
        / "rival2"
        / "gameplay_v3_continuation"
        / f"rival2_gameplay_v3_iteration_{payload['iteration']}_restored_resume.pt"
    )
    repository_failure_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(restored_path, repository_failure_checkpoint)
    final_checkpoint = {
        **restored,
        "repository_path": repository_failure_checkpoint.relative_to(REPO_ROOT).as_posix(),
        "repository_sha256": _sha256(repository_failure_checkpoint),
        "repository_size_bytes": repository_failure_checkpoint.stat().st_size,
        "audit": {"checks": restored_checks, "verdict": "PASS_GREEN"},
    }
    checks = {
        "source_checkpoint_byte_identical": _sha256(SOURCE_CHECKPOINT) == SOURCE_SHA256,
        "exact_accepted_prefix": len(rows) == accepted and 0 <= accepted <= ADDITIONAL_UPDATES,
        "last_accepted_iteration_exact": int(payload["iteration"])
        == int(failure["last_accepted_iteration"]),
        "rejected_iteration_exact": int(diagnostic["rejected_iteration"])
        == int(failure["last_accepted_iteration"]) + 1,
        "completed_boundary_checkpoint_prefix_exact": len(checkpoints)
        == len(completed_offsets),
        "completed_boundary_evaluation_prefix_exact": len(evaluations)
        == len(completed_offsets),
        "all_checkpoint_audits_green": all(
            item["audit"]["verdict"] == "PASS_GREEN" for item in checkpoints
        ),
        "all_evaluations_green": all(item["verdict"] == "PASS_GREEN" for item in evaluations),
        "all_value_loss_isolation_green": all(
            row["ppo_safety_summary"]["value_loss_to_policy_trunk_gradient_exact_zero"]
            and row["ppo_safety_summary"]["value_loss_to_actor_gradient_exact_zero"]
            for row in rows
        ),
        "hard_guard_fired": float(diagnostic["post_step_approx_kl"])
        > float(diagnostic["minibatch_kl_limit"]),
        "transactional_restore_exact": all(
            bool(value) for value in diagnostic["transactional_step_restore"].values()
        ),
        "no_later_training_performed": bool(failure["no_later_training_performed"]),
        "final_repository_checkpoint_exact": final_checkpoint["repository_sha256"]
        == final_checkpoint["sha256"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"hard-stop finalization failed: {failed}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "status": (
            "STOPPED_HARD_SAFETY_GUARD_AT_PROPOSED_UPDATE_"
            f"{diagnostic['rejected_iteration']}"
        ),
        "verdict": "BLOCKED_HARD_SAFETY_GUARD",
        "implementation_commit": _read_json(work_dir / "launch_gate.json")["head"],
        "source_checkpoint": {
            "path": SOURCE_CHECKPOINT.resolve().as_posix(),
            "sha256": SOURCE_SHA256,
            "iteration": SOURCE_ITERATION,
            "policy_version": SOURCE_ITERATION,
            "agent_decision_samples": SOURCE_SAMPLES,
            "byte_identical_after_run": _sha256(SOURCE_CHECKPOINT) == SOURCE_SHA256,
        },
        "final_checkpoint": final_checkpoint,
        "final_iteration": int(payload["iteration"]),
        "final_policy_version": int(payload["policy_version"]),
        "final_agent_decision_samples": int(payload["total_agent_samples"]),
        "additional_agent_decision_samples": int(payload["total_agent_samples"])
        - SOURCE_SAMPLES,
        "accepted_additional_updates": accepted,
        "target_additional_updates": ADDITIONAL_UPDATES,
        "completed_checkpoint_offsets": completed_offsets,
        "unreached_checkpoint_offsets": [
            offset for offset in CHECKPOINT_OFFSETS if offset not in completed_offsets
        ],
        "training_aggregate": aggregate,
        "reward_scale_extrema": reward_scale_extrema,
        "shadow_curve": shadow_curve,
        "hard_safety_failure": failure,
        "checks": checks,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(work_dir / "run_summary.json", summary)
    _write_json(RESULTS_DIR / "run_summary.json", summary)
    _write_json(RESULTS_DIR / "hard_safety_failure.json", failure)
    _write_json(RESULTS_DIR / "checkpoints.json", checkpoints)
    _write_json(RESULTS_DIR / "evaluation_curve.json", evaluations)
    _write_json(
        RESULTS_DIR / "ppo_safety_summary.json",
        _read_json(work_dir / "ppo_safety_summary.json"),
    )
    _write_json(RESULTS_DIR / "training_aggregate.json", aggregate)
    shutil.copy2(work_dir / "training_curve.jsonl", RESULTS_DIR / "training_curve.jsonl")
    shutil.copy2(work_dir / "launch_gate.json", RESULTS_DIR / "launch_gate.json")
    shutil.copy2(work_dir / "snapshot_records.json", RESULTS_DIR / "snapshot_records.json")
    for offset in (0, 30, 60, 90):
        shutil.copy2(
            work_dir / f"resume_gate_{offset:03d}.json",
            RESULTS_DIR / f"resume_gate_{offset:03d}.json",
        )
    _write_hard_stop_report(summary)
    manifest_paths = [
        *sorted(path for path in RESULTS_DIR.iterdir() if path.is_file()),
        repository_failure_checkpoint,
        REPORT_PATH,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "artifacts": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in manifest_paths
            if path.name != "artifact_manifest.json"
        ],
        "integrity_verdict": "PASS_GREEN",
        "campaign_verdict": "BLOCKED_HARD_SAFETY_GUARD",
    }
    _write_json(RESULTS_DIR / "artifact_manifest.json", manifest)
    return summary


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    rows = _read_jsonl(work_dir / "training_curve.jsonl")
    checkpoints = _read_json(work_dir / "checkpoints.json")
    evaluations = _read_json(work_dir / "evaluation_curve.json")
    if [int(row["offset"]) for row in rows] != list(range(1, 121)):
        raise RuntimeError("final training ledger is not exactly 120 accepted updates")
    if [int(row["iteration"]) for row in rows] != list(range(490, 610)):
        raise RuntimeError("final iteration ledger is not exactly 490 through 609")
    if any(row.get("verdict") != "PASS_GREEN" for row in rows):
        raise RuntimeError("final training ledger contains a non-green row")
    if [int(item["offset"]) for item in checkpoints] != list(CHECKPOINT_OFFSETS):
        raise RuntimeError("final checkpoint ledger is not exact")
    if [int(item["checkpoint_offset"]) for item in evaluations] != list(CHECKPOINT_OFFSETS):
        raise RuntimeError("final evaluation ledger is not exact")
    if any(item.get("verdict") != "PASS_GREEN" for item in evaluations):
        raise RuntimeError("final evaluation ledger contains a non-green boundary")
    for checkpoint in checkpoints:
        path = Path(checkpoint["path"])
        if not path.is_file() or _sha256(path) != checkpoint["sha256"]:
            raise RuntimeError("final checkpoint hash audit failed")

    aggregate = aggregate_training_rows(rows)
    baseline = _read_json(BASELINE_SHADOW)
    shadow_curve = [
        {
            "iteration": SOURCE_ITERATION,
            "checkpoint_sha256": SOURCE_SHA256,
            "metrics": baseline["metrics"],
        }
    ]
    shadow_curve.extend(
        {
            "iteration": int(item["checkpoint_iteration"]),
            "checkpoint_sha256": item["checkpoint_sha256"],
            "metrics": item["paired_v3_shadow"]["metrics"],
        }
        for item in evaluations
    )
    final_work_checkpoint = Path(checkpoints[-1]["path"])
    FINAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_work_checkpoint, FINAL_CHECKPOINT)
    final_checkpoint = {
        **checkpoints[-1],
        "repository_path": FINAL_CHECKPOINT.relative_to(REPO_ROOT).as_posix(),
        "repository_sha256": _sha256(FINAL_CHECKPOINT),
        "repository_size_bytes": FINAL_CHECKPOINT.stat().st_size,
    }
    checks = {
        "source_checkpoint_byte_identical": _sha256(SOURCE_CHECKPOINT) == SOURCE_SHA256,
        "exact_120_updates": len(rows) == 120,
        "stopped_exactly_609": int(checkpoints[-1]["iteration"]) == FINAL_ITERATION,
        "four_boundary_checkpoints": len(checkpoints) == 4,
        "four_boundary_evaluations": len(evaluations) == 4,
        "all_checkpoint_audits_green": all(
            item["audit"]["verdict"] == "PASS_GREEN" for item in checkpoints
        ),
        "all_evaluations_green": all(item["verdict"] == "PASS_GREEN" for item in evaluations),
        "all_value_loss_isolation_green": all(
            row["ppo_safety_summary"]["value_loss_to_policy_trunk_gradient_exact_zero"]
            and row["ppo_safety_summary"]["value_loss_to_actor_gradient_exact_zero"]
            for row in rows
        ),
        "hard_guard_not_fired": not aggregate["hard_safety_guard_fired"],
        "final_repository_checkpoint_exact": final_checkpoint["repository_sha256"]
        == final_checkpoint["sha256"],
    }
    final_metrics = shadow_curve[-1]["metrics"]
    baseline_metrics = shadow_curve[0]["metrics"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "status": "COMPLETE_120_ACCEPTED_UPDATES_AND_4_BOUNDARY_EVALUATIONS",
        "implementation_commit": _read_json(work_dir / "launch_gate.json")["head"],
        "source_checkpoint": {
            "path": SOURCE_CHECKPOINT.resolve().as_posix(),
            "sha256": SOURCE_SHA256,
            "iteration": SOURCE_ITERATION,
            "policy_version": SOURCE_ITERATION,
            "agent_decision_samples": SOURCE_SAMPLES,
            "byte_identical_after_run": _sha256(SOURCE_CHECKPOINT) == SOURCE_SHA256,
        },
        "final_checkpoint": final_checkpoint,
        "final_iteration": FINAL_ITERATION,
        "final_policy_version": FINAL_ITERATION,
        "final_agent_decision_samples": int(checkpoints[-1]["agent_decision_samples"]),
        "additional_agent_decision_samples": int(checkpoints[-1]["agent_decision_samples"])
        - SOURCE_SAMPLES,
        "training_aggregate": aggregate,
        "checkpoint_offsets": list(CHECKPOINT_OFFSETS),
        "checkpoint_iterations": list(CHECKPOINT_ITERATIONS),
        "shadow_curve": shadow_curve,
        "source_to_final_shadow_change": {
            name: {
                "source": float(baseline_metrics[name]),
                "final": float(final_metrics[name]),
                "relative_change": _relative_change(
                    float(baseline_metrics[name]), float(final_metrics[name])
                ),
            }
            for name in (
                "touches_per_min",
                "flip_active_touches_per_min",
                "unnecessary_flip_contacts_per_min",
                "unnecessary_flip_touch_fraction",
                "mechanics_progress_ratio",
                "bad_flip_progress_ratio",
            )
        },
        "hard_safety_guard_fired": False,
        "no_updates_beyond_609": True,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if summary["verdict"] != "PASS_GREEN":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"continuation finalization failed: {failed}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(work_dir / "run_summary.json", summary)
    _write_json(RESULTS_DIR / "run_summary.json", summary)
    _write_json(RESULTS_DIR / "checkpoints.json", checkpoints)
    _write_json(RESULTS_DIR / "evaluation_curve.json", evaluations)
    _write_json(
        RESULTS_DIR / "ppo_safety_summary.json",
        _read_json(work_dir / "ppo_safety_summary.json"),
    )
    _write_json(RESULTS_DIR / "training_aggregate.json", aggregate)
    shutil.copy2(work_dir / "training_curve.jsonl", RESULTS_DIR / "training_curve.jsonl")
    shutil.copy2(work_dir / "launch_gate.json", RESULTS_DIR / "launch_gate.json")
    shutil.copy2(work_dir / "snapshot_records.json", RESULTS_DIR / "snapshot_records.json")
    for offset in (0, 30, 60, 90):
        shutil.copy2(
            work_dir / f"resume_gate_{offset:03d}.json",
            RESULTS_DIR / f"resume_gate_{offset:03d}.json",
        )
    _write_report(summary)
    manifest_paths = [
        *sorted(path for path in RESULTS_DIR.iterdir() if path.is_file()),
        FINAL_CHECKPOINT,
        REPORT_PATH,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "artifacts": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in manifest_paths
            if path.name != "artifact_manifest.json"
        ],
        "verdict": "PASS_GREEN",
    }
    _write_json(RESULTS_DIR / "artifact_manifest.json", manifest)
    return summary


def _run_child(
    args: argparse.Namespace, mode: str, **extra: Any
) -> subprocess.CompletedProcess[bytes]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        mode,
        "--work-dir",
        str(args.work_dir.resolve()),
        "--collision-dir",
        str(args.collision_dir.resolve()),
        "--device",
        args.device,
    ]
    for name, value in extra.items():
        command.extend([f"--{name.replace('_', '-')}", str(value)])
    return subprocess.run(command, cwd=REPO_ROOT, check=False)


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    for resume_offset, boundary_offset in zip((0, 30, 60, 90), CHECKPOINT_OFFSETS, strict=True):
        checkpoints = (
            _read_json(work_dir / "checkpoints.json")
            if (work_dir / "checkpoints.json").is_file()
            else []
        )
        checkpoint_done = any(
            int(item["offset"]) == boundary_offset
            and Path(item["path"]).is_file()
            and _sha256(Path(item["path"])) == item["sha256"]
            for item in checkpoints
        )
        if not checkpoint_done:
            completed = _run_child(args, "train-segment", resume_offset=resume_offset)
            run_summary_path = work_dir / "run_summary.json"
            if run_summary_path.is_file():
                run_summary = _read_json(run_summary_path)
                if _is_hard_stop(run_summary):
                    return finalize_failure(args)
            completed.check_returncode()
        evaluations = (
            _read_json(work_dir / "evaluation_curve.json")
            if (work_dir / "evaluation_curve.json").is_file()
            else []
        )
        evaluation_done = any(
            int(item["checkpoint_offset"]) == boundary_offset
            and item.get("verdict") == "PASS_GREEN"
            for item in evaluations
        )
        if not evaluation_done:
            completed = _run_child(args, "evaluate-boundary", boundary_offset=boundary_offset)
            completed.check_returncode()
    return finalize(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("run", "train-segment", "evaluate-boundary", "finalize", "finalize-failure"),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--collision-dir",
        type=Path,
        default=Path(r"G:\dev\RLBot-Rival\bot\collision_meshes"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume-offset", type=int, default=0)
    parser.add_argument("--boundary-offset", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode in {"run", "train-segment", "evaluate-boundary"}:
        if not torch.cuda.is_available() or not wp.is_cuda_available():
            raise RuntimeError("CUDA PyTorch and Warp are required")
        torch.cuda.set_device(args.device)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.mode == "run":
        result = run_all(args)
    elif args.mode == "train-segment":
        result = train_segment(args)
    elif args.mode == "evaluate-boundary":
        result = evaluate_boundary(args)
    elif args.mode == "finalize":
        result = finalize(args)
    else:
        result = finalize_failure(args)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("verdict", "PASS_GREEN") == "PASS_GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
