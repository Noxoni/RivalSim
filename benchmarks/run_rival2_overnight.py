"""Execute the authorized Rival 2.0 acquisition-to-base-reward overnight curriculum."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmarks.run_rival2_campaign01 as campaign01
import benchmarks.run_rival2_campaign02 as campaign02
import benchmarks.run_rival2_campaign03 as campaign03
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    REWARD_CONTRACT_HASH,
    REWARD_V2_CONTRACT_HASH,
    RIVAL2_REWARD_V2_VERSION,
    RIVAL2_REWARD_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_training import Rival2Trainer

AUTHORIZED_HEAD = "83b0934fe254f6662cccc6bd45e0e2be4c449499"
CAMPAIGN04_CLOSEOUT = "4c121fab8c4bfe38fbf60f1c81a47d2dce898235"
RESUME_CHECKPOINT = Path(
    "checkpoints/rival2/campaign04/rival2_campaign04_1b_resume.pt"
)
RESUME_CHECKPOINT_SHA256 = (
    "DB5AA09B2CAD40D4C1F5DB1014FDE245C58994A6948458212751724F782BE6B0"
)
START_ITERATION = 120
START_SAMPLES = 1_006_632_960
WORLDS = 131072
ROLLOUT_AGENT_SAMPLES = 8_388_608
CAMPAIGN_SEED = 20260826
EVALUATION_SEED = 920260826
EVALUATION_WORLDS = 4096
PHASE_A_EVALUATION_INTERVAL = 30
PHASE_A_THRESHOLD = 0.01
PHASE_A_REQUIRED_CONSECUTIVE = 2
PHASE_B_UPDATES = 239
PHASE_B_ADDITIONAL_SAMPLES = 2_004_877_312
PHASE_B_EVALUATION_OFFSETS = (60, 120, 180, 239)
PHASE_C_THRESHOLDS = (
    ("1h", 3_600.0),
    ("2h", 7_200.0),
    ("3h", 10_800.0),
    ("4h", 14_400.0),
    ("5h", 18_000.0),
    ("6h", 21_600.0),
)
PHASE_C_DURATION_SECONDS = 21_600.0
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def frozen_configuration() -> dict[str, Any]:
    campaign04_config = json.loads(
        Path("results/rival2/campaign04/config.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "Rival 2.0 uninterrupted overnight acquisition-to-base-reward curriculum",
        "user_steering_override": (
            "Phase C extended prospectively from three to six real elapsed hours before the "
            "published run started"
        ),
        "authorized_head": AUTHORIZED_HEAD,
        "campaign04_closeout": CAMPAIGN04_CLOSEOUT,
        "resume_checkpoint": RESUME_CHECKPOINT.as_posix(),
        "resume_checkpoint_sha256": RESUME_CHECKPOINT_SHA256,
        "start_iteration": START_ITERATION,
        "start_agent_decision_samples": START_SAMPLES,
        "campaign_seed": CAMPAIGN_SEED,
        "worlds": WORLDS,
        "rollout_agent_decision_samples": ROLLOUT_AGENT_SAMPLES,
        "policy_config": campaign04_config["policy_config"],
        "policy_config_hash": campaign04_config["policy_config_hash"],
        "ppo_config": campaign04_config["ppo_config"],
        "ppo_config_hash": campaign04_config["ppo_config_hash"],
        "self_play_config": campaign04_config["self_play_config"],
        "phase_a": {
            "reward_version": RIVAL2_REWARD_V2_VERSION,
            "reward_contract_hash": REWARD_V2_CONTRACT_HASH,
            "evaluation_interval_updates": PHASE_A_EVALUATION_INTERVAL,
            "no_touch_truncated_fraction_threshold": PHASE_A_THRESHOLD,
            "required_consecutive_passing_evaluations": PHASE_A_REQUIRED_CONSECUTIVE,
            "sample_cap": None,
            "stop_rule": (
                "first evaluation checkpoint completing two consecutive ordinary held-out "
                "evaluations with no_touch_truncated_fraction <= 0.01"
            ),
        },
        "reward_transition": {
            "source_reward_version": RIVAL2_REWARD_V2_VERSION,
            "source_reward_contract_hash": REWARD_V2_CONTRACT_HASH,
            "destination_reward_version": RIVAL2_REWARD_VERSION,
            "destination_reward_contract_hash": REWARD_CONTRACT_HASH,
            "changed_semantics": ["reward_contract"],
            "removed_term": "per-agent approach-distance reward",
            "preserve_all_non_reward_training_and_runtime_state": True,
        },
        "phase_b": {
            "reward_version": RIVAL2_REWARD_VERSION,
            "additional_updates": PHASE_B_UPDATES,
            "additional_agent_decision_samples": PHASE_B_ADDITIONAL_SAMPLES,
            "evaluation_update_offsets": list(PHASE_B_EVALUATION_OFFSETS),
            "stop_rule": "exactly 239 completed PPO updates after the reward switch",
        },
        "phase_c": {
            "reward_version": RIVAL2_REWARD_VERSION,
            "timer": "time.perf_counter monotonic wall clock",
            "thresholds_seconds": [seconds for _, seconds in PHASE_C_THRESHOLDS],
            "checkpoint_labels": [label for label, _ in PHASE_C_THRESHOLDS],
            "checkpoint_and_evaluation_overhead_counts": True,
            "stop_rule": (
                "first completed PPO update at or after 21600 elapsed seconds from the "
                "post-Phase-B boundary"
            ),
            "predetermined_sample_target": None,
        },
        "evaluation": {
            "seed": EVALUATION_SEED,
            "worlds": EVALUATION_WORLDS,
            "mode": "ordinary stochastic self-play",
            "episode_scope": "first completed episode per world",
        },
        "snapshot_schedule": (
            "continue the established Campaign 04 schedule by adding the current policy to the "
            "bounded historical pool at every authorized evaluation checkpoint before saving; "
            "when the bounded pool evicts a version, deterministically return live assignments "
            "to that unavailable version to current-policy self-play"
        ),
        "prohibited": [
            "preflight, smoke, parity, regression, lint, compile, or test ceremony",
            "PPO, model, observation, action, episode, or self-play changes",
            "reward changes other than the one authorized V2-to-V1 transition",
            "viewer implementation",
            "v0.6 work",
        ],
    }


def checkpoint_authority(configuration: dict[str, Any]) -> dict[str, Any]:
    if campaign01._git("rev-parse", "HEAD") != AUTHORIZED_HEAD:
        raise RuntimeError("overnight curriculum must start from the authorized HEAD")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", CAMPAIGN04_CLOSEOUT, "HEAD"],
        check=True,
        capture_output=True,
    )
    if not RESUME_CHECKPOINT.is_file():
        raise RuntimeError("Campaign 04 final resume checkpoint is missing")
    actual_sha256 = campaign01._sha256_file(RESUME_CHECKPOINT)
    payload = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=False)
    historical_versions = [int(entry["version"]) for entry in payload["historical_opponents"]]
    checks = {
        "campaign04_is_ancestor": True,
        "checkpoint_sha256_exact": actual_sha256 == RESUME_CHECKPOINT_SHA256,
        "checkpoint_format_exact": payload["format"] == "RIVAL2_CHECKPOINT_V1",
        "reward_version_exact": payload["reward_version"] == RIVAL2_REWARD_V2_VERSION,
        "contract_hashes_exact": payload["contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_V2_VERSION),
        "policy_config_hash_exact": payload["policy_config_hash"]
        == configuration["policy_config_hash"],
        "ppo_config_hash_exact": payload["ppo_config_hash"]
        == configuration["ppo_config_hash"],
        "ppo_config_exact": payload["ppo_config"] == configuration["ppo_config"],
        "self_play_config_exact": payload["self_play_config"]
        == configuration["self_play_config"],
        "iteration_exact": int(payload["iteration"]) == START_ITERATION,
        "policy_version_exact": int(payload["policy_version"]) == START_ITERATION,
        "sample_count_exact": int(payload["total_agent_samples"]) == START_SAMPLES,
        "optimizer_state_present": bool(payload["optimizer"]["state"]),
        "torch_cpu_rng_present": payload["torch_cpu_rng_state"].numel() > 0,
        "torch_cuda_rng_present": payload["torch_cuda_rng_state"].numel() > 0,
        "policy_generator_rng_present": payload["policy_generator_state"].numel() > 0,
        "opponent_generator_rng_present": payload["opponent_generator_state"].numel() > 0,
        "opponent_assignment_shape_exact": list(payload["opponent_assignment"].shape)
        == [WORLDS],
        "historical_versions_exact": historical_versions
        == [0, 2, 3, 6, 12, 30, 60, 90, 120],
        "no_prior_curriculum_transition": "curriculum_transition" not in payload,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "checkpoint_path": RESUME_CHECKPOINT.as_posix(),
        "checkpoint_sha256": actual_sha256,
        "checkpoint_size_bytes": RESUME_CHECKPOINT.stat().st_size,
        "resume_iteration": int(payload["iteration"]),
        "resume_policy_version": int(payload["policy_version"]),
        "resume_agent_decision_samples": int(payload["total_agent_samples"]),
        "historical_policy_versions": historical_versions,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    del payload
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"overnight resume authority failed: {checks}")
    return result


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, separators=(",", ":")) + "\n")


def _save_checkpoint(label: str, trainer: Rival2Trainer, work_dir: Path) -> dict[str, Any]:
    path = work_dir / "checkpoints" / f"rival2_overnight_{label}_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    record = campaign01._checkpoint_record(path, label, trainer)
    record.update(
        {
            "reward_version": trainer.env.reward_version,
            "contract_hashes": dict(trainer.env.contract_hashes),
            "curriculum_transition_recorded": trainer.curriculum_transition is not None,
        }
    )
    return record


@torch.no_grad()
def _evaluate_checkpoint(
    *,
    phase: str,
    label: str,
    trainer: Rival2Trainer,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    phase_c_elapsed_at_update_completion: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    mode = campaign01._evaluate_mode(
        mode="stochastic_self_play",
        checkpoint_model=trainer.model,
        initialization_model=trainer.model,
        collision_dir=collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=device,
        reward_version=trainer.env.reward_version,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "phase": phase,
        "checkpoint_label": label,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "reward_version": trainer.env.reward_version,
        "contract_hashes": dict(trainer.env.contract_hashes),
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_worlds": EVALUATION_WORLDS,
        "mode": "ordinary stochastic self-play",
        "result": mode,
        "wall_seconds": time.perf_counter() - started,
        "verdict": mode["verdict"],
    }
    if phase_c_elapsed_at_update_completion is not None:
        result["phase_c_elapsed_seconds_at_update_completion"] = (
            phase_c_elapsed_at_update_completion
        )
    return result


def _training_point(
    *,
    phase: str,
    phase_update_offset: int,
    trainer: Rival2Trainer,
    rollout: Any,
    metrics: dict[str, torch.Tensor],
    policy_version_before: int,
    samples_before: int,
    iteration_seconds: float,
    phase_c_elapsed_seconds: float | None,
) -> dict[str, Any]:
    integrity = campaign01._rollout_integrity(
        trainer,
        rollout,
        metrics,
        policy_version_before=policy_version_before,
        samples_before=samples_before,
    )
    transfer = trainer.env.hot_path_transfer_bytes()
    values = integrity["metrics"]
    loss_without_entropy = values["policy_loss"] + (
        trainer.ppo_config.value_loss_coefficient * values["value_loss"]
    )
    loss_identity_error = abs(values["total_loss"] - loss_without_entropy)
    expected_contracts = contract_hashes_for_reward(trainer.env.reward_version)
    integrity["checks"].update(
        {
            "reward_version_active": trainer.env.reward_version
            in (RIVAL2_REWARD_V2_VERSION, RIVAL2_REWARD_VERSION),
            "reward_contract_exact": trainer.env.contract_hashes == expected_contracts,
            "entropy_coefficient_exact_zero": trainer.ppo_config.entropy_coefficient == 0.0,
            "total_loss_excludes_entropy": loss_identity_error <= 2e-7,
            "zero_hot_h2d": transfer["h2d"] == 0,
            "zero_hot_d2h": transfer["d2h"] == 0,
        }
    )
    integrity["hot_path_transfer_bytes"] = transfer
    integrity["verdict"] = "PASS_GREEN" if all(integrity["checks"].values()) else "FAIL_RED"
    result = {
        "phase": phase,
        "phase_update_offset": phase_update_offset,
        "reward_version": trainer.env.reward_version,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "iteration_agent_decision_samples": trainer.total_agent_samples - samples_before,
        "wall_seconds": iteration_seconds,
        "agent_decisions_per_second": (trainer.total_agent_samples - samples_before)
        / iteration_seconds,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device),
        "integrity": integrity,
        "ppo_stability": {
            "entropy_coefficient": 0.0,
            "entropy_optimization_contribution": 0.0,
            "loss_without_entropy": loss_without_entropy,
            "reported_total_loss": values["total_loss"],
            "loss_identity_absolute_error": loss_identity_error,
            "approximate_kl": values["approx_kl"],
            "clip_fraction": values["clip_fraction"],
            "gradient_norm": values["gradient_norm"],
            "post_clip_gradient_norm": values["post_clip_gradient_norm"],
        },
    }
    if phase_c_elapsed_seconds is not None:
        result["phase_c_elapsed_seconds_at_update_completion"] = phase_c_elapsed_seconds
    return result


def _train_one_update(
    *,
    phase: str,
    phase_start_iteration: int,
    trainer: Rival2Trainer,
    args: argparse.Namespace,
    training_curve_path: Path,
    phase_c_started: float | None = None,
) -> dict[str, Any]:
    policy_version_before = trainer.policy_version
    samples_before = trainer.total_agent_samples
    trainer.env.reset_transfer_counters()
    torch.cuda.reset_peak_memory_stats(args.device)
    iteration_started = time.perf_counter()
    rollout, metrics = trainer.train_iteration()
    torch.cuda.synchronize()
    iteration_seconds = time.perf_counter() - iteration_started
    phase_c_elapsed = (
        time.perf_counter() - phase_c_started if phase_c_started is not None else None
    )
    point = _training_point(
        phase=phase,
        phase_update_offset=trainer.iteration - phase_start_iteration,
        trainer=trainer,
        rollout=rollout,
        metrics=metrics,
        policy_version_before=policy_version_before,
        samples_before=samples_before,
        iteration_seconds=iteration_seconds,
        phase_c_elapsed_seconds=phase_c_elapsed,
    )
    _append_jsonl(training_curve_path, point)
    values = point["integrity"]["metrics"]
    elapsed_text = (
        "" if phase_c_elapsed is None else f" phase_c_elapsed={phase_c_elapsed:.3f}"
    )
    print(
        f"overnight phase={phase} update={trainer.iteration} "
        f"samples={trainer.total_agent_samples} seconds={iteration_seconds:.3f} "
        f"kl={values['approx_kl']:.6f} clip={values['clip_fraction']:.6f} "
        f"integrity={point['integrity']['verdict']}{elapsed_text}",
        flush=True,
    )
    if point["integrity"]["verdict"] != "PASS_GREEN":
        trainer.save_checkpoint(
            args.work_dir / "checkpoints" / f"integrity_failure_{trainer.iteration}.pt"
        )
        raise RuntimeError(f"training integrity failed at update {trainer.iteration}")
    del rollout, metrics
    gc.collect()
    return point


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _transition_reward(
    *,
    trainer: Rival2Trainer,
    acquisition_checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    acquisition_path = Path(acquisition_checkpoint["path"])
    before_payload = torch.load(acquisition_path, map_location="cpu", weights_only=False)
    runtime_identity_before = {
        "model": id(trainer.model),
        "optimizer": id(trainer.optimizer),
        "policy_generator": id(trainer.policy_generator),
        "opponent_generator": id(trainer.opponent_generator),
        "opponent_assignment_data_ptr": trainer.opponent_assignment.data_ptr(),
        "historical_policy_objects": [id(policy) for policy in trainer.opponent_pool.policies],
        "world": id(trainer.env.world),
        "observation_data_ptr": trainer.env.observation.data_ptr(),
        "decision_count": trainer.env.decision_count,
    }
    transition = trainer.transition_reward_curriculum(
        source_reward_version=RIVAL2_REWARD_V2_VERSION,
        destination_reward_version=RIVAL2_REWARD_VERSION,
        transition_record={
            "schema_version": SCHEMA_VERSION,
            "authority": "handoff/rival2-overnight/README.md reward-only curriculum migration",
            "authorized_head": AUTHORIZED_HEAD,
            "parent_checkpoint_path": acquisition_path.resolve().as_posix(),
            "parent_checkpoint_sha256": acquisition_checkpoint["sha256"],
            "parent_checkpoint_iteration": acquisition_checkpoint["iteration"],
            "parent_checkpoint_agent_decision_samples": acquisition_checkpoint[
                "agent_decision_samples"
            ],
            "authorized_transition": "RIVAL2_REWARD_V2 -> RIVAL2_REWARD_V1",
            "removed_term": "per-agent approach-distance reward",
        },
    )
    runtime_identity_after = {
        "model": id(trainer.model),
        "optimizer": id(trainer.optimizer),
        "policy_generator": id(trainer.policy_generator),
        "opponent_generator": id(trainer.opponent_generator),
        "opponent_assignment_data_ptr": trainer.opponent_assignment.data_ptr(),
        "historical_policy_objects": [id(policy) for policy in trainer.opponent_pool.policies],
        "world": id(trainer.env.world),
        "observation_data_ptr": trainer.env.observation.data_ptr(),
        "decision_count": trainer.env.decision_count,
    }
    transition_checkpoint = _save_checkpoint("reward_v1_transition", trainer, args.work_dir)
    after_payload = torch.load(
        Path(transition_checkpoint["path"]), map_location="cpu", weights_only=False
    )
    preserved_fields = sorted(
        set(before_payload)
        - {"contract_hashes", "reward_version", "curriculum_transition"}
    )
    field_checks = {
        field: _nested_exact(before_payload[field], after_payload[field])
        for field in preserved_fields
    }
    checks = {
        "runtime_objects_preserved_in_place": runtime_identity_before
        == runtime_identity_after,
        "all_checkpoint_state_fields_exact": all(field_checks.values()),
        "only_curriculum_metadata_added": set(after_payload)
        == set(before_payload) | {"curriculum_transition"},
        "source_reward_v2_exact": before_payload["reward_version"]
        == RIVAL2_REWARD_V2_VERSION,
        "source_contracts_exact": before_payload["contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_V2_VERSION),
        "destination_reward_v1_exact": after_payload["reward_version"]
        == RIVAL2_REWARD_VERSION,
        "destination_contracts_exact": after_payload["contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_VERSION),
        "transition_record_exact": after_payload["curriculum_transition"] == transition,
        "parent_sha_recorded": transition["parent_checkpoint_sha256"]
        == acquisition_checkpoint["sha256"],
        "iteration_unchanged": transition_checkpoint["iteration"]
        == acquisition_checkpoint["iteration"],
        "policy_version_unchanged": transition_checkpoint["policy_version"]
        == acquisition_checkpoint["policy_version"],
        "sample_count_unchanged": transition_checkpoint["agent_decision_samples"]
        == acquisition_checkpoint["agent_decision_samples"],
    }
    reload_gate = campaign02._exact_reload_gate(
        trainer, Path(transition_checkpoint["path"]), campaign03.campaign03_ppo_config()
    )
    checks["post_transition_checkpoint_exact_reload"] = reload_gate["verdict"] == "PASS_GREEN"
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "transition": transition,
        "source_checkpoint": acquisition_checkpoint,
        "post_transition_checkpoint": transition_checkpoint,
        "preserved_checkpoint_fields": preserved_fields,
        "field_exact_checks": field_checks,
        "runtime_identity_preserved": runtime_identity_before == runtime_identity_after,
        "checks": checks,
        "post_transition_checkpoint_reload": reload_gate,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    del before_payload, after_payload
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"authorized reward transition failed: {checks}")
    campaign01._write_json(args.work_dir / "reward_transition.json", result)
    return result, transition_checkpoint


def run_curriculum(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    resume_authority: dict[str, Any],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> int:
    torch.manual_seed(CAMPAIGN_SEED)
    torch.cuda.manual_seed(CAMPAIGN_SEED)
    kickoff_selector = (np.arange(WORLDS, dtype=np.int32) + CAMPAIGN_SEED) % 5
    env = Rival2Env(
        WORLDS,
        args.collision_dir,
        device=args.device,
        seed=CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_V2_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
    )
    trainer = Rival2Trainer(
        env,
        ppo_config=campaign03.campaign03_ppo_config(),
        seed=CAMPAIGN_SEED,
    )
    trainer.load_checkpoint(RESUME_CHECKPOINT)
    loaded_checks = {
        "resume_authority_green": resume_authority["verdict"] == "PASS_GREEN",
        "iteration_exact": trainer.iteration == START_ITERATION,
        "policy_version_exact": trainer.policy_version == START_ITERATION,
        "sample_count_exact": trainer.total_agent_samples == START_SAMPLES,
        "historical_versions_exact": trainer.opponent_pool.versions
        == [0, 2, 3, 6, 12, 30, 60, 90, 120],
        "reward_v2_active": env.reward_version == RIVAL2_REWARD_V2_VERSION,
        "contract_hashes_exact": env.contract_hashes
        == contract_hashes_for_reward(RIVAL2_REWARD_V2_VERSION),
        "entropy_coefficient_zero": trainer.ppo_config.entropy_coefficient == 0.0,
        "no_prior_transition": trainer.curriculum_transition is None,
    }
    if not all(loaded_checks.values()):
        raise RuntimeError(f"overnight loaded-state check failed: {loaded_checks}")
    campaign01._write_json(
        args.work_dir / "loaded_state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": campaign01._utc_now(),
            "checks": loaded_checks,
            "verdict": "PASS_GREEN",
        },
    )

    training_curve_path = args.work_dir / "training_curve.jsonl"
    if training_curve_path.exists():
        raise RuntimeError("overnight work directory already contains a training curve")
    curriculum_started = time.perf_counter()

    # Phase A: continue Reward V2 until two consecutive held-out evaluations are <= 1%.
    phase_a_started = time.perf_counter()
    phase_a_evaluations: list[dict[str, Any]] = []
    phase_a_checkpoints: list[dict[str, Any]] = []
    consecutive_passing = 0
    next_evaluation_iteration = START_ITERATION + PHASE_A_EVALUATION_INTERVAL
    acquisition_checkpoint: dict[str, Any] | None = None
    while consecutive_passing < PHASE_A_REQUIRED_CONSECUTIVE:
        _train_one_update(
            phase="A_REWARD_V2_ACQUISITION",
            phase_start_iteration=START_ITERATION,
            trainer=trainer,
            args=args,
            training_curve_path=training_curve_path,
        )
        if trainer.iteration != next_evaluation_iteration:
            continue
        trainer.add_historical_snapshot()
        label = f"phase_a_update_{trainer.iteration}"
        checkpoint = _save_checkpoint(label, trainer, args.work_dir)
        evaluation = _evaluate_checkpoint(
            phase="A_REWARD_V2_ACQUISITION",
            label=label,
            trainer=trainer,
            collision_dir=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
        )
        campaign01._write_json(args.work_dir / f"evaluation_{label}.json", evaluation)
        if evaluation["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"Phase A evaluation integrity failed at update {trainer.iteration}")
        no_touch = evaluation["result"]["no_touch_truncated_fraction"]
        threshold_passed = no_touch <= PHASE_A_THRESHOLD
        consecutive_passing = consecutive_passing + 1 if threshold_passed else 0
        evaluation["acquisition_threshold"] = PHASE_A_THRESHOLD
        evaluation["acquisition_threshold_passed"] = threshold_passed
        evaluation["consecutive_passing_evaluations"] = consecutive_passing
        campaign01._write_json(args.work_dir / f"evaluation_{label}.json", evaluation)
        phase_a_checkpoints.append(checkpoint)
        phase_a_evaluations.append(evaluation)
        campaign01._write_json(
            args.work_dir / "phase_a_progress.json",
            {
                "checkpoints": phase_a_checkpoints,
                "evaluations": phase_a_evaluations,
                "consecutive_passing_evaluations": consecutive_passing,
            },
        )
        print(
            f"overnight phase=A evaluation_update={trainer.iteration} "
            f"touches_per_minute={evaluation['result']['touches_per_simulated_minute']:.6f} "
            f"no_touch={no_touch:.6f} threshold_passed={threshold_passed} "
            f"consecutive={consecutive_passing}",
            flush=True,
        )
        if consecutive_passing == PHASE_A_REQUIRED_CONSECUTIVE:
            acquisition_checkpoint = checkpoint
            break
        next_evaluation_iteration += PHASE_A_EVALUATION_INTERVAL

    if acquisition_checkpoint is None:
        raise RuntimeError("Phase A ended without an acquisition-complete checkpoint")
    phase_a_summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": "COMPLETE",
        "reward_version": RIVAL2_REWARD_V2_VERSION,
        "start_iteration": START_ITERATION,
        "start_agent_decision_samples": START_SAMPLES,
        "final_iteration": trainer.iteration,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "continuation_updates": trainer.iteration - START_ITERATION,
        "evaluation_count": len(phase_a_evaluations),
        "threshold": PHASE_A_THRESHOLD,
        "required_consecutive": PHASE_A_REQUIRED_CONSECUTIVE,
        "confirming_evaluation_labels": [
            evaluation["checkpoint_label"] for evaluation in phase_a_evaluations[-2:]
        ],
        "confirming_no_touch_fractions": [
            evaluation["result"]["no_touch_truncated_fraction"]
            for evaluation in phase_a_evaluations[-2:]
        ],
        "acquisition_complete_checkpoint": acquisition_checkpoint,
        "wall_seconds_including_evaluations": time.perf_counter() - phase_a_started,
    }
    campaign01._write_json(args.work_dir / "phase_a_summary.json", phase_a_summary)
    print(
        f"overnight phase=A COMPLETE update={trainer.iteration} "
        f"samples={trainer.total_agent_samples} evaluations={len(phase_a_evaluations)}",
        flush=True,
    )

    # Explicit authorized V2 -> V1 migration, preserving all runtime/trainer state.
    transition_result, transition_checkpoint = _transition_reward(
        trainer=trainer,
        acquisition_checkpoint=acquisition_checkpoint,
        args=args,
    )
    print(
        f"overnight reward_transition=PASS_GREEN update={trainer.iteration} "
        f"parent_sha256={acquisition_checkpoint['sha256']} "
        f"checkpoint_sha256={transition_checkpoint['sha256']}",
        flush=True,
    )

    # Phase B: exactly 239 Reward V1 updates.
    phase_b_started = time.perf_counter()
    phase_b_start_iteration = trainer.iteration
    phase_b_start_samples = trainer.total_agent_samples
    phase_b_checkpoints: list[dict[str, Any]] = []
    phase_b_evaluations: list[dict[str, Any]] = []
    next_phase_b_evaluation = 0
    while trainer.iteration - phase_b_start_iteration < PHASE_B_UPDATES:
        _train_one_update(
            phase="B_REWARD_V1_2B",
            phase_start_iteration=phase_b_start_iteration,
            trainer=trainer,
            args=args,
            training_curve_path=training_curve_path,
        )
        offset = trainer.iteration - phase_b_start_iteration
        if offset != PHASE_B_EVALUATION_OFFSETS[next_phase_b_evaluation]:
            continue
        trainer.add_historical_snapshot()
        label = f"phase_b_plus_{offset}"
        checkpoint = _save_checkpoint(label, trainer, args.work_dir)
        evaluation = _evaluate_checkpoint(
            phase="B_REWARD_V1_2B",
            label=label,
            trainer=trainer,
            collision_dir=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
        )
        campaign01._write_json(args.work_dir / f"evaluation_{label}.json", evaluation)
        if evaluation["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"Phase B evaluation integrity failed at offset {offset}")
        phase_b_checkpoints.append(checkpoint)
        phase_b_evaluations.append(evaluation)
        print(
            f"overnight phase=B evaluation_offset={offset} update={trainer.iteration} "
            f"touches_per_minute={evaluation['result']['touches_per_simulated_minute']:.6f} "
            f"goals_per_minute={evaluation['result']['goals_per_simulated_minute']:.6f} "
            f"no_touch={evaluation['result']['no_touch_truncated_fraction']:.6f}",
            flush=True,
        )
        next_phase_b_evaluation += 1

    phase_b_exact = (
        trainer.iteration == phase_b_start_iteration + PHASE_B_UPDATES
        and trainer.policy_version == phase_b_start_iteration + PHASE_B_UPDATES
        and trainer.total_agent_samples == phase_b_start_samples + PHASE_B_ADDITIONAL_SAMPLES
        and next_phase_b_evaluation == len(PHASE_B_EVALUATION_OFFSETS)
    )
    if not phase_b_exact:
        raise RuntimeError("Phase B did not end at its exact 239-update boundary")
    phase_b_checkpoint = phase_b_checkpoints[-1]
    phase_b_reload = campaign02._exact_reload_gate(
        trainer, Path(phase_b_checkpoint["path"]), campaign03.campaign03_ppo_config()
    )
    if phase_b_reload["verdict"] != "PASS_GREEN":
        raise RuntimeError("Phase B 2B-base-reward checkpoint exact reload failed")
    phase_b_summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": "COMPLETE",
        "reward_version": RIVAL2_REWARD_VERSION,
        "start_iteration": phase_b_start_iteration,
        "start_agent_decision_samples": phase_b_start_samples,
        "final_iteration": trainer.iteration,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "additional_updates": trainer.iteration - phase_b_start_iteration,
        "additional_agent_decision_samples": trainer.total_agent_samples
        - phase_b_start_samples,
        "checkpoints": phase_b_checkpoints,
        "evaluation_labels": [item["checkpoint_label"] for item in phase_b_evaluations],
        "base_reward_2b_checkpoint": phase_b_checkpoint,
        "base_reward_2b_checkpoint_reload": phase_b_reload,
        "wall_seconds_including_evaluations": time.perf_counter() - phase_b_started,
    }
    campaign01._write_json(args.work_dir / "phase_b_summary.json", phase_b_summary)
    print(
        f"overnight phase=B COMPLETE update={trainer.iteration} "
        f"samples={trainer.total_agent_samples} checkpoint_sha256={phase_b_checkpoint['sha256']}",
        flush=True,
    )

    # Phase C: timer starts only after the complete Phase B checkpoint/evaluation/reload boundary.
    phase_c_started = time.perf_counter()
    phase_c_start_utc = campaign01._utc_now()
    phase_c_start_iteration = trainer.iteration
    phase_c_start_samples = trainer.total_agent_samples
    phase_c_checkpoints: list[dict[str, Any]] = []
    phase_c_evaluations: list[dict[str, Any]] = []
    next_phase_c_threshold = 0
    while next_phase_c_threshold < len(PHASE_C_THRESHOLDS):
        point = _train_one_update(
            phase="C_REWARD_V1_SIX_HOURS",
            phase_start_iteration=phase_c_start_iteration,
            trainer=trainer,
            args=args,
            training_curve_path=training_curve_path,
            phase_c_started=phase_c_started,
        )
        label, threshold_seconds = PHASE_C_THRESHOLDS[next_phase_c_threshold]
        elapsed_at_update = point["phase_c_elapsed_seconds_at_update_completion"]
        if elapsed_at_update < threshold_seconds:
            continue
        trainer.add_historical_snapshot()
        checkpoint_label = f"phase_c_{label}"
        checkpoint = _save_checkpoint(checkpoint_label, trainer, args.work_dir)
        checkpoint["phase_c_threshold_seconds"] = threshold_seconds
        checkpoint["phase_c_elapsed_seconds_at_update_completion"] = elapsed_at_update
        checkpoint["phase_c_elapsed_seconds_after_checkpoint"] = (
            time.perf_counter() - phase_c_started
        )
        evaluation = _evaluate_checkpoint(
            phase="C_REWARD_V1_SIX_HOURS",
            label=checkpoint_label,
            trainer=trainer,
            collision_dir=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
            phase_c_elapsed_at_update_completion=elapsed_at_update,
        )
        evaluation["phase_c_threshold_seconds"] = threshold_seconds
        evaluation["phase_c_elapsed_seconds_after_evaluation"] = (
            time.perf_counter() - phase_c_started
        )
        campaign01._write_json(
            args.work_dir / f"evaluation_{checkpoint_label}.json", evaluation
        )
        if evaluation["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"Phase C {label} evaluation integrity failed")
        phase_c_checkpoints.append(checkpoint)
        phase_c_evaluations.append(evaluation)
        print(
            f"overnight phase=C checkpoint={label} update={trainer.iteration} "
            f"samples={trainer.total_agent_samples} elapsed_at_update={elapsed_at_update:.3f} "
            f"elapsed_after_evaluation={evaluation['phase_c_elapsed_seconds_after_evaluation']:.3f} "
            f"touches_per_minute={evaluation['result']['touches_per_simulated_minute']:.6f} "
            f"goals_per_minute={evaluation['result']['goals_per_simulated_minute']:.6f} "
            f"no_touch={evaluation['result']['no_touch_truncated_fraction']:.6f}",
            flush=True,
        )
        next_phase_c_threshold += 1

    final_checkpoint = phase_c_checkpoints[-1]
    final_reload = campaign02._exact_reload_gate(
        trainer, Path(final_checkpoint["path"]), campaign03.campaign03_ppo_config()
    )
    if final_reload["verdict"] != "PASS_GREEN":
        raise RuntimeError("final six-hour checkpoint exact reload failed")
    phase_c_summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": "COMPLETE",
        "reward_version": RIVAL2_REWARD_VERSION,
        "timer_started_utc": phase_c_start_utc,
        "start_iteration": phase_c_start_iteration,
        "start_agent_decision_samples": phase_c_start_samples,
        "final_iteration": trainer.iteration,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "additional_updates": trainer.iteration - phase_c_start_iteration,
        "additional_agent_decision_samples": trainer.total_agent_samples
        - phase_c_start_samples,
        "stop_elapsed_seconds_at_update_completion": phase_c_checkpoints[-1][
            "phase_c_elapsed_seconds_at_update_completion"
        ],
        "total_elapsed_seconds_after_final_evaluation_and_reload": (
            time.perf_counter() - phase_c_started
        ),
        "checkpoints": phase_c_checkpoints,
        "evaluation_labels": [item["checkpoint_label"] for item in phase_c_evaluations],
        "final_checkpoint": final_checkpoint,
        "final_checkpoint_reload": final_reload,
    }
    campaign01._write_json(args.work_dir / "phase_c_summary.json", phase_c_summary)

    run_summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": "COMPLETE",
        "stop_detail": (
            "completed Phase A threshold, exact V2-to-V1 transition, 239-update Phase B, and "
            "the first Phase C update at or after 21600 elapsed seconds"
        ),
        "resume_checkpoint_sha256": resume_authority["checkpoint_sha256"],
        "start_iteration": START_ITERATION,
        "start_agent_decision_samples": START_SAMPLES,
        "phase_a": phase_a_summary,
        "reward_transition": {
            "verdict": transition_result["verdict"],
            "source_checkpoint_sha256": acquisition_checkpoint["sha256"],
            "post_transition_checkpoint_sha256": transition_checkpoint["sha256"],
        },
        "phase_b": phase_b_summary,
        "phase_c": phase_c_summary,
        "final_iteration": trainer.iteration,
        "final_policy_version": trainer.policy_version,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "total_continuation_updates": trainer.iteration - START_ITERATION,
        "total_additional_agent_decision_samples": trainer.total_agent_samples
        - START_SAMPLES,
        "curriculum_wall_seconds_including_all_evaluations": time.perf_counter()
        - curriculum_started,
        "runtime": campaign01._runtime_identity(args.device),
        "final_reward_version": trainer.env.reward_version,
        "final_contract_hashes": dict(trainer.env.contract_hashes),
        "final_historical_policy_versions": list(trainer.opponent_pool.versions),
        "final_checkpoint_reload": final_reload,
        "preflight_regression_lint_ceremony_run": False,
        "viewer_built": False,
        "v06_begun": False,
    }
    campaign01._write_json(args.work_dir / "run_summary.json", run_summary)
    print(
        f"overnight COMPLETE update={trainer.iteration} samples={trainer.total_agent_samples} "
        f"phase_c_elapsed={phase_c_summary['stop_elapsed_seconds_at_update_completion']:.3f} "
        f"final_sha256={final_checkpoint['sha256']}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configuration = frozen_configuration()
    configuration_path = args.work_dir / "config_frozen_before_training.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise RuntimeError("existing overnight configuration differs")
    else:
        campaign01._write_json(configuration_path, configuration)
    campaign01._initialize_runtime(args.device)
    resume_authority = checkpoint_authority(configuration)
    campaign01._write_json(args.work_dir / "resume_authority.json", resume_authority)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    return run_curriculum(args, configuration, resume_authority, geometry, meshes)


if __name__ == "__main__":
    raise SystemExit(main())
