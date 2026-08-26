"""Continue the exact Rival 2.0 Campaign 03 state through the Campaign 04 1B boundary."""

from __future__ import annotations

import argparse
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
    REWARD_V2_CONTRACT_HASH,
    RIVAL2_REWARD_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_training import Rival2Trainer

AUTHORIZED_HEAD = "d8ffef9de46c96a41b9c1a4a73ebd7606d94e7f4"
CAMPAIGN03_CLOSEOUT = "67b51452df98696a54f4465ea83924c6b9e75b4d"
RESUME_CHECKPOINT = Path(
    "checkpoints/rival2/campaign03/rival2_campaign03_100m_resume.pt"
)
RESUME_CHECKPOINT_SHA256 = (
    "A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991"
)
START_ITERATION = 12
START_SAMPLES = 100_663_296
TARGET_ITERATION = 120
TARGET_SAMPLES = 1_006_632_960
CAMPAIGN04_WORLDS = 131072
CAMPAIGN04_SEED = 20260826
EVALUATION_SEED = 920260826
EVALUATION_WORLDS = 4096
CHECKPOINT_THRESHOLDS = (
    ("250m", 30, 251_658_240),
    ("500m", 60, 503_316_480),
    ("750m", 90, 754_974_720),
    ("1b", 120, 1_006_632_960),
)
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def frozen_configuration() -> dict[str, Any]:
    campaign03_config = json.loads(
        Path("results/rival2/campaign03/config.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "Rival 2.0 Campaign 04 Reward V2 long-run continuation",
        "authorized_head": AUTHORIZED_HEAD,
        "campaign03_closeout": CAMPAIGN03_CLOSEOUT,
        "resume_checkpoint": RESUME_CHECKPOINT.as_posix(),
        "resume_checkpoint_sha256": RESUME_CHECKPOINT_SHA256,
        "start_iteration": START_ITERATION,
        "start_agent_decision_samples": START_SAMPLES,
        "target_iteration": TARGET_ITERATION,
        "target_agent_decision_samples": TARGET_SAMPLES,
        "stop_rule": "stop after update 120; never run update 121",
        "campaign_seed": CAMPAIGN04_SEED,
        "worlds": CAMPAIGN04_WORLDS,
        "policy_config": campaign03_config["policy_config"],
        "policy_config_hash": campaign03_config["policy_config_hash"],
        "ppo_config": campaign03_config["ppo_config"],
        "ppo_config_hash": campaign03_config["ppo_config_hash"],
        "self_play_config": campaign03_config["self_play_config"],
        "reward_version": RIVAL2_REWARD_V2_VERSION,
        "reward_v2_contract_hash": REWARD_V2_CONTRACT_HASH,
        "active_contract_hashes": contract_hashes_for_reward(RIVAL2_REWARD_V2_VERSION),
        "checkpoint_thresholds": [
            {"label": label, "iteration": iteration, "samples": samples}
            for label, iteration, samples in CHECKPOINT_THRESHOLDS
        ],
        "evaluation": {
            "seed": EVALUATION_SEED,
            "worlds": EVALUATION_WORLDS,
            "mode": "ordinary stochastic self-play",
            "episode_scope": "first completed episode per world",
            "labels": [label for label, _, _ in CHECKPOINT_THRESHOLDS],
            "campaign03_100m_baseline": "reuse published result; do not rerun",
        },
        "behavioral_trend_rule_frozen_before_continuation": {
            "axes": {
                "touches_per_simulated_minute": "higher is better",
                "no_touch_truncated_fraction": "lower is better",
            },
            "continuing": "both 1B axes improve versus 750M",
            "degrading": "both 1B axes worsen versus 750M",
            "flattening": "the two 1B axes are mixed or unchanged versus 750M",
            "no_numeric_success_threshold": True,
        },
        "prohibited": [
            "preflight or capacity checks",
            "reward smoke or initialization evaluation",
            "world-count sweep",
            "inherited parity or regression suites",
            "extra checkpoint or evaluation",
            "post-run pytest, Ruff, or compileall ceremony",
            "viewer implementation",
            "v0.6 RocketSim or RLBot transfer work",
        ],
    }


def checkpoint_authority(configuration: dict[str, Any]) -> dict[str, Any]:
    if campaign01._git("rev-parse", "HEAD") != AUTHORIZED_HEAD:
        raise RuntimeError("Campaign 04 must start from the authorized HEAD")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", CAMPAIGN03_CLOSEOUT, "HEAD"],
        check=True,
        capture_output=True,
    )
    if not RESUME_CHECKPOINT.is_file():
        raise RuntimeError("Campaign 03 final resume checkpoint is missing")
    actual_sha256 = campaign01._sha256_file(RESUME_CHECKPOINT)
    payload = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=False)
    historical_versions = [int(entry["version"]) for entry in payload["historical_opponents"]]
    checks = {
        "campaign03_is_ancestor": True,
        "checkpoint_sha256_exact": actual_sha256 == RESUME_CHECKPOINT_SHA256,
        "checkpoint_format_exact": payload["format"] == "RIVAL2_CHECKPOINT_V1",
        "reward_version_exact": payload["reward_version"] == RIVAL2_REWARD_V2_VERSION,
        "contract_hashes_exact": payload["contract_hashes"]
        == configuration["active_contract_hashes"],
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
        == [CAMPAIGN04_WORLDS],
        "historical_versions_exact": historical_versions == [0, 2, 3, 6, 12],
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
        raise RuntimeError(f"Campaign 04 resume authority failed: {checks}")
    return result


def _save_checkpoint(label: str, trainer: Rival2Trainer, work_dir: Path) -> dict[str, Any]:
    path = work_dir / "checkpoints" / f"rival2_campaign04_{label}_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return campaign01._checkpoint_record(path, label, trainer)


@torch.no_grad()
def _evaluate_checkpoint(
    *,
    label: str,
    trainer: Rival2Trainer,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
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
        reward_version=RIVAL2_REWARD_V2_VERSION,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "checkpoint_label": label,
        "iteration": trainer.iteration,
        "agent_decision_samples": trainer.total_agent_samples,
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_worlds": EVALUATION_WORLDS,
        "mode": "ordinary stochastic self-play",
        "result": mode,
        "wall_seconds": time.perf_counter() - started,
        "verdict": mode["verdict"],
    }


def run_campaign(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    resume_authority: dict[str, Any],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> int:
    torch.manual_seed(CAMPAIGN04_SEED)
    torch.cuda.manual_seed(CAMPAIGN04_SEED)
    kickoff_selector = (
        np.arange(CAMPAIGN04_WORLDS, dtype=np.int32) + CAMPAIGN04_SEED
    ) % 5
    env = Rival2Env(
        CAMPAIGN04_WORLDS,
        args.collision_dir,
        device=args.device,
        seed=CAMPAIGN04_SEED,
        reward_version=RIVAL2_REWARD_V2_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
    )
    trainer = Rival2Trainer(
        env,
        ppo_config=campaign03.campaign03_ppo_config(),
        seed=CAMPAIGN04_SEED,
    )
    trainer.load_checkpoint(RESUME_CHECKPOINT)
    loaded_checks = {
        "resume_authority_green": resume_authority["verdict"] == "PASS_GREEN",
        "iteration_exact": trainer.iteration == START_ITERATION,
        "policy_version_exact": trainer.policy_version == START_ITERATION,
        "sample_count_exact": trainer.total_agent_samples == START_SAMPLES,
        "historical_versions_exact": trainer.opponent_pool.versions == [0, 2, 3, 6, 12],
        "reward_v2_active": env.reward_version == RIVAL2_REWARD_V2_VERSION,
        "contract_hashes_exact": env.contract_hashes
        == configuration["active_contract_hashes"],
        "entropy_coefficient_zero": trainer.ppo_config.entropy_coefficient == 0.0,
    }
    if not all(loaded_checks.values()):
        raise RuntimeError(f"Campaign 04 loaded-state check failed: {loaded_checks}")
    campaign01._write_json(
        args.work_dir / "loaded_state.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": campaign01._utc_now(),
            "checks": loaded_checks,
            "verdict": "PASS_GREEN",
        },
    )

    threshold_index = 0
    checkpoints: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    campaign_started = time.perf_counter()
    execution_status = "COMPLETE"
    stop_detail = "completed update 120, the first completed update crossing 1B samples"

    while trainer.iteration < TARGET_ITERATION:
        policy_version_before = trainer.policy_version
        samples_before = trainer.total_agent_samples
        env.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(args.device)
        iteration_started = time.perf_counter()
        rollout, metrics = trainer.train_iteration()
        torch.cuda.synchronize()
        iteration_seconds = time.perf_counter() - iteration_started
        point = campaign03._training_point(
            trainer=trainer,
            rollout=rollout,
            metrics=metrics,
            policy_version_before=policy_version_before,
            samples_before=samples_before,
            iteration_seconds=iteration_seconds,
        )
        curve.append(point)
        campaign01._write_json(args.work_dir / "training_curve.json", curve)
        values = point["integrity"]["metrics"]
        print(
            f"campaign04 update={trainer.iteration} samples={trainer.total_agent_samples} "
            f"seconds={iteration_seconds:.3f} kl={values['approx_kl']:.6f} "
            f"clip={values['clip_fraction']:.6f} integrity={point['integrity']['verdict']}",
            flush=True,
        )
        if point["integrity"]["verdict"] != "PASS_GREEN":
            execution_status = "STOP_NUMERICAL"
            stop_detail = f"integrity failure at update {trainer.iteration}"
            trainer.save_checkpoint(
                args.work_dir / "checkpoints" / f"integrity_failure_{trainer.iteration}.pt"
            )
            break
        del rollout, metrics
        gc.collect()

        if threshold_index < len(CHECKPOINT_THRESHOLDS):
            label, expected_iteration, expected_samples = CHECKPOINT_THRESHOLDS[threshold_index]
            if trainer.total_agent_samples >= expected_samples:
                if trainer.iteration != expected_iteration:
                    raise RuntimeError(f"{label} threshold crossed on an unexpected update")
                trainer.add_historical_snapshot()
                checkpoint = _save_checkpoint(label, trainer, args.work_dir)
                checkpoints.append(checkpoint)
                campaign01._write_json(
                    args.work_dir / "checkpoints.json", {"checkpoints": checkpoints}
                )
                print(
                    f"campaign04 checkpoint={label} sha256={checkpoint['sha256']}",
                    flush=True,
                )
                evaluation = _evaluate_checkpoint(
                    label=label,
                    trainer=trainer,
                    collision_dir=args.collision_dir,
                    geometry=geometry,
                    meshes=meshes,
                    device=args.device,
                )
                evaluations.append(evaluation)
                campaign01._write_json(
                    args.work_dir / f"evaluation_{label}.json", evaluation
                )
                print(
                    f"campaign04 evaluation={label} verdict={evaluation['verdict']} "
                    f"touches_per_minute={evaluation['result']['touches_per_simulated_minute']:.6f} "
                    f"no_touch={evaluation['result']['no_touch_truncated_fraction']:.6f}",
                    flush=True,
                )
                if evaluation["verdict"] != "PASS_GREEN":
                    execution_status = "STOP_NUMERICAL"
                    stop_detail = f"evaluation integrity failure at {label}"
                    break
                threshold_index += 1

    if execution_status == "COMPLETE":
        exact_stop = (
            trainer.iteration == TARGET_ITERATION
            and trainer.policy_version == TARGET_ITERATION
            and trainer.total_agent_samples == TARGET_SAMPLES
            and threshold_index == len(CHECKPOINT_THRESHOLDS)
        )
        if not exact_stop:
            execution_status = "STOP_ARCHITECTURAL"
            stop_detail = "Campaign 04 did not reach the exact 1B boundary"

    final_checkpoint = Path(checkpoints[-1]["path"]) if checkpoints else RESUME_CHECKPOINT
    reload_gate = (
        campaign02._exact_reload_gate(
            trainer, final_checkpoint, campaign03.campaign03_ppo_config()
        )
        if execution_status == "COMPLETE"
        else {"verdict": "NOT_RUN", "reason": stop_detail}
    )
    if execution_status == "COMPLETE" and reload_gate["verdict"] != "PASS_GREEN":
        execution_status = "STOP_ARCHITECTURAL"
        stop_detail = "final Campaign 04 checkpoint exact reload failed"

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": execution_status,
        "stop_detail": stop_detail,
        "resume_checkpoint_sha256": resume_authority["checkpoint_sha256"],
        "start_iteration": START_ITERATION,
        "start_agent_decision_samples": START_SAMPLES,
        "final_iteration": trainer.iteration,
        "final_policy_version": trainer.policy_version,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "continuation_updates": trainer.iteration - START_ITERATION,
        "campaign_wall_seconds_including_evaluations": time.perf_counter() - campaign_started,
        "checkpoints": checkpoints,
        "evaluations": [
            {
                "label": evaluation["checkpoint_label"],
                "iteration": evaluation["iteration"],
                "agent_decision_samples": evaluation["agent_decision_samples"],
                "verdict": evaluation["verdict"],
            }
            for evaluation in evaluations
        ],
        "final_checkpoint_reload": reload_gate,
        "runtime": campaign01._runtime_identity(args.device),
        "reward_version": RIVAL2_REWARD_V2_VERSION,
        "reward_v2_contract_hash": REWARD_V2_CONTRACT_HASH,
        "active_contract_hashes": configuration["active_contract_hashes"],
        "ppo_config": configuration["ppo_config"],
        "authorized_evaluation_count": len(evaluations),
        "viewer_built": False,
        "v06_begun": False,
    }
    campaign01._write_json(args.work_dir / "run_summary.json", result)
    return 0 if execution_status == "COMPLETE" else 4


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configuration = frozen_configuration()
    configuration_path = args.work_dir / "config_frozen_before_training.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise RuntimeError("existing Campaign 04 configuration differs")
    else:
        campaign01._write_json(configuration_path, configuration)
    campaign01._initialize_runtime(args.device)
    resume_authority = checkpoint_authority(configuration)
    campaign01._write_json(args.work_dir / "resume_authority.json", resume_authority)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    return run_campaign(args, configuration, resume_authority, geometry, meshes)


if __name__ == "__main__":
    raise SystemExit(main())
