"""Execute Rival 2.0 Campaign 03's direct Reward V2 training authority."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmarks.run_rival2_campaign01 as campaign01
import benchmarks.run_rival2_campaign02 as campaign02
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.kernels.rival2 import NO_TOUCH_TIMEOUT_TICKS, PHYSICS_TICKS_PER_DECISION
from rivalsim.rival2_contracts import (
    APPROACH_DISTANCE_SCALE,
    CONTRACT_HASHES,
    OBS_DIM,
    OBS_FIELD_NAMES,
    REWARD_CONTRACT_HASH,
    REWARD_V2_CONTRACT,
    REWARD_V2_CONTRACT_HASH,
    RIVAL2_REWARD_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import Rival2PolicyConfig
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer

AUTHORIZED_HEAD = "0b385492523922dc666d32c75fc86eb60c0c0f4c"
CAMPAIGN02_CLOSEOUT = "816c66b455d253b0f563bb378e53316a09ffd48e"
CAMPAIGN03_WORLDS = 131072
CAMPAIGN03_SEED = 20260826
EVALUATION_SEED = 920260826
EVALUATION_WORLDS = 4096
TARGET_SAMPLES = 100_000_000
SNAPSHOT_THRESHOLDS = (
    ("000m", 0),
    ("010m", 10_000_000),
    ("025m", 25_000_000),
    ("050m", 50_000_000),
    ("100m", 100_000_000),
)
CHECKPOINT_LABELS = frozenset(("025m", "050m", "100m"))
CAMPAIGN02_FINAL = {
    "touches_per_simulated_minute": 0.291182,
    "goals_per_simulated_minute": 0.040362,
    "goal_terminated_fraction": 0.010254,
    "no_touch_truncated_fraction": 0.989746,
}
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def campaign03_ppo_config():
    """Return the unchanged Campaign 02 entropy-off PPO baseline."""

    return campaign02.campaign02_ppo_config()


def frozen_configuration() -> dict[str, Any]:
    policy = Rival2PolicyConfig()
    ppo = campaign03_ppo_config()
    self_play = Rival2SelfPlayConfig()
    active_hashes = contract_hashes_for_reward(RIVAL2_REWARD_V2_VERSION)
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "Rival 2.0 Campaign 03 direct reward-density training",
        "authorized_head": AUTHORIZED_HEAD,
        "campaign02_closeout": CAMPAIGN02_CLOSEOUT,
        "campaign_seed": CAMPAIGN03_SEED,
        "worlds": CAMPAIGN03_WORLDS,
        "target_agent_decision_samples": TARGET_SAMPLES,
        "stop_rule": "first completed PPO update at or above 100M samples",
        "policy_config": asdict(policy),
        "policy_config_hash": policy.content_hash,
        "ppo_config": asdict(ppo),
        "ppo_config_hash": ppo.content_hash,
        "self_play_config": asdict(self_play),
        "reward_v1_custody_hash": REWARD_CONTRACT_HASH,
        "reward_v2_contract": REWARD_V2_CONTRACT,
        "reward_v2_contract_hash": REWARD_V2_CONTRACT_HASH,
        "active_contract_hashes": active_hashes,
        "snapshot_thresholds": {label: threshold for label, threshold in SNAPSHOT_THRESHOLDS},
        "checkpoint_thresholds": {
            label: threshold
            for label, threshold in SNAPSHOT_THRESHOLDS
            if label in CHECKPOINT_LABELS
        },
        "evaluation": {
            "seed": EVALUATION_SEED,
            "worlds": EVALUATION_WORLDS,
            "mode": "ordinary stochastic self-play",
            "count": 1,
            "timing": "after the final 100M checkpoint only",
            "campaign02_final_reference": CAMPAIGN02_FINAL,
        },
        "launch_gate": "targeted GPU reward-sign/reset-leakage smoke only",
        "prohibited_before_training": [
            "capacity preflight",
            "initialization evaluation",
            "inherited simulator parity or regression gates",
            "world-count sweep",
            "held-out evaluation",
        ],
        "hard_boundaries": [
            "RIVAL2_REWARD_V2 approach term is the only learning-semantic change",
            "no curriculum, action mask, extra reward, tuning, or hyperparameter change",
            "no intermediate checkpoint evaluation",
            "no v0.6 RocketSim or RLBot transfer work",
        ],
    }


def verify_authority(configuration: dict[str, Any]) -> dict[str, bool]:
    if campaign01._git("rev-parse", "HEAD") != AUTHORIZED_HEAD:
        raise RuntimeError("Campaign 03 must start from the authorized HEAD")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", CAMPAIGN02_CLOSEOUT, "HEAD"],
        check=True,
        capture_output=True,
    )
    campaign02_config = json.loads(
        Path("results/rival2/campaign02/config.json").read_text(encoding="utf-8")
    )
    checks = {
        "campaign02_is_ancestor": True,
        "v1_reward_hash_preserved": campaign02_config["contract_hashes"][
            "RIVAL2_REWARD_V1"
        ]
        == REWARD_CONTRACT_HASH,
        "v1_contract_bundle_preserved": CONTRACT_HASHES
        == campaign01.EXPECTED_CONTRACT_HASHES,
        "ppo_matches_campaign02": configuration["ppo_config"]
        == campaign02_config["ppo_config"],
        "policy_matches_campaign02": configuration["policy_config"]
        == campaign02_config["policy_config"],
        "self_play_matches_campaign02": configuration["self_play_config"]
        == campaign02_config["self_play_config"],
        "worlds_match_campaign02": configuration["worlds"]
        == campaign02_config["worlds"]
        == CAMPAIGN03_WORLDS,
        "reward_v2_is_active": configuration["active_contract_hashes"]
        == contract_hashes_for_reward(RIVAL2_REWARD_V2_VERSION),
        "entropy_is_zero": configuration["ppo_config"]["entropy_coefficient"] == 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Campaign 03 authority check failed: {checks}")
    return checks


@torch.no_grad()
def targeted_reward_smoke(
    *,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
) -> dict[str, Any]:
    """Run only the handoff-authorized Reward V2 GPU launch gate."""

    env = Rival2Env(
        1,
        collision_dir,
        device=device,
        seed=CAMPAIGN03_SEED,
        reward_version=RIVAL2_REWARD_V2_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=np.asarray([0], dtype=np.int32),
    )
    relative_start = OBS_FIELD_NAMES.index("relative.ball_position.x")
    relative = slice(relative_start, relative_start + 3)
    synthetic_before = torch.zeros((3, 2, OBS_DIM), dtype=torch.float32, device=device)
    synthetic_after = torch.zeros_like(synthetic_before)
    synthetic_before[0, :, relative] = torch.tensor(
        (2000.0 / 4096.0, 0.0, 0.0), device=device
    )
    synthetic_after[0, :, relative] = torch.tensor(
        (1000.0 / 4096.0, 0.0, 0.0), device=device
    )
    synthetic_before[1, :, relative] = torch.tensor(
        (1000.0 / 4096.0, 0.0, 0.0), device=device
    )
    synthetic_after[1, :, relative] = torch.tensor(
        (2000.0 / 4096.0, 0.0, 0.0), device=device
    )
    unchanged = torch.tensor(
        (1200.0 / 4096.0, -2200.0 / 5120.0, 300.0 / 2044.0), device=device
    )
    synthetic_before[2, :, relative] = unchanged
    synthetic_after[2, :, relative] = unchanged
    synthetic_approach = env.bridge.approach_reward(synthetic_before, synthetic_after)

    no_touch = env.bridge.views["rival2.no_touch_ticks"]
    no_touch.fill_(NO_TOUCH_TIMEOUT_TICKS - PHYSICS_TICKS_PER_DECISION)
    actual_before = env.observation.clone()
    action = torch.zeros((1, 2, 8), dtype=torch.float32, device=device)
    transition = env.step(action)
    pre_reset_expected = env.bridge.approach_reward(
        actual_before, transition.transition_observation
    )
    post_reset_wrong = env.bridge.approach_reward(actual_before, transition.observation)
    blue_progress = 0.5 * (
        transition.transition_observation[:, 0, 1] - actual_before[:, 0, 1]
    )
    v1_reward = torch.stack((blue_progress, -blue_progress), dim=1)
    integrated_approach = transition.reward - v1_reward
    touch_index = OBS_FIELD_NAMES.index("lifecycle.self_touch_event")
    demo_index = OBS_FIELD_NAMES.index("lifecycle.self_demoed_event")

    checks = {
        "decreasing_distance_positive": bool((synthetic_approach[0] > 0.0).all().item()),
        "increasing_distance_negative": bool((synthetic_approach[1] < 0.0).all().item()),
        "unchanged_distance_zero": bool(
            torch.allclose(synthetic_approach[2], torch.zeros(2, device=device), atol=1e-7)
        ),
        "reset_transition_triggered": bool(
            transition.reset_mask.all().item() and transition.truncated.all().item()
        ),
        "reset_case_has_no_v1_event_terms": bool(
            (transition.transition_observation[..., touch_index] == 0.0).all().item()
            and (transition.transition_observation[..., demo_index] == 0.0).all().item()
            and not transition.terminated.any().item()
        ),
        "integrated_reward_uses_pre_reset_delta": bool(
            torch.allclose(integrated_approach, pre_reset_expected, atol=2e-7, rtol=0.0)
        ),
        "post_reset_motion_would_change_delta": bool(
            not torch.allclose(post_reset_wrong, pre_reset_expected, atol=1e-7, rtol=0.0)
        ),
        "tensors_finite": bool(
            torch.isfinite(synthetic_approach).all().item()
            and torch.isfinite(transition.reward).all().item()
            and torch.isfinite(pre_reset_expected).all().item()
        ),
        "tensors_device_resident": all(
            tensor.is_cuda
            for tensor in (
                synthetic_before,
                synthetic_after,
                synthetic_approach,
                actual_before,
                transition.transition_observation,
                transition.observation,
                transition.reward,
                pre_reset_expected,
            )
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "gate": "Campaign 03 targeted GPU reward-sign/reset-leakage smoke",
        "reward_version": RIVAL2_REWARD_V2_VERSION,
        "reward_v2_contract_hash": REWARD_V2_CONTRACT_HASH,
        "device": str(synthetic_approach.device),
        "approach_scale": APPROACH_DISTANCE_SCALE,
        "synthetic_approach": synthetic_approach.cpu().tolist(),
        "reset_case": {
            "pre_reset_expected": pre_reset_expected.cpu().tolist(),
            "integrated_reward_minus_v1": integrated_approach.cpu().tolist(),
            "post_reset_contaminated_delta": post_reset_wrong.cpu().tolist(),
        },
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    del env
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _save_checkpoint(label: str, trainer: Rival2Trainer, work_dir: Path) -> dict[str, Any]:
    path = work_dir / "checkpoints" / f"rival2_campaign03_{label}_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return campaign01._checkpoint_record(path, label, trainer)


def _training_point(
    *,
    trainer: Rival2Trainer,
    rollout: Any,
    metrics: dict[str, torch.Tensor],
    policy_version_before: int,
    samples_before: int,
    iteration_seconds: float,
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
    integrity["checks"].update(
        {
            "reward_v2_active": trainer.env.reward_version == RIVAL2_REWARD_V2_VERSION,
            "reward_v2_contract_exact": trainer.env.contract_hashes
            == contract_hashes_for_reward(RIVAL2_REWARD_V2_VERSION),
            "entropy_coefficient_exact_zero": trainer.ppo_config.entropy_coefficient == 0.0,
            "total_loss_excludes_entropy": loss_identity_error <= 2e-7,
            "zero_hot_h2d": transfer["h2d"] == 0,
            "zero_hot_d2h": transfer["d2h"] == 0,
        }
    )
    integrity["hot_path_transfer_bytes"] = transfer
    integrity["verdict"] = "PASS_GREEN" if all(integrity["checks"].values()) else "FAIL_RED"
    return {
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


def run_campaign(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> int:
    torch.manual_seed(CAMPAIGN03_SEED)
    torch.cuda.manual_seed(CAMPAIGN03_SEED)
    kickoff_selector = (
        np.arange(CAMPAIGN03_WORLDS, dtype=np.int32) + CAMPAIGN03_SEED
    ) % 5
    env = Rival2Env(
        CAMPAIGN03_WORLDS,
        args.collision_dir,
        device=args.device,
        seed=CAMPAIGN03_SEED,
        reward_version=RIVAL2_REWARD_V2_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
    )
    trainer = Rival2Trainer(
        env,
        ppo_config=campaign03_ppo_config(),
        seed=CAMPAIGN03_SEED,
    )
    initialization_state = {
        name: tensor.detach().cpu().clone() for name, tensor in trainer.model.state_dict().items()
    }
    initialization_sha256 = campaign01._state_dict_sha256(initialization_state)
    del initialization_state
    trainer.add_historical_snapshot()
    next_snapshot_index = 1
    checkpoints: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    execution_status = "COMPLETE"
    stop_detail = "first completed PPO update crossing 100M agent decision samples"
    campaign_started = time.perf_counter()

    while trainer.total_agent_samples < TARGET_SAMPLES:
        policy_version_before = trainer.policy_version
        samples_before = trainer.total_agent_samples
        env.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(args.device)
        iteration_started = time.perf_counter()
        rollout, metrics = trainer.train_iteration()
        torch.cuda.synchronize()
        iteration_seconds = time.perf_counter() - iteration_started
        point = _training_point(
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
            f"campaign03 update={trainer.iteration} samples={trainer.total_agent_samples} "
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

        while (
            next_snapshot_index < len(SNAPSHOT_THRESHOLDS)
            and trainer.total_agent_samples >= SNAPSHOT_THRESHOLDS[next_snapshot_index][1]
        ):
            label = SNAPSHOT_THRESHOLDS[next_snapshot_index][0]
            trainer.add_historical_snapshot()
            if label in CHECKPOINT_LABELS:
                checkpoint = _save_checkpoint(label, trainer, args.work_dir)
                checkpoints.append(checkpoint)
                print(
                    f"campaign03 checkpoint={label} sha256={checkpoint['sha256']}",
                    flush=True,
                )
            next_snapshot_index += 1

    if execution_status != "COMPLETE":
        campaign01._write_json(
            args.work_dir / "run_summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_utc": campaign01._utc_now(),
                "execution_status": execution_status,
                "stop_detail": stop_detail,
                "final_iteration": trainer.iteration,
                "final_agent_decision_samples": trainer.total_agent_samples,
                "no_v06_work": True,
            },
        )
        return 4

    final_checkpoint = Path(checkpoints[-1]["path"])
    reload_gate = campaign02._exact_reload_gate(
        trainer, final_checkpoint, campaign03_ppo_config()
    )
    final_model = copy.deepcopy(trainer.model).to(args.device).eval().requires_grad_(False)
    training_wall_seconds = time.perf_counter() - campaign_started
    del trainer, env
    gc.collect()
    torch.cuda.empty_cache()

    evaluation_started = time.perf_counter()
    evaluation_mode = campaign01._evaluate_mode(
        mode="stochastic_self_play",
        checkpoint_model=final_model,
        initialization_model=final_model,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        reward_version=RIVAL2_REWARD_V2_VERSION,
    )
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "checkpoint_label": "100m",
        "agent_decision_samples": checkpoints[-1]["agent_decision_samples"],
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_worlds": EVALUATION_WORLDS,
        "mode": "ordinary stochastic self-play",
        "result": evaluation_mode,
        "wall_seconds": time.perf_counter() - evaluation_started,
        "verdict": evaluation_mode["verdict"],
    }
    campaign01._write_json(args.work_dir / "final_evaluation.json", evaluation)
    campaign01._write_json(
        args.work_dir / "checkpoints.json", {"checkpoints": checkpoints}
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": "COMPLETE",
        "stop_detail": stop_detail,
        "selected_worlds": CAMPAIGN03_WORLDS,
        "campaign_seed": CAMPAIGN03_SEED,
        "initialization_model_sha256": initialization_sha256,
        "final_agent_decision_samples": checkpoints[-1]["agent_decision_samples"],
        "final_iteration": checkpoints[-1]["iteration"],
        "first_update_crossing_target": checkpoints[-1]["iteration"],
        "campaign_training_wall_seconds": training_wall_seconds,
        "checkpoints": checkpoints,
        "final_checkpoint_reload": reload_gate,
        "single_final_evaluation_verdict": evaluation["verdict"],
        "runtime": campaign01._runtime_identity(args.device),
        "reward_version": RIVAL2_REWARD_V2_VERSION,
        "reward_v2_contract_hash": REWARD_V2_CONTRACT_HASH,
        "active_contract_hashes": configuration["active_contract_hashes"],
        "ppo_config": configuration["ppo_config"],
        "no_preflight_or_intermediate_evaluations_run": True,
        "no_v06_work": True,
    }
    campaign01._write_json(args.work_dir / "run_summary.json", result)
    return 0 if evaluation["verdict"] == "PASS_GREEN" else 5


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configuration = frozen_configuration()
    configuration_path = args.work_dir / "config_frozen_before_training.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise RuntimeError("existing Campaign 03 configuration differs")
    else:
        campaign01._write_json(configuration_path, configuration)
    campaign01._initialize_runtime(args.device)
    authority_checks = verify_authority(configuration)
    campaign01._write_json(args.work_dir / "authority_checks.json", authority_checks)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    smoke = targeted_reward_smoke(
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
    )
    campaign01._write_json(args.work_dir / "reward_smoke.json", smoke)
    print(f"campaign03 targeted_reward_smoke={smoke['verdict']}", flush=True)
    if smoke["verdict"] != "PASS_GREEN":
        return 3
    return run_campaign(args, configuration, geometry, meshes)


if __name__ == "__main__":
    raise SystemExit(main())
