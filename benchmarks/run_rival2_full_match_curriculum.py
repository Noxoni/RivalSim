"""Run the mechanics-corrective Rival 2.0 curriculum in complete 1v1 matches."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmarks.run_rival2_campaign01 as campaign01
import benchmarks.run_rival2_campaign03 as campaign03
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    FULL_MATCH_EPISODE_CONTRACT_HASH,
    REWARD_GOAL_ONLY_CONTRACT_HASH,
    REWARD_V2_CONTRACT_HASH,
    RIVAL2_FULL_MATCH_EPISODE_VERSION,
    RIVAL2_REWARD_GOAL_ONLY_VERSION,
    RIVAL2_REWARD_V2_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.rival2_policy import Rival2PolicyConfig, sample_hybrid_action
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer

MECHANICS_FIX_COMMIT = "0fa27d2fd846b4e8d4b9955a0ad88c2c2af91037"
ROCKETSIM_COMMIT = "c2baacb8f4b441dd8505e63c2aeb5a1679b60b02"
ROCKETSIM_BINDING_COMMIT = "2da51b1dac7b8127127613a5ff30e490bdd70dd8"
MECHANICS_EVIDENCE = Path(
    "results/rival2/mechanics_correction/movement_mechanics_parity.json"
)
AUTHORITY = Path("handoff/rival2-mechanics-full-match/README.md")

WORLDS = 131_072
CAMPAIGN_SEED = 20_260_826
ROLLOUT_AGENT_SAMPLES = 8_388_608
EVALUATION_WORLDS = 4_096
EVALUATION_SEED = 920_260_826
PHASE_A_EVALUATION_INTERVAL = 30
PHASE_A_THRESHOLD = 0.01
PHASE_A_REQUIRED_CONSECUTIVE = 2
PHASE_B_UPDATES = 239
PHASE_B_ADDITIONAL_SAMPLES = 2_004_877_312
PHASE_B_EVALUATION_OFFSETS = (60, 120, 180, 239)
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/rival2/full_match_curriculum"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    return campaign01._sha256_file(path)


def frozen_configuration() -> dict[str, Any]:
    policy = Rival2PolicyConfig()
    ppo = campaign03.campaign03_ppo_config()
    self_play = Rival2SelfPlayConfig()
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY.as_posix(),
        "mechanics_fix_commit": MECHANICS_FIX_COMMIT,
        "rocketsim_commit": ROCKETSIM_COMMIT,
        "rocketsim_binding_commit": ROCKETSIM_BINDING_COMMIT,
        "campaign_seed": CAMPAIGN_SEED,
        "worlds": WORLDS,
        "rollout_agent_decision_samples": ROLLOUT_AGENT_SAMPLES,
        "policy_config": asdict(policy),
        "policy_config_hash": policy.content_hash,
        "ppo_config": asdict(ppo),
        "ppo_config_hash": ppo.content_hash,
        "self_play_config": asdict(self_play),
        "episode_version": RIVAL2_FULL_MATCH_EPISODE_VERSION,
        "episode_contract_hash": FULL_MATCH_EPISODE_CONTRACT_HASH,
        "all_training_episode_mode": "complete standard five-minute 1v1 match",
        "phase_a": {
            "reward_version": RIVAL2_REWARD_V2_VERSION,
            "reward_contract_hash": REWARD_V2_CONTRACT_HASH,
            "evaluation_interval_updates": PHASE_A_EVALUATION_INTERVAL,
            "counterfactual_no_touch_kickoff_segment_threshold": PHASE_A_THRESHOLD,
            "required_consecutive_passing_evaluations": (
                PHASE_A_REQUIRED_CONSECUTIVE
            ),
            "sample_cap": None,
        },
        "reward_transition": {
            "source": RIVAL2_REWARD_V2_VERSION,
            "destination": RIVAL2_REWARD_GOAL_ONLY_VERSION,
            "destination_contract_hash": REWARD_GOAL_ONLY_CONTRACT_HASH,
            "removed": [
                "approach",
                "ball progress",
                "touch",
                "demolition",
            ],
            "retained": ["goal +10/-10"],
        },
        "phase_b": {
            "additional_updates": PHASE_B_UPDATES,
            "additional_agent_decision_samples": PHASE_B_ADDITIONAL_SAMPLES,
            "evaluation_update_offsets": list(PHASE_B_EVALUATION_OFFSETS),
        },
        "phase_c": {
            "bound": "one fresh complete standard match per resident world",
            "worlds": WORLDS,
            "active_sample_mask": (
                "transitions after a world's first match completion are excluded from PPO"
            ),
            "stop_rule": "every world has completed exactly one counted match",
            "wall_clock_bound": None,
        },
        "evaluation": {
            "seed": EVALUATION_SEED,
            "worlds": EVALUATION_WORLDS,
            "mode": "stochastic current-policy self-play",
            "scope": "first complete standard match per held-out world",
            "overtime": "unbounded sudden death",
            "hard_timeout": None,
        },
    }


def verify_launch(configuration: dict[str, Any]) -> dict[str, Any]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", MECHANICS_FIX_COMMIT, "HEAD"],
        check=True,
    )
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    tracked_clean = subprocess.run(["git", "diff", "--quiet"]).returncode == 0
    staged_clean = (
        subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0
    )
    mechanics = json.loads(MECHANICS_EVIDENCE.read_text(encoding="utf-8"))
    checks = {
        "mechanics_fix_is_ancestor": True,
        "head_pushed_to_origin_main": head == origin,
        "tracked_worktree_clean": tracked_clean,
        "index_clean": staged_clean,
        "authority_present": AUTHORITY.is_file(),
        "mechanics_gate_green": mechanics["status"] == "PASS",
        "mechanics_gate_no_failures": mechanics["failures"] == [],
        "mechanics_case_count_exact": mechanics["case_count"] == 24,
        "rocketsim_commit_exact": (
            mechanics["authority"]["rocketsim_commit"] == ROCKETSIM_COMMIT
        ),
        "rocketsim_binding_commit_exact": (
            mechanics["authority"]["rocketsim_binding_commit"]
            == ROCKETSIM_BINDING_COMMIT
        ),
        "full_match_reward_v2_contract_exact": (
            contract_hashes_for_reward(
                RIVAL2_REWARD_V2_VERSION,
                RIVAL2_FULL_MATCH_EPISODE_VERSION,
            )
            == {
                "RIVAL2_OBS_V1": (
                    contract_hashes_for_reward(
                        RIVAL2_REWARD_V2_VERSION,
                        RIVAL2_FULL_MATCH_EPISODE_VERSION,
                    )["RIVAL2_OBS_V1"]
                ),
                "RIVAL2_ACTION_V1": (
                    contract_hashes_for_reward(
                        RIVAL2_REWARD_V2_VERSION,
                        RIVAL2_FULL_MATCH_EPISODE_VERSION,
                    )["RIVAL2_ACTION_V1"]
                ),
                RIVAL2_REWARD_V2_VERSION: REWARD_V2_CONTRACT_HASH,
                RIVAL2_FULL_MATCH_EPISODE_VERSION: (
                    FULL_MATCH_EPISODE_CONTRACT_HASH
                ),
            }
        ),
        "ppo_entropy_zero": configuration["ppo_config"]["entropy_coefficient"]
        == 0.0,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "head": head,
        "origin_main": origin,
        "mechanics_evidence_sha256": _sha256(MECHANICS_EVIDENCE),
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"full-match campaign launch gate failed: {checks}")
    return result


def _checkpoint_record(
    path: Path, label: str, trainer: Rival2Trainer
) -> dict[str, Any]:
    trainer.save_checkpoint(path)
    return {
        "label": label,
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "reward_version": trainer.env.reward_version,
        "episode_version": trainer.env.episode_version,
        "contract_hashes": dict(trainer.env.contract_hashes),
        "historical_policy_versions": list(trainer.opponent_pool.versions),
    }


def _save_checkpoint(
    label: str, trainer: Rival2Trainer, work_dir: Path
) -> dict[str, Any]:
    path = work_dir / "checkpoints" / f"rival2_full_match_{label}_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return _checkpoint_record(path, label, trainer)


def _checkpoint_audit(record: dict[str, Any], trainer: Rival2Trainer) -> dict[str, Any]:
    path = Path(record["path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    current = trainer.checkpoint_payload()
    model_exact = all(
        torch.equal(payload["model"][name], value.detach().cpu())
        for name, value in trainer.model.state_dict().items()
    )
    checks = {
        "sha256_exact": _sha256(path) == record["sha256"],
        "format_exact": payload["format"] == "RIVAL2_CHECKPOINT_V1",
        "model_exact": model_exact,
        "optimizer_state_present": bool(payload["optimizer"]["state"]),
        "iteration_exact": payload["iteration"] == trainer.iteration,
        "policy_version_exact": payload["policy_version"] == trainer.policy_version,
        "sample_count_exact": (
            payload["total_agent_samples"] == trainer.total_agent_samples
        ),
        "reward_exact": payload["reward_version"] == trainer.env.reward_version,
        "episode_exact": (
            payload["episode_version"] == RIVAL2_FULL_MATCH_EPISODE_VERSION
        ),
        "contracts_exact": payload["contract_hashes"] == trainer.env.contract_hashes,
        "policy_rng_exact": torch.equal(
            payload["policy_generator_state"],
            current["policy_generator_state"].cpu(),
        ),
        "opponent_rng_exact": torch.equal(
            payload["opponent_generator_state"],
            current["opponent_generator_state"].cpu(),
        ),
        "historical_versions_exact": [
            int(item["version"]) for item in payload["historical_opponents"]
        ]
        == trainer.opponent_pool.versions,
    }
    del payload, current
    return {
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


@torch.no_grad()
def evaluate_full_matches(
    *,
    trainer: Rival2Trainer,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    phase: str,
    label: str,
) -> dict[str, Any]:
    """Run one complete held-out match in every world with no timeout."""

    started = time.perf_counter()
    kickoff_selector = (
        np.arange(EVALUATION_WORLDS, dtype=np.int32) + EVALUATION_SEED
    ) % 5
    env = Rival2FullMatchEnv(
        EVALUATION_WORLDS,
        collision_dir,
        device=device,
        seed=EVALUATION_SEED,
        reward_version=trainer.env.reward_version,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    generator = torch.Generator(device=env.device).manual_seed(EVALUATION_SEED)
    model = trainer.model
    was_training = model.training
    model.eval()
    active = torch.ones(EVALUATION_WORLDS, dtype=torch.bool, device=env.device)
    completed = torch.zeros_like(active)
    capture_names = (
        "completed_matches",
        "completed_blue_wins",
        "completed_orange_wins",
        "completed_overtime_matches",
        "completed_blue_goals",
        "completed_orange_goals",
        "completed_blue_touches",
        "completed_orange_touches",
        "completed_match_goals",
        "completed_match_ticks",
        "kickoff_segments_total",
        "no_touch_segments_total",
    )
    captured = {
        name: torch.full(
            (EVALUATION_WORLDS,), -1, dtype=torch.int64, device=env.device
        )
        for name in capture_names
    }
    action_sum = torch.zeros((2, 8), dtype=torch.float64, device=env.device)
    action_abs_sum = torch.zeros_like(action_sum)
    action_sq_sum = torch.zeros_like(action_sum)
    action_min = torch.full(
        (2, 8), float("inf"), dtype=torch.float32, device=env.device
    )
    action_max = torch.full(
        (2, 8), float("-inf"), dtype=torch.float32, device=env.device
    )
    active_decisions = torch.zeros((), dtype=torch.int64, device=env.device)
    truncated_count = torch.zeros((), dtype=torch.int64, device=env.device)
    decision = 0
    while True:
        actor, _value = model(env.observation.reshape(-1, trainer.policy_config.obs_dim))
        sample = sample_hybrid_action(
            actor.reshape(EVALUATION_WORLDS, 2, 13),
            generator=generator,
            config=trainer.policy_config,
        )
        action = sample.action
        action = torch.where(active[:, None, None], action, torch.zeros_like(action))
        mask = active[:, None, None]
        mask_float = mask.to(torch.float32)
        action_sum += (action * mask_float).sum(dim=0, dtype=torch.float64)
        action_abs_sum += (action.abs() * mask_float).sum(
            dim=0, dtype=torch.float64
        )
        action_sq_sum += (action.square() * mask_float).sum(
            dim=0, dtype=torch.float64
        )
        action_min = torch.minimum(
            action_min,
            torch.where(mask, action, torch.full_like(action, float("inf"))).amin(
                dim=0
            ),
        )
        action_max = torch.maximum(
            action_max,
            torch.where(
                mask, action, torch.full_like(action, float("-inf"))
            ).amax(dim=0),
        )
        active_decisions += active.sum()
        transition = env.step(action)
        truncated_count += (transition.truncated & active).sum()
        done = transition.terminated & active
        for name in capture_names:
            source = env.full_match_views[name].to(torch.int64)
            captured[name].copy_(torch.where(done, source, captured[name]))
        completed.logical_or_(done)
        active.logical_and_(~done)
        decision += 1
        if decision % 30 == 0 and not bool(active.any().item()):
            break
        if decision % 900 == 0:
            print(
                f"full-match evaluation label={label} simulated_seconds={decision / 30:.1f} "
                f"complete={int(completed.sum().item())}/{EVALUATION_WORLDS}",
                flush=True,
            )

    torch.cuda.synchronize(env.device)
    host = {
        name: value.detach().cpu().numpy().astype(np.int64, copy=True)
        for name, value in captured.items()
    }
    action_count = int(active_decisions.item())
    sum_host = action_sum.cpu().numpy()
    abs_host = action_abs_sum.cpu().numpy()
    sq_host = action_sq_sum.cpu().numpy()
    min_host = action_min.cpu().numpy()
    max_host = action_max.cpu().numpy()
    truncated_seen = int(truncated_count.item()) != 0
    model.train(was_training)

    matches = int(host["completed_matches"].sum())
    blue_wins = int(host["completed_blue_wins"].sum())
    orange_wins = int(host["completed_orange_wins"].sum())
    overtime_matches = int(host["completed_overtime_matches"].sum())
    blue_goals = int(host["completed_blue_goals"].sum())
    orange_goals = int(host["completed_orange_goals"].sum())
    blue_touches = int(host["completed_blue_touches"].sum())
    orange_touches = int(host["completed_orange_touches"].sum())
    total_ticks = int(host["completed_match_ticks"].sum())
    kickoff_segments = int(host["kickoff_segments_total"].sum())
    no_touch_segments = int(host["no_touch_segments_total"].sum())
    simulated_minutes = total_ticks / (120.0 * 60.0)
    no_touch_fraction = (
        no_touch_segments / kickoff_segments if kickoff_segments else 1.0
    )
    action_denominator = float(action_count)
    action_names = (
        "throttle",
        "steer",
        "pitch",
        "yaw",
        "roll",
        "jump",
        "boost",
        "handbrake",
    )
    controller: dict[str, Any] = {}
    for side, side_name in enumerate(("Blue", "Orange")):
        controller[side_name] = {
            name: {
                "mean": float(sum_host[side, channel] / action_denominator),
                "mean_absolute": float(
                    abs_host[side, channel] / action_denominator
                ),
                "rms": float(
                    np.sqrt(sq_host[side, channel] / action_denominator)
                ),
                "minimum": float(min_host[side, channel]),
                "maximum": float(max_host[side, channel]),
                "active_fraction": (
                    float(sum_host[side, channel] / action_denominator)
                    if channel >= 5
                    else None
                ),
            }
            for channel, name in enumerate(action_names)
        }

    duration_seconds = host["completed_match_ticks"].astype(np.float64) / 120.0
    goal_values, goal_counts = np.unique(
        host["completed_match_goals"], return_counts=True
    )
    checks = {
        "all_worlds_completed_once": bool(
            np.all(host["completed_matches"] == 1)
        ),
        "completed_world_count_exact": int(completed.sum().item())
        == EVALUATION_WORLDS,
        "no_truncation": not truncated_seen,
        "wins_cover_matches": blue_wins + orange_wins == matches,
        "goal_totals_consistent": (
            blue_goals + orange_goals
            == int(host["completed_match_goals"].sum())
        ),
        "kickoff_segment_denominator_nonzero": kickoff_segments > 0,
        "episode_contract_exact": (
            env.episode_version == RIVAL2_FULL_MATCH_EPISODE_VERSION
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "phase": phase,
        "checkpoint_label": label,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "reward_version": trainer.env.reward_version,
        "episode_version": trainer.env.episode_version,
        "contract_hashes": dict(trainer.env.contract_hashes),
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_worlds": EVALUATION_WORLDS,
        "mode": "stochastic current-policy self-play; first complete match per world",
        "result": {
            "completed_matches": matches,
            "blue_wins": blue_wins,
            "orange_wins": orange_wins,
            "blue_win_fraction": blue_wins / matches,
            "orange_win_fraction": orange_wins / matches,
            "overtime_matches": overtime_matches,
            "overtime_fraction": overtime_matches / matches,
            "blue_goals": blue_goals,
            "orange_goals": orange_goals,
            "goals": blue_goals + orange_goals,
            "goals_per_simulated_minute": (
                (blue_goals + orange_goals) / simulated_minutes
            ),
            "blue_touches": blue_touches,
            "orange_touches": orange_touches,
            "touches": blue_touches + orange_touches,
            "touches_per_simulated_minute": (
                (blue_touches + orange_touches) / simulated_minutes
            ),
            "simulated_match_minutes": simulated_minutes,
            "kickoff_segments": kickoff_segments,
            "first_touch_within_15s_segments": (
                kickoff_segments - no_touch_segments
            ),
            "counterfactual_no_touch_kickoff_segments": no_touch_segments,
            "counterfactual_no_touch_kickoff_segment_fraction": (
                no_touch_fraction
            ),
            "mean_match_duration_seconds": float(duration_seconds.mean()),
            "median_match_duration_seconds": float(np.median(duration_seconds)),
            "maximum_match_duration_seconds": float(duration_seconds.max()),
            "match_duration_seconds_percentiles": {
                str(percentile): float(np.percentile(duration_seconds, percentile))
                for percentile in (50, 90, 95, 99, 100)
            },
            "goals_per_match_distribution": {
                str(int(value)): int(count)
                for value, count in zip(goal_values, goal_counts, strict=True)
            },
            "controller": controller,
        },
        "wall_seconds": time.perf_counter() - started,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    del env, captured, active, completed
    gc.collect()
    torch.cuda.empty_cache()
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"full-match evaluation integrity failed: {checks}")
    return result


def _training_point(
    *,
    phase: str,
    phase_start_iteration: int,
    trainer: Rival2Trainer,
    rollout: Any,
    metrics: dict[str, torch.Tensor],
    policy_version_before: int,
    samples_before: int,
    seconds: float,
    phase_c_elapsed: float | None,
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
    checks = {
        "inherited_integrity_green": integrity["verdict"] == "PASS_GREEN",
        "full_match_episode_active": (
            trainer.env.episode_version == RIVAL2_FULL_MATCH_EPISODE_VERSION
        ),
        "no_training_truncations": not bool(rollout.truncated.any().item()),
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(
            trainer.env.reward_version,
            RIVAL2_FULL_MATCH_EPISODE_VERSION,
        ),
        "zero_hot_h2d": transfer["h2d"] == 0,
        "zero_hot_d2h": transfer["d2h"] == 0,
        "entropy_coefficient_zero": trainer.ppo_config.entropy_coefficient == 0.0,
    }
    point: dict[str, Any] = {
        "phase": phase,
        "phase_update_offset": trainer.iteration - phase_start_iteration,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "iteration_agent_decision_samples": (
            trainer.total_agent_samples - samples_before
        ),
        "reward_version": trainer.env.reward_version,
        "episode_version": trainer.env.episode_version,
        "wall_seconds": seconds,
        "agent_decisions_per_second": (
            (trainer.total_agent_samples - samples_before) / seconds
        ),
        "terminated_world_intervals": int(rollout.terminated[..., 0].sum().item()),
        "truncated_world_intervals": int(rollout.truncated[..., 0].sum().item()),
        "metrics": values,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if phase_c_elapsed is not None:
        point["phase_c_elapsed_seconds_at_update_completion"] = phase_c_elapsed
    return point


def _train_one_update(
    *,
    phase: str,
    phase_start_iteration: int,
    trainer: Rival2Trainer,
    device: str,
    ledger: Path,
    phase_c_started: float | None = None,
) -> dict[str, Any]:
    policy_before = trainer.policy_version
    samples_before = trainer.total_agent_samples
    trainer.env.reset_transfer_counters()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    rollout, metrics = trainer.train_iteration()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    phase_c_elapsed = (
        None if phase_c_started is None else time.perf_counter() - phase_c_started
    )
    point = _training_point(
        phase=phase,
        phase_start_iteration=phase_start_iteration,
        trainer=trainer,
        rollout=rollout,
        metrics=metrics,
        policy_version_before=policy_before,
        samples_before=samples_before,
        seconds=seconds,
        phase_c_elapsed=phase_c_elapsed,
    )
    _append_jsonl(ledger, point)
    print(
        f"full-match phase={phase} update={trainer.iteration} "
        f"samples={trainer.total_agent_samples} seconds={seconds:.3f} "
        f"kl={point['metrics']['approx_kl']:.6f} "
        f"clip={point['metrics']['clip_fraction']:.6f} "
        f"terminals={point['terminated_world_intervals']} "
        f"verdict={point['verdict']}",
        flush=True,
    )
    if point["verdict"] != "PASS_GREEN":
        failure = ledger.parent / "checkpoints" / f"failure_{trainer.iteration}.pt"
        trainer.save_checkpoint(failure)
        raise RuntimeError(f"training integrity failed: {point['checks']}")
    del rollout, metrics
    gc.collect()
    return point


def _train_one_match_masked_update(
    *,
    trainer: Rival2Trainer,
    active: torch.Tensor,
    phase_start_iteration: int,
    device: str,
    ledger: Path,
) -> dict[str, Any]:
    """Train one PPO update using only worlds still in their counted match."""

    active_before = int(active.sum().item())
    policy_before = trainer.policy_version
    samples_before = trainer.total_agent_samples
    trainer.env.reset_transfer_counters()
    started = time.perf_counter()
    rollout, metrics = trainer.train_iteration(active)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    active_after = int(active.sum().item())
    sample_delta = trainer.total_agent_samples - samples_before
    finite_rollout = all(
        bool(torch.isfinite(getattr(rollout, name)).all().item())
        for name in (
            "observations",
            "actions",
            "pre_tanh",
            "old_log_probability",
            "values",
            "rewards",
            "next_values",
            "advantages",
            "returns",
        )
    )
    values = {name: float(value.item()) for name, value in metrics.items()}
    checks = {
        "finite_rollout": finite_rollout,
        "finite_metrics": all(np.isfinite(value) for value in values.values()),
        "policy_increment_exact": trainer.policy_version == policy_before + 1,
        "iteration_policy_match": trainer.iteration == trainer.policy_version,
        "positive_counted_samples": sample_delta > 0,
        "bounded_counted_samples": sample_delta
        <= trainer.ppo_config.rollout_horizon * trainer.env.num_envs * 2,
        "active_worlds_monotonic": 0 <= active_after <= active_before,
        "no_truncations": not bool(rollout.truncated.any().item()),
        "goal_only_reward_active": (
            trainer.env.reward_version == RIVAL2_REWARD_GOAL_ONLY_VERSION
        ),
        "full_match_episode_active": (
            trainer.env.episode_version == RIVAL2_FULL_MATCH_EPISODE_VERSION
        ),
    }
    point = {
        "phase": "C_GOAL_ONLY_ONE_MATCH_SET",
        "phase_update_offset": trainer.iteration - phase_start_iteration,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "counted_agent_decision_samples_this_update": sample_delta,
        "active_worlds_before": active_before,
        "active_worlds_after": active_after,
        "matches_completed_this_update": active_before - active_after,
        "wall_seconds": seconds,
        "metrics": values,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    _append_jsonl(ledger, point)
    print(
        f"full-match phase=C_ONE_MATCH update={trainer.iteration} "
        f"active={active_after}/{WORLDS} completed={WORLDS - active_after} "
        f"samples={trainer.total_agent_samples} seconds={seconds:.3f} "
        f"kl={values['approx_kl']:.6f} verdict={point['verdict']}",
        flush=True,
    )
    if point["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"one-match-set training integrity failed: {checks}")
    del rollout, metrics
    gc.collect()
    return point


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
            left, right
        )
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
    trainer: Rival2Trainer,
    acquisition: dict[str, Any],
    work_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_payload = torch.load(
        Path(acquisition["path"]), map_location="cpu", weights_only=False
    )
    full_state_before = {
        name: tensor.clone()
        for name, tensor in trainer.env.full_match_views.items()
    }
    identities_before = {
        "model": id(trainer.model),
        "optimizer": id(trainer.optimizer),
        "world": id(trainer.env.world),
        "observation_data_ptr": trainer.env.observation.data_ptr(),
        "full_match_data_ptrs": {
            name: tensor.data_ptr()
            for name, tensor in trainer.env.full_match_views.items()
        },
    }
    transition = trainer.transition_reward_curriculum(
        source_reward_version=RIVAL2_REWARD_V2_VERSION,
        destination_reward_version=RIVAL2_REWARD_GOAL_ONLY_VERSION,
        transition_record={
            "schema_version": SCHEMA_VERSION,
            "authority": AUTHORITY.as_posix(),
            "mechanics_fix_commit": MECHANICS_FIX_COMMIT,
            "parent_checkpoint_sha256": acquisition["sha256"],
            "authorized_transition": (
                "RIVAL2_REWARD_V2 -> RIVAL2_REWARD_GOAL_ONLY_V1"
            ),
            "all_training_remains_complete_standard_matches": True,
        },
    )
    identities_after = {
        "model": id(trainer.model),
        "optimizer": id(trainer.optimizer),
        "world": id(trainer.env.world),
        "observation_data_ptr": trainer.env.observation.data_ptr(),
        "full_match_data_ptrs": {
            name: tensor.data_ptr()
            for name, tensor in trainer.env.full_match_views.items()
        },
    }
    live_state_exact = all(
        torch.equal(value, trainer.env.full_match_views[name])
        for name, value in full_state_before.items()
    )
    checkpoint = _save_checkpoint("goal_only_transition", trainer, work_dir)
    destination_payload = torch.load(
        Path(checkpoint["path"]), map_location="cpu", weights_only=False
    )
    preserved_fields = (
        "model",
        "optimizer",
        "policy_config",
        "ppo_config",
        "self_play_config",
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
    )
    field_checks = {
        name: _nested_exact(source_payload[name], destination_payload[name])
        for name in preserved_fields
    }
    checks = {
        "transition_contract_recorded": bool(transition),
        "runtime_identity_preserved": identities_before == identities_after,
        "live_full_match_state_exact": live_state_exact,
        "all_checkpoint_fields_exact": all(field_checks.values()),
        "destination_reward_exact": (
            trainer.env.reward_version == RIVAL2_REWARD_GOAL_ONLY_VERSION
        ),
        "destination_episode_unchanged": (
            trainer.env.episode_version == RIVAL2_FULL_MATCH_EPISODE_VERSION
        ),
        "destination_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(
            RIVAL2_REWARD_GOAL_ONLY_VERSION,
            RIVAL2_FULL_MATCH_EPISODE_VERSION,
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "source_checkpoint": acquisition,
        "transition": transition,
        "post_transition_checkpoint": checkpoint,
        "preserved_checkpoint_fields": field_checks,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    _write_json(work_dir / "reward_transition.json", result)
    del source_payload, destination_payload, full_state_before
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"reward-only transition failed: {checks}")
    return result, checkpoint


def _evaluate_and_write(
    *,
    trainer: Rival2Trainer,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    phase: str,
    label: str,
    work_dir: Path,
) -> dict[str, Any]:
    evaluation = evaluate_full_matches(
        trainer=trainer,
        collision_dir=collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=device,
        phase=phase,
        label=label,
    )
    _write_json(work_dir / f"evaluation_{label}.json", evaluation)
    metric = evaluation["result"]
    print(
        f"full-match evaluation={label} update={trainer.iteration} "
        f"touches/min={metric['touches_per_simulated_minute']:.6f} "
        f"goals/min={metric['goals_per_simulated_minute']:.6f} "
        f"no_touch={metric['counterfactual_no_touch_kickoff_segment_fraction']:.6f} "
        f"overtime={metric['overtime_fraction']:.6f}",
        flush=True,
    )
    return evaluation


def run_curriculum(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    launch_gate: dict[str, Any],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> dict[str, Any]:
    torch.manual_seed(CAMPAIGN_SEED)
    torch.cuda.manual_seed(CAMPAIGN_SEED)
    kickoff_selector = (
        np.arange(WORLDS, dtype=np.int32) + CAMPAIGN_SEED
    ) % 5
    env = Rival2FullMatchEnv(
        WORLDS,
        args.collision_dir,
        device=args.device,
        seed=CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_V2_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2Trainer(
        env,
        ppo_config=campaign03.campaign03_ppo_config(),
        seed=CAMPAIGN_SEED,
    )
    initialization_sha256 = campaign01._state_dict_sha256(
        {
            name: tensor.detach().cpu().clone()
            for name, tensor in trainer.model.state_dict().items()
        }
    )
    trainer.add_historical_snapshot()
    ledger = args.work_dir / "training_curve.jsonl"
    if ledger.exists():
        raise RuntimeError("work directory already contains a training ledger")
    curriculum_started = time.perf_counter()

    phase_a_start = trainer.iteration
    phase_a_started = time.perf_counter()
    phase_a_evaluations: list[dict[str, Any]] = []
    phase_a_checkpoints: list[dict[str, Any]] = []
    consecutive = 0
    next_evaluation = PHASE_A_EVALUATION_INTERVAL
    while consecutive < PHASE_A_REQUIRED_CONSECUTIVE:
        _train_one_update(
            phase="A_REWARD_V2_ACQUISITION",
            phase_start_iteration=phase_a_start,
            trainer=trainer,
            device=args.device,
            ledger=ledger,
        )
        if trainer.iteration != next_evaluation:
            continue
        trainer.add_historical_snapshot()
        label = f"phase_a_update_{trainer.iteration}"
        checkpoint = _save_checkpoint(label, trainer, args.work_dir)
        evaluation = _evaluate_and_write(
            trainer=trainer,
            collision_dir=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
            phase="A_REWARD_V2_ACQUISITION",
            label=label,
            work_dir=args.work_dir,
        )
        fraction = evaluation["result"][
            "counterfactual_no_touch_kickoff_segment_fraction"
        ]
        passed = fraction <= PHASE_A_THRESHOLD
        consecutive = consecutive + 1 if passed else 0
        evaluation["acquisition_threshold"] = PHASE_A_THRESHOLD
        evaluation["acquisition_threshold_passed"] = passed
        evaluation["consecutive_passing_evaluations"] = consecutive
        _write_json(args.work_dir / f"evaluation_{label}.json", evaluation)
        phase_a_checkpoints.append(checkpoint)
        phase_a_evaluations.append(evaluation)
        _write_json(
            args.work_dir / "phase_a_progress.json",
            {
                "checkpoints": phase_a_checkpoints,
                "evaluations": phase_a_evaluations,
                "consecutive_passing_evaluations": consecutive,
            },
        )
        next_evaluation += PHASE_A_EVALUATION_INTERVAL

    acquisition = phase_a_checkpoints[-1]
    phase_a_summary = {
        "status": "COMPLETE",
        "start_iteration": phase_a_start,
        "final_iteration": trainer.iteration,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "updates": trainer.iteration - phase_a_start,
        "evaluation_count": len(phase_a_evaluations),
        "threshold": PHASE_A_THRESHOLD,
        "confirming_fractions": [
            item["result"][
                "counterfactual_no_touch_kickoff_segment_fraction"
            ]
            for item in phase_a_evaluations[-2:]
        ],
        "acquisition_checkpoint": acquisition,
        "wall_seconds_including_evaluations": time.perf_counter()
        - phase_a_started,
    }
    _write_json(args.work_dir / "phase_a_summary.json", phase_a_summary)

    transition, transition_checkpoint = _transition_reward(
        trainer, acquisition, args.work_dir
    )

    phase_b_start_iteration = trainer.iteration
    phase_b_start_samples = trainer.total_agent_samples
    phase_b_started = time.perf_counter()
    phase_b_checkpoints: list[dict[str, Any]] = []
    phase_b_evaluations: list[dict[str, Any]] = []
    next_index = 0
    while trainer.iteration - phase_b_start_iteration < PHASE_B_UPDATES:
        _train_one_update(
            phase="B_GOAL_ONLY_2B",
            phase_start_iteration=phase_b_start_iteration,
            trainer=trainer,
            device=args.device,
            ledger=ledger,
        )
        offset = trainer.iteration - phase_b_start_iteration
        if offset != PHASE_B_EVALUATION_OFFSETS[next_index]:
            continue
        trainer.add_historical_snapshot()
        label = f"phase_b_plus_{offset}"
        checkpoint = _save_checkpoint(label, trainer, args.work_dir)
        evaluation = _evaluate_and_write(
            trainer=trainer,
            collision_dir=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
            phase="B_GOAL_ONLY_2B",
            label=label,
            work_dir=args.work_dir,
        )
        phase_b_checkpoints.append(checkpoint)
        phase_b_evaluations.append(evaluation)
        next_index += 1

    if not (
        trainer.iteration == phase_b_start_iteration + PHASE_B_UPDATES
        and trainer.total_agent_samples
        == phase_b_start_samples + PHASE_B_ADDITIONAL_SAMPLES
        and next_index == len(PHASE_B_EVALUATION_OFFSETS)
    ):
        raise RuntimeError("Phase B did not stop at the exact 239-update boundary")
    phase_b_checkpoint = phase_b_checkpoints[-1]
    phase_b_audit = _checkpoint_audit(phase_b_checkpoint, trainer)
    if phase_b_audit["verdict"] != "PASS_GREEN":
        raise RuntimeError("Phase B checkpoint audit failed")
    phase_b_summary = {
        "status": "COMPLETE",
        "start_iteration": phase_b_start_iteration,
        "start_agent_decision_samples": phase_b_start_samples,
        "final_iteration": trainer.iteration,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "additional_updates": PHASE_B_UPDATES,
        "additional_agent_decision_samples": PHASE_B_ADDITIONAL_SAMPLES,
        "checkpoints": phase_b_checkpoints,
        "checkpoint_audit": phase_b_audit,
        "wall_seconds_including_evaluations": time.perf_counter()
        - phase_b_started,
    }
    _write_json(args.work_dir / "phase_b_summary.json", phase_b_summary)

    # User-steered replacement for the former six-hour continuation: one
    # prospectively counted complete match per resident world. Worlds that
    # finish early remain resident but are masked out of every later PPO batch.
    trainer.env.start_fresh_matches()
    phase_c_started = time.perf_counter()
    phase_c_start_iteration = trainer.iteration
    phase_c_start_samples = trainer.total_agent_samples
    active_match_worlds = torch.ones(WORLDS, dtype=torch.bool, device=trainer.device)
    final_match_ledger = args.work_dir / "final_match_training_curve.jsonl"
    while bool(active_match_worlds.any().item()):
        _train_one_match_masked_update(
            trainer=trainer,
            active=active_match_worlds,
            phase_start_iteration=phase_c_start_iteration,
            device=args.device,
            ledger=final_match_ledger,
        )

    trainer.add_historical_snapshot()
    final_checkpoint = _save_checkpoint(
        "goal_only_final_match_set", trainer, args.work_dir
    )
    final_evaluation = _evaluate_and_write(
        trainer=trainer,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        phase="C_GOAL_ONLY_ONE_MATCH_SET",
        label="goal_only_final_match_set",
        work_dir=args.work_dir,
    )
    final_audit = _checkpoint_audit(final_checkpoint, trainer)
    if final_audit["verdict"] != "PASS_GREEN":
        raise RuntimeError("final checkpoint audit failed")
    phase_c_summary = {
        "status": "COMPLETE",
        "bound": "one fresh complete standard match per resident world",
        "start_iteration": phase_c_start_iteration,
        "start_agent_decision_samples": phase_c_start_samples,
        "final_iteration": trainer.iteration,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "additional_updates": trainer.iteration - phase_c_start_iteration,
        "additional_agent_decision_samples": trainer.total_agent_samples
        - phase_c_start_samples,
        "counted_complete_matches": WORLDS,
        "remaining_active_worlds": int(active_match_worlds.sum().item()),
        "wall_seconds_including_final_evaluation": time.perf_counter()
        - phase_c_started,
        "training_curve": final_match_ledger.resolve().as_posix(),
        "final_checkpoint": final_checkpoint,
        "final_evaluation_label": final_evaluation["checkpoint_label"],
        "final_checkpoint_audit": final_audit,
    }
    _write_json(args.work_dir / "phase_c_summary.json", phase_c_summary)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": "COMPLETE",
        "source_head": launch_gate["head"],
        "mechanics_fix_commit": MECHANICS_FIX_COMMIT,
        "initialization_model_sha256": initialization_sha256,
        "configuration": configuration,
        "phase_a": phase_a_summary,
        "reward_transition": transition,
        "reward_transition_checkpoint": transition_checkpoint,
        "phase_b": phase_b_summary,
        "phase_c": phase_c_summary,
        "final_iteration": trainer.iteration,
        "final_policy_version": trainer.policy_version,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "final_reward_version": trainer.env.reward_version,
        "final_episode_version": trainer.env.episode_version,
        "final_contract_hashes": dict(trainer.env.contract_hashes),
        "final_historical_policy_versions": trainer.opponent_pool.versions,
        "curriculum_wall_seconds_including_evaluations": time.perf_counter()
        - curriculum_started,
        "all_training_used_complete_matches": True,
        "training_truncation_contract": None,
        "viewer_built": False,
        "v06_begun": False,
    }
    _write_json(args.work_dir / "run_summary.json", summary)
    return summary


def publish_results(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    launch_gate: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    results = args.results_dir.resolve()
    if results.exists() and any(results.iterdir()):
        raise RuntimeError("results directory is not empty")
    results.mkdir(parents=True, exist_ok=True)
    _write_json(results / "config.json", configuration)
    _write_json(results / "launch_gate.json", launch_gate)
    for name in (
        "phase_a_progress.json",
        "phase_a_summary.json",
        "reward_transition.json",
        "phase_b_summary.json",
        "phase_c_summary.json",
        "run_summary.json",
        "training_curve.jsonl",
        "final_match_training_curve.jsonl",
    ):
        shutil.copy2(args.work_dir / name, results / name)
    for evaluation in sorted(args.work_dir.glob("evaluation_*.json")):
        shutil.copy2(evaluation, results / evaluation.name)

    selected = {
        "acquisition_complete": Path(
            summary["phase_a"]["acquisition_checkpoint"]["path"]
        ),
        "goal_only_transition": Path(
            summary["reward_transition_checkpoint"]["path"]
        ),
        "goal_only_2b": Path(summary["phase_b"]["checkpoints"][-1]["path"]),
        "goal_only_final_match_set": Path(
            summary["phase_c"]["final_checkpoint"]["path"]
        ),
    }
    checkpoint_records = []
    checkpoint_dir = Path("checkpoints/rival2/full_match_curriculum")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for label, source in selected.items():
        destination = checkpoint_dir / f"rival2_full_match_{label}_resume.pt"
        shutil.copy2(source, destination)
        checkpoint_records.append(
            {
                "label": label,
                "path": destination.as_posix(),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    _write_json(results / "checkpoints.json", {"checkpoints": checkpoint_records})

    evaluations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results.glob("evaluation_*.json"))
    ]
    evaluations.sort(key=lambda item: (item["iteration"], item["checkpoint_label"]))
    curve = [
        {
            "label": item["checkpoint_label"],
            "phase": item["phase"],
            "iteration": item["iteration"],
            "agent_decision_samples": item["agent_decision_samples"],
            "reward_version": item["reward_version"],
            "touches_per_simulated_minute": item["result"][
                "touches_per_simulated_minute"
            ],
            "goals_per_simulated_minute": item["result"][
                "goals_per_simulated_minute"
            ],
            "counterfactual_no_touch_kickoff_segment_fraction": item[
                "result"
            ]["counterfactual_no_touch_kickoff_segment_fraction"],
            "blue_win_fraction": item["result"]["blue_win_fraction"],
            "orange_win_fraction": item["result"]["orange_win_fraction"],
            "overtime_fraction": item["result"]["overtime_fraction"],
        }
        for item in evaluations
    ]
    _write_json(results / "behavioral_curve.json", curve)
    final = curve[-1]
    report = f"""# Rival 2.0 mechanics-corrective full-match curriculum

Status: **COMPLETE**

- Mechanics correction: `{MECHANICS_FIX_COMMIT}`.
- Every rollout used the complete five-minute regulation/sudden-death-overtime
  contract `{RIVAL2_FULL_MATCH_EPISODE_VERSION}`; no training truncation or
  short-episode reset was active.
- Phase A ended at update `{summary['phase_a']['final_iteration']}` with two
  consecutive counterfactual no-touch kickoff-segment fractions
  `{summary['phase_a']['confirming_fractions']}`.
- The reward-only transition to `{RIVAL2_REWARD_GOAL_ONLY_VERSION}` preserved
  trainer and live full-match state and passed its exact transition audit.
- Phase B completed exactly `{PHASE_B_UPDATES}` updates / `{PHASE_B_ADDITIONAL_SAMPLES:,}`
  samples.
- The final phase trained on exactly one prospectively counted complete match
  in each of `{WORLDS:,}` worlds and completed at update
  `{summary['phase_c']['final_iteration']}`. Samples after an individual
  world's match completion were excluded from PPO.
- Final held-out complete-match evaluation: touches/min
  `{final['touches_per_simulated_minute']:.6f}`, goals/min
  `{final['goals_per_simulated_minute']:.6f}`, counterfactual no-touch fraction
  `{final['counterfactual_no_touch_kickoff_segment_fraction']:.6f}`, Blue wins
  `{final['blue_win_fraction']:.6f}`, Orange wins
  `{final['orange_win_fraction']:.6f}`, overtime `{final['overtime_fraction']:.6f}`.

The machine-readable evaluation files preserve Blue and Orange separately,
complete match durations, score/touch totals, kickoff acquisition counts, and
controller activation summaries. The campaign stops here and does not begin
v0.6.
"""
    (results / "README.md").write_text(report, encoding="utf-8")

    manifest = []
    for path in sorted(
        [item for item in results.rglob("*") if item.is_file()]
        + [Path(item["path"]) for item in checkpoint_records]
    ):
        manifest.append(
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _write_json(results / "manifest.json", {"files": manifest})


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise RuntimeError("work directory must be empty")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configuration = frozen_configuration()
    _write_json(args.work_dir / "config_frozen_before_training.json", configuration)
    campaign01._initialize_runtime(args.device)
    launch_gate = verify_launch(configuration)
    _write_json(args.work_dir / "launch_gate.json", launch_gate)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    summary = run_curriculum(args, configuration, launch_gate, geometry, meshes)
    publish_results(args, configuration, launch_gate, summary)
    print(
        f"full-match curriculum COMPLETE update={summary['final_iteration']} "
        f"samples={summary['final_agent_decision_samples']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
