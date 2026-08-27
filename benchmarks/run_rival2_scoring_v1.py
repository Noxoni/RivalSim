"""Run the bounded Rival 2.0 scoring curriculum from acquisition V1."""

from __future__ import annotations

import argparse
import copy
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
import benchmarks.run_rival2_full_match_curriculum as full_match
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    ACTION_NAMES,
    CAR_LINEAR_SPEED_SCALE,
    FULL_MATCH_EPISODE_CONTRACT_HASH,
    OBS_DIM,
    OBS_FIELD_NAMES,
    POSITION_SCALE,
    REWARD_SCORING_V1_CONTRACT,
    REWARD_SCORING_V1_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_FULL_MATCH_EPISODE_VERSION,
    RIVAL2_REWARD_ACQUISITION_V1_VERSION,
    RIVAL2_REWARD_SCORING_V1_VERSION,
    SCORING_APPROACH_COEFFICIENT,
    SCORING_DEMOLITION_REWARD,
    SCORING_FLIP_ONSET_COST,
    SCORING_JUMP_RISING_EDGE_COST,
    SCORING_PROGRESS_COEFFICIENT,
    SCORING_TOUCH_REWARD,
    contract_hashes_for_reward,
)
from rivalsim.rival2_full_match_env import Rival2FullMatchEnv
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    sample_hybrid_action,
)
from rivalsim.rival2_ppo import Rival2PPOConfig
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer

AUTHORITY = Path("handoff/rival2-scoring-v1/README.md")
SOURCE_COMMIT = "61307571d86508f3026402c4948f759f310ff36c"
SOURCE_CHECKPOINT = Path(
    "checkpoints/rival2/acquisition_v1/rival2_acquisition_resume.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "4FB7A3B134B25D595374E3968E2EDFA150A9CD6F8910B903BF892B59D7F8BC9A"
)
MECHANICS_EVIDENCE = Path(
    "results/rival2/mechanics_correction/movement_mechanics_parity.json"
)
ACQUISITION_EVALUATION = Path(
    "results/rival2/acquisition_v1/evaluation_update_120.json"
)

WORLDS = 131_072
CAMPAIGN_SEED = 20_260_827
ROLLOUT_AGENT_SAMPLES = 8_388_608
ADDITIONAL_UPDATES = 239
ADDITIONAL_AGENT_SAMPLES = 2_004_877_312
CHECKPOINT_OFFSETS = (60, 120, 180, 239)
EVALUATION_WORLDS = 1_024
EVALUATION_SEED = 930_260_827
REFERENCE_WORLDS_PER_SIDE = 256
REFERENCE_WORLDS = REFERENCE_WORLDS_PER_SIDE * 2
REFERENCE_SEED = 940_260_827
NO_TOUCH_WARNING_THRESHOLD = 0.01
SCHEMA_VERSION = 1

_BALL_Y_INDEX = OBS_FIELD_NAMES.index("ball.position.y")
_SELF_VELOCITY_START = OBS_FIELD_NAMES.index("self.linear_velocity.x")
_SELF_BOOST_INDEX = OBS_FIELD_NAMES.index("self.boost")
_SELF_ON_GROUND_INDEX = OBS_FIELD_NAMES.index("self.on_ground")
_SELF_HAS_FLIPPED_INDEX = OBS_FIELD_NAMES.index("self.has_flipped")
_SELF_SUPERSONIC_INDEX = OBS_FIELD_NAMES.index("self.is_supersonic")
_RELATIVE_BALL_START = OBS_FIELD_NAMES.index("relative.ball_position.x")
_PREVIOUS_JUMP_INDEX = OBS_FIELD_NAMES.index("previous_action.jump")
_KICKOFF_INDEX = OBS_FIELD_NAMES.index("lifecycle.kickoff_reset")
_SELF_TOUCH_INDEX = OBS_FIELD_NAMES.index("lifecycle.self_touch_event")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/rival2/scoring_v1"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
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


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
            left.cpu(), right.cpu()
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


def frozen_configuration(source_checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    policy = Rival2PolicyConfig(**payload["policy_config"])
    ppo = Rival2PPOConfig(**payload["ppo_config"])
    self_play = Rival2SelfPlayConfig(**payload["self_play_config"])
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY.as_posix(),
        "source_commit": SOURCE_COMMIT,
        "source_checkpoint": source_checkpoint.as_posix(),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_iteration": int(payload["iteration"]),
        "source_policy_version": int(payload["policy_version"]),
        "source_agent_decision_samples": int(payload["total_agent_samples"]),
        "source_reward_version": payload["reward_version"],
        "source_episode_version": payload["episode_version"],
        "campaign_seed": CAMPAIGN_SEED,
        "worlds": WORLDS,
        "policy_config": asdict(policy),
        "policy_config_hash": policy.content_hash,
        "ppo_config": asdict(ppo),
        "ppo_config_hash": ppo.content_hash,
        "self_play_config": asdict(self_play),
        "destination_reward_version": RIVAL2_REWARD_SCORING_V1_VERSION,
        "destination_reward_contract_hash": REWARD_SCORING_V1_CONTRACT_HASH,
        "destination_reward_contract": REWARD_SCORING_V1_CONTRACT,
        "destination_episode_version": RIVAL2_FULL_MATCH_EPISODE_VERSION,
        "destination_episode_contract_hash": FULL_MATCH_EPISODE_CONTRACT_HASH,
        "training": {
            "additional_updates": ADDITIONAL_UPDATES,
            "additional_agent_decision_samples": ADDITIONAL_AGENT_SAMPLES,
            "rollout_agent_decision_samples": ROLLOUT_AGENT_SAMPLES,
            "checkpoint_offsets": list(CHECKPOINT_OFFSETS),
            "ppo_boundaries_reset_matches": False,
            "nexto_training": False,
        },
        "evaluation": {
            "self_play_worlds": EVALUATION_WORLDS,
            "self_play_seed": EVALUATION_SEED,
            "reference_worlds_per_current_policy_side": REFERENCE_WORLDS_PER_SIDE,
            "reference_seed": REFERENCE_SEED,
            "stochastic": True,
            "complete_full_matches": True,
            "no_touch_warning_threshold": NO_TOUCH_WARNING_THRESHOLD,
        },
    }


def verify_launch(
    configuration: dict[str, Any], source_checkpoint: Path
) -> dict[str, Any]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], check=True
    )
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    mechanics = json.loads(MECHANICS_EVIDENCE.read_text(encoding="utf-8"))
    source_hash = _sha256(source_checkpoint)
    expected_samples = configuration["source_agent_decision_samples"] + (
        ADDITIONAL_UPDATES * ROLLOUT_AGENT_SAMPLES
    )
    checks = {
        "source_commit_is_ancestor": True,
        "head_pushed_to_origin_main": head == origin,
        "tracked_worktree_clean": subprocess.run(
            ["git", "diff", "--quiet"]
        ).returncode
        == 0,
        "index_clean": subprocess.run(
            ["git", "diff", "--cached", "--quiet"]
        ).returncode
        == 0,
        "authority_present": AUTHORITY.is_file(),
        "source_checkpoint_present": source_checkpoint.is_file(),
        "source_checkpoint_sha256_exact": source_hash == SOURCE_CHECKPOINT_SHA256,
        "source_reward_exact": configuration["source_reward_version"]
        == RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        "source_episode_exact": configuration["source_episode_version"]
        == RIVAL2_EPISODE_VERSION,
        "source_world_count_exact": configuration["worlds"] == WORLDS,
        "entropy_zero": configuration["ppo_config"]["entropy_coefficient"] == 0.0,
        "horizon_32": configuration["ppo_config"]["rollout_horizon"] == 32,
        "sample_arithmetic_exact": expected_samples == 3_011_510_272,
        "mechanics_evidence_green": mechanics["status"] == "PASS",
        "mechanics_evidence_no_failures": mechanics["failures"] == [],
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "head": head,
        "origin_main": origin,
        "source_checkpoint_sha256": source_hash,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"scoring launch gate failed: {checks}")
    return result


def reward_scale_accounting() -> dict[str, Any]:
    acquisition = json.loads(ACQUISITION_EVALUATION.read_text(encoding="utf-8"))
    movement = acquisition["result"]["movement"]
    simulated_minutes = float(acquisition["result"]["simulated_minutes"])
    jump_edges_per_minute = sum(
        float(movement[side]["jump_rising_edges"]) / simulated_minutes
        for side in ("Blue", "Orange")
    ) / 2.0
    flips_per_minute = sum(
        float(movement[side]["flips_per_simulated_minute"])
        for side in ("Blue", "Orange")
    ) / 2.0
    reference_jump_cost = jump_edges_per_minute * abs(
        SCORING_JUMP_RISING_EDGE_COST
    )
    reference_flip_cost = flips_per_minute * abs(SCORING_FLIP_ONSET_COST)
    alternating_jump_edges_per_minute = 30 * 60 / 2
    flip_torque_envelope_per_minute = 60.0 / 0.65
    planning_envelope = (
        alternating_jump_edges_per_minute * abs(SCORING_JUMP_RISING_EDGE_COST)
        + flip_torque_envelope_per_minute * abs(SCORING_FLIP_ONSET_COST)
    )
    examples = {
        "one_goal": 10.0,
        "one_full_center_to_goal_ball_advance_5120uu": (
            SCORING_PROGRESS_COEFFICIENT
        ),
        "one_meaningful_512uu_ball_advance": (
            SCORING_PROGRESS_COEFFICIENT * 512.0 / 5120.0
        ),
        "one_1000uu_car_ball_approach": (
            SCORING_APPROACH_COEFFICIENT * 1000.0 / 4096.0
        ),
        "one_unique_touch": SCORING_TOUCH_REWARD,
        "one_unique_demolition": SCORING_DEMOLITION_REWARD,
        "one_jump_rising_edge": SCORING_JUMP_RISING_EDGE_COST,
        "one_actual_flip_onset": SCORING_FLIP_ONSET_COST,
    }
    checks = {
        "goal_is_largest_single_event": examples["one_goal"]
        > 100.0 * abs(examples["one_actual_flip_onset"]),
        "meaningful_progress_exceeds_flip_cost": (
            examples["one_meaningful_512uu_ball_advance"]
            > abs(examples["one_actual_flip_onset"])
        ),
        "touch_exceeds_flip_cost": examples["one_unique_touch"]
        > abs(examples["one_actual_flip_onset"]),
        "jump_edge_is_smaller_than_touch": abs(examples["one_jump_rising_edge"])
        < examples["one_unique_touch"],
        "reference_spam_penalty_below_20pct_of_goal": (
            reference_jump_cost + reference_flip_cost < 2.0
        ),
        "source_timing_planning_envelope_below_goal": planning_envelope < 10.0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "reward_version": RIVAL2_REWARD_SCORING_V1_VERSION,
        "reward_contract_hash": REWARD_SCORING_V1_CONTRACT_HASH,
        "equation": (
            "goal + 0.5*canonical_delta_ball_y/5120 + "
            "0.10*(distance_before-distance_after)/4096 + 0.02*own_touch + "
            "0.10*(own_demo-opponent_demo) - 0.002*jump_edge - 0.01*flip_onset"
        ),
        "single_event_examples": examples,
        "acquisition_reference": {
            "evaluation": ACQUISITION_EVALUATION.as_posix(),
            "jump_rising_edges_per_player_minute": jump_edges_per_minute,
            "actual_flips_per_player_minute": flips_per_minute,
            "expected_jump_cost_magnitude_per_player_minute": reference_jump_cost,
            "expected_flip_cost_magnitude_per_player_minute": reference_flip_cost,
            "expected_total_control_cost_magnitude_per_player_minute": (
                reference_jump_cost + reference_flip_cost
            ),
        },
        "planning_envelope": {
            "alternating_jump_edges_per_minute": alternating_jump_edges_per_minute,
            "flip_onsets_per_minute_at_0_65s_source_torque_envelope": (
                flip_torque_envelope_per_minute
            ),
            "combined_cost_magnitude_per_minute": planning_envelope,
            "qualification": (
                "Static scale envelope, not a claim that back-to-back flips at the "
                "0.65-second torque duration are physically achievable."
            ),
        },
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _checkpoint_record(
    path: Path, label: str, offset: int, trainer: Rival2Trainer
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    return {
        "label": label,
        "offset_updates": offset,
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


def _rng_snapshot(trainer: Rival2Trainer) -> dict[str, torch.Tensor]:
    return {
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": torch.cuda.get_rng_state(trainer.device).cpu().clone(),
        "policy": trainer.policy_generator.get_state().cpu().clone(),
        "opponent": trainer.opponent_generator.get_state().cpu().clone(),
    }


def _rng_exact(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _capture_completed(
    env: Rival2FullMatchEnv,
    captured: dict[str, torch.Tensor],
    done: torch.Tensor,
) -> None:
    for name, target in captured.items():
        source = env.full_match_views[name].to(torch.int64)
        target.copy_(torch.where(done, source, target))


def _score_distribution(blue: np.ndarray, orange: np.ndarray) -> dict[str, int]:
    values: dict[str, int] = {}
    for blue_score, orange_score in zip(blue, orange, strict=True):
        key = f"{int(blue_score)}-{int(orange_score)}"
        values[key] = values.get(key, 0) + 1
    return dict(sorted(values.items()))


@torch.no_grad()
def evaluate_self_play(
    *,
    trainer: Rival2Trainer,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    label: str,
    offset: int,
) -> dict[str, Any]:
    """Run one complete stochastic held-out match with rich low-overhead telemetry."""

    started = time.perf_counter()
    rng_before = _rng_snapshot(trainer)
    kickoff_selector = (
        np.arange(EVALUATION_WORLDS, dtype=np.int32) + EVALUATION_SEED
    ) % 5
    env = Rival2FullMatchEnv(
        EVALUATION_WORLDS,
        collision_dir,
        device=device,
        seed=EVALUATION_SEED,
        reward_version=RIVAL2_REWARD_SCORING_V1_VERSION,
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
    side_decisions = torch.zeros(2, dtype=torch.float64, device=env.device)
    boost_sum = torch.zeros(2, dtype=torch.float64, device=env.device)
    boost_starved = torch.zeros_like(boost_sum)
    speed_sum = torch.zeros_like(boost_sum)
    max_speed = torch.zeros(2, dtype=torch.float32, device=env.device)
    grounded = torch.zeros_like(boost_sum)
    supersonic = torch.zeros_like(boost_sum)
    distance_sum = torch.zeros_like(boost_sum)
    jump_edges = torch.zeros_like(boost_sum)
    flip_onsets = torch.zeros_like(boost_sum)
    ball_progress_uu = torch.zeros_like(boost_sum)
    first_touch_latency_decisions = torch.zeros(
        (), dtype=torch.float64, device=env.device
    )
    first_touch_latency_count = torch.zeros(
        (), dtype=torch.int64, device=env.device
    )
    segment_waiting = torch.ones_like(active)
    segment_age = torch.zeros(
        EVALUATION_WORLDS, dtype=torch.int32, device=env.device
    )
    position_scale = torch.tensor(
        POSITION_SCALE, dtype=torch.float32, device=env.device
    )
    truncated_count = torch.zeros((), dtype=torch.int64, device=env.device)
    decision = 0
    while True:
        observation = env.observation
        actor, _value = model(observation.reshape(-1, OBS_DIM))
        sample = sample_hybrid_action(
            actor.reshape(EVALUATION_WORLDS, 2, 13),
            generator=generator,
            config=trainer.policy_config,
        )
        action = torch.where(
            active[:, None, None], sample.action, torch.zeros_like(sample.action)
        )
        mask = active[:, None]
        mask3 = mask[..., None]
        mask_float = mask.to(torch.float32)
        active_count = active.sum().to(torch.float64)
        side_decisions += active_count
        action_sum += (action * mask3).sum(dim=0, dtype=torch.float64)
        action_abs_sum += (action.abs() * mask3).sum(dim=0, dtype=torch.float64)
        action_sq_sum += (action.square() * mask3).sum(dim=0, dtype=torch.float64)
        boost = observation[..., _SELF_BOOST_INDEX] * 100.0
        boost_sum += (boost * mask_float).sum(dim=0, dtype=torch.float64)
        boost_starved += ((boost <= 1.0) & mask).sum(dim=0, dtype=torch.float64)
        velocity = (
            observation[..., _SELF_VELOCITY_START : _SELF_VELOCITY_START + 3]
            * CAR_LINEAR_SPEED_SCALE
        )
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        speed_sum += (speed * mask_float).sum(dim=0, dtype=torch.float64)
        max_speed = torch.maximum(
            max_speed,
            torch.where(mask, speed, torch.zeros_like(speed)).amax(dim=0),
        )
        grounded += (
            (observation[..., _SELF_ON_GROUND_INDEX] > 0.5) & mask
        ).sum(dim=0, dtype=torch.float64)
        supersonic += (
            (observation[..., _SELF_SUPERSONIC_INDEX] > 0.5) & mask
        ).sum(dim=0, dtype=torch.float64)
        relative = (
            observation[..., _RELATIVE_BALL_START : _RELATIVE_BALL_START + 3]
            * position_scale
        )
        distance_sum += (
            torch.linalg.vector_norm(relative, dim=-1) * mask_float
        ).sum(dim=0, dtype=torch.float64)
        jump_edges += (
            (action[..., 5] >= 0.5)
            & (observation[..., _PREVIOUS_JUMP_INDEX] < 0.5)
            & mask
        ).sum(dim=0, dtype=torch.float64)
        flipped_before = observation[..., _SELF_HAS_FLIPPED_INDEX] >= 0.5
        segment_age += (active & segment_waiting).to(torch.int32)

        transition = env.step(action)
        after = transition.transition_observation
        truncated_count += (transition.truncated & active).sum()
        flip_onsets += (
            (after[..., _SELF_HAS_FLIPPED_INDEX] >= 0.5)
            & ~flipped_before
            & mask
        ).sum(dim=0, dtype=torch.float64)
        ball_progress_uu += (
            (after[..., _BALL_Y_INDEX] - observation[..., _BALL_Y_INDEX])
            * 5120.0
            * mask_float
        ).sum(dim=0, dtype=torch.float64)
        touched = (after[..., _SELF_TOUCH_INDEX] > 0.5) & mask
        first_touch = touched.any(dim=1) & segment_waiting & active
        first_touch_latency_decisions += segment_age[first_touch].sum(
            dtype=torch.float64
        )
        first_touch_latency_count += first_touch.sum()
        segment_waiting &= ~first_touch
        segment_waiting &= segment_age < 15 * 30

        done = transition.terminated & active
        _capture_completed(env, captured, done)
        completed |= done
        active &= ~done
        kickoff = (
            transition.observation[:, 0, _KICKOFF_INDEX] > 0.5
        ) & active
        segment_waiting |= kickoff
        segment_age.copy_(torch.where(kickoff, torch.zeros_like(segment_age), segment_age))
        decision += 1
        if decision % 30 == 0 and not bool(active.any().item()):
            break
        if decision % 900 == 0:
            print(
                f"scoring self-play eval={label} simulated_seconds={decision / 30:.1f} "
                f"complete={int(completed.sum().item())}/{EVALUATION_WORLDS}",
                flush=True,
            )

    torch.cuda.synchronize(env.device)
    host = {
        name: tensor.cpu().numpy().astype(np.int64, copy=True)
        for name, tensor in captured.items()
    }
    model.train(was_training)
    matches = int(host["completed_matches"].sum())
    blue_wins = int(host["completed_blue_wins"].sum())
    orange_wins = int(host["completed_orange_wins"].sum())
    blue_goals = int(host["completed_blue_goals"].sum())
    orange_goals = int(host["completed_orange_goals"].sum())
    blue_touches = int(host["completed_blue_touches"].sum())
    orange_touches = int(host["completed_orange_touches"].sum())
    total_ticks = int(host["completed_match_ticks"].sum())
    simulated_minutes = total_ticks / (120.0 * 60.0)
    kickoff_segments = int(host["kickoff_segments_total"].sum())
    no_touch_segments = int(host["no_touch_segments_total"].sum())
    denominator = side_decisions.clamp_min(1.0)
    action_mean = action_sum / denominator[:, None]
    action_abs_mean = action_abs_sum / denominator[:, None]
    action_rms = torch.sqrt(action_sq_sum / denominator[:, None])
    movement: dict[str, Any] = {}
    controller: dict[str, Any] = {}
    for side_index, side in enumerate(("Blue", "Orange")):
        movement[side] = {
            "mean_speed_uu_per_s": float(
                (speed_sum[side_index] / denominator[side_index]).item()
            ),
            "maximum_speed_uu_per_s": float(max_speed[side_index].item()),
            "mean_boost_level": float(
                (boost_sum[side_index] / denominator[side_index]).item()
            ),
            "boost_starved_fraction": float(
                (boost_starved[side_index] / denominator[side_index]).item()
            ),
            "grounded_fraction": float(
                (grounded[side_index] / denominator[side_index]).item()
            ),
            "airborne_fraction": float(
                1.0 - (grounded[side_index] / denominator[side_index]).item()
            ),
            "supersonic_fraction": float(
                (supersonic[side_index] / denominator[side_index]).item()
            ),
            "mean_car_ball_distance_uu": float(
                (distance_sum[side_index] / denominator[side_index]).item()
            ),
            "jump_rising_edges": int(jump_edges[side_index].item()),
            "jump_rising_edges_per_simulated_minute": float(
                jump_edges[side_index].item() / simulated_minutes
            ),
            "actual_flip_onsets": int(flip_onsets[side_index].item()),
            "actual_flip_onsets_per_simulated_minute": float(
                flip_onsets[side_index].item() / simulated_minutes
            ),
            "canonical_ball_progress_uu": float(ball_progress_uu[side_index].item()),
            "canonical_ball_progress_uu_per_simulated_minute": float(
                ball_progress_uu[side_index].item() / simulated_minutes
            ),
        }
        controller[side] = {
            name: {
                "mean": float(action_mean[side_index, channel].item()),
                "mean_absolute": float(action_abs_mean[side_index, channel].item()),
                "rms": float(action_rms[side_index, channel].item()),
                "active_fraction": (
                    float(action_mean[side_index, channel].item())
                    if channel >= 5
                    else None
                ),
            }
            for channel, name in enumerate(ACTION_NAMES)
        }
    duration_seconds = host["completed_match_ticks"].astype(np.float64) / 120.0
    overtime_matches = int(host["completed_overtime_matches"].sum())
    rng_after = _rng_snapshot(trainer)
    checks = {
        "all_worlds_completed_once": bool(np.all(host["completed_matches"] == 1)),
        "completed_world_count_exact": int(completed.sum().item())
        == EVALUATION_WORLDS,
        "no_truncation": int(truncated_count.item()) == 0,
        "wins_cover_matches": blue_wins + orange_wins == matches,
        "goal_totals_consistent": blue_goals + orange_goals
        == int(host["completed_match_goals"].sum()),
        "kickoff_denominator_nonzero": kickoff_segments > 0,
        "training_rng_unchanged": _rng_exact(rng_before, rng_after),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "checkpoint_label": label,
        "offset_updates": offset,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "reward_version": trainer.env.reward_version,
        "episode_version": trainer.env.episode_version,
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_worlds": EVALUATION_WORLDS,
        "mode": "stochastic current-policy self-play; one complete match per world",
        "result": {
            "completed_matches": matches,
            "blue_wins": blue_wins,
            "orange_wins": orange_wins,
            "goals": blue_goals + orange_goals,
            "goals_per_match": (blue_goals + orange_goals) / matches,
            "goals_per_simulated_minute": (blue_goals + orange_goals)
            / simulated_minutes,
            "blue_goals": blue_goals,
            "orange_goals": orange_goals,
            "score_distribution": _score_distribution(
                host["completed_blue_goals"], host["completed_orange_goals"]
            ),
            "touches": blue_touches + orange_touches,
            "blue_touches": blue_touches,
            "orange_touches": orange_touches,
            "touches_per_simulated_minute": (blue_touches + orange_touches)
            / simulated_minutes,
            "simulated_match_minutes": simulated_minutes,
            "regulation_completions": matches - overtime_matches,
            "regulation_completion_fraction": (matches - overtime_matches) / matches,
            "overtime_matches": overtime_matches,
            "overtime_fraction": overtime_matches / matches,
            "mean_match_duration_seconds": float(duration_seconds.mean()),
            "median_match_duration_seconds": float(np.median(duration_seconds)),
            "kickoff_segments": kickoff_segments,
            "counterfactual_no_touch_kickoff_segments": no_touch_segments,
            "counterfactual_no_touch_kickoff_segment_fraction": (
                no_touch_segments / kickoff_segments
            ),
            "first_touch_latency_seconds_contacted": {
                "count": int(first_touch_latency_count.item()),
                "mean": (
                    float(
                        first_touch_latency_decisions.item()
                        / first_touch_latency_count.item()
                        / 30.0
                    )
                    if first_touch_latency_count.item()
                    else None
                ),
            },
            "movement": movement,
            "controller": controller,
        },
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        "wall_seconds": time.perf_counter() - started,
    }
    del env, captured, active, completed
    gc.collect()
    torch.cuda.empty_cache()
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"scoring self-play evaluation failed: {checks}")
    return result


def _load_frozen_model(
    payload: dict[str, Any], trainer: Rival2Trainer
) -> Rival2ActorCritic:
    rng = _rng_snapshot(trainer)
    model = Rival2ActorCritic(trainer.policy_config).to(trainer.device)
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False)
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state(rng["torch_cuda"], trainer.device)
    trainer.policy_generator.set_state(rng["policy"])
    trainer.opponent_generator.set_state(rng["opponent"])
    return model


@torch.no_grad()
def evaluate_frozen_reference(
    *,
    trainer: Rival2Trainer,
    frozen_model: Rival2ActorCritic,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    label: str,
    offset: int,
) -> dict[str, Any]:
    """Run 256 current-Blue and 256 current-Orange full matches."""

    started = time.perf_counter()
    rng_before = _rng_snapshot(trainer)
    kickoff_selector = (
        np.arange(REFERENCE_WORLDS, dtype=np.int32) + REFERENCE_SEED
    ) % 5
    env = Rival2FullMatchEnv(
        REFERENCE_WORLDS,
        collision_dir,
        device=device,
        seed=REFERENCE_SEED,
        reward_version=RIVAL2_REWARD_SCORING_V1_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    current_side = torch.cat(
        (
            torch.zeros(
                REFERENCE_WORLDS_PER_SIDE, dtype=torch.long, device=env.device
            ),
            torch.ones(
                REFERENCE_WORLDS_PER_SIDE, dtype=torch.long, device=env.device
            ),
        )
    )
    opponent_side = 1 - current_side
    rows = torch.arange(REFERENCE_WORLDS, device=env.device)
    current_generator = torch.Generator(device=env.device).manual_seed(REFERENCE_SEED)
    frozen_generator = torch.Generator(device=env.device).manual_seed(
        REFERENCE_SEED ^ 0xA5A55A5A
    )
    was_training = trainer.model.training
    trainer.model.eval()
    active = torch.ones(REFERENCE_WORLDS, dtype=torch.bool, device=env.device)
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
        "completed_match_ticks",
    )
    captured = {
        name: torch.full(
            (REFERENCE_WORLDS,), -1, dtype=torch.int64, device=env.device
        )
        for name in capture_names
    }
    current_progress = torch.zeros(
        REFERENCE_WORLDS, dtype=torch.float64, device=env.device
    )
    frozen_progress = torch.zeros_like(current_progress)
    truncated_count = torch.zeros((), dtype=torch.int64, device=env.device)
    decision = 0
    while True:
        observation = env.observation
        current_observation = observation[rows, current_side]
        frozen_observation = observation[rows, opponent_side]
        current_actor, _ = trainer.model(current_observation)
        frozen_actor, _ = frozen_model(frozen_observation)
        current_action = sample_hybrid_action(
            current_actor,
            generator=current_generator,
            config=trainer.policy_config,
        ).action
        frozen_action = sample_hybrid_action(
            frozen_actor,
            generator=frozen_generator,
            config=trainer.policy_config,
        ).action
        action = torch.zeros(
            (REFERENCE_WORLDS, 2, 8), dtype=torch.float32, device=env.device
        )
        action[rows, current_side] = current_action
        action[rows, opponent_side] = frozen_action
        action = torch.where(active[:, None, None], action, torch.zeros_like(action))
        transition = env.step(action)
        after = transition.transition_observation
        current_progress += (
            (
                after[rows, current_side, _BALL_Y_INDEX]
                - observation[rows, current_side, _BALL_Y_INDEX]
            )
            * 5120.0
            * active
        ).to(torch.float64)
        frozen_progress += (
            (
                after[rows, opponent_side, _BALL_Y_INDEX]
                - observation[rows, opponent_side, _BALL_Y_INDEX]
            )
            * 5120.0
            * active
        ).to(torch.float64)
        truncated_count += (transition.truncated & active).sum()
        done = transition.terminated & active
        _capture_completed(env, captured, done)
        completed |= done
        active &= ~done
        decision += 1
        if decision % 30 == 0 and not bool(active.any().item()):
            break
        if decision % 900 == 0:
            print(
                f"scoring frozen eval={label} simulated_seconds={decision / 30:.1f} "
                f"complete={int(completed.sum().item())}/{REFERENCE_WORLDS}",
                flush=True,
            )

    torch.cuda.synchronize(env.device)
    host = {
        name: tensor.cpu().numpy().astype(np.int64, copy=True)
        for name, tensor in captured.items()
    }
    progress_host = current_progress.cpu().numpy()
    frozen_progress_host = frozen_progress.cpu().numpy()
    trainer.model.train(was_training)

    def summarize(start: int, stop: int, side_name: str) -> dict[str, Any]:
        selection = slice(start, stop)
        current_is_blue = side_name == "Blue"
        current_wins = (
            host["completed_blue_wins"][selection]
            if current_is_blue
            else host["completed_orange_wins"][selection]
        )
        frozen_wins = (
            host["completed_orange_wins"][selection]
            if current_is_blue
            else host["completed_blue_wins"][selection]
        )
        goals_for = (
            host["completed_blue_goals"][selection]
            if current_is_blue
            else host["completed_orange_goals"][selection]
        )
        goals_against = (
            host["completed_orange_goals"][selection]
            if current_is_blue
            else host["completed_blue_goals"][selection]
        )
        touches_for = (
            host["completed_blue_touches"][selection]
            if current_is_blue
            else host["completed_orange_touches"][selection]
        )
        touches_against = (
            host["completed_orange_touches"][selection]
            if current_is_blue
            else host["completed_blue_touches"][selection]
        )
        minutes = host["completed_match_ticks"][selection].sum() / 7200.0
        return {
            "matches": stop - start,
            "wins": int(current_wins.sum()),
            "losses": int(frozen_wins.sum()),
            "win_fraction": float(current_wins.mean()),
            "goals_for": int(goals_for.sum()),
            "goals_against": int(goals_against.sum()),
            "goal_differential": int(goals_for.sum() - goals_against.sum()),
            "touches_for": int(touches_for.sum()),
            "touches_against": int(touches_against.sum()),
            "touch_differential": int(touches_for.sum() - touches_against.sum()),
            "current_canonical_ball_progress_uu_per_simulated_minute": float(
                progress_host[selection].sum() / minutes
            ),
            "frozen_canonical_ball_progress_uu_per_simulated_minute": float(
                frozen_progress_host[selection].sum() / minutes
            ),
            "simulated_match_minutes": float(minutes),
        }

    blue = summarize(0, REFERENCE_WORLDS_PER_SIDE, "Blue")
    orange = summarize(REFERENCE_WORLDS_PER_SIDE, REFERENCE_WORLDS, "Orange")
    total_minutes = blue["simulated_match_minutes"] + orange["simulated_match_minutes"]
    overall = {
        "matches": REFERENCE_WORLDS,
        "wins": blue["wins"] + orange["wins"],
        "losses": blue["losses"] + orange["losses"],
        "win_fraction": (blue["wins"] + orange["wins"]) / REFERENCE_WORLDS,
        "goals_for": blue["goals_for"] + orange["goals_for"],
        "goals_against": blue["goals_against"] + orange["goals_against"],
        "goal_differential": blue["goal_differential"] + orange["goal_differential"],
        "touches_for": blue["touches_for"] + orange["touches_for"],
        "touches_against": blue["touches_against"] + orange["touches_against"],
        "touch_differential": blue["touch_differential"] + orange["touch_differential"],
        "current_canonical_ball_progress_uu_per_simulated_minute": float(
            progress_host.sum() / total_minutes
        ),
        "frozen_canonical_ball_progress_uu_per_simulated_minute": float(
            frozen_progress_host.sum() / total_minutes
        ),
        "simulated_match_minutes": total_minutes,
    }
    rng_after = _rng_snapshot(trainer)
    checks = {
        "all_worlds_completed_once": bool(np.all(host["completed_matches"] == 1)),
        "world_count_exact": int(completed.sum().item()) == REFERENCE_WORLDS,
        "side_counts_exact": blue["matches"] == orange["matches"]
        == REFERENCE_WORLDS_PER_SIDE,
        "no_truncation": int(truncated_count.item()) == 0,
        "wins_cover_matches": overall["wins"] + overall["losses"]
        == REFERENCE_WORLDS,
        "training_rng_unchanged": _rng_exact(rng_before, rng_after),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "checkpoint_label": label,
        "offset_updates": offset,
        "iteration": trainer.iteration,
        "agent_decision_samples": trainer.total_agent_samples,
        "mode": (
            "stochastic current scoring policy versus frozen acquisition policy; "
            "complete full matches"
        ),
        "current_policy_checkpoint": label,
        "frozen_policy_checkpoint": SOURCE_CHECKPOINT.as_posix(),
        "frozen_policy_sha256": SOURCE_CHECKPOINT_SHA256,
        "seed": REFERENCE_SEED,
        "result": {"overall": overall, "current_as_Blue": blue, "current_as_Orange": orange},
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        "wall_seconds": time.perf_counter() - started,
    }
    del env, captured, active, completed
    gc.collect()
    torch.cuda.empty_cache()
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"frozen acquisition evaluation failed: {checks}")
    return result


def transition_checkpoint(
    *,
    trainer: Rival2Trainer,
    source_checkpoint: Path,
    work_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    transition = trainer.load_checkpoint_curriculum_transition(
        source_checkpoint,
        source_reward_version=RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        source_episode_version=RIVAL2_EPISODE_VERSION,
        transition_record={
            "schema_version": SCHEMA_VERSION,
            "authority": AUTHORITY.as_posix(),
            "source_commit": SOURCE_COMMIT,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "authorized_change": (
                "completed short acquisition -> fresh full-match scoring curriculum"
            ),
        },
    )
    transition_path = work_dir / "checkpoints" / "rival2_scoring_transition_resume.pt"
    record = _checkpoint_record(transition_path, "transition", 0, trainer)
    destination_payload = torch.load(
        transition_path, map_location="cpu", weights_only=False
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
    fresh = trainer.env.full_match_views
    checks = {
        "source_sha256_exact": _sha256(source_checkpoint)
        == SOURCE_CHECKPOINT_SHA256,
        "all_compatible_fields_exact": all(field_checks.values()),
        "destination_reward_exact": trainer.env.reward_version
        == RIVAL2_REWARD_SCORING_V1_VERSION,
        "destination_episode_exact": trainer.env.episode_version
        == RIVAL2_FULL_MATCH_EPISODE_VERSION,
        "fresh_scores_zero": bool(
            torch.all(fresh["blue_score"] == 0).item()
            and torch.all(fresh["orange_score"] == 0).item()
        ),
        "fresh_regulation_clock_exact": bool(
            torch.all(
                fresh["regulation_ticks_remaining"] == 5 * 60 * 120
            ).item()
        ),
        "fresh_match_not_done": bool(torch.all(fresh["match_done"] == 0).item()),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "source_checkpoint": {
            "path": source_checkpoint.resolve().as_posix(),
            "sha256": SOURCE_CHECKPOINT_SHA256,
        },
        "transition": transition,
        "transition_checkpoint": record,
        "preserved_checkpoint_fields": field_checks,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    _write_json(work_dir / "checkpoint_transition.json", result)
    del source_payload, destination_payload
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"checkpoint transition failed: {checks}")
    return result, record


def _average_movement(evaluation: dict[str, Any], name: str) -> float:
    movement = evaluation["result"]["movement"]
    return (float(movement["Blue"][name]) + float(movement["Orange"][name])) / 2.0


def behavioral_assessment(
    baseline: dict[str, Any],
    final: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    base = baseline["result"]
    end = final["result"]
    duel = comparison["result"]["overall"]
    return {
        "A_scoring_improves": {
            "value": end["goals_per_simulated_minute"]
            > base["goals_per_simulated_minute"],
            "baseline_goals_per_minute": base["goals_per_simulated_minute"],
            "final_goals_per_minute": end["goals_per_simulated_minute"],
        },
        "B_goal_directed_ball_progress_improves": {
            "value": duel["current_canonical_ball_progress_uu_per_simulated_minute"]
            > duel["frozen_canonical_ball_progress_uu_per_simulated_minute"],
            "current": duel["current_canonical_ball_progress_uu_per_simulated_minute"],
            "frozen": duel["frozen_canonical_ball_progress_uu_per_simulated_minute"],
        },
        "C_acquisition_remains_intact": {
            "value": end["counterfactual_no_touch_kickoff_segment_fraction"]
            <= NO_TOUCH_WARNING_THRESHOLD,
            "warning_threshold": NO_TOUCH_WARNING_THRESHOLD,
            "final_fraction": end[
                "counterfactual_no_touch_kickoff_segment_fraction"
            ],
        },
        "D_flip_and_jump_spam_decreases": {
            "value": _average_movement(
                final, "actual_flip_onsets_per_simulated_minute"
            )
            < _average_movement(
                baseline, "actual_flip_onsets_per_simulated_minute"
            )
            and _average_movement(
                final, "jump_rising_edges_per_simulated_minute"
            )
            < _average_movement(
                baseline, "jump_rising_edges_per_simulated_minute"
            ),
            "baseline_flips_per_minute": _average_movement(
                baseline, "actual_flip_onsets_per_simulated_minute"
            ),
            "final_flips_per_minute": _average_movement(
                final, "actual_flip_onsets_per_simulated_minute"
            ),
            "baseline_jump_edges_per_minute": _average_movement(
                baseline, "jump_rising_edges_per_simulated_minute"
            ),
            "final_jump_edges_per_minute": _average_movement(
                final, "jump_rising_edges_per_simulated_minute"
            ),
        },
        "E_ground_air_balance_less_pathological": {
            "value": _average_movement(final, "grounded_fraction")
            > _average_movement(baseline, "grounded_fraction"),
            "baseline_grounded_fraction": _average_movement(
                baseline, "grounded_fraction"
            ),
            "final_grounded_fraction": _average_movement(
                final, "grounded_fraction"
            ),
        },
        "F_beats_frozen_acquisition": {
            "value": duel["wins"] > duel["losses"],
            "wins": duel["wins"],
            "losses": duel["losses"],
            "goal_differential": duel["goal_differential"],
            "touch_differential": duel["touch_differential"],
        },
    }


def run(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    launch_gate: dict[str, Any],
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
) -> dict[str, Any]:
    source_payload = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    policy_config = Rival2PolicyConfig(**source_payload["policy_config"])
    ppo_config = Rival2PPOConfig(**source_payload["ppo_config"])
    self_play_config = Rival2SelfPlayConfig(**source_payload["self_play_config"])
    kickoff_selector = (
        np.arange(WORLDS, dtype=np.int32) + CAMPAIGN_SEED
    ) % 5
    env = Rival2FullMatchEnv(
        WORLDS,
        args.collision_dir,
        device=args.device,
        seed=CAMPAIGN_SEED,
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
    transition, transition_checkpoint_record = transition_checkpoint(
        trainer=trainer,
        source_checkpoint=args.source_checkpoint,
        work_dir=args.work_dir,
    )
    frozen_model = _load_frozen_model(source_payload, trainer)
    del source_payload
    scale = reward_scale_accounting()
    _write_json(args.work_dir / "reward_scale.json", scale)
    if scale["verdict"] != "PASS_GREEN":
        raise RuntimeError("reward scale accounting failed")

    baseline = evaluate_self_play(
        trainer=trainer,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        label="plus_000_acquisition_baseline",
        offset=0,
    )
    _write_json(args.work_dir / "evaluation_plus_000.json", baseline)
    print(
        "scoring baseline "
        f"goals/min={baseline['result']['goals_per_simulated_minute']:.6f} "
        f"touches/min={baseline['result']['touches_per_simulated_minute']:.6f} "
        f"no_touch={baseline['result']['counterfactual_no_touch_kickoff_segment_fraction']:.6f}",
        flush=True,
    )

    start_iteration = trainer.iteration
    start_samples = trainer.total_agent_samples
    started = time.perf_counter()
    ledger = args.work_dir / "training_curve.jsonl"
    if ledger.exists():
        raise RuntimeError("work directory already contains a training ledger")
    checkpoints: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    next_checkpoint = 0
    while trainer.iteration - start_iteration < ADDITIONAL_UPDATES:
        full_match._train_one_update(
            phase="SCORING_V1_FULL_MATCH",
            phase_start_iteration=start_iteration,
            trainer=trainer,
            device=args.device,
            ledger=ledger,
        )
        offset = trainer.iteration - start_iteration
        if offset != CHECKPOINT_OFFSETS[next_checkpoint]:
            continue
        trainer.add_historical_snapshot()
        label = f"plus_{offset:03d}"
        record = _checkpoint_record(
            args.work_dir
            / "checkpoints"
            / f"rival2_scoring_{label}_resume.pt",
            label,
            offset,
            trainer,
        )
        audit = full_match._checkpoint_audit(record, trainer)
        if audit["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"checkpoint audit failed at {label}")
        record["audit"] = audit
        evaluation = evaluate_self_play(
            trainer=trainer,
            collision_dir=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
            label=label,
            offset=offset,
        )
        comparison = evaluate_frozen_reference(
            trainer=trainer,
            frozen_model=frozen_model,
            collision_dir=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
            label=label,
            offset=offset,
        )
        _write_json(args.work_dir / f"evaluation_{label}.json", evaluation)
        _write_json(args.work_dir / f"frozen_comparison_{label}.json", comparison)
        checkpoints.append(record)
        evaluations.append(evaluation)
        comparisons.append(comparison)
        _write_json(
            args.work_dir / "progress.json",
            {
                "checkpoints": checkpoints,
                "evaluations": evaluations,
                "frozen_comparisons": comparisons,
            },
        )
        duel = comparison["result"]["overall"]
        metric = evaluation["result"]
        print(
            f"scoring checkpoint={label} goals/min={metric['goals_per_simulated_minute']:.6f} "
            f"touches/min={metric['touches_per_simulated_minute']:.6f} "
            f"no_touch={metric['counterfactual_no_touch_kickoff_segment_fraction']:.6f} "
            f"vs_acquisition={duel['wins']}-{duel['losses']} "
            f"goal_diff={duel['goal_differential']}",
            flush=True,
        )
        next_checkpoint += 1

    expected_iteration = start_iteration + ADDITIONAL_UPDATES
    expected_samples = start_samples + ADDITIONAL_AGENT_SAMPLES
    final = evaluations[-1]
    final_comparison = comparisons[-1]
    assessment = behavioral_assessment(baseline, final, final_comparison)
    checks = {
        "additional_updates_exact": trainer.iteration == expected_iteration,
        "additional_samples_exact": trainer.total_agent_samples == expected_samples,
        "all_checkpoints_saved": next_checkpoint == len(CHECKPOINT_OFFSETS),
        "final_offset_exact": checkpoints[-1]["offset_updates"] == 239,
        "reward_contract_exact": trainer.env.contract_hashes
        == contract_hashes_for_reward(
            RIVAL2_REWARD_SCORING_V1_VERSION,
            RIVAL2_FULL_MATCH_EPISODE_VERSION,
        ),
        "no_nexto_training": True,
        "no_post_239_continuation": True,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "execution_status": "COMPLETE" if all(checks.values()) else "FAIL_RED",
        "source_head": launch_gate["head"],
        "configuration": configuration,
        "checkpoint_transition": transition,
        "transition_checkpoint": transition_checkpoint_record,
        "reward_scale": scale,
        "baseline_evaluation": baseline,
        "checkpoints": checkpoints,
        "evaluations": evaluations,
        "frozen_comparisons": comparisons,
        "behavioral_assessment": assessment,
        "start_iteration": start_iteration,
        "final_iteration": trainer.iteration,
        "start_agent_decision_samples": start_samples,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "additional_updates": trainer.iteration - start_iteration,
        "additional_agent_decision_samples": trainer.total_agent_samples
        - start_samples,
        "final_checkpoint": checkpoints[-1],
        "wall_seconds_including_evaluations": time.perf_counter() - started,
        "checks": checks,
    }
    _write_json(args.work_dir / "run_summary.json", summary)
    if summary["execution_status"] != "COMPLETE":
        raise RuntimeError(f"scoring curriculum boundary failed: {checks}")
    return summary


def _viewer_commands(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [
        {
            "label": "acquisition_start",
            "checkpoint": SOURCE_CHECKPOINT.as_posix(),
            "command": (
                ".\\.venv\\Scripts\\python.exe -m rivalsim.viewer --checkpoint "
                "checkpoints\\rival2\\acquisition_v1\\rival2_acquisition_resume.pt "
                "--stochastic --seed 20260827"
            ),
        }
    ]
    for record in checkpoints:
        filename = f"rival2_scoring_{record['label']}_resume.pt"
        entries.append(
            {
                "label": record["label"],
                "checkpoint": f"checkpoints/rival2/scoring_v1/{filename}",
                "command": (
                    ".\\.venv\\Scripts\\python.exe -m rivalsim.viewer --checkpoint "
                    f"checkpoints\\rival2\\scoring_v1\\{filename} "
                    "--stochastic --seed 20260827"
                ),
            }
        )
    return {"schema_version": SCHEMA_VERSION, "viewer_commands": entries}


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
        "checkpoint_transition.json",
        "reward_scale.json",
        "training_curve.jsonl",
        "evaluation_plus_000.json",
        "progress.json",
        "run_summary.json",
    ):
        shutil.copy2(args.work_dir / name, results / name)
    for path in sorted(args.work_dir.glob("evaluation_plus_*.json")):
        if path.name != "evaluation_plus_000.json":
            shutil.copy2(path, results / path.name)
    for path in sorted(args.work_dir.glob("frozen_comparison_plus_*.json")):
        shutil.copy2(path, results / path.name)

    checkpoint_dir = Path("checkpoints/rival2/scoring_v1")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    published_checkpoints: list[dict[str, Any]] = []
    for record in summary["checkpoints"]:
        source = Path(record["path"])
        destination = checkpoint_dir / f"rival2_scoring_{record['label']}_resume.pt"
        shutil.copy2(source, destination)
        published = {
            **copy.deepcopy(record),
            "path": destination.as_posix(),
            "sha256": _sha256(destination),
            "size_bytes": destination.stat().st_size,
        }
        published_checkpoints.append(published)
    _write_json(results / "checkpoints.json", {"checkpoints": published_checkpoints})
    viewer = _viewer_commands(published_checkpoints)
    _write_json(results / "viewer_commands.json", viewer)

    evaluation_rows = []
    for evaluation, comparison in zip(
        summary["evaluations"], summary["frozen_comparisons"], strict=True
    ):
        metric = evaluation["result"]
        movement = metric["movement"]
        duel = comparison["result"]["overall"]
        evaluation_rows.append(
            {
                "offset": evaluation["offset_updates"],
                "goals_per_match": metric["goals_per_match"],
                "goals_per_minute": metric["goals_per_simulated_minute"],
                "touches_per_minute": metric["touches_per_simulated_minute"],
                "no_touch_fraction": metric[
                    "counterfactual_no_touch_kickoff_segment_fraction"
                ],
                "mean_progress_uu_per_minute": (
                    movement["Blue"][
                        "canonical_ball_progress_uu_per_simulated_minute"
                    ]
                    + movement["Orange"][
                        "canonical_ball_progress_uu_per_simulated_minute"
                    ]
                )
                / 2.0,
                "mean_grounded_fraction": (
                    movement["Blue"]["grounded_fraction"]
                    + movement["Orange"]["grounded_fraction"]
                )
                / 2.0,
                "mean_jump_edges_per_minute": (
                    movement["Blue"]["jump_rising_edges_per_simulated_minute"]
                    + movement["Orange"]["jump_rising_edges_per_simulated_minute"]
                )
                / 2.0,
                "mean_flips_per_minute": (
                    movement["Blue"]["actual_flip_onsets_per_simulated_minute"]
                    + movement["Orange"]["actual_flip_onsets_per_simulated_minute"]
                )
                / 2.0,
                "reference_wins": duel["wins"],
                "reference_losses": duel["losses"],
                "reference_goal_differential": duel["goal_differential"],
                "reference_touch_differential": duel["touch_differential"],
            }
        )
    _write_json(results / "evaluation_table.json", evaluation_rows)

    lines = [
        "# Rival 2.0 scoring curriculum V1 results",
        "",
        "The bounded scoring stage resumed the exact completed acquisition checkpoint,",
        "changed only the explicit reward and episode contracts, initialized fresh full-match",
        "world state, trained exactly 239 updates, evaluated the four scheduled checkpoints,",
        "and stopped without a continuation.",
        "",
        "## Reward",
        "",
        f"- Contract: `{RIVAL2_REWARD_SCORING_V1_VERSION}`.",
        f"- Contract SHA-256: `{REWARD_SCORING_V1_CONTRACT_HASH}`.",
        f"- Equation: `{summary['reward_scale']['equation']}`.",
        "",
        "## Evaluation curve",
        "",
        "| Offset | Goals/match | Goals/min | Touches/min | No-touch | Grounded | "
        "Jump edges/min | Flips/min | vs acquisition W-L | Goal diff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation_rows:
        lines.append(
            f"| +{row['offset']} | {row['goals_per_match']:.6f} | "
            f"{row['goals_per_minute']:.6f} | {row['touches_per_minute']:.6f} | "
            f"{row['no_touch_fraction']:.6%} | {row['mean_grounded_fraction']:.6%} | "
            f"{row['mean_jump_edges_per_minute']:.6f} | "
            f"{row['mean_flips_per_minute']:.6f} | "
            f"{row['reference_wins']}-{row['reference_losses']} | "
            f"{row['reference_goal_differential']:+d} |"
        )
    final = published_checkpoints[-1]
    lines.extend(
        [
            "",
            "## Final boundary",
            "",
            f"- Final iteration: `{summary['final_iteration']}`.",
            f"- Final cumulative samples: `{summary['final_agent_decision_samples']:,}`.",
            f"- Final checkpoint: `{final['path']}`.",
            f"- Final checkpoint SHA-256: `{final['sha256']}`.",
            "- Wall time including evaluations: "
            f"`{summary['wall_seconds_including_evaluations']:.3f}` seconds.",
            "",
            "The six requested behavioral questions are recorded independently in",
            "`results/rival2/scoring_v1/run_summary.json`; no aggregate scalar pass/fail",
            "was invented for behavior.",
            "",
        ]
    )
    report = Path("docs/RIVAL2_SCORING_V1_RESULTS.md")
    report.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    manifest_paths = [
        *sorted(path for path in results.iterdir() if path.name != "manifest.json"),
        *[Path(item["path"]) for item in published_checkpoints],
        report,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "artifacts": [
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in manifest_paths
        ],
    }
    _write_json(results / "manifest.json", manifest)


def main() -> int:
    args = parse_args()
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise RuntimeError(f"work directory is not empty: {args.work_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    configuration = frozen_configuration(args.source_checkpoint)
    launch_gate = verify_launch(configuration, args.source_checkpoint)
    _write_json(args.work_dir / "config.json", configuration)
    _write_json(args.work_dir / "launch_gate.json", launch_gate)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    summary = run(args, configuration, launch_gate, geometry, meshes)
    publish(args, configuration, launch_gate, summary)
    print(
        "RIVAL2_SCORING_V1 COMPLETE "
        f"iteration={summary['final_iteration']} "
        f"samples={summary['final_agent_decision_samples']} "
        f"checkpoint_sha256={summary['final_checkpoint']['sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
