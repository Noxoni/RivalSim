"""Run the bounded acquisition-to-gameplay Rival 2.0 curriculum."""

from __future__ import annotations

import argparse
import gc
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmarks.run_rival2_acquisition_v1 as acquisition
import benchmarks.run_rival2_campaign01 as campaign01
from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    ACTION_NAMES,
    ANALOG_ACTION_NAMES,
    BALL_LINEAR_SPEED_SCALE,
    BUTTON_ACTION_NAMES,
    CAR_LINEAR_SPEED_SCALE,
    EPISODE_CONTRACT_HASH,
    GAMEPLAY_BIG_PAD_PICKUP_REWARD,
    GAMEPLAY_BOOST_USE_REWARD,
    GAMEPLAY_SAVE_REWARD,
    GAMEPLAY_SMALL_PAD_PICKUP_REWARD,
    GAMEPLAY_SPEED_COEFFICIENT,
    GAMEPLAY_SUPERSONIC_REWARD,
    OBS_DIM,
    OBS_FIELD_NAMES,
    REWARD_CONTRACT_HASH,
    REWARD_GAMEPLAY_V1_CONTRACT,
    REWARD_GAMEPLAY_V1_CONTRACT_HASH,
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_ACQUISITION_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import Rival2PolicyConfig, sample_hybrid_action
from rivalsim.rival2_ppo import (
    Rival2KLGuardConfig,
    Rival2PolicyDisplacementRejected,
    Rival2PPOConfig,
)
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer

SCHEMA_VERSION = 1
AUTHORITY = Path("handoff/rival2-gameplay-v1/README.md")
SOURCE_COMMIT = "61307571d86508f3026402c4948f759f310ff36c"
SOURCE_CHECKPOINT = Path("checkpoints/rival2/acquisition_v1/rival2_acquisition_resume.pt")
SOURCE_CHECKPOINT_SHA256 = "4FB7A3B134B25D595374E3968E2EDFA150A9CD6F8910B903BF892B59D7F8BC9A"
HISTORICAL_V1_HASH = "E3C97C7B3EA97D15F6AFB3AF21C40BAFBD206F0ED1124BAD6EA2C5A2ED14786F"

WORLDS = 131_072
CAMPAIGN_SEED = 20_260_827
EVALUATION_WORLDS = 4_096
EVALUATION_SEED = 920_260_827
MAX_EVALUATION_DECISIONS = 45 * 30
ADDITIONAL_UPDATES = 239
SAMPLES_PER_UPDATE = 8_388_608
ADDITIONAL_SAMPLES = ADDITIONAL_UPDATES * SAMPLES_PER_UPDATE
CHECKPOINT_OFFSETS = (60, 120, 180, 239)
KL_GUARD = Rival2KLGuardConfig(
    minibatch_kl_limit=0.10,
    completed_update_mean_kl_limit=0.05,
)

_SELF_VELOCITY_START = OBS_FIELD_NAMES.index("self.linear_velocity.x")
_SELF_BOOST_INDEX = OBS_FIELD_NAMES.index("self.boost")
_SELF_ON_GROUND_INDEX = OBS_FIELD_NAMES.index("self.on_ground")
_SELF_HAS_FLIPPED_INDEX = OBS_FIELD_NAMES.index("self.has_flipped")
_SELF_SUPERSONIC_INDEX = OBS_FIELD_NAMES.index("self.is_supersonic")
_NO_TOUCH_AGE_INDEX = OBS_FIELD_NAMES.index("lifecycle.no_touch_age")

COMPONENT_NAMES = (
    "v1_progress_component",
    "v1_goal_component",
    "v1_touch_component",
    "v1_demo_component",
    "speed_component",
    "supersonic_component",
    "boost_use_component",
    "boost_pickup_component",
    "save_component",
)
INCIDENTAL_COMPONENTS = (
    "speed_component",
    "supersonic_component",
    "boost_use_component",
    "boost_pickup_component",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/rival2/gameplay_v1"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    acquisition._write_json(path, value)


def _append_jsonl(path: Path, value: object) -> None:
    acquisition._append_jsonl(path, value)


def _sha256(path: Path) -> str:
    return acquisition._sha256(path)


def _nested_exact(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
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


def frozen_configuration(source: dict[str, Any]) -> dict[str, Any]:
    policy = Rival2PolicyConfig(**source["policy_config"])
    ppo = Rival2PPOConfig(**source["ppo_config"])
    self_play = Rival2SelfPlayConfig(**source["self_play_config"])
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY.as_posix(),
        "source_commit": SOURCE_COMMIT,
        "source_checkpoint": SOURCE_CHECKPOINT.as_posix(),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_iteration": int(source["iteration"]),
        "source_policy_version": int(source["policy_version"]),
        "source_agent_decision_samples": int(source["total_agent_samples"]),
        "source_reward_version": source["reward_version"],
        "source_episode_version": source["episode_version"],
        "destination_reward_version": RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        "destination_reward_contract": REWARD_GAMEPLAY_V1_CONTRACT,
        "destination_reward_contract_hash": REWARD_GAMEPLAY_V1_CONTRACT_HASH,
        "destination_episode_version": RIVAL2_EPISODE_VERSION,
        "destination_episode_contract_hash": EPISODE_CONTRACT_HASH,
        "policy_config": asdict(policy),
        "policy_config_hash": policy.content_hash,
        "ppo_config": asdict(ppo),
        "ppo_config_hash": ppo.content_hash,
        "self_play_config": asdict(self_play),
        "worlds": WORLDS,
        "campaign_seed": CAMPAIGN_SEED,
        "training": {
            "additional_updates": ADDITIONAL_UPDATES,
            "samples_per_update": SAMPLES_PER_UPDATE,
            "additional_agent_decision_samples": ADDITIONAL_SAMPLES,
            "checkpoint_offsets": list(CHECKPOINT_OFFSETS),
            "expected_final_iteration": 359,
            "expected_final_agent_decision_samples": (
                int(source["total_agent_samples"]) + ADDITIONAL_SAMPLES
            ),
        },
        "kl_guard": asdict(KL_GUARD),
        "evaluation": {
            "worlds": EVALUATION_WORLDS,
            "seed": EVALUATION_SEED,
            "labels": ["source", "plus_060", "plus_120", "plus_180", "plus_239"],
            "stochastic_current_policy_self_play": True,
            "one_original_short_episode_per_world": True,
        },
        "five_minute_matches": False,
        "nexto_training": False,
    }


def verify_launch(configuration: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"], check=True)
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/main")
    source_sha = _sha256(SOURCE_CHECKPOINT)
    expected_contracts = contract_hashes_for_reward(
        RIVAL2_REWARD_ACQUISITION_V1_VERSION, RIVAL2_EPISODE_VERSION
    )
    destination_contracts = contract_hashes_for_reward(
        RIVAL2_REWARD_GAMEPLAY_V1_VERSION, RIVAL2_EPISODE_VERSION
    )
    ppo = configuration["ppo_config"]
    checks = {
        "source_commit_is_ancestor": True,
        "head_pushed_to_origin_main": head == origin,
        "tracked_worktree_clean": subprocess.run(["git", "diff", "--quiet"]).returncode == 0,
        "index_clean": subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0,
        "authority_present": AUTHORITY.is_file(),
        "source_checkpoint_present": SOURCE_CHECKPOINT.is_file(),
        "source_checkpoint_sha256_exact": source_sha == SOURCE_CHECKPOINT_SHA256,
        "source_iteration_120": int(source["iteration"]) == 120,
        "source_policy_version_120": int(source["policy_version"]) == 120,
        "source_reward_acquisition_v1": source["reward_version"]
        == RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        "source_episode_short_v1": source["episode_version"] == RIVAL2_EPISODE_VERSION,
        "source_contracts_exact": source["contract_hashes"] == expected_contracts,
        "historical_v1_immutable": REWARD_CONTRACT_HASH == HISTORICAL_V1_HASH,
        "destination_contract_exact": destination_contracts[RIVAL2_REWARD_GAMEPLAY_V1_VERSION]
        == REWARD_GAMEPLAY_V1_CONTRACT_HASH,
        "destination_zero_sum": REWARD_GAMEPLAY_V1_CONTRACT["zero_sum"] is True,
        "world_count_exact": configuration["worlds"] == WORLDS,
        "entropy_zero": ppo["entropy_coefficient"] == 0.0,
        "learning_rate_unchanged": ppo["learning_rate"] == 0.0003,
        "clip_range_unchanged": ppo["clip_range"] == 0.2,
        "value_loss_coefficient_unchanged": ppo["value_loss_coefficient"] == 0.5,
        "max_gradient_norm_unchanged": ppo["max_gradient_norm"] == 0.5,
        "horizon_32": ppo["rollout_horizon"] == 32,
        "sample_arithmetic_exact": ADDITIONAL_SAMPLES == 2_004_877_312,
        "no_five_minute_matches": not configuration["five_minute_matches"],
        "no_nexto_training": not configuration["nexto_training"],
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "head": head,
        "origin_main": origin,
        "source_checkpoint_sha256": source_sha,
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"gameplay launch gate failed: {checks}")
    return result


def transition_preservation_gate(
    source: dict[str, Any], trainer: Rival2Trainer, transition: dict[str, Any]
) -> dict[str, Any]:
    destination = trainer.checkpoint_payload()
    preserved = (
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
    checks = {f"{name}_exact": _nested_exact(source[name], destination[name]) for name in preserved}
    checks.update(
        {
            "reward_is_only_contract_change": transition["changed_semantics"]
            == ["reward_contract", "fresh_world_state"],
            "episode_contract_unchanged": transition["source_episode_version"]
            == transition["destination_episode_version"]
            == RIVAL2_EPISODE_VERSION,
            "fresh_short_episode_state": bool(
                torch.all(trainer.env.bridge.views["rival2.episode_ticks"] == 0).item()
                and torch.all(trainer.env.bridge.views["rival2.no_touch_ticks"] == 0).item()
            ),
        }
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "preserved_fields": list(preserved),
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"gameplay transition gate failed: {checks}")
    return result


def _channels(values: torch.Tensor, names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        side: {
            name: float(values[side_index, channel].item()) for channel, name in enumerate(names)
        }
        for side_index, side in enumerate(("Blue", "Orange"))
    }


@torch.no_grad()
def evaluate(
    *,
    trainer: Rival2Trainer,
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    label: str,
) -> dict[str, Any]:
    """Run exactly one held-out original short episode in each world."""

    started = time.perf_counter()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(trainer.device).clone()
    policy_rng = trainer.policy_generator.get_state().clone()
    opponent_rng = trainer.opponent_generator.get_state().clone()
    was_training = trainer.model.training
    kickoff_selector = (np.arange(EVALUATION_WORLDS, dtype=np.int32) + EVALUATION_SEED) % 5
    env = Rival2Env(
        EVALUATION_WORLDS,
        collision_dir,
        device=device,
        seed=EVALUATION_SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        episode_version=RIVAL2_EPISODE_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    model = trainer.model
    model.eval()
    generator = torch.Generator(device=env.device).manual_seed(EVALUATION_SEED)
    active = torch.ones(EVALUATION_WORLDS, dtype=torch.bool, device=env.device)
    completed = torch.zeros_like(active)
    episode_decisions = torch.zeros(EVALUATION_WORLDS, dtype=torch.int32, device=env.device)
    side_touch = torch.zeros(2, dtype=torch.float64, device=env.device)
    goals = torch.zeros((), dtype=torch.float64, device=env.device)
    no_touch = torch.zeros((), dtype=torch.float64, device=env.device)
    hard = torch.zeros((), dtype=torch.float64, device=env.device)
    world_decisions = torch.zeros((), dtype=torch.float64, device=env.device)
    action_count = torch.zeros(2, dtype=torch.float64, device=env.device)
    action_sum = torch.zeros((2, 8), dtype=torch.float64, device=env.device)
    action_abs_sum = torch.zeros_like(action_sum)
    analog_saturation = torch.zeros((2, 5), dtype=torch.float64, device=env.device)
    analog_policy_std_sum = torch.zeros((2, 5), dtype=torch.float64, device=env.device)
    button_probability_sum = torch.zeros((2, 3), dtype=torch.float64, device=env.device)
    button_activation_sum = torch.zeros((2, 3), dtype=torch.float64, device=env.device)
    boost_level_sum = torch.zeros(2, dtype=torch.float64, device=env.device)
    boost_consumed_sum = torch.zeros(2, dtype=torch.float64, device=env.device)
    boost_use_intervals = torch.zeros(2, dtype=torch.float64, device=env.device)
    small_pickups = torch.zeros(2, dtype=torch.float64, device=env.device)
    big_pickups = torch.zeros(2, dtype=torch.float64, device=env.device)
    speed_sum = torch.zeros(2, dtype=torch.float64, device=env.device)
    supersonic = torch.zeros(2, dtype=torch.float64, device=env.device)
    grounded = torch.zeros(2, dtype=torch.float64, device=env.device)
    jump_edges = torch.zeros(2, dtype=torch.float64, device=env.device)
    flip_onsets = torch.zeros(2, dtype=torch.float64, device=env.device)
    saves = torch.zeros(2, dtype=torch.float64, device=env.device)
    previous_jump = torch.zeros((EVALUATION_WORLDS, 2), dtype=torch.bool, device=env.device)
    reward_total = torch.zeros(2, dtype=torch.float64, device=env.device)
    component_total = {
        name: torch.zeros((), dtype=torch.float64, device=env.device) for name in COMPONENT_NAMES
    }
    component_abs_total = {
        name: torch.zeros((), dtype=torch.float64, device=env.device) for name in COMPONENT_NAMES
    }
    positive_shaping_total = {
        name: torch.zeros(2, dtype=torch.float64, device=env.device)
        for name in ("speed", "supersonic", "boost_use", "boost_pickup", "save")
    }
    zero_sum_max_error = torch.zeros((), dtype=torch.float32, device=env.device)
    component_identity_max_error = torch.zeros((), dtype=torch.float32, device=env.device)

    for _decision in range(MAX_EVALUATION_DECISIONS):
        observation = env.observation
        actor, _value = model(observation.reshape(-1, OBS_DIM))
        actor = actor.reshape(EVALUATION_WORLDS, 2, 13)
        sample = sample_hybrid_action(actor, generator=generator, config=trainer.policy_config)
        action = torch.where(active[:, None, None], sample.action, torch.zeros_like(sample.action))
        mask = active[:, None]
        mask3 = mask[..., None]
        mask_float = mask.to(torch.float32)
        active_count = active.sum().double()
        action_count += active_count
        action_sum += (action * mask3).sum(dim=0, dtype=torch.float64)
        action_abs_sum += (action.abs() * mask3).sum(dim=0, dtype=torch.float64)
        analog_saturation += ((action[..., :5].abs() > 0.95) & mask3).sum(
            dim=0, dtype=torch.float64
        )
        log_std = actor[..., 5:10].clamp(
            trainer.policy_config.log_std_min, trainer.policy_config.log_std_max
        )
        analog_policy_std_sum += (torch.exp(log_std) * mask3).sum(dim=0, dtype=torch.float64)
        button_probability_sum += (torch.sigmoid(actor[..., 10:13]) * mask3).sum(
            dim=0, dtype=torch.float64
        )
        button_activation_sum += (action[..., 5:8] * mask3).sum(dim=0, dtype=torch.float64)

        current_jump = action[..., 5] > 0.5
        jump_edges += (current_jump & ~previous_jump & mask).sum(dim=0, dtype=torch.float64)
        previous_jump.copy_(current_jump & mask)
        boost_before = observation[..., _SELF_BOOST_INDEX] * 100.0
        boost_level_sum += (boost_before * mask_float).sum(dim=0, dtype=torch.float64)
        velocity_before = (
            observation[..., _SELF_VELOCITY_START : _SELF_VELOCITY_START + 3]
            * CAR_LINEAR_SPEED_SCALE
        )
        speed_before = torch.linalg.vector_norm(velocity_before, dim=-1)
        speed_sum += (speed_before * mask_float).sum(dim=0, dtype=torch.float64)
        supersonic += ((observation[..., _SELF_SUPERSONIC_INDEX] > 0.5) & mask).sum(
            dim=0, dtype=torch.float64
        )
        grounded += ((observation[..., _SELF_ON_GROUND_INDEX] > 0.5) & mask).sum(
            dim=0, dtype=torch.float64
        )
        flipped_before = observation[..., _SELF_HAS_FLIPPED_INDEX] > 0.5

        transition = env.step(action)
        episode_decisions += active.to(torch.int32)
        world_decisions += active_count
        transition_observation = transition.transition_observation
        reward_total += (transition.reward * mask_float).sum(dim=0, dtype=torch.float64)
        zero_sum_max_error = torch.maximum(
            zero_sum_max_error,
            (transition.reward.sum(dim=-1) * active.to(torch.float32)).abs().amax(),
        )
        component_blue = torch.zeros(EVALUATION_WORLDS, dtype=torch.float32, device=env.device)
        for name in COMPONENT_NAMES:
            values = env.bridge.views[f"rival2.{name}"]
            component_total[name] += (values * active).sum(dtype=torch.float64)
            component_abs_total[name] += (values.abs() * active).sum(dtype=torch.float64)
            component_blue += values
        component_identity_max_error = torch.maximum(
            component_identity_max_error,
            ((transition.reward[:, 0] - component_blue) * active.to(torch.float32)).abs().amax(),
        )

        touch_count = env.bridge.views["rival2.touch_count"].reshape(EVALUATION_WORLDS, 2)
        side_touch += (touch_count * mask).sum(dim=0, dtype=torch.float64)
        boost_after = transition_observation[..., _SELF_BOOST_INDEX] * 100.0
        boost_gained = env.bridge.views["rival2.boost_gained_amount"].reshape(EVALUATION_WORLDS, 2)
        boost_consumed_sum += (
            (boost_before + boost_gained - boost_after).clamp_min(0.0) * mask_float
        ).sum(dim=0, dtype=torch.float64)
        boost_use = env.bridge.views["rival2.boost_use_event"].reshape(EVALUATION_WORLDS, 2)
        boost_use_intervals += (boost_use * mask).sum(dim=0, dtype=torch.float64)
        small = env.bridge.views["rival2.small_pad_pickup_count"].reshape(EVALUATION_WORLDS, 2)
        big = env.bridge.views["rival2.big_pad_pickup_count"].reshape(EVALUATION_WORLDS, 2)
        save = env.bridge.views["rival2.save_count"].reshape(EVALUATION_WORLDS, 2)
        small_pickups += (small * mask).sum(dim=0, dtype=torch.float64)
        big_pickups += (big * mask).sum(dim=0, dtype=torch.float64)
        saves += (save * mask).sum(dim=0, dtype=torch.float64)
        velocity_after = (
            transition_observation[..., _SELF_VELOCITY_START : _SELF_VELOCITY_START + 3]
            * CAR_LINEAR_SPEED_SCALE
        )
        speed_after = torch.linalg.vector_norm(velocity_after, dim=-1)
        supersonic_after = transition_observation[..., _SELF_SUPERSONIC_INDEX] > 0.5
        positive_shaping_total["speed"] += (
            GAMEPLAY_SPEED_COEFFICIENT
            * (speed_after / CAR_LINEAR_SPEED_SCALE).clamp(0.0, 1.0)
            * mask_float
        ).sum(dim=0, dtype=torch.float64)
        positive_shaping_total["supersonic"] += (
            GAMEPLAY_SUPERSONIC_REWARD * supersonic_after.to(torch.float32) * mask_float
        ).sum(dim=0, dtype=torch.float64)
        positive_shaping_total["boost_use"] += (GAMEPLAY_BOOST_USE_REWARD * boost_use * mask).sum(
            dim=0, dtype=torch.float64
        )
        positive_shaping_total["boost_pickup"] += (
            (GAMEPLAY_SMALL_PAD_PICKUP_REWARD * small + GAMEPLAY_BIG_PAD_PICKUP_REWARD * big) * mask
        ).sum(dim=0, dtype=torch.float64)
        positive_shaping_total["save"] += (GAMEPLAY_SAVE_REWARD * save * mask).sum(
            dim=0, dtype=torch.float64
        )
        flipped_after = transition_observation[..., _SELF_HAS_FLIPPED_INDEX] > 0.5
        flip_onsets += (flipped_after & ~flipped_before & mask).sum(dim=0, dtype=torch.float64)

        done = active & (transition.terminated | transition.truncated)
        goals += (done & transition.terminated).sum().double()
        no_touch_now = (
            done & transition.truncated & (transition_observation[:, 0, _NO_TOUCH_AGE_INDEX] >= 1.0)
        )
        no_touch += no_touch_now.sum().double()
        hard += (done & transition.truncated & ~no_touch_now).sum().double()
        completed |= done
        active &= ~done
        if not bool(active.any().item()):
            break

    torch.cuda.synchronize(env.device)
    episodes = int(completed.sum().item())
    decisions = float(world_decisions.item())
    simulated_minutes = decisions / 30.0 / 60.0
    denominator = action_count.clamp_min(1.0)
    action_mean = action_sum / denominator[:, None]
    action_abs_mean = action_abs_sum / denominator[:, None]
    movement: dict[str, Any] = {}
    for side_index, side in enumerate(("Blue", "Orange")):
        movement[side] = {
            "mean_speed_uu_per_s": float((speed_sum[side_index] / denominator[side_index]).item()),
            "supersonic_fraction": float((supersonic[side_index] / denominator[side_index]).item()),
            "grounded_fraction": float((grounded[side_index] / denominator[side_index]).item()),
            "airborne_fraction": float(
                1.0 - (grounded[side_index] / denominator[side_index]).item()
            ),
            "average_boost_level": float(
                (boost_level_sum[side_index] / denominator[side_index]).item()
            ),
            "net_observed_boost_consumed": float(boost_consumed_sum[side_index].item()),
            "boost_use_interval_fraction": float(
                (boost_use_intervals[side_index] / denominator[side_index]).item()
            ),
            "small_boost_pad_pickups": int(small_pickups[side_index].item()),
            "large_boost_pad_pickups": int(big_pickups[side_index].item()),
            "jump_rising_edges": int(jump_edges[side_index].item()),
            "actual_flip_onsets": int(flip_onsets[side_index].item()),
            "flips_per_simulated_minute": float(flip_onsets[side_index].item() / simulated_minutes),
            "save_events": int(saves[side_index].item()),
        }
    competitive = {name: float(value.item()) for name, value in component_total.items()}
    competitive_abs = {name: float(value.item()) for name, value in component_abs_total.items()}
    v1_total = sum(competitive[name] for name in COMPONENT_NAMES[:4])
    added_total = sum(competitive[name] for name in COMPONENT_NAMES[4:])
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "checkpoint_label": label,
        "iteration": trainer.iteration,
        "policy_version": trainer.policy_version,
        "agent_decision_samples": trainer.total_agent_samples,
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_worlds": EVALUATION_WORLDS,
        "episode_version": RIVAL2_EPISODE_VERSION,
        "reward_version": RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        "result": {
            "completed_episodes": episodes,
            "goal_terminated_episodes": int(goals.item()),
            "goal_terminated_fraction": float(goals.item()) / episodes,
            "no_touch_truncated_episodes": int(no_touch.item()),
            "no_touch_truncated_fraction": float(no_touch.item()) / episodes,
            "hard_truncated_episodes": int(hard.item()),
            "hard_truncated_fraction": float(hard.item()) / episodes,
            "mean_episode_duration_seconds": decisions / episodes / 30.0,
            "touches_per_simulated_minute": float(side_touch.sum().item()) / simulated_minutes,
            "goals_per_simulated_minute": float(goals.item()) / simulated_minutes,
            "save_events_per_simulated_minute": float(saves.sum().item()) / simulated_minutes,
            "unique_touches": {
                "Blue": int(side_touch[0].item()),
                "Orange": int(side_touch[1].item()),
            },
            "controller_mean": _channels(action_mean, ACTION_NAMES),
            "controller_mean_absolute": _channels(action_abs_mean, ACTION_NAMES),
            "analog_saturation_fraction": _channels(
                analog_saturation / denominator[:, None], ANALOG_ACTION_NAMES
            ),
            "analog_policy_std": _channels(
                analog_policy_std_sum / denominator[:, None], ANALOG_ACTION_NAMES
            ),
            "button_activation_fraction": _channels(
                button_activation_sum / denominator[:, None], BUTTON_ACTION_NAMES
            ),
            "button_probability": _channels(
                button_probability_sum / denominator[:, None], BUTTON_ACTION_NAMES
            ),
            "movement": movement,
            "reward_components": {
                "competitive_blue_totals": competitive,
                "competitive_orange_totals": {name: -value for name, value in competitive.items()},
                "competitive_blue_mean_absolute_totals": competitive_abs,
                "historical_v1_blue_total": v1_total,
                "added_gameplay_blue_total": added_total,
                "player_positive_totals": {
                    name: {
                        side: float(values[side_index].item())
                        for side_index, side in enumerate(("Blue", "Orange"))
                    }
                    for name, values in positive_shaping_total.items()
                },
                "observed_total_reward": {
                    "Blue": float(reward_total[0].item()),
                    "Orange": float(reward_total[1].item()),
                },
                "per_interval_component_identity_max_error": float(
                    component_identity_max_error.item()
                ),
            },
            "simulated_minutes": simulated_minutes,
            "world_decision_intervals": decisions,
        },
        "checks": {
            "all_worlds_completed_once": episodes == EVALUATION_WORLDS,
            "done_partition_exact": int(goals.item() + no_touch.item() + hard.item()) == episodes,
            "reward_zero_sum_exact": float(zero_sum_max_error.item()) == 0.0,
            "reward_component_identity": float(component_identity_max_error.item()) <= 2.0e-6,
            "no_approach_component": "approach_component" not in competitive,
            "short_episode_boundary": decisions <= EVALUATION_WORLDS * MAX_EVALUATION_DECISIONS,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    result["verdict"] = "PASS_GREEN" if all(result["checks"].values()) else "FAIL_RED"
    model.train(was_training)
    del env
    gc.collect()
    torch.cuda.empty_cache()
    trainer.policy_generator.set_state(policy_rng)
    trainer.opponent_generator.set_state(opponent_rng)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, trainer.device)
    if result["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"gameplay evaluation failed: {result['checks']}")
    return result


def reward_scale_sanity(source_evaluation: dict[str, Any]) -> dict[str, Any]:
    seconds_per_decision = 1.0 / 30.0
    max_progress_per_decision = 0.5 * BALL_LINEAR_SPEED_SCALE * seconds_per_decision / 5120.0
    theoretical = {
        "goal_event": 10.0,
        "save_event": GAMEPLAY_SAVE_REWARD,
        "v1_ball_progress_max_at_6000_uu_per_s_one_decision": max_progress_per_decision,
        "v1_touch_event": 0.05,
        "v1_demo_event": 0.10,
        "speed_max_per_player_decision": GAMEPLAY_SPEED_COEFFICIENT,
        "supersonic_per_player_decision": GAMEPLAY_SUPERSONIC_REWARD,
        "boost_use_per_player_decision": GAMEPLAY_BOOST_USE_REWARD,
        "small_pad_positive_gain_event": GAMEPLAY_SMALL_PAD_PICKUP_REWARD,
        "large_pad_positive_gain_event": GAMEPLAY_BIG_PAD_PICKUP_REWARD,
        "speed_max_one_player_45_seconds": GAMEPLAY_SPEED_COEFFICIENT * 45 * 30,
        "supersonic_max_one_player_45_seconds": GAMEPLAY_SUPERSONIC_REWARD * 45 * 30,
        "boost_use_max_one_player_45_seconds": GAMEPLAY_BOOST_USE_REWARD * 45 * 30,
    }
    result = source_evaluation["result"]
    decisions = result["world_decision_intervals"]
    observed_abs = result["reward_components"]["competitive_blue_mean_absolute_totals"]
    observed_per_decision = {name: value / decisions for name, value in observed_abs.items()}
    incidental_per_decision = sum(observed_per_decision[name] for name in INCIDENTAL_COMPONENTS)
    checks = {
        "goal_greater_than_save": theoretical["goal_event"] > theoretical["save_event"],
        "save_greater_than_v1_demo": theoretical["save_event"] > theoretical["v1_demo_event"],
        "v1_touch_greater_than_any_incidental_unit": theoretical["v1_touch_event"]
        > max(
            theoretical["speed_max_per_player_decision"],
            theoretical["supersonic_per_player_decision"],
            theoretical["boost_use_per_player_decision"],
            theoretical["small_pad_positive_gain_event"],
            theoretical["large_pad_positive_gain_event"],
        ),
        "observed_incidental_mean_abs_below_one_v1_touch_per_decision": (
            incidental_per_decision < theoretical["v1_touch_event"]
        ),
        "historical_v1_hash_unchanged": REWARD_CONTRACT_HASH == HISTORICAL_V1_HASH,
        "zero_sum_evaluation": source_evaluation["checks"]["reward_zero_sum_exact"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "theoretical": theoretical,
        "observed_source_mean_absolute_competitive_component_per_decision": (observed_per_decision),
        "observed_source_incidental_mean_absolute_per_decision": (incidental_per_decision),
        "ordering": (
            "goal > save > normal V1 event shaping > incidental movement/resource unit shaping"
        ),
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _training_integrity(
    trainer: Rival2Trainer,
    rollout: Any,
    metrics: dict[str, torch.Tensor],
    *,
    policy_before: int,
    samples_before: int,
) -> dict[str, Any]:
    checks = {
        "finite_metrics": all(torch.isfinite(value).item() for value in metrics.values()),
        "finite_rewards": bool(torch.isfinite(rollout.rewards).all().item()),
        "finite_returns": bool(torch.isfinite(rollout.returns).all().item()),
        "finite_advantages": bool(torch.isfinite(rollout.advantages).all().item()),
        "reward_zero_sum_exact": bool((rollout.rewards.sum(dim=-1) == 0.0).all().item()),
        "termination_team_consistent": bool(
            torch.equal(rollout.terminated[..., 0], rollout.terminated[..., 1])
        ),
        "truncation_team_consistent": bool(
            torch.equal(rollout.truncated[..., 0], rollout.truncated[..., 1])
        ),
        "policy_increment_exact": trainer.policy_version == policy_before + 1,
        "iteration_matches_policy": trainer.iteration == trainer.policy_version,
        "sample_increment_exact": trainer.total_agent_samples - samples_before
        == SAMPLES_PER_UPDATE,
        "completed_update_kl_within_guard": float(metrics["approx_kl"].item())
        <= KL_GUARD.completed_update_mean_kl_limit,
        "minibatch_kl_within_guard": float(metrics["optimizer_post_step_approx_kl_max"].item())
        <= KL_GUARD.minibatch_kl_limit,
        "hot_path_zero_transfer": trainer.env.hot_path_transfer_bytes() == {"h2d": 0, "d2h": 0},
    }
    return {
        "checks": checks,
        "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
    }


def _checkpoint(label: str, trainer: Rival2Trainer, work_dir: Path) -> dict[str, Any]:
    path = work_dir / "checkpoints" / f"rival2_gameplay_{label}_resume.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    finite_model = all(torch.isfinite(value).all().item() for value in payload["model"].values())
    checks = {
        "format_exact": payload["format"] == "RIVAL2_CHECKPOINT_V1",
        "iteration_exact": int(payload["iteration"]) == trainer.iteration,
        "policy_version_exact": int(payload["policy_version"]) == trainer.policy_version,
        "sample_count_exact": int(payload["total_agent_samples"]) == trainer.total_agent_samples,
        "reward_exact": payload["reward_version"] == RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        "episode_exact": payload["episode_version"] == RIVAL2_EPISODE_VERSION,
        "contracts_exact": payload["contract_hashes"] == trainer.env.contract_hashes,
        "model_finite": finite_model,
        "curriculum_transition_present": "curriculum_transition" in payload,
    }
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
        "audit": {
            "checks": checks,
            "verdict": "PASS_GREEN" if all(checks.values()) else "FAIL_RED",
        },
    }


def _write_report(
    destination: Path,
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> None:
    lines = [
        "# Rival 2.0 Gameplay V1 bounded curriculum",
        "",
        f"Status: `{summary['status']}`.",
        "",
        f"Source commit: `{SOURCE_COMMIT}`.",
        "",
        f"Source checkpoint SHA-256: `{SOURCE_CHECKPOINT_SHA256}`.",
        "",
        f"Gameplay reward contract SHA-256: `{REWARD_GAMEPLAY_V1_CONTRACT_HASH}`.",
        "",
        (
            "The run resumed only the pinned acquisition-complete learned/training "
            "state into fresh `RIVAL2_EPISODE_V1` worlds. Historical "
            "`RIVAL2_REWARD_V1` remained immutable; only the new reward identity "
            "and fresh short-lifecycle state changed."
        ),
        "",
        "```text",
        "Blue = historical_V1_blue",
        "     + speed(Blue) - speed(Orange)",
        "     + supersonic(Blue) - supersonic(Orange)",
        "     + actual_boost_use(Blue) - actual_boost_use(Orange)",
        "     + positive_boost_pickups(Blue) - positive_boost_pickups(Orange)",
        "     + 0.75 * BlueSaves - 0.75 * OrangeSaves",
        "Orange = -Blue",
        "```",
        "",
        "## Held-out curve",
        "",
        (
            "| label | iteration | samples | touches/min | goals/min | goal fraction "
            "| no-touch | hard-time | mean seconds | saves/min |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in evaluations:
        result = item["result"]
        lines.append(
            f"| {item['checkpoint_label']} | {item['iteration']} | "
            f"{item['agent_decision_samples']} | "
            f"{result['touches_per_simulated_minute']:.6f} | "
            f"{result['goals_per_simulated_minute']:.6f} | "
            f"{result['goal_terminated_fraction']:.6f} | "
            f"{result['no_touch_truncated_fraction']:.6f} | "
            f"{result['hard_truncated_fraction']:.6f} | "
            f"{result['mean_episode_duration_seconds']:.3f} | "
            f"{result['save_events_per_simulated_minute']:.6f} |"
        )
    if summary["status"] == "STOPPED_KL_GUARD_REJECTION":
        diagnostic = summary["diagnostic"]
        lines.extend(
            [
                "",
                "## Mandatory policy-displacement stop",
                "",
                f"Rejected update: `{diagnostic['rejected_iteration']}`.",
                "",
                f"Reason: `{diagnostic['reason']}`.",
                "",
                (
                    "The transactional pre-update model, optimizer, gradients, and "
                    "RNG state were restored. No later update was run."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            "| label | iteration | samples | SHA-256 | audit |",
            "|---|---:|---:|---|---|",
        ]
    )
    for checkpoint in checkpoints:
        lines.append(
            f"| {checkpoint['label']} | {checkpoint['iteration']} | "
            f"{checkpoint['agent_decision_samples']} | "
            f"`{checkpoint['sha256']}` | `{checkpoint['audit']['verdict']}` |"
        )
    lines.extend(
        [
            "",
            "## Safety boundary",
            "",
            (
                "Every update used a transactional post-step minibatch KL limit of "
                f"`{KL_GUARD.minibatch_kl_limit}` and a complete-rollout final-policy "
                "mean KL limit of "
                f"`{KL_GUARD.completed_update_mean_kl_limit}`. A violation restores "
                "model, optimizer, gradients, and RNG state and ends the run; no "
                "automatic retuning is permitted."
            ),
            "",
            (
                "Full per-update optimizer, actor-distribution, "
                "value/return/advantage, saturation, and rollout reward evidence, "
                "plus held-out reward-component, movement, save, and boost evidence, "
                "is machine-readable under "
                "`results/rival2/gameplay_v1/`."
            ),
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def publish_complete(
    args: argparse.Namespace,
    summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> None:
    results = args.results_dir.resolve()
    if results.exists() and any(results.iterdir()):
        raise RuntimeError("results directory must be absent or empty")
    results.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "launch_gate.json",
        "transition.json",
        "transition_gate.json",
        "reward_scale_sanity.json",
        "training_curve.jsonl",
        "evaluation_curve.json",
        "checkpoints.json",
        "run_summary.json",
    ):
        shutil.copy2(args.work_dir / name, results / name)
    for path in sorted(args.work_dir.glob("evaluation_*.json")):
        shutil.copy2(path, results / path.name)
    final_source = Path(checkpoints[-1]["path"])
    final_destination = Path("checkpoints/rival2/gameplay_v1/rival2_gameplay_resume.pt")
    final_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_source, final_destination)
    _write_json(
        results / "final_checkpoint.json",
        {
            "path": final_destination.as_posix(),
            "sha256": _sha256(final_destination),
            "size_bytes": final_destination.stat().st_size,
            "source_work_path": final_source.as_posix(),
        },
    )
    _write_report(
        Path("docs/RIVAL2_GAMEPLAY_V1_RESULTS.md"),
        summary,
        evaluations,
        checkpoints,
    )


def publish_rejection(
    args: argparse.Namespace,
    summary: dict[str, Any],
    diagnostic: dict[str, Any],
    checkpoint: dict[str, Any],
    evaluations: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> None:
    results = args.results_dir.resolve()
    if results.exists() and any(results.iterdir()):
        raise RuntimeError("results directory must be absent or empty")
    results.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "launch_gate.json",
        "transition.json",
        "transition_gate.json",
        "reward_scale_sanity.json",
        "training_curve.jsonl",
        "evaluation_curve.json",
        "checkpoints.json",
        "kl_rejection.json",
        "run_summary.json",
    ):
        source = args.work_dir / name
        if source.is_file():
            shutil.copy2(source, results / name)
    for path in sorted(args.work_dir.glob("evaluation_*.json")):
        shutil.copy2(path, results / path.name)
    restored_source = Path(checkpoint["path"])
    restored_destination = Path(
        "checkpoints/rival2/gameplay_v1/rival2_gameplay_rejected_pre_update_resume.pt"
    )
    restored_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(restored_source, restored_destination)
    _write_json(
        results / "rejected_pre_update_checkpoint.json",
        {
            "path": restored_destination.as_posix(),
            "sha256": _sha256(restored_destination),
            "diagnostic_reason": diagnostic["reason"],
        },
    )
    _write_report(
        Path("docs/RIVAL2_GAMEPLAY_V1_RESULTS.md"),
        summary,
        evaluations,
        [*checkpoints, checkpoint],
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    configuration = frozen_configuration(source)
    launch = verify_launch(configuration, source)
    _write_json(args.work_dir / "config.json", configuration)
    _write_json(args.work_dir / "launch_gate.json", launch)
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    kickoff_selector = (np.arange(WORLDS, dtype=np.int32) + CAMPAIGN_SEED) % 5
    env = Rival2Env(
        WORLDS,
        args.collision_dir,
        device=args.device,
        seed=CAMPAIGN_SEED,
        reward_version=RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        episode_version=RIVAL2_EPISODE_VERSION,
        geometry=geometry,
        meshes=meshes,
        kickoff_selector=kickoff_selector,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2Trainer(
        env,
        policy_config=Rival2PolicyConfig(**source["policy_config"]),
        ppo_config=Rival2PPOConfig(**source["ppo_config"]),
        self_play_config=Rival2SelfPlayConfig(**source["self_play_config"]),
        seed=CAMPAIGN_SEED,
    )
    transition = trainer.load_checkpoint_curriculum_transition(
        SOURCE_CHECKPOINT,
        source_reward_version=RIVAL2_REWARD_ACQUISITION_V1_VERSION,
        source_episode_version=RIVAL2_EPISODE_VERSION,
        transition_record={
            "schema_version": SCHEMA_VERSION,
            "authority": AUTHORITY.as_posix(),
            "authorized_change": (
                "acquisition-complete checkpoint -> fresh short-lifecycle Gameplay V1 reward"
            ),
            "source_commit": SOURCE_COMMIT,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "collapsed_scoring_v1_lineage_used": False,
        },
    )
    transition_gate = transition_preservation_gate(source, trainer, transition)
    _write_json(args.work_dir / "transition.json", transition)
    _write_json(args.work_dir / "transition_gate.json", transition_gate)

    evaluations: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    source_evaluation = evaluate(
        trainer=trainer,
        collision_dir=args.collision_dir,
        geometry=geometry,
        meshes=meshes,
        device=args.device,
        label="source",
    )
    evaluations.append(source_evaluation)
    _write_json(args.work_dir / "evaluation_source.json", source_evaluation)
    _write_json(args.work_dir / "evaluation_curve.json", evaluations)
    sanity = reward_scale_sanity(source_evaluation)
    _write_json(args.work_dir / "reward_scale_sanity.json", sanity)
    if sanity["verdict"] != "PASS_GREEN":
        raise RuntimeError(f"reward-scale sanity failed: {sanity['checks']}")

    ledger = args.work_dir / "training_curve.jsonl"
    started = time.perf_counter()
    for offset in range(1, ADDITIONAL_UPDATES + 1):
        policy_before = trainer.policy_version
        samples_before = trainer.total_agent_samples
        trainer.env.reset_transfer_counters()
        update_started = time.perf_counter()
        rollout = trainer.collect_rollout()
        try:
            metrics = trainer.update(rollout, kl_guard=KL_GUARD)
        except Rival2PolicyDisplacementRejected as error:
            torch.cuda.synchronize(args.device)
            label = f"pre_rejected_update_{trainer.iteration + 1:05d}"
            checkpoint = _checkpoint(label, trainer, args.work_dir)
            diagnostic = {
                **error.diagnostics,
                "schema_version": SCHEMA_VERSION,
                "created_utc": campaign01._utc_now(),
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "restored_checkpoint": checkpoint,
                "no_later_training_performed": True,
            }
            summary = {
                "schema_version": SCHEMA_VERSION,
                "created_utc": campaign01._utc_now(),
                "status": "STOPPED_KL_GUARD_REJECTION",
                "source_commit": SOURCE_COMMIT,
                "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "reward_version": RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
                "reward_contract_hash": REWARD_GAMEPLAY_V1_CONTRACT_HASH,
                "episode_version": RIVAL2_EPISODE_VERSION,
                "rejected_update": diagnostic["rejected_iteration"],
                "restored_iteration": trainer.iteration,
                "restored_policy_version": trainer.policy_version,
                "completed_additional_updates": trainer.iteration - int(source["iteration"]),
                "agent_decision_samples_after_rejected_rollout": (trainer.total_agent_samples),
                "restored_checkpoint": checkpoint,
                "evaluation_labels": [item["checkpoint_label"] for item in evaluations],
                "diagnostic": diagnostic,
            }
            _write_json(args.work_dir / "kl_rejection.json", diagnostic)
            _write_json(args.work_dir / "run_summary.json", summary)
            _write_json(args.work_dir / "checkpoints.json", [*checkpoints, checkpoint])
            _write_json(args.work_dir / "evaluation_curve.json", evaluations)
            publish_rejection(
                args,
                summary,
                diagnostic,
                checkpoint,
                evaluations,
                checkpoints,
            )
            return summary
        torch.cuda.synchronize(args.device)
        seconds = time.perf_counter() - update_started
        integrity = _training_integrity(
            trainer,
            rollout,
            metrics,
            policy_before=policy_before,
            samples_before=samples_before,
        )
        values = {name: float(value.item()) for name, value in metrics.items()}
        values.update(
            {
                "rollout_reward_mean_blue": float(rollout.rewards[..., 0].mean().item()),
                "rollout_reward_mean_orange": float(rollout.rewards[..., 1].mean().item()),
                "rollout_reward_mean_absolute": float(rollout.rewards.abs().mean().item()),
                "rollout_reward_max_absolute": float(rollout.rewards.abs().amax().item()),
            }
        )
        point = {
            "phase": "GAMEPLAY_V1_SHORT_EPISODE",
            "offset": offset,
            "iteration": trainer.iteration,
            "policy_version": trainer.policy_version,
            "agent_decision_samples": trainer.total_agent_samples,
            "iteration_agent_decision_samples": trainer.total_agent_samples - samples_before,
            "reward_version": trainer.env.reward_version,
            "episode_version": trainer.env.episode_version,
            "wall_seconds": seconds,
            "agent_decisions_per_second": (trainer.total_agent_samples - samples_before) / seconds,
            "terminated_world_intervals": int(rollout.terminated[..., 0].sum().item()),
            "truncated_world_intervals": int(rollout.truncated[..., 0].sum().item()),
            "metrics": values,
            "integrity": integrity,
            "verdict": integrity["verdict"],
        }
        _append_jsonl(ledger, point)
        print(
            f"gameplay update={trainer.iteration} offset={offset}/239 "
            f"samples={trainer.total_agent_samples} seconds={seconds:.3f} "
            f"kl={values['approx_kl']:.6f} "
            f"mb_kl_max={values['optimizer_post_step_approx_kl_max']:.6f} "
            f"value={values['value_loss']:.6f} "
            f"sat_throttle={values['emitted_action_saturation_fraction_throttle']:.6f} "
            f"verdict={point['verdict']}",
            flush=True,
        )
        if point["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"training integrity failure at {trainer.iteration}")
        del rollout, metrics
        gc.collect()
        if offset not in CHECKPOINT_OFFSETS:
            continue
        trainer.add_historical_snapshot()
        label = f"plus_{offset:03d}"
        checkpoint = _checkpoint(label, trainer, args.work_dir)
        if checkpoint["audit"]["verdict"] != "PASS_GREEN":
            raise RuntimeError(f"checkpoint audit failed at {label}")
        evaluation = evaluate(
            trainer=trainer,
            collision_dir=args.collision_dir,
            geometry=geometry,
            meshes=meshes,
            device=args.device,
            label=label,
        )
        evaluations.append(evaluation)
        checkpoints.append(checkpoint)
        _write_json(args.work_dir / f"evaluation_{label}.json", evaluation)
        _write_json(args.work_dir / "evaluation_curve.json", evaluations)
        _write_json(args.work_dir / "checkpoints.json", checkpoints)
        result = evaluation["result"]
        print(
            f"gameplay evaluation={label} "
            f"touches/min={result['touches_per_simulated_minute']:.6f} "
            f"goals/min={result['goals_per_simulated_minute']:.6f} "
            f"no_touch={result['no_touch_truncated_fraction']:.6f} "
            f"saves/min={result['save_events_per_simulated_minute']:.6f}",
            flush=True,
        )

    final = checkpoints[-1]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": campaign01._utc_now(),
        "status": "COMPLETE_239_UPDATE_BOUNDARY",
        "source_commit": SOURCE_COMMIT,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "reward_version": RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
        "reward_contract_hash": REWARD_GAMEPLAY_V1_CONTRACT_HASH,
        "episode_version": RIVAL2_EPISODE_VERSION,
        "final_iteration": trainer.iteration,
        "final_policy_version": trainer.policy_version,
        "final_agent_decision_samples": trainer.total_agent_samples,
        "additional_updates": trainer.iteration - int(source["iteration"]),
        "additional_agent_decision_samples": trainer.total_agent_samples
        - int(source["total_agent_samples"]),
        "final_checkpoint": final,
        "evaluation_labels": [item["checkpoint_label"] for item in evaluations],
        "source_no_touch_fraction": source_evaluation["result"]["no_touch_truncated_fraction"],
        "final_no_touch_fraction": evaluations[-1]["result"]["no_touch_truncated_fraction"],
        "acquisition_regression_indicator_final": evaluations[-1]["result"][
            "no_touch_truncated_fraction"
        ]
        <= 0.01,
        "kl_guard_rejections": 0,
        "five_minute_matches_run": False,
        "nexto_training_run": False,
        "collapsed_scoring_checkpoint_used": False,
        "wall_seconds_including_evaluations": time.perf_counter() - started,
        "recommendation": (
            "Review acquisition retention, goal/save behavior, component scale, "
            "and KL/action trends before authorizing any continuation beyond +239."
        ),
    }
    _write_json(args.work_dir / "run_summary.json", summary)
    _write_json(args.work_dir / "evaluation_curve.json", evaluations)
    _write_json(args.work_dir / "checkpoints.json", checkpoints)
    publish_complete(args, summary, evaluations, checkpoints)
    return summary


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise RuntimeError("work directory must be absent or empty")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise RuntimeError("results directory must be absent or empty")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    campaign01._initialize_runtime(args.device)
    summary = run(args)
    print(
        f"gameplay status={summary['status']} "
        f"iteration={summary.get('final_iteration', summary.get('restored_iteration'))}",
        flush=True,
    )
    return 0 if summary["status"] == "COMPLETE_239_UPDATE_BOUNDARY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
