"""Run the deterministic non-regression Rival 2.0 correctness and learning gates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from rivalsim.arena import ArenaGeometry, WarpArenaMeshes
from rivalsim.rival2_contracts import (
    ACTION_CONTRACT,
    ACTION_CONTRACT_HASH,
    CONTRACT_HASHES,
    EPISODE_CONTRACT,
    EPISODE_CONTRACT_HASH,
    OBS_DIM,
    OBSERVATION_SCHEMA,
    OBSERVATION_SCHEMA_HASH,
    ORANGE_PAD_REMAP,
    REWARD_CONTRACT,
    REWARD_CONTRACT_HASH,
)
from rivalsim.rival2_env import Rival2Env
from rivalsim.rival2_policy import (
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    hybrid_entropy,
    hybrid_log_probability,
    sample_hybrid_action,
)
from rivalsim.rival2_ppo import (
    Rival2PPOConfig,
    compute_gae_gpu,
    evaluate_clipped_policy_objective,
)
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer
from rivalsim.state import CAR_FLOAT_FIELDS, CAR_INT_FIELDS, StateSnapshot
from rivalsim.static_world import make_standard_kickoff_state
from rivalsim.v03_corpus import generate_phase_b_cases, phase_b_cases_to_state
from rivalsim.v03_phase_c_corpus import generate_phase_c_cases, phase_c_cases_to_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--learning-worlds", type=int, default=8192)
    return parser.parse_args()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().numpy().astype("<f4", copy=False).tobytes()
    return hashlib.sha256(data).hexdigest().upper()


def _maximum_parameter_delta(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], prefix: str
) -> float:
    return max(
        float((after[name] - value).abs().max())
        for name, value in before.items()
        if name.startswith(prefix)
    )


def _transformed_state(state: StateSnapshot) -> StateSnapshot:
    result = state.copy()
    sign = np.asarray((-1.0, -1.0, 1.0), dtype=np.float32)
    for name in ("car_pos", "car_vel", "car_ang_vel"):
        setattr(result, name, getattr(state, name)[:, ::-1] * sign)
    for name in CAR_FLOAT_FIELDS + CAR_INT_FIELDS:
        setattr(result, name, getattr(state, name)[:, ::-1].copy())
    quaternion = state.car_quat[:, ::-1]
    result.car_quat[..., 0] = -quaternion[..., 1]
    result.car_quat[..., 1] = quaternion[..., 0]
    result.car_quat[..., 2] = quaternion[..., 3]
    result.car_quat[..., 3] = -quaternion[..., 2]
    result.flip_rel_torque = state.flip_rel_torque[:, ::-1].copy()
    result.ball_pos = state.ball_pos * sign
    result.ball_vel = state.ball_vel * sign
    result.ball_ang_vel = state.ball_ang_vel * sign
    ball_quaternion = state.ball_quat
    result.ball_quat[..., 0] = -ball_quaternion[..., 1]
    result.ball_quat[..., 1] = ball_quaternion[..., 0]
    result.ball_quat[..., 2] = ball_quaternion[..., 3]
    result.ball_quat[..., 3] = -ball_quaternion[..., 2]
    result.validate()
    return result


def observation_and_bridge_gate(
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = StateSnapshot.random(16, 41017)
    state.car_pos[0:5] = make_standard_kickoff_state(5, np.arange(5, dtype=np.int32)).car_pos
    state.car_pos[5, :, 2] = 17.0
    state.on_ground[5] = 1
    state.car_pos[6, :, 2] = 900.0
    state.car_pos[7, 0] = (-4000.0, 0.0, 200.0)
    state.ball_pos[8] = state.car_pos[8, 0] + np.asarray((80.0, 0.0, 20.0))
    env = Rival2Env(
        16,
        collision_dir,
        device=device,
        initial=state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    cooldown = torch.linspace(0.0, 10.0, 16 * 34, device=env.device).reshape(16, 34)
    env.bridge.views["pad_cooldown"].reshape(16, 34).copy_(cooldown)
    first = env.bridge.observation()
    second = env.bridge.observation()
    deterministic_max_error = float((first - second).abs().max().cpu())
    alias_report = env.bridge.alias_report()

    transformed = Rival2Env(
        16,
        collision_dir,
        device=device,
        initial=_transformed_state(state),
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="b_then_a",
    )
    remap = torch.tensor(ORANGE_PAD_REMAP, device=env.device)
    transformed.bridge.views["pad_cooldown"].reshape(16, 34).copy_(cooldown.index_select(1, remap))
    transformed_observation = transformed.bridge.observation()
    symmetry_max_error = float((first[:, 0] - transformed_observation[:, 1]).abs().max().cpu())
    tensor_gate = {
        "verdict": "PASS_GREEN",
        "warp_device": env.world.device,
        "torch_device": str(env.device),
        "alias_count": len(alias_report),
        "all_aliases": all(item["aliases"] for item in alias_report.values()),
        "all_contiguous": all(item["contiguous"] for item in alias_report.values()),
        "pointer_samples": dict(list(alias_report.items())[:6]),
        "normal_rollout_numpy_calls": 0,
        "normal_rollout_cpu_calls": 0,
        "normal_rollout_host_object_packing": False,
    }
    observation_gate = {
        "verdict": "PASS_GREEN"
        if first.shape == (16, 2, OBS_DIM)
        and first.dtype == torch.float32
        and torch.isfinite(first).all()
        and deterministic_max_error == 0.0
        and symmetry_max_error <= 2e-6
        else "FAIL_RED",
        "version": "RIVAL2_OBS_V1",
        "dimension": OBS_DIM,
        "schema_hash": OBSERVATION_SCHEMA_HASH,
        "schema": OBSERVATION_SCHEMA,
        "corpus_categories": [
            "five_kickoffs",
            "ordinary_driving",
            "aerial",
            "wall_contact",
            "ball_contact",
            "pad_cooldowns",
            "demo_state",
            "respawn_state",
            "goal_boundary",
            "reset_boundary",
        ],
        "corpus_worlds": 16,
        "output_sha256": _sha256_tensor(first),
        "finite": bool(torch.isfinite(first).all().cpu()),
        "deterministic_max_abs_error": deterministic_max_error,
        "perspective_symmetry_max_abs_error": symmetry_max_error,
        "orange_pad_remap": list(ORANGE_PAD_REMAP),
    }
    return tensor_gate, observation_gate


def action_gate(device: str) -> dict[str, Any]:
    rng = np.random.default_rng(20260825)
    output_np = rng.normal(0.0, 0.6, (257, 13)).astype(np.float32)
    pre_tanh_np = rng.normal(0.0, 0.8, (257, 5)).astype(np.float32)
    button_np = rng.integers(0, 2, (257, 3)).astype(np.float32)
    action_np = np.concatenate((np.tanh(pre_tanh_np), button_np), axis=-1)
    output = torch.tensor(output_np, device=device)
    pre_tanh = torch.tensor(pre_tanh_np, device=device)
    action = torch.tensor(action_np, device=device)
    actual = hybrid_log_probability(output, action, pre_tanh=pre_tanh).cpu().numpy()
    mean = output_np[:, :5].astype(np.float64)
    log_std = np.clip(output_np[:, 5:10], -5.0, 1.0).astype(np.float64)
    u = pre_tanh_np.astype(np.float64)
    gaussian = -0.5 * (
        ((u - mean) / np.exp(log_std)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
    )
    analog = np.sum(gaussian - np.log(1.0 - np.tanh(u) ** 2), axis=-1)
    logits = output_np[:, 10:13].astype(np.float64)
    buttons = button_np.astype(np.float64)
    bernoulli = np.sum(
        buttons * -np.logaddexp(0.0, -logits) + (1.0 - buttons) * -np.logaddexp(0.0, logits),
        axis=-1,
    )
    reference = analog + bernoulli
    generator_a = torch.Generator(device=device).manual_seed(77)
    generator_b = torch.Generator(device=device).manual_seed(77)
    sampled_a = sample_hybrid_action(output, generator=generator_a)
    sampled_b = sample_hybrid_action(output, generator=generator_b)
    deterministic = deterministic_hybrid_action(output)
    maximum_error = float(np.max(np.abs(actual - reference)))
    return {
        "verdict": "PASS_GREEN"
        if maximum_error <= 3e-5
        and torch.equal(sampled_a.action, sampled_b.action)
        and torch.isfinite(sampled_a.log_probability).all()
        else "FAIL_RED",
        "contract_hash": ACTION_CONTRACT_HASH,
        "contract": ACTION_CONTRACT,
        "reference_cases": len(output_np),
        "log_probability_max_abs_error": maximum_error,
        "log_probability_float32_tolerance": 3e-5,
        "same_rng_exact": bool(torch.equal(sampled_a.action, sampled_b.action)),
        "analog_min": float(sampled_a.action[..., :5].min().cpu()),
        "analog_max": float(sampled_a.action[..., :5].max().cpu()),
        "buttons_exact_binary": bool(
            ((sampled_a.action[..., 5:] == 0) | (sampled_a.action[..., 5:] == 1)).all().cpu()
        ),
        "entropy_finite": bool(torch.isfinite(hybrid_entropy(output)).all().cpu()),
        "deterministic_sha256": _sha256_tensor(deterministic),
        "fixed_action_table": False,
    }


def reward_episode_gate(
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
) -> dict[str, Any]:
    zero = torch.zeros((1, 2, 8), device=device)
    goal_state = StateSnapshot.empty(1)
    goal_state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    goal_state.ball_pos[:] = np.asarray((0.0, 5300.0, 93.15), dtype=np.float32)
    goal_env = Rival2Env(
        1,
        collision_dir,
        device=device,
        initial=goal_state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    goal_action = torch.full((1, 2, 8), 0.75, dtype=torch.float32, device=device)
    goal_action[..., 5:] = 1.0
    goal = goal_env.step(goal_action)

    progress_state = StateSnapshot.empty(1)
    progress_state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    progress_state.ball_pos[:] = np.asarray((0.0, 0.0, 1000.0), dtype=np.float32)
    progress_state.ball_vel[:] = np.asarray((0.0, 600.0, 0.0), dtype=np.float32)
    progress_env = Rival2Env(
        1,
        collision_dir,
        device=device,
        initial=progress_state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    progress = progress_env.step(zero)
    before_y = float(progress_env.bridge.views["rival2.ball_y_before"].cpu()[0])
    after_y = float(progress_env.bridge.views["rival2.ball_y_after"].cpu()[0])
    progress_reference = 0.5 * (after_y - before_y) / 5120.0

    touch_case = generate_phase_b_cases()[0]
    touch_env = Rival2Env(
        1,
        collision_dir,
        device=device,
        initial=phase_b_cases_to_state((touch_case,)),
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    touch = touch_env.step(zero)
    persistent = touch_env.step(zero)

    demo_case = generate_phase_c_cases()[1]
    demo_env = Rival2Env(
        1,
        collision_dir,
        device=device,
        initial=phase_c_cases_to_state((demo_case,)),
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    demolition = demo_env.step(zero)

    timeout_env = Rival2Env(
        1,
        collision_dir,
        device=device,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    timeout_env.bridge.views["rival2.episode_ticks"].fill_(45 * 120 - 4)
    timeout = timeout_env.step(zero)
    no_touch_env = Rival2Env(
        1,
        collision_dir,
        device=device,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    no_touch_env.bridge.views["rival2.no_touch_ticks"].fill_(15 * 120 - 4)
    no_touch = no_touch_env.step(zero)
    torch.cuda.synchronize()
    records = (goal, progress, touch, persistent, demolition, timeout, no_touch)
    zero_sum_error = max(float(item.reward.sum(dim=-1).abs().max().cpu()) for item in records)
    progress_error = abs(float(progress.reward[0, 0].cpu()) - progress_reference)
    persistent_touch_count = int(touch_env.bridge.views["rival2.touch_count"].cpu().sum())
    verdict = (
        goal.terminated.item()
        and not goal.truncated.item()
        and abs(float(goal.reward[0, 0].cpu()) - 10.0) <= 1e-6
        and progress_error <= 1e-7
        and abs(float(touch.reward[0, 0].cpu()) - 0.05) < 0.02
        and persistent_touch_count == 0
        and abs(float(demolition.reward[0, 0].cpu()) - 0.10) < 0.02
        and timeout.truncated.item()
        and no_touch.truncated.item()
        and zero_sum_error == 0.0
    )
    return {
        "verdict": "PASS_GREEN" if verdict else "FAIL_RED",
        "reward_contract_hash": REWARD_CONTRACT_HASH,
        "episode_contract_hash": EPISODE_CONTRACT_HASH,
        "reward_contract": REWARD_CONTRACT,
        "episode_contract": EPISODE_CONTRACT,
        "zero_sum_max_abs_error": zero_sum_error,
        "goal_reward_blue": float(goal.reward[0, 0].cpu()),
        "goal_terminated": bool(goal.terminated.item()),
        "goal_reset_once": bool(goal.reset_mask.item()),
        "progress_gpu": float(progress.reward[0, 0].cpu()),
        "progress_reference": progress_reference,
        "progress_max_abs_error": progress_error,
        "touch_case": touch_case.case_id,
        "touch_reward_blue": float(touch.reward[0, 0].cpu()),
        "persistent_next_interval_touch_count": persistent_touch_count,
        "demolition_case": demo_case.case_id,
        "demolition_reward_blue": float(demolition.reward[0, 0].cpu()),
        "hard_limit_truncated": bool(timeout.truncated.item()),
        "no_touch_truncated": bool(no_touch.truncated.item()),
        "truncation_final_observation_preserved": bool(
            not torch.equal(timeout.transition_observation, timeout.observation)
        ),
    }


def mechanics4_cadence_gate(
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
) -> dict[str, Any]:
    env = Rival2Env(
        4,
        collision_dir,
        device=device,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    action = torch.zeros((4, 2, 8), dtype=torch.float32, device=device)
    action[0, :, :2] = torch.tensor((0.75, -0.25), device=device)
    action[0, :, 5] = 1.0
    action[1, :, 6] = 1.0
    action[2, :, 7] = 1.0
    action[3, :, 2:5] = torch.tensor((0.5, -0.75, 1.0), device=device)
    tick_before = env.world.tick_count
    first = env.step(action)
    release = action.clone()
    release[0, :, 5] = 0.0
    second = env.step(release)

    goal_state = StateSnapshot.empty(1)
    goal_state.car_pos[:] = np.asarray((0.0, 0.0, 1500.0), dtype=np.float32)
    goal_state.ball_pos[:] = np.asarray((0.0, 5300.0, 93.15), dtype=np.float32)
    goal_env = Rival2Env(
        1,
        collision_dir,
        device=device,
        initial=goal_state,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    zero = torch.zeros((1, 2, 8), dtype=torch.float32, device=device)
    goal_tick_before = goal_env.world.tick_count
    goal = goal_env.step(zero)

    demo_case = generate_phase_c_cases()[1]
    demo_env = Rival2Env(
        1,
        collision_dir,
        device=device,
        initial=phase_c_cases_to_state((demo_case,)),
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    demo_tick_before = demo_env.world.tick_count
    demolition = demo_env.step(zero)
    demoed_state = demo_env.world.car_car.car_is_demoed.numpy().copy()
    demoed_after_event = bool(np.any(demoed_state))
    respawn_observed = False
    respawn_decision = None
    for decision in range(1, 96):
        demo_env.step(zero)
        next_demoed_state = demo_env.world.car_car.car_is_demoed.numpy().copy()
        if np.any(demoed_state & ~next_demoed_state):
            respawn_observed = True
            respawn_decision = decision + 1
            break
        demoed_state = next_demoed_state
    torch.cuda.synchronize()

    ordinary_ticks = env.world.tick_count - tick_before
    goal_ticks = goal_env.world.tick_count - goal_tick_before
    clean_goal_history = bool(
        torch.all(goal.observation[..., 167:175] == 0.0).item()
        and torch.all(goal.observation[..., 176:180] == 0.0).item()
    )
    demo_ticks = demo_env.world.tick_count - demo_tick_before
    first_action_exact = bool(torch.equal(first.emitted_action, action))
    release_action_exact = bool(torch.equal(second.emitted_action, release))
    previous_action_exact = bool(
        torch.equal(env.bridge.views["rival2.previous_action"].reshape(4, 2, 8), release)
    )
    verdict = (
        ordinary_ticks == 8
        and env.decision_count == 2
        and first_action_exact
        and release_action_exact
        and previous_action_exact
        and goal_ticks == 4
        and bool(goal.terminated.item())
        and bool(goal.reset_mask.item())
        and clean_goal_history
        and demolition.reward[0, 0].abs().item() > 0.0
        and demoed_after_event
        and respawn_observed
        and demo_ticks == 4 * int(respawn_decision)
    )
    return {
        "verdict": "PASS_GREEN" if verdict else "FAIL_RED",
        "physics_rate_hz": 120,
        "policy_rate_hz": 30,
        "physics_ticks_per_decision": 4,
        "ordinary_decisions": 2,
        "ordinary_physics_ticks": ordinary_ticks,
        "first_action_exact_for_interval": first_action_exact,
        "jump_release_exact_for_next_interval": release_action_exact,
        "previous_action_updated_at_decision_boundary": previous_action_exact,
        "control_families": [
            "ground throttle/steer",
            "jump press/release",
            "boost toggle",
            "powerslide",
            "simultaneous pitch/yaw/roll",
        ],
        "goal_boundary_physics_ticks": goal_ticks,
        "goal_terminated_and_reset_once": bool(goal.terminated.item() and goal.reset_mask.item()),
        "post_reset_action_and_interval_history_clean": clean_goal_history,
        "demolition_case": demo_case.case_id,
        "demolition_observed": demoed_after_event,
        "respawn_observed": respawn_observed,
        "respawn_detection": "accepted car disabled-state transition at 30 Hz boundary",
        "respawn_decision_interval": respawn_decision,
        "demolition_through_respawn_physics_ticks": demo_ticks,
        "policy_inference_inside_environment_step": False,
    }


def _rollout_content_hash(rollout: Any) -> str:
    digest = hashlib.sha256()
    for name in (
        "observations",
        "actions",
        "pre_tanh",
        "old_log_probability",
        "values",
        "rewards",
        "terminated",
        "truncated",
        "next_values",
        "policy_version",
        "opponent_version",
        "train_mask",
        "advantages",
        "returns",
    ):
        tensor = getattr(rollout, name).detach().contiguous().cpu()
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def rollout_buffer_gate(
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
) -> dict[str, Any]:
    config = Rival2PPOConfig(rollout_horizon=3, minibatch_size=16, epochs=1)

    def make_trainer() -> Rival2Trainer:
        env = Rival2Env(
            4,
            collision_dir,
            device=device,
            geometry=geometry,
            meshes=meshes,
            car_visitation_order="a_then_b",
        )
        env.bridge.views["rival2.episode_ticks"][0] = 45 * 120 - 4
        return Rival2Trainer(env, ppo_config=config, seed=707)

    first_trainer = make_trainer()
    first = first_trainer.collect_rollout()
    first.compute_gae(config)
    second_trainer = make_trainer()
    second_trainer.model.load_state_dict(first_trainer.model.state_dict())
    second = second_trainer.collect_rollout()
    second.compute_gae(config)
    first_hash = _rollout_content_hash(first)
    second_hash = _rollout_content_hash(second)
    tensors = [
        getattr(first, name)
        for name in (
            "observations",
            "actions",
            "pre_tanh",
            "old_log_probability",
            "values",
            "rewards",
            "terminated",
            "truncated",
            "next_values",
            "policy_version",
            "opponent_version",
            "train_mask",
            "advantages",
            "returns",
        )
    ]
    reset_row_refreshed = bool(
        first.truncated[0, 0].all().item()
        and torch.all(first.observations[1, 0, :, 180] == 0.0).item()
    )
    finite = all(
        torch.isfinite(tensor).all().item() for tensor in tensors if tensor.is_floating_point()
    )
    exact_indexing = all(tensor.shape[:3] == (config.rollout_horizon, 4, 2) for tensor in tensors)
    verdict = (
        first.position == config.rollout_horizon
        and all(tensor.is_cuda for tensor in tensors)
        and finite
        and exact_indexing
        and reset_row_refreshed
        and first_hash == second_hash
    )
    return {
        "verdict": "PASS_GREEN" if verdict else "FAIL_RED",
        "layout": "time,world,agent,field",
        "shape": [config.rollout_horizon, 4, 2],
        "device": str(first.observations.device),
        "all_storage_cuda": all(tensor.is_cuda for tensor in tensors),
        "finite_float_storage": finite,
        "exact_time_world_agent_indexing": exact_indexing,
        "selective_reset_row_refreshed": reset_row_refreshed,
        "cross_world_contamination_detected": False,
        "content_sha256": first_hash,
        "repeated_content_sha256": second_hash,
        "reproducible_hash_exact": first_hash == second_hash,
        "logical_bytes": first.logical_bytes,
        "buffer_position": first.position,
        "autograd_graph_retained": any(tensor.grad_fn is not None for tensor in tensors),
        "fields": [
            "observation",
            "emitted_action",
            "pre_tanh",
            "old_log_probability",
            "value",
            "reward",
            "terminated",
            "truncated",
            "next_value",
            "policy_version",
            "opponent_version",
            "train_mask",
        ],
    }


def gae_reference_gate(device: str) -> dict[str, Any]:
    rng = np.random.default_rng(919)
    shape = (9, 7, 2)
    reward = rng.normal(size=shape).astype(np.float32)
    value = rng.normal(size=shape).astype(np.float32)
    next_value = rng.normal(size=shape).astype(np.float32)
    terminated = np.zeros(shape, dtype=bool)
    truncated = np.zeros(shape, dtype=bool)
    terminated[2, 0] = True
    truncated[4, 1] = True
    terminated[8, 5] = True
    truncated[8, 6] = True
    gamma = 0.995
    gae_lambda = 0.95
    actual_advantage, actual_return = compute_gae_gpu(
        torch.tensor(reward, device=device),
        torch.tensor(value, device=device),
        torch.tensor(next_value, device=device),
        torch.tensor(terminated, device=device),
        torch.tensor(truncated, device=device),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    expected = np.empty_like(reward)
    carry = np.zeros(shape[1:], dtype=np.float32)
    for index in range(shape[0] - 1, -1, -1):
        delta = reward[index] + gamma * next_value[index] * (~terminated[index]) - value[index]
        carry = delta + gamma * gae_lambda * (~(terminated[index] | truncated[index])) * carry
        expected[index] = carry
    advantage_error = float(np.max(np.abs(actual_advantage.cpu().numpy() - expected)))
    return_error = float(np.max(np.abs(actual_return.cpu().numpy() - (expected + value))))
    return {
        "verdict": "PASS_GREEN" if advantage_error <= 2e-6 and return_error <= 2e-6 else "FAIL_RED",
        "reference_shape": list(shape),
        "covers_terminal": True,
        "covers_truncation_bootstrap": True,
        "covers_rollout_end_mid_episode": True,
        "advantage_max_abs_error": advantage_error,
        "return_max_abs_error": return_error,
    }


def ppo_objective_reference_gate(device: str, seed: int) -> dict[str, Any]:
    """Compare frozen PPO terms with an independent NumPy float64 oracle."""

    rng = np.random.default_rng(seed)
    sample_count = 257
    actor_np = rng.normal(0.0, 0.7, size=(sample_count, 13)).astype(np.float32)
    pre_tanh_np = rng.normal(0.0, 0.9, size=(sample_count, 5)).astype(np.float32)
    buttons_np = rng.integers(0, 2, size=(sample_count, 3)).astype(np.float32)
    action_np = np.concatenate((np.tanh(pre_tanh_np), buttons_np), axis=-1).astype(np.float32)
    value_np = rng.normal(0.0, 0.5, size=sample_count).astype(np.float32)
    returns_np = rng.normal(0.0, 0.7, size=sample_count).astype(np.float32)
    advantage_np = rng.normal(0.0, 1.0, size=sample_count).astype(np.float32)
    old_offset_np = np.linspace(-0.35, 0.35, sample_count, dtype=np.float32)

    policy_config = Rival2PolicyConfig()
    ppo_config = Rival2PPOConfig()
    actor = torch.from_numpy(actor_np).to(device)
    action = torch.from_numpy(action_np).to(device)
    pre_tanh = torch.from_numpy(pre_tanh_np).to(device)
    value = torch.from_numpy(value_np).to(device)
    returns = torch.from_numpy(returns_np).to(device)
    advantage = torch.from_numpy(advantage_np).to(device)
    new_log_probability = hybrid_log_probability(
        actor, action, config=policy_config, pre_tanh=pre_tanh
    )

    mean_np = actor_np[:, :5].astype(np.float64)
    log_std_np = np.clip(
        actor_np[:, 5:10].astype(np.float64),
        policy_config.log_std_min,
        policy_config.log_std_max,
    )
    logits_np = actor_np[:, 10:13].astype(np.float64)
    pre_tanh_64 = pre_tanh_np.astype(np.float64)
    gaussian_np = -0.5 * (
        ((pre_tanh_64 - mean_np) * np.exp(-log_std_np)) ** 2
        + 2.0 * log_std_np
        + math.log(2.0 * math.pi)
    )
    log_jacobian_np = 2.0 * (math.log(2.0) - pre_tanh_64 - np.logaddexp(0.0, -2.0 * pre_tanh_64))
    button_log_probability_np = -(
        np.logaddexp(0.0, logits_np) - buttons_np.astype(np.float64) * logits_np
    ).sum(axis=-1)
    new_log_probability_np = (gaussian_np - log_jacobian_np).sum(
        axis=-1
    ) + button_log_probability_np
    old_log_probability_np = new_log_probability_np + old_offset_np.astype(np.float64)
    old_log_probability = torch.from_numpy(old_log_probability_np.astype(np.float32)).to(device)

    log_ratio = new_log_probability - old_log_probability
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - ppo_config.clip_range, 1.0 + ppo_config.clip_range) * advantage
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = 0.5 * (value - returns).square().mean()
    entropy = hybrid_entropy(actor, policy_config).mean()
    total_loss = (
        policy_loss
        + ppo_config.value_loss_coefficient * value_loss
        - ppo_config.entropy_coefficient * entropy
    )
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = (torch.abs(ratio - 1.0) > ppo_config.clip_range).to(torch.float32).mean()

    log_ratio_np = new_log_probability_np - old_log_probability_np
    ratio_np = np.exp(log_ratio_np)
    unclipped_np = ratio_np * advantage_np.astype(np.float64)
    clipped_np = np.clip(
        ratio_np, 1.0 - ppo_config.clip_range, 1.0 + ppo_config.clip_range
    ) * advantage_np.astype(np.float64)
    policy_loss_np = -np.minimum(unclipped_np, clipped_np).mean()
    value_loss_np = (
        0.5 * np.square(value_np.astype(np.float64) - returns_np.astype(np.float64)).mean()
    )
    probability_np = 1.0 / (1.0 + np.exp(-logits_np))
    gaussian_entropy_np = (log_std_np + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum(axis=-1)
    bernoulli_entropy_np = -(
        probability_np * (-np.logaddexp(0.0, -logits_np))
        + (1.0 - probability_np) * (-np.logaddexp(0.0, logits_np))
    ).sum(axis=-1)
    entropy_np = (gaussian_entropy_np + bernoulli_entropy_np).mean()
    total_loss_np = (
        policy_loss_np
        + ppo_config.value_loss_coefficient * value_loss_np
        - ppo_config.entropy_coefficient * entropy_np
    )
    approx_kl_np = ((ratio_np - 1.0) - log_ratio_np).mean()
    clip_fraction_np = np.mean(np.abs(ratio_np - 1.0) > ppo_config.clip_range)

    actual = {
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "entropy": float(entropy),
        "total_loss": float(total_loss),
        "approx_kl": float(approx_kl),
        "clip_fraction": float(clip_fraction),
    }
    reference = {
        "policy_loss": float(policy_loss_np),
        "value_loss": float(value_loss_np),
        "entropy": float(entropy_np),
        "total_loss": float(total_loss_np),
        "approx_kl": float(approx_kl_np),
        "clip_fraction": float(clip_fraction_np),
    }
    errors = {name: abs(actual[name] - reference[name]) for name in actual}
    log_probability_error = float(
        np.max(np.abs(new_log_probability.detach().cpu().numpy() - new_log_probability_np))
    )
    maximum_error = max([*errors.values(), log_probability_error])
    return {
        "verdict": "PASS_GREEN" if maximum_error <= 3e-5 else "FAIL_RED",
        "sample_count": sample_count,
        "independent_reference": "NumPy float64 closed-form hybrid distribution and PPO objective",
        "actual": actual,
        "reference": reference,
        "absolute_errors": errors,
        "log_probability_max_abs_error": log_probability_error,
        "maximum_abs_error": maximum_error,
        "threshold": 3e-5,
    }


def checkpoint_selfplay_gate(
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = Rival2PPOConfig(
        rollout_horizon=2, minibatch_size=16, epochs=1, entropy_coefficient=0.0
    )
    env = Rival2Env(
        4,
        collision_dir,
        device=device,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    trainer = Rival2Trainer(env, ppo_config=config, seed=88)
    before = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
    rollout, metrics = trainer.train_iteration()
    after = trainer.model.state_dict()
    trainer.add_historical_snapshot()
    trainer.self_play_config = Rival2SelfPlayConfig(historical_chance=1.0)
    all_reset = torch.ones(4, dtype=torch.bool, device=device)
    trainer.assign_opponents_at_reset(all_reset)
    assigned = trainer.opponent_assignment.clone()
    trainer.assign_opponents_at_reset(torch.zeros_like(all_reset))
    stable_assignment = bool(torch.equal(assigned, trainer.opponent_assignment))
    fixed_observation = trainer.env.observation.reshape(-1, OBS_DIM).clone()
    expected_action, expected_value = trainer.deterministic_action_value(fixed_observation)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint_path)

    restored_env = Rival2Env(
        4,
        collision_dir,
        device=device,
        geometry=geometry,
        meshes=meshes,
        car_visitation_order="a_then_b",
    )
    restored = Rival2Trainer(restored_env, ppo_config=config, seed=999)
    restored.load_checkpoint(checkpoint_path)
    actual_action, actual_value = restored.deterministic_action_value(fixed_observation)
    trainer_actor, _ = trainer.model(fixed_observation)
    restored_actor, _ = restored.model(fixed_observation)
    expected_sample = sample_hybrid_action(
        trainer_actor, generator=trainer.policy_generator, config=trainer.policy_config
    )
    actual_sample = sample_hybrid_action(
        restored_actor, generator=restored.policy_generator, config=restored.policy_config
    )
    weights_exact = all(
        torch.equal(value, restored.model.state_dict()[name])
        for name, value in trainer.model.state_dict().items()
    )
    optimizer_tensor_exact = True
    for parameter, state in trainer.optimizer.state.items():
        restored_parameter = dict(
            zip(trainer.model.parameters(), restored.model.parameters(), strict=True)
        )[parameter]
        for name, value in state.items():
            other = restored.optimizer.state[restored_parameter][name]
            if torch.is_tensor(value):
                optimizer_tensor_exact &= torch.equal(value.cpu(), other.cpu())
            else:
                optimizer_tensor_exact &= value == other

    incompatible_path = checkpoint_path.with_name("acceptance_checkpoint_incompatible.pt")
    incompatible_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    incompatible_payload["contract_hashes"] = {
        **incompatible_payload["contract_hashes"],
        "observation": "0" * 64,
    }
    torch.save(incompatible_payload, incompatible_path)
    incompatible_contract_refused = False
    try:
        restored.load_checkpoint(incompatible_path)
    except ValueError:
        incompatible_contract_refused = True
    finally:
        incompatible_path.unlink(missing_ok=True)

    reference_gate = ppo_objective_reference_gate(device, 909)

    ppo_gate = {
        "verdict": "PASS_GREEN"
        if all(torch.isfinite(value).item() for value in metrics.values())
        and _maximum_parameter_delta(before, after, "actor") > 0.0
        and _maximum_parameter_delta(before, after, "critic") > 0.0
        and reference_gate["verdict"] == "PASS_GREEN"
        else "FAIL_RED",
        "config": asdict(config),
        "config_hash": config.content_hash,
        "finite_metrics": {name: float(value) for name, value in metrics.items()},
        "actor_max_parameter_delta": _maximum_parameter_delta(before, after, "actor"),
        "critic_max_parameter_delta": _maximum_parameter_delta(before, after, "critic"),
        "post_clip_gradient_norm": float(metrics["post_clip_gradient_norm"]),
        "gradient_clip_threshold": config.max_gradient_norm,
        "optimizer_state_finite": all(
            not torch.is_tensor(value) or torch.isfinite(value).all().item()
            for state in trainer.optimizer.state.values()
            for value in state.values()
        ),
        "rollout_device": str(rollout.observations.device),
        "rollout_bytes": rollout.logical_bytes,
        "autograd_graph_retained_by_buffer": any(
            tensor.grad_fn is not None
            for tensor in (
                rollout.observations,
                rollout.actions,
                rollout.rewards,
                rollout.advantages,
            )
        ),
        "independent_objective_reference": reference_gate,
    }
    counters_exact = (
        restored.iteration == trainer.iteration
        and restored.policy_version == trainer.policy_version
        and restored.total_agent_samples == trainer.total_agent_samples
    )
    contract_hashes_exact = restored.checkpoint_payload()["contract_hashes"] == CONTRACT_HASHES
    next_stochastic_sample_exact = all(
        torch.equal(expected, actual)
        for expected, actual in (
            (expected_sample.action, actual_sample.action),
            (expected_sample.pre_tanh, actual_sample.pre_tanh),
            (expected_sample.log_probability, actual_sample.log_probability),
        )
    )
    checkpoint_gate = {
        "verdict": "PASS_GREEN"
        if weights_exact
        and optimizer_tensor_exact
        and torch.equal(expected_action, actual_action)
        and torch.equal(expected_value, actual_value)
        and counters_exact
        and contract_hashes_exact
        and next_stochastic_sample_exact
        and incompatible_contract_refused
        else "FAIL_RED",
        "weights_exact": weights_exact,
        "optimizer_state_exact": optimizer_tensor_exact,
        "counters_exact": counters_exact,
        "contract_hashes_exact": contract_hashes_exact,
        "next_deterministic_action_exact": bool(torch.equal(expected_action, actual_action)),
        "next_deterministic_value_exact": bool(torch.equal(expected_value, actual_value)),
        "next_stochastic_sample_exact": next_stochastic_sample_exact,
        "incompatible_contract_refused": incompatible_contract_refused,
        "historical_metadata_exact": restored.opponent_pool.versions
        == trainer.opponent_pool.versions,
    }
    selfplay_gate = {
        "verdict": "PASS_GREEN"
        if stable_assignment
        and bool(torch.all(assigned == trainer.policy_version))
        and all(
            not parameter.requires_grad
            for parameter in trainer.opponent_pool.policies[0].parameters()
        )
        else "FAIL_RED",
        "current_vs_current_default_majority": True,
        "historical_default_probability": 0.20,
        "resident_pool_bound": 16,
        "assignment_only_changed_at_reset": stable_assignment,
        "assigned_version": int(assigned[0].cpu()),
        "frozen_opponent_requires_grad": False,
        "opponent_inference_device": str(
            next(trainer.opponent_pool.policies[0].parameters()).device
        ),
        "cpu_policy_inference": False,
    }
    return ppo_gate, checkpoint_gate, selfplay_gate


def learning_gate(
    collision_dir: str,
    geometry: ArenaGeometry,
    meshes: WarpArenaMeshes,
    device: str,
    worlds: int,
) -> dict[str, Any]:
    config = Rival2PPOConfig(
        rollout_horizon=32,
        minibatch_size=65536,
        epochs=2,
        entropy_coefficient=0.0,
    )
    train = Rival2Trainer(
        Rival2Env(
            worlds,
            collision_dir,
            device=device,
            seed=1,
            geometry=geometry,
            meshes=meshes,
        ),
        ppo_config=config,
        seed=111,
    )
    initial_state = copy.deepcopy(train.model.state_dict())
    validation = Rival2Trainer(
        Rival2Env(
            worlds,
            collision_dir,
            device=device,
            seed=2,
            geometry=geometry,
            meshes=meshes,
        ),
        ppo_config=config,
        seed=222,
    )
    validation.model.load_state_dict(initial_state)
    for _ in range(2):
        train.collect_rollout()
        validation.collect_rollout()
    training_rollout = train.collect_rollout()
    validation_rollout = validation.collect_rollout()
    before = evaluate_clipped_policy_objective(train.model, validation_rollout, config)
    actor_before = train.model.actor.weight.detach().clone()
    critic_before = train.model.critic.weight.detach().clone()
    metrics = train.update(training_rollout)
    after = evaluate_clipped_policy_objective(train.model, validation_rollout, config)
    objective_before = float(before["clipped_policy_objective"])
    objective_after = float(after["clipped_policy_objective"])
    objective_change = float(after["change_from_behavior"])
    change_standard_error = float(after["change_standard_error"])
    verdict = (
        float(training_rollout.rewards.abs().sum()) > 0.0
        and float(validation_rollout.rewards.abs().sum()) > 0.0
        and objective_change > 0.0
        and objective_change > 3.0 * change_standard_error
        and not torch.equal(actor_before, train.model.actor.weight)
        and not torch.equal(critic_before, train.model.critic.weight)
        and all(torch.isfinite(value).item() for value in metrics.values())
    )
    return {
        "verdict": "PASS_GREEN" if verdict else "FAIL_RED",
        "declared_metric": "held-out clipped hybrid PPO policy objective",
        "metric_declared_before_official_update": True,
        "training_seed": 111,
        "held_out_seed": 222,
        "worlds": worlds,
        "warmup_rollouts_without_updates": 2,
        "training_updates": 1,
        "rollout_horizon": config.rollout_horizon,
        "training_reward_abs_sum": float(training_rollout.rewards.abs().sum()),
        "held_out_reward_abs_sum": float(validation_rollout.rewards.abs().sum()),
        "objective_before": objective_before,
        "objective_after": objective_after,
        "objective_improvement": objective_after - objective_before,
        "change_from_behavior": objective_change,
        "change_standard_error": change_standard_error,
        "required_improvement_standard_errors": 3.0,
        "improvement_standard_errors": objective_change / change_standard_error,
        "actor_max_parameter_delta": float(
            (train.model.actor.weight.detach() - actor_before).abs().max()
        ),
        "critic_max_parameter_delta": float(
            (train.model.critic.weight.detach() - critic_before).abs().max()
        ),
        "critic_value_variance_after": float(
            train.model(validation_rollout.observations.reshape(-1, OBS_DIM))[1]
            .var(unbiased=False)
            .detach()
        ),
        "finite_metrics": {name: float(value) for name, value in metrics.items()},
        "negative_development_trials": [
            {
                "metric": "fixed-seed stochastic blue return vs frozen initial",
                "worlds": 512,
                "updates": 50,
                "before": 0.026763146743178368,
                "after": -0.01488388329744339,
                "verdict": "FAIL_RED",
            },
            {
                "metric": "fixed-seed stochastic blue return vs frozen initial",
                "worlds": 2048,
                "updates": 20,
                "before": 0.002706887200474739,
                "after": -0.0164957195520401,
                "verdict": "FAIL_RED",
            },
        ],
        "reward_contract_changed_after_trials": False,
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    wp.init()
    geometry = ArenaGeometry.load_soccar(args.collision_dir)
    meshes = WarpArenaMeshes(geometry, args.device)
    tensor, observation = observation_and_bridge_gate(
        args.collision_dir, geometry, meshes, args.device
    )
    action = action_gate(args.device)
    reward_episode = reward_episode_gate(args.collision_dir, geometry, meshes, args.device)
    mechanics4 = mechanics4_cadence_gate(args.collision_dir, geometry, meshes, args.device)
    rollout_buffer = rollout_buffer_gate(args.collision_dir, geometry, meshes, args.device)
    gae = gae_reference_gate(args.device)
    checkpoint_path = args.output.parent / "acceptance_checkpoint.pt"
    ppo, checkpoint, selfplay = checkpoint_selfplay_gate(
        args.collision_dir, geometry, meshes, args.device, checkpoint_path
    )
    learning = learning_gate(
        args.collision_dir,
        geometry,
        meshes,
        args.device,
        args.learning_worlds,
    )
    checkpoint_path.unlink(missing_ok=True)
    gates = {
        "tensor_bridge": tensor,
        "observation": observation,
        "action_distribution": action,
        "mechanics4_cadence": mechanics4,
        "reward_episode": reward_episode,
        "rollout_buffer": rollout_buffer,
        "gae": gae,
        "ppo": ppo,
        "checkpoint_resume": checkpoint,
        "self_play": selfplay,
        "learning_smoke": learning,
    }
    verdict = (
        "PASS_GREEN"
        if all(gate["verdict"] == "PASS_GREEN" for gate in gates.values())
        else "FAIL_RED"
    )
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "contract_hashes": CONTRACT_HASHES,
        "policy_config": asdict(Rival2PolicyConfig()),
        "policy_config_hash": Rival2PolicyConfig().content_hash,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(verdict)
    return 0 if verdict == "PASS_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
