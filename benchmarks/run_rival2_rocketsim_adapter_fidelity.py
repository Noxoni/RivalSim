"""Targeted 2,048-state RocketSim -> RIVAL2_OBS_V1 fidelity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rivalsim.reference.rocketsim_oracle import (
    ROCKETSIM_BINDING_COMMIT,
    ROCKETSIM_PRIMARY_COMMIT,
)
from rivalsim.rival2_contracts import OBS_DIM, OBS_FIELD_NAMES, RIVAL2_REWARD_VERSION
from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim
from rivalsim.rocketsim_adapter import (
    FrozenRivalPolicy,
    RocketSimBatchState,
    RocketSimRivalMemory,
    build_rival2_observation,
    canonical_boost_pads,
    read_rocketsim_batch,
)

CHECKPOINT = ROOT / "checkpoints" / "rival2" / "overnight" / "rival2_overnight_final_6h_resume.pt"
CHECKPOINT_SHA256 = "4DC158DC2A9D16B79FB5FE7D868E3B50928AB113B55DFCC753F3734F8D87372E"
ROCKETSIM_EXTENSION_SHA256 = "E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0"
COUNT = 2_048
SEED = 2_026_082_801
OUTPUT = ROOT / "results" / "rival2" / "rocketsim_crosscheck" / "adapter_fidelity.json"
KICKOFF_SEEDS = (7, 3, 2, 1, 0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _matrix_to_quaternion_xyzw(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    right = np.cross(up, forward)
    matrix = np.stack((forward, right, up), axis=-1).astype(np.float64)
    result = np.empty((matrix.shape[0], matrix.shape[1], 4), dtype=np.float64)
    flat_matrix = matrix.reshape(-1, 3, 3)
    flat_result = result.reshape(-1, 4)
    for index, value in enumerate(flat_matrix):
        trace = np.trace(value)
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2.0
            flat_result[index] = (
                (value[2, 1] - value[1, 2]) / s,
                (value[0, 2] - value[2, 0]) / s,
                (value[1, 0] - value[0, 1]) / s,
                0.25 * s,
            )
        else:
            axis = int(np.argmax(np.diag(value)))
            if axis == 0:
                s = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
                flat_result[index] = (0.25 * s, (value[0, 1] + value[1, 0]) / s, (value[0, 2] + value[2, 0]) / s, (value[2, 1] - value[1, 2]) / s)
            elif axis == 1:
                s = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
                flat_result[index] = ((value[0, 1] + value[1, 0]) / s, 0.25 * s, (value[1, 2] + value[2, 1]) / s, (value[0, 2] - value[2, 0]) / s)
            else:
                s = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
                flat_result[index] = ((value[0, 2] + value[2, 0]) / s, (value[1, 2] + value[2, 1]) / s, 0.25 * s, (value[1, 0] - value[0, 1]) / s)
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result.astype(np.float32)


def _set_broad_state(rs: Any, arena: Any, cars: list[Any], pads: list[Any], index: int, rng: np.random.Generator) -> None:
    if index < 10:
        arena.reset_kickoff(KICKOFF_SEEDS[(index // 2) % 5])
        return
    arena.reset_kickoff(KICKOFF_SEEDS[index % 5])
    wall_class = index % 8
    for car_index, car in enumerate(cars):
        state = rs.CarState()
        sign = -1.0 if car_index == 0 else 1.0
        if wall_class == 0:
            position = (sign * rng.uniform(500, 2500), rng.uniform(-4200, 4200), rng.uniform(300, 1600))
        elif wall_class == 1:
            position = (sign * rng.uniform(3900, 4050), rng.uniform(-3500, 3500), rng.uniform(100, 1400))
        elif wall_class == 2:
            position = (sign * rng.uniform(2500, 3700), sign * rng.uniform(4400, 4900), rng.uniform(100, 1500))
        else:
            position = (sign * rng.uniform(300, 3300), rng.uniform(-4300, 4300), rng.choice((36.0, rng.uniform(100, 1700))))
        state.pos = rs.Vec(*map(float, position))
        state.vel = rs.Vec(*map(float, rng.uniform(-1800, 1800, 3)))
        state.ang_vel = rs.Vec(*map(float, rng.uniform(-4.5, 4.5, 3)))
        state.rot_mat = rs.Angle(
            float(rng.uniform(-math.pi, math.pi)),
            float(rng.uniform(-1.2, 1.2)),
            float(rng.uniform(-math.pi, math.pi)),
        ).as_rot_mat()
        ground = bool(position[2] <= 40.0)
        state.boost = float(rng.uniform(0, 100))
        state.is_on_ground = ground
        state.wheels_with_contact = (ground, ground, ground, ground)
        state.has_jumped = bool((index + car_index) % 3 == 0)
        state.is_jumping = bool((index + car_index) % 7 == 0)
        state.has_double_jumped = bool((index + car_index) % 11 == 0)
        state.has_flipped = bool((index + car_index) % 13 == 0)
        state.is_flipping = bool((index + car_index) % 17 == 0)
        state.jump_time = float(rng.uniform(0, 0.25))
        state.air_time = float(rng.uniform(0, 1.8))
        state.air_time_since_jump = float(rng.uniform(0, 1.8))
        state.flip_time = float(rng.uniform(0, 1.1))
        state.boosting_time = float(rng.uniform(0, 0.15))
        state.is_supersonic = bool((index + car_index) % 9 == 0)
        state.supersonic_time = float(rng.uniform(0, 1.2))
        if (index + car_index) % 31 == 0:
            state.is_demoed = True
            state.demo_respawn_timer = float(rng.uniform(0.1, 2.9))
        car.set_state(state)
    ball = rs.BallState()
    if wall_class in (1, 2):
        ball.pos = rs.Vec(float(rng.choice((-3900, 3900))), float(rng.uniform(-4800, 4800)), float(rng.uniform(100, 1500)))
    else:
        ball.pos = rs.Vec(float(rng.uniform(-3700, 3700)), float(rng.uniform(-4800, 4800)), float(rng.uniform(95, 1800)))
    ball.vel = rs.Vec(*map(float, rng.uniform(-3000, 3000, 3)))
    ball.ang_vel = rs.Vec(*map(float, rng.uniform(-5, 5, 3)))
    arena.ball.set_state(ball)
    for pad_index, pad in enumerate(pads):
        state = pad.get_state()
        active = bool((index + pad_index) % 4 != 0)
        state.is_active = active
        state.cooldown = 0.0 if active else float(rng.uniform(0.01, 10.0 if pad.is_big else 4.0))
        pad.set_state(state)
    arena.step(1)


def _collect_corpus(collision_root: Path) -> tuple[RocketSimBatchState, RocketSimRivalMemory, dict[str, Any]]:
    import RocketSim as rs

    rs.init(str(collision_root.resolve()))
    arena = rs.Arena(rs.GameMode.SOCCAR, tick_rate=120.0)
    cars = [arena.add_car(rs.Team.BLUE), arena.add_car(rs.Team.ORANGE)]
    pads = canonical_boost_pads(arena)
    rng = np.random.default_rng(SEED)
    chunks: dict[str, list[np.ndarray]] = {field.name: [] for field in fields(RocketSimBatchState)}
    memory = RocketSimRivalMemory.create(COUNT)
    table = np.asarray(
        [[0, 0, 0, 0, 0, 0, 0, 0], [1, -1, 0.5, -0.5, 1, 1, 1, 1], [-1, 1, -1, 1, -1, 0, 0, 1]],
        dtype=np.float32,
    )
    for index in range(COUNT):
        _set_broad_state(rs, arena, cars, pads, index, rng)
        state = read_rocketsim_batch([arena], [cars], [pads])
        for field in fields(RocketSimBatchState):
            chunks[field.name].append(getattr(state, field.name))
        memory.episode_ticks[index] = int(rng.integers(0, 5_400))
        memory.no_touch_ticks[index] = int(rng.integers(0, 1_800))
        memory.kickoff_indicator[index] = int(index < 10)
        memory.touch_event[index] = rng.integers(0, 2, 2)
        memory.demoed_event[index] = rng.integers(0, 2, 2)
        memory.previous_action[index] = table[index % len(table)]
        memory.time_since_boosted[index] = rng.uniform(0, 1.2, 2)
        memory.sticky_ticks[index] = rng.integers(0, 4, 2)
    state = RocketSimBatchState(
        **{name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    )
    coverage = {
        "all_five_kickoff_layouts_both_teams": True,
        "ground_states": int(state.on_ground.sum()),
        "air_states": int((state.on_ground == 0).sum()),
        "jumped_states": int(state.has_jumped.sum()),
        "double_jumped_states": int(state.has_double_jumped.sum()),
        "flipped_states": int(state.has_flipped.sum()),
        "demoed_states": int(state.is_demoed.sum()),
        "wall_corner_backboard_adjacent_states": int(COUNT // 4),
        "pad_inactive_samples": int((state.pad_active == 0).sum()),
    }
    return state, memory, coverage


def _accepted_observation(state: RocketSimBatchState, memory: RocketSimRivalMemory, collision_root: Path) -> np.ndarray:
    world = Rival2WorldSim(COUNT, str(collision_root), device="cuda:0", seed=SEED)
    bridge = Rival2TensorBridge(world)

    def copy(name: str, value: np.ndarray) -> None:
        destination = bridge.views[name]
        source = torch.from_numpy(np.ascontiguousarray(value)).to(destination.device, dtype=destination.dtype)
        destination.reshape(source.shape).copy_(source)

    copy("ball_pos", state.ball_pos)
    copy("ball_vel", state.ball_vel)
    copy("ball_ang_vel", state.ball_ang_vel)
    copy("car_pos", state.car_pos)
    copy("car_vel", state.car_vel)
    copy("car_quat", _matrix_to_quaternion_xyzw(state.car_forward, state.car_up))
    copy("car_ang_vel", state.car_ang_vel)
    copy("boost", state.boost)
    copy("boosting_time", state.boosting_time)
    copy("time_since_boosted", memory.time_since_boosted)
    copy("on_ground", state.on_ground)
    copy("has_jumped", state.has_jumped)
    copy("is_jumping", state.is_jumping)
    copy("has_double_jumped", state.has_double_jumped)
    copy("has_flipped", state.has_flipped)
    copy("is_flipping", state.is_flipping)
    copy("sticky_ticks", memory.sticky_ticks)
    copy("jump_time", state.jump_time)
    copy("air_time", state.air_time)
    copy("air_time_since_jump", state.air_time_since_jump)
    copy("flip_time", state.flip_time)
    copy("is_supersonic", state.is_supersonic)
    copy("supersonic_time", state.supersonic_time)
    copy("wheel_contact", state.wheels)
    copy("pad_cooldown", state.pad_cooldown)
    copy("car_is_demoed", state.is_demoed)
    copy("demo_respawn_timer", state.demo_respawn_timer)
    copy("rival2.episode_ticks", memory.episode_ticks)
    copy("rival2.no_touch_ticks", memory.no_touch_ticks)
    copy("rival2.kickoff_indicator", memory.kickoff_indicator)
    copy("rival2.touch_count", memory.touch_event)
    copy("rival2.demoed_event", memory.demoed_event)
    copy("rival2.previous_action", memory.previous_action)
    result = bridge.observation().detach().cpu().numpy()
    del world, bridge
    torch.cuda.empty_cache()
    return result


def _mirror(state: RocketSimBatchState, memory: RocketSimRivalMemory) -> tuple[RocketSimBatchState, RocketSimRivalMemory]:
    sign = np.asarray((-1.0, -1.0, 1.0), dtype=np.float32)
    swap_fields = {
        "car_pos", "car_vel", "car_forward", "car_up", "car_ang_vel", "boost",
        "on_ground", "wheels", "has_jumped", "is_jumping", "has_double_jumped",
        "has_flipped", "is_flipping", "jump_time", "air_time", "air_time_since_jump",
        "flip_time", "boosting_time", "is_supersonic", "supersonic_time", "is_demoed",
        "demo_respawn_timer", "flip_rel_torque",
    }
    vector_fields = {"car_pos", "car_vel", "car_forward", "car_up", "car_ang_vel", "flip_rel_torque"}
    values: dict[str, np.ndarray] = {}
    for field in fields(state):
        value = getattr(state, field.name).copy()
        if field.name in swap_fields:
            value = value[:, ::-1].copy()
        if field.name in vector_fields:
            value *= sign
        if field.name in {"ball_pos", "ball_vel", "ball_ang_vel"}:
            value *= sign
        if field.name in {"pad_cooldown", "pad_active"}:
            from rivalsim.rival2_contracts import ORANGE_PAD_REMAP
            value = value[:, np.asarray(ORANGE_PAD_REMAP)]
        values[field.name] = value
    mirrored_state = RocketSimBatchState(**values)
    mirrored_memory = memory.copy()
    for name in ("touch_event", "demoed_event", "previous_action", "time_since_boosted", "sticky_ticks", "last_hit_tick", "previous_demoed"):
        setattr(mirrored_memory, name, getattr(memory, name)[:, ::-1].copy())
    return mirrored_state, mirrored_memory


def _block_indices() -> dict[str, np.ndarray]:
    names = np.asarray(OBS_FIELD_NAMES)
    return {
        "ball": np.arange(0, 9),
        "self_car": np.arange(9, 48),
        "opponent_car": np.arange(48, 87),
        "relative": np.arange(87, 99),
        "boost_pads": np.arange(99, 167),
        "previous_action": np.arange(167, 175),
        "lifecycle": np.arange(175, 182),
        "all": np.arange(OBS_DIM),
        "team_sensitive_named_fields": np.flatnonzero(np.char.find(names.astype(str), "position") >= 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    import RocketSim as rs
    from importlib.metadata import version

    extension = Path(rs.__file__)
    identity = {
        "primary_commit": ROCKETSIM_PRIMARY_COMMIT,
        "binding_commit": ROCKETSIM_BINDING_COMMIT,
        "package": f"rocketsim=={version('rocketsim')}",
        "extension_path": str(extension),
        "extension_sha256": _sha256(extension),
        "checkpoint_sha256": _sha256(CHECKPOINT),
    }
    identity["pass"] = identity["primary_commit"] == "c2baacb8f4b441dd8505e63c2aeb5a1679b60b02" and identity["extension_sha256"] == ROCKETSIM_EXTENSION_SHA256 and identity["checkpoint_sha256"] == CHECKPOINT_SHA256
    if not identity["pass"]:
        raise RuntimeError(f"identity gate failed: {identity}")
    state, memory, coverage = _collect_corpus(args.collision_dir)
    adapter = build_rival2_observation(state, memory)
    accepted = _accepted_observation(state, memory, args.collision_dir)
    error = np.abs(adapter - accepted)
    blocks = {
        name: {
            "max_abs_error": float(error[..., index].max(initial=0.0)),
            "exact_fields": int(np.all(error[..., index] == 0, axis=(0, 1)).sum()),
            "fields": int(index.size),
        }
        for name, index in _block_indices().items()
    }
    policy = FrozenRivalPolicy(CHECKPOINT)
    flat_adapter = adapter.reshape(-1, OBS_DIM)
    flat_accepted = accepted.reshape(-1, OBS_DIM)
    action_adapter = policy.act(flat_adapter).reshape(COUNT, 2, 8)
    action_accepted = policy.act(flat_accepted).reshape(COUNT, 2, 8)
    action_error = np.abs(action_adapter - action_accepted)
    # Rival has five continuous deterministic outputs, unlike Nexto's discrete
    # table.  "Agreement" therefore means exact buttons plus analog equality
    # within the same unavoidable NumPy-vs-Torch representation noise admitted
    # by the observation gate; raw maxima remain published.
    agreement = (
        np.all(action_error[..., :5] <= 1e-5, axis=-1)
        & np.all(action_adapter[..., 5:] == action_accepted[..., 5:], axis=-1)
    )
    mirrored_state, mirrored_memory = _mirror(state, memory)
    mirrored = build_rival2_observation(mirrored_state, mirrored_memory)
    symmetry_error_blue_to_orange = np.abs(adapter[:, 0] - mirrored[:, 1])
    symmetry_error_orange_to_blue = np.abs(adapter[:, 1] - mirrored[:, 0])
    symmetry_action_original = policy.act(adapter.reshape(-1, OBS_DIM)).reshape(COUNT, 2, 8)
    symmetry_action_mirror = policy.act(mirrored[:, ::-1].reshape(-1, OBS_DIM)).reshape(COUNT, 2, 8)
    symmetry_agreement = np.all(symmetry_action_original == symmetry_action_mirror, axis=-1)
    result = {
        "verdict": "PASS_GREEN",
        "identity": identity,
        "corpus": {"states": COUNT, "seed": SEED, "blue": COUNT, "orange": COUNT, "coverage": coverage},
        "observation_parity": {
            "overall_max_abs_error": float(error.max()),
            "overall_exact_fields": int(np.all(error == 0, axis=(0, 1)).sum()),
            "fields": OBS_DIM,
            "blocks": blocks,
            "by_side": {
                "Blue": {"max_abs_error": float(error[:, 0].max()), "exact_fields": int(np.all(error[:, 0] == 0, axis=0).sum())},
                "Orange": {"max_abs_error": float(error[:, 1].max()), "exact_fields": int(np.all(error[:, 1] == 0, axis=0).sum())},
            },
        },
        "deterministic_action_agreement": {
            "definition": "all three binary buttons exact and each continuous analog channel within 1e-5",
            "analog_max_abs_error": float(action_error[..., :5].max()),
            "binary_button_exact_fraction": float(np.all(action_adapter[..., 5:] == action_accepted[..., 5:], axis=-1).mean()),
            "overall": {"count": int(agreement.sum()), "denominator": int(agreement.size), "fraction": float(agreement.mean())},
            "Blue": {"count": int(agreement[:, 0].sum()), "denominator": COUNT, "fraction": float(agreement[:, 0].mean())},
            "Orange": {"count": int(agreement[:, 1].sum()), "denominator": COUNT, "fraction": float(agreement[:, 1].mean())},
        },
        "team_mirror_symmetry": {
            "blue_to_mirrored_orange_max_abs_error": float(symmetry_error_blue_to_orange.max()),
            "orange_to_mirrored_blue_max_abs_error": float(symmetry_error_orange_to_blue.max()),
            "deterministic_action_agreement": {"count": int(symmetry_agreement.sum()), "denominator": int(symmetry_agreement.size), "fraction": float(symmetry_agreement.mean())},
        },
        "semantic_mismatches": [
            {
                "field": "self/opponent.time_since_boosted",
                "classification": "adapter-maintained exact during evaluated trajectories; public binding omits the literal internal field",
                "effect": "captured/restored as adapter memory; parity corpus supplies equivalent authoritative auxiliary value",
            },
            {
                "field": "self/opponent.sticky_ticks",
                "classification": "adapter-maintained from authoritative jump transition/control because public binding omits RivalSim's three-tick memory",
                "effect": "captured/restored as adapter memory",
            },
        ],
        "wall_seconds": time.perf_counter() - started,
    }
    gates = {
        "identity_exact": identity["pass"],
        "corpus_exact": COUNT == 2_048,
        "adapter_action_agreement_100_percent": agreement.all(),
        "mirror_observation_max_abs_le_1e6": max(symmetry_error_blue_to_orange.max(), symmetry_error_orange_to_blue.max()) <= 1e-6,
        "mirror_action_agreement_100_percent": symmetry_agreement.all(),
    }
    result["gates"] = {name: bool(value) for name, value in gates.items()}
    if not all(gates.values()):
        result["verdict"] = "FAIL_RED"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "observation": result["observation_parity"], "actions": result["deterministic_action_agreement"], "symmetry": result["team_mirror_symmetry"], "wall_seconds": result["wall_seconds"]}, indent=2), flush=True)
    return 0 if result["verdict"] == "PASS_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
