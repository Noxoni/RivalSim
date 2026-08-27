"""Scalar pinned-upstream Wisp reference used only by the fidelity gate."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _basis(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
            (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
            (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _prediction(pos: np.ndarray, vel: np.ndarray, tick: int) -> tuple[np.ndarray, np.ndarray]:
    time = np.float32(tick / 120.0)
    result = pos + vel * time
    result = result.astype(np.float32)
    result[2] += np.float32(-325.0) * time * time
    result_vel = vel.copy()
    result_vel[2] += np.float32(-650.0) * time
    lower, upper = np.float32(91.25), np.float32(2044.0 - 91.25)
    span = upper - lower
    phase = np.remainder(result[2] - lower, 2 * span)
    descending = phase > span
    result[2] = lower + (2 * span - phase if descending else phase)
    result_vel[2] = (-result_vel[2] if descending else result_vel[2]) * np.float32(0.6)
    return result, result_vel


def main() -> None:
    args = parse_args()
    source = args.upstream_root.resolve() / "src"
    sys.path.insert(0, str(source))
    import obs_builder as obs_builder_module
    from action_parser import XMirroredActionParser
    from backend.gamestate.action import Action
    from backend.gamestate.gamestate import GameState
    from backend.gamestate.phys_obj import PhysObj
    from backend.gamestate.player import Player
    from backend.gamestate.rot_mat import RotMat
    from backend.gamestate.team import Team
    from backend.gamestate.vec import Vec
    from backend.model import ActivationType, ModelInfo, ModelSet
    from backend.rlbot_conversion import convert_vec3
    from eta import linear_eta, rough_eta
    from obs_builder import CustomObs

    eta_trace: dict[int, list[tuple[int, float, float, float]]] = {}

    def traced_rough_eta(player, prediction):
        estimate = traced_rough_eta.cache.setdefault(player.index, 0.0)
        calls = eta_trace.setdefault(player.index, [])
        for _eta_pass in range(2):
            tick = min(int(estimate * 120), 599)
            target = convert_vec3(prediction.slices[tick].physics.location)
            delta = target - player.pos
            distance = delta.length()
            direction = delta / distance
            initial_velocity = player.vel.dot(direction)
            target_distance = distance - 136.875
            estimate = linear_eta(
                initial_velocity,
                target_distance,
                player.boost / 33.3,
            )
            calls.append((tick, initial_velocity, target_distance, estimate))
        traced_rough_eta.cache[player.index] = estimate
        return estimate

    traced_rough_eta.cache = {}
    obs_builder_module.rough_eta = traced_rough_eta

    data = np.load(args.input, allow_pickle=False)
    random.seed(2026082703)
    count = int(data["side"].shape[0])
    parser = XMirroredActionParser()
    models = ModelSet(
        ModelInfo(source / "models/POLICY.lt", ActivationType.RELU),
        ModelInfo(source / "models/SHARED_HEAD.lt", ActivationType.RELU),
        device="cpu",
    )
    observations = np.empty((count, 432), dtype=np.float32)
    masks = np.empty((count, 90), dtype=np.bool_)
    indices = np.empty(count, dtype=np.int64)
    controllers = np.empty((count, 8), dtype=np.float32)
    opponent_slots = np.empty(count, dtype=np.int64)
    eta_v0 = np.zeros((count, 2, 2), dtype=np.float64)
    eta_x0 = np.zeros((count, 2, 2), dtype=np.float64)
    eta_time = np.zeros((count, 2, 2), dtype=np.float64)
    eta_tick = np.zeros((count, 2, 2), dtype=np.int64)
    for row in range(count):
        for physical in range(2):
            estimate = 0.0
            for eta_pass in range(2):
                tick = min(int(estimate * 120), 599)
                predicted_position, _ = _prediction(
                    data["ball_pos"][row], data["ball_vel"][row], tick + 1
                )
                delta = predicted_position - data["car_pos"][row, physical]
                distance = np.linalg.norm(delta)
                direction = delta / distance
                initial_velocity = np.dot(data["car_vel"][row, physical], direction)
                target_distance = distance - 136.875
                estimate = linear_eta(
                    initial_velocity,
                    target_distance,
                    float(data["boost"][row, physical]) / 33.3,
                )
                eta_tick[row, physical, eta_pass] = tick
                eta_v0[row, physical, eta_pass] = initial_velocity
                eta_x0[row, physical, eta_pass] = target_distance
                eta_time[row, physical, eta_pass] = estimate
        rough_eta.cache = {}
        traced_rough_eta.cache = {}
        eta_trace.clear()
        state = GameState()
        state.ball = PhysObj(
            pos=Vec(data["ball_pos"][row]),
            rot_mat=RotMat(),
            vel=Vec(data["ball_vel"][row]),
            ang_vel=Vec(data["ball_ang_vel"][row]),
        )
        state.boost_pad_timers = data["wisp_pad_cooldown"][row].copy()
        state.boost_pads = state.boost_pad_timers == 0
        state.boost_pad_timers_inv = state.boost_pad_timers[::-1].copy()
        state.boost_pads_inv = state.boost_pads[::-1].copy()
        for physical in range(2):
            player = Player(Team.BLUE if physical == 0 else Team.ORANGE)
            player.index = physical
            player.pos = Vec(data["car_pos"][row, physical])
            player.vel = Vec(data["car_vel"][row, physical])
            player.ang_vel = Vec(data["car_ang_vel"][row, physical])
            player.rot_mat = RotMat(_basis(data["car_quat"][row, physical]).tolist())
            player.boost = float(data["boost"][row, physical])
            player.is_on_ground = bool(data["on_ground"][row, physical])
            player.is_jumping = bool(data["is_jumping"][row, physical])
            player.has_jumped = bool(data["has_jumped"][row, physical])
            player.has_double_jumped = bool(data["has_double_jumped"][row, physical])
            player.has_flipped = bool(data["has_flipped"][row, physical])
            player.is_demoed = bool(data["demoed"][row, physical])
            player.air_time_since_jump = float(data["air_time_since_jump"][row, physical])
            player.ball_touched_step = bool(data["touch_count"][row, physical])
            player.handbrake_val = float(data["handbrake_value"][row, physical])
            player.prev_action = Action(
                data["previous_action"][row] if physical == data["side"][row] else np.zeros(8)
            )
            state.players.append(player)
        slices = []
        for tick in range(600):
            position, velocity = _prediction(data["ball_pos"][row], data["ball_vel"][row], tick + 1)
            slices.append(
                SimpleNamespace(
                    physics=SimpleNamespace(
                        location=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
                        velocity=SimpleNamespace(x=velocity[0], y=velocity[1], z=velocity[2]),
                        angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                        rotation=SimpleNamespace(yaw=0.0, pitch=0.0, roll=0.0),
                    )
                )
            )
        prediction = SimpleNamespace(slices=slices)
        controlled = state.players[int(data["side"][row])]
        observation = (
            CustomObs()
            .build_obs(controlled, state, prediction, int(data["score_diff"][row]))
            .get_np()
        )
        for physical in range(2):
            calls = eta_trace.get(physical, [])
            for eta_pass, (tick, initial_velocity, target_distance, estimate) in enumerate(calls):
                eta_tick[row, physical, eta_pass] = tick
                eta_v0[row, physical, eta_pass] = initial_velocity
                eta_x0[row, physical, eta_pass] = target_distance
                eta_time[row, physical, eta_pass] = estimate
        action_mask = parser.get_action_mask(controlled, state).numpy()
        action_index = models.get_action(observation, action_mask, True)
        action = parser.get_action(action_index, controlled, state).get_np()
        observations[row] = observation
        masks[row] = action_mask
        indices[row] = action_index
        controllers[row] = action
        opponent_blocks = observation[279:432].reshape(3, 51)
        opponent_slots[row] = int(np.argmax(np.abs(opponent_blocks).sum(axis=1)))
    np.savez_compressed(
        args.output,
        observation=observations,
        action_mask=masks,
        action_index=indices,
        controller=controllers,
        opponent_slot=opponent_slots,
        eta_v0=eta_v0,
        eta_x0=eta_x0,
        eta_time=eta_time,
        eta_tick=eta_tick,
    )


if __name__ == "__main__":
    main()
