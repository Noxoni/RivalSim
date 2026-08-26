"""Evaluation-only RocketSim adapters for frozen Rival 2.0 and public Nexto.

The production Rival contract is not redefined here.  This module reads the
public state of pinned ``rocketsim==2.2.1`` and reconstructs the already-frozen
``RIVAL2_OBS_V1`` tensor, while retaining the small policy/runtime memory which
RocketSim does not own (previous actions and evaluation episode counters).
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS
from rivalsim.rival2_contracts import (
    AIR_TIME_SCALE,
    ANGULAR_SPEED_SCALE,
    BALL_LINEAR_SPEED_SCALE,
    BOOSTING_TIME_SCALE,
    BOOST_SCALE,
    CAR_LINEAR_SPEED_SCALE,
    DEMO_TIMER_SCALE,
    EPISODE_AGE_SCALE_TICKS,
    FLIP_TIME_SCALE,
    JUMP_TIME_SCALE,
    NO_TOUCH_AGE_SCALE_TICKS,
    OBS_DIM,
    ORANGE_PAD_REMAP,
    POSITION_SCALE,
    STICKY_TICK_SCALE,
    SUPERSONIC_TIME_SCALE,
    TIME_SINCE_BOOSTED_SCALE,
)
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    sample_hybrid_action,
)

PHYSICS_HZ = 120
DT = np.float32(1.0 / PHYSICS_HZ)
RIVAL_CADENCE_TICKS = 4
NEXTO_CADENCE_TICKS = 8
NEXTO_MODEL_SHA256 = "BF5343B5EEACAC6BF7CDB75DAC4A5C14BA0F94D820EAE75F00A211B6119D69FA"


def _vec(value: Any) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float32)


def _matrix(value: Any) -> np.ndarray:
    return np.asarray(
        (
            (value.forward.x, value.right.x, value.up.x),
            (value.forward.y, value.right.y, value.up.y),
            (value.forward.z, value.right.z, value.up.z),
        ),
        dtype=np.float32,
    )


def _timer(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(value.astype(np.float32) / np.float32(scale), 0.0, 1.0)[..., None]


@dataclass(slots=True)
class RocketSimRivalMemory:
    """Non-physics memory required by ``RIVAL2_OBS_V1`` and its cadence."""

    episode_ticks: np.ndarray
    no_touch_ticks: np.ndarray
    kickoff_indicator: np.ndarray
    touch_event: np.ndarray
    demoed_event: np.ndarray
    previous_action: np.ndarray
    time_since_boosted: np.ndarray
    sticky_ticks: np.ndarray
    last_hit_tick: np.ndarray
    previous_demoed: np.ndarray

    @classmethod
    def create(cls, worlds: int) -> "RocketSimRivalMemory":
        return cls(
            episode_ticks=np.zeros(worlds, dtype=np.int32),
            no_touch_ticks=np.zeros(worlds, dtype=np.int32),
            kickoff_indicator=np.ones(worlds, dtype=np.int32),
            touch_event=np.zeros((worlds, 2), dtype=np.int32),
            demoed_event=np.zeros((worlds, 2), dtype=np.int32),
            previous_action=np.zeros((worlds, 2, 8), dtype=np.float32),
            time_since_boosted=np.zeros((worlds, 2), dtype=np.float32),
            sticky_ticks=np.zeros((worlds, 2), dtype=np.int32),
            last_hit_tick=np.full((worlds, 2), -1, dtype=np.int64),
            previous_demoed=np.zeros((worlds, 2), dtype=bool),
        )

    def reset_rows(self, rows: np.ndarray | Sequence[int]) -> None:
        index = np.asarray(rows)
        self.episode_ticks[index] = 0
        self.no_touch_ticks[index] = 0
        self.kickoff_indicator[index] = 1
        self.touch_event[index] = 0
        self.demoed_event[index] = 0
        self.previous_action[index] = 0
        self.time_since_boosted[index] = 0
        self.sticky_ticks[index] = 0
        self.last_hit_tick[index] = -1
        self.previous_demoed[index] = False

    def clear_interval_events(self) -> None:
        self.touch_event.fill(0)
        self.demoed_event.fill(0)
        self.kickoff_indicator.fill(0)

    def copy(self) -> "RocketSimRivalMemory":
        return RocketSimRivalMemory(
            **{field.name: getattr(self, field.name).copy() for field in fields(self)}
        )


@dataclass(slots=True)
class RocketSimBatchState:
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    ball_ang_vel: np.ndarray
    car_pos: np.ndarray
    car_vel: np.ndarray
    car_forward: np.ndarray
    car_up: np.ndarray
    car_ang_vel: np.ndarray
    boost: np.ndarray
    on_ground: np.ndarray
    wheels: np.ndarray
    has_jumped: np.ndarray
    is_jumping: np.ndarray
    has_double_jumped: np.ndarray
    has_flipped: np.ndarray
    is_flipping: np.ndarray
    jump_time: np.ndarray
    air_time: np.ndarray
    air_time_since_jump: np.ndarray
    flip_time: np.ndarray
    boosting_time: np.ndarray
    is_supersonic: np.ndarray
    supersonic_time: np.ndarray
    is_demoed: np.ndarray
    demo_respawn_timer: np.ndarray
    pad_cooldown: np.ndarray
    pad_active: np.ndarray
    flip_rel_torque: np.ndarray

    @property
    def worlds(self) -> int:
        return int(self.ball_pos.shape[0])


def canonical_boost_pads(arena: Any) -> list[Any]:
    """Return RocketSim pads in RivalSim's six-big-then-small canonical order."""

    unordered = arena.get_boost_pads()
    result: list[Any] = []
    for expected in SOCCAR_PAD_POSITIONS:
        matches = [
            pad
            for pad in unordered
            if np.array_equal(_vec(pad.get_pos()), np.asarray(expected, dtype=np.float32))
        ]
        if len(matches) != 1:
            raise RuntimeError(f"failed to map RocketSim pad {expected.tolist()}")
        result.append(matches[0])
    return result


def read_rocketsim_batch(
    arenas: Sequence[Any],
    cars: Sequence[Sequence[Any]],
    pads: Sequence[Sequence[Any]],
) -> RocketSimBatchState:
    worlds = len(arenas)
    if len(cars) != worlds or len(pads) != worlds:
        raise ValueError("arena/car/pad batch lengths differ")
    ball_pos = np.empty((worlds, 3), dtype=np.float32)
    ball_vel = np.empty_like(ball_pos)
    ball_ang_vel = np.empty_like(ball_pos)
    car_pos = np.empty((worlds, 2, 3), dtype=np.float32)
    car_vel = np.empty_like(car_pos)
    car_forward = np.empty_like(car_pos)
    car_up = np.empty_like(car_pos)
    car_ang_vel = np.empty_like(car_pos)
    flip_rel_torque = np.empty_like(car_pos)
    scalar_float = {
        name: np.empty((worlds, 2), dtype=np.float32)
        for name in (
            "boost", "jump_time", "air_time", "air_time_since_jump", "flip_time",
            "boosting_time", "supersonic_time", "demo_respawn_timer",
        )
    }
    scalar_int = {
        name: np.empty((worlds, 2), dtype=np.int32)
        for name in (
            "on_ground", "has_jumped", "is_jumping", "has_double_jumped",
            "has_flipped", "is_flipping", "is_supersonic", "is_demoed",
        )
    }
    wheels = np.empty((worlds, 2, 4), dtype=np.int32)
    pad_cooldown = np.empty((worlds, 34), dtype=np.float32)
    pad_active = np.empty((worlds, 34), dtype=np.int32)
    for world, arena in enumerate(arenas):
        ball = arena.ball.get_state()
        ball_pos[world] = _vec(ball.pos)
        ball_vel[world] = _vec(ball.vel)
        ball_ang_vel[world] = _vec(ball.ang_vel)
        for car_index, car in enumerate(cars[world]):
            state = car.get_state()
            car_pos[world, car_index] = _vec(state.pos)
            car_vel[world, car_index] = _vec(state.vel)
            matrix = _matrix(state.rot_mat)
            car_forward[world, car_index] = matrix[:, 0]
            car_up[world, car_index] = matrix[:, 2]
            car_ang_vel[world, car_index] = _vec(state.ang_vel)
            flip_rel_torque[world, car_index] = _vec(state.flip_rel_torque)
            scalar_float["boost"][world, car_index] = state.boost
            scalar_float["jump_time"][world, car_index] = state.jump_time
            scalar_float["air_time"][world, car_index] = state.air_time
            scalar_float["air_time_since_jump"][world, car_index] = state.air_time_since_jump
            scalar_float["flip_time"][world, car_index] = state.flip_time
            scalar_float["boosting_time"][world, car_index] = state.boosting_time
            scalar_float["supersonic_time"][world, car_index] = state.supersonic_time
            scalar_float["demo_respawn_timer"][world, car_index] = state.demo_respawn_timer
            scalar_int["on_ground"][world, car_index] = state.is_on_ground
            scalar_int["has_jumped"][world, car_index] = state.has_jumped
            scalar_int["is_jumping"][world, car_index] = state.is_jumping
            scalar_int["has_double_jumped"][world, car_index] = state.has_double_jumped
            scalar_int["has_flipped"][world, car_index] = state.has_flipped
            scalar_int["is_flipping"][world, car_index] = state.is_flipping
            scalar_int["is_supersonic"][world, car_index] = state.is_supersonic
            scalar_int["is_demoed"][world, car_index] = state.is_demoed
            wheels[world, car_index] = np.asarray(state.wheels_with_contact, dtype=np.int32)
        for pad_index, pad in enumerate(pads[world]):
            state = pad.get_state()
            pad_cooldown[world, pad_index] = state.cooldown
            pad_active[world, pad_index] = state.is_active
    return RocketSimBatchState(
        ball_pos=ball_pos, ball_vel=ball_vel, ball_ang_vel=ball_ang_vel,
        car_pos=car_pos, car_vel=car_vel, car_forward=car_forward, car_up=car_up,
        car_ang_vel=car_ang_vel, wheels=wheels, pad_cooldown=pad_cooldown,
        pad_active=pad_active, flip_rel_torque=flip_rel_torque,
        **scalar_float, **scalar_int,
    )


def build_rival2_observation(
    state: RocketSimBatchState, memory: RocketSimRivalMemory
) -> np.ndarray:
    """Direct NumPy transcription of accepted ``Rival2TensorBridge.observation``."""

    worlds = state.worlds
    if memory.episode_ticks.shape != (worlds,):
        raise ValueError("adapter memory batch mismatch")
    position_scale = np.asarray(POSITION_SCALE, dtype=np.float32)
    signs = np.asarray(((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0)), dtype=np.float32)
    pad_durations = np.asarray([10.0] * 6 + [4.0] * 28, dtype=np.float32)
    pad_maps = (np.arange(34, dtype=np.int64), np.asarray(ORANGE_PAD_REMAP, dtype=np.int64))

    def car_block(car: int, sign: np.ndarray) -> np.ndarray:
        on_ground = state.on_ground[:, car].astype(np.float32)
        has_jumped = state.has_jumped[:, car].astype(np.float32)
        has_double = state.has_double_jumped[:, car].astype(np.float32)
        has_flipped = state.has_flipped[:, car].astype(np.float32)
        dodge = (
            (has_double == 0)
            & (has_flipped == 0)
            & (
                (on_ground != 0)
                | ((has_jumped != 0) & (state.air_time_since_jump[:, car] < AIR_TIME_SCALE))
            )
        ).astype(np.float32)
        return np.concatenate(
            (
                state.car_pos[:, car] * sign / position_scale,
                state.car_vel[:, car] * sign / np.float32(CAR_LINEAR_SPEED_SCALE),
                state.car_forward[:, car] * sign,
                state.car_up[:, car] * sign,
                state.car_ang_vel[:, car] * sign / np.float32(ANGULAR_SPEED_SCALE),
                (state.boost[:, car] / np.float32(BOOST_SCALE))[:, None],
                on_ground[:, None],
                has_jumped[:, None],
                state.is_jumping[:, car, None].astype(np.float32),
                has_double[:, None],
                has_flipped[:, None],
                state.is_flipping[:, car, None].astype(np.float32),
                (has_jumped == 0).astype(np.float32)[:, None],
                dodge[:, None],
                state.is_demoed[:, car, None].astype(np.float32),
                _timer(state.demo_respawn_timer[:, car], DEMO_TIMER_SCALE),
                state.wheels[:, car].astype(np.float32),
                _timer(state.jump_time[:, car], JUMP_TIME_SCALE),
                _timer(state.air_time[:, car], AIR_TIME_SCALE),
                _timer(state.air_time_since_jump[:, car], AIR_TIME_SCALE),
                _timer(state.flip_time[:, car], FLIP_TIME_SCALE),
                _timer(state.boosting_time[:, car], BOOSTING_TIME_SCALE),
                _timer(memory.time_since_boosted[:, car], TIME_SINCE_BOOSTED_SCALE),
                state.is_supersonic[:, car, None].astype(np.float32),
                _timer(state.supersonic_time[:, car], SUPERSONIC_TIME_SCALE),
                _timer(memory.sticky_ticks[:, car], STICKY_TICK_SCALE),
            ),
            axis=1,
        )

    observations: list[np.ndarray] = []
    for agent in range(2):
        opponent = 1 - agent
        sign = signs[agent]
        pad_index = pad_maps[agent]
        cooldown = state.pad_cooldown[:, pad_index]
        duration = pad_durations[pad_index]
        pads = np.stack(
            ((cooldown == 0).astype(np.float32), np.clip(cooldown / duration, 0.0, 1.0)),
            axis=-1,
        ).reshape(worlds, 68)
        ball = np.concatenate(
            (
                state.ball_pos * sign / position_scale,
                state.ball_vel * sign / np.float32(BALL_LINEAR_SPEED_SCALE),
                state.ball_ang_vel * sign / np.float32(ANGULAR_SPEED_SCALE),
            ),
            axis=1,
        )
        relative = np.concatenate(
            (
                (state.ball_pos - state.car_pos[:, agent]) * sign / position_scale,
                (state.ball_vel - state.car_vel[:, agent]) * sign / np.float32(BALL_LINEAR_SPEED_SCALE),
                (state.car_pos[:, opponent] - state.car_pos[:, agent]) * sign / position_scale,
                (state.car_vel[:, opponent] - state.car_vel[:, agent]) * sign / np.float32(CAR_LINEAR_SPEED_SCALE),
            ),
            axis=1,
        )
        lifecycle = np.concatenate(
            (
                memory.kickoff_indicator.astype(np.float32)[:, None],
                (memory.touch_event[:, agent] > 0).astype(np.float32)[:, None],
                (memory.touch_event[:, opponent] > 0).astype(np.float32)[:, None],
                (memory.demoed_event[:, agent] > 0).astype(np.float32)[:, None],
                (memory.demoed_event[:, opponent] > 0).astype(np.float32)[:, None],
                _timer(memory.episode_ticks, EPISODE_AGE_SCALE_TICKS),
                _timer(memory.no_touch_ticks, NO_TOUCH_AGE_SCALE_TICKS),
            ),
            axis=1,
        )
        observation = np.concatenate(
            (
                ball,
                car_block(agent, sign),
                car_block(opponent, sign),
                relative,
                pads,
                memory.previous_action[:, agent],
                lifecycle,
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        if observation.shape != (worlds, OBS_DIM):
            raise RuntimeError(f"RocketSim Rival observation shape mismatch: {observation.shape}")
        observations.append(observation)
    return np.stack(observations, axis=1)


def update_rival_memory_after_tick(
    memory: RocketSimRivalMemory,
    before: RocketSimBatchState,
    after: RocketSimBatchState,
    controls: np.ndarray,
    car_hit_ticks: np.ndarray,
    active: np.ndarray | None = None,
) -> None:
    """Advance adapter-only fields from authoritative before/after RocketSim state."""

    if active is None:
        active = np.ones(memory.episode_ticks.shape, dtype=bool)
    active = np.asarray(active, dtype=bool)
    active_car = active[:, None]
    new_touch = car_hit_ticks > memory.last_hit_tick
    new_touch &= car_hit_ticks >= 0
    new_touch &= active_car
    memory.touch_event += new_touch.astype(np.int32)
    any_touch = new_touch.any(axis=1)
    memory.no_touch_ticks[active] = np.where(
        any_touch[active], 0, memory.no_touch_ticks[active] + 1
    ).astype(np.int32)
    memory.last_hit_tick[active] = np.maximum(
        memory.last_hit_tick[active], car_hit_ticks[active]
    )
    demoed = after.is_demoed != 0
    newly_demoed = demoed & ~memory.previous_demoed
    memory.demoed_event[active] |= newly_demoed[active].astype(np.int32)
    memory.previous_demoed[active] = demoed[active]
    memory.episode_ticks[active] += 1
    memory.time_since_boosted[active] = np.where(
        after.boosting_time[active] > 0,
        np.float32(0.0),
        memory.time_since_boosted[active] + DT,
    ).astype(np.float32)
    started_jump = (
        (before.is_jumping == 0)
        & (after.is_jumping != 0)
        & (before.on_ground != 0)
        & (controls[..., 5] >= 0.5)
    )
    memory.sticky_ticks[active] = np.where(
        started_jump[active],
        2,
        np.maximum(memory.sticky_ticks[active] - 1, 0),
    ).astype(np.int32)


def car_hit_ticks(cars: Sequence[Sequence[Any]]) -> np.ndarray:
    result = np.full((len(cars), 2), -1, dtype=np.int64)
    for world, pair in enumerate(cars):
        for index, car in enumerate(pair):
            info = car.get_state().ball_hit_info
            if info.is_valid:
                result[world, index] = int(info.tick_count_when_hit)
    return result


def actions_to_controls(rs: Any, actions: np.ndarray) -> list[list[Any]]:
    actions = np.asarray(actions, dtype=np.float32)
    result: list[list[Any]] = []
    for world in range(actions.shape[0]):
        pair: list[Any] = []
        for car in range(2):
            row = actions[world, car]
            pair.append(
                rs.CarControls(
                    throttle=float(np.clip(row[0], -1, 1)),
                    steer=float(np.clip(row[1], -1, 1)),
                    pitch=float(np.clip(row[2], -1, 1)),
                    yaw=float(np.clip(row[3], -1, 1)),
                    roll=float(np.clip(row[4], -1, 1)),
                    jump=bool(row[5] >= 0.5),
                    boost=bool(row[6] >= 0.5),
                    handbrake=bool(row[7] >= 0.5),
                )
            )
        result.append(pair)
    return result


class FrozenRivalPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda:0",
        stochastic: bool = False,
        seed: int = 0,
    ):
        self.device = torch.device(device)
        self.stochastic = bool(stochastic)
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        config = Rival2PolicyConfig(**payload["policy_config"])
        if payload.get("policy_config_hash") != config.content_hash:
            raise RuntimeError("Rival checkpoint policy config mismatch")
        self.model = Rival2ActorCritic(config).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.identity = {
            "policy_version": int(payload["policy_version"]),
            "total_agent_samples": int(payload["total_agent_samples"]),
            "policy_config_hash": config.content_hash,
            "reward_version": payload["reward_version"],
        }

    def act(self, observation: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(observation)).to(self.device)
        with torch.inference_mode():
            actor, _ = self.model(tensor)
            if self.stochastic:
                action = sample_hybrid_action(actor, generator=self.generator).action
            else:
                action = deterministic_hybrid_action(actor)
        return action.detach().cpu().numpy().astype(np.float32, copy=False)


def _nexto_action_table() -> np.ndarray:
    actions: list[list[float]] = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    return np.asarray(actions, dtype=np.float32)


def _nexto_kickoff_sequence() -> np.ndarray:
    rows: list[list[float]] = []
    rows += 11 * 4 * [[1, 0, 0, 0, 0, 0, 1, 0]]
    rows += 4 * 4 * [[1, -1, 0, -1, 0, 0, 1, 0]]
    rows += 2 * 4 * [[1, 0, 0, 0, 0, 1, 1, 0]]
    rows += 1 * 4 * [[1, 0, 0, 0, 0, 0, 1, 0]]
    rows += 1 * 4 * [[1, 0, -0.7, 0.8, 0, 1, 1, 0]]
    rows += 13 * 4 * [[1, 0, 1, 0, 0, 0, 1, 0]]
    rows += 10 * 4 * [[1, 0, 0.5, 0, 1, 0, 0, 0]]
    return np.asarray(rows, dtype=np.float32)


def build_nexto_source_observation(
    state: RocketSimBatchState,
    side: np.ndarray,
    previous_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct float64 transcription of pinned public ``nexto_obs.py`` for 1v1."""

    worlds = state.worlds
    q = np.zeros((worlds, 1, 32), dtype=np.float64)
    kv = np.zeros((worlds, 37, 24), dtype=np.float64)
    mask = np.zeros((worlds, 37), dtype=np.float64)
    kv[:, 2, 3] = 1
    kv[:, 2, 5:8] = state.ball_pos
    kv[:, 2, 8:11] = state.ball_vel
    kv[:, 2, 17:20] = state.ball_ang_vel
    kv[:, 3:, 4] = 1
    # Upstream hard-coded source order is RLGym pad order.  Map canonical
    # physical pads into that order through the already-validated coordinates.
    from third_party.nexto.adapter import NEXTO_BOOST_LOCATIONS, nexto_pad_mapping

    mapping, _ = nexto_pad_mapping()
    kv[:, 3:, 5:8] = NEXTO_BOOST_LOCATIONS.astype(np.float64)
    kv[:, 3:, 20] = 0.12 + 0.88 * (NEXTO_BOOST_LOCATIONS[:, 2] > 72)
    kv[:, 3:, 21] = state.pad_active[:, mapping]
    rows = np.arange(worlds)
    physical_order = np.stack((side, 1 - side), axis=1)
    teams = physical_order.astype(np.float64)
    kv[:, :2, 1] = 1 - teams
    kv[:, :2, 2] = teams
    for entity in range(2):
        physical = physical_order[:, entity]
        kv[:, entity, 5:8] = state.car_pos[rows, physical]
        kv[:, entity, 8:11] = state.car_vel[rows, physical]
        kv[:, entity, 11:14] = state.car_forward[rows, physical]
        kv[:, entity, 14:17] = state.car_up[rows, physical]
        kv[:, entity, 17:20] = state.car_ang_vel[rows, physical]
        kv[:, entity, 20] = state.boost[rows, physical] / 100.0
        kv[:, entity, 21] = state.is_demoed[rows, physical]
        kv[:, entity, 22] = state.on_ground[rows, physical]
        kv[:, entity, 23] = (
            (state.has_flipped[rows, physical] == 0)
            & (state.has_double_jumped[rows, physical] == 0)
            & (state.air_time_since_jump[rows, physical] < 1.25)
        )
    kv[:, 0, 0] = 1
    invert = np.asarray([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    orange = side == 1
    kv[orange] *= invert
    mate = kv[orange, :, 1].copy()
    kv[orange, :, 1] = kv[orange, :, 2]
    kv[orange, :, 2] = mate
    norm = np.asarray([1.0] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)
    kv /= norm
    q[:, 0, :24] = kv[:, 0, :]
    q[:, 0, 24:32] = previous_action
    kv[:, :, 5:8] -= q[:, :, 5:8]
    forward = q[:, :, 11:14]
    theta = np.expand_dims(np.arctan2(forward[..., 0], forward[..., 1]), axis=-1)
    ct = np.cos(theta)
    st = np.sin(theta)
    xs = kv[:, :, 5:20:3]
    ys = kv[:, :, 6:20:3]
    nx = ct * xs - st * ys
    ny = st * xs + ct * ys
    kv[:, :, 5:20:3] = nx
    kv[:, :, 6:20:3] = ny
    return q.astype(np.float32), kv.astype(np.float32), mask.astype(np.float32)


class SourceNextoPolicy:
    """Pinned public CPU TorchScript policy with exact source cadence/kickoff."""

    def __init__(self, worlds: int, model_path: str | Path):
        model_path = Path(model_path)
        if hashlib.sha256(model_path.read_bytes()).hexdigest().upper() != NEXTO_MODEL_SHA256:
            raise RuntimeError("pinned Nexto model SHA-256 mismatch")
        self.worlds = int(worlds)
        self.actor = torch.jit.load(str(model_path), map_location="cpu").eval()
        torch.set_num_threads(1)
        self.table = _nexto_action_table()
        self.kickoff_sequence = _nexto_kickoff_sequence()
        self.previous_action = np.zeros((worlds, 8), dtype=np.float32)
        self.kickoff_index = np.full(worlds, -1, dtype=np.int32)
        self.cadence_tick = 0
        self.inference_calls = 0

    def reset_policy_memory(self, rows: np.ndarray | Sequence[int]) -> None:
        index = np.asarray(rows)
        self.previous_action[index] = 0
        self.kickoff_index[index] = 0

    def tick(
        self,
        state: RocketSimBatchState,
        side: np.ndarray,
        kickoff_active: np.ndarray,
        active: np.ndarray | None = None,
    ) -> np.ndarray:
        if active is None:
            active = np.ones(self.worlds, dtype=bool)
        active = np.asarray(active, dtype=bool)
        if self.cadence_tick == 0:
            observation = build_nexto_source_observation(state, side, self.previous_action)
            rows = np.flatnonzero(active)
            if rows.size:
                with torch.inference_mode():
                    logits, _weights = self.actor(
                        tuple(torch.from_numpy(item[rows]) for item in observation)
                    )
                indices = torch.argmax(logits, dim=-1).cpu().numpy().reshape(-1)
                self.previous_action[rows] = self.table[indices]
                self.inference_calls += 1
        inactive = active & ~kickoff_active
        self.kickoff_index[inactive] = -1
        newly_active = active & kickoff_active & (self.kickoff_index < 0)
        self.kickoff_index[newly_active] = 0
        in_sequence = (
            active
            & kickoff_active
            & (self.kickoff_index >= 0)
            & (self.kickoff_index < len(self.kickoff_sequence))
            & (state.ball_pos[:, 1] == 0.0)
        )
        if np.any(in_sequence):
            self.previous_action[in_sequence] = self.kickoff_sequence[self.kickoff_index[in_sequence]]
        self.kickoff_index[active & kickoff_active] += 1
        self.cadence_tick = (self.cadence_tick + 1) % NEXTO_CADENCE_TICKS
        return self.previous_action


def mirror_vec(value: Any, rs: Any) -> Any:
    return rs.Vec(-float(value.x), -float(value.y), float(value.z))


def mirror_rot_mat(value: Any, rs: Any) -> Any:
    return rs.RotMat(
        mirror_vec(value.forward, rs),
        mirror_vec(value.right, rs),
        mirror_vec(value.up, rs),
    )


def mirror_car_state(state: Any, rs: Any) -> Any:
    result = copy.deepcopy(state)
    result.pos = mirror_vec(state.pos, rs)
    result.vel = mirror_vec(state.vel, rs)
    result.ang_vel = mirror_vec(state.ang_vel, rs)
    result.rot_mat = mirror_rot_mat(state.rot_mat, rs)
    return result


def mirror_ball_state(state: Any, rs: Any) -> Any:
    result = copy.deepcopy(state)
    result.pos = mirror_vec(state.pos, rs)
    result.vel = mirror_vec(state.vel, rs)
    result.ang_vel = mirror_vec(state.ang_vel, rs)
    result.rot_mat = mirror_rot_mat(state.rot_mat, rs)
    return result


__all__ = [
    "FrozenRivalPolicy", "NEXTO_CADENCE_TICKS", "PHYSICS_HZ",
    "RIVAL_CADENCE_TICKS", "RocketSimBatchState", "RocketSimRivalMemory",
    "SourceNextoPolicy", "actions_to_controls", "build_nexto_source_observation",
    "build_rival2_observation", "canonical_boost_pads", "car_hit_ticks",
    "mirror_ball_state", "mirror_car_state", "read_rocketsim_batch",
    "update_rival_memory_after_tick",
]
