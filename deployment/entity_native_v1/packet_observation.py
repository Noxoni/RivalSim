"""Vendored packet observation builder; exact values are NOT claimed for proxies.

Copied from RLBot-Rival bot/rival2_live/runtime.py (source hash in manifest).
Only the observation adapter is included, not its obsolete policy loader.
Wheel/sticky proxies and packet-timer qualifications are recorded per field.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


ACTIVE_PHASES = {"Kickoff", "Active"}
RESET_PHASES = {"Countdown"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _phase_name(packet: Any) -> str:
    return str(packet.match_info.match_phase).split(".")[-1]


def _vec3(value: Any) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float32)


def _rotation_basis(rotator: Any) -> tuple[np.ndarray, np.ndarray]:
    pitch = float(rotator.pitch)
    yaw = float(rotator.yaw)
    roll = float(rotator.roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = np.asarray((cp * cy, cp * sy, sp), dtype=np.float32)
    up = np.asarray(
        (-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr),
        dtype=np.float32,
    )
    return forward, up


def _controller_row(value: Any) -> np.ndarray:
    return np.asarray(
        (
            float(value.throttle),
            float(value.steer),
            float(value.pitch),
            float(value.yaw),
            float(value.roll),
            float(bool(value.jump)),
            float(bool(value.boost)),
            float(bool(value.handbrake)),
        ),
        dtype=np.float32,
    )


def _timer(value: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(value.astype(np.float32) / np.float32(scale), 0.0, 1.0)[..., None]


@dataclass(slots=True)
class LiveBatchState:
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


@dataclass(slots=True)
class LiveMemory:
    episode_ticks: np.ndarray
    no_touch_ticks: np.ndarray
    kickoff_indicator: np.ndarray
    touch_event: np.ndarray
    demoed_event: np.ndarray
    previous_action: np.ndarray
    time_since_boosted: np.ndarray
    sticky_ticks: np.ndarray
    jump_time: np.ndarray
    air_time: np.ndarray
    air_time_since_jump: np.ndarray
    boosting_time: np.ndarray
    supersonic_time: np.ndarray
    previous_on_ground: np.ndarray
    previous_is_jumping: np.ndarray
    previous_demoed: np.ndarray
    latest_touch_seconds: np.ndarray

    @classmethod
    def create(cls) -> "LiveMemory":
        return cls(
            episode_ticks=np.zeros(1, dtype=np.int32),
            no_touch_ticks=np.zeros(1, dtype=np.int32),
            kickoff_indicator=np.ones(1, dtype=np.int32),
            touch_event=np.zeros((1, 2), dtype=np.int32),
            demoed_event=np.zeros((1, 2), dtype=np.int32),
            previous_action=np.zeros((1, 2, 8), dtype=np.float32),
            time_since_boosted=np.zeros((1, 2), dtype=np.float32),
            sticky_ticks=np.zeros((1, 2), dtype=np.int32),
            jump_time=np.zeros((1, 2), dtype=np.float32),
            air_time=np.zeros((1, 2), dtype=np.float32),
            air_time_since_jump=np.zeros((1, 2), dtype=np.float32),
            boosting_time=np.zeros((1, 2), dtype=np.float32),
            supersonic_time=np.zeros((1, 2), dtype=np.float32),
            previous_on_ground=np.zeros((1, 2), dtype=np.int32),
            previous_is_jumping=np.zeros((1, 2), dtype=np.int32),
            previous_demoed=np.zeros((1, 2), dtype=bool),
            latest_touch_seconds=np.full((1, 2), -np.inf, dtype=np.float64),
        )

    def clear_interval_events(self) -> None:
        self.touch_event.fill(0)
        self.demoed_event.fill(0)
        self.kickoff_indicator.fill(0)


class Rival2LiveAdapter:
    """Build 182 fields from packets; unavailable wheel/sticky values use documented proxies."""

    def __init__(self, manifest: dict[str, Any], field_info: Any):
        self.manifest = manifest
        self.spec = manifest["observation"]
        self.memory = LiveMemory.create()
        expected = np.asarray(
            self.spec["canonical_boost_pad_positions"], dtype=np.float32
        )
        static = list(field_info.boost_pads)
        if expected.shape != (34, 3) or len(static) != 34:
            raise RuntimeError("Rival 2 requires the standard 34-pad Soccar field")
        actual = np.stack([_vec3(pad.location) for pad in static])
        mapping = []
        for position in expected:
            # RivalSim records the boost pickup trigger height, while RLBot's
            # live FieldInfo reports the pad's rendered floor elevation. The
            # horizontal centers are the authoritative stable pad identity.
            distances = np.linalg.norm(actual[:, :2] - position[:2], axis=1)
            candidates = np.flatnonzero(distances <= 4.0)
            if candidates.size != 1:
                nearest = int(np.argmin(distances))
                raise RuntimeError(
                    "failed to map canonical boost pad by horizontal center at "
                    f"{position.tolist()}; nearest live pad={actual[nearest].tolist()} "
                    f"xy_distance={float(distances[nearest])}"
                )
            mapping.append(int(candidates[0]))
        if len(set(mapping)) != 34:
            raise RuntimeError("canonical boost-pad mapping is not one-to-one")
        self.pad_mapping = np.asarray(mapping, dtype=np.int64)
        self.pad_durations = np.asarray(
            self.spec["canonical_boost_pad_durations"], dtype=np.float32
        )
        self.player_indices = np.zeros(2, dtype=np.int64)

    @staticmethod
    def _team_indices(packet: Any) -> np.ndarray:
        indices = []
        for team in (0, 1):
            candidates = [i for i, player in enumerate(packet.players) if player.team == team]
            if len(candidates) != 1:
                raise RuntimeError(
                    "Rival 2 live deployment supports standard 1v1 only: "
                    f"team {team} has {len(candidates)} cars"
                )
            indices.append(candidates[0])
        return np.asarray(indices, dtype=np.int64)

    def reset(self, packet: Any) -> None:
        self.memory = LiveMemory.create()
        self.player_indices = self._team_indices(packet)
        for team, index in enumerate(self.player_indices):
            player = packet.players[int(index)]
            on_ground = int(str(player.air_state).split(".")[-1] == "OnGround")
            jumping = int(str(player.air_state).split(".")[-1] == "Jumping")
            demoed = bool(float(player.demolished_timeout) > 0.0)
            self.memory.previous_on_ground[0, team] = on_ground
            self.memory.previous_is_jumping[0, team] = jumping
            self.memory.previous_demoed[0, team] = demoed
            touch = player.latest_touch
            if touch is not None:
                self.memory.latest_touch_seconds[0, team] = float(touch.game_seconds)

    def advance(self, packet: Any, delta_ticks: int) -> None:
        delta_ticks = max(1, int(delta_ticks))
        dt = np.float32(delta_ticks / 120.0)
        self.memory.episode_ticks[0] += delta_ticks
        touch_seen = False
        for team, index in enumerate(self.player_indices):
            player = packet.players[int(index)]
            phase = str(player.air_state).split(".")[-1]
            on_ground = int(phase == "OnGround")
            is_jumping = int(phase == "Jumping")
            is_demoed = bool(float(player.demolished_timeout) > 0.0)
            last_input = player.last_input
            physically_boosting = bool(last_input.boost) and float(player.boost) > 0.0

            touch = player.latest_touch
            if touch is not None:
                seconds = float(touch.game_seconds)
                if seconds > self.memory.latest_touch_seconds[0, team] + 1e-6:
                    self.memory.touch_event[0, team] += 1
                    touch_seen = True
                    self.memory.latest_touch_seconds[0, team] = seconds
            if is_demoed and not self.memory.previous_demoed[0, team]:
                self.memory.demoed_event[0, team] = 1

            if on_ground:
                self.memory.air_time[0, team] = 0.0
                self.memory.air_time_since_jump[0, team] = 0.0
            else:
                self.memory.air_time[0, team] += dt
                if bool(player.has_jumped) or self.memory.air_time_since_jump[0, team] > 0:
                    self.memory.air_time_since_jump[0, team] += dt
            if is_jumping:
                self.memory.jump_time[0, team] = min(
                    np.float32(0.2), self.memory.jump_time[0, team] + dt
                )
            else:
                self.memory.jump_time[0, team] = 0.0
            if physically_boosting:
                self.memory.boosting_time[0, team] += dt
                self.memory.time_since_boosted[0, team] = 0.0
            else:
                self.memory.boosting_time[0, team] = 0.0
                self.memory.time_since_boosted[0, team] += dt
            if bool(player.is_supersonic):
                self.memory.supersonic_time[0, team] += dt
            else:
                self.memory.supersonic_time[0, team] = 0.0

            started_jump = (
                self.memory.previous_is_jumping[0, team] == 0
                and is_jumping != 0
                and self.memory.previous_on_ground[0, team] != 0
                and bool(last_input.jump)
            )
            self.memory.sticky_ticks[0, team] = (
                2
                if started_jump
                else max(0, int(self.memory.sticky_ticks[0, team]) - delta_ticks)
            )
            self.memory.previous_on_ground[0, team] = on_ground
            self.memory.previous_is_jumping[0, team] = is_jumping
            self.memory.previous_demoed[0, team] = is_demoed
        if touch_seen:
            latest = max(
                (
                    float(packet.players[int(index)].latest_touch.game_seconds)
                    for index in self.player_indices
                    if packet.players[int(index)].latest_touch is not None
                ),
                default=float(packet.match_info.seconds_elapsed),
            )
            elapsed = max(0.0, float(packet.match_info.seconds_elapsed) - latest)
            self.memory.no_touch_ticks[0] = int(round(elapsed * 120.0))
        else:
            self.memory.no_touch_ticks[0] += delta_ticks

    def _state(self, packet: Any) -> LiveBatchState:
        self.player_indices = self._team_indices(packet)
        ball = packet.balls[0].physics
        ball_pos = _vec3(ball.location)[None, :]
        ball_vel = _vec3(ball.velocity)[None, :]
        ball_ang_vel = _vec3(ball.angular_velocity)[None, :]
        car_pos = np.empty((1, 2, 3), dtype=np.float32)
        car_vel = np.empty_like(car_pos)
        car_forward = np.empty_like(car_pos)
        car_up = np.empty_like(car_pos)
        car_ang_vel = np.empty_like(car_pos)
        float_fields = {
            name: np.empty((1, 2), dtype=np.float32)
            for name in ("boost", "flip_time", "demo_respawn_timer")
        }
        int_fields = {
            name: np.empty((1, 2), dtype=np.int32)
            for name in (
                "on_ground",
                "has_jumped",
                "is_jumping",
                "has_double_jumped",
                "has_flipped",
                "is_flipping",
                "is_supersonic",
                "is_demoed",
            )
        }
        wheels = np.empty((1, 2, 4), dtype=np.int32)
        for team, index in enumerate(self.player_indices):
            player = packet.players[int(index)]
            physics = player.physics
            phase = str(player.air_state).split(".")[-1]
            on_ground = int(phase == "OnGround")
            car_pos[0, team] = _vec3(physics.location)
            car_vel[0, team] = _vec3(physics.velocity)
            car_forward[0, team], car_up[0, team] = _rotation_basis(physics.rotation)
            car_ang_vel[0, team] = _vec3(physics.angular_velocity)
            float_fields["boost"][0, team] = float(player.boost)
            float_fields["flip_time"][0, team] = (
                max(0.0, float(player.dodge_elapsed))
                if bool(player.has_dodged) and not on_ground
                else 0.0
            )
            float_fields["demo_respawn_timer"][0, team] = max(
                0.0, float(player.demolished_timeout)
            )
            int_fields["on_ground"][0, team] = on_ground
            int_fields["has_jumped"][0, team] = int(bool(player.has_jumped))
            int_fields["is_jumping"][0, team] = int(phase == "Jumping")
            int_fields["has_double_jumped"][0, team] = int(
                bool(player.has_double_jumped)
            )
            int_fields["has_flipped"][0, team] = int(bool(player.has_dodged))
            int_fields["is_flipping"][0, team] = int(phase == "Dodging")
            int_fields["is_supersonic"][0, team] = int(bool(player.is_supersonic))
            int_fields["is_demoed"][0, team] = int(
                float(player.demolished_timeout) > 0.0
            )
            wheels[0, team].fill(on_ground)

        pad_active = np.empty((1, 34), dtype=np.int32)
        pad_cooldown = np.empty((1, 34), dtype=np.float32)
        for canonical, packet_index in enumerate(self.pad_mapping):
            pad = packet.boost_pads[int(packet_index)]
            active = bool(pad.is_active)
            pad_active[0, canonical] = int(active)
            pad_cooldown[0, canonical] = (
                0.0
                if active
                else max(0.0, float(self.pad_durations[canonical]) - float(pad.timer))
            )
        return LiveBatchState(
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            ball_ang_vel=ball_ang_vel,
            car_pos=car_pos,
            car_vel=car_vel,
            car_forward=car_forward,
            car_up=car_up,
            car_ang_vel=car_ang_vel,
            wheels=wheels,
            jump_time=self.memory.jump_time.copy(),
            air_time=self.memory.air_time.copy(),
            air_time_since_jump=self.memory.air_time_since_jump.copy(),
            boosting_time=self.memory.boosting_time.copy(),
            supersonic_time=self.memory.supersonic_time.copy(),
            pad_cooldown=pad_cooldown,
            pad_active=pad_active,
            **float_fields,
            **int_fields,
        )

    def observation(self, packet: Any) -> np.ndarray:
        state = self._state(packet)
        spec = self.spec
        memory = self.memory
        position_scale = np.asarray(spec["position_scale"], dtype=np.float32)
        signs = np.asarray(((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0)), dtype=np.float32)
        pad_maps = (
            np.arange(34, dtype=np.int64),
            np.asarray(spec["orange_pad_remap"], dtype=np.int64),
        )

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
                    | (
                        (has_jumped != 0)
                        & (state.air_time_since_jump[:, car] < spec["air_time_scale"])
                    )
                )
            ).astype(np.float32)
            return np.concatenate(
                (
                    state.car_pos[:, car] * sign / position_scale,
                    state.car_vel[:, car]
                    * sign
                    / np.float32(spec["car_linear_speed_scale"]),
                    state.car_forward[:, car] * sign,
                    state.car_up[:, car] * sign,
                    state.car_ang_vel[:, car]
                    * sign
                    / np.float32(spec["angular_speed_scale"]),
                    (state.boost[:, car] / np.float32(spec["boost_scale"]))[:, None],
                    on_ground[:, None],
                    has_jumped[:, None],
                    state.is_jumping[:, car, None].astype(np.float32),
                    has_double[:, None],
                    has_flipped[:, None],
                    state.is_flipping[:, car, None].astype(np.float32),
                    (has_jumped == 0).astype(np.float32)[:, None],
                    dodge[:, None],
                    state.is_demoed[:, car, None].astype(np.float32),
                    _timer(state.demo_respawn_timer[:, car], spec["demo_timer_scale"]),
                    state.wheels[:, car].astype(np.float32),
                    _timer(state.jump_time[:, car], spec["jump_time_scale"]),
                    _timer(state.air_time[:, car], spec["air_time_scale"]),
                    _timer(state.air_time_since_jump[:, car], spec["air_time_scale"]),
                    _timer(state.flip_time[:, car], spec["flip_time_scale"]),
                    _timer(state.boosting_time[:, car], spec["boosting_time_scale"]),
                    _timer(
                        memory.time_since_boosted[:, car],
                        spec["time_since_boosted_scale"],
                    ),
                    state.is_supersonic[:, car, None].astype(np.float32),
                    _timer(state.supersonic_time[:, car], spec["supersonic_time_scale"]),
                    _timer(memory.sticky_ticks[:, car], spec["sticky_tick_scale"]),
                ),
                axis=1,
            )

        observations = []
        for agent in range(2):
            opponent = 1 - agent
            sign = signs[agent]
            pad_index = pad_maps[agent]
            cooldown = state.pad_cooldown[:, pad_index]
            duration = self.pad_durations[pad_index]
            pads = np.stack(
                (
                    (cooldown == 0).astype(np.float32),
                    np.clip(cooldown / duration, 0.0, 1.0),
                ),
                axis=-1,
            ).reshape(1, 68)
            ball = np.concatenate(
                (
                    state.ball_pos * sign / position_scale,
                    state.ball_vel
                    * sign
                    / np.float32(spec["ball_linear_speed_scale"]),
                    state.ball_ang_vel
                    * sign
                    / np.float32(spec["angular_speed_scale"]),
                ),
                axis=1,
            )
            relative = np.concatenate(
                (
                    (state.ball_pos - state.car_pos[:, agent])
                    * sign
                    / position_scale,
                    (state.ball_vel - state.car_vel[:, agent])
                    * sign
                    / np.float32(spec["ball_linear_speed_scale"]),
                    (state.car_pos[:, opponent] - state.car_pos[:, agent])
                    * sign
                    / position_scale,
                    (state.car_vel[:, opponent] - state.car_vel[:, agent])
                    * sign
                    / np.float32(spec["car_linear_speed_scale"]),
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
                    _timer(memory.episode_ticks, spec["episode_age_scale_ticks"]),
                    _timer(memory.no_touch_ticks, spec["no_touch_age_scale_ticks"]),
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
            if observation.shape != (1, int(spec["dimension"])):
                raise RuntimeError(f"Rival 2 observation shape mismatch: {observation.shape}")
            observations.append(observation)
        return np.stack(observations, axis=1)
