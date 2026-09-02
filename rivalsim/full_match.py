"""Reusable device-resident 120 Hz full-match runtime for fixed policies.

This module is deliberately separate from Rival 2.0's frozen training episode
semantics.  It reuses the accepted kickoff reset kernel but owns regulation,
overtime, match completion, and match telemetry without changing rewards,
observations, PPO, policy architecture, or simulator physics.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from rivalsim.ball_world_state import MAX_BALL_CONTACTS
from rivalsim.behavioral_telemetry import (
    GOAL_SCORING_PLANE_Y_UU,
    SURFACE_BACKBOARD,
    SURFACE_SIDE_WALL,
    _surface_category,
)
from rivalsim.rival2_contracts import (
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    RIVAL2_REWARD_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    sample_hybrid_action,
)
from third_party.nexto.adapter import NextoPolicyAdapter, NextoStateTensors

PHYSICS_HZ = 120
REGULATION_TICKS = 5 * 60 * PHYSICS_HZ
RIVAL_CADENCE_TICKS = 4
NEXTO_CADENCE_TICKS = 8
GOAL_CAPACITY_PER_WORLD = 128
DIRECTION_THRESHOLD_UU_PER_SECOND = 100.0

TOUCH_BACKWARD = 0
TOUCH_NEUTRAL = 1
TOUCH_FORWARD = 2


@wp.func
def _direction_category(value: float) -> int:
    result = TOUCH_NEUTRAL
    if value >= DIRECTION_THRESHOLD_UU_PER_SECOND:
        result = TOUCH_FORWARD
    elif value <= -DIRECTION_THRESHOLD_UU_PER_SECOND:
        result = TOUCH_BACKWARD
    return result


@wp.func
def _finalize_possession(
    env: int,
    position: wp.vec3,
    last_toucher: wp.array(dtype=wp.int32),
    touch_start_position: wp.array(dtype=wp.vec3),
    max_forward_y: wp.array(dtype=wp.float32),
    max_backward_y: wp.array(dtype=wp.float32),
    active_surface_bits: wp.array(dtype=wp.int32),
    displacement_count: wp.array(dtype=wp.int32),
    wall_continuation_count: wp.array(dtype=wp.int32),
    backboard_continuation_count: wp.array(dtype=wp.int32),
):
    toucher = last_toucher[env]
    if toucher >= 0:
        sign = 1.0
        if toucher == 1:
            sign = -1.0
        net_y = sign * (position[1] - touch_start_position[env][1])
        category = _direction_category(net_y)
        car = env * 2 + toucher
        displacement_count[car * 3 + category] = (
            displacement_count[car * 3 + category] + 1
        )
        bits = active_surface_bits[env]
        side_wall_bit = wp.int32(1 << (SURFACE_SIDE_WALL - 1))
        backboard_bit = wp.int32(1 << (SURFACE_BACKBOARD - 1))
        if (bits & side_wall_bit) != 0:
            wall_continuation_count[car] = wall_continuation_count[car] + 1
        if (bits & backboard_bit) != 0:
            backboard_continuation_count[car] = backboard_continuation_count[car] + 1
        max_forward_y[env] = wp.max(max_forward_y[env], net_y)
        max_backward_y[env] = wp.max(max_backward_y[env], -net_y)


@wp.kernel(enable_backward=False)
def collect_full_match_tick(
    goal_capacity: int,
    ball_position: wp.array(dtype=wp.vec3),
    ball_velocity: wp.array(dtype=wp.vec3),
    car_a_hit_this_tick: wp.array(dtype=wp.int32),
    car_b_hit_this_tick: wp.array(dtype=wp.int32),
    car_a_pre_ball_position_bt: wp.array(dtype=wp.vec3),
    car_b_pre_ball_position_bt: wp.array(dtype=wp.vec3),
    car_a_pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    car_b_pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    pre_tick_first_car: wp.array(dtype=wp.int32),
    ball_contact_count: wp.array(dtype=wp.int32),
    ball_contact_normal: wp.array(dtype=wp.vec3),
    goal_scored: wp.array(dtype=wp.int32),
    scoring_team: wp.array(dtype=wp.int32),
    bump_event_count: wp.array(dtype=wp.int32),
    bump_event_bumper: wp.array(dtype=wp.int32),
    bump_event_is_demo: wp.array(dtype=wp.int32),
    regulation_ticks_remaining: wp.array(dtype=wp.int32),
    blue_score: wp.array(dtype=wp.int32),
    orange_score: wp.array(dtype=wp.int32),
    overtime: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
    goal_count: wp.array(dtype=wp.int32),
    total_ticks: wp.array(dtype=wp.int32),
    overtime_ticks: wp.array(dtype=wp.int32),
    pending_reset: wp.array(dtype=wp.int32),
    kickoff_active: wp.array(dtype=wp.int32),
    kickoff_touch_count: wp.array(dtype=wp.int32),
    touch_contact_latched: wp.array(dtype=wp.int32),
    last_toucher: wp.array(dtype=wp.int32),
    previous_ball_position: wp.array(dtype=wp.vec3),
    touch_start_position: wp.array(dtype=wp.vec3),
    max_forward_y: wp.array(dtype=wp.float32),
    max_backward_y: wp.array(dtype=wp.float32),
    active_surface_bits: wp.array(dtype=wp.int32),
    touch_count: wp.array(dtype=wp.int32),
    kickoff_first_touch_count: wp.array(dtype=wp.int32),
    kickoff_goal_count: wp.array(dtype=wp.int32),
    possession_total: wp.array(dtype=wp.int32),
    possession_same: wp.array(dtype=wp.int32),
    possession_opponent: wp.array(dtype=wp.int32),
    direction_count: wp.array(dtype=wp.int32),
    displacement_count: wp.array(dtype=wp.int32),
    wall_continuation_count: wp.array(dtype=wp.int32),
    backboard_continuation_count: wp.array(dtype=wp.int32),
    demo_count: wp.array(dtype=wp.int32),
    goal_overflow: wp.array(dtype=wp.int32),
    goal_scorer: wp.array(dtype=wp.int32),
    goal_tick: wp.array(dtype=wp.int32),
    goal_overtime: wp.array(dtype=wp.int32),
    goal_kickoff: wp.array(dtype=wp.int32),
    goal_entry_valid: wp.array(dtype=wp.int32),
    goal_entry_x: wp.array(dtype=wp.float32),
    goal_entry_z: wp.array(dtype=wp.float32),
):
    env = wp.tid()
    if done[env] != 0:
        return

    position_after = ball_position[env]
    velocity_after = ball_velocity[env]
    car_base = env * 2
    reports_a = wp.int32(car_a_hit_this_tick[env] != 0)
    reports_b = wp.int32(car_b_hit_this_tick[env] != 0)
    touched_a = wp.int32(
        reports_a != 0 and touch_contact_latched[car_base] == 0
    )
    touched_b = wp.int32(
        reports_b != 0 and touch_contact_latched[car_base + 1] == 0
    )
    touch_contact_latched[car_base] = reports_a
    touch_contact_latched[car_base + 1] = reports_b

    first_toucher = pre_tick_first_car[env]
    for ordinal in range(2):
        local_toucher = first_toucher
        if ordinal == 1:
            local_toucher = 1 - first_toucher
        accepted = touched_a
        position_before = car_a_pre_ball_position_bt[env] * 50.0
        velocity_before = car_a_pre_ball_velocity_bt[env] * 50.0
        if local_toucher == 1:
            accepted = touched_b
            position_before = car_b_pre_ball_position_bt[env] * 50.0
            velocity_before = car_b_pre_ball_velocity_bt[env] * 50.0
        if accepted != 0:
            previous_toucher = last_toucher[env]
            if previous_toucher >= 0:
                _finalize_possession(
                    env,
                    position_before,
                    last_toucher,
                    touch_start_position,
                    max_forward_y,
                    max_backward_y,
                    active_surface_bits,
                    displacement_count,
                    wall_continuation_count,
                    backboard_continuation_count,
                )
                previous_car = car_base + previous_toucher
                possession_total[previous_car] = possession_total[previous_car] + 1
                if previous_toucher == local_toucher:
                    possession_same[previous_car] = possession_same[previous_car] + 1
                else:
                    possession_opponent[previous_car] = (
                        possession_opponent[previous_car] + 1
                    )

            car = car_base + local_toucher
            touch_count[car] = touch_count[car] + 1
            if kickoff_touch_count[env] == 0:
                kickoff_first_touch_count[car] = kickoff_first_touch_count[car] + 1
            kickoff_touch_count[env] = kickoff_touch_count[env] + 1
            kickoff_active[env] = 0
            sign = 1.0
            if local_toucher == 1:
                sign = -1.0
            longitudinal_delta = sign * (
                velocity_after[1] - velocity_before[1]
            )
            category = _direction_category(longitudinal_delta)
            direction_count[car * 3 + category] = (
                direction_count[car * 3 + category] + 1
            )
            last_toucher[env] = local_toucher
            touch_start_position[env] = position_after
            max_forward_y[env] = 0.0
            max_backward_y[env] = 0.0
            active_surface_bits[env] = 0

    current_toucher = last_toucher[env]
    if current_toucher >= 0:
        sign = 1.0
        if current_toucher == 1:
            sign = -1.0
        displacement = sign * (
            position_after[1] - touch_start_position[env][1]
        )
        max_forward_y[env] = wp.max(max_forward_y[env], displacement)
        max_backward_y[env] = wp.max(max_backward_y[env], -displacement)
        contacts = ball_contact_count[env]
        contact_base = env * MAX_BALL_CONTACTS
        for relative in range(MAX_BALL_CONTACTS):
            if relative < contacts:
                surface = _surface_category(
                    ball_contact_normal[contact_base + relative]
                )
                active_surface_bits[env] = (
                    active_surface_bits[env]
                    | wp.int32(1 << (surface - 1))
                )

    event_base = env * 4
    bump_count = bump_event_count[env]
    for relative in range(4):
        if relative < bump_count:
            event = event_base + relative
            if bump_event_is_demo[event] != 0:
                bumper = bump_event_bumper[event]
                if bumper >= 0 and bumper < 2:
                    demo_count[car_base + bumper] = demo_count[car_base + bumper] + 1

    scored = goal_scored[env] != 0
    if scored:
        _finalize_possession(
            env,
            position_after,
            last_toucher,
            touch_start_position,
            max_forward_y,
            max_backward_y,
            active_surface_bits,
            displacement_count,
            wall_continuation_count,
            backboard_continuation_count,
        )
        scorer = scoring_team[env]
        if scorer == 0:
            blue_score[env] = blue_score[env] + 1
        else:
            orange_score[env] = orange_score[env] + 1
        if kickoff_touch_count[env] <= 1:
            kickoff_goal_count[car_base + scorer] = kickoff_goal_count[car_base + scorer] + 1

        slot = goal_count[env]
        goal_count[env] = slot + 1
        if slot >= goal_capacity:
            goal_overflow[env] = 1
        else:
            goal_event = env * goal_capacity + slot
            goal_scorer[goal_event] = scorer
            goal_tick[goal_event] = total_ticks[env]
            goal_overtime[goal_event] = overtime[env]
            goal_kickoff[goal_event] = wp.int32(kickoff_touch_count[env] <= 1)
            goal_entry_valid[goal_event] = 0
            scoring_sign = 1.0
            if scorer == 1:
                scoring_sign = -1.0
            scoring_plane = scoring_sign * GOAL_SCORING_PLANE_Y_UU
            before = previous_ball_position[env]
            delta_y = position_after[1] - before[1]
            if wp.abs(delta_y) > 0.000001:
                fraction = (scoring_plane - before[1]) / delta_y
                if fraction >= 0.0 and fraction <= 1.0:
                    crossing = before + (position_after - before) * fraction
                    goal_entry_valid[goal_event] = 1
                    goal_entry_x[goal_event] = scoring_sign * crossing[0]
                    goal_entry_z[goal_event] = crossing[2]
        last_toucher[env] = -1
        active_surface_bits[env] = 0
        if overtime[env] != 0:
            done[env] = 1
            winner[env] = scorer
        else:
            pending_reset[env] = 1

    if kickoff_active[env] != 0 and position_after[1] != 0.0:
        kickoff_active[env] = 0

    total_ticks[env] = total_ticks[env] + 1
    if overtime[env] != 0:
        overtime_ticks[env] = overtime_ticks[env] + 1
    else:
        remaining = regulation_ticks_remaining[env] - 1
        regulation_ticks_remaining[env] = remaining
        if remaining == 0:
            if blue_score[env] == orange_score[env]:
                overtime[env] = 1
                pending_reset[env] = 1
            else:
                done[env] = 1
                winner[env] = wp.int32(blue_score[env] < orange_score[env])
                pending_reset[env] = 0

    previous_ball_position[env] = position_after


@wp.kernel(enable_backward=False)
def after_full_match_reset(
    reset_mask: wp.array(dtype=wp.int32),
    ball_position: wp.array(dtype=wp.vec3),
    pending_reset: wp.array(dtype=wp.int32),
    kickoff_active: wp.array(dtype=wp.int32),
    kickoff_touch_count: wp.array(dtype=wp.int32),
    touch_contact_latched: wp.array(dtype=wp.int32),
    last_toucher: wp.array(dtype=wp.int32),
    previous_ball_position: wp.array(dtype=wp.vec3),
    active_surface_bits: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    if reset_mask[env] != 0:
        pending_reset[env] = 0
        kickoff_active[env] = 1
        kickoff_touch_count[env] = 0
        touch_contact_latched[env * 2] = 0
        touch_contact_latched[env * 2 + 1] = 0
        last_toucher[env] = -1
        previous_ball_position[env] = ball_position[env]
        active_surface_bits[env] = 0


class FullMatchState:
    """Device-resident regulation/overtime state and immutable assignments."""

    def __init__(
        self,
        num_worlds: int,
        device: str,
        starting_layout: np.ndarray,
        rival_side: np.ndarray,
    ):
        self.num_worlds = int(num_worlds)
        self.device = device
        self.starting_layout = wp.array(starting_layout, dtype=wp.int32, device=device)
        self.rival_side = wp.array(rival_side, dtype=wp.int32, device=device)
        self.regulation_ticks_remaining = wp.full(
            num_worlds, REGULATION_TICKS, dtype=wp.int32, device=device
        )
        for name in (
            "blue_score",
            "orange_score",
            "overtime",
            "done",
            "goal_count",
            "total_ticks",
            "overtime_ticks",
            "pending_reset",
            "kickoff_touch_count",
            "rival_scheduler_tick",
            "nexto_scheduler_tick",
        ):
            setattr(self, name, wp.zeros(num_worlds, dtype=wp.int32, device=device))
        self.winner = wp.full(num_worlds, -1, dtype=wp.int32, device=device)
        self.kickoff_active = wp.ones(num_worlds, dtype=wp.int32, device=device)

    def torch_views(self) -> dict[str, torch.Tensor]:
        return {
            name: wp.to_torch(getattr(self, name))
            for name in (
                "starting_layout",
                "rival_side",
                "regulation_ticks_remaining",
                "blue_score",
                "orange_score",
                "overtime",
                "done",
                "winner",
                "goal_count",
                "total_ticks",
                "overtime_ticks",
                "pending_reset",
                "kickoff_active",
                "kickoff_touch_count",
                "rival_scheduler_tick",
                "nexto_scheduler_tick",
            )
        }


class FullMatchTelemetry:
    """Bounded device-resident per-world/per-side match telemetry."""

    _CAR_FIELDS = (
        "touch_count",
        "kickoff_first_touch_count",
        "kickoff_goal_count",
        "possession_total",
        "possession_same",
        "possession_opponent",
        "wall_continuation_count",
        "backboard_continuation_count",
        "demo_count",
    )
    _PRE_DIRECTION_CAR_FIELDS = (
        "touch_count",
        "kickoff_first_touch_count",
        "kickoff_goal_count",
        "possession_total",
        "possession_same",
        "possession_opponent",
    )
    _GOAL_INT_FIELDS = (
        "goal_scorer",
        "goal_tick",
        "goal_overtime",
        "goal_kickoff",
        "goal_entry_valid",
    )

    def __init__(
        self,
        state: FullMatchState,
        *,
        goal_capacity: int = GOAL_CAPACITY_PER_WORLD,
    ):
        self.state = state
        self.num_worlds = state.num_worlds
        self.device = state.device
        self.goal_capacity = int(goal_capacity)
        car_count = self.num_worlds * 2
        self.touch_contact_latched = wp.zeros(car_count, dtype=wp.int32, device=self.device)
        self.last_toucher = wp.full(
            self.num_worlds, -1, dtype=wp.int32, device=self.device
        )
        self.previous_ball_position = wp.zeros(
            self.num_worlds, dtype=wp.vec3, device=self.device
        )
        self.touch_start_position = wp.zeros(
            self.num_worlds, dtype=wp.vec3, device=self.device
        )
        self.max_forward_y = wp.zeros(
            self.num_worlds, dtype=wp.float32, device=self.device
        )
        self.max_backward_y = wp.zeros(
            self.num_worlds, dtype=wp.float32, device=self.device
        )
        self.active_surface_bits = wp.zeros(
            self.num_worlds, dtype=wp.int32, device=self.device
        )
        for name in self._CAR_FIELDS:
            setattr(self, name, wp.zeros(car_count, dtype=wp.int32, device=self.device))
        self.direction_count = wp.zeros(car_count * 3, dtype=wp.int32, device=self.device)
        self.displacement_count = wp.zeros(car_count * 3, dtype=wp.int32, device=self.device)
        self.goal_overflow = wp.zeros(
            self.num_worlds, dtype=wp.int32, device=self.device
        )
        capacity = self.num_worlds * self.goal_capacity
        for name in self._GOAL_INT_FIELDS:
            initial = -1 if name in {"goal_scorer", "goal_tick"} else 0
            setattr(
                self,
                name,
                wp.full(capacity, initial, dtype=wp.int32, device=self.device),
            )
        self.goal_entry_x = wp.zeros(capacity, dtype=wp.float32, device=self.device)
        self.goal_entry_z = wp.zeros(capacity, dtype=wp.float32, device=self.device)
        self._original_launch: Any | None = None

    def attach(self, world: Rival2WorldSim) -> None:
        if self._original_launch is not None:
            raise RuntimeError("full-match telemetry is already attached")
        wp.copy(self.previous_ball_position, world.state.ball_pos)
        original_launch = world._launch_tick
        self._original_launch = original_launch

        def instrumented_launch() -> None:
            original_launch()
            self._launch_after_tick(world)

        world._launch_tick = instrumented_launch

    def _launch_after_tick(self, world: Rival2WorldSim) -> None:
        match = self.state
        wp.launch(
            collect_full_match_tick,
            dim=self.num_worlds,
            inputs=[
                self.goal_capacity,
                world.state.ball_pos,
                world.state.ball_vel,
                world.car_ball.hit_this_tick,
                world.car_ball_b.hit_this_tick,
                world.car_ball.pre_ball_position_bt,
                world.car_ball_b.pre_ball_position_bt,
                world.car_ball.pre_ball_velocity_bt,
                world.car_ball_b.pre_ball_velocity_bt,
                world.car_car.pre_tick_first_car,
                world.ball_world.contact_count,
                world.ball_world.contact_normal,
                world.lifecycle.goal_scored,
                world.lifecycle.scoring_team,
                world.car_car.event_count,
                world.car_car.event_bumper,
                world.car_car.event_is_demo,
                match.regulation_ticks_remaining,
                match.blue_score,
                match.orange_score,
                match.overtime,
                match.done,
                match.winner,
                match.goal_count,
                match.total_ticks,
                match.overtime_ticks,
                match.pending_reset,
                match.kickoff_active,
                match.kickoff_touch_count,
                self.touch_contact_latched,
                self.last_toucher,
                self.previous_ball_position,
                self.touch_start_position,
                self.max_forward_y,
                self.max_backward_y,
                self.active_surface_bits,
                *[getattr(self, name) for name in self._PRE_DIRECTION_CAR_FIELDS],
                self.direction_count,
                self.displacement_count,
                self.wall_continuation_count,
                self.backboard_continuation_count,
                self.demo_count,
                self.goal_overflow,
                *[getattr(self, name) for name in self._GOAL_INT_FIELDS],
                self.goal_entry_x,
                self.goal_entry_z,
            ],
            device=self.device,
        )

    def after_resets(self, world: Rival2WorldSim, reset_mask: wp.array) -> None:
        wp.launch(
            after_full_match_reset,
            dim=self.num_worlds,
            inputs=[
                reset_mask,
                world.state.ball_pos,
                self.state.pending_reset,
                self.state.kickoff_active,
                self.state.kickoff_touch_count,
                self.touch_contact_latched,
                self.last_toucher,
                self.previous_ball_position,
                self.active_surface_bits,
            ],
            device=self.device,
        )

    def numpy(self) -> dict[str, np.ndarray]:
        wp.synchronize_device(self.device)
        values: dict[str, np.ndarray] = {}
        for name, tensor in self.state.torch_views().items():
            values[f"match.{name}"] = tensor.cpu().numpy().copy()
        for name in self._CAR_FIELDS:
            values[name] = np.asarray(getattr(self, name).numpy()).reshape(
                self.num_worlds, 2
            )
        values["direction_count"] = np.asarray(self.direction_count.numpy()).reshape(
            self.num_worlds, 2, 3
        )
        values["displacement_count"] = np.asarray(
            self.displacement_count.numpy()
        ).reshape(self.num_worlds, 2, 3)
        values["goal_overflow"] = np.asarray(self.goal_overflow.numpy())
        for name in self._GOAL_INT_FIELDS:
            values[name] = np.asarray(getattr(self, name).numpy()).reshape(
                self.num_worlds, self.goal_capacity
            )
        values["goal_entry_x"] = np.asarray(self.goal_entry_x.numpy()).reshape(
            self.num_worlds, self.goal_capacity
        )
        values["goal_entry_z"] = np.asarray(self.goal_entry_z.numpy()).reshape(
            self.num_worlds, self.goal_capacity
        )
        return values


@dataclass(frozen=True, slots=True)
class MatchRunTiming:
    physics_ticks_requested: int
    seconds: float
    world_ticks_per_second: float


class FullMatchRunner:
    """Mixed checkpoint-native/15/120 Hz Rival-vs-fixed-policy scheduler."""

    def __init__(
        self,
        num_worlds: int,
        collision_root: str,
        checkpoint_path: str | Path,
        *,
        starting_layout: np.ndarray,
        rival_side: np.ndarray,
        stochastic_rival: bool,
        evaluation_seed: int,
        device: str = "cuda:0",
    ):
        self.num_worlds = int(num_worlds)
        self.device = torch.device(device)
        layout = np.asarray(starting_layout, dtype=np.int32).reshape(self.num_worlds)
        side = np.asarray(rival_side, dtype=np.int32).reshape(self.num_worlds)
        if np.any((layout < 0) | (layout >= 5)):
            raise ValueError("starting layouts must be in [0, 5)")
        if np.any((side < 0) | (side > 1)):
            raise ValueError("Rival side must be Blue=0 or Orange=1")
        self.world = Rival2WorldSim(
            self.num_worlds,
            collision_root,
            device=device,
            seed=evaluation_seed,
            kickoff_selector=layout,
            car_lifecycle_seed=evaluation_seed,
        )
        self.warp_stream = wp.get_stream(self.world.device)
        self.torch_stream = wp.stream_to_torch(self.warp_stream)
        self._activate_stream()
        self.bridge = Rival2TensorBridge(self.world)
        self.match = FullMatchState(self.num_worlds, self.world.device, layout, side)
        self.match_views = self.match.torch_views()
        self.telemetry = FullMatchTelemetry(self.match)
        self.telemetry.attach(self.world)

        checkpoint_path = Path(checkpoint_path)
        checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "RIVAL2_CHECKPOINT_V1":
            raise RuntimeError("unsupported Rival checkpoint format")
        checkpoint_reward = payload.get("reward_version")
        supported_rewards = (
            RIVAL2_REWARD_VERSION,
            RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
            RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
            RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
            RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
            RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
        )
        if checkpoint_reward not in supported_rewards:
            raise RuntimeError("full-match runner requires a Gameplay V1/V2/V3 checkpoint")
        modern_gameplay_checkpoint = checkpoint_reward != RIVAL2_REWARD_VERSION
        if modern_gameplay_checkpoint:
            if payload.get("episode_version") != RIVAL2_EPISODE_VERSION:
                raise RuntimeError("checkpoint episode identity is not RIVAL2_EPISODE_V1")
            expected_contracts = contract_hashes_for_reward(
                checkpoint_reward, RIVAL2_EPISODE_VERSION
            )
            if payload.get("contract_hashes") != expected_contracts:
                raise RuntimeError(
                    "Rival checkpoint observation/action/reward contract mismatch"
                )
        self.rival_policy_hz = int(payload.get("policy_hz", 30))
        if self.rival_policy_hz <= 0 or PHYSICS_HZ % self.rival_policy_hz != 0:
            raise RuntimeError("Rival policy Hz must be a positive divisor of physics Hz")
        self.rival_cadence_ticks = PHYSICS_HZ // self.rival_policy_hz
        self.lifecycle_cadence_ticks = RIVAL_CADENCE_TICKS
        policy_config = Rival2PolicyConfig(**payload["policy_config"])
        if policy_config.content_hash != payload["policy_config_hash"]:
            raise RuntimeError("Rival checkpoint policy contract mismatch")
        self.rival_policy = Rival2ActorCritic(policy_config).to(self.device)
        self.rival_policy.load_state_dict(payload["model"])
        self.rival_policy.eval()
        self.checkpoint_identity = {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
            "iteration": int(payload["iteration"]),
            "policy_version": int(payload["policy_version"]),
            "total_agent_samples": int(payload["total_agent_samples"]),
            "policy_config": asdict(policy_config),
            "policy_config_hash": policy_config.content_hash,
            "reward_version": payload["reward_version"],
            "episode_version": payload.get("episode_version"),
            "contract_hashes": payload.get("contract_hashes"),
            "policy_hz": self.rival_policy_hz,
        }
        del payload

        self.stochastic_rival = bool(stochastic_rival)
        self.rival_generator = torch.Generator(device=self.device)
        self.rival_generator.manual_seed(int(evaluation_seed))
        self.rival_side = self.match_views["rival_side"].to(torch.long)
        self.nexto_side = 1 - self.rival_side
        self.batch_index = torch.arange(self.num_worlds, device=self.device)
        self.nexto = NextoPolicyAdapter(self.num_worlds, device=self.device)
        self.nexto.set_player_index(self.nexto_side)
        self.nexto_state = NextoStateTensors.from_bridge(self.bridge)
        self.rival_observation = self.bridge.observation()
        self.rival_action = torch.zeros(
            (self.num_worlds, 8), dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros(
            (self.num_worlds, 2, 8), dtype=torch.float32, device=self.device
        )
        self.host_tick = 0
        self.world.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(self.device)
        # The graph reads mutable controller and match-state storage; capture
        # does not advance physics and removes dozens of per-tick kernel launches.
        self.world.capture_graph(block_ticks=1)

    def _activate_stream(self) -> None:
        torch.cuda.set_stream(self.torch_stream)
        wp.set_stream(self.warp_stream, device=self.world.device, sync=False)

    def _update_rival_action(self) -> None:
        observation = self.rival_observation[self.batch_index, self.rival_side]
        with torch.inference_mode():
            actor, _value = self.rival_policy(observation)
            if self.stochastic_rival:
                self.rival_action.copy_(
                    sample_hybrid_action(
                        actor, generator=self.rival_generator
                    ).action
                )
            else:
                self.rival_action.copy_(deterministic_hybrid_action(actor))

    def tick(self) -> None:
        self._activate_stream()
        if self.host_tick % self.rival_cadence_ticks == 0:
            self._update_rival_action()
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            self.world.begin_decision()
        kickoff_active = self.match_views["kickoff_active"] != 0
        nexto_action, _indices = self.nexto.tick_action(
            self.nexto_state, kickoff_active
        )
        self.actions[self.batch_index, self.rival_side] = self.rival_action
        self.actions[self.batch_index, self.nexto_side] = nexto_action
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)

        self.match_views["rival_scheduler_tick"].add_(1).remainder_(
            self.rival_cadence_ticks
        )
        self.match_views["nexto_scheduler_tick"].add_(1).remainder_(NEXTO_CADENCE_TICKS)
        self.host_tick += 1
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            # Replace the training timeout mask with the match-owned goal or
            # regulation-tie reset relation before invoking the accepted reset.
            wp.copy(self.world.rival2.reset_mask, self.match.pending_reset)
            reset_mask = self.match_views["pending_reset"] != 0
            self.nexto.notify_kickoff(reset_mask)
            self.world.apply_interval_resets()
            self.telemetry.after_resets(self.world, self.world.rival2.reset_mask)
        if self.host_tick % self.rival_cadence_ticks == 0:
            self.rival_observation = self.bridge.observation()

    def run_ticks(self, ticks: int) -> MatchRunTiming:
        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for _ in range(ticks):
            self.tick()
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        return MatchRunTiming(
            physics_ticks_requested=ticks,
            seconds=seconds,
            world_ticks_per_second=(self.num_worlds * ticks / seconds),
        )

    def profile_ticks(self, ticks: int = 8) -> tuple[MatchRunTiming, list[str]]:
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            for _ in range(ticks):
                self.tick()
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        transfer_names = [
            event.name
            for event in profile.events()
            if "memcpy htod" in event.name.lower()
            or "memcpy dtoh" in event.name.lower()
        ]
        return (
            MatchRunTiming(
                physics_ticks_requested=ticks,
                seconds=seconds,
                world_ticks_per_second=self.num_worlds * ticks / seconds,
            ),
            transfer_names,
        )

    def phase_status(self) -> dict[str, np.ndarray]:
        """Export only match-boundary state needed to decide overtime progress.

        This deliberately excludes telemetry.  The caller may inspect it once
        after regulation and at coarse overtime boundaries without introducing
        any transfer into the timed per-tick policy/physics loop.
        """

        self._activate_stream()
        torch.cuda.synchronize(self.device)
        return {
            name: self.match_views[name].detach().cpu().numpy().copy()
            for name in (
                "blue_score",
                "orange_score",
                "overtime",
                "done",
                "winner",
                "total_ticks",
                "overtime_ticks",
            )
        }

    def export(self) -> dict[str, Any]:
        raw = self.telemetry.numpy()
        return {
            "raw": raw,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "world_host_to_device_bytes_after_initialization": int(
                self.world.host_to_device_bytes
            ),
            "world_device_to_host_bytes_before_export": int(
                self.world.device_to_host_bytes
            ),
            "nexto_timed_h2d_bytes": int(self.nexto.timed_h2d_bytes),
            "nexto_timed_d2h_bytes": int(self.nexto.timed_d2h_bytes),
            "checkpoint": self.checkpoint_identity,
        }


__all__ = [
    "DIRECTION_THRESHOLD_UU_PER_SECOND",
    "FullMatchRunner",
    "FullMatchState",
    "FullMatchTelemetry",
    "GOAL_CAPACITY_PER_WORLD",
    "MatchRunTiming",
    "NEXTO_CADENCE_TICKS",
    "PHYSICS_HZ",
    "REGULATION_TICKS",
    "RIVAL_CADENCE_TICKS",
    "TOUCH_BACKWARD",
    "TOUCH_FORWARD",
    "TOUCH_NEUTRAL",
]
