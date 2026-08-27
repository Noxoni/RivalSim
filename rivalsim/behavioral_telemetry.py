"""Read-only 120 Hz behavioral telemetry for the bounded Rival 2.0 evaluation.

The collector is attached only by the behavioral evaluator.  It launches after
the ordinary :class:`Rival2WorldSim` tick and reads existing source-backed event
and contact arrays.  It does not write simulator, controller, observation,
reward, policy, or lifecycle state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

from rivalsim.ball_world_state import MAX_BALL_CONTACTS
from rivalsim.kernels.lifecycle import (
    GOAL_HALF_WIDTH,
    GOAL_HEIGHT,
    GOAL_SCORING_PLANE_Y,
)

PHYSICS_HZ = 120
GOAL_SCORING_PLANE_Y_UU = float(GOAL_SCORING_PLANE_Y)
GOAL_CENTER_Y_UU = 5120.0
GOAL_HALF_WIDTH_UU = float(GOAL_HALF_WIDTH)
GOAL_HEIGHT_UU = float(GOAL_HEIGHT)
GOAL_CENTER_Z_UU = GOAL_HEIGHT_UU / 2.0

MAX_TOUCHES_PER_WORLD = 256
MAX_SURFACE_SEQUENCE = 128

SURFACE_GROUND = 1
SURFACE_SIDE_WALL = 2
SURFACE_BACKBOARD = 3
SURFACE_CEILING = 4

END_NEXT_TOUCH = 1
END_GOAL = 2
END_EPISODE = 3


@wp.func
def _surface_category(normal: wp.vec3) -> int:
    """Classify a retained arena contact by its dominant source normal axis."""

    absolute_x = wp.abs(normal[0])
    absolute_y = wp.abs(normal[1])
    absolute_z = wp.abs(normal[2])
    result = SURFACE_BACKBOARD
    if absolute_z >= absolute_x and absolute_z >= absolute_y:
        result = SURFACE_GROUND if normal[2] >= 0.0 else SURFACE_CEILING
    elif absolute_x >= absolute_y:
        result = SURFACE_SIDE_WALL
    return result


@wp.func
def _update_excursion(
    event: int,
    position: wp.vec3,
    toucher: wp.array(dtype=wp.int32),
    touch_position_after: wp.array(dtype=wp.vec3),
    net_y: wp.array(dtype=wp.float32),
    max_forward_y: wp.array(dtype=wp.float32),
    max_backward_y: wp.array(dtype=wp.float32),
):
    sign = 1.0
    if toucher[event] == 1:
        sign = -1.0
    displacement = sign * (position[1] - touch_position_after[event][1])
    net_y[event] = displacement
    max_forward_y[event] = wp.max(max_forward_y[event], displacement)
    max_backward_y[event] = wp.max(max_backward_y[event], -displacement)


@wp.func
def _finalize_touch(
    event: int,
    tick: int,
    position: wp.vec3,
    next_player: int,
    reason: int,
    toucher: wp.array(dtype=wp.int32),
    touch_position_after: wp.array(dtype=wp.vec3),
    end_tick: wp.array(dtype=wp.int32),
    next_toucher: wp.array(dtype=wp.int32),
    end_reason: wp.array(dtype=wp.int32),
    net_y: wp.array(dtype=wp.float32),
    max_forward_y: wp.array(dtype=wp.float32),
    max_backward_y: wp.array(dtype=wp.float32),
):
    _update_excursion(
        event,
        position,
        toucher,
        touch_position_after,
        net_y,
        max_forward_y,
        max_backward_y,
    )
    end_tick[event] = tick
    next_toucher[event] = next_player
    end_reason[event] = reason


@wp.kernel(enable_backward=False)
def collect_behavioral_tick(
    max_touches_per_world: int,
    max_surface_sequence: int,
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
    episode_ticks: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
    episode_open: wp.array(dtype=wp.int32),
    touch_contact_latched: wp.array(dtype=wp.int32),
    touch_count: wp.array(dtype=wp.int32),
    touch_overflow: wp.array(dtype=wp.int32),
    active_event: wp.array(dtype=wp.int32),
    last_touch_event: wp.array(dtype=wp.int32),
    previous_ball_position: wp.array(dtype=wp.vec3),
    surface_sequence_length: wp.array(dtype=wp.int32),
    surface_sequence_previous_bits: wp.array(dtype=wp.int32),
    surface_sequence_overflow: wp.array(dtype=wp.int32),
    surface_sequence: wp.array(dtype=wp.int32),
    event_tick: wp.array(dtype=wp.int32),
    event_toucher: wp.array(dtype=wp.int32),
    event_end_tick: wp.array(dtype=wp.int32),
    event_next_toucher: wp.array(dtype=wp.int32),
    event_end_reason: wp.array(dtype=wp.int32),
    event_surface_bits: wp.array(dtype=wp.int32),
    event_horizon_valid_bits: wp.array(dtype=wp.int32),
    event_projection_defined: wp.array(dtype=wp.int32),
    event_projection_inside_mouth: wp.array(dtype=wp.int32),
    event_position_before: wp.array(dtype=wp.vec3),
    event_position_after: wp.array(dtype=wp.vec3),
    event_velocity_before: wp.array(dtype=wp.vec3),
    event_velocity_after: wp.array(dtype=wp.vec3),
    event_position_025s: wp.array(dtype=wp.vec3),
    event_position_050s: wp.array(dtype=wp.vec3),
    event_position_100s: wp.array(dtype=wp.vec3),
    event_position_200s: wp.array(dtype=wp.vec3),
    event_net_y: wp.array(dtype=wp.float32),
    event_max_forward_y: wp.array(dtype=wp.float32),
    event_max_backward_y: wp.array(dtype=wp.float32),
    event_longitudinal_delta: wp.array(dtype=wp.float32),
    event_heading_to_goal_3d: wp.array(dtype=wp.float32),
    event_heading_to_goal_planar: wp.array(dtype=wp.float32),
    event_projection_time: wp.array(dtype=wp.float32),
    event_projection_x: wp.array(dtype=wp.float32),
    event_projection_z: wp.array(dtype=wp.float32),
    goal_valid: wp.array(dtype=wp.int32),
    goal_scoring_side: wp.array(dtype=wp.int32),
    goal_tick: wp.array(dtype=wp.int32),
    goal_last_touch_event: wp.array(dtype=wp.int32),
    goal_position: wp.array(dtype=wp.vec3),
    goal_velocity: wp.array(dtype=wp.vec3),
    goal_crossing_valid: wp.array(dtype=wp.int32),
    goal_crossing_position: wp.array(dtype=wp.vec3),
    goal_surface_sequence_length: wp.array(dtype=wp.int32),
    goal_surface_sequence: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    tick = episode_ticks[env]
    position_after = ball_position[env]
    velocity_after = ball_velocity[env]
    reports_a = wp.int32(car_a_hit_this_tick[env] != 0)
    reports_b = wp.int32(car_b_hit_this_tick[env] != 0)
    car_base = env * 2
    touched_a = wp.int32(reports_a != 0 and touch_contact_latched[car_base] == 0)
    touched_b = wp.int32(
        reports_b != 0 and touch_contact_latched[car_base + 1] == 0
    )
    touch_contact_latched[car_base] = reports_a
    touch_contact_latched[car_base + 1] = reports_b

    if episode_open[env] != 0:
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
                previous_event = active_event[env]
                if previous_event >= 0:
                    _finalize_touch(
                        previous_event,
                        tick,
                        position_before,
                        local_toucher,
                        END_NEXT_TOUCH,
                        event_toucher,
                        event_position_after,
                        event_end_tick,
                        event_next_toucher,
                        event_end_reason,
                        event_net_y,
                        event_max_forward_y,
                        event_max_backward_y,
                    )

                slot = touch_count[env]
                touch_count[env] = slot + 1
                active_event[env] = -1
                if slot >= max_touches_per_world:
                    touch_overflow[env] = 1
                else:
                    event = env * max_touches_per_world + slot
                    active_event[env] = event
                    last_touch_event[env] = event
                    event_tick[event] = tick
                    event_toucher[event] = local_toucher
                    event_end_tick[event] = -1
                    event_next_toucher[event] = -1
                    event_end_reason[event] = 0
                    event_surface_bits[event] = 0
                    event_horizon_valid_bits[event] = 0
                    event_projection_defined[event] = 0
                    event_projection_inside_mouth[event] = 0
                    event_position_before[event] = position_before
                    event_position_after[event] = position_after
                    event_velocity_before[event] = velocity_before
                    event_velocity_after[event] = velocity_after
                    event_net_y[event] = 0.0
                    event_max_forward_y[event] = 0.0
                    event_max_backward_y[event] = 0.0

                    sign = 1.0
                    if local_toucher == 1:
                        sign = -1.0
                    canonical_position = wp.vec3(
                        sign * position_after[0],
                        sign * position_after[1],
                        position_after[2],
                    )
                    canonical_velocity = wp.vec3(
                        sign * velocity_after[0],
                        sign * velocity_after[1],
                        velocity_after[2],
                    )
                    canonical_pre_velocity_y = sign * velocity_before[1]
                    event_longitudinal_delta[event] = (
                        canonical_velocity[1] - canonical_pre_velocity_y
                    )

                    goal_vector = wp.vec3(
                        -canonical_position[0],
                        GOAL_CENTER_Y_UU - canonical_position[1],
                        GOAL_CENTER_Z_UU - canonical_position[2],
                    )
                    speed = wp.length(canonical_velocity)
                    goal_distance = wp.length(goal_vector)
                    heading_3d = 0.0
                    if speed > 0.0 and goal_distance > 0.0:
                        cosine = wp.clamp(
                            wp.dot(canonical_velocity, goal_vector)
                            / (speed * goal_distance),
                            -1.0,
                            1.0,
                        )
                        heading_3d = wp.acos(cosine)
                    event_heading_to_goal_3d[event] = heading_3d

                    planar_speed = wp.sqrt(
                        canonical_velocity[0] * canonical_velocity[0]
                        + canonical_velocity[1] * canonical_velocity[1]
                    )
                    planar_goal_distance = wp.sqrt(
                        goal_vector[0] * goal_vector[0]
                        + goal_vector[1] * goal_vector[1]
                    )
                    heading_planar = 0.0
                    if planar_speed > 0.0 and planar_goal_distance > 0.0:
                        planar_cosine = wp.clamp(
                            (
                                canonical_velocity[0] * goal_vector[0]
                                + canonical_velocity[1] * goal_vector[1]
                            )
                            / (planar_speed * planar_goal_distance),
                            -1.0,
                            1.0,
                        )
                        heading_planar = wp.acos(planar_cosine)
                    event_heading_to_goal_planar[event] = heading_planar

                    if (
                        canonical_velocity[1] > 0.000001
                        and canonical_position[1] < GOAL_SCORING_PLANE_Y_UU
                    ):
                        projection_time = (
                            GOAL_SCORING_PLANE_Y_UU - canonical_position[1]
                        ) / canonical_velocity[1]
                        projection_x = (
                            canonical_position[0]
                            + canonical_velocity[0] * projection_time
                        )
                        projection_z = (
                            canonical_position[2]
                            + canonical_velocity[2] * projection_time
                        )
                        event_projection_defined[event] = 1
                        event_projection_time[event] = projection_time
                        event_projection_x[event] = projection_x
                        event_projection_z[event] = projection_z
                        event_projection_inside_mouth[event] = wp.int32(
                            wp.abs(projection_x) <= GOAL_HALF_WIDTH_UU
                            and projection_z >= 0.0
                            and projection_z <= GOAL_HEIGHT_UU
                        )
                    else:
                        event_projection_time[event] = 0.0
                        event_projection_x[event] = 0.0
                        event_projection_z[event] = 0.0

                    surface_sequence_length[env] = 0
                    surface_sequence_previous_bits[env] = 0

        current_event = active_event[env]
        if current_event >= 0:
            contacts = ball_contact_count[env]
            contact_base = env * MAX_BALL_CONTACTS
            current_surface_bits = wp.int32(0)
            for relative in range(MAX_BALL_CONTACTS):
                if relative < contacts:
                    category = _surface_category(
                        ball_contact_normal[contact_base + relative]
                    )
                    category_bit = wp.int32(1 << (category - 1))
                    current_surface_bits = current_surface_bits | category_bit
                    event_surface_bits[current_event] = (
                        event_surface_bits[current_event] | category_bit
                    )

            new_surface_bits = (
                current_surface_bits & ~surface_sequence_previous_bits[env]
            )
            for category_offset in range(4):
                category = category_offset + 1
                category_bit = wp.int32(1 << category_offset)
                if (new_surface_bits & category_bit) != 0:
                    sequence_slot = surface_sequence_length[env]
                    if sequence_slot < max_surface_sequence:
                        surface_sequence[
                            env * max_surface_sequence + sequence_slot
                        ] = category
                        surface_sequence_length[env] = sequence_slot + 1
                    else:
                        surface_sequence_overflow[env] = 1
            surface_sequence_previous_bits[env] = current_surface_bits

            _update_excursion(
                current_event,
                position_after,
                event_toucher,
                event_position_after,
                event_net_y,
                event_max_forward_y,
                event_max_backward_y,
            )
            age = tick - event_tick[current_event]
            if age == 30:
                event_position_025s[current_event] = position_after
                event_horizon_valid_bits[current_event] = (
                    event_horizon_valid_bits[current_event] | 1
                )
            elif age == 60:
                event_position_050s[current_event] = position_after
                event_horizon_valid_bits[current_event] = (
                    event_horizon_valid_bits[current_event] | 2
                )
            elif age == 120:
                event_position_100s[current_event] = position_after
                event_horizon_valid_bits[current_event] = (
                    event_horizon_valid_bits[current_event] | 4
                )
            elif age == 240:
                event_position_200s[current_event] = position_after
                event_horizon_valid_bits[current_event] = (
                    event_horizon_valid_bits[current_event] | 8
                )

        if goal_scored[env] != 0:
            goal_valid[env] = 1
            goal_scoring_side[env] = scoring_team[env]
            goal_tick[env] = tick
            goal_last_touch_event[env] = last_touch_event[env]
            goal_position[env] = position_after
            goal_velocity[env] = velocity_after
            goal_crossing_valid[env] = 0
            scoring_sign = 1.0
            if scoring_team[env] == 1:
                scoring_sign = -1.0
            scoring_plane = scoring_sign * GOAL_SCORING_PLANE_Y_UU
            position_before_tick = previous_ball_position[env]
            delta_y = position_after[1] - position_before_tick[1]
            if wp.abs(delta_y) > 0.000001:
                fraction = (scoring_plane - position_before_tick[1]) / delta_y
                if fraction >= 0.0 and fraction <= 1.0:
                    goal_crossing_valid[env] = 1
                    goal_crossing_position[env] = position_before_tick + (
                        position_after - position_before_tick
                    ) * fraction

            sequence_length = surface_sequence_length[env]
            goal_surface_sequence_length[env] = sequence_length
            for sequence_slot in range(MAX_SURFACE_SEQUENCE):
                if sequence_slot < sequence_length:
                    goal_surface_sequence[
                        env * max_surface_sequence + sequence_slot
                    ] = surface_sequence[
                        env * max_surface_sequence + sequence_slot
                    ]

            current_event = active_event[env]
            if current_event >= 0:
                _finalize_touch(
                    current_event,
                    tick,
                    position_after,
                    -1,
                    END_GOAL,
                    event_toucher,
                    event_position_after,
                    event_end_tick,
                    event_next_toucher,
                    event_end_reason,
                    event_net_y,
                    event_max_forward_y,
                    event_max_backward_y,
                )
            active_event[env] = -1
            episode_open[env] = 0
        elif reset_mask[env] != 0:
            current_event = active_event[env]
            if current_event >= 0:
                _finalize_touch(
                    current_event,
                    tick,
                    position_after,
                    -1,
                    END_EPISODE,
                    event_toucher,
                    event_position_after,
                    event_end_tick,
                    event_next_toucher,
                    event_end_reason,
                    event_net_y,
                    event_max_forward_y,
                    event_max_backward_y,
                )
            active_event[env] = -1
            episode_open[env] = 0

    previous_ball_position[env] = position_after


@wp.kernel(enable_backward=False)
def initialize_behavioral_telemetry(
    ball_position: wp.array(dtype=wp.vec3),
    previous_ball_position: wp.array(dtype=wp.vec3),
):
    env = wp.tid()
    previous_ball_position[env] = ball_position[env]


class BehavioralTelemetry:
    """GPU-resident touch/goal recorder attached to one evaluation world."""

    _EVENT_INT_FIELDS = (
        "event_tick",
        "event_toucher",
        "event_end_tick",
        "event_next_toucher",
        "event_end_reason",
        "event_surface_bits",
        "event_horizon_valid_bits",
        "event_projection_defined",
        "event_projection_inside_mouth",
    )
    _EVENT_VECTOR_FIELDS = (
        "event_position_before",
        "event_position_after",
        "event_velocity_before",
        "event_velocity_after",
        "event_position_025s",
        "event_position_050s",
        "event_position_100s",
        "event_position_200s",
    )
    _EVENT_FLOAT_FIELDS = (
        "event_net_y",
        "event_max_forward_y",
        "event_max_backward_y",
        "event_longitudinal_delta",
        "event_heading_to_goal_3d",
        "event_heading_to_goal_planar",
        "event_projection_time",
        "event_projection_x",
        "event_projection_z",
    )

    def __init__(
        self,
        num_envs: int,
        device: str,
        *,
        max_touches_per_world: int = MAX_TOUCHES_PER_WORLD,
        max_surface_sequence: int = MAX_SURFACE_SEQUENCE,
    ):
        self.num_envs = num_envs
        self.device = device
        self.max_touches_per_world = max_touches_per_world
        self.max_surface_sequence = max_surface_sequence
        event_capacity = num_envs * max_touches_per_world
        self.episode_open = wp.ones(num_envs, dtype=wp.int32, device=device)
        self.touch_contact_latched = wp.zeros(
            num_envs * 2, dtype=wp.int32, device=device
        )
        self.touch_count = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.touch_overflow = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.active_event = wp.full(num_envs, -1, dtype=wp.int32, device=device)
        self.last_touch_event = wp.full(num_envs, -1, dtype=wp.int32, device=device)
        self.previous_ball_position = wp.zeros(
            num_envs, dtype=wp.vec3, device=device
        )
        self.surface_sequence_length = wp.zeros(
            num_envs, dtype=wp.int32, device=device
        )
        self.surface_sequence_previous_bits = wp.zeros(
            num_envs, dtype=wp.int32, device=device
        )
        self.surface_sequence_overflow = wp.zeros(
            num_envs, dtype=wp.int32, device=device
        )
        self.surface_sequence = wp.zeros(
            num_envs * max_surface_sequence, dtype=wp.int32, device=device
        )
        for name in self._EVENT_INT_FIELDS:
            initial = -1 if name in {
                "event_end_tick",
                "event_next_toucher",
            } else 0
            setattr(
                self,
                name,
                wp.full(event_capacity, initial, dtype=wp.int32, device=device),
            )
        for name in self._EVENT_VECTOR_FIELDS:
            setattr(self, name, wp.zeros(event_capacity, dtype=wp.vec3, device=device))
        for name in self._EVENT_FLOAT_FIELDS:
            setattr(
                self,
                name,
                wp.zeros(event_capacity, dtype=wp.float32, device=device),
            )

        self.goal_valid = wp.zeros(num_envs, dtype=wp.int32, device=device)
        self.goal_scoring_side = wp.full(
            num_envs, -1, dtype=wp.int32, device=device
        )
        self.goal_tick = wp.full(num_envs, -1, dtype=wp.int32, device=device)
        self.goal_last_touch_event = wp.full(
            num_envs, -1, dtype=wp.int32, device=device
        )
        self.goal_position = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self.goal_velocity = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self.goal_crossing_valid = wp.zeros(
            num_envs, dtype=wp.int32, device=device
        )
        self.goal_crossing_position = wp.zeros(
            num_envs, dtype=wp.vec3, device=device
        )
        self.goal_surface_sequence_length = wp.zeros(
            num_envs, dtype=wp.int32, device=device
        )
        self.goal_surface_sequence = wp.zeros(
            num_envs * max_surface_sequence, dtype=wp.int32, device=device
        )
        self._original_launch: Callable[[], None] | None = None

    def attach(self, world: Any) -> None:
        """Wrap the ordinary tick with a post-tick read-only telemetry launch."""

        if self._original_launch is not None:
            raise RuntimeError("behavioral telemetry is already attached")
        if world.num_envs != self.num_envs or world.device != self.device:
            raise ValueError("telemetry/world shape or device mismatch")
        wp.launch(
            initialize_behavioral_telemetry,
            dim=self.num_envs,
            inputs=[world.state.ball_pos, self.previous_ball_position],
            device=self.device,
        )
        original_launch = world._launch_tick
        self._original_launch = original_launch

        def instrumented_launch() -> None:
            original_launch()
            self._launch_after_tick(world)

        world._launch_tick = instrumented_launch

    def _launch_after_tick(self, world: Any) -> None:
        pair_a = world.car_ball
        pair_b = world.car_ball_b
        ball = world.ball_world
        state = world.rival2
        wp.launch(
            collect_behavioral_tick,
            dim=self.num_envs,
            inputs=[
                self.max_touches_per_world,
                self.max_surface_sequence,
                world.state.ball_pos,
                world.state.ball_vel,
                pair_a.hit_this_tick,
                pair_b.hit_this_tick,
                pair_a.pre_ball_position_bt,
                pair_b.pre_ball_position_bt,
                pair_a.pre_ball_velocity_bt,
                pair_b.pre_ball_velocity_bt,
                world.car_car.pre_tick_first_car,
                ball.contact_count,
                ball.contact_normal,
                world.lifecycle.goal_scored,
                world.lifecycle.scoring_team,
                state.episode_ticks,
                state.reset_mask,
                self.episode_open,
                self.touch_contact_latched,
                self.touch_count,
                self.touch_overflow,
                self.active_event,
                self.last_touch_event,
                self.previous_ball_position,
                self.surface_sequence_length,
                self.surface_sequence_previous_bits,
                self.surface_sequence_overflow,
                self.surface_sequence,
                *[getattr(self, name) for name in self._EVENT_INT_FIELDS],
                *[getattr(self, name) for name in self._EVENT_VECTOR_FIELDS],
                *[getattr(self, name) for name in self._EVENT_FLOAT_FIELDS],
                self.goal_valid,
                self.goal_scoring_side,
                self.goal_tick,
                self.goal_last_touch_event,
                self.goal_position,
                self.goal_velocity,
                self.goal_crossing_valid,
                self.goal_crossing_position,
                self.goal_surface_sequence_length,
                self.goal_surface_sequence,
            ],
            device=self.device,
        )

    def numpy(self) -> dict[str, np.ndarray]:
        """Synchronize once and return the complete bounded raw recorder."""

        wp.synchronize_device(self.device)
        fields = {
            "episode_open": self.episode_open,
            "touch_count": self.touch_count,
            "touch_overflow": self.touch_overflow,
            "surface_sequence_overflow": self.surface_sequence_overflow,
            "goal_valid": self.goal_valid,
            "goal_scoring_side": self.goal_scoring_side,
            "goal_tick": self.goal_tick,
            "goal_last_touch_event": self.goal_last_touch_event,
            "goal_position": self.goal_position,
            "goal_velocity": self.goal_velocity,
            "goal_crossing_valid": self.goal_crossing_valid,
            "goal_crossing_position": self.goal_crossing_position,
            "goal_surface_sequence_length": self.goal_surface_sequence_length,
            "goal_surface_sequence": self.goal_surface_sequence,
        }
        for name in (
            *self._EVENT_INT_FIELDS,
            *self._EVENT_VECTOR_FIELDS,
            *self._EVENT_FLOAT_FIELDS,
        ):
            fields[name] = getattr(self, name)
        return {name: np.asarray(value.numpy()).copy() for name, value in fields.items()}


__all__ = [
    "END_EPISODE",
    "END_GOAL",
    "END_NEXT_TOUCH",
    "GOAL_CENTER_Y_UU",
    "GOAL_CENTER_Z_UU",
    "GOAL_HALF_WIDTH_UU",
    "GOAL_HEIGHT_UU",
    "GOAL_SCORING_PLANE_Y_UU",
    "MAX_SURFACE_SEQUENCE",
    "MAX_TOUCHES_PER_WORLD",
    "PHYSICS_HZ",
    "SURFACE_BACKBOARD",
    "SURFACE_CEILING",
    "SURFACE_GROUND",
    "SURFACE_SIDE_WALL",
    "BehavioralTelemetry",
]
