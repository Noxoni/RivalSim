"""Paired short-lifecycle Rival 2.0 versus pinned Nexto evaluation runtime.

This module is evaluation-only.  It preserves ``RIVAL2_EPISODE_V1`` lifecycle
semantics while scheduling Rival at 30 Hz, pinned Nexto inference at 15 Hz, and
Nexto's stock kickoff controller at 120 Hz.  Device-resident telemetry records
match outcomes, movement, controls, and inspectable jump/dodge/contact sequences
without affecting policy output, reward, lifecycle, or simulator physics.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from rivalsim.constants import DOUBLEJUMP_MAX_DELAY
from rivalsim.kernels.rival2 import (
    EPISODE_LIMIT_TICKS,
    NO_TOUCH_TIMEOUT_TICKS,
    REWARD_MODE_GAMEPLAY,
)
from rivalsim.rival2_contracts import (
    RIVAL2_EPISODE_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    contract_hashes_for_reward,
)
from rivalsim.rival2_env import Rival2TensorBridge, Rival2WorldSim
from rivalsim.rival2_policy import (
    Rival2ActorCritic,
    Rival2PolicyConfig,
    deterministic_hybrid_action,
    sample_hybrid_action,
)
from third_party.nexto.adapter import (
    MODEL_SHA256 as NEXTO_MODEL_SHA256,
)
from third_party.nexto.adapter import (
    UPSTREAM_COMMIT as NEXTO_UPSTREAM_COMMIT,
)
from third_party.nexto.adapter import (
    NextoPolicyAdapter,
    NextoStateTensors,
)

PHYSICS_HZ = 120
RIVAL_CADENCE_TICKS = 4
NEXTO_CADENCE_TICKS = 8
DEFAULT_DASH_EVENT_CAPACITY = 64
SUPPORTED_GAMEPLAY_CHECKPOINT_REWARDS = (
    RIVAL2_REWARD_GAMEPLAY_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    RIVAL2_REWARD_GAMEPLAY_V3_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V1_VERSION,
    RIVAL2_REWARD_GAMEPLAY_120_V2_VERSION,
)
POST_LANDING_SAMPLE_TICKS = 4

# These are prospective operational classification windows, not hidden engine
# tolerances.  Raw timing, suspension, contact, orientation, and velocity
# evidence is retained for every classified event.
LOW_AIR_TIME_TICKS = 42  # 0.35 seconds at 120 Hz
WAVEDASH_LANDING_WINDOW_TICKS = 24  # 0.20 seconds
RAPID_LANDING_TO_JUMP_TICKS = 12  # 0.10 seconds
RAPID_JUMP_TO_FLIP_TICKS = 30  # 0.25 seconds
DOUBLE_DASH_WINDOW_TICKS = 90  # 0.75 seconds

TERMINATION_GOAL = 1
TERMINATION_NO_TOUCH = 2
TERMINATION_HARD_TIME = 3

SURFACE_UNKNOWN = 0
SURFACE_FLOOR_CEILING = 1
SURFACE_CURVED = 2
SURFACE_WALL = 3


@wp.func
def _contact_mask(
    car: int,
    wheel_contact: wp.array(dtype=wp.int32),
) -> int:
    mask = wp.int32(0)
    base = car * 4
    for wheel in range(4):
        if wheel_contact[base + wheel] != 0:
            mask = mask | wp.int32(1 << wheel)
    return mask


@wp.func
def _speed(value: wp.vec3) -> float:
    return wp.sqrt(wp.dot(value, value))


@wp.func
def _planar_speed(value: wp.vec3) -> float:
    return wp.sqrt(value[0] * value[0] + value[1] * value[1])


@wp.func
def _quaternion_up_z(value: wp.quat) -> float:
    return 1.0 - 2.0 * (value[0] * value[0] + value[1] * value[1])


@wp.func
def _average_wheel_normal(
    car: int,
    mask: int,
    wheel_hit_normal: wp.array(dtype=wp.vec3),
) -> wp.vec3:
    normal = wp.vec3(0.0)
    count = 0.0
    base = car * 4
    for wheel in range(4):
        if (mask & wp.int32(1 << wheel)) != 0:
            normal = normal + wheel_hit_normal[base + wheel]
            count = count + 1.0
    if count > 0.0:
        normal = normal / count
        length_squared = wp.dot(normal, normal)
        if length_squared > 1.0e-20:
            normal = normal / wp.sqrt(length_squared)
    return normal


@wp.kernel(enable_backward=False)
def initialize_short_eval_mechanics(
    car_velocity: wp.array(dtype=wp.vec3),
    car_quaternion: wp.array(dtype=wp.quat),
    boost: wp.array(dtype=wp.float32),
    on_ground: wp.array(dtype=wp.int32),
    has_jumped: wp.array(dtype=wp.int32),
    has_double_jumped: wp.array(dtype=wp.int32),
    has_flipped: wp.array(dtype=wp.int32),
    air_time: wp.array(dtype=wp.float32),
    air_time_since_jump: wp.array(dtype=wp.float32),
    wheel_contact: wp.array(dtype=wp.int32),
    suspension_length: wp.array(dtype=wp.float32),
    suspension_velocity: wp.array(dtype=wp.float32),
    previous_velocity: wp.array(dtype=wp.vec3),
    previous_quaternion: wp.array(dtype=wp.quat),
    previous_boost: wp.array(dtype=wp.float32),
    previous_on_ground: wp.array(dtype=wp.int32),
    previous_has_jumped: wp.array(dtype=wp.int32),
    previous_has_double_jumped: wp.array(dtype=wp.int32),
    previous_has_flipped: wp.array(dtype=wp.int32),
    previous_air_time: wp.array(dtype=wp.float32),
    previous_air_time_since_jump: wp.array(dtype=wp.float32),
    previous_wheel_mask: wp.array(dtype=wp.int32),
    previous_suspension_length: wp.array(dtype=wp.float32),
    previous_suspension_velocity: wp.array(dtype=wp.float32),
):
    car = wp.tid()
    previous_velocity[car] = car_velocity[car]
    previous_quaternion[car] = car_quaternion[car]
    previous_boost[car] = boost[car]
    previous_on_ground[car] = on_ground[car]
    previous_has_jumped[car] = has_jumped[car]
    previous_has_double_jumped[car] = has_double_jumped[car]
    previous_has_flipped[car] = has_flipped[car]
    previous_air_time[car] = air_time[car]
    previous_air_time_since_jump[car] = air_time_since_jump[car]
    previous_wheel_mask[car] = _contact_mask(car, wheel_contact)
    for wheel in range(4):
        index = car * 4 + wheel
        previous_suspension_length[index] = suspension_length[index]
        previous_suspension_velocity[index] = suspension_velocity[index]


@wp.kernel(enable_backward=False)
def collect_short_eval_mechanics_tick(
    event_capacity: int,
    episode_ticks: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    car_velocity: wp.array(dtype=wp.vec3),
    car_quaternion: wp.array(dtype=wp.quat),
    boost: wp.array(dtype=wp.float32),
    on_ground: wp.array(dtype=wp.int32),
    has_jumped: wp.array(dtype=wp.int32),
    has_double_jumped: wp.array(dtype=wp.int32),
    has_flipped: wp.array(dtype=wp.int32),
    is_flipping: wp.array(dtype=wp.int32),
    air_time: wp.array(dtype=wp.float32),
    air_time_since_jump: wp.array(dtype=wp.float32),
    flip_rel_torque: wp.array(dtype=wp.vec3),
    wheel_contact: wp.array(dtype=wp.int32),
    wheel_hit_normal: wp.array(dtype=wp.vec3),
    suspension_length: wp.array(dtype=wp.float32),
    suspension_velocity: wp.array(dtype=wp.float32),
    control_throttle: wp.array(dtype=wp.float32),
    control_steer: wp.array(dtype=wp.float32),
    control_pitch: wp.array(dtype=wp.float32),
    control_yaw: wp.array(dtype=wp.float32),
    control_roll: wp.array(dtype=wp.float32),
    control_jump: wp.array(dtype=wp.int32),
    control_boost: wp.array(dtype=wp.int32),
    control_handbrake: wp.array(dtype=wp.int32),
    previous_velocity: wp.array(dtype=wp.vec3),
    previous_quaternion: wp.array(dtype=wp.quat),
    previous_boost: wp.array(dtype=wp.float32),
    previous_on_ground: wp.array(dtype=wp.int32),
    previous_has_jumped: wp.array(dtype=wp.int32),
    previous_has_double_jumped: wp.array(dtype=wp.int32),
    previous_has_flipped: wp.array(dtype=wp.int32),
    previous_air_time: wp.array(dtype=wp.float32),
    previous_air_time_since_jump: wp.array(dtype=wp.float32),
    previous_wheel_mask: wp.array(dtype=wp.int32),
    previous_jump_control: wp.array(dtype=wp.int32),
    previous_suspension_length: wp.array(dtype=wp.float32),
    previous_suspension_velocity: wp.array(dtype=wp.float32),
    last_wheel_contact_tick: wp.array(dtype=wp.int32),
    last_takeoff_tick: wp.array(dtype=wp.int32),
    last_landing_tick: wp.array(dtype=wp.int32),
    last_landing_wheel_mask: wp.array(dtype=wp.int32),
    last_landing_normal: wp.array(dtype=wp.vec3),
    last_jump_rise_tick: wp.array(dtype=wp.int32),
    last_first_jump_tick: wp.array(dtype=wp.int32),
    last_first_jump_wheel_mask_before: wp.array(dtype=wp.int32),
    last_first_jump_wheel_mask_after: wp.array(dtype=wp.int32),
    last_first_jump_velocity_z: wp.array(dtype=wp.float32),
    last_first_jump_suspension_velocity: wp.array(dtype=wp.float32),
    last_flip_event_slot: wp.array(dtype=wp.int32),
    post_landing_event_slot: wp.array(dtype=wp.int32),
    post_landing_sample_tick: wp.array(dtype=wp.int32),
    jump_rising_edges: wp.array(dtype=wp.int32),
    first_jump_onsets: wp.array(dtype=wp.int32),
    double_jump_onsets: wp.array(dtype=wp.int32),
    flip_onsets: wp.array(dtype=wp.int32),
    event_count: wp.array(dtype=wp.int32),
    event_overflow: wp.array(dtype=wp.int32),
    event_tick: wp.array(dtype=wp.int32),
    event_on_ground_before: wp.array(dtype=wp.int32),
    event_on_ground_after: wp.array(dtype=wp.int32),
    event_wheel_mask_before: wp.array(dtype=wp.int32),
    event_wheel_mask_after: wp.array(dtype=wp.int32),
    event_last_wheel_contact_tick: wp.array(dtype=wp.int32),
    event_last_takeoff_tick: wp.array(dtype=wp.int32),
    event_last_landing_tick: wp.array(dtype=wp.int32),
    event_prior_landing_wheel_mask: wp.array(dtype=wp.int32),
    event_last_jump_rise_tick: wp.array(dtype=wp.int32),
    event_last_first_jump_tick: wp.array(dtype=wp.int32),
    event_first_jump_wheel_mask_before: wp.array(dtype=wp.int32),
    event_first_jump_wheel_mask_after: wp.array(dtype=wp.int32),
    event_landing_tick: wp.array(dtype=wp.int32),
    event_landing_wheel_mask: wp.array(dtype=wp.int32),
    event_landing_is_flipping: wp.array(dtype=wp.int32),
    event_action: wp.array(dtype=wp.float32),
    event_flip_rel_torque: wp.array(dtype=wp.vec3),
    event_air_time_before: wp.array(dtype=wp.float32),
    event_air_time_since_jump_before: wp.array(dtype=wp.float32),
    event_first_jump_velocity_z: wp.array(dtype=wp.float32),
    event_first_jump_suspension_velocity: wp.array(dtype=wp.float32),
    event_velocity_before: wp.array(dtype=wp.vec3),
    event_velocity_after: wp.array(dtype=wp.vec3),
    event_up_z_before: wp.array(dtype=wp.float32),
    event_up_z_after: wp.array(dtype=wp.float32),
    event_suspension_length_before: wp.array(dtype=wp.float32),
    event_suspension_velocity_before: wp.array(dtype=wp.float32),
    event_landing_velocity: wp.array(dtype=wp.vec3),
    event_landing_up_z: wp.array(dtype=wp.float32),
    event_landing_normal: wp.array(dtype=wp.vec3),
    event_prior_landing_normal: wp.array(dtype=wp.vec3),
    event_landing_suspension_length: wp.array(dtype=wp.float32),
    event_landing_suspension_velocity: wp.array(dtype=wp.float32),
    event_post_landing_velocity: wp.array(dtype=wp.vec3),
):
    car = wp.tid()
    env = car // 2
    if done[env] != 0:
        return

    tick = episode_ticks[env]
    prior_velocity = previous_velocity[car]
    prior_quaternion = previous_quaternion[car]
    prior_ground = previous_on_ground[car]
    prior_has_jumped = previous_has_jumped[car]
    prior_has_double = previous_has_double_jumped[car]
    prior_has_flipped = previous_has_flipped[car]
    prior_air_time = previous_air_time[car]
    prior_air_since_jump = previous_air_time_since_jump[car]
    prior_wheels = previous_wheel_mask[car]
    current_wheels = _contact_mask(car, wheel_contact)
    jump_control = control_jump[car]
    jump_rise = jump_control != 0 and previous_jump_control[car] == 0

    if jump_rise:
        jump_rising_edges[car] = jump_rising_edges[car] + 1
        last_jump_rise_tick[car] = tick

    first_jump = prior_has_jumped == 0 and has_jumped[car] != 0
    double_jump = prior_has_double == 0 and has_double_jumped[car] != 0
    flip = prior_has_flipped == 0 and has_flipped[car] != 0

    if first_jump:
        first_jump_onsets[car] = first_jump_onsets[car] + 1
        last_first_jump_tick[car] = tick
        last_first_jump_wheel_mask_before[car] = prior_wheels
        last_first_jump_wheel_mask_after[car] = current_wheels
        last_first_jump_velocity_z[car] = prior_velocity[2]
        for wheel in range(4):
            last_first_jump_suspension_velocity[car * 4 + wheel] = previous_suspension_velocity[
                car * 4 + wheel
            ]
    if double_jump:
        double_jump_onsets[car] = double_jump_onsets[car] + 1

    if current_wheels != 0:
        last_wheel_contact_tick[car] = tick
    if prior_wheels != 0 and current_wheels == 0:
        last_takeoff_tick[car] = tick

    if flip:
        flip_onsets[car] = flip_onsets[car] + 1
        ordinal = event_count[car]
        event_count[car] = ordinal + 1
        if ordinal >= event_capacity:
            event_overflow[car] = event_overflow[car] + 1
        else:
            slot = car * event_capacity + ordinal
            event_tick[slot] = tick
            event_on_ground_before[slot] = prior_ground
            event_on_ground_after[slot] = on_ground[car]
            event_wheel_mask_before[slot] = prior_wheels
            event_wheel_mask_after[slot] = current_wheels
            event_last_wheel_contact_tick[slot] = last_wheel_contact_tick[car]
            event_last_takeoff_tick[slot] = last_takeoff_tick[car]
            event_last_landing_tick[slot] = last_landing_tick[car]
            event_prior_landing_wheel_mask[slot] = last_landing_wheel_mask[car]
            event_last_jump_rise_tick[slot] = last_jump_rise_tick[car]
            event_last_first_jump_tick[slot] = last_first_jump_tick[car]
            event_first_jump_wheel_mask_before[slot] = last_first_jump_wheel_mask_before[car]
            event_first_jump_wheel_mask_after[slot] = last_first_jump_wheel_mask_after[car]
            event_flip_rel_torque[slot] = flip_rel_torque[car]
            event_air_time_before[slot] = prior_air_time
            event_air_time_since_jump_before[slot] = prior_air_since_jump
            event_first_jump_velocity_z[slot] = last_first_jump_velocity_z[car]
            event_first_jump_suspension_velocity[slot] = 0.0
            event_velocity_before[slot] = prior_velocity
            event_velocity_after[slot] = car_velocity[car]
            event_up_z_before[slot] = _quaternion_up_z(prior_quaternion)
            event_up_z_after[slot] = _quaternion_up_z(car_quaternion[car])
            event_landing_tick[slot] = -1
            event_landing_wheel_mask[slot] = 0
            event_landing_is_flipping[slot] = 0
            event_landing_normal[slot] = wp.vec3(0.0)
            event_prior_landing_normal[slot] = last_landing_normal[car]
            event_post_landing_velocity[slot] = wp.vec3(0.0)
            action_base = slot * 8
            event_action[action_base] = control_throttle[car]
            event_action[action_base + 1] = control_steer[car]
            event_action[action_base + 2] = control_pitch[car]
            event_action[action_base + 3] = control_yaw[car]
            event_action[action_base + 4] = control_roll[car]
            event_action[action_base + 5] = float(control_jump[car])
            event_action[action_base + 6] = float(control_boost[car])
            event_action[action_base + 7] = float(control_handbrake[car])
            for wheel in range(4):
                wheel_slot = slot * 4 + wheel
                car_wheel = car * 4 + wheel
                event_suspension_length_before[wheel_slot] = previous_suspension_length[car_wheel]
                event_suspension_velocity_before[wheel_slot] = previous_suspension_velocity[
                    car_wheel
                ]
                event_first_jump_suspension_velocity[slot] = wp.max(
                    event_first_jump_suspension_velocity[slot],
                    wp.abs(last_first_jump_suspension_velocity[car_wheel]),
                )
            last_flip_event_slot[car] = slot

    landing = prior_wheels == 0 and current_wheels != 0
    if landing:
        last_landing_tick[car] = tick
        last_landing_wheel_mask[car] = current_wheels
        last_landing_normal[car] = _average_wheel_normal(car, current_wheels, wheel_hit_normal)
        slot = last_flip_event_slot[car]
        if slot >= 0 and event_landing_tick[slot] < 0:
            event_landing_tick[slot] = tick
            event_landing_wheel_mask[slot] = current_wheels
            event_landing_is_flipping[slot] = is_flipping[car]
            event_landing_velocity[slot] = car_velocity[car]
            event_landing_up_z[slot] = _quaternion_up_z(car_quaternion[car])
            event_landing_normal[slot] = _average_wheel_normal(
                car, current_wheels, wheel_hit_normal
            )
            for wheel in range(4):
                wheel_slot = slot * 4 + wheel
                car_wheel = car * 4 + wheel
                event_landing_suspension_length[wheel_slot] = suspension_length[car_wheel]
                event_landing_suspension_velocity[wheel_slot] = suspension_velocity[car_wheel]
            post_landing_event_slot[car] = slot
            post_landing_sample_tick[car] = tick + POST_LANDING_SAMPLE_TICKS

    pending = post_landing_event_slot[car]
    if pending >= 0 and tick >= post_landing_sample_tick[car]:
        event_post_landing_velocity[pending] = car_velocity[car]
        post_landing_event_slot[car] = -1
        post_landing_sample_tick[car] = -1

    previous_velocity[car] = car_velocity[car]
    previous_quaternion[car] = car_quaternion[car]
    previous_boost[car] = boost[car]
    previous_on_ground[car] = on_ground[car]
    previous_has_jumped[car] = has_jumped[car]
    previous_has_double_jumped[car] = has_double_jumped[car]
    previous_has_flipped[car] = has_flipped[car]
    previous_air_time[car] = air_time[car]
    previous_air_time_since_jump[car] = air_time_since_jump[car]
    previous_wheel_mask[car] = current_wheels
    previous_jump_control[car] = jump_control
    for wheel in range(4):
        index = car * 4 + wheel
        previous_suspension_length[index] = suspension_length[index]
        previous_suspension_velocity[index] = suspension_velocity[index]


@wp.kernel(enable_backward=False)
def collect_short_eval_metrics_tick(
    touch_event_capacity: int,
    ball_position: wp.array(dtype=wp.vec3),
    car_velocity: wp.array(dtype=wp.vec3),
    boost: wp.array(dtype=wp.float32),
    is_boosting: wp.array(dtype=wp.int32),
    is_supersonic: wp.array(dtype=wp.int32),
    on_ground: wp.array(dtype=wp.int32),
    pad_boost_gained: wp.array(dtype=wp.float32),
    car_a_hit_this_tick: wp.array(dtype=wp.int32),
    car_b_hit_this_tick: wp.array(dtype=wp.int32),
    pre_tick_first_car: wp.array(dtype=wp.int32),
    interval_tick: wp.array(dtype=wp.int32),
    episode_ticks: wp.array(dtype=wp.int32),
    no_touch_ticks: wp.array(dtype=wp.int32),
    save_count_interval: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
    scoring_team_latched: wp.array(dtype=wp.int32),
    control_throttle: wp.array(dtype=wp.float32),
    control_steer: wp.array(dtype=wp.float32),
    control_pitch: wp.array(dtype=wp.float32),
    control_yaw: wp.array(dtype=wp.float32),
    control_roll: wp.array(dtype=wp.float32),
    control_jump: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.int32),
    winner: wp.array(dtype=wp.int32),
    termination_kind: wp.array(dtype=wp.int32),
    duration_ticks: wp.array(dtype=wp.int32),
    first_toucher: wp.array(dtype=wp.int32),
    touch_contact_latched: wp.array(dtype=wp.int32),
    previous_boost: wp.array(dtype=wp.float32),
    previous_save_count_interval: wp.array(dtype=wp.int32),
    simulated_ticks: wp.array(dtype=wp.int32),
    touch_count: wp.array(dtype=wp.int32),
    touch_event_count: wp.array(dtype=wp.int32),
    touch_event_overflow: wp.array(dtype=wp.int32),
    airborne_touch_count: wp.array(dtype=wp.int32),
    touch_ball_height: wp.array(dtype=wp.float32),
    save_count: wp.array(dtype=wp.int32),
    speed_sum: wp.array(dtype=wp.float32),
    supersonic_ticks: wp.array(dtype=wp.int32),
    grounded_ticks: wp.array(dtype=wp.int32),
    airborne_ticks: wp.array(dtype=wp.int32),
    boost_active_ticks: wp.array(dtype=wp.int32),
    boost_consumed: wp.array(dtype=wp.float32),
    boost_pickups: wp.array(dtype=wp.int32),
    jump_active_ticks: wp.array(dtype=wp.int32),
    analog_saturated_count: wp.array(dtype=wp.int32),
    analog_absolute_sum: wp.array(dtype=wp.float32),
):
    env = wp.tid()
    if done[env] != 0:
        return

    car_base = env * 2
    reports_a = wp.int32(car_a_hit_this_tick[env] != 0)
    reports_b = wp.int32(car_b_hit_this_tick[env] != 0)
    onset_a = wp.int32(reports_a != 0 and touch_contact_latched[car_base] == 0)
    onset_b = wp.int32(reports_b != 0 and touch_contact_latched[car_base + 1] == 0)
    touch_contact_latched[car_base] = reports_a
    touch_contact_latched[car_base + 1] = reports_b
    if onset_a != 0:
        touch_count[car_base] = touch_count[car_base] + 1
        ordinal = touch_event_count[car_base]
        touch_event_count[car_base] = ordinal + 1
        if ordinal < touch_event_capacity:
            touch_ball_height[car_base * touch_event_capacity + ordinal] = ball_position[env][2]
        else:
            touch_event_overflow[car_base] = touch_event_overflow[car_base] + 1
        if on_ground[car_base] == 0:
            airborne_touch_count[car_base] = airborne_touch_count[car_base] + 1
    if onset_b != 0:
        touch_count[car_base + 1] = touch_count[car_base + 1] + 1
        ordinal = touch_event_count[car_base + 1]
        touch_event_count[car_base + 1] = ordinal + 1
        if ordinal < touch_event_capacity:
            touch_ball_height[(car_base + 1) * touch_event_capacity + ordinal] = (
                ball_position[env][2]
            )
        else:
            touch_event_overflow[car_base + 1] = touch_event_overflow[car_base + 1] + 1
        if on_ground[car_base + 1] == 0:
            airborne_touch_count[car_base + 1] = airborne_touch_count[car_base + 1] + 1
    if first_toucher[env] < 0 and (onset_a != 0 or onset_b != 0):
        if onset_a != 0 and onset_b != 0:
            first_toucher[env] = pre_tick_first_car[env]
        elif onset_a != 0:
            first_toucher[env] = 0
        else:
            first_toucher[env] = 1

    for side in range(2):
        car = car_base + side
        velocity = car_velocity[car]
        simulated_ticks[car] = simulated_ticks[car] + 1
        speed_sum[car] = speed_sum[car] + _speed(velocity)
        if is_supersonic[car] != 0:
            supersonic_ticks[car] = supersonic_ticks[car] + 1
        if on_ground[car] != 0:
            grounded_ticks[car] = grounded_ticks[car] + 1
        else:
            airborne_ticks[car] = airborne_ticks[car] + 1
        if is_boosting[car] != 0:
            boost_active_ticks[car] = boost_active_ticks[car] + 1
        gained = pad_boost_gained[car]
        if gained > 0.0:
            boost_pickups[car] = boost_pickups[car] + 1
        delta = previous_boost[car] - boost[car]
        if delta > 0.0 and gained <= 0.0:
            boost_consumed[car] = boost_consumed[car] + delta
        previous_boost[car] = boost[car]
        if control_jump[car] != 0:
            jump_active_ticks[car] = jump_active_ticks[car] + 1

        analog0 = control_throttle[car]
        analog1 = control_steer[car]
        analog2 = control_pitch[car]
        analog3 = control_yaw[car]
        analog4 = control_roll[car]
        analog_absolute_sum[car] = analog_absolute_sum[car] + (
            wp.abs(analog0) + wp.abs(analog1) + wp.abs(analog2) + wp.abs(analog3) + wp.abs(analog4)
        )
        analog_saturated_count[car] = analog_saturated_count[car] + wp.int32(wp.abs(analog0) > 0.95)
        analog_saturated_count[car] = analog_saturated_count[car] + wp.int32(wp.abs(analog1) > 0.95)
        analog_saturated_count[car] = analog_saturated_count[car] + wp.int32(wp.abs(analog2) > 0.95)
        analog_saturated_count[car] = analog_saturated_count[car] + wp.int32(wp.abs(analog3) > 0.95)
        analog_saturated_count[car] = analog_saturated_count[car] + wp.int32(wp.abs(analog4) > 0.95)

        current_save = save_count_interval[car]
        previous_save = previous_save_count_interval[car]
        if interval_tick[env] == 1:
            previous_save = 0
        if current_save > previous_save:
            save_count[car] = save_count[car] + (current_save - previous_save)
        previous_save_count_interval[car] = current_save

    if reset_mask[env] != 0:
        done[env] = 1
        duration_ticks[env] = episode_ticks[env]
        if terminated[env] != 0:
            termination_kind[env] = TERMINATION_GOAL
            winner[env] = scoring_team_latched[env]
        elif no_touch_ticks[env] >= NO_TOUCH_TIMEOUT_TICKS:
            termination_kind[env] = TERMINATION_NO_TOUCH
            winner[env] = -1
        else:
            termination_kind[env] = TERMINATION_HARD_TIME
            winner[env] = -1


class ShortEvalTelemetry:
    """Bounded, device-resident telemetry for exactly one episode per world."""

    _CAR_INT_FIELDS = (
        "simulated_ticks",
        "touch_count",
        "touch_event_count",
        "touch_event_overflow",
        "airborne_touch_count",
        "save_count",
        "supersonic_ticks",
        "grounded_ticks",
        "airborne_ticks",
        "boost_active_ticks",
        "boost_pickups",
        "jump_active_ticks",
        "analog_saturated_count",
        "jump_rising_edges",
        "first_jump_onsets",
        "double_jump_onsets",
        "flip_onsets",
        "event_count",
        "event_overflow",
    )
    _CAR_FLOAT_FIELDS = (
        "speed_sum",
        "boost_consumed",
        "analog_absolute_sum",
    )
    _EVENT_INT_FIELDS = (
        "event_tick",
        "event_on_ground_before",
        "event_on_ground_after",
        "event_wheel_mask_before",
        "event_wheel_mask_after",
        "event_last_wheel_contact_tick",
        "event_last_takeoff_tick",
        "event_last_landing_tick",
        "event_prior_landing_wheel_mask",
        "event_last_jump_rise_tick",
        "event_last_first_jump_tick",
        "event_first_jump_wheel_mask_before",
        "event_first_jump_wheel_mask_after",
        "event_landing_tick",
        "event_landing_wheel_mask",
        "event_landing_is_flipping",
    )
    _EVENT_FLOAT_FIELDS = (
        "event_air_time_before",
        "event_air_time_since_jump_before",
        "event_first_jump_velocity_z",
        "event_first_jump_suspension_velocity",
        "event_up_z_before",
        "event_up_z_after",
        "event_landing_up_z",
    )
    _EVENT_VEC_FIELDS = (
        "event_flip_rel_torque",
        "event_velocity_before",
        "event_velocity_after",
        "event_landing_velocity",
        "event_landing_normal",
        "event_prior_landing_normal",
        "event_post_landing_velocity",
    )

    def __init__(
        self,
        world: Rival2WorldSim,
        *,
        event_capacity: int = DEFAULT_DASH_EVENT_CAPACITY,
    ):
        self.world = world
        self.num_worlds = world.num_envs
        self.device = world.device
        self.car_count = self.num_worlds * 2
        self.event_capacity = int(event_capacity)
        if self.event_capacity <= 0:
            raise ValueError("event capacity must be positive")

        self.done = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.winner = wp.full(self.num_worlds, -1, dtype=wp.int32, device=self.device)
        self.termination_kind = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.duration_ticks = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.first_toucher = wp.full(self.num_worlds, -1, dtype=wp.int32, device=self.device)
        self.touch_contact_latched = wp.zeros(self.car_count, dtype=wp.int32, device=self.device)
        self.previous_save_count_interval = wp.zeros(
            self.car_count, dtype=wp.int32, device=self.device
        )

        for name in self._CAR_INT_FIELDS:
            setattr(self, name, wp.zeros(self.car_count, dtype=wp.int32, device=self.device))
        for name in self._CAR_FLOAT_FIELDS:
            setattr(self, name, wp.zeros(self.car_count, dtype=wp.float32, device=self.device))

        self.previous_velocity = wp.zeros(self.car_count, dtype=wp.vec3, device=self.device)
        self.previous_quaternion = wp.zeros(self.car_count, dtype=wp.quat, device=self.device)
        self.previous_boost = wp.zeros(self.car_count, dtype=wp.float32, device=self.device)
        self.metric_previous_boost = wp.zeros(self.car_count, dtype=wp.float32, device=self.device)
        wp.copy(self.metric_previous_boost, world.state.boost)
        for name in (
            "previous_on_ground",
            "previous_has_jumped",
            "previous_has_double_jumped",
            "previous_has_flipped",
            "previous_wheel_mask",
            "previous_jump_control",
        ):
            setattr(self, name, wp.zeros(self.car_count, dtype=wp.int32, device=self.device))
        for name in ("previous_air_time", "previous_air_time_since_jump"):
            setattr(self, name, wp.zeros(self.car_count, dtype=wp.float32, device=self.device))
        self.previous_suspension_length = wp.zeros(
            self.car_count * 4, dtype=wp.float32, device=self.device
        )
        self.previous_suspension_velocity = wp.zeros(
            self.car_count * 4, dtype=wp.float32, device=self.device
        )

        for name in (
            "last_wheel_contact_tick",
            "last_takeoff_tick",
            "last_landing_tick",
            "last_jump_rise_tick",
            "last_first_jump_tick",
            "last_flip_event_slot",
            "post_landing_event_slot",
            "post_landing_sample_tick",
        ):
            setattr(self, name, wp.full(self.car_count, -1, dtype=wp.int32, device=self.device))
        self.last_landing_wheel_mask = wp.zeros(self.car_count, dtype=wp.int32, device=self.device)
        self.last_landing_normal = wp.zeros(self.car_count, dtype=wp.vec3, device=self.device)
        self.last_first_jump_wheel_mask_before = wp.zeros(
            self.car_count, dtype=wp.int32, device=self.device
        )
        self.last_first_jump_wheel_mask_after = wp.zeros(
            self.car_count, dtype=wp.int32, device=self.device
        )
        self.last_first_jump_velocity_z = wp.zeros(
            self.car_count, dtype=wp.float32, device=self.device
        )
        self.last_first_jump_suspension_velocity = wp.zeros(
            self.car_count * 4, dtype=wp.float32, device=self.device
        )

        event_count = self.car_count * self.event_capacity
        for name in self._EVENT_INT_FIELDS:
            initial = (
                -1
                if name
                in {
                    "event_tick",
                    "event_last_wheel_contact_tick",
                    "event_last_takeoff_tick",
                    "event_last_landing_tick",
                    "event_last_jump_rise_tick",
                    "event_last_first_jump_tick",
                    "event_landing_tick",
                }
                else 0
            )
            setattr(
                self,
                name,
                wp.full(event_count, initial, dtype=wp.int32, device=self.device),
            )
        for name in self._EVENT_FLOAT_FIELDS:
            setattr(self, name, wp.zeros(event_count, dtype=wp.float32, device=self.device))
        for name in self._EVENT_VEC_FIELDS:
            setattr(self, name, wp.zeros(event_count, dtype=wp.vec3, device=self.device))
        self.event_action = wp.zeros(event_count * 8, dtype=wp.float32, device=self.device)
        self.touch_ball_height = wp.zeros(
            event_count, dtype=wp.float32, device=self.device
        )
        for name in (
            "event_suspension_length_before",
            "event_suspension_velocity_before",
            "event_landing_suspension_length",
            "event_landing_suspension_velocity",
        ):
            setattr(
                self,
                name,
                wp.zeros(event_count * 4, dtype=wp.float32, device=self.device),
            )

        wp.launch(
            initialize_short_eval_mechanics,
            dim=self.car_count,
            inputs=[
                world.state.car_vel,
                world.state.car_quat,
                world.state.boost,
                world.state.on_ground,
                world.state.has_jumped,
                world.state.has_double_jumped,
                world.state.has_flipped,
                world.state.air_time,
                world.state.air_time_since_jump,
                world.vehicle.wheel_contact,
                world.vehicle.suspension_length,
                world.vehicle.suspension_velocity,
                self.previous_velocity,
                self.previous_quaternion,
                self.previous_boost,
                self.previous_on_ground,
                self.previous_has_jumped,
                self.previous_has_double_jumped,
                self.previous_has_flipped,
                self.previous_air_time,
                self.previous_air_time_since_jump,
                self.previous_wheel_mask,
                self.previous_suspension_length,
                self.previous_suspension_velocity,
            ],
            device=self.device,
        )
        self._original_launch: Any | None = None

    def attach(self) -> None:
        if self._original_launch is not None:
            raise RuntimeError("short-evaluation telemetry is already attached")
        world = self.world
        original_launch = world._launch_tick
        self._original_launch = original_launch

        def instrumented_launch() -> None:
            original_launch()
            self._launch_mechanics_tick()
            self._launch_metrics_tick()

        world._launch_tick = instrumented_launch

    def _launch_mechanics_tick(self) -> None:
        world = self.world
        controls = world.controls
        wp.launch(
            collect_short_eval_mechanics_tick,
            dim=self.car_count,
            inputs=[
                self.event_capacity,
                world.rival2.episode_ticks,
                self.done,
                world.state.car_vel,
                world.state.car_quat,
                world.state.boost,
                world.state.on_ground,
                world.state.has_jumped,
                world.state.has_double_jumped,
                world.state.has_flipped,
                world.state.is_flipping,
                world.state.air_time,
                world.state.air_time_since_jump,
                world.state.flip_rel_torque,
                world.vehicle.wheel_contact,
                world.vehicle.wheel_hit_normal,
                world.vehicle.suspension_length,
                world.vehicle.suspension_velocity,
                controls.throttle,
                controls.steer,
                controls.pitch,
                controls.yaw,
                controls.roll,
                controls.jump,
                controls.boost,
                controls.handbrake,
                self.previous_velocity,
                self.previous_quaternion,
                self.previous_boost,
                self.previous_on_ground,
                self.previous_has_jumped,
                self.previous_has_double_jumped,
                self.previous_has_flipped,
                self.previous_air_time,
                self.previous_air_time_since_jump,
                self.previous_wheel_mask,
                self.previous_jump_control,
                self.previous_suspension_length,
                self.previous_suspension_velocity,
                self.last_wheel_contact_tick,
                self.last_takeoff_tick,
                self.last_landing_tick,
                self.last_landing_wheel_mask,
                self.last_landing_normal,
                self.last_jump_rise_tick,
                self.last_first_jump_tick,
                self.last_first_jump_wheel_mask_before,
                self.last_first_jump_wheel_mask_after,
                self.last_first_jump_velocity_z,
                self.last_first_jump_suspension_velocity,
                self.last_flip_event_slot,
                self.post_landing_event_slot,
                self.post_landing_sample_tick,
                self.jump_rising_edges,
                self.first_jump_onsets,
                self.double_jump_onsets,
                self.flip_onsets,
                self.event_count,
                self.event_overflow,
                *[getattr(self, name) for name in self._EVENT_INT_FIELDS],
                self.event_action,
                self.event_flip_rel_torque,
                self.event_air_time_before,
                self.event_air_time_since_jump_before,
                self.event_first_jump_velocity_z,
                self.event_first_jump_suspension_velocity,
                self.event_velocity_before,
                self.event_velocity_after,
                self.event_up_z_before,
                self.event_up_z_after,
                self.event_suspension_length_before,
                self.event_suspension_velocity_before,
                self.event_landing_velocity,
                self.event_landing_up_z,
                self.event_landing_normal,
                self.event_prior_landing_normal,
                self.event_landing_suspension_length,
                self.event_landing_suspension_velocity,
                self.event_post_landing_velocity,
            ],
            device=self.device,
        )

    def _launch_metrics_tick(self) -> None:
        world = self.world
        controls = world.controls
        wp.launch(
            collect_short_eval_metrics_tick,
            dim=self.num_worlds,
            inputs=[
                self.event_capacity,
                world.state.ball_pos,
                world.state.car_vel,
                world.state.boost,
                world.state.is_boosting,
                world.state.is_supersonic,
                world.state.on_ground,
                world.lifecycle.pad_boost_gained,
                world.car_ball.hit_this_tick,
                world.car_ball_b.hit_this_tick,
                world.car_car.pre_tick_first_car,
                world.rival2.interval_tick,
                world.rival2.episode_ticks,
                world.rival2.no_touch_ticks,
                world.rival2.save_count,
                world.rival2.terminated,
                world.rival2.truncated,
                world.rival2.reset_mask,
                world.rival2.scoring_team_latched,
                controls.throttle,
                controls.steer,
                controls.pitch,
                controls.yaw,
                controls.roll,
                controls.jump,
                self.done,
                self.winner,
                self.termination_kind,
                self.duration_ticks,
                self.first_toucher,
                self.touch_contact_latched,
                self.metric_previous_boost,
                self.previous_save_count_interval,
                self.simulated_ticks,
                self.touch_count,
                self.touch_event_count,
                self.touch_event_overflow,
                self.airborne_touch_count,
                self.touch_ball_height,
                self.save_count,
                self.speed_sum,
                self.supersonic_ticks,
                self.grounded_ticks,
                self.airborne_ticks,
                self.boost_active_ticks,
                self.boost_consumed,
                self.boost_pickups,
                self.jump_active_ticks,
                self.analog_saturated_count,
                self.analog_absolute_sum,
            ],
            device=self.device,
        )

    def numpy(self) -> dict[str, np.ndarray]:
        wp.synchronize_device(self.device)
        raw: dict[str, np.ndarray] = {
            "done": np.asarray(self.done.numpy()).copy(),
            "winner": np.asarray(self.winner.numpy()).copy(),
            "termination_kind": np.asarray(self.termination_kind.numpy()).copy(),
            "duration_ticks": np.asarray(self.duration_ticks.numpy()).copy(),
            "first_toucher": np.asarray(self.first_toucher.numpy()).copy(),
        }
        for name in self._CAR_INT_FIELDS + self._CAR_FLOAT_FIELDS:
            raw[name] = np.asarray(getattr(self, name).numpy()).reshape(self.num_worlds, 2).copy()
        event_shape = (self.num_worlds, 2, self.event_capacity)
        for name in self._EVENT_INT_FIELDS + self._EVENT_FLOAT_FIELDS:
            raw[name] = np.asarray(getattr(self, name).numpy()).reshape(event_shape).copy()
        for name in self._EVENT_VEC_FIELDS:
            raw[name] = np.asarray(getattr(self, name).numpy()).reshape(*event_shape, 3).copy()
        raw["event_action"] = np.asarray(self.event_action.numpy()).reshape(*event_shape, 8).copy()
        raw["touch_ball_height"] = np.asarray(self.touch_ball_height.numpy()).reshape(
            event_shape
        ).copy()
        for name in (
            "event_suspension_length_before",
            "event_suspension_velocity_before",
            "event_landing_suspension_length",
            "event_landing_suspension_velocity",
        ):
            raw[name] = np.asarray(getattr(self, name).numpy()).reshape(*event_shape, 4).copy()
        return raw


@dataclass(frozen=True, slots=True)
class ShortEvalTiming:
    physics_ticks_requested: int
    seconds: float
    world_ticks_per_second: float


class NextoShortEpisodeRunner:
    """Mixed 30/15/120 Hz runner governed by the original short lifecycle."""

    def __init__(
        self,
        num_worlds: int,
        collision_root: str,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        starting_layout: np.ndarray,
        rival_side: np.ndarray,
        stochastic_rival: bool,
        evaluation_seed: int,
        device: str = "cuda:0",
        dash_event_capacity: int = DEFAULT_DASH_EVENT_CAPACITY,
        rival_policy_hz: int = 30,
        accepted_stage1_checkpoint_format: str | None = None,
        collect_open_play_telemetry: bool = False,
    ):
        self.num_worlds = int(num_worlds)
        self.device = torch.device(device)
        if rival_policy_hz <= 0 or PHYSICS_HZ % int(rival_policy_hz) != 0:
            raise ValueError("Rival policy Hz must be a positive divisor of physics Hz")
        self.rival_policy_hz = int(rival_policy_hz)
        self.rival_cadence_ticks = PHYSICS_HZ // self.rival_policy_hz
        # The original short lifecycle/reward accounting remains a four-physics-
        # tick interval.  A 120 Hz evaluation policy may refresh observations and
        # actions every tick without repeatedly reopening that lifecycle interval.
        self.lifecycle_cadence_ticks = RIVAL_CADENCE_TICKS
        layout = np.asarray(starting_layout, dtype=np.int32).reshape(self.num_worlds)
        side = np.asarray(rival_side, dtype=np.int32).reshape(self.num_worlds)
        if np.any((layout < 0) | (layout >= 5)):
            raise ValueError("starting layouts must be in [0, 5)")
        if np.any((side < 0) | (side > 1)):
            raise ValueError("Rival side must be Blue=0 or Orange=1")
        self.starting_layout = layout
        self.rival_side_host = side
        self.world = Rival2WorldSim(
            self.num_worlds,
            collision_root,
            device=device,
            seed=evaluation_seed,
            kickoff_selector=layout,
            car_lifecycle_seed=evaluation_seed,
            reward_mode=REWARD_MODE_GAMEPLAY,
        )
        self.warp_stream = wp.get_stream(self.world.device)
        self.torch_stream = wp.stream_to_torch(self.warp_stream)
        self._activate_stream()
        self.bridge = Rival2TensorBridge(self.world)

        checkpoint_path = Path(checkpoint_path)
        checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()
        if checkpoint_sha != expected_checkpoint_sha256.upper():
            raise RuntimeError(
                f"Rival checkpoint SHA-256 mismatch: {checkpoint_sha} != "
                f"{expected_checkpoint_sha256.upper()}"
            )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_format = payload.get("format")
        standard_checkpoint = checkpoint_format == "RIVAL2_CHECKPOINT_V1"
        stage1_checkpoint = (
            accepted_stage1_checkpoint_format is not None
            and checkpoint_format == accepted_stage1_checkpoint_format
        )
        if not standard_checkpoint and not stage1_checkpoint:
            raise RuntimeError("unsupported Rival checkpoint format")
        checkpoint_reward = payload.get("reward_version")
        if standard_checkpoint:
            if checkpoint_reward not in SUPPORTED_GAMEPLAY_CHECKPOINT_REWARDS:
                raise RuntimeError("short evaluator requires a Gameplay V1/V2/V3 checkpoint")
            if payload.get("episode_version") != RIVAL2_EPISODE_VERSION:
                raise RuntimeError("checkpoint episode identity is not RIVAL2_EPISODE_V1")
        policy_config = Rival2PolicyConfig(**payload["policy_config"])
        if (
            payload.get("policy_config_hash", policy_config.content_hash)
            != policy_config.content_hash
        ):
            raise RuntimeError("Rival checkpoint policy contract mismatch")
        if standard_checkpoint:
            expected_contracts = contract_hashes_for_reward(
                checkpoint_reward, RIVAL2_EPISODE_VERSION
            )
            if payload.get("contract_hashes") != expected_contracts:
                raise RuntimeError("Rival checkpoint observation/action/reward contract mismatch")
        self.rival_policy = Rival2ActorCritic(policy_config).to(self.device)
        self.rival_policy.load_state_dict(payload["model"])
        self.rival_policy.eval()
        self.checkpoint_identity = {
            "path": checkpoint_path.as_posix(),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
            "format": checkpoint_format,
            "iteration": int(payload.get("iteration", 0)),
            "policy_version": int(payload.get("policy_version", payload.get("selected_step", 0))),
            "total_agent_samples": int(payload.get("total_agent_samples", 0)),
            "selected_step": payload.get("selected_step"),
            "policy_config": asdict(policy_config),
            "policy_config_hash": policy_config.content_hash,
            "reward_version": payload.get("reward_version"),
            "episode_version": payload.get("episode_version"),
            "contract_hashes": payload.get("contract_hashes"),
            "evaluation_only_stage1_load": bool(stage1_checkpoint),
        }
        del payload

        self.stochastic_rival = bool(stochastic_rival)
        self.rival_generator = torch.Generator(device=self.device)
        self.rival_generator.manual_seed(int(evaluation_seed))
        self.rival_side = torch.as_tensor(side, dtype=torch.long, device=self.device)
        self.nexto_side = 1 - self.rival_side
        self.batch_index = torch.arange(self.num_worlds, device=self.device)
        self.nexto = NextoPolicyAdapter(self.num_worlds, device=self.device)
        self.nexto.set_player_index(self.nexto_side)
        self.nexto_state = NextoStateTensors.from_bridge(self.bridge)
        self.rival_observation = self.bridge.observation()
        self.rival_action = torch.zeros(
            (self.num_worlds, 8), dtype=torch.float32, device=self.device
        )
        self.actions = torch.zeros((self.num_worlds, 2, 8), dtype=torch.float32, device=self.device)
        self.telemetry = ShortEvalTelemetry(self.world, event_capacity=dash_event_capacity)
        self.telemetry.attach()
        self.open_play_telemetry = None
        if collect_open_play_telemetry:
            from rivalsim.open_play import OpenPlayTelemetry

            self.open_play_telemetry = OpenPlayTelemetry(self.world)
            self.open_play_telemetry.attach(self.world)
        self.host_tick = 0
        self.world.reset_transfer_counters()
        torch.cuda.reset_peak_memory_stats(self.device)
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
                    sample_hybrid_action(actor, generator=self.rival_generator).action
                )
            else:
                self.rival_action.copy_(deterministic_hybrid_action(actor))

    def tick(self) -> None:
        self._activate_stream()
        if self.host_tick % self.rival_cadence_ticks == 0:
            self._update_rival_action()
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            self.world.begin_decision()
        kickoff_active = self.bridge.views["rival2.kickoff_indicator"] != 0
        nexto_action, _indices = self.nexto.tick_action(self.nexto_state, kickoff_active)
        self.actions[self.batch_index, self.rival_side] = self.rival_action
        self.actions[self.batch_index, self.nexto_side] = nexto_action
        self.bridge.set_actions(self.actions)
        self.world.step_graph(1)
        self.host_tick += 1
        if self.host_tick % self.lifecycle_cadence_ticks == 0:
            reset_mask = self.bridge.views["rival2.reset_mask"].to(torch.bool)
            self.nexto.notify_kickoff(reset_mask)
            self.world.apply_interval_resets()
        if self.host_tick % self.rival_cadence_ticks == 0:
            self.rival_observation = self.bridge.observation()

    def run(self) -> ShortEvalTiming:
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for _ in range(EPISODE_LIMIT_TICKS):
            self.tick()
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        return ShortEvalTiming(
            physics_ticks_requested=EPISODE_LIMIT_TICKS,
            seconds=seconds,
            world_ticks_per_second=(self.num_worlds * EPISODE_LIMIT_TICKS / seconds),
        )

    def export(self) -> dict[str, Any]:
        raw = self.telemetry.numpy()
        result = {
            "raw": raw,
            "checkpoint_identity": self.checkpoint_identity,
            "starting_layout": self.starting_layout.copy(),
            "rival_side": self.rival_side_host.copy(),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "world_host_to_device_bytes_after_initialization": int(self.world.host_to_device_bytes),
            "world_device_to_host_bytes_after_initialization": int(self.world.device_to_host_bytes),
            "nexto_inference_calls": int(self.nexto.inference_calls),
            "nexto_observation_builds": int(self.nexto.observation_builds),
            "nexto_timed_h2d_bytes": int(self.nexto.timed_h2d_bytes),
            "nexto_timed_d2h_bytes": int(self.nexto.timed_d2h_bytes),
            "rival_policy_hz": self.rival_policy_hz,
            "rival_cadence_ticks": self.rival_cadence_ticks,
            "lifecycle_cadence_ticks": self.lifecycle_cadence_ticks,
        }
        if self.open_play_telemetry is not None:
            result["open_play_raw"] = self.open_play_telemetry.numpy()
        return result


def _mask_bits(mask: int) -> list[int]:
    return [index for index in range(4) if mask & (1 << index)]


def _wheel_names(mask: int) -> list[str]:
    names = ("front_left", "front_right", "back_left", "back_right")
    return [name for index, name in enumerate(names) if mask & (1 << index)]


def _direction(action: np.ndarray) -> str:
    pitch = float(action[2])
    lateral = float(action[3] + action[4])
    if abs(pitch) + abs(lateral) < 0.5:
        return "neutral"
    forward = -pitch
    if abs(forward) >= 2.0 * abs(lateral):
        return "forward" if forward > 0 else "backward"
    if abs(lateral) >= 2.0 * abs(forward):
        return "right" if lateral > 0 else "left"
    longitudinal = "forward" if forward > 0 else "backward"
    side = "right" if lateral > 0 else "left"
    return f"{longitudinal}_{side}"


def _surface(normal: np.ndarray) -> tuple[int, str]:
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-9:
        return SURFACE_UNKNOWN, "unknown"
    z = abs(float(normal[2]) / norm)
    if z >= 0.85:
        return SURFACE_FLOOR_CEILING, "floor_or_ceiling"
    if z <= 0.25:
        return SURFACE_WALL, "wall"
    return SURFACE_CURVED, "curved_transition"


def _tangent_speed(velocity: np.ndarray, normal: np.ndarray) -> float:
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1.0e-9:
        return float(np.linalg.norm(velocity[:2]))
    unit_normal = normal / normal_length
    tangent_velocity = velocity - unit_normal * float(np.dot(velocity, unit_normal))
    return float(np.linalg.norm(tangent_velocity))


def classify_dash_events(
    raw: dict[str, np.ndarray],
    *,
    rival_side: np.ndarray,
    starting_layout: np.ndarray,
    checkpoint_label: str,
    opponent_name: str = "Nexto",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify inspectable candidates without inferring intent.

    Every label includes its underlying state/timing evidence.  Engine facts
    (actual flip onset, wheel masks, jump flags, suspension, and velocity) remain
    separate from the prospective operational windows used to group events.
    """

    worlds = int(raw["event_count"].shape[0])
    capacity = int(raw["event_tick"].shape[2])
    classified: list[dict[str, Any]] = []
    all_events_by_car: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for world in range(worlds):
        for side in range(2):
            count = min(int(raw["event_count"][world, side]), capacity)
            per_car: list[dict[str, Any]] = []
            for ordinal in range(count):
                tick = int(raw["event_tick"][world, side, ordinal])
                if tick < 0:
                    continue
                action = raw["event_action"][world, side, ordinal].astype(np.float64)
                before_velocity = raw["event_velocity_before"][world, side, ordinal].astype(
                    np.float64
                )
                after_velocity = raw["event_velocity_after"][world, side, ordinal].astype(
                    np.float64
                )
                landing_velocity = raw["event_landing_velocity"][world, side, ordinal].astype(
                    np.float64
                )
                post_velocity = raw["event_post_landing_velocity"][world, side, ordinal].astype(
                    np.float64
                )
                landing_tick = int(raw["event_landing_tick"][world, side, ordinal])
                last_contact = int(raw["event_last_wheel_contact_tick"][world, side, ordinal])
                last_takeoff = int(raw["event_last_takeoff_tick"][world, side, ordinal])
                last_landing = int(raw["event_last_landing_tick"][world, side, ordinal])
                prior_landing_mask = int(
                    raw["event_prior_landing_wheel_mask"][world, side, ordinal]
                )
                last_jump_rise = int(raw["event_last_jump_rise_tick"][world, side, ordinal])
                last_first_jump = int(raw["event_last_first_jump_tick"][world, side, ordinal])
                wheel_before = int(raw["event_wheel_mask_before"][world, side, ordinal])
                wheel_after = int(raw["event_wheel_mask_after"][world, side, ordinal])
                landing_mask = int(raw["event_landing_wheel_mask"][world, side, ordinal])
                air_seconds = float(raw["event_air_time_before"][world, side, ordinal])
                air_ticks = round(air_seconds * PHYSICS_HZ)
                flip_to_landing = None if landing_tick < 0 else landing_tick - tick
                normal = raw["event_landing_normal"][world, side, ordinal].astype(np.float64)
                surface_code, surface_name = _surface(normal)
                before_speed = float(np.linalg.norm(before_velocity))
                after_speed = float(np.linalg.norm(after_velocity))
                landing_speed = float(np.linalg.norm(landing_velocity))
                post_speed = float(np.linalg.norm(post_velocity))
                before_planar = float(np.linalg.norm(before_velocity[:2]))
                landing_planar = float(np.linalg.norm(landing_velocity[:2]))
                post_planar = float(np.linalg.norm(post_velocity[:2]))
                before_tangent = _tangent_speed(before_velocity, normal)
                landing_tangent = _tangent_speed(landing_velocity, normal)
                post_tangent = _tangent_speed(post_velocity, normal)
                has_post_sample = bool(np.any(post_velocity != 0.0))
                retained_tangent = post_tangent if has_post_sample else landing_tangent

                labels: list[str] = []
                evidence: dict[str, Any] = {}
                if wheel_before != 0 or wheel_after != 0:
                    labels.append("ground_contact_dodge_candidate")
                    evidence["ground_contact_dodge_candidate"] = {
                        "wheel_mask_before": wheel_before,
                        "wheel_mask_after": wheel_after,
                    }

                wavedash = (
                    wheel_before == 0
                    and int(raw["event_on_ground_before"][world, side, ordinal]) == 0
                    and flip_to_landing is not None
                    and 0 <= flip_to_landing <= WAVEDASH_LANDING_WINDOW_TICKS
                    and 0 <= air_ticks <= LOW_AIR_TIME_TICKS
                )
                if wavedash:
                    labels.append("wavedash_candidate")
                    evidence["wavedash_candidate"] = {
                        "actual_flip_onset": True,
                        "airborne_zero_wheels_before": True,
                        "air_time_before_ticks": air_ticks,
                        "flip_to_first_landing_ticks": flip_to_landing,
                        "landing_wheel_mask": landing_mask,
                        "landing_surface": surface_name,
                    }
                    speed_increase = retained_tangent > before_tangent
                    if speed_increase:
                        labels.append("speed_increasing_wavedash_candidate")
                        evidence["speed_increasing_wavedash_candidate"] = {
                            "surface_tangent_speed_before": before_tangent,
                            "surface_tangent_speed_after_landing_sample": retained_tangent,
                            "surface_tangent_speed_delta": retained_tangent - before_tangent,
                            "landing_surface": surface_name,
                            "post_landing_sample_available": has_post_sample,
                        }
                    if surface_code == SURFACE_WALL:
                        labels.append("wall_dash_candidate")
                        evidence["wall_dash_candidate"] = {
                            "landing_normal": normal.tolist(),
                            "surface_rule": "abs(normal_z) <= 0.25",
                        }
                    elif surface_code == SURFACE_CURVED:
                        labels.append("curved_surface_dash_candidate")
                        evidence["curved_surface_dash_candidate"] = {
                            "landing_normal": normal.tolist(),
                            "surface_rule": "0.25 < abs(normal_z) < 0.85",
                        }

                    landing_to_jump = (
                        None
                        if last_landing < 0 or last_first_jump < last_landing
                        else last_first_jump - last_landing
                    )
                    jump_to_flip = None if last_first_jump < 0 else tick - last_first_jump
                    first_jump_mask = int(
                        raw["event_first_jump_wheel_mask_before"][world, side, ordinal]
                    )
                    # RivalSim's authoritative grounded state requires at least
                    # three wheel contacts.  A zapdash therefore has a
                    # separately observable front-wheel-first landing, then a
                    # three-wheel (not flat four-wheel) first-jump onset before
                    # the directional landing dodge.  Keeping each mask makes
                    # this an inspectable state sequence rather than a timing-
                    # only guess.
                    front_first_landing = (prior_landing_mask & 0b0011) != 0 and (
                        prior_landing_mask & 0b1100
                    ) == 0
                    partial_grounded_jump = (
                        first_jump_mask.bit_count() >= 3
                        and first_jump_mask != 0b1111
                        and (first_jump_mask & 0b0011) != 0
                    )
                    if (
                        landing_to_jump is not None
                        and 0 <= landing_to_jump <= RAPID_LANDING_TO_JUMP_TICKS
                        and jump_to_flip is not None
                        and 0 <= jump_to_flip <= RAPID_JUMP_TO_FLIP_TICKS
                        and front_first_landing
                        and partial_grounded_jump
                    ):
                        labels.append("zapdash_candidate")
                        evidence["zapdash_candidate"] = {
                            "prior_landing_to_first_jump_ticks": landing_to_jump,
                            "first_jump_to_flip_ticks": jump_to_flip,
                            "prior_landing_wheel_mask": prior_landing_mask,
                            "prior_landing_wheels": _wheel_names(prior_landing_mask),
                            "front_wheels_contacted_before_rear": front_first_landing,
                            "first_jump_wheel_mask_before": int(
                                raw["event_first_jump_wheel_mask_before"][world, side, ordinal]
                            ),
                            "first_jump_wheels_before": _wheel_names(first_jump_mask),
                            "partial_grounded_jump": partial_grounded_jump,
                            "first_jump_wheel_mask_after": int(
                                raw["event_first_jump_wheel_mask_after"][world, side, ordinal]
                            ),
                            "first_jump_vertical_velocity_before": float(
                                raw["event_first_jump_velocity_z"][world, side, ordinal]
                            ),
                            "first_jump_max_abs_suspension_velocity": float(
                                raw["event_first_jump_suspension_velocity"][world, side, ordinal]
                            ),
                        }

                event = {
                    "checkpoint": checkpoint_label,
                    "world": world,
                    "starting_layout": int(starting_layout[world]),
                    "rival_side": int(rival_side[world]),
                    "physical_side": side,
                    "policy": "Rival" if side == int(rival_side[world]) else opponent_name,
                    "ordinal": ordinal,
                    "tick": tick,
                    "seconds": tick / PHYSICS_HZ,
                    "direction": _direction(action),
                    "controller": action.tolist(),
                    "flip_relative_torque": raw["event_flip_rel_torque"][world, side, ordinal]
                    .astype(float)
                    .tolist(),
                    "on_ground_before": bool(raw["event_on_ground_before"][world, side, ordinal]),
                    "on_ground_after": bool(raw["event_on_ground_after"][world, side, ordinal]),
                    "wheel_mask_before": wheel_before,
                    "wheel_indices_before": _mask_bits(wheel_before),
                    "wheels_before": _wheel_names(wheel_before),
                    "wheel_mask_after": wheel_after,
                    "wheel_indices_after": _mask_bits(wheel_after),
                    "wheels_after": _wheel_names(wheel_after),
                    "last_wheel_contact_tick": last_contact,
                    "last_takeoff_tick": last_takeoff,
                    "last_landing_tick": last_landing,
                    "prior_landing_wheel_mask": prior_landing_mask,
                    "prior_landing_wheels": _wheel_names(prior_landing_mask),
                    "prior_landing_normal": raw["event_prior_landing_normal"][world, side, ordinal]
                    .astype(float)
                    .tolist(),
                    "last_jump_rising_edge_tick": last_jump_rise,
                    "last_first_jump_tick": last_first_jump,
                    "ticks_last_contact_to_flip": (
                        None if last_contact < 0 else tick - last_contact
                    ),
                    "ticks_takeoff_to_flip": (None if last_takeoff < 0 else tick - last_takeoff),
                    "ticks_first_jump_to_flip": (
                        None if last_first_jump < 0 else tick - last_first_jump
                    ),
                    "air_time_before_seconds": air_seconds,
                    "air_time_since_jump_before_seconds": float(
                        raw["event_air_time_since_jump_before"][world, side, ordinal]
                    ),
                    "velocity_before": before_velocity.tolist(),
                    "velocity_after_flip_tick": after_velocity.tolist(),
                    "speed_before": before_speed,
                    "speed_after_flip_tick": after_speed,
                    "planar_speed_before": before_planar,
                    "surface_tangent_speed_before": before_tangent,
                    "up_z_before": float(raw["event_up_z_before"][world, side, ordinal]),
                    "up_z_after_flip_tick": float(raw["event_up_z_after"][world, side, ordinal]),
                    "suspension_length_before": raw["event_suspension_length_before"][
                        world, side, ordinal
                    ]
                    .astype(float)
                    .tolist(),
                    "suspension_velocity_before": raw["event_suspension_velocity_before"][
                        world, side, ordinal
                    ]
                    .astype(float)
                    .tolist(),
                    "landing_tick": None if landing_tick < 0 else landing_tick,
                    "flip_to_landing_ticks": flip_to_landing,
                    "landing_wheel_mask": landing_mask,
                    "landing_wheel_indices": _mask_bits(landing_mask),
                    "landing_wheels": _wheel_names(landing_mask),
                    "landing_is_flipping": bool(
                        raw["event_landing_is_flipping"][world, side, ordinal]
                    ),
                    "landing_surface": surface_name,
                    "landing_normal": normal.tolist(),
                    "landing_velocity": landing_velocity.tolist(),
                    "landing_speed": landing_speed,
                    "landing_planar_speed": landing_planar,
                    "landing_surface_tangent_speed": landing_tangent,
                    "landing_up_z": float(raw["event_landing_up_z"][world, side, ordinal]),
                    "landing_suspension_length": raw["event_landing_suspension_length"][
                        world, side, ordinal
                    ]
                    .astype(float)
                    .tolist(),
                    "landing_suspension_velocity": raw["event_landing_suspension_velocity"][
                        world, side, ordinal
                    ]
                    .astype(float)
                    .tolist(),
                    "post_landing_sample_ticks": (
                        None if landing_tick < 0 else POST_LANDING_SAMPLE_TICKS
                    ),
                    "post_landing_velocity": post_velocity.tolist(),
                    "post_landing_speed": post_speed,
                    "post_landing_planar_speed": post_planar,
                    "post_landing_surface_tangent_speed": post_tangent,
                    "candidate_labels": labels,
                    "classification_evidence": evidence,
                }
                per_car.append(event)
                if labels:
                    classified.append(event)
            all_events_by_car[(world, side)] = per_car

    strict_pair_counts: dict[str, int] = {"Rival": 0, opponent_name: 0}
    for events in all_events_by_car.values():
        wavedashes = [
            event for event in events if "wavedash_candidate" in event["candidate_labels"]
        ]
        for previous, current in pairwise(wavedashes):
            ticks_between = int(current["tick"] - previous["tick"])
            landing_tick = previous["landing_tick"]
            if (
                landing_tick is not None
                and landing_tick <= current["tick"]
                and 0 < ticks_between <= DOUBLE_DASH_WINDOW_TICKS
            ):
                strict_pair_counts[current["policy"]] += 1
                evidence = {
                    "previous_world": previous["world"],
                    "previous_ordinal": previous["ordinal"],
                    "current_ordinal": current["ordinal"],
                    "ticks_between_flip_onsets": ticks_between,
                    "intervening_landing_tick": landing_tick,
                    "fresh_first_jump_between": bool(
                        current["last_first_jump_tick"] is not None
                        and current["last_first_jump_tick"] > landing_tick
                    ),
                }
                for event in (previous, current):
                    if "double_dash_candidate" not in event["candidate_labels"]:
                        event["candidate_labels"].append("double_dash_candidate")
                    event["classification_evidence"]["double_dash_candidate"] = evidence

    # Rebuild to include any double-dash labels added to previously unclassified
    # events while retaining one JSON row per actual flip onset.
    classified = [
        event
        for events in all_events_by_car.values()
        for event in events
        if event["candidate_labels"]
    ]
    counts: dict[str, dict[str, int]] = {"Rival": {}, opponent_name: {}}
    for event in classified:
        policy_counts = counts[event["policy"]]
        for label in event["candidate_labels"]:
            policy_counts[label] = policy_counts.get(label, 0) + 1

    summary = {
        "classification_windows": {
            "physics_hz": PHYSICS_HZ,
            "engine_double_jump_dodge_availability_seconds": float(DOUBLEJUMP_MAX_DELAY),
            "low_air_time_ticks": LOW_AIR_TIME_TICKS,
            "low_air_time_seconds": LOW_AIR_TIME_TICKS / PHYSICS_HZ,
            "wavedash_flip_to_landing_ticks": WAVEDASH_LANDING_WINDOW_TICKS,
            "wavedash_flip_to_landing_seconds": WAVEDASH_LANDING_WINDOW_TICKS / PHYSICS_HZ,
            "zap_landing_to_jump_ticks": RAPID_LANDING_TO_JUMP_TICKS,
            "zap_jump_to_flip_ticks": RAPID_JUMP_TO_FLIP_TICKS,
            "double_dash_flip_to_flip_ticks": DOUBLE_DASH_WINDOW_TICKS,
            "post_landing_speed_sample_ticks": POST_LANDING_SAMPLE_TICKS,
        },
        "definitions": {
            "wavedash_candidate": (
                "actual has_flipped onset from zero wheel contact, pre-flip air "
                "time <=0.35 s, followed by first wheel contact within 0.20 s"
            ),
            "speed_increasing_wavedash_candidate": (
                "wavedash candidate whose measured velocity tangent to the contacted "
                "surface after landing is strictly greater than immediately before "
                "the dodge"
            ),
            "zapdash_candidate": (
                "wavedash candidate preceded by a front-wheel-only first landing, "
                "a non-flat three-wheel first-jump onset within 0.10 s, and the "
                "directional landing dodge within 0.25 s; named wheel masks, "
                "descent velocity, and suspension velocity are retained"
            ),
            "double_dash_candidate": (
                "two wavedash candidates within 0.75 s with an intervening wheel "
                "contact; evidence states whether another first jump occurred"
            ),
            "wall_dash_candidate": (
                "wavedash candidate whose first landing normal has abs(z)<=0.25"
            ),
            "curved_surface_dash_candidate": (
                "wavedash candidate whose first landing normal has 0.25<abs(z)<0.85"
            ),
            "qualification": (
                "labels are prospective state-transition classifications, not "
                "claims of intent; every label retains the underlying controller, "
                "wheel, jump, suspension, orientation, landing, and velocity trace"
            ),
        },
        "actual_flip_events_retained": int(np.minimum(raw["event_count"], capacity).sum()),
        "event_capacity_per_car": capacity,
        "event_overflow_total": int(raw["event_overflow"].sum()),
        "classified_event_count": len(classified),
        "candidate_event_counts_by_policy": counts,
        "strict_double_dash_pair_count_by_policy": strict_pair_counts,
    }
    return classified, summary


__all__ = [
    "DEFAULT_DASH_EVENT_CAPACITY",
    "DOUBLE_DASH_WINDOW_TICKS",
    "LOW_AIR_TIME_TICKS",
    "NEXTO_CADENCE_TICKS",
    "NEXTO_MODEL_SHA256",
    "NEXTO_UPSTREAM_COMMIT",
    "PHYSICS_HZ",
    "RAPID_JUMP_TO_FLIP_TICKS",
    "RAPID_LANDING_TO_JUMP_TICKS",
    "RIVAL_CADENCE_TICKS",
    "TERMINATION_GOAL",
    "TERMINATION_HARD_TIME",
    "TERMINATION_NO_TOUCH",
    "WAVEDASH_LANDING_WINDOW_TICKS",
    "NextoShortEpisodeRunner",
    "ShortEvalTiming",
    "classify_dash_events",
]
