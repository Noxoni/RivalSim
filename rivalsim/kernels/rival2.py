"""GPU-native Rival 2.0 interval accounting and selective reset kernels."""

from __future__ import annotations

import warp as wp

from rivalsim.kernels.boost_pad import PAD_COUNT
from rivalsim.kernels.lifecycle import (
    BALL_REST_Z,
    CAR_SPAWN_BOOST,
    _kickoff_position,
    _kickoff_quaternion,
)

PHYSICS_TICKS_PER_DECISION = 4
NO_TOUCH_TIMEOUT_TICKS = 15 * 120
EPISODE_LIMIT_TICKS = 45 * 120
GOAL_PROGRESS_SCALE_Y = 5120.0
REWARD_MODE_BASE = 0
REWARD_MODE_ACQUISITION = 1
REWARD_MODE_GOAL_ONLY = 2


@wp.kernel(enable_backward=False)
def rival2_begin_decision(
    ball_pos: wp.array(dtype=wp.vec3),
    interval_tick: wp.array(dtype=wp.int32),
    ball_y_before: wp.array(dtype=wp.float32),
    ball_y_after: wp.array(dtype=wp.float32),
    touch_count: wp.array(dtype=wp.int32),
    first_contact_count: wp.array(dtype=wp.int32),
    demo_by_count: wp.array(dtype=wp.int32),
    demoed_event: wp.array(dtype=wp.int32),
    goal_latched: wp.array(dtype=wp.int32),
    scoring_team_latched: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
    kickoff_indicator: wp.array(dtype=wp.int32),
):
    """Open one 30 Hz decision interval without touching episode-age state."""

    env = wp.tid()
    car_base = env * 2
    interval_tick[env] = 0
    ball_y_before[env] = ball_pos[env][1]
    ball_y_after[env] = ball_pos[env][1]
    goal_latched[env] = 0
    scoring_team_latched[env] = -1
    terminated[env] = 0
    truncated[env] = 0
    reset_mask[env] = 0
    kickoff_indicator[env] = 0
    for local_car in range(2):
        car = car_base + local_car
        touch_count[car] = 0
        first_contact_count[car] = 0
        demo_by_count[car] = 0
        demoed_event[car] = 0
        reward[car] = 0.0


@wp.kernel(enable_backward=False)
def rival2_accumulate_tick(
    reward_mode: int,
    ball_pos: wp.array(dtype=wp.vec3),
    goal_scored: wp.array(dtype=wp.int32),
    scoring_team: wp.array(dtype=wp.int32),
    car_a_hit_this_tick: wp.array(dtype=wp.int32),
    car_b_hit_this_tick: wp.array(dtype=wp.int32),
    bump_event_count: wp.array(dtype=wp.int32),
    bump_event_bumper: wp.array(dtype=wp.int32),
    bump_event_victim: wp.array(dtype=wp.int32),
    bump_event_is_demo: wp.array(dtype=wp.int32),
    interval_tick: wp.array(dtype=wp.int32),
    episode_ticks: wp.array(dtype=wp.int32),
    no_touch_ticks: wp.array(dtype=wp.int32),
    ball_y_before: wp.array(dtype=wp.float32),
    ball_y_after: wp.array(dtype=wp.float32),
    touch_count: wp.array(dtype=wp.int32),
    touch_contact_latched: wp.array(dtype=wp.int32),
    episode_player_touched: wp.array(dtype=wp.int32),
    first_contact_count: wp.array(dtype=wp.int32),
    demo_by_count: wp.array(dtype=wp.int32),
    demoed_event: wp.array(dtype=wp.int32),
    goal_latched: wp.array(dtype=wp.int32),
    scoring_team_latched: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
):
    """Accumulate source-backed events and finish reward/done on tick four."""

    env = wp.tid()
    car_base = env * 2
    reports_a = wp.int32(car_a_hit_this_tick[env] != 0)
    reports_b = wp.int32(car_b_hit_this_tick[env] != 0)
    touched_a = wp.int32(reports_a != 0 and touch_contact_latched[car_base] == 0)
    touched_b = wp.int32(reports_b != 0 and touch_contact_latched[car_base + 1] == 0)
    touch_contact_latched[car_base] = reports_a
    touch_contact_latched[car_base + 1] = reports_b
    if touched_a != 0:
        touch_count[car_base] = touch_count[car_base] + 1
        if episode_player_touched[car_base] == 0:
            episode_player_touched[car_base] = 1
            first_contact_count[car_base] = first_contact_count[car_base] + 1
    if touched_b != 0:
        touch_count[car_base + 1] = touch_count[car_base + 1] + 1
        if episode_player_touched[car_base + 1] == 0:
            episode_player_touched[car_base + 1] = 1
            first_contact_count[car_base + 1] = first_contact_count[car_base + 1] + 1
    if touched_a != 0 or touched_b != 0:
        no_touch_ticks[env] = 0
    else:
        no_touch_ticks[env] = no_touch_ticks[env] + 1

    event_base = env * 4
    count = bump_event_count[env]
    for relative in range(4):
        if relative < count:
            event = event_base + relative
            if bump_event_is_demo[event] != 0:
                bumper = bump_event_bumper[event]
                victim = bump_event_victim[event]
                if bumper >= 0 and bumper < 2:
                    demo_by_count[car_base + bumper] = demo_by_count[car_base + bumper] + 1
                if victim >= 0 and victim < 2:
                    demoed_event[car_base + victim] = 1

    if goal_scored[env] != 0 and goal_latched[env] == 0:
        goal_latched[env] = 1
        scoring_team_latched[env] = scoring_team[env]

    episode_ticks[env] = episode_ticks[env] + 1
    ball_y_after[env] = ball_pos[env][1]
    next_interval_tick = interval_tick[env] + 1
    interval_tick[env] = next_interval_tick

    if next_interval_tick == PHYSICS_TICKS_PER_DECISION:
        terminal = goal_latched[env]
        timeout = wp.int32(
            terminal == 0
            and (
                no_touch_ticks[env] >= NO_TOUCH_TIMEOUT_TICKS
                or episode_ticks[env] >= EPISODE_LIMIT_TICKS
            )
        )
        terminated[env] = terminal
        truncated[env] = timeout
        reset_mask[env] = wp.int32(terminal != 0 or timeout != 0)

        blue_reward = 0.0
        orange_reward = 0.0
        if reward_mode == REWARD_MODE_BASE:
            # Preserve the exact historical RIVAL2_REWARD_V1/V2 arithmetic and
            # operation order. Orange remains the final negation of Blue.
            blue_reward = (
                0.5
                * (ball_y_after[env] - ball_y_before[env])
                / GOAL_PROGRESS_SCALE_Y
            )
            if terminal != 0:
                if scoring_team_latched[env] == 0:
                    blue_reward = blue_reward + 10.0
                else:
                    blue_reward = blue_reward - 10.0
            blue_reward = blue_reward + 0.05 * float(
                touch_count[car_base] - touch_count[car_base + 1]
            )
            blue_reward = blue_reward + 0.10 * float(
                demo_by_count[car_base] - demo_by_count[car_base + 1]
            )
            orange_reward = -blue_reward
        elif reward_mode == REWARD_MODE_ACQUISITION:
            progress_reward = (
                0.5
                * (ball_y_after[env] - ball_y_before[env])
                / GOAL_PROGRESS_SCALE_Y
            )
            blue_reward = progress_reward
            orange_reward = -progress_reward
            if terminal != 0:
                if scoring_team_latched[env] == 0:
                    blue_reward = blue_reward + 10.0
                    orange_reward = orange_reward - 10.0
                else:
                    blue_reward = blue_reward - 10.0
                    orange_reward = orange_reward + 10.0
            blue_reward = blue_reward + 0.20 * float(touch_count[car_base])
            orange_reward = orange_reward + 0.20 * float(touch_count[car_base + 1])
            blue_reward = blue_reward + float(first_contact_count[car_base])
            orange_reward = orange_reward + float(first_contact_count[car_base + 1])
            no_touch_failure = wp.int32(
                terminal == 0 and no_touch_ticks[env] >= NO_TOUCH_TIMEOUT_TICKS
            )
            if no_touch_failure != 0:
                if episode_player_touched[car_base] == 0:
                    blue_reward = blue_reward - 0.5
                if episode_player_touched[car_base + 1] == 0:
                    orange_reward = orange_reward - 0.5
            demo_reward = 0.10 * float(
                demo_by_count[car_base] - demo_by_count[car_base + 1]
            )
            blue_reward = blue_reward + demo_reward
            orange_reward = orange_reward - demo_reward
        else:
            # Goal-only mode intentionally excludes progress, touch, approach,
            # no-touch, demo, and mechanic shaping.
            if terminal != 0:
                if scoring_team_latched[env] == 0:
                    blue_reward = blue_reward + 10.0
                    orange_reward = orange_reward - 10.0
                else:
                    blue_reward = blue_reward - 10.0
                    orange_reward = orange_reward + 10.0
        reward[car_base] = blue_reward
        reward[car_base + 1] = orange_reward


@wp.kernel(enable_backward=False)
def rival2_after_interval_reset(
    reset_mask: wp.array(dtype=wp.int32),
    episode_ticks: wp.array(dtype=wp.int32),
    no_touch_ticks: wp.array(dtype=wp.int32),
    kickoff_indicator: wp.array(dtype=wp.int32),
    touch_count: wp.array(dtype=wp.int32),
    touch_contact_latched: wp.array(dtype=wp.int32),
    episode_player_touched: wp.array(dtype=wp.int32),
    first_contact_count: wp.array(dtype=wp.int32),
    demo_by_count: wp.array(dtype=wp.int32),
    demoed_event: wp.array(dtype=wp.int32),
    previous_action: wp.array(dtype=wp.float32),
):
    """Reset trainer-owned timers after an accepted selective kickoff."""

    env = wp.tid()
    if reset_mask[env] != 0:
        episode_ticks[env] = 0
        no_touch_ticks[env] = 0
        kickoff_indicator[env] = 1
        car_base = env * 2
        for local_car in range(2):
            car = car_base + local_car
            touch_count[car] = 0
            touch_contact_latched[car] = 0
            episode_player_touched[car] = 0
            first_contact_count[car] = 0
            demo_by_count[car] = 0
            demoed_event[car] = 0
            action_base = car * 8
            for channel in range(8):
                previous_action[action_base + channel] = 0.0


@wp.kernel(enable_backward=False)
def rival2_interval_reset(
    reset_mask: wp.array(dtype=wp.int32),
    episode_tick: wp.array(dtype=wp.int32),
    kickoff_reset: wp.array(dtype=wp.int32),
    kickoff_layout: wp.array(dtype=wp.int32),
    kickoff_selector: wp.array(dtype=wp.int32),
    reset_required: wp.array(dtype=wp.int32),
    ball_scored_last: wp.array(dtype=wp.int32),
    demo_respawn_timer: wp.array(dtype=wp.float32),
    demo_held_valid: wp.array(dtype=wp.int32),
    demo_request: wp.array(dtype=wp.int32),
    respawn_pending: wp.array(dtype=wp.int32),
    car_is_demoed: wp.array(dtype=wp.int32),
    car_contact_id: wp.array(dtype=wp.int32),
    car_contact_cooldown: wp.array(dtype=wp.float32),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    boost: wp.array(dtype=wp.float32),
    boosting_time: wp.array(dtype=wp.float32),
    time_since_boosted: wp.array(dtype=wp.float32),
    on_ground: wp.array(dtype=wp.int32),
    air_control_disabled: wp.array(dtype=wp.int32),
    has_jumped: wp.array(dtype=wp.int32),
    is_jumping: wp.array(dtype=wp.int32),
    has_double_jumped: wp.array(dtype=wp.int32),
    has_flipped: wp.array(dtype=wp.int32),
    is_flipping: wp.array(dtype=wp.int32),
    sticky_ticks: wp.array(dtype=wp.int32),
    jump_time: wp.array(dtype=wp.float32),
    air_time: wp.array(dtype=wp.float32),
    air_time_since_jump: wp.array(dtype=wp.float32),
    flip_time: wp.array(dtype=wp.float32),
    flip_rel_torque: wp.array(dtype=wp.vec3),
    auto_flip_timer: wp.array(dtype=wp.float32),
    auto_flip_torque_scale: wp.array(dtype=wp.float32),
    is_auto_flipping: wp.array(dtype=wp.int32),
    is_boosting: wp.array(dtype=wp.int32),
    is_supersonic: wp.array(dtype=wp.int32),
    supersonic_time: wp.array(dtype=wp.float32),
    prev_throttle: wp.array(dtype=wp.float32),
    prev_steer: wp.array(dtype=wp.float32),
    prev_pitch: wp.array(dtype=wp.float32),
    prev_yaw: wp.array(dtype=wp.float32),
    prev_roll: wp.array(dtype=wp.float32),
    prev_jump: wp.array(dtype=wp.int32),
    prev_boost: wp.array(dtype=wp.int32),
    prev_handbrake: wp.array(dtype=wp.int32),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_quat: wp.array(dtype=wp.quat),
    ball_ang_vel: wp.array(dtype=wp.vec3),
    ball_position_bt: wp.array(dtype=wp.vec3),
    ball_velocity_bt: wp.array(dtype=wp.vec3),
    rigid_position_bt: wp.array(dtype=wp.vec3),
    rigid_velocity_bt: wp.array(dtype=wp.vec3),
    solver_position: wp.array(dtype=wp.vec3),
    solver_orientation: wp.array(dtype=wp.quat),
    solver_velocity: wp.array(dtype=wp.vec3),
    solver_angular_velocity: wp.array(dtype=wp.vec3),
    pad_cooldown: wp.array(dtype=wp.float32),
    pad_previous_locked_car: wp.array(dtype=wp.int32),
):
    """Apply the accepted v0.4 deterministic kickoff writes to selected worlds."""

    env = wp.tid()
    if reset_mask[env] == 0:
        return
    car_base = env * 2
    pad_base = env * PAD_COUNT
    layout = kickoff_selector[env]
    kickoff_selector[env] = (layout + 1) % 5
    kickoff_layout[env] = layout
    kickoff_reset[env] = 1
    reset_required[env] = 0
    episode_tick[env] = 0
    ball_scored_last[env] = 0

    for local_car in range(2):
        car = car_base + local_car
        position = _kickoff_position(layout, local_car)
        position_bt = position * 0.02
        position = position_bt * 50.0
        quat = _kickoff_quaternion(layout, local_car)
        car_pos[car] = position
        car_vel[car] = wp.vec3(0.0)
        car_quat[car] = quat
        car_ang_vel[car] = wp.vec3(0.0)
        boost[car] = CAR_SPAWN_BOOST
        boosting_time[car] = 0.0
        time_since_boosted[car] = 0.0
        on_ground[car] = 1
        air_control_disabled[car] = 0
        has_jumped[car] = 0
        is_jumping[car] = 0
        has_double_jumped[car] = 0
        has_flipped[car] = 0
        is_flipping[car] = 0
        sticky_ticks[car] = 0
        jump_time[car] = 0.0
        air_time[car] = 0.0
        air_time_since_jump[car] = 0.0
        flip_time[car] = 0.0
        flip_rel_torque[car] = wp.vec3(0.0)
        auto_flip_timer[car] = 0.0
        auto_flip_torque_scale[car] = 0.0
        is_auto_flipping[car] = 0
        is_boosting[car] = 0
        is_supersonic[car] = 0
        supersonic_time[car] = 0.0
        prev_throttle[car] = 0.0
        prev_steer[car] = 0.0
        prev_pitch[car] = 0.0
        prev_yaw[car] = 0.0
        prev_roll[car] = 0.0
        prev_jump[car] = 0
        prev_boost[car] = 0
        prev_handbrake[car] = 0
        car_contact_id[car] = -1
        car_contact_cooldown[car] = 0.0
        car_is_demoed[car] = 0
        demo_respawn_timer[car] = 0.0
        demo_held_valid[car] = 0
        demo_request[car] = 0
        respawn_pending[car] = 0
        rigid_position_bt[car] = position_bt
        rigid_velocity_bt[car] = wp.vec3(0.0)
        solver_position[car] = position
        solver_orientation[car] = quat
        solver_velocity[car] = wp.vec3(0.0)
        solver_angular_velocity[car] = wp.vec3(0.0)

    reset_ball_bt = wp.vec3(0.0, 0.0, BALL_REST_Z * 0.02)
    ball_position_bt[env] = reset_ball_bt
    ball_velocity_bt[env] = wp.vec3(0.0)
    ball_pos[env] = reset_ball_bt * 50.0
    ball_vel[env] = wp.vec3(0.0)
    ball_quat[env] = wp.quat_identity()
    ball_ang_vel[env] = wp.vec3(0.0)
    for pad in range(PAD_COUNT):
        index = pad_base + pad
        pad_cooldown[index] = 0.0
        pad_previous_locked_car[index] = 0


__all__ = [
    "EPISODE_LIMIT_TICKS",
    "NO_TOUCH_TIMEOUT_TICKS",
    "PHYSICS_TICKS_PER_DECISION",
    "REWARD_MODE_ACQUISITION",
    "REWARD_MODE_BASE",
    "REWARD_MODE_GOAL_ONLY",
    "rival2_accumulate_tick",
    "rival2_after_interval_reset",
    "rival2_begin_decision",
    "rival2_interval_reset",
]
