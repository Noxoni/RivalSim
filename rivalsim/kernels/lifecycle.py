"""Source-ordered RocketSim v0.4 lifecycle kernels."""

from __future__ import annotations

import warp as wp

from rivalsim.kernels.boost_pad import BIG_PAD_COUNT, PAD_COUNT

DT = 1.0 / 120.0
DEMO_RESPAWN_TIME = 3.0
BALL_REST_Z = 93.15
BALL_RADIUS = 91.25
GOAL_BASE_THRESHOLD_Y = 5124.25
GOAL_SCORING_PLANE_Y = GOAL_BASE_THRESHOLD_Y + BALL_RADIUS
GOAL_HALF_WIDTH = 892.755
GOAL_HEIGHT = 642.775
CAR_SPAWN_REST_Z = 17.0
CAR_RESPAWN_Z = 36.0
CAR_SPAWN_BOOST = 100.0 / 3.0


@wp.func
def _z_quaternion(yaw: float) -> wp.quat:
    half = yaw * 0.5
    return wp.quat(0.0, 0.0, wp.sin(half), wp.cos(half))


@wp.func
def _kickoff_position(layout: int, orange: int) -> wp.vec3:
    result = wp.vec3(0.0, -4608.0, CAR_SPAWN_REST_Z)
    if layout == 0:
        result = wp.vec3(-2048.0, -2560.0, CAR_SPAWN_REST_Z)
    elif layout == 1:
        result = wp.vec3(2048.0, -2560.0, CAR_SPAWN_REST_Z)
    elif layout == 2:
        result = wp.vec3(-256.0, -3840.0, CAR_SPAWN_REST_Z)
    elif layout == 3:
        result = wp.vec3(256.0, -3840.0, CAR_SPAWN_REST_Z)
    if orange != 0:
        result = wp.vec3(-result[0], -result[1], result[2])
    return result


@wp.func
def _kickoff_quaternion(layout: int, orange: int) -> wp.quat:
    yaw = 1.5707963267948966
    if layout == 0:
        yaw = 0.7853981633974483
    elif layout == 1:
        yaw = 2.356194490192345
    if orange != 0:
        yaw = yaw + 3.141592653589793
    return _z_quaternion(yaw)


@wp.func
def _respawn_position(location: int, orange: int) -> wp.vec3:
    x = -2304.0
    if location == 1:
        x = -2688.0
    elif location == 2:
        x = 2304.0
    elif location == 3:
        x = 2688.0
    y = -4608.0
    if orange != 0:
        y = 4608.0
    return wp.vec3(x, y, CAR_RESPAWN_Z)


@wp.func
def _save_held_float(
    car: int,
    held: wp.array2d(dtype=wp.float32),
    pos: wp.vec3,
    vel: wp.vec3,
    quat: wp.quat,
    ang_vel: wp.vec3,
    boost: float,
    boosting_time: float,
    time_since_boosted: float,
    jump_time: float,
    air_time: float,
    air_time_since_jump: float,
    flip_time: float,
    flip_rel_torque: wp.vec3,
    auto_flip_timer: float,
    auto_flip_torque_scale: float,
    supersonic_time: float,
    prev_throttle: float,
    prev_steer: float,
    prev_pitch: float,
    prev_yaw: float,
    prev_roll: float,
):
    held[car, 0] = pos[0]
    held[car, 1] = pos[1]
    held[car, 2] = pos[2]
    held[car, 3] = vel[0]
    held[car, 4] = vel[1]
    held[car, 5] = vel[2]
    held[car, 6] = quat[0]
    held[car, 7] = quat[1]
    held[car, 8] = quat[2]
    held[car, 9] = quat[3]
    held[car, 10] = ang_vel[0]
    held[car, 11] = ang_vel[1]
    held[car, 12] = ang_vel[2]
    held[car, 13] = boost
    held[car, 14] = boosting_time
    held[car, 15] = time_since_boosted
    held[car, 16] = jump_time
    held[car, 17] = air_time
    held[car, 18] = air_time_since_jump
    held[car, 19] = flip_time
    held[car, 20] = flip_rel_torque[0]
    held[car, 21] = flip_rel_torque[1]
    held[car, 22] = flip_rel_torque[2]
    held[car, 23] = auto_flip_timer
    held[car, 24] = auto_flip_torque_scale
    held[car, 25] = supersonic_time
    held[car, 26] = prev_throttle
    held[car, 27] = prev_steer
    held[car, 28] = prev_pitch
    held[car, 29] = prev_yaw
    held[car, 30] = prev_roll


@wp.func
def _save_held_int(
    car: int,
    held: wp.array2d(dtype=wp.int32),
    on_ground: int,
    has_jumped: int,
    is_jumping: int,
    has_double_jumped: int,
    has_flipped: int,
    is_flipping: int,
    is_auto_flipping: int,
    is_boosting: int,
    sticky_ticks: int,
    is_supersonic: int,
    prev_jump: int,
    prev_boost: int,
    prev_handbrake: int,
):
    held[car, 0] = on_ground
    held[car, 1] = has_jumped
    held[car, 2] = is_jumping
    held[car, 3] = has_double_jumped
    held[car, 4] = has_flipped
    held[car, 5] = is_flipping
    held[car, 6] = is_auto_flipping
    held[car, 7] = is_boosting
    held[car, 8] = sticky_ticks
    held[car, 9] = is_supersonic
    held[car, 10] = prev_jump
    held[car, 11] = prev_boost
    held[car, 12] = prev_handbrake


@wp.kernel(enable_backward=False)
def lifecycle_pre_tick(
    demo_respawn_timer: wp.array(dtype=wp.float32),
    demo_held_valid: wp.array(dtype=wp.int32),
    respawn_pending: wp.array(dtype=wp.int32),
    respawn_event: wp.array(dtype=wp.int32),
    respawn_location: wp.array(dtype=wp.int32),
    respawn_selector: wp.array(dtype=wp.int32),
    car_is_demoed: wp.array(dtype=wp.int32),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    pad_cooldown: wp.array(dtype=wp.float32),
    pad_cooldown_before: wp.array(dtype=wp.float32),
):
    env = wp.tid()
    pad_base = env * PAD_COUNT
    for pad in range(PAD_COUNT):
        pad_cooldown_before[pad_base + pad] = pad_cooldown[pad_base + pad]
    car_base = env * 2
    for local_car in range(2):
        car = car_base + local_car
        respawn_event[car] = 0
        respawn_location[car] = -1
        respawn_pending[car] = 0
        if car_is_demoed[car] != 0:
            timer = wp.max(demo_respawn_timer[car] - DT, 0.0)
            demo_respawn_timer[car] = timer
            if timer == 0.0:
                location = respawn_selector[car]
                respawn_pending[car] = 1
                respawn_event[car] = 1
                respawn_location[car] = location
                respawn_selector[car] = (location + 1) % 4
            # RocketSim disables the rigid body for the complete transition
            # tick. Park only the execution state; post-tick restores the
            # retained public state or writes the selected respawn state.
            park = 1000000.0 + float(car) * 1000.0
            car_pos[car] = wp.vec3(park, park, park)
            car_vel[car] = wp.vec3(0.0)
            car_quat[car] = wp.quat_identity()
            car_ang_vel[car] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def lifecycle_post_tick(
    world_tick: wp.array(dtype=wp.int32),
    episode_tick: wp.array(dtype=wp.int32),
    blue_score: wp.array(dtype=wp.int32),
    orange_score: wp.array(dtype=wp.int32),
    goal_scored: wp.array(dtype=wp.int32),
    scoring_team: wp.array(dtype=wp.int32),
    kickoff_reset: wp.array(dtype=wp.int32),
    kickoff_layout: wp.array(dtype=wp.int32),
    kickoff_selector: wp.array(dtype=wp.int32),
    full_reset: wp.array(dtype=wp.int32),
    reset_required: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    ball_scored_last: wp.array(dtype=wp.int32),
    auto_kickoff: wp.array(dtype=wp.int32),
    full_reset_interval: wp.array(dtype=wp.int32),
    pad_cooldown_before: wp.array(dtype=wp.float32),
    pad_pickup_car: wp.array(dtype=wp.int32),
    pad_reactivated: wp.array(dtype=wp.int32),
    demo_respawn_timer: wp.array(dtype=wp.float32),
    demo_held_valid: wp.array(dtype=wp.int32),
    demo_request: wp.array(dtype=wp.int32),
    respawn_pending: wp.array(dtype=wp.int32),
    respawn_location: wp.array(dtype=wp.int32),
    held_float: wp.array2d(dtype=wp.float32),
    held_int: wp.array2d(dtype=wp.int32),
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
    env = wp.tid()
    car_base = env * 2
    pad_base = env * PAD_COUNT

    # Car::_PreTickUpdate demo timer/disable followed by Car::_PostTickUpdate.
    for local_car in range(2):
        car = car_base + local_car
        if demo_request[car] != 0:
            car_is_demoed[car] = 1
            demo_request[car] = 0
        if respawn_pending[car] != 0:
            location = respawn_location[car]
            position = _respawn_position(location, local_car)
            position_bt = position * 0.02
            position = position_bt * 50.0
            quat = _z_quaternion(1.5707963267948966 + float(local_car) * 3.141592653589793)
            car_pos[car] = position
            car_vel[car] = wp.vec3(0.0)
            car_quat[car] = quat
            car_ang_vel[car] = wp.vec3(0.0)
            boost[car] = CAR_SPAWN_BOOST
            boosting_time[car] = 0.0
            time_since_boosted[car] = 0.0
            on_ground[car] = 0
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
        elif car_is_demoed[car] != 0:
            if demo_held_valid[car] == 0:
                demo_respawn_timer[car] = DEMO_RESPAWN_TIME
                _save_held_float(
                    car,
                    held_float,
                    car_pos[car],
                    car_vel[car],
                    car_quat[car],
                    car_ang_vel[car],
                    boost[car],
                    boosting_time[car],
                    time_since_boosted[car],
                    jump_time[car],
                    air_time[car],
                    air_time_since_jump[car],
                    flip_time[car],
                    flip_rel_torque[car],
                    auto_flip_timer[car],
                    auto_flip_torque_scale[car],
                    supersonic_time[car],
                    prev_throttle[car],
                    prev_steer[car],
                    prev_pitch[car],
                    prev_yaw[car],
                    prev_roll[car],
                )
                _save_held_int(
                    car,
                    held_int,
                    on_ground[car],
                    has_jumped[car],
                    is_jumping[car],
                    has_double_jumped[car],
                    has_flipped[car],
                    is_flipping[car],
                    is_auto_flipping[car],
                    is_boosting[car],
                    sticky_ticks[car],
                    is_supersonic[car],
                    prev_jump[car],
                    prev_boost[car],
                    prev_handbrake[car],
                )
                demo_held_valid[car] = 1
            car_pos[car] = wp.vec3(held_float[car, 0], held_float[car, 1], held_float[car, 2])
            car_vel[car] = wp.vec3(held_float[car, 3], held_float[car, 4], held_float[car, 5])
            car_quat[car] = wp.quat(
                held_float[car, 6],
                held_float[car, 7],
                held_float[car, 8],
                held_float[car, 9],
            )
            car_ang_vel[car] = wp.vec3(
                held_float[car, 10], held_float[car, 11], held_float[car, 12]
            )
            boost[car] = held_float[car, 13]
            boosting_time[car] = held_float[car, 14]
            time_since_boosted[car] = held_float[car, 15]
            jump_time[car] = held_float[car, 16]
            air_time[car] = held_float[car, 17]
            air_time_since_jump[car] = held_float[car, 18]
            flip_time[car] = held_float[car, 19]
            flip_rel_torque[car] = wp.vec3(
                held_float[car, 20], held_float[car, 21], held_float[car, 22]
            )
            auto_flip_timer[car] = held_float[car, 23]
            auto_flip_torque_scale[car] = held_float[car, 24]
            supersonic_time[car] = held_float[car, 25]
            prev_throttle[car] = held_float[car, 26]
            prev_steer[car] = held_float[car, 27]
            prev_pitch[car] = held_float[car, 28]
            prev_yaw[car] = held_float[car, 29]
            prev_roll[car] = held_float[car, 30]
            on_ground[car] = held_int[car, 0]
            has_jumped[car] = held_int[car, 1]
            is_jumping[car] = held_int[car, 2]
            has_double_jumped[car] = held_int[car, 3]
            has_flipped[car] = held_int[car, 4]
            is_flipping[car] = held_int[car, 5]
            is_auto_flipping[car] = held_int[car, 6]
            is_boosting[car] = held_int[car, 7]
            sticky_ticks[car] = held_int[car, 8]
            is_supersonic[car] = held_int[car, 9]
            prev_jump[car] = held_int[car, 10]
            prev_boost[car] = held_int[car, 11]
            prev_handbrake[car] = held_int[car, 12]
            rigid_position_bt[car] = car_pos[car] * 0.02
            rigid_velocity_bt[car] = car_vel[car] * 0.02

    # BoostPad::_PreTickUpdate/_PostTickUpdate event readback.
    for pad in range(PAD_COUNT):
        index = pad_base + pad
        before = pad_cooldown_before[index]
        after = pad_cooldown[index]
        duration = 4.0
        if pad < BIG_PAD_COUNT:
            duration = 10.0
        pickup = wp.int32(0)
        if after == duration and before != duration:
            pickup = pad_previous_locked_car[index]
        pad_pickup_car[index] = pickup
        pad_reactivated[index] = wp.int32(before > 0.0 and after == 0.0 and pickup == 0)

    goal_scored[env] = 0
    scoring_team[env] = -1
    kickoff_reset[env] = 0
    kickoff_layout[env] = -1
    full_reset[env] = 0
    reset_required[env] = 0
    terminated[env] = 0
    truncated[env] = 0

    scored = wp.int32(wp.abs(ball_position_bt[env][1] * 50.0) > GOAL_BASE_THRESHOLD_Y + BALL_RADIUS)
    new_goal = wp.int32(scored != 0 and ball_scored_last[env] == 0)
    ball_scored_last[env] = scored
    if new_goal != 0:
        team = wp.int32(1)
        if ball_position_bt[env][1] > 0.0:
            team = 0
            blue_score[env] = blue_score[env] + 1
        else:
            orange_score[env] = orange_score[env] + 1
        goal_scored[env] = 1
        scoring_team[env] = team
        if auto_kickoff[env] == 0:
            reset_required[env] = 1

    next_world_tick = world_tick[env] + 1
    interval = full_reset_interval[env]
    do_full_reset = wp.int32(interval > 0 and next_world_tick % interval == 0)
    do_kickoff = wp.int32(new_goal != 0 and auto_kickoff[env] != 0)
    if do_full_reset != 0:
        do_kickoff = 1
        full_reset[env] = 1
        blue_score[env] = 0
        orange_score[env] = 0

    if do_kickoff != 0:
        layout = kickoff_selector[env]
        kickoff_selector[env] = (layout + 1) % 5
        kickoff_layout[env] = layout
        kickoff_reset[env] = 1
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
    else:
        episode_tick[env] = episode_tick[env] + 1
    world_tick[env] = next_world_tick


__all__ = ["lifecycle_post_tick", "lifecycle_pre_tick"]
