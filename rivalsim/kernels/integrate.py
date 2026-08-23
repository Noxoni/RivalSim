"""Fused one-thread-per-car Warp kernel for one v0.1 physics tick.

The jump, dodge, boost, and air-torque ordering follows RocketSim Car.cpp at
the pinned source revisions listed in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import warp as wp

from rivalsim import constants as c

DT = float(c.DT)
GRAVITY_Z = float(c.GRAVITY_Z)
CAR_MAX_SPEED = float(c.CAR_MAX_SPEED)
CAR_MAX_ANG_SPEED = float(c.CAR_MAX_ANG_SPEED)
BALL_MAX_SPEED = float(c.BALL_MAX_SPEED)
BALL_MAX_ANG_SPEED = float(c.BALL_MAX_ANG_SPEED)
BALL_DRAG_FACTOR = float(c.BALL_DRAG_FACTOR)
BOOST_USED_PER_SECOND = float(c.BOOST_USED_PER_SECOND)
BOOST_MIN_TIME = float(c.BOOST_MIN_TIME)
BOOST_ACCEL_AIR = float(c.BOOST_ACCEL_AIR)
THROTTLE_AIR_ACCEL = float(c.THROTTLE_AIR_ACCEL)
JUMP_ACCEL = float(c.JUMP_ACCEL)
JUMP_IMMEDIATE_FORCE = float(c.JUMP_IMMEDIATE_FORCE)
JUMP_MIN_TIME = float(c.JUMP_MIN_TIME)
JUMP_RESET_TIME_PAD = float(c.JUMP_RESET_TIME_PAD)
JUMP_MAX_TIME = float(c.JUMP_MAX_TIME)
JUMP_PRE_MIN_ACCEL_SCALE = float(c.JUMP_PRE_MIN_ACCEL_SCALE)
JUMP_STICKY_ACCEL = float(c.JUMP_STICKY_ACCEL)
JUMP_STICKY_TICKS = int(c.JUMP_STICKY_TICKS)
DOUBLEJUMP_MAX_DELAY = float(c.DOUBLEJUMP_MAX_DELAY)
DODGE_DEADZONE = float(c.DODGE_DEADZONE)
FLIP_Z_DAMP_120 = float(c.FLIP_Z_DAMP_120)
FLIP_Z_DAMP_START = float(c.FLIP_Z_DAMP_START)
FLIP_Z_DAMP_END = float(c.FLIP_Z_DAMP_END)
FLIP_TORQUE_TIME = float(c.FLIP_TORQUE_TIME)
FLIP_PITCHLOCK_EXTRA_TIME = float(c.FLIP_PITCHLOCK_EXTRA_TIME)
FLIP_INITIAL_VEL_SCALE = float(c.FLIP_INITIAL_VEL_SCALE)
FLIP_TORQUE_X = float(c.FLIP_TORQUE_X)
FLIP_TORQUE_Y = float(c.FLIP_TORQUE_Y)
FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE = float(c.FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE)
FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE = float(c.FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE)
FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE = float(c.FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE)
FLIP_BACKWARD_IMPULSE_SCALE_X = float(c.FLIP_BACKWARD_IMPULSE_SCALE_X)
CAR_TORQUE_SCALE = float(c.CAR_TORQUE_SCALE)
SUPERSONIC_START_SPEED = float(c.SUPERSONIC_START_SPEED)
SUPERSONIC_MAINTAIN_MIN_SPEED = float(c.SUPERSONIC_MAINTAIN_MIN_SPEED)
SUPERSONIC_MAINTAIN_MAX_TIME = float(c.SUPERSONIC_MAINTAIN_MAX_TIME)


@wp.func
def _clamp_unit(value: float) -> float:
    return wp.max(-1.0, wp.min(1.0, value))


@wp.func
def _cap_vector(value: wp.vec3, maximum: float) -> wp.vec3:
    length_sq = wp.dot(value, value)
    if length_sq > maximum * maximum:
        return value * (maximum / wp.sqrt(length_sq))
    return value


@wp.func
def _integrate_quaternion(quat: wp.quat, ang_vel: wp.vec3) -> wp.quat:
    angle_sq = wp.dot(ang_vel, ang_vel)
    angle = wp.sqrt(angle_sq)
    limited = angle
    max_angle = (0.25 * wp.pi) / DT
    if limited > max_angle:
        limited = max_angle

    scale = 0.0
    if limited < 0.001:
        scale = 0.5 * DT - DT * DT * DT * 0.020833333333 * limited * limited
    else:
        scale = wp.sin(0.5 * limited * DT) / angle

    delta = wp.quat(
        ang_vel[0] * scale,
        ang_vel[1] * scale,
        ang_vel[2] * scale,
        wp.cos(0.5 * limited * DT),
    )
    result = wp.mul(delta, quat)
    norm = wp.sqrt(
        result[0] * result[0]
        + result[1] * result[1]
        + result[2] * result[2]
        + result[3] * result[3]
    )
    if norm > 1.0e-20:
        inv = 1.0 / norm
        return wp.quat(result[0] * inv, result[1] * inv, result[2] * inv, result[3] * inv)
    return quat


@wp.kernel
def integrate_tick(
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
    control_throttle: wp.array(dtype=wp.float32),
    control_steer: wp.array(dtype=wp.float32),
    control_pitch: wp.array(dtype=wp.float32),
    control_yaw: wp.array(dtype=wp.float32),
    control_roll: wp.array(dtype=wp.float32),
    control_jump: wp.array(dtype=wp.int32),
    control_boost: wp.array(dtype=wp.int32),
    control_handbrake: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    env = tid // 2

    pos = car_pos[tid]
    vel = car_vel[tid]
    quat = car_quat[tid]
    ang_vel = car_ang_vel[tid]
    car_boost = boost[tid]
    car_boosting_time = boosting_time[tid]
    car_time_since_boosted = time_since_boosted[tid]
    car_on_ground = on_ground[tid]
    car_has_jumped = has_jumped[tid]
    car_is_jumping = is_jumping[tid]
    car_has_double_jumped = has_double_jumped[tid]
    car_has_flipped = has_flipped[tid]
    car_is_flipping = is_flipping[tid]
    car_sticky_ticks = sticky_ticks[tid]
    car_jump_time = jump_time[tid]
    car_air_time = air_time[tid]
    car_air_since_jump = air_time_since_jump[tid]
    car_flip_time = flip_time[tid]
    car_flip_rel_torque = flip_rel_torque[tid]
    car_is_boosting = is_boosting[tid]
    car_is_supersonic = is_supersonic[tid]
    car_supersonic_time = supersonic_time[tid]

    throttle = _clamp_unit(control_throttle[tid])
    steer = _clamp_unit(control_steer[tid])
    pitch = _clamp_unit(control_pitch[tid])
    yaw = _clamp_unit(control_yaw[tid])
    roll = _clamp_unit(control_roll[tid])
    jump = control_jump[tid] != 0
    boost_pressed = control_boost[tid] != 0
    handbrake = control_handbrake[tid] != 0
    jump_pressed = jump and prev_jump[tid] == 0

    forward = wp.quat_rotate(quat, wp.vec3(1.0, 0.0, 0.0))
    right = wp.quat_rotate(quat, wp.vec3(0.0, 1.0, 0.0))
    up = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))
    forward_speed = wp.dot(vel, forward)
    airborne = car_on_ground == 0
    acceleration = wp.vec3(0.0, 0.0, 0.0)
    angular_acceleration = wp.vec3(0.0, 0.0, 0.0)

    # RocketSim _UpdateAirTorque runs before jump and dodge state transitions.
    if car_is_flipping != 0:
        if car_has_flipped != 0 and car_flip_time < FLIP_TORQUE_TIME:
            car_is_flipping = 1
        else:
            car_is_flipping = 0

    flipping = car_is_flipping != 0 and airborne
    rel_nonzero = wp.dot(car_flip_rel_torque, car_flip_rel_torque) > 1.0e-20
    cancel = False
    rel_torque = car_flip_rel_torque
    if flipping and rel_nonzero:
        if rel_torque[1] != 0.0 and pitch != 0.0:
            if (rel_torque[1] > 0.0 and pitch > 0.0) or (rel_torque[1] < 0.0 and pitch < 0.0):
                rel_torque[1] = rel_torque[1] * (1.0 - wp.abs(pitch))
                cancel = True
        dodge_local = wp.vec3(
            rel_torque[0] * FLIP_TORQUE_X,
            rel_torque[1] * FLIP_TORQUE_Y,
            0.0,
        )
        angular_acceleration = angular_acceleration + wp.quat_rotate(quat, dodge_local)

    do_air_control = airborne and ((not flipping) or (not rel_nonzero) or cancel)
    if do_air_control:
        pitch_scale = 1.0
        if flipping:
            pitch_scale = 0.0
        elif car_has_flipped != 0:
            if car_flip_time < FLIP_TORQUE_TIME + FLIP_PITCHLOCK_EXTRA_TIME:
                pitch_scale = 0.0

        dir_pitch = -right
        dir_yaw = up
        dir_roll = -forward
        torque = (
            pitch * dir_pitch * pitch_scale * 130.0 + yaw * dir_yaw * 95.0 + roll * dir_roll * 400.0
        )
        damp_pitch = wp.dot(dir_pitch, ang_vel) * 30.0 * (1.0 - wp.abs(pitch * pitch_scale))
        damp_yaw = wp.dot(dir_yaw, ang_vel) * 20.0 * (1.0 - wp.abs(yaw))
        damp_roll = wp.dot(dir_roll, ang_vel) * 50.0
        damping = dir_pitch * damp_pitch + dir_yaw * damp_yaw + dir_roll * damp_roll
        angular_acceleration = angular_acceleration + (torque - damping) * CAR_TORQUE_SCALE

    if airborne:
        acceleration = acceleration + forward * throttle * THROTTLE_AIR_ACCEL

    on_ground_for_tick = car_on_ground != 0
    if on_ground_for_tick and car_is_jumping == 0:
        if not (car_has_jumped != 0 and car_jump_time < JUMP_MIN_TIME + JUMP_RESET_TIME_PAD):
            car_has_jumped = 0
            car_jump_time = 0.0

    if car_is_jumping != 0:
        if not (car_jump_time < JUMP_MIN_TIME or (jump and car_jump_time < JUMP_MAX_TIME)):
            car_is_jumping = 0
    elif on_ground_for_tick and jump_pressed:
        car_is_jumping = 1
        car_jump_time = 0.0
        vel = vel + up * JUMP_IMMEDIATE_FORCE
        car_sticky_ticks = JUMP_STICKY_TICKS

    started_jump = on_ground_for_tick and jump_pressed and car_is_jumping != 0
    if car_is_jumping != 0:
        car_has_jumped = 1
        jump_scale = 1.0
        if car_jump_time < JUMP_MIN_TIME:
            jump_scale = JUMP_PRE_MIN_ACCEL_SCALE
        acceleration = acceleration + up * JUMP_ACCEL * jump_scale

    if car_sticky_ticks > 0:
        acceleration = acceleration - up * JUMP_STICKY_ACCEL
        car_sticky_ticks = car_sticky_ticks - 1

    if car_is_jumping != 0 or car_has_jumped != 0:
        car_jump_time = car_jump_time + DT

    # RocketSim _UpdateDoubleJumpOrFlip.
    if on_ground_for_tick:
        car_has_double_jumped = 0
        car_has_flipped = 0
        car_air_time = 0.0
        car_air_since_jump = 0.0
        car_flip_time = 0.0
    else:
        car_air_time = car_air_time + DT
        if car_has_jumped != 0 and car_is_jumping == 0:
            car_air_since_jump = car_air_since_jump + DT
        else:
            car_air_since_jump = 0.0

        eligible = (
            jump_pressed
            and car_air_since_jump < DOUBLEJUMP_MAX_DELAY
            and car_has_double_jumped == 0
            and car_has_flipped == 0
        )
        if eligible:
            input_magnitude = wp.abs(yaw) + wp.abs(pitch) + wp.abs(roll)
            if input_magnitude >= DODGE_DEADZONE:
                car_flip_time = 0.0
                car_has_flipped = 1
                car_is_flipping = 1

                dodge_dir = wp.vec3(-pitch, yaw + roll, 0.0)
                if wp.abs(yaw + roll) < 0.1 and wp.abs(pitch) < 0.1:
                    dodge_dir = wp.vec3(0.0, 0.0, 0.0)
                else:
                    dodge_len_sq = wp.dot(dodge_dir, dodge_dir)
                    if dodge_len_sq > 1.0e-20:
                        dodge_dir = dodge_dir / wp.sqrt(dodge_len_sq)

                car_flip_rel_torque = wp.vec3(-dodge_dir[1], dodge_dir[0], 0.0)
                impulse_dir = dodge_dir
                if wp.abs(impulse_dir[0]) < 0.1:
                    impulse_dir[0] = 0.0
                if wp.abs(impulse_dir[1]) < 0.1:
                    impulse_dir[1] = 0.0

                if wp.dot(impulse_dir, impulse_dir) > 1.0e-20:
                    backwards = False
                    if wp.abs(forward_speed) < 100.0:
                        backwards = impulse_dir[0] < 0.0
                    else:
                        backwards = (impulse_dir[0] >= 0.0) != (forward_speed >= 0.0)

                    speed_ratio = wp.abs(forward_speed) / CAR_MAX_SPEED
                    max_x = FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE
                    if backwards:
                        max_x = FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE
                    initial_x = impulse_dir[0] * FLIP_INITIAL_VEL_SCALE
                    initial_y = impulse_dir[1] * FLIP_INITIAL_VEL_SCALE
                    initial_x = initial_x * ((max_x - 1.0) * speed_ratio + 1.0)
                    initial_y = initial_y * (
                        (FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE - 1.0) * speed_ratio + 1.0
                    )
                    if backwards:
                        initial_x = initial_x * FLIP_BACKWARD_IMPULSE_SCALE_X

                    forward_2d = wp.vec3(forward[0], forward[1], 0.0)
                    length_2d = wp.sqrt(wp.dot(forward_2d, forward_2d))
                    if length_2d > 1.0e-20:
                        forward_2d = forward_2d / length_2d
                        right_2d = wp.vec3(-forward_2d[1], forward_2d[0], 0.0)
                        vel = vel + forward_2d * initial_x + right_2d * initial_y
            else:
                vel = vel + up * JUMP_IMMEDIATE_FORCE
                car_has_double_jumped = 1

    if car_is_flipping != 0:
        car_flip_time = car_flip_time + DT
        if car_flip_time <= FLIP_TORQUE_TIME:
            if car_flip_time >= FLIP_Z_DAMP_START and (
                vel[2] < 0.0 or car_flip_time < FLIP_Z_DAMP_END
            ):
                vel[2] = vel[2] * (1.0 - FLIP_Z_DAMP_120)
    elif car_has_flipped != 0:
        car_flip_time = car_flip_time + DT

    if started_jump:
        car_on_ground = 0

    # RocketSim _UpdateBoost, using the v0.1 airborne acceleration.
    has_boost_now = car_boost > 0.0
    if has_boost_now:
        if car_is_boosting != 0:
            if boost_pressed or car_boosting_time < BOOST_MIN_TIME:
                car_is_boosting = 1
            else:
                car_is_boosting = 0
        elif boost_pressed:
            car_is_boosting = 1
    else:
        car_is_boosting = 0

    if car_is_boosting != 0:
        car_boosting_time = car_boosting_time + DT
        car_boost = wp.max(0.0, car_boost - BOOST_USED_PER_SECOND * DT)
        acceleration = acceleration + forward * BOOST_ACCEL_AIR
        car_time_since_boosted = 0.0
    else:
        car_boosting_time = 0.0
        car_time_since_boosted = car_time_since_boosted + DT

    acceleration[2] = acceleration[2] + GRAVITY_Z
    vel = vel + acceleration * DT
    ang_vel = ang_vel + angular_acceleration * DT
    pos = pos + vel * DT
    quat = _integrate_quaternion(quat, ang_vel)

    speed_sq = wp.dot(vel, vel)
    if car_is_supersonic != 0 and car_supersonic_time < SUPERSONIC_MAINTAIN_MAX_TIME:
        if speed_sq >= SUPERSONIC_MAINTAIN_MIN_SPEED * SUPERSONIC_MAINTAIN_MIN_SPEED:
            car_is_supersonic = 1
        else:
            car_is_supersonic = 0
    elif speed_sq >= SUPERSONIC_START_SPEED * SUPERSONIC_START_SPEED:
        car_is_supersonic = 1
    else:
        car_is_supersonic = 0
    if car_is_supersonic != 0:
        car_supersonic_time = car_supersonic_time + DT
    else:
        car_supersonic_time = 0.0

    vel = _cap_vector(vel, CAR_MAX_SPEED)
    ang_vel = _cap_vector(ang_vel, CAR_MAX_ANG_SPEED)

    car_pos[tid] = pos
    car_vel[tid] = vel
    car_quat[tid] = quat
    car_ang_vel[tid] = ang_vel
    boost[tid] = car_boost
    boosting_time[tid] = car_boosting_time
    time_since_boosted[tid] = car_time_since_boosted
    on_ground[tid] = car_on_ground
    has_jumped[tid] = car_has_jumped
    is_jumping[tid] = car_is_jumping
    has_double_jumped[tid] = car_has_double_jumped
    has_flipped[tid] = car_has_flipped
    is_flipping[tid] = car_is_flipping
    sticky_ticks[tid] = car_sticky_ticks
    jump_time[tid] = car_jump_time
    air_time[tid] = car_air_time
    air_time_since_jump[tid] = car_air_since_jump
    flip_time[tid] = car_flip_time
    flip_rel_torque[tid] = car_flip_rel_torque
    is_boosting[tid] = car_is_boosting
    is_supersonic[tid] = car_is_supersonic
    supersonic_time[tid] = car_supersonic_time
    prev_throttle[tid] = throttle
    prev_steer[tid] = steer
    prev_pitch[tid] = pitch
    prev_yaw[tid] = yaw
    prev_roll[tid] = roll
    prev_jump[tid] = wp.int32(jump)
    prev_boost[tid] = wp.int32(boost_pressed)
    prev_handbrake[tid] = wp.int32(handbrake)

    # The even car thread advances its world's one free ball.
    if tid % 2 == 0:
        bpos = ball_pos[env]
        bvel = ball_vel[env] * BALL_DRAG_FACTOR
        bquat = ball_quat[env]
        bang = ball_ang_vel[env]
        bvel[2] = bvel[2] + GRAVITY_Z * DT
        bpos = bpos + bvel * DT
        bquat = _integrate_quaternion(bquat, bang)
        bvel = _cap_vector(bvel, BALL_MAX_SPEED)
        bang = _cap_vector(bang, BALL_MAX_ANG_SPEED)
        ball_pos[env] = bpos
        ball_vel[env] = bvel
        ball_quat[env] = bquat
        ball_ang_vel[env] = bang
