"""Vectorized CPU implementation of the exact RivalSim v0.1 equations."""

from __future__ import annotations

import numpy as np

from rivalsim import constants as c
from rivalsim.controls import ControlBatch
from rivalsim.math import cap_vectors, integrate_quaternion, quat_rotate
from rivalsim.state import StateSnapshot

X_AXIS = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
Y_AXIS = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
Z_AXIS = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)


class CpuSimulator:
    """NumPy batch simulator used for same-equation correctness and throughput."""

    def __init__(self, state: StateSnapshot, controls: ControlBatch | None = None):
        state.validate()
        self.state = state.copy()
        self.controls = (controls or ControlBatch.zeros(state.num_envs)).clamped()
        if self.controls.num_envs != state.num_envs:
            raise ValueError("state/control world counts differ")
        self.tick_count = 0

    @property
    def num_envs(self) -> int:
        return self.state.num_envs

    def set_controls(self, controls: ControlBatch) -> None:
        controls.validate()
        if controls.num_envs != self.num_envs:
            raise ValueError("control world count differs")
        self.controls = controls.clamped()

    def reset(self, state: StateSnapshot) -> None:
        if state.num_envs != self.num_envs:
            raise ValueError("reset world count differs")
        state.validate()
        self.state = state.copy()
        self.tick_count = 0

    def step(self, ticks: int = 1) -> None:
        if ticks < 0:
            raise ValueError("ticks must be non-negative")
        for _ in range(ticks):
            self._tick()
        self.tick_count += ticks

    def snapshot(self) -> StateSnapshot:
        return self.state.copy()

    def _tick(self) -> None:
        state = self.state
        controls = self.controls
        dt = c.DT

        throttle = np.clip(controls.throttle, -1.0, 1.0).astype(np.float32)
        steer = np.clip(controls.steer, -1.0, 1.0).astype(np.float32)
        pitch = np.clip(controls.pitch, -1.0, 1.0).astype(np.float32)
        yaw = np.clip(controls.yaw, -1.0, 1.0).astype(np.float32)
        roll = np.clip(controls.roll, -1.0, 1.0).astype(np.float32)
        jump = controls.jump != 0
        boost_input = controls.boost != 0
        handbrake = controls.handbrake != 0

        forward = quat_rotate(state.car_quat, X_AXIS)
        right = quat_rotate(state.car_quat, Y_AXIS)
        up = quat_rotate(state.car_quat, Z_AXIS)
        forward_speed = np.sum(state.car_vel * forward, axis=-1)
        airborne = state.on_ground == 0
        acceleration = np.zeros_like(state.car_vel)
        angular_acceleration = np.zeros_like(state.car_ang_vel)

        # RocketSim _UpdateAirTorque executes before jump/dodge state updates.
        was_flipping = state.is_flipping != 0
        state.is_flipping[was_flipping] = (state.has_flipped[was_flipping] != 0) & (
            state.flip_time[was_flipping] < c.FLIP_TORQUE_TIME
        )
        flipping = (state.is_flipping != 0) & airborne
        rel_torque = state.flip_rel_torque.copy()
        rel_nonzero = np.sum(rel_torque * rel_torque, axis=-1) > np.float32(1e-20)
        cancel = (
            flipping
            & rel_nonzero
            & (rel_torque[..., 1] != 0.0)
            & (pitch != 0.0)
            & (np.sign(rel_torque[..., 1]) == np.sign(pitch))
        )
        rel_torque[..., 1] *= np.where(cancel, 1.0 - np.abs(pitch), 1.0).astype(np.float32)
        dodge_local = np.zeros_like(rel_torque)
        dodge_local[..., 0] = rel_torque[..., 0] * c.FLIP_TORQUE_X
        dodge_local[..., 1] = rel_torque[..., 1] * c.FLIP_TORQUE_Y
        dodge_mask = flipping & rel_nonzero
        if np.any(dodge_mask):
            dodge_world = quat_rotate(state.car_quat, dodge_local)
            angular_acceleration[dodge_mask] += dodge_world[dodge_mask]

        do_air_control = airborne & ((~flipping) | (~rel_nonzero) | cancel)
        pitch_scale = np.ones_like(pitch)
        pitch_scale[flipping] = 0.0
        post_flip_lock = (
            (state.has_flipped != 0)
            & (~flipping)
            & (state.flip_time < c.FLIP_TORQUE_TIME + c.FLIP_PITCHLOCK_EXTRA_TIME)
        )
        pitch_scale[post_flip_lock] = 0.0

        dir_pitch = -right
        dir_yaw = up
        dir_roll = -forward
        torque = (
            pitch[..., None] * dir_pitch * pitch_scale[..., None] * c.CAR_AIR_CONTROL_TORQUE[0]
            + yaw[..., None] * dir_yaw * c.CAR_AIR_CONTROL_TORQUE[1]
            + roll[..., None] * dir_roll * c.CAR_AIR_CONTROL_TORQUE[2]
        )
        ang_vel = state.car_ang_vel
        damp_pitch = (
            np.sum(dir_pitch * ang_vel, axis=-1)
            * c.CAR_AIR_CONTROL_DAMPING[0]
            * (1.0 - np.abs(pitch * pitch_scale))
        )
        damp_yaw = (
            np.sum(dir_yaw * ang_vel, axis=-1) * c.CAR_AIR_CONTROL_DAMPING[1] * (1.0 - np.abs(yaw))
        )
        damp_roll = np.sum(dir_roll * ang_vel, axis=-1) * c.CAR_AIR_CONTROL_DAMPING[2]
        damping = (
            dir_pitch * damp_pitch[..., None]
            + dir_yaw * damp_yaw[..., None]
            + dir_roll * damp_roll[..., None]
        )
        if np.any(do_air_control):
            angular_acceleration[do_air_control] += (torque - damping)[
                do_air_control
            ] * c.CAR_TORQUE_SCALE
        acceleration[airborne] += (forward * throttle[..., None] * c.THROTTLE_AIR_ACCEL)[airborne]

        jump_pressed = jump & (state.prev_jump == 0)
        on_ground_for_tick = state.on_ground != 0
        ground_not_jumping = on_ground_for_tick & (state.is_jumping == 0)
        reset_jump = ground_not_jumping & ~(
            (state.has_jumped != 0) & (state.jump_time < c.JUMP_MIN_TIME + c.JUMP_RESET_TIME_PAD)
        )
        state.has_jumped[reset_jump] = 0
        state.jump_time[reset_jump] = 0.0

        jumping = state.is_jumping != 0
        stop_jump = jumping & ~(
            (state.jump_time < c.JUMP_MIN_TIME) | (jump & (state.jump_time < c.JUMP_MAX_TIME))
        )
        state.is_jumping[stop_jump] = 0
        start_jump = (state.is_jumping == 0) & on_ground_for_tick & jump_pressed
        state.is_jumping[start_jump] = 1
        state.jump_time[start_jump] = 0.0
        state.car_vel[start_jump] += (up * c.JUMP_IMMEDIATE_FORCE)[start_jump]
        state.sticky_ticks[start_jump] = c.JUMP_STICKY_TICKS

        jumping = state.is_jumping != 0
        state.has_jumped[jumping] = 1
        jump_scale = np.where(
            state.jump_time < c.JUMP_MIN_TIME, c.JUMP_PRE_MIN_ACCEL_SCALE, np.float32(1.0)
        )
        acceleration[jumping] += (up * c.JUMP_ACCEL * jump_scale[..., None])[jumping]
        sticky = state.sticky_ticks > 0
        acceleration[sticky] -= (up * c.JUMP_STICKY_ACCEL)[sticky]
        state.sticky_ticks[sticky] -= 1
        jump_clock = (state.is_jumping != 0) | (state.has_jumped != 0)
        state.jump_time[jump_clock] += dt

        # RocketSim _UpdateDoubleJumpOrFlip.
        state.has_double_jumped[on_ground_for_tick] = 0
        state.has_flipped[on_ground_for_tick] = 0
        state.air_time[on_ground_for_tick] = 0.0
        state.air_time_since_jump[on_ground_for_tick] = 0.0
        state.flip_time[on_ground_for_tick] = 0.0
        in_air = ~on_ground_for_tick
        state.air_time[in_air] += dt
        since_jump = in_air & (state.has_jumped != 0) & (state.is_jumping == 0)
        state.air_time_since_jump[since_jump] += dt
        state.air_time_since_jump[in_air & ~since_jump] = 0.0

        eligible = (
            in_air
            & jump_pressed
            & (state.air_time_since_jump < c.DOUBLEJUMP_MAX_DELAY)
            & (state.has_double_jumped == 0)
            & (state.has_flipped == 0)
        )
        input_magnitude = np.abs(yaw) + np.abs(pitch) + np.abs(roll)
        flip_start = eligible & (input_magnitude >= c.DODGE_DEADZONE)
        double_start = eligible & ~flip_start
        if np.any(flip_start):
            state.flip_time[flip_start] = 0.0
            state.has_flipped[flip_start] = 1
            state.is_flipping[flip_start] = 1
            dodge_dir = np.zeros_like(state.car_vel)
            dodge_dir[..., 0] = -pitch
            dodge_dir[..., 1] = yaw + roll
            dodge_norm = np.linalg.norm(dodge_dir, axis=-1)
            normalize = (np.abs(yaw + roll) >= 0.1) | (np.abs(pitch) >= 0.1)
            safe = normalize & (dodge_norm > np.float32(1e-20))
            dodge_dir[safe] /= dodge_norm[safe, None]
            dodge_dir[~normalize] = 0.0
            state.flip_rel_torque[..., 0][flip_start] = -dodge_dir[..., 1][flip_start]
            state.flip_rel_torque[..., 1][flip_start] = dodge_dir[..., 0][flip_start]
            state.flip_rel_torque[..., 2][flip_start] = 0.0

            impulse_dir = dodge_dir.copy()
            impulse_dir[..., 0][np.abs(impulse_dir[..., 0]) < 0.1] = 0.0
            impulse_dir[..., 1][np.abs(impulse_dir[..., 1]) < 0.1] = 0.0
            active_impulse = flip_start & (
                np.sum(impulse_dir * impulse_dir, axis=-1) > np.float32(1e-20)
            )
            speed_ratio = np.abs(forward_speed) / c.CAR_MAX_SPEED
            backwards = np.where(
                np.abs(forward_speed) < 100.0,
                impulse_dir[..., 0] < 0.0,
                (impulse_dir[..., 0] >= 0.0) != (forward_speed >= 0.0),
            )
            max_x = np.where(
                backwards,
                c.FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE,
                c.FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE,
            )
            initial = impulse_dir * c.FLIP_INITIAL_VEL_SCALE
            initial[..., 0] *= (max_x - 1.0) * speed_ratio + 1.0
            initial[..., 1] *= (c.FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE - 1.0) * speed_ratio + 1.0
            initial[..., 0] *= np.where(backwards, c.FLIP_BACKWARD_IMPULSE_SCALE_X, np.float32(1.0))
            forward_2d = forward.copy()
            forward_2d[..., 2] = 0.0
            forward_2d_norm = np.linalg.norm(forward_2d, axis=-1)
            valid_forward = forward_2d_norm > np.float32(1e-20)
            forward_2d[valid_forward] /= forward_2d_norm[valid_forward, None]
            forward_2d[~valid_forward] = 0.0
            right_2d = np.zeros_like(forward_2d)
            right_2d[..., 0] = -forward_2d[..., 1]
            right_2d[..., 1] = forward_2d[..., 0]
            delta = initial[..., 0, None] * forward_2d + initial[..., 1, None] * right_2d
            state.car_vel[active_impulse] += delta[active_impulse]

        state.car_vel[double_start] += (up * c.JUMP_IMMEDIATE_FORCE)[double_start]
        state.has_double_jumped[double_start] = 1

        flipping_now = state.is_flipping != 0
        state.flip_time[flipping_now] += dt
        damp_z = (
            flipping_now
            & (state.flip_time <= c.FLIP_TORQUE_TIME)
            & (state.flip_time >= c.FLIP_Z_DAMP_START)
            & ((state.car_vel[..., 2] < 0.0) | (state.flip_time < c.FLIP_Z_DAMP_END))
        )
        state.car_vel[..., 2][damp_z] *= np.float32(1.0) - c.FLIP_Z_DAMP_120
        after_flip = (state.is_flipping == 0) & (state.has_flipped != 0)
        state.flip_time[after_flip] += dt
        state.on_ground[start_jump] = 0

        # RocketSim _UpdateBoost.
        has_boost = state.boost > 0.0
        was_boosting = state.is_boosting != 0
        keep_boosting = was_boosting & (boost_input | (state.boosting_time < c.BOOST_MIN_TIME))
        start_boosting = (~was_boosting) & boost_input
        state.is_boosting[...] = has_boost & (keep_boosting | start_boosting)
        boosting = state.is_boosting != 0
        state.boosting_time[boosting] += dt
        state.boosting_time[~boosting] = 0.0
        state.boost[boosting] = np.maximum(
            state.boost[boosting] - c.BOOST_USED_PER_SECOND * dt, 0.0
        )
        acceleration[boosting] += (forward * c.BOOST_ACCEL_AIR)[boosting]
        state.time_since_boosted[boosting] = 0.0
        state.time_since_boosted[~boosting] += dt

        acceleration[..., 2] += c.GRAVITY_Z
        state.car_vel += acceleration * dt
        state.car_ang_vel += angular_acceleration * dt
        state.car_pos += state.car_vel * dt
        state.car_quat[...] = integrate_quaternion(state.car_quat, state.car_ang_vel, dt)

        # Bullet linear damping, force integration, transform integration, then RocketSim caps.
        state.ball_vel *= c.BALL_DRAG_FACTOR
        state.ball_vel[..., 2] += c.GRAVITY_Z * dt
        state.ball_pos += state.ball_vel * dt
        state.ball_quat[...] = integrate_quaternion(state.ball_quat, state.ball_ang_vel, dt)

        speed_sq = np.sum(state.car_vel * state.car_vel, axis=-1)
        maintaining = (state.is_supersonic != 0) & (
            state.supersonic_time < c.SUPERSONIC_MAINTAIN_MAX_TIME
        )
        state.is_supersonic[...] = np.where(
            maintaining,
            speed_sq >= c.SUPERSONIC_MAINTAIN_MIN_SPEED**2,
            speed_sq >= c.SUPERSONIC_START_SPEED**2,
        )
        state.supersonic_time[state.is_supersonic != 0] += dt
        state.supersonic_time[state.is_supersonic == 0] = 0.0

        cap_vectors(state.car_vel, c.CAR_MAX_SPEED)
        cap_vectors(state.car_ang_vel, c.CAR_MAX_ANG_SPEED)
        cap_vectors(state.ball_vel, c.BALL_MAX_SPEED)
        cap_vectors(state.ball_ang_vel, c.BALL_MAX_ANG_SPEED)

        state.prev_throttle[...] = throttle
        state.prev_steer[...] = steer
        state.prev_pitch[...] = pitch
        state.prev_yaw[...] = yaw
        state.prev_roll[...] = roll
        state.prev_jump[...] = jump
        state.prev_boost[...] = boost_input
        state.prev_handbrake[...] = handbrake
