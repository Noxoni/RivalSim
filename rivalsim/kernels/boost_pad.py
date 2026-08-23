"""GPU-resident RocketSim Soccar boost-pad state and collision semantics."""

from __future__ import annotations

import numpy as np
import warp as wp

DT = 1.0 / 120.0
PAD_COUNT = 34
BIG_PAD_COUNT = 6

# RocketSim RLConst::BoostPads order: six full pads, then 28 small pads.
SOCCAR_PAD_POSITIONS = np.asarray(
    (
        (-3584.0, 0.0, 73.0),
        (3584.0, 0.0, 73.0),
        (-3072.0, 4096.0, 73.0),
        (3072.0, 4096.0, 73.0),
        (-3072.0, -4096.0, 73.0),
        (3072.0, -4096.0, 73.0),
        (0.0, -4240.0, 70.0),
        (-1792.0, -4184.0, 70.0),
        (1792.0, -4184.0, 70.0),
        (-940.0, -3308.0, 70.0),
        (940.0, -3308.0, 70.0),
        (0.0, -2816.0, 70.0),
        (-3584.0, -2484.0, 70.0),
        (3584.0, -2484.0, 70.0),
        (-1788.0, -2300.0, 70.0),
        (1788.0, -2300.0, 70.0),
        (-2048.0, -1036.0, 70.0),
        (0.0, -1024.0, 70.0),
        (2048.0, -1036.0, 70.0),
        (-1024.0, 0.0, 70.0),
        (1024.0, 0.0, 70.0),
        (-2048.0, 1036.0, 70.0),
        (0.0, 1024.0, 70.0),
        (2048.0, 1036.0, 70.0),
        (-1788.0, 2300.0, 70.0),
        (1788.0, 2300.0, 70.0),
        (-3584.0, 2484.0, 70.0),
        (3584.0, 2484.0, 70.0),
        (0.0, 2816.0, 70.0),
        (-940.0, 3308.0, 70.0),
        (940.0, 3308.0, 70.0),
        (-1792.0, 4184.0, 70.0),
        (1792.0, 4184.0, 70.0),
        (0.0, 4240.0, 70.0),
    ),
    dtype=np.float32,
)

HITBOX_COLLISION_HALF = wp.vec3(
    60.18645668029785,
    43.28265380859375,
    19.26250457763672,
)
HITBOX_OFFSET = wp.vec3(13.875699996948242, 0.0, 20.7549991607666)


@wp.func
def _pad_overlap_locked_car(
    pos: wp.vec3,
    quat: wp.quat,
    pad_pos: wp.vec3,
    pad_radius: float,
) -> bool:
    """Match BoostPad's strict AABB test for its previously locked car."""

    center = pos + wp.quat_rotate(quat, HITBOX_OFFSET)
    axis_x = wp.quat_rotate(quat, wp.vec3(1.0, 0.0, 0.0))
    axis_y = wp.quat_rotate(quat, wp.vec3(0.0, 1.0, 0.0))
    axis_z = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))
    half = wp.vec3(
        wp.abs(axis_x[0]) * HITBOX_COLLISION_HALF[0]
        + wp.abs(axis_y[0]) * HITBOX_COLLISION_HALF[1]
        + wp.abs(axis_z[0]) * HITBOX_COLLISION_HALF[2],
        wp.abs(axis_x[1]) * HITBOX_COLLISION_HALF[0]
        + wp.abs(axis_y[1]) * HITBOX_COLLISION_HALF[1]
        + wp.abs(axis_z[1]) * HITBOX_COLLISION_HALF[2],
        wp.abs(axis_x[2]) * HITBOX_COLLISION_HALF[0]
        + wp.abs(axis_y[2]) * HITBOX_COLLISION_HALF[1]
        + wp.abs(axis_z[2]) * HITBOX_COLLISION_HALF[2],
    )
    car_min = center - half
    car_max = center + half
    pad_min = wp.vec3(pad_pos[0] - pad_radius, pad_pos[1] - pad_radius, pad_pos[2])
    pad_max = wp.vec3(
        pad_pos[0] + pad_radius,
        pad_pos[1] + pad_radius,
        pad_pos[2] + 64.0,
    )
    return (
        pad_max[0] > car_min[0]
        and pad_min[0] < car_max[0]
        and pad_max[1] > car_min[1]
        and pad_min[1] < car_max[1]
        and pad_max[2] > car_min[2]
        and pad_min[2] < car_max[2]
    )


@wp.kernel
def boost_pad_tick(
    pad_positions: wp.array(dtype=wp.vec3),
    car_pos: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_boost: wp.array(dtype=wp.float32),
    cooldown: wp.array(dtype=wp.float32),
    previous_locked_car: wp.array(dtype=wp.int32),
):
    """Run pad pre-tick, two-car collision checks, then post-tick pickup."""

    env = wp.tid()
    pad_base = env * PAD_COUNT
    car_base = env * 2
    for pad in range(PAD_COUNT):
        index = pad_base + pad
        pad_pos = pad_positions[pad]
        is_big = pad < BIG_PAD_COUNT
        current_cooldown = cooldown[index]
        if current_cooldown > 0.0:
            current_cooldown = wp.max(0.0, current_cooldown - DT)
        active = current_cooldown == 0.0
        locked_car = wp.int32(0)

        cylinder_radius = 144.0
        box_radius = 120.0
        if is_big:
            cylinder_radius = 208.0
            box_radius = 160.0

        for local_car in range(2):
            car = car_base + local_car
            car_id = wp.int32(local_car + 1)
            pos = car_pos[car]
            colliding = False
            if previous_locked_car[index] == car_id:
                colliding = _pad_overlap_locked_car(
                    pos,
                    car_quat[car],
                    pad_pos,
                    box_radius,
                )
            else:
                delta_x = pos[0] - pad_pos[0]
                delta_y = pos[1] - pad_pos[1]
                colliding = (
                    delta_x * delta_x + delta_y * delta_y
                    < cylinder_radius * cylinder_radius
                    and wp.abs(pos[2] - pad_pos[2]) < 95.0
                )
            if colliding:
                locked_car = car_id

        if locked_car != 0 and active:
            car = car_base + locked_car - 1
            if is_big:
                car_boost[car] = 100.0
                current_cooldown = 10.0
            else:
                car_boost[car] = wp.min(100.0, car_boost[car] + 12.0)
                current_cooldown = 4.0

        cooldown[index] = current_cooldown
        previous_locked_car[index] = locked_car
