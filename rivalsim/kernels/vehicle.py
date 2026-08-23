"""RocketSim-derived Octane wheel forces and static-world chassis contacts."""

import warp as wp

DT = 1.0 / 120.0
CAR_MASS = 180.0
INV_MASS = 1.0 / CAR_MASS
INV_INERTIA_LOCAL = wp.vec3(
    1.0 / 135169.68393410725,
    1.0 / 240246.95911884043,
    1.0 / 330580.8636451765,
)
HITBOX_HALF = wp.vec3(120.50700378417969 / 2.0, 86.69940185546875 / 2.0, 38.65909957885742 / 2.0)
HITBOX_OFFSET = wp.vec3(13.875699996948242, 0.0, 20.7549991607666)
MAX_SUSPENSION_TRAVEL = 12.0
SUSPENSION_SUBTRACTION = 0.05
SUSPENSION_STIFFNESS = 500.0
SUSPENSION_DAMPING_COMPRESSION = 25.0
SUSPENSION_DAMPING_RELAXATION = 40.0
SUSPENSION_SCALE_FRONT = 35.75
SUSPENSION_SCALE_BACK = 54.265
POWERSLIDE_RISE_RATE = 5.0
POWERSLIDE_FALL_RATE = 2.0
STOPPING_FORWARD_VEL = 25.0
THROTTLE_DEADZONE = 0.001
COASTING_BRAKE_FACTOR = 0.15
BRAKING_NO_THROTTLE_SPEED_THRESH = 0.01
SOCCAR_EXTENT_X = 4096.0
SOCCAR_HEIGHT = 2048.0
CONTACT_FRICTION = 0.3
CONTACT_RESTITUTION = 0.3
CONTACT_SLOP = 0.02
CONTACT_CORRECTION = 0.7


@wp.func
def _linear(value: float, x0: float, y0: float, x1: float, y1: float) -> float:
    alpha = wp.clamp((value - x0) / (x1 - x0), 0.0, 1.0)
    return y0 + (y1 - y0) * alpha


@wp.func
def _drive_curve(speed: float) -> float:
    if speed <= 1400.0:
        return _linear(speed, 0.0, 1.0, 1400.0, 0.1)
    return _linear(speed, 1400.0, 0.1, 1410.0, 0.0)


@wp.func
def _steer_curve(speed: float) -> float:
    if speed <= 500.0:
        return _linear(speed, 0.0, 0.53356, 500.0, 0.31930)
    if speed <= 1000.0:
        return _linear(speed, 500.0, 0.31930, 1000.0, 0.18203)
    if speed <= 1500.0:
        return _linear(speed, 1000.0, 0.18203, 1500.0, 0.10570)
    if speed <= 1750.0:
        return _linear(speed, 1500.0, 0.10570, 1750.0, 0.08507)
    return _linear(speed, 1750.0, 0.08507, 3000.0, 0.03454)


@wp.func
def _powerslide_steer_curve(speed: float) -> float:
    return _linear(speed, 0.0, 0.39235, 2500.0, 0.12610)


@wp.func
def _non_sticky_curve(normal_z: float) -> float:
    if normal_z <= 0.7075:
        return _linear(normal_z, 0.0, 0.1, 0.7075, 0.5)
    return _linear(normal_z, 0.7075, 0.5, 1.0, 1.0)


@wp.func
def _world_ray(mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, maximum: float) -> wp.vec4:
    """Return distance and outward normal; distance > maximum encodes a miss."""

    nearest = maximum + 1.0
    found_normal = wp.vec3(0.0, 0.0, 0.0)
    query = wp.mesh_query_ray(mesh_id, origin, direction, maximum)
    if query.result:
        nearest = query.t
        found_normal = query.normal
        if wp.dot(found_normal, direction) > 0.0:
            found_normal = -found_normal

    denominator = direction[2]
    if wp.abs(denominator) > 1.0e-8:
        candidate = -origin[2] / denominator
        if candidate >= 0.0 and candidate <= maximum and candidate < nearest:
            nearest = candidate
            found_normal = wp.vec3(0.0, 0.0, 1.0)
        candidate = (SOCCAR_HEIGHT - origin[2]) / denominator
        if candidate >= 0.0 and candidate <= maximum and candidate < nearest:
            nearest = candidate
            found_normal = wp.vec3(0.0, 0.0, -1.0)

    denominator = direction[0]
    if wp.abs(denominator) > 1.0e-8:
        candidate = (-SOCCAR_EXTENT_X - origin[0]) / denominator
        if candidate >= 0.0 and candidate <= maximum and candidate < nearest:
            nearest = candidate
            found_normal = wp.vec3(1.0, 0.0, 0.0)
        candidate = (SOCCAR_EXTENT_X - origin[0]) / denominator
        if candidate >= 0.0 and candidate <= maximum and candidate < nearest:
            nearest = candidate
            found_normal = wp.vec3(-1.0, 0.0, 0.0)
    return wp.vec4(nearest, found_normal[0], found_normal[1], found_normal[2])


@wp.func
def _inverse_inertia_world(quat: wp.quat, value: wp.vec3) -> wp.vec3:
    local = wp.quat_rotate_inv(quat, value)
    local = wp.cw_mul(local, INV_INERTIA_LOCAL)
    return wp.quat_rotate(quat, local)


@wp.func
def _axis_penetration(
    axis_raw: wp.vec3,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    half: wp.vec3,
) -> float:
    length_sq = wp.dot(axis_raw, axis_raw)
    if length_sq < 1.0e-12:
        return 1.0e9
    axis = axis_raw / wp.sqrt(length_sq)
    p0 = wp.dot(v0, axis)
    p1 = wp.dot(v1, axis)
    p2 = wp.dot(v2, axis)
    tri_min = wp.min(p0, wp.min(p1, p2))
    tri_max = wp.max(p0, wp.max(p1, p2))
    radius = half[0] * wp.abs(axis[0]) + half[1] * wp.abs(axis[1]) + half[2] * wp.abs(axis[2])
    return wp.min(tri_max, radius) - wp.max(tri_min, -radius)


@wp.func
def _triangle_obb_sat(
    world_v0: wp.vec3,
    world_v1: wp.vec3,
    world_v2: wp.vec3,
    center: wp.vec3,
    quat: wp.quat,
) -> wp.vec4:
    """Return local separating axis and penetration, with negative penetration on miss."""

    v0 = wp.quat_rotate_inv(quat, world_v0 - center)
    v1 = wp.quat_rotate_inv(quat, world_v1 - center)
    v2 = wp.quat_rotate_inv(quat, world_v2 - center)
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    tri_center = (v0 + v1 + v2) / 3.0
    best = 1.0e9
    best_axis = wp.vec3(0.0, 0.0, 1.0)

    axis = wp.vec3(1.0, 0.0, 0.0)
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.vec3(0.0, 1.0, 0.0)
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.vec3(0.0, 0.0, 1.0)
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis

    axis = wp.cross(e0, e1)
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis

    # Nine edge x box-axis tests complete the triangle/AABB SAT.
    axis = wp.cross(e0, wp.vec3(1.0, 0.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e0, wp.vec3(0.0, 1.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e0, wp.vec3(0.0, 0.0, 1.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e1, wp.vec3(1.0, 0.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e1, wp.vec3(0.0, 1.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e1, wp.vec3(0.0, 0.0, 1.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e2, wp.vec3(1.0, 0.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e2, wp.vec3(0.0, 1.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e2, wp.vec3(0.0, 0.0, 1.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_HALF)
    if penetration < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis

    best_axis = wp.normalize(best_axis)
    if wp.dot(best_axis, tri_center) > 0.0:
        best_axis = -best_axis
    return wp.vec4(best_axis[0], best_axis[1], best_axis[2], best)


@wp.kernel
def load_action_tape(
    tick_counter: wp.array(dtype=wp.int32),
    hold_ticks: int,
    tape_length: int,
    tape_throttle: wp.array(dtype=wp.float32),
    tape_steer: wp.array(dtype=wp.float32),
    tape_pitch: wp.array(dtype=wp.float32),
    tape_yaw: wp.array(dtype=wp.float32),
    tape_roll: wp.array(dtype=wp.float32),
    tape_jump: wp.array(dtype=wp.int32),
    tape_boost: wp.array(dtype=wp.int32),
    tape_handbrake: wp.array(dtype=wp.int32),
    throttle: wp.array(dtype=wp.float32),
    steer: wp.array(dtype=wp.float32),
    pitch: wp.array(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    roll: wp.array(dtype=wp.float32),
    jump: wp.array(dtype=wp.int32),
    boost: wp.array(dtype=wp.int32),
    handbrake: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    slot = ((tick_counter[0] / hold_ticks) + tid * 17) % tape_length
    throttle[tid] = tape_throttle[slot]
    steer[tid] = tape_steer[slot]
    pitch[tid] = tape_pitch[slot]
    yaw[tid] = tape_yaw[slot]
    roll[tid] = tape_roll[slot]
    jump[tid] = tape_jump[slot]
    boost[tid] = tape_boost[slot]
    handbrake[tid] = tape_handbrake[slot]


@wp.kernel
def increment_tick_counter(tick_counter: wp.array(dtype=wp.int32)):
    tick_counter[0] = tick_counter[0] + 1


@wp.kernel
def wheel_pre_tick(
    ray_mesh_id: wp.uint64,
    enable_forces: int,
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    on_ground: wp.array(dtype=wp.int32),
    boost_amount: wp.array(dtype=wp.float32),
    control_throttle: wp.array(dtype=wp.float32),
    control_steer: wp.array(dtype=wp.float32),
    control_boost: wp.array(dtype=wp.int32),
    control_handbrake: wp.array(dtype=wp.int32),
    wheel_ray_start: wp.array(dtype=wp.vec3),
    wheel_direction: wp.array(dtype=wp.vec3),
    wheel_hit_point: wp.array(dtype=wp.vec3),
    wheel_hit_normal: wp.array(dtype=wp.vec3),
    wheel_hit_distance: wp.array(dtype=wp.float32),
    wheel_hit_face: wp.array(dtype=wp.int32),
    suspension_length: wp.array(dtype=wp.float32),
    suspension_velocity: wp.array(dtype=wp.float32),
    suspension_clipped_factor: wp.array(dtype=wp.float32),
    suspension_force: wp.array(dtype=wp.float32),
    suspension_pushback: wp.array(dtype=wp.float32),
    engine_acceleration: wp.array(dtype=wp.float32),
    brake_acceleration: wp.array(dtype=wp.float32),
    steer_angle: wp.array(dtype=wp.float32),
    lateral_friction: wp.array(dtype=wp.float32),
    longitudinal_friction: wp.array(dtype=wp.float32),
    wheel_contact: wp.array(dtype=wp.int32),
    wheel_world_contact: wp.array(dtype=wp.int32),
    handbrake_value: wp.array(dtype=wp.float32),
    wheels_with_contact: wp.array(dtype=wp.int32),
):
    car = wp.tid()
    pos = car_pos[car]
    vel = car_vel[car]
    quat = car_quat[car]
    ang_vel = car_ang_vel[car]
    up = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))
    forward = wp.quat_rotate(quat, wp.vec3(1.0, 0.0, 0.0))
    right = wp.quat_rotate(quat, wp.vec3(0.0, 1.0, 0.0))
    forward_speed = wp.dot(vel, forward)
    abs_speed = wp.abs(forward_speed)
    contact_count = 0
    normal_sum = wp.vec3(0.0, 0.0, 0.0)

    for wheel in range(4):
        wheel_index = car * 4 + wheel
        front = wheel < 2
        left = (wheel % 2) == 1
        connection = wp.vec3(-33.75, 29.5, 20.755)
        radius = 15.0
        configured_rest = 37.055
        force_scale = SUSPENSION_SCALE_BACK
        if front:
            connection = wp.vec3(51.25, 25.9, 20.755)
            radius = 12.5
            configured_rest = 38.755
            force_scale = SUSPENSION_SCALE_FRONT
        if left:
            connection[1] = -connection[1]
        rest = configured_rest - MAX_SUSPENSION_TRAVEL
        ray_length = rest + MAX_SUSPENSION_TRAVEL + radius - SUSPENSION_SUBTRACTION
        source = pos + wp.quat_rotate(quat, connection)
        direction = -up
        distance = ray_length + 1.0
        normal = wp.vec3(0.0, 0.0, 0.0)
        hit_face = -1
        ray_query = wp.mesh_query_ray(ray_mesh_id, source, direction, ray_length)
        if ray_query.result:
            distance = ray_query.t
            normal = ray_query.normal
            if wp.dot(normal, direction) > 0.0:
                normal = -normal
            hit_face = ray_query.face
        denominator_plane = direction[2]
        if wp.abs(denominator_plane) > 1.0e-8:
            candidate_plane = -source[2] / denominator_plane
            if (
                candidate_plane >= 0.0
                and candidate_plane <= ray_length
                and candidate_plane < distance
            ):
                distance = candidate_plane
                normal = wp.vec3(0.0, 0.0, 1.0)
                hit_face = -2
            candidate_plane = (SOCCAR_HEIGHT - source[2]) / denominator_plane
            if (
                candidate_plane >= 0.0
                and candidate_plane <= ray_length
                and candidate_plane < distance
            ):
                distance = candidate_plane
                normal = wp.vec3(0.0, 0.0, -1.0)
                hit_face = -3
        denominator_plane = direction[0]
        if wp.abs(denominator_plane) > 1.0e-8:
            candidate_plane = (-SOCCAR_EXTENT_X - source[0]) / denominator_plane
            if (
                candidate_plane >= 0.0
                and candidate_plane <= ray_length
                and candidate_plane < distance
            ):
                distance = candidate_plane
                normal = wp.vec3(1.0, 0.0, 0.0)
                hit_face = -4
            candidate_plane = (SOCCAR_EXTENT_X - source[0]) / denominator_plane
            if (
                candidate_plane >= 0.0
                and candidate_plane <= ray_length
                and candidate_plane < distance
            ):
                distance = candidate_plane
                normal = wp.vec3(-1.0, 0.0, 0.0)
                hit_face = -5
        hit = distance <= ray_length
        hit_point = source + direction * wp.min(distance, ray_length)
        sus_length = rest + MAX_SUSPENSION_TRAVEL
        sus_velocity = 0.0
        clipped = 1.0
        pushback = 0.0
        force_value = 0.0

        if hit:
            contact_count = contact_count + 1
            normal_sum = normal_sum + normal
            trace_distance = wp.dot(source - hit_point, up)
            sus_length = wp.clamp(
                trace_distance - radius,
                rest - MAX_SUSPENSION_TRAVEL,
                rest + MAX_SUSPENSION_TRAVEL,
            )
            denominator = wp.dot(normal, up)
            contact_offset = hit_point - pos
            velocity_at_contact = vel + wp.cross(ang_vel, contact_offset)
            projected_velocity = wp.dot(normal, velocity_at_contact)
            if denominator > 0.1:
                clipped = 1.0 / denominator
                sus_velocity = projected_velocity * clipped
            else:
                clipped = 10.0
                sus_velocity = 0.0
            push_threshold = rest + radius - SUSPENSION_SUBTRACTION
            if trace_distance < push_threshold:
                pushback = (push_threshold - trace_distance) * 0.25

            compression = (rest - sus_length) * SUSPENSION_STIFFNESS * clipped
            damping = SUSPENSION_DAMPING_RELAXATION
            if sus_velocity < 0.0:
                damping = SUSPENSION_DAMPING_COMPRESSION
            force_value = wp.max(0.0, (compression - damping * sus_velocity) * force_scale)

            if enable_forces != 0:
                impulse = normal * (force_value * DT + pushback * CAR_MASS)
                vel = vel + impulse * INV_MASS
                ang_vel = ang_vel + _inverse_inertia_world(quat, wp.cross(contact_offset, impulse))

                old_steer = steer_angle[wheel_index]
                axle = right * wp.cos(old_steer) - forward * wp.sin(old_steer)
                axle = wp.normalize(axle - normal * wp.dot(axle, normal))
                wheel_forward = wp.normalize(wp.cross(normal, axle))
                relative = vel + wp.cross(ang_vel, contact_offset)

                old_engine = engine_acceleration[wheel_index]
                if old_engine != 0.0:
                    drive_impulse = -wheel_forward * old_engine * CAR_MASS * DT
                    vel = vel + drive_impulse * INV_MASS
                    ang_vel = ang_vel + _inverse_inertia_world(
                        quat, wp.cross(contact_offset, drive_impulse)
                    )

                old_brake = brake_acceleration[wheel_index]
                longitudinal_speed = wp.dot(relative, wheel_forward)
                if old_brake > 0.0 and wp.abs(longitudinal_speed) > 1.0e-5:
                    brake_delta = wp.min(wp.abs(longitudinal_speed), old_brake * DT)
                    brake_impulse = (
                        -wp.sign(longitudinal_speed) * wheel_forward * brake_delta * CAR_MASS
                    )
                    vel = vel + brake_impulse * INV_MASS
                    ang_vel = ang_vel + _inverse_inertia_world(
                        quat, wp.cross(contact_offset, brake_impulse)
                    )

                lateral_speed = wp.dot(relative, axle)
                lateral_delta = lateral_speed * wp.min(
                    1.0, lateral_friction[wheel_index] * 12.0 * DT
                )
                lateral_impulse = -axle * lateral_delta * CAR_MASS
                vel = vel + lateral_impulse * INV_MASS
                ang_vel = ang_vel + _inverse_inertia_world(
                    quat, wp.cross(contact_offset, lateral_impulse)
                )

        wheel_ray_start[wheel_index] = source
        wheel_direction[wheel_index] = direction
        wheel_hit_point[wheel_index] = hit_point
        wheel_hit_normal[wheel_index] = normal
        wheel_hit_distance[wheel_index] = distance
        wheel_hit_face[wheel_index] = hit_face
        suspension_length[wheel_index] = sus_length
        suspension_velocity[wheel_index] = sus_velocity
        suspension_clipped_factor[wheel_index] = clipped
        suspension_force[wheel_index] = force_value
        suspension_pushback[wheel_index] = pushback
        wheel_contact[wheel_index] = wp.int32(hit)
        wheel_world_contact[wheel_index] = wp.int32(hit)

    if enable_forces != 0:
        handbrake = handbrake_value[car]
        if control_handbrake[car] != 0:
            handbrake = handbrake + POWERSLIDE_RISE_RATE * DT
        else:
            handbrake = handbrake - POWERSLIDE_FALL_RATE * DT
        handbrake = wp.clamp(handbrake, 0.0, 1.0)
        handbrake_value[car] = handbrake

        real_throttle = wp.clamp(control_throttle[car], -1.0, 1.0)
        if control_boost[car] != 0 and boost_amount[car] > 0.0:
            real_throttle = 1.0
        engine_throttle = real_throttle
        real_brake = 0.0
        if control_handbrake[car] == 0:
            if wp.abs(real_throttle) >= THROTTLE_DEADZONE:
                opposite = (real_throttle > 0.0) != (forward_speed > 0.0)
                if abs_speed > STOPPING_FORWARD_VEL and opposite:
                    real_brake = 1.0
                    if abs_speed > BRAKING_NO_THROTTLE_SPEED_THRESH:
                        engine_throttle = 0.0
            else:
                engine_throttle = 0.0
                real_brake = COASTING_BRAKE_FACTOR
                if abs_speed < STOPPING_FORWARD_VEL:
                    real_brake = 1.0
        drive_scale = _drive_curve(abs_speed)
        if contact_count < 3:
            drive_scale = drive_scale * 0.25
        new_engine = engine_throttle * 400.0 * drive_scale
        new_brake = real_brake * 875.0
        new_steer = _steer_curve(abs_speed)
        if handbrake > 0.0:
            new_steer = new_steer + (_powerslide_steer_curve(abs_speed) - new_steer) * handbrake
        new_steer = new_steer * wp.clamp(control_steer[car], -1.0, 1.0)

        for wheel in range(4):
            wheel_index = car * 4 + wheel
            engine_acceleration[wheel_index] = new_engine
            brake_acceleration[wheel_index] = new_brake
            if wheel < 2:
                steer_angle[wheel_index] = new_steer
            else:
                steer_angle[wheel_index] = 0.0
            friction_input = 0.0
            if wheel_contact[wheel_index] != 0:
                offset = wheel_ray_start[wheel_index] - pos
                local_velocity = vel + wp.cross(ang_vel, offset)
                lateral_speed = wp.abs(wp.dot(local_velocity, right))
                forward_abs = wp.abs(wp.dot(local_velocity, forward))
                if lateral_speed > 5.0:
                    friction_input = lateral_speed / (forward_abs + lateral_speed)
            lat = 1.0 - 0.8 * friction_input
            long_factor = 1.0
            if handbrake > 0.0:
                lat = lat * (1.0 + (0.1 - 1.0) * handbrake)
                long_factor = long_factor * (
                    1.0 + (_linear(friction_input, 0.0, 0.5, 1.0, 0.9) - 1.0) * handbrake
                )
            if real_throttle == 0.0 and wheel_contact[wheel_index] != 0:
                sticky = _non_sticky_curve(wheel_hit_normal[wheel_index][2])
                lat = lat * sticky
                long_factor = long_factor * sticky
            lateral_friction[wheel_index] = lat
            longitudinal_friction[wheel_index] = long_factor

        if contact_count > 0:
            upwards = wp.normalize(normal_sum)
            sticky_scale = 0.5
            if real_throttle != 0.0 or abs_speed > STOPPING_FORWARD_VEL:
                sticky_scale = sticky_scale + 1.0 - wp.abs(upwards[2])
            vel = vel + upwards * (-650.0 * sticky_scale * DT)
        on_ground[car] = wp.int32(contact_count >= 3)
        car_vel[car] = vel
        car_ang_vel[car] = ang_vel
    wheels_with_contact[car] = contact_count


@wp.func
def _solve_contact(
    pos: wp.vec3,
    vel: wp.vec3,
    quat: wp.quat,
    ang_vel: wp.vec3,
    point: wp.vec3,
    normal: wp.vec3,
    penetration: float,
) -> wp.mat44:
    """Pack corrected position/velocity/angular velocity into a 4x4 matrix."""

    offset = point - pos
    point_velocity = vel + wp.cross(ang_vel, offset)
    normal_speed = wp.dot(point_velocity, normal)
    normal_impulse_magnitude = 0.0
    if normal_speed < 0.0:
        cross_rn = wp.cross(offset, normal)
        angular_term = wp.dot(normal, wp.cross(_inverse_inertia_world(quat, cross_rn), offset))
        denominator = INV_MASS + angular_term
        if denominator > 1.0e-9:
            normal_impulse_magnitude = -(1.0 + CONTACT_RESTITUTION) * normal_speed / denominator
            impulse = normal * normal_impulse_magnitude
            vel = vel + impulse * INV_MASS
            ang_vel = ang_vel + _inverse_inertia_world(quat, wp.cross(offset, impulse))

            post_velocity = vel + wp.cross(ang_vel, offset)
            tangent = post_velocity - normal * wp.dot(post_velocity, normal)
            tangent_length = wp.length(tangent)
            if tangent_length > 1.0e-6:
                tangent = tangent / tangent_length
                cross_rt = wp.cross(offset, tangent)
                tangent_denominator = INV_MASS + wp.dot(
                    tangent,
                    wp.cross(_inverse_inertia_world(quat, cross_rt), offset),
                )
                if tangent_denominator > 1.0e-9:
                    tangent_impulse = -wp.dot(post_velocity, tangent) / tangent_denominator
                    limit = CONTACT_FRICTION * normal_impulse_magnitude
                    tangent_impulse = wp.clamp(tangent_impulse, -limit, limit)
                    friction_impulse = tangent * tangent_impulse
                    vel = vel + friction_impulse * INV_MASS
                    ang_vel = ang_vel + _inverse_inertia_world(
                        quat, wp.cross(offset, friction_impulse)
                    )
    correction = wp.max(0.0, penetration - CONTACT_SLOP) * CONTACT_CORRECTION
    pos = pos + normal * correction
    return wp.mat44(
        pos[0],
        pos[1],
        pos[2],
        0.0,
        vel[0],
        vel[1],
        vel[2],
        0.0,
        ang_vel[0],
        ang_vel[1],
        ang_vel[2],
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel
def chassis_contacts(
    aabb_mesh_id: wp.uint64,
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    candidate_count: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    candidate_total: wp.array(dtype=wp.float32),
    contact_total: wp.array(dtype=wp.float32),
    candidate_max: wp.array(dtype=wp.int32),
    contact_max: wp.array(dtype=wp.int32),
    penetration_max: wp.array(dtype=wp.float32),
    contact_point: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_penetration: wp.array(dtype=wp.float32),
):
    car = wp.tid()
    pos = car_pos[car]
    vel = car_vel[car]
    quat = car_quat[car]
    ang_vel = car_ang_vel[car]
    center = pos + wp.quat_rotate(quat, HITBOX_OFFSET)
    axis_x = wp.quat_rotate(quat, wp.vec3(1.0, 0.0, 0.0))
    axis_y = wp.quat_rotate(quat, wp.vec3(0.0, 1.0, 0.0))
    axis_z = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))
    aabb_half = wp.vec3(
        wp.abs(axis_x[0]) * HITBOX_HALF[0]
        + wp.abs(axis_y[0]) * HITBOX_HALF[1]
        + wp.abs(axis_z[0]) * HITBOX_HALF[2],
        wp.abs(axis_x[1]) * HITBOX_HALF[0]
        + wp.abs(axis_y[1]) * HITBOX_HALF[1]
        + wp.abs(axis_z[1]) * HITBOX_HALF[2],
        wp.abs(axis_x[2]) * HITBOX_HALF[0]
        + wp.abs(axis_y[2]) * HITBOX_HALF[1]
        + wp.abs(axis_z[2]) * HITBOX_HALF[2],
    )
    candidates = 0
    contacts = 0
    maximum_penetration = 0.0

    # RocketSim adds these four infinite planes beside its CMF shapes.
    for plane in range(4):
        normal = wp.vec3(0.0, 0.0, 1.0)
        plane_point = wp.vec3(0.0, 0.0, 0.0)
        if plane == 1:
            normal = wp.vec3(0.0, 0.0, -1.0)
            plane_point = wp.vec3(0.0, 0.0, SOCCAR_HEIGHT)
        elif plane == 2:
            normal = wp.vec3(1.0, 0.0, 0.0)
            plane_point = wp.vec3(-SOCCAR_EXTENT_X, 0.0, 0.0)
        elif plane == 3:
            normal = wp.vec3(-1.0, 0.0, 0.0)
            plane_point = wp.vec3(SOCCAR_EXTENT_X, 0.0, 0.0)
        radius = (
            HITBOX_HALF[0] * wp.abs(wp.dot(normal, axis_x))
            + HITBOX_HALF[1] * wp.abs(wp.dot(normal, axis_y))
            + HITBOX_HALF[2] * wp.abs(wp.dot(normal, axis_z))
        )
        signed_distance = wp.dot(center - plane_point, normal)
        penetration = radius - signed_distance
        if penetration > 0.0 and contacts < 4:
            maximum_penetration = wp.max(maximum_penetration, penetration)
            candidates = candidates + 1
            point = center - normal * radius
            packed = _solve_contact(pos, vel, quat, ang_vel, point, normal, penetration)
            pos = wp.vec3(packed[0, 0], packed[0, 1], packed[0, 2])
            vel = wp.vec3(packed[1, 0], packed[1, 1], packed[1, 2])
            ang_vel = wp.vec3(packed[2, 0], packed[2, 1], packed[2, 2])
            output_index = car * 4 + contacts
            contact_point[output_index] = point
            contact_normal[output_index] = normal
            contact_penetration[output_index] = penetration
            contacts = contacts + 1

    query = wp.mesh_query_aabb(aabb_mesh_id, center - aabb_half, center + aabb_half)
    for face in query:
        candidates = candidates + 1
        if contacts < 4:
            v0 = wp.mesh_eval_position(aabb_mesh_id, face, 1.0, 0.0)
            v1 = wp.mesh_eval_position(aabb_mesh_id, face, 0.0, 1.0)
            v2 = wp.mesh_eval_position(aabb_mesh_id, face, 0.0, 0.0)
            sat = _triangle_obb_sat(v0, v1, v2, center, quat)
            penetration = sat[3]
            if penetration >= 0.0:
                maximum_penetration = wp.max(maximum_penetration, penetration)
                local_normal = wp.vec3(sat[0], sat[1], sat[2])
                normal = wp.quat_rotate(quat, local_normal)
                radius = (
                    HITBOX_HALF[0] * wp.abs(local_normal[0])
                    + HITBOX_HALF[1] * wp.abs(local_normal[1])
                    + HITBOX_HALF[2] * wp.abs(local_normal[2])
                )
                point = center - normal * radius
                packed = _solve_contact(pos, vel, quat, ang_vel, point, normal, penetration)
                pos = wp.vec3(packed[0, 0], packed[0, 1], packed[0, 2])
                vel = wp.vec3(packed[1, 0], packed[1, 1], packed[1, 2])
                ang_vel = wp.vec3(packed[2, 0], packed[2, 1], packed[2, 2])
                output_index = car * 4 + contacts
                contact_point[output_index] = point
                contact_normal[output_index] = normal
                contact_penetration[output_index] = penetration
                contacts = contacts + 1

    for index in range(4):
        if index >= contacts:
            output_index = car * 4 + index
            contact_point[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_normal[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_penetration[output_index] = 0.0
    car_pos[car] = pos
    car_vel[car] = vel
    car_ang_vel[car] = ang_vel
    candidate_count[car] = candidates
    contact_count[car] = contacts
    candidate_total[car] = candidate_total[car] + float(candidates)
    contact_total[car] = contact_total[car] + float(contacts)
    candidate_max[car] = wp.max(candidate_max[car], candidates)
    contact_max[car] = wp.max(contact_max[car], contacts)
    penetration_max[car] = wp.max(penetration_max[car], maximum_penetration)
