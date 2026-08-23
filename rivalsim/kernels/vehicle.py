"""RocketSim-derived Octane wheel forces and static-world chassis contacts."""

import warp as wp

DT = 1.0 / 120.0
CAR_MASS = 180.0
INV_MASS = 1.0 / CAR_MASS
# Bullet's btBoxShape constructor reduces each effective Octane half extent by
# 0.001341 BT when its safe margin replaces the 0.04 BT default.  These values
# and inverse inertias are read from the pinned native diagnostic build.
HITBOX_MARGIN_REDUCTION = 0.06704521179199219
INV_INERTIA_LOCAL = wp.vec3(
    0.0185644571 / 2500.0,
    0.0104337428 / 2500.0,
    0.0075815497 / 2500.0,
)
INV_INERTIA_LOCAL_BT = wp.vec3(0.0185644571, 0.0104337428, 0.0075815497)
HITBOX_HALF = wp.vec3(120.50700378417969 / 2.0, 86.69940185546875 / 2.0, 38.65909957885742 / 2.0)
HITBOX_COLLISION_HALF = HITBOX_HALF - wp.vec3(
    HITBOX_MARGIN_REDUCTION,
    HITBOX_MARGIN_REDUCTION,
    HITBOX_MARGIN_REDUCTION,
)
HITBOX_MARGIN = 1.932954975
HITBOX_CORE_HALF = HITBOX_COLLISION_HALF - wp.vec3(
    HITBOX_MARGIN,
    HITBOX_MARGIN,
    HITBOX_MARGIN,
)
HITBOX_OFFSET = wp.vec3(13.875699996948242, 0.0, 20.7549991607666)
MAX_SUSPENSION_TRAVEL = 12.0
# RLConst::BTVehicle::SUSPENSION_SUBTRACTION is stored in Bullet units.
# RocketSim uses 50 Unreal Units per Bullet unit, so 0.05 BT is 2.5 UU.
SUSPENSION_SUBTRACTION = 2.5
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
# The compound Octane shape's relative Bullet breaking threshold is
# 0.0406245552 BT at the pinned build (50 UU per BT).
CONTACT_BREAKING_THRESHOLD = 2.03122776
# The CMF triangle path operates in UU while Bullet compares its native BT
# separation. At Soccar's 4096-UU wall coordinate, one FP32 world-coordinate
# ULP is 2^-11 UU; reserve that final comparison band so a rounded-down UU
# distance cannot create a contact that the native BT comparison rejects.
CONTACT_BREAKING_ROUNDING_GUARD = 0.00048828125
CONTACT_LINEAR_SLOP = 0.0
CONTACT_ERP2 = 0.8
CONTACT_SPLIT_TURN_ERP = 0.1
CONTACT_SOLVER_ITERATIONS = 10
GJK_MAX_ITERATIONS = 16
GJK_EQUAL_VERTEX_THRESHOLD_BT2 = 0.0001
GJK_SUPPORT_TIE_BT2 = 0.00001
GJK_REL_ERROR2 = 1.0e-6
GJK_SIMD_EPSILON = 1.1920929e-7
GJK_CORE_HALF_BT = wp.vec3(1.16507006, 0.826994002, 0.346590996)
GJK_OFFSET_BT = wp.vec3(0.277513981, 0.0, 0.415099978)
GJK_MARGIN_BT = 0.0386590995
GJK_MAXIMUM_DISTANCE_BT = 0.0792836547
AUTO_ROLL_ACCELERATION = 100.0
AUTO_ROLL_ANGULAR_ACCELERATION = 80.0
FRICTION_SCALE = CAR_MASS / 3.0
BILATERAL_CONTACT_DAMPING = 0.2
ROLLING_FRICTION_SCALE_MAGIC = 113.73963
SOLVER_ERP = 0.2
SOLVER_INITIAL_DT = 1.0 / 60.0


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
def _bullet_quaternion_matrix(quat: wp.quat) -> wp.mat33:
    """Match btMatrix3x3::setRotation's normalized-quaternion construction."""

    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]
    length_sq = (x * x + z * z) + (y * y + w * w)
    scale = 2.0 / length_sq
    return wp.mat33(
        1.0 + ((-y * y) + (-z * z)) * scale,
        ((x * y) + (-w * z)) * scale,
        ((x * z) + (w * y)) * scale,
        ((x * y) + (w * z)) * scale,
        1.0 + ((-x * x) + (-z * z)) * scale,
        ((y * z) + (-w * x)) * scale,
        ((x * z) + (-w * y)) * scale,
        ((y * z) + (w * x)) * scale,
        1.0 + ((-x * x) + (-y * y)) * scale,
    )


@wp.func
def _inverse_inertia_world(
    quat: wp.quat, value: wp.vec3, transpose_mix: float, bt_units: int
) -> wp.vec3:
    # btRigidBody materializes basis.scaled(invInertiaLocal) * basis.transpose()
    # and then matrix-multiplies impulses by that tensor. Keeping that order is
    # important near support-vertex sign changes in long resting contacts.
    matrix = _bullet_quaternion_matrix(quat)
    inverse_local = INV_INERTIA_LOCAL
    if bt_units != 0:
        inverse_local = INV_INERTIA_LOCAL_BT
    row0 = wp.vec3(
        matrix[0, 0] * inverse_local[0],
        matrix[0, 1] * inverse_local[1],
        matrix[0, 2] * inverse_local[2],
    )
    row1 = wp.vec3(
        matrix[1, 0] * inverse_local[0],
        matrix[1, 1] * inverse_local[1],
        matrix[1, 2] * inverse_local[2],
    )
    row2 = wp.vec3(
        matrix[2, 0] * inverse_local[0],
        matrix[2, 1] * inverse_local[1],
        matrix[2, 2] * inverse_local[2],
    )
    tensor = wp.mat33(
        (row0[0] * matrix[0, 0] + row0[1] * matrix[0, 1])
        + row0[2] * matrix[0, 2],
        (row0[0] * matrix[1, 0] + row0[1] * matrix[1, 1])
        + row0[2] * matrix[1, 2],
        (row0[0] * matrix[2, 0] + row0[1] * matrix[2, 1])
        + row0[2] * matrix[2, 2],
        (row1[0] * matrix[0, 0] + row1[1] * matrix[0, 1])
        + row1[2] * matrix[0, 2],
        (row1[0] * matrix[1, 0] + row1[1] * matrix[1, 1])
        + row1[2] * matrix[1, 2],
        (row1[0] * matrix[2, 0] + row1[1] * matrix[2, 1])
        + row1[2] * matrix[2, 2],
        (row2[0] * matrix[0, 0] + row2[1] * matrix[0, 1])
        + row2[2] * matrix[0, 2],
        (row2[0] * matrix[1, 0] + row2[1] * matrix[1, 1])
        + row2[2] * matrix[1, 2],
        (row2[0] * matrix[2, 0] + row2[1] * matrix[2, 1])
        + row2[2] * matrix[2, 2],
    )
    transpose_result = wp.vec3(
        (value[0] * tensor[0, 0] + value[1] * tensor[1, 0])
        + value[2] * tensor[2, 0],
        (value[0] * tensor[0, 1] + value[1] * tensor[1, 1])
        + value[2] * tensor[2, 1],
        (value[0] * tensor[0, 2] + value[1] * tensor[1, 2])
        + value[2] * tensor[2, 2],
    )
    direct_result = wp.vec3(
        (tensor[0, 0] * value[0] + tensor[0, 1] * value[1])
        + tensor[0, 2] * value[2],
        (tensor[1, 0] * value[0] + tensor[1, 1] * value[1])
        + tensor[1, 2] * value[2],
        (tensor[2, 0] * value[0] + tensor[2, 1] * value[1])
        + tensor[2, 2] * value[2],
    )
    return direct_result + (transpose_result - direct_result) * transpose_mix


@wp.func
def _impulse_denominator(
    quat: wp.quat,
    offset: wp.vec3,
    direction: wp.vec3,
    transpose_mix: float,
    bt_units: int,
) -> float:
    angular_axis = wp.cross(offset, direction)
    return INV_MASS + wp.dot(
        angular_axis,
        _inverse_inertia_world(quat, angular_axis, transpose_mix, bt_units),
    )


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
    if tri_min > radius:
        return radius - tri_min
    if tri_max < -radius:
        return tri_max + radius
    # A triangle has no volume along its face normal. Its interval-overlap
    # length is therefore zero even while it penetrates the box; use the
    # minimum separating translation instead of overlap length.
    return wp.min(radius - tri_min, tri_max + radius)


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
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.vec3(0.0, 1.0, 0.0)
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.vec3(0.0, 0.0, 1.0)
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis

    axis = wp.cross(e0, e1)
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis

    # Nine edge x box-axis tests complete the triangle/AABB SAT.
    axis = wp.cross(e0, wp.vec3(1.0, 0.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e0, wp.vec3(0.0, 1.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e0, wp.vec3(0.0, 0.0, 1.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e1, wp.vec3(1.0, 0.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e1, wp.vec3(0.0, 1.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e1, wp.vec3(0.0, 0.0, 1.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e2, wp.vec3(1.0, 0.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e2, wp.vec3(0.0, 1.0, 0.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
        return wp.vec4(0.0, 0.0, 0.0, -1.0)
    if penetration < best:
        best = penetration
        best_axis = axis
    axis = wp.cross(e2, wp.vec3(0.0, 0.0, 1.0))
    penetration = _axis_penetration(axis, v0, v1, v2, HITBOX_COLLISION_HALF)
    if penetration < -CONTACT_BREAKING_THRESHOLD:
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
    tick_counter: wp.array(dtype=wp.int32),
    ray_mesh_id: wp.uint64,
    enable_forces: int,
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    solver_position: wp.array(dtype=wp.vec3),
    rigid_position_bt: wp.array(dtype=wp.vec3),
    solver_orientation: wp.array(dtype=wp.quat),
    solver_velocity: wp.array(dtype=wp.vec3),
    rigid_velocity_bt: wp.array(dtype=wp.vec3),
    solver_angular_velocity: wp.array(dtype=wp.vec3),
    auto_roll_acceleration: wp.array(dtype=wp.vec3),
    auto_roll_angular_acceleration: wp.array(dtype=wp.vec3),
    previous_contact_count: wp.array(dtype=wp.int32),
    previous_contact_normal: wp.array(dtype=wp.vec3),
    previous_contact_face: wp.array(dtype=wp.int32),
    on_ground: wp.array(dtype=wp.int32),
    air_control_disabled: wp.array(dtype=wp.int32),
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
    wheel_axle: wp.array(dtype=wp.vec3),
    wheel_forward: wp.array(dtype=wp.vec3),
    wheel_friction_impulse: wp.array(dtype=wp.vec3),
    side_impulse: wp.array(dtype=wp.float32),
    rolling_impulse: wp.array(dtype=wp.float32),
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
    base_vel = car_vel[car]
    quat = car_quat[car]
    base_ang_vel = car_ang_vel[car]
    solver_position[car] = pos
    if tick_counter[0] == 0:
        rigid_position_bt[car] = pos * 0.02
        rigid_velocity_bt[car] = base_vel * 0.02
    solver_orientation[car] = quat
    auto_roll_acceleration[car] = wp.vec3(0.0, 0.0, 0.0)
    auto_roll_angular_acceleration[car] = wp.vec3(0.0, 0.0, 0.0)
    up = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))
    forward = wp.quat_rotate(quat, wp.vec3(1.0, 0.0, 0.0))
    right = wp.quat_rotate(quat, wp.vec3(0.0, 1.0, 0.0))
    forward_speed = wp.dot(base_vel, forward)
    abs_speed = wp.abs(forward_speed)
    contact_count = 0
    normal_sum = wp.vec3(0.0, 0.0, 0.0)
    solver_dt = DT
    if tick_counter[0] == 0:
        # btContactSolverInfo starts at 60 Hz and is changed by the first
        # stepSimulation call, which occurs after the vehicle pre-tick.
        solver_dt = SOLVER_INITIAL_DT

    # RocketSim updateVehicleFirst: update every wheel transform/ray and cache
    # all friction impulses from one unchanged rigid-body state.  Wheel forces
    # are deliberately not applied in this loop.
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
                and candidate_plane < distance - 1.0e-4
            ):
                distance = candidate_plane
                normal = wp.vec3(0.0, 0.0, 1.0)
                if wp.dot(normal, direction) > 0.0:
                    normal = -normal
                hit_face = -2
            candidate_plane = (SOCCAR_HEIGHT - source[2]) / denominator_plane
            if (
                candidate_plane >= 0.0
                and candidate_plane <= ray_length
                and candidate_plane < distance - 1.0e-4
            ):
                distance = candidate_plane
                normal = wp.vec3(0.0, 0.0, -1.0)
                if wp.dot(normal, direction) > 0.0:
                    normal = -normal
                hit_face = -3
        denominator_plane = direction[0]
        if wp.abs(denominator_plane) > 1.0e-8:
            candidate_plane = (-SOCCAR_EXTENT_X - source[0]) / denominator_plane
            if (
                candidate_plane >= 0.0
                and candidate_plane <= ray_length
                and candidate_plane < distance - 1.0e-4
            ):
                distance = candidate_plane
                normal = wp.vec3(1.0, 0.0, 0.0)
                if wp.dot(normal, direction) > 0.0:
                    normal = -normal
                hit_face = -4
            candidate_plane = (SOCCAR_EXTENT_X - source[0]) / denominator_plane
            if (
                candidate_plane >= 0.0
                and candidate_plane <= ray_length
                and candidate_plane < distance - 1.0e-4
            ):
                distance = candidate_plane
                normal = wp.vec3(-1.0, 0.0, 0.0)
                if wp.dot(normal, direction) > 0.0:
                    normal = -normal
                hit_face = -5
        hit = distance <= ray_length
        hit_point = source + direction * wp.min(distance, ray_length)
        sus_length = rest + MAX_SUSPENSION_TRAVEL
        sus_velocity = 0.0
        clipped = 1.0
        # btVehicleRL only clears m_extraPushback when the ray misses.  While a
        # wheel remains on a static object, a value produced by an earlier
        # below-threshold trace persists until another below-threshold solve
        # replaces it.
        pushback = suspension_pushback[wheel_index]
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
            velocity_at_contact = base_vel + wp.cross(base_ang_vel, contact_offset)
            projected_velocity = wp.dot(normal, velocity_at_contact)
            if denominator > 0.1:
                clipped = 1.0 / denominator
                sus_velocity = projected_velocity * clipped
            else:
                clipped = 10.0
                sus_velocity = 0.0
            push_threshold = rest + radius - SUSPENSION_SUBTRACTION
            if trace_distance < push_threshold:
                # RocketSim calls Bullet resolveSingleCollision without applying
                # its result, divides the BT impulse among the wheels, and adds
                # it to the later suspension impulse.  The expression below is
                # the same calculation in UU momentum units.
                distance_bt = (trace_distance - push_threshold) / 50.0
                relative_velocity_bt = projected_velocity / 50.0
                effective_denominator = _impulse_denominator(
                    quat, contact_offset, normal, 1.0, 0
                )
                if effective_denominator > 1.0e-9:
                    positional_error = SOLVER_ERP * -distance_bt / solver_dt
                    velocity_error = -relative_velocity_bt
                    pushback_bt = wp.max(
                        0.0,
                        (positional_error + velocity_error) / effective_denominator,
                    )
                    pushback = pushback_bt * 50.0 / 4.0

            compression = (rest - sus_length) * SUSPENSION_STIFFNESS * clipped
            damping = SUSPENSION_DAMPING_RELAXATION
            if sus_velocity < 0.0:
                damping = SUSPENSION_DAMPING_COMPRESSION
            force_value = wp.max(0.0, (compression - damping * sus_velocity) * force_scale)

            old_steer = steer_angle[wheel_index]
            axle = right * wp.cos(old_steer) - forward * wp.sin(old_steer)
            axle = wp.normalize(axle - normal * wp.dot(axle, normal))
            forward_at_wheel = wp.normalize(wp.cross(normal, axle))
            relative = base_vel + wp.cross(base_ang_vel, contact_offset)

            # Bullet resolveSingleBilateral uses a 0.2 contact damping term and
            # the complete angular effective mass at the wheel contact.
            lateral_speed = wp.dot(relative, axle)
            side_denominator = _impulse_denominator(
                quat, contact_offset, axle, 1.0, 0
            )
            side_value = 0.0
            if side_denominator > 1.0e-9:
                side_value = (
                    -BILATERAL_CONTACT_DAMPING
                    * lateral_speed
                    / side_denominator
                    * FRICTION_SCALE
                    * DT
                )

            # Cache RocketSim's rolling impulse using the previous tick's
            # engine/brake values.  Both branches are expressed as UU momentum.
            old_engine = engine_acceleration[wheel_index]
            old_brake = brake_acceleration[wheel_index]
            longitudinal_speed = wp.dot(relative, forward_at_wheel)
            rolling_value = 0.0
            if old_engine != 0.0:
                rolling_value = -old_engine * CAR_MASS * DT
            elif old_brake > 0.0:
                rolling_limit = old_brake * 3.0 * FRICTION_SCALE * DT
                rolling_value = wp.clamp(
                    -longitudinal_speed * ROLLING_FRICTION_SCALE_MAGIC * FRICTION_SCALE * DT,
                    -rolling_limit,
                    rolling_limit,
                )

            cached_impulse = (
                forward_at_wheel
                * rolling_value
                * longitudinal_friction[wheel_index]
                + axle * side_value * lateral_friction[wheel_index]
            )
            wheel_axle[wheel_index] = axle
            wheel_forward[wheel_index] = forward_at_wheel
            wheel_friction_impulse[wheel_index] = cached_impulse
            side_impulse[wheel_index] = side_value
            rolling_impulse[wheel_index] = rolling_value
        else:
            pushback = 0.0
            wheel_axle[wheel_index] = wp.vec3(0.0, 0.0, 0.0)
            wheel_forward[wheel_index] = wp.vec3(0.0, 0.0, 0.0)
            wheel_friction_impulse[wheel_index] = wp.vec3(0.0, 0.0, 0.0)
            side_impulse[wheel_index] = 0.0
            rolling_impulse[wheel_index] = 0.0

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
        air_control_disabled[car] = wp.int32(contact_count > 0)
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
            old_steer = steer_angle[wheel_index]
            if wheel_contact[wheel_index] != 0:
                friction_input = 0.0
                offset = wheel_ray_start[wheel_index] - pos
                local_velocity = base_vel + wp.cross(base_ang_vel, offset)
                # Car::_UpdateWheels evaluates its slip curve with the raw
                # wheel-transform right axis.  calcFrictionImpulses separately
                # projects that axis onto the contact plane; reusing the
                # projected solver axle here overstates grip on a tilted car.
                lateral_direction = (
                    right * wp.cos(old_steer) - forward * wp.sin(old_steer)
                )
                longitudinal_direction = wp.cross(
                    lateral_direction, wheel_hit_normal[wheel_index]
                )
                lateral_speed = wp.abs(wp.dot(local_velocity, lateral_direction))
                forward_abs = wp.abs(wp.dot(local_velocity, longitudinal_direction))
                if lateral_speed > 5.0:
                    friction_input = lateral_speed / (forward_abs + lateral_speed)
                lat = 1.0 - 0.8 * friction_input
                long_factor = 1.0
                if handbrake > 0.0:
                    lat = lat * (1.0 + (0.1 - 1.0) * handbrake)
                    long_factor = long_factor * (
                        1.0
                        + (_linear(friction_input, 0.0, 0.5, 1.0, 0.9) - 1.0)
                        * handbrake
                    )
                if real_throttle == 0.0:
                    sticky = _non_sticky_curve(wheel_hit_normal[wheel_index][2])
                    lat = lat * sticky
                    long_factor = long_factor * sticky
                lateral_friction[wheel_index] = lat
                longitudinal_friction[wheel_index] = long_factor
            engine_acceleration[wheel_index] = new_engine
            brake_acceleration[wheel_index] = new_brake
            if wheel < 2:
                steer_angle[wheel_index] = new_steer
            else:
                steer_angle[wheel_index] = 0.0

        # RocketSim updateVehicleSecond: suspension impulses first, then the
        # friction impulses cached before control setup.  The friction lever arm
        # is projected onto the chassis plane (ROLLING_INFLUENCE_FIX).
        vel = base_vel
        ang_vel = base_ang_vel
        for wheel in range(4):
            wheel_index = car * 4 + wheel
            if wheel_contact[wheel_index] != 0 and suspension_force[wheel_index] != 0.0:
                suspension_impulse = wheel_hit_normal[wheel_index] * (
                    suspension_force[wheel_index] * DT
                    + suspension_pushback[wheel_index]
                )
                suspension_offset = wheel_hit_point[wheel_index] - pos
                vel = vel + suspension_impulse * INV_MASS
                ang_vel = ang_vel + _inverse_inertia_world(
                    quat,
                    wp.cross(suspension_offset, suspension_impulse),
                    1.0,
                    0,
                )

        for wheel in range(4):
            wheel_index = car * 4 + wheel
            friction = wheel_friction_impulse[wheel_index]
            if wp.dot(friction, friction) > 0.0:
                wheel_offset = wheel_hit_point[wheel_index] - pos
                wheel_offset = wheel_offset - up * wp.dot(up, wheel_offset)
                vel = vel + friction * INV_MASS
                ang_vel = ang_vel + _inverse_inertia_world(
                    quat,
                    wp.cross(wheel_offset, friction),
                    1.0,
                    0,
                )

        # Car::_UpdateAutoRoll consumes the previous tick's chassis contact (or
        # the current wheel-normal average), then queues a central force and a
        # torque for Bullet's external-force integration.
        if control_throttle[car] != 0.0 and (
            (contact_count > 0 and contact_count < 4) or previous_contact_count[car] > 0
        ):
            ground_up = wp.vec3(0.0, 0.0, 0.0)
            if contact_count > 0:
                ground_up = wp.normalize(normal_sum)
            else:
                # RocketSim's collision callback keeps the last processed
                # world-contact normal.  Mesh contacts arrive in Bullet BVH
                # order, while the separately dispatched arena planes arrive
                # after them even though the solver stores plane rows first.
                previous_index = car * 4
                last_previous_index = car * 4 + previous_contact_count[car] - 1
                if previous_contact_count[car] > 1:
                    first_previous_normal = previous_contact_normal[car * 4]
                    last_previous_normal = previous_contact_normal[last_previous_index]
                    if wp.dot(first_previous_normal, last_previous_normal) < 0.9999:
                        previous_index = last_previous_index
                for previous_contact in range(4):
                    if (
                        previous_contact < previous_contact_count[car]
                        and previous_contact_face[car * 4 + previous_contact] < 0
                    ):
                        previous_index = car * 4 + previous_contact
                ground_up = previous_contact_normal[previous_index]
            ground_down = -ground_up
            cross_right = wp.cross(ground_up, forward)
            cross_forward = wp.cross(ground_down, cross_right)
            right_factor = 1.0 - wp.clamp(wp.dot(right, cross_right), 0.0, 1.0)
            forward_factor = 1.0 - wp.clamp(wp.dot(forward, cross_forward), 0.0, 1.0)
            right_sign = 1.0
            if wp.dot(right, ground_up) >= 0.0:
                right_sign = -1.0
            forward_sign = -1.0
            if wp.dot(forward, ground_up) >= 0.0:
                forward_sign = 1.0
            torque_right = forward * (right_sign * right_factor)
            torque_forward = right * (forward_sign * forward_factor)
            auto_roll_acceleration[car] = ground_down * AUTO_ROLL_ACCELERATION
            auto_roll_angular_acceleration[car] = (
                torque_forward + torque_right
            ) * AUTO_ROLL_ANGULAR_ACCELERATION

        # Bullet's restitution curve reads the rigid-body velocity before
        # external forces.  Suspension/friction above are direct impulses, but
        # sticky is an external central force and therefore must be excluded
        # from this saved pre-force velocity.
        solver_velocity[car] = vel
        solver_angular_velocity[car] = ang_vel
        if contact_count > 0:
            upwards = wp.normalize(normal_sum)
            sticky_scale = 0.5
            if real_throttle != 0.0 or abs_speed > STOPPING_FORWARD_VEL:
                sticky_scale = sticky_scale + 1.0 - wp.abs(upwards[2])
            vel = vel + upwards * (-650.0 * sticky_scale * DT)
        on_ground[car] = wp.int32(contact_count >= 3)
        car_vel[car] = vel
        car_ang_vel[car] = ang_vel
    else:
        air_control_disabled[car] = 0
        solver_velocity[car] = base_vel
        solver_angular_velocity[car] = base_ang_vel
    wheels_with_contact[car] = contact_count


@wp.func
def _contact_support_point(
    pos: wp.vec3,
    pos_bt: wp.vec3,
    quat: wp.quat,
    normal: wp.vec3,
    plane_bt_mode: int,
    retain_previous: int,
    support_hysteresis: float,
    plane_support_direction: wp.array(dtype=wp.float32),
    support_base: int,
) -> wp.vec3:
    """Return Bullet btBoxShape's support point toward ``-normal`` in UU."""

    local_direction = wp.quat_rotate_inv(quat, -normal)
    local_support = wp.vec3(
        HITBOX_COLLISION_HALF[0],
        HITBOX_COLLISION_HALF[1],
        HITBOX_COLLISION_HALF[2],
    )
    for axis in range(3):
        direction = local_direction[axis]
        use_negative = direction < 0.0
        prior_direction = plane_support_direction[support_base + axis]
        if (
            retain_previous != 0
            and wp.abs(direction) <= support_hysteresis
            and direction >= 0.0
            and prior_direction > 0.0
            and wp.abs(direction) < wp.abs(prior_direction)
        ):
            # A positive component that is still shrinking through Bullet's
            # float32 tie region reaches the negative support vertex one step
            # earlier in the pinned SSE reference.  Do not delay a component
            # that has already crossed negative: its literal sign is correct.
            use_negative = True
        if use_negative:
            local_support[axis] = -local_support[axis]
        plane_support_direction[support_base + axis] = direction
    if (plane_bt_mode & 5) != 0 and normal[2] > 0.5:
        matrix = _bullet_quaternion_matrix(quat)
        local_bt = (HITBOX_OFFSET + local_support) * 0.02
        support_origin_bt = pos * 0.02
        if (plane_bt_mode & 4) != 0:
            support_origin_bt = pos_bt
        point_bt = support_origin_bt + wp.vec3(
            (matrix[0, 0] * local_bt[0] + matrix[0, 1] * local_bt[1])
            + matrix[0, 2] * local_bt[2],
            (matrix[1, 0] * local_bt[0] + matrix[1, 1] * local_bt[1])
            + matrix[1, 2] * local_bt[2],
            (matrix[2, 0] * local_bt[0] + matrix[2, 1] * local_bt[1])
            + matrix[2, 2] * local_bt[2],
        )
        return point_bt * 50.0
    return pos + wp.quat_rotate(quat, HITBOX_OFFSET + local_support)


@wp.func
def _contact_core_support_point(
    pos: wp.vec3, quat: wp.quat, normal: wp.vec3
) -> wp.vec3:
    local_direction = wp.quat_rotate_inv(quat, -normal)
    local_support = wp.vec3(
        HITBOX_CORE_HALF[0],
        HITBOX_CORE_HALF[1],
        HITBOX_CORE_HALF[2],
    )
    if local_direction[0] < 0.0:
        local_support[0] = -local_support[0]
    if local_direction[1] < 0.0:
        local_support[1] = -local_support[1]
    if local_direction[2] < 0.0:
        local_support[2] = -local_support[2]
    return pos + wp.quat_rotate(quat, HITBOX_OFFSET + local_support)


@wp.func
def _closest_point_triangle(
    point: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
) -> wp.vec3:
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + ab * v

    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + ac * w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + (c - b) * w

    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    return a + ab * v + ac * w


@wp.func
def _closest_segment_parameters(
    p1: wp.vec3,
    q1: wp.vec3,
    p2: wp.vec3,
    q2: wp.vec3,
) -> wp.vec2:
    d1 = q1 - p1
    d2 = q2 - p2
    relative = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, relative)
    s = 0.0
    t = 0.0
    if a <= 1.0e-12 and e <= 1.0e-12:
        return wp.vec2(0.0, 0.0)
    if a <= 1.0e-12:
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, relative)
        if e <= 1.0e-12:
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            denominator = a * e - b * b
            if wp.abs(denominator) > 1.0e-12:
                s = wp.clamp((b * f - c * e) / denominator, 0.0, 1.0)
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)
    return wp.vec2(s, t)


@wp.func
def _point_segment_distance_sq(point: wp.vec3, start: wp.vec3, end: wp.vec3) -> float:
    edge = end - start
    length_sq = wp.dot(edge, edge)
    if length_sq <= 1.0e-12:
        return wp.dot(point - start, point - start)
    fraction = wp.clamp(wp.dot(point - start, edge) / length_sq, 0.0, 1.0)
    delta = point - (start + edge * fraction)
    return wp.dot(delta, delta)


@wp.struct
class _SimplexClosest:
    closest: wp.vec3
    weights: wp.vec4
    used: wp.int32
    valid: wp.int32


@wp.struct
class _GjkClosest:
    core_point: wp.vec3
    triangle_point: wp.vec3
    contact_point: wp.vec3
    normal: wp.vec3
    distance: wp.float32
    valid: wp.int32


@wp.func
def _simplex_triangle_origin(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> _SimplexClosest:
    """Bullet btVoronoiSimplexSolver's origin/triangle reduction."""

    result = _SimplexClosest()
    result.valid = 1
    ab = b - a
    ac = c - a
    ap = -a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        result.closest = a
        result.weights = wp.vec4(1.0, 0.0, 0.0, 0.0)
        result.used = 1
        return result

    bp = -b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        result.closest = b
        result.weights = wp.vec4(0.0, 1.0, 0.0, 0.0)
        result.used = 2
        return result

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        result.closest = a + v * ab
        result.weights = wp.vec4(1.0 - v, v, 0.0, 0.0)
        result.used = 3
        return result

    cp = -c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        result.closest = c
        result.weights = wp.vec4(0.0, 0.0, 1.0, 0.0)
        result.used = 4
        return result

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        result.closest = a + w * ac
        result.weights = wp.vec4(1.0 - w, 0.0, w, 0.0)
        result.used = 5
        return result

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        result.closest = b + w * (c - b)
        result.weights = wp.vec4(0.0, 1.0 - w, w, 0.0)
        result.used = 6
        return result

    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    result.closest = a + ab * v + ac * w
    result.weights = wp.vec4(1.0 - v - w, v, w, 0.0)
    result.used = 7
    return result


@wp.func
def _simplex_point_outside_plane(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    opposite: wp.vec3,
) -> wp.int32:
    normal = wp.cross(b - a, c - a)
    sign_origin = wp.dot(-a, normal)
    sign_opposite = wp.dot(opposite - a, normal)
    if sign_opposite * sign_opposite < 1.0e-8:
        return -1
    return wp.int32(sign_origin * sign_opposite < 0.0)


@wp.func
def _simplex_tetrahedron_origin(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
) -> _SimplexClosest:
    """Bullet's tetrahedron face selection and barycentric remapping."""

    result = _SimplexClosest()
    result.closest = wp.vec3(0.0, 0.0, 0.0)
    result.weights = wp.vec4(0.25, 0.25, 0.25, 0.25)
    result.used = 15
    result.valid = 1
    outside_abc = _simplex_point_outside_plane(a, b, c, d)
    outside_acd = _simplex_point_outside_plane(a, c, d, b)
    outside_adb = _simplex_point_outside_plane(a, d, b, c)
    outside_bdc = _simplex_point_outside_plane(b, d, c, a)
    if outside_abc < 0 or outside_acd < 0 or outside_adb < 0 or outside_bdc < 0:
        result.valid = 0
        return result
    if outside_abc == 0 and outside_acd == 0 and outside_adb == 0 and outside_bdc == 0:
        result.closest = wp.vec3(0.0, 0.0, 0.0)
        return result

    best_distance_sq = 3.402823466e38
    if outside_abc != 0:
        candidate = _simplex_triangle_origin(a, b, c)
        candidate_distance_sq = wp.dot(candidate.closest, candidate.closest)
        if candidate_distance_sq < best_distance_sq:
            best_distance_sq = candidate_distance_sq
            result.closest = candidate.closest
            result.weights = wp.vec4(
                candidate.weights[0],
                candidate.weights[1],
                candidate.weights[2],
                0.0,
            )
            result.used = candidate.used

    if outside_acd != 0:
        candidate = _simplex_triangle_origin(a, c, d)
        candidate_distance_sq = wp.dot(candidate.closest, candidate.closest)
        if candidate_distance_sq < best_distance_sq:
            best_distance_sq = candidate_distance_sq
            result.closest = candidate.closest
            result.weights = wp.vec4(
                candidate.weights[0],
                0.0,
                candidate.weights[1],
                candidate.weights[2],
            )
            result.used = wp.int32(0)
            if candidate.used & 1 != 0:
                result.used = result.used | 1
            if candidate.used & 2 != 0:
                result.used = result.used | 4
            if candidate.used & 4 != 0:
                result.used = result.used | 8

    if outside_adb != 0:
        candidate = _simplex_triangle_origin(a, d, b)
        candidate_distance_sq = wp.dot(candidate.closest, candidate.closest)
        if candidate_distance_sq < best_distance_sq:
            best_distance_sq = candidate_distance_sq
            result.closest = candidate.closest
            result.weights = wp.vec4(
                candidate.weights[0],
                candidate.weights[2],
                0.0,
                candidate.weights[1],
            )
            result.used = wp.int32(0)
            if candidate.used & 1 != 0:
                result.used = result.used | 1
            if candidate.used & 2 != 0:
                result.used = result.used | 8
            if candidate.used & 4 != 0:
                result.used = result.used | 2

    if outside_bdc != 0:
        candidate = _simplex_triangle_origin(b, d, c)
        candidate_distance_sq = wp.dot(candidate.closest, candidate.closest)
        if candidate_distance_sq < best_distance_sq:
            result.closest = candidate.closest
            result.weights = wp.vec4(
                0.0,
                candidate.weights[0],
                candidate.weights[2],
                candidate.weights[1],
            )
            result.used = wp.int32(0)
            if candidate.used & 1 != 0:
                result.used = result.used | 2
            if candidate.used & 2 != 0:
                result.used = result.used | 8
            if candidate.used & 4 != 0:
                result.used = result.used | 4
    return result


@wp.func
def _simplex_closest(
    count: wp.int32,
    w0: wp.vec3,
    w1: wp.vec3,
    w2: wp.vec3,
    w3: wp.vec3,
) -> _SimplexClosest:
    result = _SimplexClosest()
    result.valid = 1
    if count == 1:
        result.closest = w0
        result.weights = wp.vec4(1.0, 0.0, 0.0, 0.0)
        result.used = 1
        return result
    if count == 2:
        difference = w1 - w0
        projection = wp.dot(difference, -w0)
        t = 0.0
        if projection > 0.0:
            length_sq = wp.dot(difference, difference)
            if projection < length_sq:
                t = projection / length_sq
                result.used = 3
            else:
                t = 1.0
                result.used = 2
        else:
            result.used = 1
        result.closest = w0 + t * difference
        result.weights = wp.vec4(1.0 - t, t, 0.0, 0.0)
        return result
    if count == 3:
        return _simplex_triangle_origin(w0, w1, w2)
    return _simplex_tetrahedron_origin(w0, w1, w2, w3)


@wp.func
def _gjk_repeated_vertex(
    count: wp.int32,
    w: wp.vec3,
    w0: wp.vec3,
    w1: wp.vec3,
    w2: wp.vec3,
    w3: wp.vec3,
    last_w: wp.vec3,
) -> wp.int32:
    repeated = wp.int32(0)
    if count > 0 and wp.dot(w - w0, w - w0) <= GJK_EQUAL_VERTEX_THRESHOLD_BT2:
        repeated = 1
    if count > 1 and wp.dot(w - w1, w - w1) <= GJK_EQUAL_VERTEX_THRESHOLD_BT2:
        repeated = 1
    if count > 2 and wp.dot(w - w2, w - w2) <= GJK_EQUAL_VERTEX_THRESHOLD_BT2:
        repeated = 1
    if count > 3 and wp.dot(w - w3, w - w3) <= GJK_EQUAL_VERTEX_THRESHOLD_BT2:
        repeated = 1
    if w[0] == last_w[0] and w[1] == last_w[1] and w[2] == last_w[2]:
        repeated = 1
    return repeated


@wp.func
def _gjk_box_triangle(
    pos: wp.vec3,
    quat: wp.quat,
    v0_uu: wp.vec3,
    v1_uu: wp.vec3,
    v2_uu: wp.vec3,
) -> _GjkClosest:
    """Positive-distance Bullet-style GJK in native BT coordinates."""

    result = _GjkClosest()
    result.valid = 0
    body_origin = pos / 50.0
    transform_origin = body_origin + wp.quat_rotate(quat, GJK_OFFSET_BT)
    midpoint = transform_origin * 0.5
    local_origin = transform_origin - midpoint
    v0 = v0_uu / 50.0 - midpoint
    v1 = v1_uu / 50.0 - midpoint
    v2 = v2_uu / 50.0 - midpoint

    w0 = wp.vec3(0.0, 0.0, 0.0)
    w1 = wp.vec3(0.0, 0.0, 0.0)
    w2 = wp.vec3(0.0, 0.0, 0.0)
    w3 = wp.vec3(0.0, 0.0, 0.0)
    p0 = wp.vec3(0.0, 0.0, 0.0)
    p1 = wp.vec3(0.0, 0.0, 0.0)
    p2 = wp.vec3(0.0, 0.0, 0.0)
    p3 = wp.vec3(0.0, 0.0, 0.0)
    q0 = wp.vec3(0.0, 0.0, 0.0)
    q1 = wp.vec3(0.0, 0.0, 0.0)
    q2 = wp.vec3(0.0, 0.0, 0.0)
    q3 = wp.vec3(0.0, 0.0, 0.0)
    count = wp.int32(0)
    axis = wp.vec3(0.0, 1.0, 0.0)
    squared_distance = float(1.0e18)  # noqa: UP018 - Warp dynamic loop variable
    cached_core = wp.vec3(0.0, 0.0, 0.0)
    cached_triangle = wp.vec3(0.0, 0.0, 0.0)
    last_w = wp.vec3(1.0e18, 1.0e18, 1.0e18)
    have_cached = wp.int32(0)
    check_simplex = wp.int32(0)

    for _iteration in range(GJK_MAX_ITERATIONS):
        local_direction = wp.quat_rotate_inv(quat, -axis)
        local_support = wp.vec3(
            GJK_CORE_HALF_BT[0],
            GJK_CORE_HALF_BT[1],
            GJK_CORE_HALF_BT[2],
        )
        if local_direction[0] < 0.0:
            local_support[0] = -local_support[0]
        if local_direction[1] < 0.0:
            local_support[1] = -local_support[1]
        if local_direction[2] < 0.0:
            local_support[2] = -local_support[2]
        point_a = local_origin + wp.quat_rotate(quat, local_support)

        dot0 = wp.dot(axis, v0)
        dot1 = wp.dot(axis, v1)
        dot2 = wp.dot(axis, v2)
        point_b = v0
        if dot0 < dot1:
            # CPU Bullet and CUDA can round a support projection on a
            # near-parallel triangle edge in opposite directions.  This band
            # is well inside Bullet's equal-vertex squared tolerance and
            # preserves its observed later-endpoint choice.
            if dot1 < dot2 or wp.abs(dot1 - dot2) <= GJK_SUPPORT_TIE_BT2:
                point_b = v2
            else:
                point_b = v1
        elif dot0 < dot2:
            point_b = v2
        w = point_a - point_b
        repeated = _gjk_repeated_vertex(count, w, w0, w1, w2, w3, last_w)
        delta = wp.dot(axis, w)
        if delta > 0.0 and (
            delta * delta
            > squared_distance * GJK_MAXIMUM_DISTANCE_BT * GJK_MAXIMUM_DISTANCE_BT
        ):
            check_simplex = 1
            break
        if repeated != 0:
            check_simplex = 1
            break
        f0 = squared_distance - delta
        if f0 <= squared_distance * GJK_REL_ERROR2:
            check_simplex = 1
            break

        last_w = w
        if count == 0:
            w0 = w
            p0 = point_a
            q0 = point_b
        elif count == 1:
            w1 = w
            p1 = point_a
            q1 = point_b
        elif count == 2:
            w2 = w
            p2 = point_a
            q2 = point_b
        else:
            w3 = w
            p3 = point_a
            q3 = point_b
        count = count + 1

        closest = _simplex_closest(count, w0, w1, w2, w3)
        if closest.valid == 0:
            check_simplex = 1
            break
        cached_core = (
            p0 * closest.weights[0]
            + p1 * closest.weights[1]
            + p2 * closest.weights[2]
            + p3 * closest.weights[3]
        )
        cached_triangle = (
            q0 * closest.weights[0]
            + q1 * closest.weights[1]
            + q2 * closest.weights[2]
            + q3 * closest.weights[3]
        )
        have_cached = 1
        new_axis = cached_core - cached_triangle
        new_squared_distance = wp.dot(new_axis, new_axis)
        if new_squared_distance < GJK_REL_ERROR2:
            axis = new_axis
            squared_distance = new_squared_distance
            check_simplex = 1
            break
        previous_squared_distance = squared_distance
        squared_distance = new_squared_distance
        if (
            previous_squared_distance - squared_distance
            <= GJK_SIMD_EPSILON * previous_squared_distance
        ):
            check_simplex = 1
            break
        axis = new_axis

        used = closest.used
        if count >= 4 and used & 8 == 0:
            count = count - 1
        if count >= 3 and used & 4 == 0:
            if count == 4:
                w2 = w3
                p2 = p3
                q2 = q3
            count = count - 1
        if count >= 2 and used & 2 == 0:
            if count == 4:
                w1 = w3
                p1 = p3
                q1 = q3
            elif count == 3:
                w1 = w2
                p1 = p2
                q1 = q2
            count = count - 1
        if count >= 1 and used & 1 == 0:
            if count == 4:
                w0 = w3
                p0 = p3
                q0 = q3
            elif count == 3:
                w0 = w2
                p0 = p2
                q0 = q2
            elif count == 2:
                w0 = w1
                p0 = p1
                q0 = q1
            count = count - 1

        if count == 4:
            break

    axis_length_sq = wp.dot(axis, axis)
    if (
        check_simplex != 0
        and have_cached != 0
        and squared_distance > 0.0
        and axis_length_sq > GJK_SIMD_EPSILON * GJK_SIMD_EPSILON
    ):
        axis_length = wp.sqrt(axis_length_sq)
        simplex_length = wp.sqrt(squared_distance)
        normal = axis / axis_length
        result.core_point = (cached_core + midpoint) * 50.0
        result.triangle_point = (cached_triangle + midpoint) * 50.0
        result.contact_point = (
            cached_core - axis * (GJK_MARGIN_BT / simplex_length) + midpoint
        ) * 50.0
        result.normal = normal
        result.distance = (axis_length - GJK_MARGIN_BT) * 50.0
        result.valid = 1
    return result


@wp.func
def _contact_tangent(
    normal: wp.vec3, point_velocity: wp.vec3, bt_units: int
) -> wp.vec3:
    tangent = point_velocity - normal * wp.dot(point_velocity, normal)
    tangent_length_sq = wp.dot(tangent, tangent)
    tangent_threshold = 1.0e-12
    if bt_units != 0:
        tangent_threshold = GJK_SIMD_EPSILON
    if tangent_length_sq > tangent_threshold:
        return tangent / wp.sqrt(tangent_length_sq)
    if wp.abs(normal[2]) > 0.7071067811865476:
        scale = 1.0 / wp.sqrt(normal[1] * normal[1] + normal[2] * normal[2])
        return wp.vec3(0.0, -normal[2] * scale, normal[1] * scale)
    scale = 1.0 / wp.sqrt(normal[0] * normal[0] + normal[1] * normal[1])
    return wp.vec3(-normal[1] * scale, normal[0] * scale, 0.0)


@wp.func
def _bullet_matrix_quaternion(matrix: wp.mat33) -> wp.quat:
    """Match btMatrix3x3::getRotation's SSE branch and tie ordering."""

    trace = (matrix[0, 0] + matrix[1, 1]) + matrix[2, 2]
    x = float(0.0)  # noqa: UP018 - Warp dynamic branch variable
    y = float(0.0)  # noqa: UP018 - Warp dynamic branch variable
    z = float(0.0)  # noqa: UP018 - Warp dynamic branch variable
    w = float(0.0)  # noqa: UP018 - Warp dynamic branch variable
    root_argument = float(0.0)  # noqa: UP018 - Warp dynamic branch variable
    if trace > 0.0:
        root_argument = trace + 1.0
        x = matrix[2, 1] - matrix[1, 2]
        y = matrix[0, 2] - matrix[2, 0]
        z = matrix[1, 0] - matrix[0, 1]
        w = root_argument
    elif matrix[0, 0] < matrix[1, 1]:
        if matrix[1, 1] < matrix[2, 2]:
            root_argument = matrix[2, 2] - matrix[0, 0] - matrix[1, 1] + 1.0
            x = matrix[0, 2] + matrix[2, 0]
            y = matrix[1, 2] + matrix[2, 1]
            z = root_argument
            w = matrix[1, 0] - matrix[0, 1]
        else:
            root_argument = matrix[1, 1] - matrix[2, 2] - matrix[0, 0] + 1.0
            x = matrix[0, 1] + matrix[1, 0]
            y = root_argument
            z = matrix[2, 1] + matrix[1, 2]
            w = matrix[0, 2] - matrix[2, 0]
    elif matrix[0, 0] < matrix[2, 2]:
        root_argument = matrix[2, 2] - matrix[0, 0] - matrix[1, 1] + 1.0
        x = matrix[0, 2] + matrix[2, 0]
        y = matrix[1, 2] + matrix[2, 1]
        z = root_argument
        w = matrix[1, 0] - matrix[0, 1]
    else:
        root_argument = matrix[0, 0] - matrix[1, 1] - matrix[2, 2] + 1.0
        x = root_argument
        y = matrix[1, 0] + matrix[0, 1]
        z = matrix[2, 0] + matrix[0, 2]
        w = matrix[2, 1] - matrix[1, 2]
    scale = 0.5 / wp.sqrt(root_argument)
    return wp.quat(x * scale, y * scale, z * scale, w * scale)


@wp.func
def _contact_integrate_quaternion(quat: wp.quat, ang_vel: wp.vec3) -> wp.quat:
    # btTransform stores a basis, not a quaternion.  Every Bullet transform
    # integration begins by recovering orn0 from that matrix; preserving this
    # round-trip avoids carrying a subtly different predicted quaternion into
    # the next static-contact step.
    quat = _bullet_matrix_quaternion(_bullet_quaternion_matrix(quat))
    angle = wp.length(ang_vel)
    limited = wp.min(angle, (0.25 * wp.pi) / DT)
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
    # Match btQuaternion::normalize()'s SIMD horizontal reduction.  Bullet
    # sums the x/z and y/w lanes first, then combines those partials; that
    # rounding order matters after hundreds of near-rest contact steps.
    length_sq = (
        result[0] * result[0] + result[2] * result[2]
    ) + (
        result[1] * result[1] + result[3] * result[3]
    )
    if length_sq > 1.1920929e-7:
        inverse_length = 1.0 / wp.sqrt(length_sq)
        return wp.quat(
            result[0] * inverse_length,
            result[1] * inverse_length,
            result[2] * inverse_length,
            result[3] * inverse_length,
        )
    return quat


@wp.func
def _contact_cap(value: wp.vec3, maximum: float) -> wp.vec3:
    length_sq = wp.dot(value, value)
    if length_sq > maximum * maximum:
        return value * (maximum / wp.sqrt(length_sq))
    return value


@wp.func
def _rotate_axis_angle(
    value: wp.vec3, axis: wp.vec3, angle: float
) -> wp.vec3:
    axis_length_sq = wp.dot(axis, axis)
    if axis_length_sq <= 0.0:
        return value
    unit_axis = axis / wp.sqrt(axis_length_sq)
    sine = wp.sin(angle)
    cosine = wp.cos(angle)
    return (
        value * cosine
        + wp.cross(unit_axis, value) * sine
        + unit_axis * wp.dot(unit_axis, value) * (1.0 - cosine)
    )


@wp.kernel
def chassis_contacts_v021(
    aabb_mesh_id: wp.uint64,
    internal_edge_angles: wp.array(dtype=wp.vec3),
    internal_edge_flags: wp.array(dtype=wp.int32),
    bullet_bvh_rank: wp.array(dtype=wp.int32),
    inertia_transpose_mix: float,
    plane_bt_mode: int,
    support_hysteresis: float,
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    solver_position: wp.array(dtype=wp.vec3),
    rigid_position_bt: wp.array(dtype=wp.vec3),
    solver_orientation: wp.array(dtype=wp.quat),
    solver_velocity: wp.array(dtype=wp.vec3),
    rigid_velocity_bt: wp.array(dtype=wp.vec3),
    solver_angular_velocity: wp.array(dtype=wp.vec3),
    auto_roll_acceleration: wp.array(dtype=wp.vec3),
    auto_roll_angular_acceleration: wp.array(dtype=wp.vec3),
    candidate_count: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    world_contact_normal: wp.array(dtype=wp.vec3),
    candidate_total: wp.array(dtype=wp.float32),
    contact_total: wp.array(dtype=wp.float32),
    candidate_max: wp.array(dtype=wp.int32),
    contact_max: wp.array(dtype=wp.int32),
    penetration_max: wp.array(dtype=wp.float32),
    contact_point: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_tangent: wp.array(dtype=wp.vec3),
    contact_face: wp.array(dtype=wp.int32),
    contact_distance: wp.array(dtype=wp.float32),
    contact_penetration: wp.array(dtype=wp.float32),
    contact_normal_jacobian: wp.array(dtype=wp.float32),
    contact_tangent_jacobian: wp.array(dtype=wp.float32),
    contact_normal_rhs: wp.array(dtype=wp.float32),
    contact_tangent_rhs: wp.array(dtype=wp.float32),
    contact_normal_impulse: wp.array(dtype=wp.float32),
    contact_tangent_impulse: wp.array(dtype=wp.float32),
    contact_push_impulse: wp.array(dtype=wp.float32),
    contact_lifetime: wp.array(dtype=wp.int32),
    plane_support_direction: wp.array(dtype=wp.float32),
):
    """Minimal static Bullet contact generation and ten-iteration PGS solve."""

    car = wp.tid()
    pos = solver_position[car]
    quat = solver_orientation[car]
    pre_force_vel = solver_velocity[car]
    pre_force_ang_vel = solver_angular_velocity[car]
    force_vel = car_vel[car] + auto_roll_acceleration[car] * DT
    force_ang_vel = car_ang_vel[car] + auto_roll_angular_acceleration[car] * DT
    if (plane_bt_mode & 2) != 0:
        public_delta = car_vel[car] - pre_force_vel
        if (
            wp.abs(public_delta[0]) < 1.0e-6
            and wp.abs(public_delta[1]) < 1.0e-6
            and wp.abs(public_delta[2] - (-650.0 * DT)) < 1.0e-4
        ):
            force_vel = (
                pre_force_vel * 0.02 + wp.vec3(0.0, 0.0, -13.0) * DT
            ) * 50.0
    center = pos + wp.quat_rotate(quat, HITBOX_OFFSET)
    axis_x = wp.quat_rotate(quat, wp.vec3(1.0, 0.0, 0.0))
    axis_y = wp.quat_rotate(quat, wp.vec3(0.0, 1.0, 0.0))
    axis_z = wp.quat_rotate(quat, wp.vec3(0.0, 0.0, 1.0))
    aabb_half = wp.vec3(
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
    candidates = 0
    contacts = 0
    maximum_penetration = 0.0
    previous_face_0 = contact_face[car * 4]
    previous_face_1 = contact_face[car * 4 + 1]
    previous_face_2 = contact_face[car * 4 + 2]
    previous_face_3 = contact_face[car * 4 + 3]
    # btConvexPlaneCollisionAlgorithm emits one supporting vertex per plane.
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
        point = _contact_support_point(
            pos,
            rigid_position_bt[car],
            quat,
            normal,
            plane_bt_mode,
            wp.int32(
                previous_face_0 == -10 - plane
                or previous_face_1 == -10 - plane
                or previous_face_2 == -10 - plane
                or previous_face_3 == -10 - plane
            ),
            support_hysteresis,
            plane_support_direction,
            car * 12 + plane * 3,
        )
        distance = wp.dot(point - plane_point, normal)
        if (plane_bt_mode & 5) != 0 and plane == 0:
            distance = wp.dot(
                point * 0.02 - plane_point * 0.02,
                normal,
            ) * 50.0
        breaking_threshold = CONTACT_BREAKING_THRESHOLD
        if plane >= 2:
            breaking_threshold = (
                breaking_threshold - CONTACT_BREAKING_ROUNDING_GUARD
            )
        if distance < breaking_threshold and contacts < 4:
            candidates = candidates + 1
            output_index = car * 4 + contacts
            contact_point[output_index] = point
            contact_normal[output_index] = normal
            contact_face[output_index] = -10 - plane
            contact_distance[output_index] = distance
            contact_penetration[output_index] = wp.max(0.0, -distance)
            contact_lifetime[output_index] = 1
            maximum_penetration = wp.max(maximum_penetration, -distance)
            contacts = contacts + 1

    mesh_contact_start = contacts
    # Retain the v0.2 SAT candidate path for CMF triangles while the v0.2.1
    # solver below supplies Bullet ordering, effective mass and split impulse.
    query = wp.mesh_query_aabb(aabb_mesh_id, center - aabb_half, center + aabb_half)
    for face in query:
        candidates = candidates + 1
        if mesh_contact_start < 4:
            v0 = wp.mesh_eval_position(aabb_mesh_id, face, 1.0, 0.0)
            v1 = wp.mesh_eval_position(aabb_mesh_id, face, 0.0, 1.0)
            v2 = wp.mesh_eval_position(aabb_mesh_id, face, 0.0, 0.0)
            sat = _triangle_obb_sat(v0, v1, v2, center, quat)
            penetration = sat[3]
            if penetration >= -CONTACT_BREAKING_THRESHOLD and (
                sat[0] * sat[0] + sat[1] * sat[1] + sat[2] * sat[2] > 0.5
            ):
                local_normal = wp.vec3(sat[0], sat[1], sat[2])
                normal = wp.quat_rotate(quat, local_normal)
                # Bullet's box/triangle detector works against the marginless
                # box, then moves point A outward by the convex margin.  A
                # face-axis contact is a core support point projected to the
                # triangle; an edge-axis contact is the closest pair between
                # the implicated core-box edge and each triangle edge.
                core_point = _contact_core_support_point(pos, quat, normal)
                triangle_point = _closest_point_triangle(core_point, v0, v1, v2)
                abs_local_normal = wp.vec3(
                    wp.abs(local_normal[0]),
                    wp.abs(local_normal[1]),
                    wp.abs(local_normal[2]),
                )
                minimum_component = wp.min(
                    abs_local_normal[0],
                    wp.min(abs_local_normal[1], abs_local_normal[2]),
                )
                maximum_component = wp.max(
                    abs_local_normal[0],
                    wp.max(abs_local_normal[1], abs_local_normal[2]),
                )
                if minimum_component < 1.0e-3 and maximum_component < 0.999:
                    edge_axis = 0
                    if abs_local_normal[1] < abs_local_normal[edge_axis]:
                        edge_axis = 1
                    if abs_local_normal[2] < abs_local_normal[edge_axis]:
                        edge_axis = 2

                    # Select the core-box edge toward the triangle (the
                    # support direction is -normal), leaving the implicated
                    # box-axis component free over the entire edge.
                    local_start = wp.vec3(
                        HITBOX_CORE_HALF[0],
                        HITBOX_CORE_HALF[1],
                        HITBOX_CORE_HALF[2],
                    )
                    if local_normal[0] > 0.0:
                        local_start[0] = -local_start[0]
                    if local_normal[1] > 0.0:
                        local_start[1] = -local_start[1]
                    if local_normal[2] > 0.0:
                        local_start[2] = -local_start[2]
                    local_end = local_start
                    if edge_axis == 0:
                        local_start[0] = -HITBOX_CORE_HALF[0]
                        local_end[0] = HITBOX_CORE_HALF[0]
                    elif edge_axis == 1:
                        local_start[1] = -HITBOX_CORE_HALF[1]
                        local_end[1] = HITBOX_CORE_HALF[1]
                    else:
                        local_start[2] = -HITBOX_CORE_HALF[2]
                        local_end[2] = HITBOX_CORE_HALF[2]

                    box_start = pos + wp.quat_rotate(
                        quat, HITBOX_OFFSET + local_start
                    )
                    box_end = pos + wp.quat_rotate(quat, HITBOX_OFFSET + local_end)
                    best_distance_sq = 1.0e30
                    for triangle_edge in range(3):
                        triangle_start = v0
                        triangle_end = v1
                        if triangle_edge == 1:
                            triangle_start = v1
                            triangle_end = v2
                        elif triangle_edge == 2:
                            triangle_start = v2
                            triangle_end = v0
                        parameters = _closest_segment_parameters(
                            box_start,
                            box_end,
                            triangle_start,
                            triangle_end,
                        )
                        candidate_core = box_start + (box_end - box_start) * parameters[0]
                        candidate_triangle = triangle_start + (
                            triangle_end - triangle_start
                        ) * parameters[1]
                        candidate_delta = candidate_triangle - candidate_core
                        candidate_distance_sq = wp.dot(candidate_delta, candidate_delta)
                        if candidate_distance_sq < best_distance_sq:
                            best_distance_sq = candidate_distance_sq
                            core_point = candidate_core
                            triangle_point = candidate_triangle

                    # For a separated edge/edge pair, Bullet's GJK detector
                    # reports the normalized closest-pair vector, rather than
                    # the SAT axis used only to identify the feature pair.
                    # Reconstruct that direction before applying the convex
                    # margin so the manifold normal and distance agree.

                point = core_point - normal * HITBOX_MARGIN
                distance = -HITBOX_MARGIN - wp.dot(
                    triangle_point - core_point,
                    normal,
                )

                # Bullet runs btGjkPairDetector even after the outer-margin
                # SAT overlaps.  Its finite-precision repeat threshold
                # intentionally preserves the current simplex witness, which
                # can differ materially from a closest-feature construction.
                # Below Bullet's 0.01-BT degenerate-core threshold, retain the
                # penetration fallback assembled above; RocketSim routes that
                # region through its penetration-depth solver.
                gjk = _gjk_box_triangle(pos, quat, v0, v1, v2)
                if (
                    gjk.valid != 0
                    and penetration <= HITBOX_MARGIN
                    and gjk.distance + HITBOX_MARGIN >= 0.5
                ):
                    core_point = gjk.core_point
                    triangle_point = gjk.triangle_point
                    normal = gjk.normal
                    point = gjk.contact_point
                    distance = gjk.distance
                # RocketSim invokes btAdjustInternalEdgeContacts for every CMF
                # manifold point. Reproduce its closest-edge selection,
                # planar/concave normal replacement, and convex angle clamp.
                edge_angles = internal_edge_angles[face]
                edge_flags = internal_edge_flags[face]
                best_edge = -1
                best_edge_distance_sq = 3.402823466e38
                for edge in range(3):
                    edge_angle = edge_angles[edge]
                    if wp.abs(edge_angle) < 6.283185307179586:
                        edge_start = v0
                        edge_end = v1
                        if edge == 1:
                            edge_start = v1
                            edge_end = v2
                        elif edge == 2:
                            edge_start = v2
                            edge_end = v0
                        edge_distance_sq = _point_segment_distance_sq(
                            triangle_point, edge_start, edge_end
                        )
                        if edge_distance_sq < best_edge_distance_sq:
                            best_edge_distance_sq = edge_distance_sq
                            best_edge = edge

                if best_edge >= 0 and best_edge_distance_sq < 25.0:
                    triangle_normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))
                    local_contact_normal = wp.normalize(normal)
                    edge_angle = edge_angles[best_edge]
                    if edge_angle == 0.0:
                        if wp.dot(triangle_normal, local_contact_normal) >= 0.0:
                            normal = triangle_normal
                    else:
                        edge_start = v0
                        edge_end = v1
                        if best_edge == 1:
                            edge_start = v1
                            edge_end = v2
                        elif best_edge == 2:
                            edge_start = v2
                            edge_end = v0
                        edge_vector = edge_start - edge_end
                        is_convex = edge_flags & (1 << best_edge) != 0
                        swap_factor = -1.0
                        if is_convex:
                            swap_factor = 1.0
                        normal_a = triangle_normal * swap_factor
                        normal_b = _rotate_axis_angle(
                            triangle_normal, edge_vector, edge_angle
                        )
                        if edge_flags & (1 << (best_edge + 3)) != 0:
                            normal_b = -normal_b
                        normal_b = normal_b * swap_factor
                        back_facing = (
                            wp.dot(local_contact_normal, normal_a) < 0.0
                            and wp.dot(local_contact_normal, normal_b) < 0.0
                        )
                        if back_facing:
                            if wp.dot(triangle_normal, local_contact_normal) >= 0.0:
                                normal = triangle_normal
                        else:
                            edge_cross = wp.normalize(
                                wp.cross(edge_vector, normal_a)
                            )
                            current_angle = wp.atan2(
                                wp.dot(local_contact_normal, edge_cross),
                                wp.dot(local_contact_normal, normal_a),
                            )
                            clamp = 0
                            if edge_angle < 0.0 and current_angle < edge_angle:
                                clamp = 1
                            elif edge_angle >= 0.0 and current_angle > edge_angle:
                                clamp = 1
                            if clamp != 0:
                                clamped_normal = _rotate_axis_angle(
                                    local_contact_normal,
                                    edge_vector,
                                    edge_angle - current_angle,
                                )
                                if wp.dot(clamped_normal, triangle_normal) > 0.0:
                                    normal = clamped_normal
                if distance < CONTACT_BREAKING_THRESHOLD:
                    # Warp's BVH does not promise Bullet's leaf visitation
                    # order. Keep the four earliest accepted constraints in
                    # the pinned btQuantizedBvh depth-first leaf order while
                    # retaining the native solver's plane-manifold prefix;
                    # sequential PGS makes both orderings part of the ABI.
                    insert_index = contacts
                    candidate_rank = bullet_bvh_rank[face]
                    for existing in range(4):
                        if existing >= mesh_contact_start and existing < contacts:
                            existing_face = contact_face[car * 4 + existing]
                            if (
                                insert_index == contacts
                                and candidate_rank < bullet_bvh_rank[existing_face]
                            ):
                                insert_index = existing
                    retained_contacts = wp.min(contacts + 1, 4)
                    for shift_offset in range(3):
                        destination = 3 - shift_offset
                        if (
                            destination > insert_index
                            and destination < retained_contacts
                        ):
                            source = car * 4 + destination - 1
                            shifted = car * 4 + destination
                            contact_point[shifted] = contact_point[source]
                            contact_normal[shifted] = contact_normal[source]
                            contact_face[shifted] = contact_face[source]
                            contact_distance[shifted] = contact_distance[source]
                            contact_penetration[shifted] = contact_penetration[source]
                            contact_lifetime[shifted] = contact_lifetime[source]
                    output_index = car * 4 + insert_index
                    contact_point[output_index] = point
                    contact_normal[output_index] = normal
                    contact_face[output_index] = face
                    contact_distance[output_index] = distance
                    contact_penetration[output_index] = wp.max(0.0, -distance)
                    contact_lifetime[output_index] = 1
                    maximum_penetration = wp.max(maximum_penetration, -distance)
                    contacts = retained_contacts

    solve_bt = 0
    if (
        (plane_bt_mode & 4) != 0
        and contacts == 1
        and contact_face[car * 4] == -10
    ):
        solve_bt = 1
    solver_pos_units = pos
    solver_pre_force_vel = pre_force_vel
    solver_force_vel = force_vel
    if solve_bt != 0:
        solver_pos_units = rigid_position_bt[car]
        prior_public_velocity = rigid_velocity_bt[car] * 50.0
        solver_pre_force_vel = rigid_velocity_bt[car] + (
            pre_force_vel - prior_public_velocity
        ) * 0.02
        solver_force_vel = solver_pre_force_vel + (
            force_vel - pre_force_vel
        ) * 0.02
        public_delta = car_vel[car] - pre_force_vel
        if (
            (plane_bt_mode & 2) != 0
            and wp.abs(public_delta[0]) < 1.0e-6
            and wp.abs(public_delta[1]) < 1.0e-6
            and wp.abs(public_delta[2] - (-650.0 * DT)) < 1.0e-4
        ):
            solver_force_vel = solver_pre_force_vel + wp.vec3(0.0, 0.0, -13.0) * DT

    # Set up every constraint from the same unchanged rigid-body state, as
    # Bullet does before applying warmstart or iteration deltas.
    for index in range(4):
        output_index = car * 4 + index
        if index < contacts:
            point = contact_point[output_index]
            if solve_bt != 0:
                point = point * 0.02
            normal = contact_normal[output_index]
            offset = point - solver_pos_units
            force_point_velocity = solver_force_vel + wp.cross(force_ang_vel, offset)
            pre_force_point_velocity = solver_pre_force_vel + wp.cross(
                pre_force_ang_vel, offset
            )
            friction_rhs_point_velocity = solver_force_vel + wp.cross(
                pre_force_ang_vel, offset
            )
            normal_denominator = _impulse_denominator(
                quat, offset, normal, inertia_transpose_mix, solve_bt
            )
            normal_jacobian = 0.0
            if normal_denominator > 1.0e-9:
                normal_jacobian = 1.0 / normal_denominator
            pre_force_normal_speed = wp.dot(pre_force_point_velocity, normal)
            restitution = 0.0
            restitution_threshold = 10.0
            if solve_bt != 0:
                restitution_threshold = 0.2
            if wp.abs(pre_force_normal_speed) >= restitution_threshold:
                restitution = wp.max(0.0, -CONTACT_RESTITUTION * pre_force_normal_speed)
            normal_rhs = (
                restitution - wp.dot(force_point_velocity, normal)
            ) * normal_jacobian
            tangent = _contact_tangent(normal, force_point_velocity, solve_bt)
            tangent_denominator = _impulse_denominator(
                quat, offset, tangent, inertia_transpose_mix, solve_bt
            )
            tangent_jacobian = 0.0
            if tangent_denominator > 1.0e-9:
                tangent_jacobian = 1.0 / tangent_denominator
            # Bullet uses external angular impulses to choose the lateral
            # direction, but its friction-row RHS includes only the external
            # linear impulse. Preserve that asymmetry exactly.
            tangent_rhs = (
                -wp.dot(friction_rhs_point_velocity, tangent) * tangent_jacobian
            )
            contact_tangent[output_index] = tangent
            contact_normal_jacobian[output_index] = normal_jacobian
            contact_tangent_jacobian[output_index] = tangent_jacobian
            contact_normal_rhs[output_index] = normal_rhs
            contact_tangent_rhs[output_index] = tangent_rhs
            contact_normal_impulse[output_index] = 0.0
            contact_tangent_impulse[output_index] = 0.0
            contact_push_impulse[output_index] = 0.0
            if contact_lifetime[output_index] <= 0:
                contact_lifetime[output_index] = 1
        else:
            contact_point[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_normal[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_tangent[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_face[output_index] = -1
            contact_distance[output_index] = 0.0
            contact_penetration[output_index] = 0.0
            contact_normal_jacobian[output_index] = 0.0
            contact_tangent_jacobian[output_index] = 0.0
            contact_normal_rhs[output_index] = 0.0
            contact_tangent_rhs[output_index] = 0.0
            contact_normal_impulse[output_index] = 0.0
            contact_tangent_impulse[output_index] = 0.0
            contact_push_impulse[output_index] = 0.0
            contact_lifetime[output_index] = 0

    # RocketSim sets the split threshold to +1e30, so every penetrating static
    # contact is recovered through push/turn velocity, independently of bounce.
    push_vel = wp.vec3(0.0, 0.0, 0.0)
    turn_vel = wp.vec3(0.0, 0.0, 0.0)
    for _iteration in range(CONTACT_SOLVER_ITERATIONS):
        for index in range(4):
            if index < contacts:
                output_index = car * 4 + index
                distance = contact_distance[output_index] + CONTACT_LINEAR_SLOP
                if solve_bt != 0:
                    distance = distance * 0.02
                if distance < 0.0:
                    normal = contact_normal[output_index]
                    point = contact_point[output_index]
                    if solve_bt != 0:
                        point = point * 0.02
                    offset = point - solver_pos_units
                    jacobian = contact_normal_jacobian[output_index]
                    rhs = -distance * CONTACT_ERP2 / DT * jacobian
                    old_impulse = contact_push_impulse[output_index]
                    point_push_speed = wp.dot(
                        normal, push_vel + wp.cross(turn_vel, offset)
                    )
                    delta_impulse = rhs - point_push_speed * jacobian
                    new_impulse = wp.max(0.0, old_impulse + delta_impulse)
                    delta_impulse = new_impulse - old_impulse
                    contact_push_impulse[output_index] = new_impulse
                    push_vel = push_vel + normal * (delta_impulse * INV_MASS)
                    turn_vel = turn_vel + _inverse_inertia_world(
                        quat,
                        wp.cross(offset, normal * delta_impulse),
                        inertia_transpose_mix,
                        solve_bt,
                    )

    delta_vel = wp.vec3(0.0, 0.0, 0.0)
    delta_ang_vel = wp.vec3(0.0, 0.0, 0.0)
    for _iteration in range(CONTACT_SOLVER_ITERATIONS):
        for index in range(4):
            if index < contacts:
                output_index = car * 4 + index
                normal = contact_normal[output_index]
                point = contact_point[output_index]
                if solve_bt != 0:
                    point = point * 0.02
                offset = point - solver_pos_units
                jacobian = contact_normal_jacobian[output_index]
                old_impulse = contact_normal_impulse[output_index]
                point_delta_speed = wp.dot(
                    normal, delta_vel + wp.cross(delta_ang_vel, offset)
                )
                delta_impulse = (
                    contact_normal_rhs[output_index] - point_delta_speed * jacobian
                )
                new_impulse = wp.max(0.0, old_impulse + delta_impulse)
                delta_impulse = new_impulse - old_impulse
                contact_normal_impulse[output_index] = new_impulse
                delta_vel = delta_vel + normal * (delta_impulse * INV_MASS)
                delta_ang_vel = delta_ang_vel + _inverse_inertia_world(
                    quat,
                    wp.cross(offset, normal * delta_impulse),
                    inertia_transpose_mix,
                    solve_bt,
                )
        for index in range(4):
            if index < contacts:
                output_index = car * 4 + index
                # Bullet skips a friction row when its paired normal impulse is
                # zero.  It does not clamp an impulse accumulated by an earlier
                # iteration back to zero, which matters as coupled contacts
                # transfer load during the PGS sweep.
                if contact_normal_impulse[output_index] > 0.0:
                    tangent = contact_tangent[output_index]
                    point = contact_point[output_index]
                    if solve_bt != 0:
                        point = point * 0.02
                    offset = point - solver_pos_units
                    jacobian = contact_tangent_jacobian[output_index]
                    old_impulse = contact_tangent_impulse[output_index]
                    point_delta_speed = wp.dot(
                        tangent, delta_vel + wp.cross(delta_ang_vel, offset)
                    )
                    delta_impulse = (
                        contact_tangent_rhs[output_index]
                        - point_delta_speed * jacobian
                    )
                    limit = CONTACT_FRICTION * contact_normal_impulse[output_index]
                    new_impulse = wp.clamp(
                        old_impulse + delta_impulse, -limit, limit
                    )
                    delta_impulse = new_impulse - old_impulse
                    contact_tangent_impulse[output_index] = new_impulse
                    delta_vel = delta_vel + tangent * (delta_impulse * INV_MASS)
                    delta_ang_vel = delta_ang_vel + _inverse_inertia_world(
                        quat,
                        wp.cross(offset, tangent * delta_impulse),
                        inertia_transpose_mix,
                        solve_bt,
                    )

    solved_vel = solver_force_vel + delta_vel
    solved_ang_vel = force_ang_vel + delta_ang_vel
    split_quat = quat
    if (
        push_vel[0] != 0.0
        or push_vel[1] != 0.0
        or push_vel[2] != 0.0
        or turn_vel[0] != 0.0
        or turn_vel[1] != 0.0
        or turn_vel[2] != 0.0
    ):
        split_quat = _contact_integrate_quaternion(
            quat, turn_vel * CONTACT_SPLIT_TURN_ERP
        )
    # Bullet first writes the split-impulse transform back to the rigid body,
    # then integrates the final velocity from that corrected transform.
    split_pos = solver_pos_units + push_vel * DT
    solved_pos = split_pos + solved_vel * DT
    if (
        solve_bt == 0
        and (plane_bt_mode & 8) != 0
        and contacts > 0
        and contact_face[car * 4] < 0
    ):
        split_pos_bt = pos * 0.02 + push_vel * 0.02 * DT
        solved_pos = (split_pos_bt + solved_vel * 0.02 * DT) * 50.0
    solved_quat = _contact_integrate_quaternion(split_quat, solved_ang_vel)

    # RocketSim caps after Bullet has integrated the transform.
    public_pos = solved_pos
    public_vel = _contact_cap(solved_vel, 2300.0)
    if solve_bt != 0:
        capped_vel_bt = _contact_cap(solved_vel, 46.0)
        rigid_position_bt[car] = solved_pos
        rigid_velocity_bt[car] = capped_vel_bt
        public_pos = solved_pos * 50.0
        public_vel = capped_vel_bt * 50.0
        for index in range(4):
            if index < contacts:
                output_index = car * 4 + index
                contact_normal_impulse[output_index] = (
                    contact_normal_impulse[output_index] * 50.0
                )
                contact_tangent_impulse[output_index] = (
                    contact_tangent_impulse[output_index] * 50.0
                )
                contact_push_impulse[output_index] = (
                    contact_push_impulse[output_index] * 50.0
                )
    else:
        rigid_position_bt[car] = public_pos * 0.02
        rigid_velocity_bt[car] = public_vel * 0.02
    car_pos[car] = public_pos
    car_quat[car] = solved_quat
    car_vel[car] = public_vel
    car_ang_vel[car] = _contact_cap(solved_ang_vel, 5.5)
    candidate_count[car] = candidates
    contact_count[car] = contacts
    callback_normal = wp.vec3(0.0, 0.0, 0.0)
    if contacts > 0:
        callback_index = car * 4
        last_callback_index = car * 4 + contacts - 1
        if contacts > 1:
            first_callback_normal = contact_normal[car * 4]
            last_callback_normal = contact_normal[last_callback_index]
            if wp.dot(first_callback_normal, last_callback_normal) < 0.9999:
                callback_index = last_callback_index
        for callback_contact in range(4):
            if (
                callback_contact < contacts
                and contact_face[car * 4 + callback_contact] < 0
            ):
                callback_index = car * 4 + callback_contact
        callback_normal = contact_normal[callback_index]
    world_contact_normal[car] = callback_normal
    candidate_total[car] = candidate_total[car] + float(candidates)
    contact_total[car] = contact_total[car] + float(contacts)
    candidate_max[car] = wp.max(candidate_max[car], candidates)
    contact_max[car] = wp.max(contact_max[car], contacts)
    penetration_max[car] = wp.max(penetration_max[car], maximum_penetration)
