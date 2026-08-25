"""Bounded shared Bullet island for two Octanes contacting one Soccar ball."""

import warp as wp

from rivalsim.ball_world_state import MAX_BALL_CONTACTS
from rivalsim.car_ball_state import MAX_CAR_BALL_CONTACTS
from rivalsim.kernels.ball_world import (
    _bullet_ball_solve_split_row,
    _bullet_ball_solve_velocity_row,
    _bullet_ball_special_contact_row,
)
from rivalsim.kernels.car_ball import (
    BALL_DAMPING,
    BALL_MAX_ANGULAR_SPEED,
    BALL_MAX_SPEED_BT,
    BALL_STATIC_FRICTION,
    BOX_HALF_WITH_MARGIN_BT,
    CAR_MAX_ANGULAR_SPEED,
    CAR_MAX_SPEED_BT,
    CAR_STATIC_FRICTION,
    CHILD_OFFSET_BT,
    CONTACT_FRICTION,
    CONTACT_SPLIT_TURN_ERP,
    DT,
    RS_CAR_PROXY_AABB_EXPANSION_BT,
    RS_EQUAL_ISLAND_PERMUTATION_WIDTH,
    RS_MAX_ISLAND_MANIFOLDS,
    SOLVER_ITERATIONS,
    _bullet_cap_amd,
    _bullet_pair_solve_split_row,
    _bullet_pair_solve_velocity_row,
    _bullet_special_average,
    _bullet_special_distance_accumulate,
    _matrix_vector,
    _rs_cell_index,
    _rs_mask_count,
    _rs_nth_mask_body,
    _rs_static_overlap_mask,
)
from rivalsim.kernels.vehicle import (
    _authority_input_quaternion_matrix,
    _bullet_integrate_external_velocities,
    _bullet_integrate_position,
    _bullet_integrate_quaternion,
    _bullet_quaternion_matrix,
    _bullet_solve_split_row,
    _bullet_solve_velocity_row,
    _bullet_transform_point,
)
from rivalsim.vehicle_state import MAX_CONTACTS_PER_CAR


@wp.func
def _integrated_sorted_manifold_code(
    sorted_position: int,
    ball_mask: wp.uint32,
    car_a_mask: wp.uint32,
    car_b_mask: wp.uint32,
    ball_manifolds: int,
    pair_a_active: int,
    pair_b_active: int,
    pair_a_before_b: int,
    car_a_manifolds: int,
    car_car_active: int,
    total_manifolds: int,
    equal_island_permutation: wp.array(dtype=wp.int32),
) -> int:
    """Map Bullet's equal-island quicksort slot to the source pair order.

    btRSBroadphase creates manifolds in handle order for this fixed world:
    ball/static, ball/dynamic-cell order, A/static, A/B, then B/static.
    The per-cell vector order is lifecycle state updated by proxy cell moves.
    """

    source = equal_island_permutation[
        total_manifolds * RS_EQUAL_ISLAND_PERMUTATION_WIDTH + sorted_position
    ]
    cursor = ball_manifolds
    if source < cursor:
        return _rs_nth_mask_body(ball_mask, source)
    if pair_a_active != 0 and pair_b_active != 0:
        if source == cursor:
            return 32 if pair_a_before_b != 0 else 33
        if source == cursor + 1:
            return 33 if pair_a_before_b != 0 else 32
        cursor = cursor + 2
    elif pair_a_active != 0:
        if source == cursor:
            return 32
        cursor = cursor + 1
    elif pair_b_active != 0:
        if source == cursor:
            return 33
        cursor = cursor + 1
    if source < cursor + car_a_manifolds:
        return 64 + _rs_nth_mask_body(car_a_mask, source - cursor)
    cursor = cursor + car_a_manifolds
    if car_car_active != 0:
        if source == cursor:
            return 96
        cursor = cursor + 1
    return 128 + _rs_nth_mask_body(car_b_mask, source - cursor)


@wp.func
def _integrated_car_proxy_min(
    position: wp.vec3,
    basis: wp.mat33,
    predicted_position: wp.vec3,
    predicted_basis: wp.mat33,
) -> wp.vec3:
    center = position + _matrix_vector(basis, CHILD_OFFSET_BT)
    predicted_center = predicted_position + _matrix_vector(
        predicted_basis, CHILD_OFFSET_BT
    )
    half = BOX_HALF_WITH_MARGIN_BT
    extent = wp.vec3(
        wp.abs(basis[0, 0]) * half[0]
        + wp.abs(basis[0, 1]) * half[1]
        + wp.abs(basis[0, 2]) * half[2],
        wp.abs(basis[1, 0]) * half[0]
        + wp.abs(basis[1, 1]) * half[1]
        + wp.abs(basis[1, 2]) * half[2],
        wp.abs(basis[2, 0]) * half[0]
        + wp.abs(basis[2, 1]) * half[1]
        + wp.abs(basis[2, 2]) * half[2],
    ) + wp.vec3(RS_CAR_PROXY_AABB_EXPANSION_BT)
    predicted_extent = wp.vec3(
        wp.abs(predicted_basis[0, 0]) * half[0]
        + wp.abs(predicted_basis[0, 1]) * half[1]
        + wp.abs(predicted_basis[0, 2]) * half[2],
        wp.abs(predicted_basis[1, 0]) * half[0]
        + wp.abs(predicted_basis[1, 1]) * half[1]
        + wp.abs(predicted_basis[1, 2]) * half[2],
        wp.abs(predicted_basis[2, 0]) * half[0]
        + wp.abs(predicted_basis[2, 1]) * half[1]
        + wp.abs(predicted_basis[2, 2]) * half[2],
    ) + wp.vec3(RS_CAR_PROXY_AABB_EXPANSION_BT)
    current_min = center - extent
    predicted_min = predicted_center - predicted_extent
    return wp.vec3(
        wp.min(current_min[0], predicted_min[0]),
        wp.min(current_min[1], predicted_min[1]),
        wp.min(current_min[2], predicted_min[2]),
    )


@wp.kernel(enable_backward=False, module="unique")
def update_integrated_broadphase_order(
    tick_counter: wp.array(dtype=wp.int32),
    pre_car_position_bt: wp.array(dtype=wp.vec3),
    pre_car_velocity_bt: wp.array(dtype=wp.vec3),
    pre_car_quaternion: wp.array(dtype=wp.quat),
    pre_car_angular_velocity: wp.array(dtype=wp.vec3),
    pre_ball_position_bt: wp.array(dtype=wp.vec3),
    pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    pre_ball_quaternion: wp.array(dtype=wp.quat),
    pre_ball_angular_velocity: wp.array(dtype=wp.vec3),
    proxy_cell: wp.array(dtype=wp.int32),
    proxy_move_rank: wp.array(dtype=wp.int32),
    move_counter: wp.array(dtype=wp.int32),
    pair_a_before_b: wp.array(dtype=wp.int32),
):
    """Port the btRSBroadphase dynamic-cell vector lifecycle for three bodies."""

    env = wp.tid()
    car_a = env * 2
    car_b = car_a + 1
    position_a = pre_car_position_bt[car_a]
    position_b = pre_car_position_bt[car_b]
    quaternion_a = pre_car_quaternion[car_a]
    quaternion_b = pre_car_quaternion[car_b]
    basis_a = _bullet_quaternion_matrix(quaternion_a)
    basis_b = _bullet_quaternion_matrix(quaternion_b)
    ball_position = pre_ball_position_bt[env]
    if tick_counter[0] == 0:
        basis_a = _authority_input_quaternion_matrix(quaternion_a)
        basis_b = _authority_input_quaternion_matrix(quaternion_b)

    predicted_a = position_a
    predicted_b = position_b
    _bullet_integrate_position(
        position_a,
        wp.vec3(0.0),
        pre_car_velocity_bt[car_a],
        DT,
        0,
        predicted_a,
    )
    _bullet_integrate_position(
        position_b,
        wp.vec3(0.0),
        pre_car_velocity_bt[car_b],
        DT,
        0,
        predicted_b,
    )
    predicted_quaternion_a = quaternion_a
    predicted_quaternion_b = quaternion_b
    _bullet_integrate_quaternion(
        basis_a,
        pre_car_angular_velocity[car_a],
        DT,
        predicted_quaternion_a,
    )
    _bullet_integrate_quaternion(
        basis_b,
        pre_car_angular_velocity[car_b],
        DT,
        predicted_quaternion_b,
    )
    predicted_basis_a = _bullet_quaternion_matrix(predicted_quaternion_a)
    predicted_basis_b = _bullet_quaternion_matrix(predicted_quaternion_b)
    ball_pre_velocity = pre_ball_velocity_bt[env] * wp.pow(1.0 - BALL_DAMPING, DT)
    predicted_ball = ball_position
    _bullet_integrate_position(
        ball_position,
        wp.vec3(0.0),
        ball_pre_velocity,
        DT,
        0,
        predicted_ball,
    )

    car_a_cell = _rs_cell_index(
        _integrated_car_proxy_min(
            position_a, basis_a, predicted_a, predicted_basis_a
        )
    )
    car_b_cell = _rs_cell_index(
        _integrated_car_proxy_min(
            position_b, basis_b, predicted_b, predicted_basis_b
        )
    )
    ball_half = wp.vec3(1.925)
    ball_current_min = ball_position - ball_half
    ball_predicted_min = predicted_ball - ball_half
    ball_cell = _rs_cell_index(
        wp.vec3(
            wp.min(ball_current_min[0], ball_predicted_min[0]),
            wp.min(ball_current_min[1], ball_predicted_min[1]),
            wp.min(ball_current_min[2], ball_predicted_min[2]),
        )
    )

    base = env * 3
    counter = move_counter[env]
    # btCollisionWorld::updateAabbs walks the fixed collision-object array:
    # ball first, then A, then B. A base-cell change removes the proxy from
    # each old cell and appends it to every new cell in exactly this order.
    if proxy_cell[base] != ball_cell:
        counter = counter + 1
        proxy_cell[base] = ball_cell
        proxy_move_rank[base] = counter
    if proxy_cell[base + 1] != car_a_cell:
        counter = counter + 1
        proxy_cell[base + 1] = car_a_cell
        proxy_move_rank[base + 1] = counter
    if proxy_cell[base + 2] != car_b_cell:
        counter = counter + 1
        proxy_cell[base + 2] = car_b_cell
        proxy_move_rank[base + 2] = counter
    move_counter[env] = counter
    pair_a_before_b[env] = wp.int32(
        proxy_move_rank[base + 1] < proxy_move_rank[base + 2]
    )


@wp.kernel(
    enable_backward=False,
    module="unique",
    module_options={"max_unroll": 4},
)
def integrated_two_car_ball_tick(
    tick_counter: wp.array(dtype=wp.int32),
    amd_rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    rs_static_cell_mask: wp.array(dtype=wp.uint32),
    rs_static_aabb_min_bt: wp.array(dtype=wp.vec3),
    rs_static_aabb_max_bt: wp.array(dtype=wp.vec3),
    rs_equal_island_permutation: wp.array(dtype=wp.int32),
    pair_a_before_b: wp.array(dtype=wp.int32),
    total_force_bt: wp.array(dtype=wp.vec3),
    total_torque_bt: wp.array(dtype=wp.vec3),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    car_is_supersonic: wp.array(dtype=wp.int32),
    car_supersonic_time: wp.array(dtype=wp.float32),
    car_rigid_position_bt: wp.array(dtype=wp.vec3),
    car_rigid_velocity_bt: wp.array(dtype=wp.vec3),
    car_static_contact_count: wp.array(dtype=wp.int32),
    car_static_local_a_bt: wp.array(dtype=wp.vec3),
    car_static_normal: wp.array(dtype=wp.vec3),
    car_static_tangent: wp.array(dtype=wp.vec3),
    car_static_mesh: wp.array(dtype=wp.int32),
    car_static_normal_jacobian: wp.array(dtype=wp.float32),
    car_static_tangent_jacobian: wp.array(dtype=wp.float32),
    car_static_normal_rhs: wp.array(dtype=wp.float32),
    car_static_tangent_rhs: wp.array(dtype=wp.float32),
    car_static_push_rhs: wp.array(dtype=wp.float32),
    car_static_normal_impulse: wp.array(dtype=wp.float32),
    car_static_tangent_impulse: wp.array(dtype=wp.float32),
    car_static_push_impulse: wp.array(dtype=wp.float32),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_quat: wp.array(dtype=wp.quat),
    ball_ang_vel: wp.array(dtype=wp.vec3),
    ball_resident_position_bt: wp.array(dtype=wp.vec3),
    ball_resident_velocity_bt: wp.array(dtype=wp.vec3),
    ball_static_contact_count: wp.array(dtype=wp.int32),
    ball_static_local_a_bt: wp.array(dtype=wp.vec3),
    ball_static_normal: wp.array(dtype=wp.vec3),
    ball_static_tangent: wp.array(dtype=wp.vec3),
    ball_static_mesh: wp.array(dtype=wp.int32),
    ball_static_normal_jacobian: wp.array(dtype=wp.float32),
    ball_static_tangent_jacobian: wp.array(dtype=wp.float32),
    ball_static_normal_rhs: wp.array(dtype=wp.float32),
    ball_static_tangent_rhs: wp.array(dtype=wp.float32),
    ball_static_push_rhs: wp.array(dtype=wp.float32),
    ball_static_normal_impulse: wp.array(dtype=wp.float32),
    ball_static_tangent_impulse: wp.array(dtype=wp.float32),
    ball_static_push_impulse: wp.array(dtype=wp.float32),
    pre_car_position_bt: wp.array(dtype=wp.vec3),
    pre_car_velocity_bt: wp.array(dtype=wp.vec3),
    pre_car_quaternion: wp.array(dtype=wp.quat),
    pre_car_angular_velocity: wp.array(dtype=wp.vec3),
    pre_car_is_supersonic: wp.array(dtype=wp.int32),
    pre_car_supersonic_time: wp.array(dtype=wp.float32),
    pre_ball_position_bt: wp.array(dtype=wp.vec3),
    pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    pre_ball_quaternion: wp.array(dtype=wp.quat),
    pre_ball_angular_velocity: wp.array(dtype=wp.vec3),
    pair_a_algorithm_active: wp.array(dtype=wp.int32),
    pair_a_contact_count: wp.array(dtype=wp.int32),
    pair_a_manifold_local_a_bt: wp.array(dtype=wp.vec3),
    pair_a_manifold_local_b_bt: wp.array(dtype=wp.vec3),
    pair_a_manifold_normal: wp.array(dtype=wp.vec3),
    pair_a_manifold_tangent: wp.array(dtype=wp.vec3),
    pair_a_manifold_normal_jacobian: wp.array(dtype=wp.float32),
    pair_a_manifold_tangent_jacobian: wp.array(dtype=wp.float32),
    pair_a_manifold_normal_rhs: wp.array(dtype=wp.float32),
    pair_a_manifold_tangent_rhs: wp.array(dtype=wp.float32),
    pair_a_manifold_push_rhs: wp.array(dtype=wp.float32),
    pair_a_manifold_normal_impulse: wp.array(dtype=wp.float32),
    pair_a_manifold_tangent_impulse: wp.array(dtype=wp.float32),
    pair_a_manifold_push_impulse: wp.array(dtype=wp.float32),
    pair_a_extra_hit_velocity_uu: wp.array(dtype=wp.vec3),
    pair_b_algorithm_active: wp.array(dtype=wp.int32),
    pair_b_contact_count: wp.array(dtype=wp.int32),
    pair_b_manifold_local_a_bt: wp.array(dtype=wp.vec3),
    pair_b_manifold_local_b_bt: wp.array(dtype=wp.vec3),
    pair_b_manifold_normal: wp.array(dtype=wp.vec3),
    pair_b_manifold_tangent: wp.array(dtype=wp.vec3),
    pair_b_manifold_normal_jacobian: wp.array(dtype=wp.float32),
    pair_b_manifold_tangent_jacobian: wp.array(dtype=wp.float32),
    pair_b_manifold_normal_rhs: wp.array(dtype=wp.float32),
    pair_b_manifold_tangent_rhs: wp.array(dtype=wp.float32),
    pair_b_manifold_push_rhs: wp.array(dtype=wp.float32),
    pair_b_manifold_normal_impulse: wp.array(dtype=wp.float32),
    pair_b_manifold_tangent_impulse: wp.array(dtype=wp.float32),
    pair_b_manifold_push_impulse: wp.array(dtype=wp.float32),
    pair_b_extra_hit_velocity_uu: wp.array(dtype=wp.vec3),
    car_car_algorithm_active: wp.array(dtype=wp.int32),
    queued_car_velocity_bt: wp.array(dtype=wp.vec3),
):
    env = wp.tid()
    pair_a_active = pair_a_algorithm_active[env]
    pair_b_active = pair_b_algorithm_active[env]
    car_pair_active = car_car_algorithm_active[env]
    # The currently required shared component is ball--A plus ball--B. A/B
    # remains on the independently accepted Phase C path when it is the sole
    # dynamic edge. The fixed corpus has no simultaneous three-edge clique.
    if pair_a_active == 0 or pair_b_active == 0 or car_pair_active != 0:
        return

    car_a = env * 2
    car_b = car_a + 1
    position_a = pre_car_position_bt[car_a]
    position_b = pre_car_position_bt[car_b]
    quaternion_a = pre_car_quaternion[car_a]
    quaternion_b = pre_car_quaternion[car_b]
    basis_a = _bullet_quaternion_matrix(quaternion_a)
    basis_b = _bullet_quaternion_matrix(quaternion_b)
    ball_position = pre_ball_position_bt[env]
    ball_quaternion_value = pre_ball_quaternion[env]
    ball_basis = _bullet_quaternion_matrix(ball_quaternion_value)
    if tick_counter[0] == 1:
        basis_a = _authority_input_quaternion_matrix(quaternion_a)
        basis_b = _authority_input_quaternion_matrix(quaternion_b)
        ball_basis = _authority_input_quaternion_matrix(ball_quaternion_value)

    velocity_a = pre_car_velocity_bt[car_a]
    velocity_b = pre_car_velocity_bt[car_b]
    angular_a = pre_car_angular_velocity[car_a]
    angular_b = pre_car_angular_velocity[car_b]
    ball_source_velocity = pre_ball_velocity_bt[env]
    ball_pre_velocity = ball_source_velocity * wp.pow(1.0 - BALL_DAMPING, DT)
    ball_pre_angular = pre_ball_angular_velocity[env]

    external_linear_a = wp.vec3(0.0)
    external_angular_a = wp.vec3(0.0)
    external_linear_b = wp.vec3(0.0)
    external_angular_b = wp.vec3(0.0)
    _bullet_integrate_external_velocities(
        basis_a,
        total_force_bt[car_a] + wp.vec3(0.0, 0.0, -2340.0),
        total_torque_bt[car_a],
        external_linear_a,
        external_angular_a,
    )
    _bullet_integrate_external_velocities(
        basis_b,
        total_force_bt[car_b] + wp.vec3(0.0, 0.0, -2340.0),
        total_torque_bt[car_b],
        external_linear_b,
        external_angular_b,
    )
    ball_external_linear = wp.vec3(0.0)
    if wp.dot(ball_source_velocity, ball_source_velocity) != 0.0 or wp.dot(
        ball_pre_angular, ball_pre_angular
    ) != 0.0:
        ball_external_linear = wp.vec3(0.0, 0.0, -13.0 * DT)
    ball_force_velocity = ball_pre_velocity + ball_external_linear

    child_center_a = position_a + _matrix_vector(basis_a, CHILD_OFFSET_BT)
    child_center_b = position_b + _matrix_vector(basis_b, CHILD_OFFSET_BT)
    extent_a = wp.vec3(
        wp.abs(basis_a[0, 0]) * BOX_HALF_WITH_MARGIN_BT[0]
        + wp.abs(basis_a[0, 1]) * BOX_HALF_WITH_MARGIN_BT[1]
        + wp.abs(basis_a[0, 2]) * BOX_HALF_WITH_MARGIN_BT[2],
        wp.abs(basis_a[1, 0]) * BOX_HALF_WITH_MARGIN_BT[0]
        + wp.abs(basis_a[1, 1]) * BOX_HALF_WITH_MARGIN_BT[1]
        + wp.abs(basis_a[1, 2]) * BOX_HALF_WITH_MARGIN_BT[2],
        wp.abs(basis_a[2, 0]) * BOX_HALF_WITH_MARGIN_BT[0]
        + wp.abs(basis_a[2, 1]) * BOX_HALF_WITH_MARGIN_BT[1]
        + wp.abs(basis_a[2, 2]) * BOX_HALF_WITH_MARGIN_BT[2],
    ) + wp.vec3(RS_CAR_PROXY_AABB_EXPANSION_BT)
    extent_b = wp.vec3(
        wp.abs(basis_b[0, 0]) * BOX_HALF_WITH_MARGIN_BT[0]
        + wp.abs(basis_b[0, 1]) * BOX_HALF_WITH_MARGIN_BT[1]
        + wp.abs(basis_b[0, 2]) * BOX_HALF_WITH_MARGIN_BT[2],
        wp.abs(basis_b[1, 0]) * BOX_HALF_WITH_MARGIN_BT[0]
        + wp.abs(basis_b[1, 1]) * BOX_HALF_WITH_MARGIN_BT[1]
        + wp.abs(basis_b[1, 2]) * BOX_HALF_WITH_MARGIN_BT[2],
        wp.abs(basis_b[2, 0]) * BOX_HALF_WITH_MARGIN_BT[0]
        + wp.abs(basis_b[2, 1]) * BOX_HALF_WITH_MARGIN_BT[1]
        + wp.abs(basis_b[2, 2]) * BOX_HALF_WITH_MARGIN_BT[2],
    ) + wp.vec3(RS_CAR_PROXY_AABB_EXPANSION_BT)
    ball_proxy_half = wp.vec3(1.925)
    ball_mask = _rs_static_overlap_mask(
        ball_position - ball_proxy_half,
        ball_position + ball_proxy_half,
        rs_static_cell_mask,
        rs_static_aabb_min_bt,
        rs_static_aabb_max_bt,
    )
    if wp.dot(ball_source_velocity, ball_source_velocity) == 0.0 and wp.dot(
        ball_pre_angular, ball_pre_angular
    ) == 0.0:
        ball_mask = wp.uint32(0)
    car_a_mask = _rs_static_overlap_mask(
        child_center_a - extent_a,
        child_center_a + extent_a,
        rs_static_cell_mask,
        rs_static_aabb_min_bt,
        rs_static_aabb_max_bt,
    )
    car_b_mask = _rs_static_overlap_mask(
        child_center_b - extent_b,
        child_center_b + extent_b,
        rs_static_cell_mask,
        rs_static_aabb_min_bt,
        rs_static_aabb_max_bt,
    )
    ball_manifolds = _rs_mask_count(ball_mask)
    car_a_manifolds = _rs_mask_count(car_a_mask)
    car_b_manifolds = _rs_mask_count(car_b_mask)
    total_manifolds = (
        ball_manifolds
        + pair_a_active
        + pair_b_active
        + car_a_manifolds
        + car_b_manifolds
    )
    car_a_contacts = car_static_contact_count[car_a]
    car_b_contacts = car_static_contact_count[car_b]
    ball_contacts = ball_static_contact_count[env]
    pair_a_contacts = pair_a_contact_count[env]
    pair_b_contacts = pair_b_contact_count[env]
    pair_a_base = env * MAX_CAR_BALL_CONTACTS
    pair_b_base = env * MAX_CAR_BALL_CONTACTS

    for relative in range(MAX_CONTACTS_PER_CAR):
        if relative < car_a_contacts:
            index = car_a * MAX_CONTACTS_PER_CAR + relative
            car_static_normal_impulse[index] = 0.0
            car_static_tangent_impulse[index] = 0.0
            car_static_push_impulse[index] = 0.0
        if relative < car_b_contacts:
            index = car_b * MAX_CONTACTS_PER_CAR + relative
            car_static_normal_impulse[index] = 0.0
            car_static_tangent_impulse[index] = 0.0
            car_static_push_impulse[index] = 0.0
    for relative in range(MAX_BALL_CONTACTS):
        if relative < ball_contacts:
            index = env * MAX_BALL_CONTACTS + relative
            ball_static_normal_impulse[index] = 0.0
            ball_static_tangent_impulse[index] = 0.0
            ball_static_push_impulse[index] = 0.0
    for relative in range(MAX_CAR_BALL_CONTACTS):
        if relative < pair_a_contacts:
            index = pair_a_base + relative
            pair_a_manifold_normal_impulse[index] = 0.0
            pair_a_manifold_tangent_impulse[index] = 0.0
            pair_a_manifold_push_impulse[index] = 0.0
        if relative < pair_b_contacts:
            index = pair_b_base + relative
            pair_b_manifold_normal_impulse[index] = 0.0
            pair_b_manifold_tangent_impulse[index] = 0.0
            pair_b_manifold_push_impulse[index] = 0.0

    ball_special_normal_sum = wp.vec3(0.0)
    ball_special_distance_sum = wp.float32(0.0)
    for sorted_slot in range(RS_MAX_ISLAND_MANIFOLDS):
        if sorted_slot < total_manifolds:
            code = _integrated_sorted_manifold_code(
                sorted_slot,
                ball_mask,
                car_a_mask,
                car_b_mask,
                ball_manifolds,
                pair_a_active,
                pair_b_active,
                pair_a_before_b[env],
                car_a_manifolds,
                0,
                total_manifolds,
                rs_equal_island_permutation,
            )
            if code < 32:
                for relative in range(MAX_BALL_CONTACTS):
                    if relative < ball_contacts:
                        index = env * MAX_BALL_CONTACTS + relative
                        if ball_static_mesh[index] == code:
                            point = _bullet_transform_point(
                                ball_position,
                                ball_basis,
                                ball_static_local_a_bt[index],
                            )
                            ball_special_normal_sum = (
                                ball_special_normal_sum + ball_static_normal[index]
                            )
                            _bullet_special_distance_accumulate(
                                point - ball_position, ball_special_distance_sum
                            )
    ball_special_normal = wp.vec3(0.0)
    ball_special_rel = wp.vec3(0.0)
    ball_special_tangent = wp.vec3(0.0)
    ball_special_normal_jacobian = wp.float32(0.0)
    ball_special_tangent_jacobian = wp.float32(0.0)
    ball_special_normal_rhs = wp.float32(0.0)
    ball_special_tangent_rhs = wp.float32(0.0)
    ball_special_normal_impulse = wp.float32(0.0)
    ball_special_tangent_impulse = wp.float32(0.0)
    if ball_contacts > 0:
        ball_special_distance = wp.float32(0.0)
        _bullet_special_average(
            ball_special_normal_sum,
            ball_special_distance_sum,
            ball_contacts,
            ball_special_normal,
            ball_special_distance,
        )
        ball_special_rel = ball_special_normal * -ball_special_distance
        ball_special_push_rhs = wp.float32(0.0)
        _bullet_ball_special_contact_row(
            wp.vec3(0.0),
            ball_basis,
            ball_special_rel,
            wp.vec3(0.0),
            ball_special_distance,
            DT,
            ball_special_normal,
            ball_pre_velocity,
            ball_pre_angular,
            ball_force_velocity,
            ball_pre_angular,
            ball_special_tangent,
            ball_special_normal_jacobian,
            ball_special_tangent_jacobian,
            ball_special_normal_rhs,
            ball_special_tangent_rhs,
            ball_special_push_rhs,
        )

    delta_velocity_a = wp.vec3(0.0)
    delta_angular_a = wp.vec3(0.0)
    delta_velocity_b = wp.vec3(0.0)
    delta_angular_b = wp.vec3(0.0)
    ball_delta_velocity = wp.vec3(0.0)
    ball_delta_angular = wp.vec3(0.0)
    push_a = wp.vec3(0.0)
    turn_a = wp.vec3(0.0)
    push_b = wp.vec3(0.0)
    turn_b = wp.vec3(0.0)
    ball_push = wp.vec3(0.0)
    ball_turn = wp.vec3(0.0)

    # solveGroupCacheFriendlySplitImpulseIterations: every ordinary normal
    # row in the one globally sorted manifold stream.
    for _iteration in range(SOLVER_ITERATIONS):
        for sorted_slot in range(RS_MAX_ISLAND_MANIFOLDS):
            if sorted_slot < total_manifolds:
                code = _integrated_sorted_manifold_code(
                    sorted_slot,
                    ball_mask,
                    car_a_mask,
                    car_b_mask,
                    ball_manifolds,
                    pair_a_active,
                    pair_b_active,
                    pair_a_before_b[env],
                    car_a_manifolds,
                    0,
                    total_manifolds,
                    rs_equal_island_permutation,
                )
                if code < 32:
                    for relative in range(MAX_BALL_CONTACTS):
                        if relative < ball_contacts:
                            index = env * MAX_BALL_CONTACTS + relative
                            if ball_static_mesh[index] == code:
                                point = _bullet_transform_point(
                                    ball_position,
                                    ball_basis,
                                    ball_static_local_a_bt[index],
                                )
                                applied = wp.float32(ball_static_push_impulse[index])
                                _bullet_ball_solve_split_row(
                                    ball_basis,
                                    ball_static_normal[index],
                                    point - ball_position,
                                    ball_static_normal_jacobian[index],
                                    ball_static_push_rhs[index],
                                    ball_push,
                                    ball_turn,
                                    applied,
                                )
                                ball_static_push_impulse[index] = applied
                elif code == 32 or code == 33:
                    contacts = pair_a_contacts
                    base = pair_a_base
                    pair_basis = basis_a
                    pair_position = position_a
                    pair_push = push_a
                    pair_turn = turn_a
                    local_a = pair_a_manifold_local_a_bt
                    local_b = pair_a_manifold_local_b_bt
                    normal = pair_a_manifold_normal
                    split_jacobians = pair_a_manifold_normal_jacobian
                    split_rhs_values = pair_a_manifold_push_rhs
                    split_impulses = pair_a_manifold_push_impulse
                    if code == 33:
                        contacts = pair_b_contacts
                        base = pair_b_base
                        pair_basis = basis_b
                        pair_position = position_b
                        pair_push = push_b
                        pair_turn = turn_b
                        local_a = pair_b_manifold_local_a_bt
                        local_b = pair_b_manifold_local_b_bt
                        normal = pair_b_manifold_normal
                        split_jacobians = pair_b_manifold_normal_jacobian
                        split_rhs_values = pair_b_manifold_push_rhs
                        split_impulses = pair_b_manifold_push_impulse
                    for relative in range(MAX_CAR_BALL_CONTACTS):
                        if relative < contacts:
                            index = base + relative
                            point_a = _bullet_transform_point(
                                pair_position, pair_basis, local_a[index]
                            )
                            point_b = _bullet_transform_point(
                                ball_position, ball_basis, local_b[index]
                            )
                            applied = wp.float32(split_impulses[index])
                            _bullet_pair_solve_split_row(
                                pair_basis,
                                ball_basis,
                                normal[index],
                                point_a - pair_position,
                                point_b - ball_position,
                                split_jacobians[index],
                                split_rhs_values[index],
                                0.0,
                                1.0e10,
                                pair_push,
                                pair_turn,
                                ball_push,
                                ball_turn,
                                applied,
                            )
                            split_impulses[index] = applied
                    if code == 32:
                        push_a = pair_push
                        turn_a = pair_turn
                    else:
                        push_b = pair_push
                        turn_b = pair_turn
                else:
                    car = car_a
                    position = position_a
                    basis = basis_a
                    push = push_a
                    turn = turn_a
                    contacts = car_a_contacts
                    static_body = code - 64
                    if code >= 128:
                        car = car_b
                        position = position_b
                        basis = basis_b
                        push = push_b
                        turn = turn_b
                        contacts = car_b_contacts
                        static_body = code - 128
                    for relative in range(MAX_CONTACTS_PER_CAR):
                        if relative < contacts:
                            index = car * MAX_CONTACTS_PER_CAR + relative
                            if car_static_mesh[index] == static_body:
                                point = _bullet_transform_point(
                                    position, basis, car_static_local_a_bt[index]
                                )
                                applied = wp.float32(car_static_push_impulse[index])
                                _bullet_solve_split_row(
                                    basis,
                                    car_static_normal[index],
                                    point - position,
                                    car_static_normal_jacobian[index],
                                    car_static_push_rhs[index],
                                    push,
                                    turn,
                                    applied,
                                )
                                car_static_push_impulse[index] = applied
                    if code >= 128:
                        push_b = push
                        turn_b = turn
                    else:
                        push_a = push
                        turn_a = turn

    # solveGroupCacheFriendlyIterations: all normals, the aggregate special
    # ball row, all frictions, then the aggregate special friction row.
    for _iteration in range(SOLVER_ITERATIONS):
        for friction_pass in range(2):
            if friction_pass == 1 and ball_contacts > 0:
                # The special normal row is appended after ordinary normals,
                # immediately before the friction pass.
                pass
            for sorted_slot in range(RS_MAX_ISLAND_MANIFOLDS):
                if sorted_slot < total_manifolds:
                    code = _integrated_sorted_manifold_code(
                        sorted_slot,
                        ball_mask,
                        car_a_mask,
                        car_b_mask,
                        ball_manifolds,
                        pair_a_active,
                        pair_b_active,
                        pair_a_before_b[env],
                        car_a_manifolds,
                        0,
                        total_manifolds,
                        rs_equal_island_permutation,
                    )
                    if code == 32 or code == 33:
                        contacts = pair_a_contacts
                        base = pair_a_base
                        pair_basis = basis_a
                        pair_position = position_a
                        pair_delta_velocity = delta_velocity_a
                        pair_delta_angular = delta_angular_a
                        local_a = pair_a_manifold_local_a_bt
                        local_b = pair_a_manifold_local_b_bt
                        normal = pair_a_manifold_normal
                        tangent = pair_a_manifold_tangent
                        pair_normal_jacobians = pair_a_manifold_normal_jacobian
                        pair_tangent_jacobians = pair_a_manifold_tangent_jacobian
                        pair_normal_rhs_values = pair_a_manifold_normal_rhs
                        pair_tangent_rhs_values = pair_a_manifold_tangent_rhs
                        pair_normal_impulses = pair_a_manifold_normal_impulse
                        pair_tangent_impulses = pair_a_manifold_tangent_impulse
                        if code == 33:
                            contacts = pair_b_contacts
                            base = pair_b_base
                            pair_basis = basis_b
                            pair_position = position_b
                            pair_delta_velocity = delta_velocity_b
                            pair_delta_angular = delta_angular_b
                            local_a = pair_b_manifold_local_a_bt
                            local_b = pair_b_manifold_local_b_bt
                            normal = pair_b_manifold_normal
                            tangent = pair_b_manifold_tangent
                            pair_normal_jacobians = pair_b_manifold_normal_jacobian
                            pair_tangent_jacobians = pair_b_manifold_tangent_jacobian
                            pair_normal_rhs_values = pair_b_manifold_normal_rhs
                            pair_tangent_rhs_values = pair_b_manifold_tangent_rhs
                            pair_normal_impulses = pair_b_manifold_normal_impulse
                            pair_tangent_impulses = pair_b_manifold_tangent_impulse
                        for relative in range(MAX_CAR_BALL_CONTACTS):
                            if relative < contacts:
                                index = base + relative
                                applied_normal = pair_normal_impulses[index]
                                if friction_pass == 0 or applied_normal > 0.0:
                                    direction = normal[index]
                                    row_jacobian = pair_normal_jacobians[index]
                                    row_rhs = pair_normal_rhs_values[index]
                                    lower = 0.0
                                    upper = 1.0e10
                                    applied = wp.float32(applied_normal)
                                    if friction_pass == 1:
                                        direction = tangent[index]
                                        row_jacobian = pair_tangent_jacobians[index]
                                        row_rhs = pair_tangent_rhs_values[index]
                                        limit = CONTACT_FRICTION * applied_normal
                                        lower = -limit
                                        upper = limit
                                        applied = wp.float32(
                                            pair_tangent_impulses[index]
                                        )
                                    point_a = _bullet_transform_point(
                                        pair_position, pair_basis, local_a[index]
                                    )
                                    point_b = _bullet_transform_point(
                                        ball_position, ball_basis, local_b[index]
                                    )
                                    _bullet_pair_solve_velocity_row(
                                        pair_basis,
                                        ball_basis,
                                        direction,
                                        point_a - pair_position,
                                        point_b - ball_position,
                                        row_jacobian,
                                        row_rhs,
                                        lower,
                                        upper,
                                        pair_delta_velocity,
                                        pair_delta_angular,
                                        ball_delta_velocity,
                                        ball_delta_angular,
                                        applied,
                                    )
                                    if friction_pass == 0:
                                        pair_normal_impulses[index] = applied
                                    else:
                                        pair_tangent_impulses[index] = applied
                        if code == 32:
                            delta_velocity_a = pair_delta_velocity
                            delta_angular_a = pair_delta_angular
                        else:
                            delta_velocity_b = pair_delta_velocity
                            delta_angular_b = pair_delta_angular
                    elif code >= 64:
                        car = car_a
                        position = position_a
                        basis = basis_a
                        delta_velocity = delta_velocity_a
                        delta_angular = delta_angular_a
                        contacts = car_a_contacts
                        static_body = code - 64
                        if code >= 128:
                            car = car_b
                            position = position_b
                            basis = basis_b
                            delta_velocity = delta_velocity_b
                            delta_angular = delta_angular_b
                            contacts = car_b_contacts
                            static_body = code - 128
                        for relative in range(MAX_CONTACTS_PER_CAR):
                            if relative < contacts:
                                index = car * MAX_CONTACTS_PER_CAR + relative
                                if car_static_mesh[index] == static_body:
                                    applied_normal = car_static_normal_impulse[index]
                                    if friction_pass == 0 or applied_normal > 0.0:
                                        direction = car_static_normal[index]
                                        row_jacobian = car_static_normal_jacobian[index]
                                        row_rhs = car_static_normal_rhs[index]
                                        lower = 0.0
                                        upper = 1.0e10
                                        applied = wp.float32(applied_normal)
                                        if friction_pass == 1:
                                            direction = car_static_tangent[index]
                                            row_jacobian = car_static_tangent_jacobian[index]
                                            row_rhs = car_static_tangent_rhs[index]
                                            limit = CAR_STATIC_FRICTION * applied_normal
                                            lower = -limit
                                            upper = limit
                                            applied = wp.float32(
                                                car_static_tangent_impulse[index]
                                            )
                                        point = _bullet_transform_point(
                                            position,
                                            basis,
                                            car_static_local_a_bt[index],
                                        )
                                        _bullet_solve_velocity_row(
                                            basis,
                                            direction,
                                            point - position,
                                            row_jacobian,
                                            row_rhs,
                                            lower,
                                            upper,
                                            delta_velocity,
                                            delta_angular,
                                            applied,
                                        )
                                        if friction_pass == 0:
                                            car_static_normal_impulse[index] = applied
                                        else:
                                            car_static_tangent_impulse[index] = applied
                        if code >= 128:
                            delta_velocity_b = delta_velocity
                            delta_angular_b = delta_angular
                        else:
                            delta_velocity_a = delta_velocity
                            delta_angular_a = delta_angular
            if friction_pass == 0 and ball_contacts > 0:
                _bullet_ball_solve_velocity_row(
                    ball_basis,
                    ball_special_normal,
                    ball_special_rel,
                    ball_special_normal_jacobian,
                    ball_special_normal_rhs,
                    0.0,
                    1.0e10,
                    ball_delta_velocity,
                    ball_delta_angular,
                    ball_special_normal_impulse,
                )
            elif friction_pass == 1 and ball_special_normal_impulse > 0.0:
                limit = BALL_STATIC_FRICTION * ball_special_normal_impulse
                _bullet_ball_solve_velocity_row(
                    ball_basis,
                    ball_special_tangent,
                    ball_special_rel,
                    ball_special_tangent_jacobian,
                    ball_special_tangent_rhs,
                    -limit,
                    limit,
                    ball_delta_velocity,
                    ball_delta_angular,
                    ball_special_tangent_impulse,
                )

    solved_velocity_a = (velocity_a + delta_velocity_a) + external_linear_a
    solved_angular_a = (angular_a + delta_angular_a) + external_angular_a
    solved_velocity_b = (velocity_b + delta_velocity_b) + external_linear_b
    solved_angular_b = (angular_b + delta_angular_b) + external_angular_b
    ball_solved_velocity = (
        ball_pre_velocity + ball_delta_velocity
    ) + ball_external_linear
    ball_solved_angular = ball_pre_angular + ball_delta_angular

    split_a = wp.int32(wp.dot(push_a, push_a) > 0.0 or wp.dot(turn_a, turn_a) > 0.0)
    split_b = wp.int32(wp.dot(push_b, push_b) > 0.0 or wp.dot(turn_b, turn_b) > 0.0)
    ball_split = wp.int32(
        wp.dot(ball_push, ball_push) > 0.0 or wp.dot(ball_turn, ball_turn) > 0.0
    )
    split_quaternion_a = quaternion_a
    split_quaternion_b = quaternion_b
    ball_split_quaternion = ball_quaternion_value
    integration_basis_a = basis_a
    integration_basis_b = basis_b
    ball_integration_basis = ball_basis
    if split_a != 0:
        _bullet_integrate_quaternion(
            basis_a, turn_a * CONTACT_SPLIT_TURN_ERP, DT, split_quaternion_a
        )
        integration_basis_a = _bullet_quaternion_matrix(split_quaternion_a)
    if split_b != 0:
        _bullet_integrate_quaternion(
            basis_b, turn_b * CONTACT_SPLIT_TURN_ERP, DT, split_quaternion_b
        )
        integration_basis_b = _bullet_quaternion_matrix(split_quaternion_b)
    if ball_split != 0:
        _bullet_integrate_quaternion(
            ball_basis,
            ball_turn * CONTACT_SPLIT_TURN_ERP,
            DT,
            ball_split_quaternion,
        )
        ball_integration_basis = _bullet_quaternion_matrix(ball_split_quaternion)
    integrated_position_a = position_a
    integrated_position_b = position_b
    integrated_ball_position = ball_position
    _bullet_integrate_position(
        position_a, push_a, solved_velocity_a, DT, split_a, integrated_position_a
    )
    _bullet_integrate_position(
        position_b, push_b, solved_velocity_b, DT, split_b, integrated_position_b
    )
    _bullet_integrate_position(
        ball_position,
        ball_push,
        ball_solved_velocity,
        DT,
        ball_split,
        integrated_ball_position,
    )
    integrated_quaternion_a = split_quaternion_a
    integrated_quaternion_b = split_quaternion_b
    integrated_ball_quaternion = ball_split_quaternion
    _bullet_integrate_quaternion(
        integration_basis_a, solved_angular_a, DT, integrated_quaternion_a
    )
    _bullet_integrate_quaternion(
        integration_basis_b, solved_angular_b, DT, integrated_quaternion_b
    )
    _bullet_integrate_quaternion(
        ball_integration_basis,
        ball_solved_angular,
        DT,
        integrated_ball_quaternion,
    )

    # RocketSim applies cached bump impulses to each car, then the sum of both
    # per-car ball-hit velocity caches, after transform integration.
    solved_velocity_a = solved_velocity_a + queued_car_velocity_bt[car_a]
    solved_velocity_b = solved_velocity_b + queued_car_velocity_bt[car_b]
    ball_velocity_cache = wp.vec3(0.0)
    ball_velocity_cache = (
        ball_velocity_cache + pair_a_extra_hit_velocity_uu[env] * 0.02
    )
    ball_velocity_cache = (
        ball_velocity_cache + pair_b_extra_hit_velocity_uu[env] * 0.02
    )
    ball_solved_velocity = ball_solved_velocity + ball_velocity_cache

    for local_car in range(2):
        car = car_a + local_car
        solved_velocity = solved_velocity_a
        if local_car == 1:
            solved_velocity = solved_velocity_b
        speed_uu = solved_velocity * 50.0
        speed_squared = wp.dot(speed_uu, speed_uu)
        supersonic = pre_car_is_supersonic[car]
        time = pre_car_supersonic_time[car]
        if supersonic != 0 and time < 1.0:
            supersonic = wp.int32(speed_squared >= 4410000.0)
        else:
            supersonic = wp.int32(speed_squared >= 4840000.0)
        if supersonic != 0:
            time = time + DT
        else:
            time = 0.0
        car_is_supersonic[car] = supersonic
        car_supersonic_time[car] = time

    _bullet_cap_amd(solved_velocity_a, CAR_MAX_SPEED_BT, amd_rsqrtss_mantissa)
    _bullet_cap_amd(solved_angular_a, CAR_MAX_ANGULAR_SPEED, amd_rsqrtss_mantissa)
    _bullet_cap_amd(solved_velocity_b, CAR_MAX_SPEED_BT, amd_rsqrtss_mantissa)
    _bullet_cap_amd(solved_angular_b, CAR_MAX_ANGULAR_SPEED, amd_rsqrtss_mantissa)
    _bullet_cap_amd(ball_solved_velocity, BALL_MAX_SPEED_BT, amd_rsqrtss_mantissa)
    _bullet_cap_amd(ball_solved_angular, BALL_MAX_ANGULAR_SPEED, amd_rsqrtss_mantissa)

    car_rigid_position_bt[car_a] = integrated_position_a
    car_rigid_position_bt[car_b] = integrated_position_b
    car_rigid_velocity_bt[car_a] = solved_velocity_a
    car_rigid_velocity_bt[car_b] = solved_velocity_b
    car_pos[car_a] = integrated_position_a * 50.0
    car_pos[car_b] = integrated_position_b * 50.0
    car_vel[car_a] = solved_velocity_a * 50.0
    car_vel[car_b] = solved_velocity_b * 50.0
    car_quat[car_a] = integrated_quaternion_a
    car_quat[car_b] = integrated_quaternion_b
    car_ang_vel[car_a] = solved_angular_a
    car_ang_vel[car_b] = solved_angular_b
    ball_resident_position_bt[env] = integrated_ball_position
    ball_resident_velocity_bt[env] = ball_solved_velocity
    ball_pos[env] = integrated_ball_position * 50.0
    ball_vel[env] = ball_solved_velocity * 50.0
    ball_quat[env] = integrated_ball_quaternion
    ball_ang_vel[env] = ball_solved_angular


__all__ = ["integrated_two_car_ball_tick"]
