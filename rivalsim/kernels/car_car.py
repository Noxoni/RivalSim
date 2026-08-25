"""Bounded RocketSim/Bullet Octane/Octane collision and bump path."""

import warp as wp

from rivalsim.car_car_state import (
    MAX_CAR_BUMP_EVENTS_PER_TICK,
    MAX_CAR_CAR_CONTACTS,
)
from rivalsim.kernels.bullet_box_box import bullet_octane_box_box
from rivalsim.kernels.car_ball import (
    CAR_INV_MASS,
    CAR_MAX_ANGULAR_SPEED,
    CAR_MAX_SPEED_BT,
    CAR_STATIC_FRICTION,
    CHILD_OFFSET_BT,
    CONTACT_SPLIT_TURN_ERP,
    DT,
    RS_CAR_PROXY_AABB_EXPANSION_BT,
    RS_EQUAL_ISLAND_PERMUTATION_WIDTH,
    RS_MAX_ISLAND_MANIFOLDS,
    SOLVER_ITERATIONS,
    _BULLET_PAIR_CONTACT_ROW,
    _BULLET_PAIR_SOLVE_SPLIT_ROW,
    _BULLET_PAIR_SOLVE_VELOCITY_ROW,
    _bullet_cap_amd,
    _bullet_pair_refresh_contact,
    _matrix_vector,
    _rs_mask_count,
    _rs_nth_mask_body,
    _rs_static_overlap_mask,
)
from rivalsim.kernels.vehicle import (
    _authority_input_quaternion_matrix,
    _bullet_integrate_external_velocities,
    _bullet_integrate_position,
    _bullet_integrate_quaternion,
    _bullet_inverse_transform_point,
    _bullet_quaternion_matrix,
    _bullet_solve_split_row,
    _bullet_solve_velocity_row,
    _bullet_transform_point,
    _bullet_vector_scale_add,
)
from rivalsim.vehicle_state import MAX_CONTACTS_PER_CAR

CONTACT_FRICTION = 0.09
CONTACT_BREAKING_THRESHOLD_BT = 0.0406245552


_BULLET_CAR_CAR_CONTACT_ROW = (
    _BULLET_PAIR_CONTACT_ROW.replace(
        "Octane/standard-ball pair", "fixed Octane/Octane pair"
    )
    .replace(
        "const float ball_inverse_local[3]={0.0250203293f,0.0250203293f,0.0250203293f};",
        "const float ball_inverse_local[3]={0.0185644571f,0.0104337428f,0.0075815497f};",
    )
    .replace("0.0333333351f", "0.00555555569f")
    .replace(
        "normal_rhs=op_mul(op_sub(0.0f,relative_normal_speed),normal_inverse);",
        """const PairV3 pre_point_a=add(car_pre_l,cross(car_pre_a,rel_a));
    const PairV3 pre_point_b=add(ball_pre_l,cross(ball_pre_a,rel_b));
    const float pre_relative_speed=dot(n,sub(pre_point_a,pre_point_b));
    float restitution=0.0f;
    if(fabsf(pre_relative_speed)>=0.2f){
        restitution=op_mul(-0.1f,pre_relative_speed);
        if(restitution<0.0f)restitution=0.0f;
    }
    normal_rhs=op_mul(op_sub(restitution,relative_normal_speed),normal_inverse);""",
    )
)


@wp.func_native(_BULLET_CAR_CAR_CONTACT_ROW)
def _bullet_car_car_contact_row(
    car_basis: wp.mat33,
    ball_basis: wp.mat33,
    relative_car_bt: wp.vec3,
    relative_ball_bt: wp.vec3,
    distance_bt: float,
    time_step: float,
    normal: wp.vec3,
    car_pre_linear_bt: wp.vec3,
    car_pre_angular_world: wp.vec3,
    car_force_linear_bt: wp.vec3,
    car_force_angular_world: wp.vec3,
    ball_pre_linear_bt: wp.vec3,
    ball_pre_angular_world: wp.vec3,
    ball_force_linear_bt: wp.vec3,
    ball_force_angular_world: wp.vec3,
    tangent: wp.ref[wp.vec3],
    normal_jacobian: wp.ref[wp.float32],
    tangent_jacobian: wp.ref[wp.float32],
    normal_rhs: wp.ref[wp.float32],
    tangent_rhs: wp.ref[wp.float32],
    push_rhs: wp.ref[wp.float32],
): ...


_BULLET_CAR_CAR_SOLVE_VELOCITY_ROW = (
    _BULLET_PAIR_SOLVE_VELOCITY_ROW.replace(
        "const float ball_inverse_local[3]={0.0250203293f,0.0250203293f,0.0250203293f};",
        "const float ball_inverse_local[3]={0.0185644571f,0.0104337428f,0.0075815497f};",
    ).replace("0.0333333351f", "0.00555555569f")
)


@wp.func_native(_BULLET_CAR_CAR_SOLVE_VELOCITY_ROW)
def _bullet_car_car_solve_velocity_row(
    car_basis: wp.mat33,
    ball_basis: wp.mat33,
    direction: wp.vec3,
    relative_car_bt: wp.vec3,
    relative_ball_bt: wp.vec3,
    jacobian: float,
    rhs: float,
    lower_limit: float,
    upper_limit: float,
    car_delta_linear_bt: wp.ref[wp.vec3],
    car_delta_angular_world: wp.ref[wp.vec3],
    ball_delta_linear_bt: wp.ref[wp.vec3],
    ball_delta_angular_world: wp.ref[wp.vec3],
    applied_impulse: wp.ref[wp.float32],
): ...


_BULLET_CAR_CAR_SOLVE_SPLIT_ROW = (
    _BULLET_PAIR_SOLVE_SPLIT_ROW.replace(
        "const float ball_inverse_local[3]={0.0250203293f,0.0250203293f,0.0250203293f};",
        "const float ball_inverse_local[3]={0.0185644571f,0.0104337428f,0.0075815497f};",
    ).replace("0.0333333351f", "0.00555555569f")
)


@wp.func_native(_BULLET_CAR_CAR_SOLVE_SPLIT_ROW)
def _bullet_car_car_solve_split_row(
    car_basis: wp.mat33,
    ball_basis: wp.mat33,
    direction: wp.vec3,
    relative_car_bt: wp.vec3,
    relative_ball_bt: wp.vec3,
    jacobian: float,
    rhs: float,
    lower_limit: float,
    upper_limit: float,
    car_delta_linear_bt: wp.ref[wp.vec3],
    car_delta_angular_world: wp.ref[wp.vec3],
    ball_delta_linear_bt: wp.ref[wp.vec3],
    ball_delta_angular_world: wp.ref[wp.vec3],
    applied_impulse: wp.ref[wp.float32],
): ...


_ROCKETSIM_CAR_CAR_EVENT = r"""
    // Arena::_BtCallback_OnCarCarCollision for one directed bumper/victim
    // test, including MathTypes::Vec and LinearPieceCurve operation order.
    auto add=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a,b);
    #else
        volatile float v=a+b;return v;
    #endif
    };
    auto sub=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a,b);
    #else
        volatile float v=a-b;return v;
    #endif
    };
    auto mul=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a,b);
    #else
        volatile float v=a*b;return v;
    #endif
    };
    auto div=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a,b);
    #else
        volatile float v=a/b;return v;
    #endif
    };
    struct V{float x,y,z,w;};
    auto make=[](float x,float y,float z)->V{V v={x,y,z,0.0f};return v;};
    auto subtract=[&](V a,V b)->V{return make(sub(a.x,b.x),sub(a.y,b.y),sub(a.z,b.z));};
    auto scale=[&](V a,float s)->V{return make(mul(a.x,s),mul(a.y,s),mul(a.z,s));};
    auto sum=[&](V a,V b)->V{return make(add(a.x,b.x),add(a.y,b.y),add(a.z,b.z));};
    auto dot=[&](V a,V b)->float{return add(add(add(mul(a.x,b.x),mul(a.y,b.y)),mul(a.z,b.z)),mul(a.w,b.w));};
    auto normalized=[&](V a)->V{
        const float length=sqrtf(dot(a,a));
        if(length>2.0194839173657902e-28f)return make(div(a.x,length),div(a.y,length),div(a.z,length));
        return make(0.0f,0.0f,0.0f);
    };
    auto curve=[&](float input,float middle,float last,float first)->float{
        if(input<=0.0f)return first;
        if(1400.0f>input){
            const float range=sub(1400.0f,0.0f);
            const float difference=sub(middle,first);
            const float factor=div(sub(input,0.0f),range);
            return add(first,mul(difference,factor));
        }
        if(2200.0f>input){
            const float range=sub(2200.0f,1400.0f);
            const float difference=sub(last,middle);
            const float factor=div(sub(input,1400.0f),range);
            return add(middle,mul(difference,factor));
        }
        return last;
    };
    const V bumper_pos=scale(make(bumper_position_bt[0],bumper_position_bt[1],bumper_position_bt[2]),50.0f);
    const V victim_pos=scale(make(victim_position_bt[0],victim_position_bt[1],victim_position_bt[2]),50.0f);
    const V bumper_vel=scale(make(bumper_velocity_bt[0],bumper_velocity_bt[1],bumper_velocity_bt[2]),50.0f);
    const V victim_vel=scale(make(victim_velocity_bt[0],victim_velocity_bt[1],victim_velocity_bt[2]),50.0f);
    const V delta=subtract(victim_pos,bumper_pos);
    if(dot(bumper_vel,delta)<=0.0f)return;
    const V velocity_direction=normalized(bumper_vel);
    const V direction_to_victim=normalized(delta);
    const float speed_towards=dot(bumper_vel,direction_to_victim);
    const float victim_away_speed=dot(victim_vel,velocity_direction);
    if(speed_towards<=victim_away_speed)return;
    if(mul(local_point_on_bumper_bt[0],50.0f)<=64.5f)return;
    triggered=1;
    if(bumper_is_supersonic!=0){is_demo=1;return;}
    const float base=curve(speed_towards,victim_on_ground!=0?1100.0f:1390.0f,victim_on_ground!=0?1530.0f:1945.0f,0.833333313f);
    const float up_amount=curve(speed_towards,278.0f,417.0f,0.333333343f);
    V up=make(0.0f,0.0f,1.0f);
    if(victim_on_ground!=0)up=make(victim_basis.data[0][2],victim_basis.data[1][2],victim_basis.data[2][2]);
    const V impulse_uu=sum(scale(velocity_direction,base),scale(up,up_amount));
    impulse_bt=wp::vec_t<3,wp::float32>(mul(impulse_uu.x,0.02f),mul(impulse_uu.y,0.02f),mul(impulse_uu.z,0.02f));
"""


@wp.func_native(_ROCKETSIM_CAR_CAR_EVENT)
def _rocketsim_car_car_event(
    bumper_position_bt: wp.vec3,
    bumper_velocity_bt: wp.vec3,
    victim_position_bt: wp.vec3,
    victim_velocity_bt: wp.vec3,
    local_point_on_bumper_bt: wp.vec3,
    bumper_is_supersonic: int,
    victim_on_ground: int,
    victim_basis: wp.mat33,
    triggered: wp.ref[wp.int32],
    is_demo: wp.ref[wp.int32],
    impulse_bt: wp.ref[wp.vec3],
): ...


@wp.func
def _compound_aabb(
    position: wp.vec3, basis: wp.mat33, broadphase_expansion: float
) -> tuple[wp.vec3, wp.vec3]:
    center = position + _matrix_vector(basis, CHILD_OFFSET_BT)
    half = wp.vec3(1.20372915, 0.865653098, 0.385250092)
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
    ) + wp.vec3(
        broadphase_expansion,
        broadphase_expansion,
        broadphase_expansion,
    )
    return center - extent, center + extent


@wp.func
def _car_car_aabb_overlap(
    position_a: wp.vec3,
    basis_a: wp.mat33,
    predicted_position_a: wp.vec3,
    predicted_basis_a: wp.mat33,
    position_b: wp.vec3,
    basis_b: wp.mat33,
    predicted_position_b: wp.vec3,
    predicted_basis_b: wp.mat33,
) -> int:
    minimum_a, maximum_a = _compound_aabb(position_a, basis_a, 0.02)
    predicted_minimum_a, predicted_maximum_a = _compound_aabb(
        predicted_position_a, predicted_basis_a, 0.02
    )
    minimum_b, maximum_b = _compound_aabb(position_b, basis_b, 0.02)
    predicted_minimum_b, predicted_maximum_b = _compound_aabb(
        predicted_position_b, predicted_basis_b, 0.02
    )
    minimum_a = wp.min(minimum_a, predicted_minimum_a)
    maximum_a = wp.max(maximum_a, predicted_maximum_a)
    minimum_b = wp.min(minimum_b, predicted_minimum_b)
    maximum_b = wp.max(maximum_b, predicted_maximum_b)
    if minimum_a[0] > maximum_b[0] or maximum_a[0] < minimum_b[0]:
        return 0
    if minimum_a[2] > maximum_b[2] or maximum_a[2] < minimum_b[2]:
        return 0
    if minimum_a[1] > maximum_b[1] or maximum_a[1] < minimum_b[1]:
        return 0
    return 1


@wp.func
def _car_car_child_aabb_overlap(
    position_a: wp.vec3,
    basis_a: wp.mat33,
    position_b: wp.vec3,
    basis_b: wp.mat33,
) -> int:
    """Return the nested compound child-algorithm gate from pinned Bullet.

    The outer btRSBroadphase pair uses continuous current-plus-interpolation
    AABBs expanded by gContactBreakingThreshold.  Once that outer pair exists,
    btCompoundCollisionAlgorithm independently culls its single child using
    the two *current* shape AABBs with no extra threshold.  A failed child cull
    destroys the box/box algorithm and its persistent manifold, although the
    outer pair still unifies both cars into one solver island.
    """

    minimum_a, maximum_a = _compound_aabb(position_a, basis_a, 0.0)
    minimum_b, maximum_b = _compound_aabb(position_b, basis_b, 0.0)
    if minimum_a[0] > maximum_b[0] or maximum_a[0] < minimum_b[0]:
        return 0
    if minimum_a[1] > maximum_b[1] or maximum_a[1] < minimum_b[1]:
        return 0
    if minimum_a[2] > maximum_b[2] or maximum_a[2] < minimum_b[2]:
        return 0
    return 1


@wp.func
def _sorted_car_car_manifold_code(
    sorted_position: int,
    car_a_mask: wp.uint32,
    car_b_mask: wp.uint32,
    car_a_manifold_count: int,
    pair_manifold_active: int,
    total_manifold_count: int,
    equal_island_permutation: wp.array(dtype=wp.int32),
) -> int:
    """Encode a sorted slot: pair=32, car A static=64+, car B=96+."""

    source = equal_island_permutation[
        total_manifold_count * RS_EQUAL_ISLAND_PERMUTATION_WIDTH
        + sorted_position
    ]
    if source < car_a_manifold_count:
        return 64 + _rs_nth_mask_body(car_a_mask, source)
    if pair_manifold_active != 0 and source == car_a_manifold_count:
        return 32
    return 96 + _rs_nth_mask_body(
        car_b_mask,
        source - car_a_manifold_count - pair_manifold_active,
    )


@wp.kernel(enable_backward=False)
def capture_car_car_inputs(
    position_bt: wp.array(dtype=wp.vec3),
    velocity_bt: wp.array(dtype=wp.vec3),
    quaternion: wp.array(dtype=wp.quat),
    angular_velocity: wp.array(dtype=wp.vec3),
    on_ground: wp.array(dtype=wp.int32),
    is_supersonic: wp.array(dtype=wp.int32),
    supersonic_time: wp.array(dtype=wp.float32),
    pre_position_bt: wp.array(dtype=wp.vec3),
    pre_velocity_bt: wp.array(dtype=wp.vec3),
    pre_quaternion: wp.array(dtype=wp.quat),
    pre_angular_velocity: wp.array(dtype=wp.vec3),
    pre_on_ground: wp.array(dtype=wp.int32),
    pre_is_supersonic: wp.array(dtype=wp.int32),
    pre_supersonic_time: wp.array(dtype=wp.float32),
):
    car = wp.tid()
    pre_position_bt[car] = position_bt[car]
    pre_velocity_bt[car] = velocity_bt[car]
    pre_quaternion[car] = quaternion[car]
    pre_angular_velocity[car] = angular_velocity[car]
    pre_on_ground[car] = on_ground[car]
    pre_is_supersonic[car] = is_supersonic[car]
    pre_supersonic_time[car] = supersonic_time[car]


@wp.kernel(
    enable_backward=False,
    module="unique",
    module_options={"max_unroll": 4},
)
def car_car_tick(
    tick_counter: wp.array(dtype=wp.int32),
    amd_rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    rs_static_cell_mask: wp.array(dtype=wp.uint32),
    rs_static_aabb_min_bt: wp.array(dtype=wp.vec3),
    rs_static_aabb_max_bt: wp.array(dtype=wp.vec3),
    rs_equal_island_permutation: wp.array(dtype=wp.int32),
    total_force_bt: wp.array(dtype=wp.vec3),
    total_torque_bt: wp.array(dtype=wp.vec3),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    car_on_ground: wp.array(dtype=wp.int32),
    car_is_supersonic: wp.array(dtype=wp.int32),
    car_supersonic_time: wp.array(dtype=wp.float32),
    rigid_position_bt: wp.array(dtype=wp.vec3),
    rigid_velocity_bt: wp.array(dtype=wp.vec3),
    static_contact_count: wp.array(dtype=wp.int32),
    static_local_a_bt: wp.array(dtype=wp.vec3),
    static_normal: wp.array(dtype=wp.vec3),
    static_tangent: wp.array(dtype=wp.vec3),
    static_mesh: wp.array(dtype=wp.int32),
    static_normal_jacobian: wp.array(dtype=wp.float32),
    static_tangent_jacobian: wp.array(dtype=wp.float32),
    static_normal_rhs: wp.array(dtype=wp.float32),
    static_tangent_rhs: wp.array(dtype=wp.float32),
    static_push_rhs: wp.array(dtype=wp.float32),
    static_normal_impulse: wp.array(dtype=wp.float32),
    static_tangent_impulse: wp.array(dtype=wp.float32),
    static_push_impulse: wp.array(dtype=wp.float32),
    pre_position_bt: wp.array(dtype=wp.vec3),
    pre_velocity_bt: wp.array(dtype=wp.vec3),
    pre_quaternion: wp.array(dtype=wp.quat),
    pre_angular_velocity: wp.array(dtype=wp.vec3),
    pre_on_ground: wp.array(dtype=wp.int32),
    pre_is_supersonic: wp.array(dtype=wp.int32),
    pre_supersonic_time: wp.array(dtype=wp.float32),
    queued_velocity_bt: wp.array(dtype=wp.vec3),
    car_contact_id: wp.array(dtype=wp.int32),
    car_contact_cooldown: wp.array(dtype=wp.float32),
    car_is_demoed: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    return_code: wp.array(dtype=wp.int32),
    algorithm_active: wp.array(dtype=wp.int32),
    contact_point_b_bt: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_distance_bt: wp.array(dtype=wp.float32),
    manifold_local_a_bt: wp.array(dtype=wp.vec3),
    manifold_local_b_bt: wp.array(dtype=wp.vec3),
    manifold_normal: wp.array(dtype=wp.vec3),
    manifold_tangent: wp.array(dtype=wp.vec3),
    manifold_distance_bt: wp.array(dtype=wp.float32),
    manifold_normal_jacobian: wp.array(dtype=wp.float32),
    manifold_tangent_jacobian: wp.array(dtype=wp.float32),
    manifold_normal_rhs: wp.array(dtype=wp.float32),
    manifold_tangent_rhs: wp.array(dtype=wp.float32),
    manifold_push_rhs: wp.array(dtype=wp.float32),
    manifold_normal_impulse: wp.array(dtype=wp.float32),
    manifold_tangent_impulse: wp.array(dtype=wp.float32),
    manifold_push_impulse: wp.array(dtype=wp.float32),
    event_count: wp.array(dtype=wp.int32),
    event_bumper: wp.array(dtype=wp.int32),
    event_victim: wp.array(dtype=wp.int32),
    event_is_demo: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    car_a = env * 2
    car_b = car_a + 1
    position_a = pre_position_bt[car_a]
    position_b = pre_position_bt[car_b]
    quaternion_a = pre_quaternion[car_a]
    quaternion_b = pre_quaternion[car_b]
    basis_a = _bullet_quaternion_matrix(quaternion_a)
    basis_b = _bullet_quaternion_matrix(quaternion_b)
    if tick_counter[0] == 1:
        basis_a = _authority_input_quaternion_matrix(quaternion_a)
        basis_b = _authority_input_quaternion_matrix(quaternion_b)
    velocity_a = pre_velocity_bt[car_a]
    velocity_b = pre_velocity_bt[car_b]
    angular_a = pre_angular_velocity[car_a]
    angular_b = pre_angular_velocity[car_b]
    base = env * MAX_CAR_CAR_CONTACTS
    event_base = env * MAX_CAR_BUMP_EVENTS_PER_TICK
    contact_count[env] = 0
    return_code[env] = 0
    event_count[env] = 0
    queued_velocity_bt[car_a] = wp.vec3(0.0)
    queued_velocity_bt[car_b] = wp.vec3(0.0)
    for relative in range(MAX_CAR_BUMP_EVENTS_PER_TICK):
        event_bumper[event_base + relative] = -1
        event_victim[event_base + relative] = -1
        event_is_demo[event_base + relative] = 0

    predicted_position_a = position_a
    predicted_position_b = position_b
    _bullet_integrate_position(
        position_a, wp.vec3(0.0), velocity_a, DT, 0, predicted_position_a
    )
    _bullet_integrate_position(
        position_b, wp.vec3(0.0), velocity_b, DT, 0, predicted_position_b
    )
    predicted_quaternion_a = quaternion_a
    predicted_quaternion_b = quaternion_b
    _bullet_integrate_quaternion(
        basis_a, angular_a, DT, predicted_quaternion_a
    )
    _bullet_integrate_quaternion(
        basis_b, angular_b, DT, predicted_quaternion_b
    )
    predicted_basis_a = _bullet_quaternion_matrix(predicted_quaternion_a)
    predicted_basis_b = _bullet_quaternion_matrix(predicted_quaternion_b)
    pair_active = _car_car_aabb_overlap(
        position_a,
        basis_a,
        predicted_position_a,
        predicted_basis_a,
        position_b,
        basis_b,
        predicted_position_b,
        predicted_basis_b,
    )
    algorithm_active[env] = pair_active
    if pair_active == 0:
        # The independent car/static islands already produced the exact rigid-
        # body output.  StaticWorldSim updates the gameplay supersonic state in
        # its pre-contact integration kernel, while RocketSim's
        # Car::_PostTickUpdate reads Bullet's velocity after the dynamics-world
        # solve.  Re-run that source operation here from the retained pre-tick
        # gameplay state and the already-solved static-world velocity.
        for local_car in range(2):
            car = car_a + local_car
            if car_is_demoed[car] == 0:
                speed_squared = wp.dot(car_vel[car], car_vel[car])
                supersonic = pre_is_supersonic[car]
                time = pre_supersonic_time[car]
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
                if car_contact_cooldown[car] > 0.0:
                    car_contact_cooldown[car] = wp.max(
                        car_contact_cooldown[car] - DT, 0.0
                    )
        return

    normal = wp.vec3(0.0)
    point0 = wp.vec3(0.0)
    point1 = wp.vec3(0.0)
    point2 = wp.vec3(0.0)
    point3 = wp.vec3(0.0)
    distance0 = wp.float32(0.0)
    distance1 = wp.float32(0.0)
    distance2 = wp.float32(0.0)
    distance3 = wp.float32(0.0)
    candidate_contacts = wp.int32(0)
    code = wp.int32(0)
    pair_manifold_active = _car_car_child_aabb_overlap(
        position_a,
        basis_a,
        position_b,
        basis_b,
    )
    if pair_manifold_active != 0:
        bullet_octane_box_box(
            position_b,
            basis_b,
            position_a,
            basis_a,
            normal,
            point0,
            point1,
            point2,
            point3,
            distance0,
            distance1,
            distance2,
            distance3,
            candidate_contacts,
            code,
        )
    return_code[env] = code
    contacts = candidate_contacts
    for relative in range(MAX_CAR_CAR_CONTACTS):
        if relative < candidate_contacts:
            point_a = point0
            distance = distance0
            if relative == 1:
                point_a = point1
                distance = distance1
            elif relative == 2:
                point_a = point2
                distance = distance2
            elif relative == 3:
                point_a = point3
                distance = distance3
            index = base + relative
            # btManifoldResult::addContactPoint materializes point A as a
            # vector scale followed by a vector add. Preserve those separate
            # float32 operations; a contracted Warp expression changes the
            # stored local witness by one ULP before manifold refresh.
            point_b = _bullet_vector_scale_add(point_a, normal, distance)
            local_a = _bullet_inverse_transform_point(position_a, basis_a, point_a)
            local_b = _bullet_inverse_transform_point(position_b, basis_b, point_b)
            contact_point_b_bt[index] = point_a
            contact_normal[index] = normal
            contact_distance_bt[index] = distance
            manifold_local_a_bt[index] = local_a
            manifold_local_b_bt[index] = local_b
            manifold_normal[index] = normal
            manifold_tangent[index] = wp.vec3(0.0)
            manifold_distance_bt[index] = distance
            manifold_normal_impulse[index] = 0.0
            manifold_tangent_impulse[index] = 0.0
            manifold_push_impulse[index] = 0.0

            # RocketSim invokes the custom-material callback once per inserted
            # manifold point, and that callback tests A->B then B->A.
            for callback_direction in range(2):
                # The nested compound/compound child dispatch presents body B
                # first to RocketSim's custom callback for this fixed pair.
                bumper = car_b
                victim = car_a
                bumper_local = local_b
                bumper_position = position_b
                victim_position = position_a
                bumper_velocity = velocity_b
                victim_velocity = velocity_a
                victim_basis = basis_a
                if callback_direction == 1:
                    bumper = car_a
                    victim = car_b
                    bumper_local = local_a
                    bumper_position = position_a
                    victim_position = position_b
                    bumper_velocity = velocity_a
                    victim_velocity = velocity_b
                    victim_basis = basis_b
                if car_is_demoed[bumper] != 0 or car_is_demoed[victim] != 0:
                    break
                victim_local = victim - env * 2
                bumper_local_id = bumper - env * 2
                if not (
                    car_contact_id[bumper] == victim_local
                    and car_contact_cooldown[bumper] > 0.0
                ):
                    triggered = wp.int32(0)
                    demo = wp.int32(0)
                    bump_velocity = wp.vec3(0.0)
                    _rocketsim_car_car_event(
                        bumper_position,
                        bumper_velocity,
                        victim_position,
                        victim_velocity,
                        bumper_local,
                        pre_is_supersonic[bumper],
                        pre_on_ground[victim],
                        victim_basis,
                        triggered,
                        demo,
                        bump_velocity,
                    )
                    if triggered != 0:
                        if demo != 0:
                            car_is_demoed[victim] = 1
                        else:
                            queued_velocity_bt[victim] = (
                                queued_velocity_bt[victim] + bump_velocity
                            )
                        car_contact_id[bumper] = victim_local
                        car_contact_cooldown[bumper] = 0.25
                        event_index = event_count[env]
                        if event_index < MAX_CAR_BUMP_EVENTS_PER_TICK:
                            event_bumper[event_base + event_index] = bumper_local_id
                            event_victim[event_base + event_index] = victim_local
                            event_is_demo[event_base + event_index] = demo
                            event_count[env] = event_index + 1

    # btBoxBoxCollisionAlgorithm refreshes the complete child manifold after
    # detector.getClosestPoints.  Body 0 is car B in this compound dispatch,
    # so Bullet localPointA is stored in manifold_local_b_bt here.
    contacts_before_refresh = contacts
    for reverse in range(MAX_CAR_CAR_CONTACTS):
        relative = contacts_before_refresh - 1 - reverse
        if relative >= 0:
            index = base + relative
            point_body0 = _bullet_transform_point(
                position_b, basis_b, manifold_local_b_bt[index]
            )
            point_body1 = _bullet_transform_point(
                position_a, basis_a, manifold_local_a_bt[index]
            )
            refreshed_distance = wp.float32(0.0)
            invalid = wp.int32(0)
            _bullet_pair_refresh_contact(
                point_body0,
                point_body1,
                manifold_normal[index],
                CONTACT_BREAKING_THRESHOLD_BT,
                refreshed_distance,
                invalid,
            )
            manifold_distance_bt[index] = refreshed_distance
    for reverse in range(MAX_CAR_CAR_CONTACTS):
        relative = contacts_before_refresh - 1 - reverse
        if relative >= 0 and relative < contacts:
            index = base + relative
            point_body0 = _bullet_transform_point(
                position_b, basis_b, manifold_local_b_bt[index]
            )
            point_body1 = _bullet_transform_point(
                position_a, basis_a, manifold_local_a_bt[index]
            )
            refreshed_distance = wp.float32(0.0)
            invalid = wp.int32(0)
            _bullet_pair_refresh_contact(
                point_body0,
                point_body1,
                manifold_normal[index],
                CONTACT_BREAKING_THRESHOLD_BT,
                refreshed_distance,
                invalid,
            )
            if invalid != 0:
                last = base + contacts - 1
                if index != last:
                    manifold_local_a_bt[index] = manifold_local_a_bt[last]
                    manifold_local_b_bt[index] = manifold_local_b_bt[last]
                    manifold_normal[index] = manifold_normal[last]
                    manifold_tangent[index] = manifold_tangent[last]
                    manifold_distance_bt[index] = manifold_distance_bt[last]
                    manifold_normal_impulse[index] = manifold_normal_impulse[last]
                    manifold_tangent_impulse[index] = manifold_tangent_impulse[last]
                    manifold_push_impulse[index] = manifold_push_impulse[last]
                contacts = contacts - 1
    contact_count[env] = contacts

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
    force_velocity_a = velocity_a + external_linear_a
    force_angular_a = angular_a + external_angular_a
    force_velocity_b = velocity_b + external_linear_b
    force_angular_b = angular_b + external_angular_b
    delta_velocity_a = wp.vec3(0.0)
    delta_angular_a = wp.vec3(0.0)
    delta_velocity_b = wp.vec3(0.0)
    delta_angular_b = wp.vec3(0.0)
    push_a = wp.vec3(0.0)
    turn_a = wp.vec3(0.0)
    push_b = wp.vec3(0.0)
    turn_b = wp.vec3(0.0)

    child_center_a = position_a + _matrix_vector(basis_a, CHILD_OFFSET_BT)
    child_center_b = position_b + _matrix_vector(basis_b, CHILD_OFFSET_BT)
    half = wp.vec3(1.20372915, 0.865653098, 0.385250092)
    extent_a = wp.vec3(
        wp.abs(basis_a[0, 0]) * half[0]
        + wp.abs(basis_a[0, 1]) * half[1]
        + wp.abs(basis_a[0, 2]) * half[2],
        wp.abs(basis_a[1, 0]) * half[0]
        + wp.abs(basis_a[1, 1]) * half[1]
        + wp.abs(basis_a[1, 2]) * half[2],
        wp.abs(basis_a[2, 0]) * half[0]
        + wp.abs(basis_a[2, 1]) * half[1]
        + wp.abs(basis_a[2, 2]) * half[2],
    ) + wp.vec3(
        RS_CAR_PROXY_AABB_EXPANSION_BT,
        RS_CAR_PROXY_AABB_EXPANSION_BT,
        RS_CAR_PROXY_AABB_EXPANSION_BT,
    )
    extent_b = wp.vec3(
        wp.abs(basis_b[0, 0]) * half[0]
        + wp.abs(basis_b[0, 1]) * half[1]
        + wp.abs(basis_b[0, 2]) * half[2],
        wp.abs(basis_b[1, 0]) * half[0]
        + wp.abs(basis_b[1, 1]) * half[1]
        + wp.abs(basis_b[1, 2]) * half[2],
        wp.abs(basis_b[2, 0]) * half[0]
        + wp.abs(basis_b[2, 1]) * half[1]
        + wp.abs(basis_b[2, 2]) * half[2],
    ) + wp.vec3(
        RS_CAR_PROXY_AABB_EXPANSION_BT,
        RS_CAR_PROXY_AABB_EXPANSION_BT,
        RS_CAR_PROXY_AABB_EXPANSION_BT,
    )
    static_mask_a = _rs_static_overlap_mask(
        child_center_a - extent_a,
        child_center_a + extent_a,
        rs_static_cell_mask,
        rs_static_aabb_min_bt,
        rs_static_aabb_max_bt,
    )
    static_mask_b = _rs_static_overlap_mask(
        child_center_b - extent_b,
        child_center_b + extent_b,
        rs_static_cell_mask,
        rs_static_aabb_min_bt,
        rs_static_aabb_max_bt,
    )
    manifolds_a = _rs_mask_count(static_mask_a)
    manifolds_b = _rs_mask_count(static_mask_b)
    island_manifolds = manifolds_a + pair_manifold_active + manifolds_b
    static_contacts_a = static_contact_count[car_a]
    static_contacts_b = static_contact_count[car_b]
    for relative in range(MAX_CONTACTS_PER_CAR):
        if relative < static_contacts_a:
            index = car_a * MAX_CONTACTS_PER_CAR + relative
            static_normal_impulse[index] = 0.0
            static_tangent_impulse[index] = 0.0
            static_push_impulse[index] = 0.0
        if relative < static_contacts_b:
            index = car_b * MAX_CONTACTS_PER_CAR + relative
            static_normal_impulse[index] = 0.0
            static_tangent_impulse[index] = 0.0
            static_push_impulse[index] = 0.0

    for relative in range(MAX_CAR_CAR_CONTACTS):
        if relative < contacts:
            index = base + relative
            point_a = _bullet_transform_point(
                position_a, basis_a, manifold_local_a_bt[index]
            )
            point_b = _bullet_transform_point(
                position_b, basis_b, manifold_local_b_bt[index]
            )
            refreshed_distance = manifold_distance_bt[index]
            tangent = wp.vec3(0.0)
            normal_jacobian = wp.float32(0.0)
            tangent_jacobian = wp.float32(0.0)
            normal_rhs_value = wp.float32(0.0)
            tangent_rhs_value = wp.float32(0.0)
            push_rhs_value = wp.float32(0.0)
            _bullet_car_car_contact_row(
                basis_b,
                basis_a,
                point_b - position_b,
                point_a - position_a,
                refreshed_distance,
                DT,
                manifold_normal[index],
                velocity_b,
                angular_b,
                force_velocity_b,
                force_angular_b,
                velocity_a,
                angular_a,
                force_velocity_a,
                force_angular_a,
                tangent,
                normal_jacobian,
                tangent_jacobian,
                normal_rhs_value,
                tangent_rhs_value,
                push_rhs_value,
            )
            manifold_tangent[index] = tangent
            manifold_normal_jacobian[index] = normal_jacobian
            manifold_tangent_jacobian[index] = tangent_jacobian
            manifold_normal_rhs[index] = normal_rhs_value
            manifold_tangent_rhs[index] = tangent_rhs_value
            manifold_push_rhs[index] = push_rhs_value

    # Split impulse: every normal row in globally sorted manifold order.
    for _iteration in range(SOLVER_ITERATIONS):
        for sorted_slot in range(RS_MAX_ISLAND_MANIFOLDS):
            if sorted_slot < island_manifolds:
                manifold_code = _sorted_car_car_manifold_code(
                    sorted_slot,
                    static_mask_a,
                    static_mask_b,
                    manifolds_a,
                    pair_manifold_active,
                    island_manifolds,
                    rs_equal_island_permutation,
                )
                if manifold_code == 32:
                    for relative in range(MAX_CAR_CAR_CONTACTS):
                        if relative < contacts:
                            index = base + relative
                            point_a = _bullet_transform_point(
                                position_a, basis_a, manifold_local_a_bt[index]
                            )
                            point_b = _bullet_transform_point(
                                position_b, basis_b, manifold_local_b_bt[index]
                            )
                            applied = wp.float32(manifold_push_impulse[index])
                            _bullet_car_car_solve_split_row(
                                basis_b,
                                basis_a,
                                manifold_normal[index],
                                point_b - position_b,
                                point_a - position_a,
                                manifold_normal_jacobian[index],
                                manifold_push_rhs[index],
                                0.0,
                                1.0e10,
                                push_b,
                                turn_b,
                                push_a,
                                turn_a,
                                applied,
                            )
                            manifold_push_impulse[index] = applied
                else:
                    car = car_a
                    position = position_a
                    basis = basis_a
                    push = push_a
                    turn = turn_a
                    static_body = manifold_code - 64
                    static_contacts = static_contacts_a
                    if manifold_code >= 96:
                        car = car_b
                        position = position_b
                        basis = basis_b
                        push = push_b
                        turn = turn_b
                        static_body = manifold_code - 96
                        static_contacts = static_contacts_b
                    for relative in range(MAX_CONTACTS_PER_CAR):
                        if relative < static_contacts:
                            index = car * MAX_CONTACTS_PER_CAR + relative
                            if static_mesh[index] == static_body:
                                point = _bullet_transform_point(
                                    position, basis, static_local_a_bt[index]
                                )
                                applied = wp.float32(static_push_impulse[index])
                                _bullet_solve_split_row(
                                    basis,
                                    static_normal[index],
                                    point - position,
                                    static_normal_jacobian[index],
                                    static_push_rhs[index],
                                    push,
                                    turn,
                                    applied,
                                )
                                static_push_impulse[index] = applied
                    if manifold_code >= 96:
                        push_b = push
                        turn_b = turn
                    else:
                        push_a = push
                        turn_a = turn

    # Velocity solve: all normal rows, then all friction rows, per iteration.
    for _iteration in range(SOLVER_ITERATIONS):
        for friction_pass in range(2):
            for sorted_slot in range(RS_MAX_ISLAND_MANIFOLDS):
                if sorted_slot < island_manifolds:
                    manifold_code = _sorted_car_car_manifold_code(
                        sorted_slot,
                        static_mask_a,
                        static_mask_b,
                        manifolds_a,
                        pair_manifold_active,
                        island_manifolds,
                        rs_equal_island_permutation,
                    )
                    if manifold_code == 32:
                        for relative in range(MAX_CAR_CAR_CONTACTS):
                            if relative < contacts:
                                index = base + relative
                                normal_impulse = manifold_normal_impulse[index]
                                if friction_pass == 0 or normal_impulse > 0.0:
                                    row_direction = manifold_normal[index]
                                    jacobian = manifold_normal_jacobian[index]
                                    rhs = manifold_normal_rhs[index]
                                    lower = 0.0
                                    upper = 1.0e10
                                    applied = wp.float32(normal_impulse)
                                    if friction_pass == 1:
                                        row_direction = manifold_tangent[index]
                                        jacobian = manifold_tangent_jacobian[index]
                                        rhs = manifold_tangent_rhs[index]
                                        limit = CONTACT_FRICTION * normal_impulse
                                        lower = -limit
                                        upper = limit
                                        applied = wp.float32(
                                            manifold_tangent_impulse[index]
                                        )
                                    point_a = _bullet_transform_point(
                                        position_a,
                                        basis_a,
                                        manifold_local_a_bt[index],
                                    )
                                    point_b = _bullet_transform_point(
                                        position_b,
                                        basis_b,
                                        manifold_local_b_bt[index],
                                    )
                                    _bullet_car_car_solve_velocity_row(
                                        basis_b,
                                        basis_a,
                                        row_direction,
                                        point_b - position_b,
                                        point_a - position_a,
                                        jacobian,
                                        rhs,
                                        lower,
                                        upper,
                                        delta_velocity_b,
                                        delta_angular_b,
                                        delta_velocity_a,
                                        delta_angular_a,
                                        applied,
                                    )
                                    if friction_pass == 0:
                                        manifold_normal_impulse[index] = applied
                                    else:
                                        manifold_tangent_impulse[index] = applied
                    else:
                        car = car_a
                        position = position_a
                        basis = basis_a
                        delta_velocity = delta_velocity_a
                        delta_angular = delta_angular_a
                        static_body = manifold_code - 64
                        static_contacts = static_contacts_a
                        if manifold_code >= 96:
                            car = car_b
                            position = position_b
                            basis = basis_b
                            delta_velocity = delta_velocity_b
                            delta_angular = delta_angular_b
                            static_body = manifold_code - 96
                            static_contacts = static_contacts_b
                        for relative in range(MAX_CONTACTS_PER_CAR):
                            if relative < static_contacts:
                                index = car * MAX_CONTACTS_PER_CAR + relative
                                if static_mesh[index] == static_body:
                                    normal_impulse = static_normal_impulse[index]
                                    if friction_pass == 0 or normal_impulse > 0.0:
                                        row_direction = static_normal[index]
                                        jacobian = static_normal_jacobian[index]
                                        rhs = static_normal_rhs[index]
                                        lower = 0.0
                                        upper = 1.0e10
                                        applied = wp.float32(normal_impulse)
                                        if friction_pass == 1:
                                            row_direction = static_tangent[index]
                                            jacobian = static_tangent_jacobian[index]
                                            rhs = static_tangent_rhs[index]
                                            limit = CAR_STATIC_FRICTION * normal_impulse
                                            lower = -limit
                                            upper = limit
                                            applied = wp.float32(
                                                static_tangent_impulse[index]
                                            )
                                        point = _bullet_transform_point(
                                            position, basis, static_local_a_bt[index]
                                        )
                                        _bullet_solve_velocity_row(
                                            basis,
                                            row_direction,
                                            point - position,
                                            jacobian,
                                            rhs,
                                            lower,
                                            upper,
                                            delta_velocity,
                                            delta_angular,
                                            applied,
                                        )
                                        if friction_pass == 0:
                                            static_normal_impulse[index] = applied
                                        else:
                                            static_tangent_impulse[index] = applied
                        if manifold_code >= 96:
                            delta_velocity_b = delta_velocity
                            delta_angular_b = delta_angular
                        else:
                            delta_velocity_a = delta_velocity
                            delta_angular_a = delta_angular


    solved_velocity_a = (velocity_a + delta_velocity_a) + external_linear_a
    solved_angular_a = (angular_a + delta_angular_a) + external_angular_a
    solved_velocity_b = (velocity_b + delta_velocity_b) + external_linear_b
    solved_angular_b = (angular_b + delta_angular_b) + external_angular_b
    split_a = wp.int32(wp.dot(push_a, push_a) > 0.0 or wp.dot(turn_a, turn_a) > 0.0)
    split_b = wp.int32(wp.dot(push_b, push_b) > 0.0 or wp.dot(turn_b, turn_b) > 0.0)
    split_quaternion_a = quaternion_a
    split_quaternion_b = quaternion_b
    integration_basis_a = basis_a
    integration_basis_b = basis_b
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
    integrated_position_a = position_a
    integrated_position_b = position_b
    _bullet_integrate_position(
        position_a, push_a, solved_velocity_a, DT, split_a, integrated_position_a
    )
    _bullet_integrate_position(
        position_b, push_b, solved_velocity_b, DT, split_b, integrated_position_b
    )
    integrated_quaternion_a = split_quaternion_a
    integrated_quaternion_b = split_quaternion_b
    _bullet_integrate_quaternion(
        integration_basis_a, solved_angular_a, DT, integrated_quaternion_a
    )
    _bullet_integrate_quaternion(
        integration_basis_b, solved_angular_b, DT, integrated_quaternion_b
    )

    # Car::_PostTickUpdate occurs before queued bump impulses are applied.
    for local_car in range(2):
        car = car_a + local_car
        solved_velocity = solved_velocity_a
        if local_car == 1:
            solved_velocity = solved_velocity_b
        speed_uu = solved_velocity * 50.0
        speed_squared = wp.dot(speed_uu, speed_uu)
        if car_is_demoed[car] == 0:
            supersonic = pre_is_supersonic[car]
            time = pre_supersonic_time[car]
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
            if car_contact_cooldown[car] > 0.0:
                car_contact_cooldown[car] = wp.max(
                    car_contact_cooldown[car] - DT, 0.0
                )
        else:
            car_is_supersonic[car] = pre_is_supersonic[car]
            car_supersonic_time[car] = pre_supersonic_time[car]

    if car_is_demoed[car_a] == 0:
        solved_velocity_a = solved_velocity_a + queued_velocity_bt[car_a]
        _bullet_cap_amd(
            solved_velocity_a, CAR_MAX_SPEED_BT, amd_rsqrtss_mantissa
        )
        _bullet_cap_amd(
            solved_angular_a, CAR_MAX_ANGULAR_SPEED, amd_rsqrtss_mantissa
        )
    if car_is_demoed[car_b] == 0:
        solved_velocity_b = solved_velocity_b + queued_velocity_bt[car_b]
        _bullet_cap_amd(
            solved_velocity_b, CAR_MAX_SPEED_BT, amd_rsqrtss_mantissa
        )
        _bullet_cap_amd(
            solved_angular_b, CAR_MAX_ANGULAR_SPEED, amd_rsqrtss_mantissa
        )
    rigid_position_bt[car_a] = integrated_position_a
    rigid_position_bt[car_b] = integrated_position_b
    rigid_velocity_bt[car_a] = solved_velocity_a
    rigid_velocity_bt[car_b] = solved_velocity_b
    car_pos[car_a] = integrated_position_a * 50.0
    car_pos[car_b] = integrated_position_b * 50.0
    car_vel[car_a] = solved_velocity_a * 50.0
    car_vel[car_b] = solved_velocity_b * 50.0
    car_quat[car_a] = integrated_quaternion_a
    car_quat[car_b] = integrated_quaternion_b
    car_ang_vel[car_a] = solved_angular_a
    car_ang_vel[car_b] = solved_angular_b


__all__ = [
    "capture_car_car_inputs",
    "car_car_tick",
]
