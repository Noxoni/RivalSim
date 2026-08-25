"""Bounded RocketSim/Bullet Octane/standard-ball collision path."""

import warp as wp

from rivalsim.ball_world_state import MAX_BALL_CONTACTS
from rivalsim.car_ball_state import MAX_CAR_BALL_CONTACTS
from rivalsim.kernels.ball_world import (
    _bullet_ball_solve_split_row,
    _bullet_ball_solve_velocity_row,
    _bullet_ball_special_contact_row,
)
from rivalsim.kernels.bullet_box_sphere import (
    BALL_RADIUS_BT,
    bullet_box_sphere_closest,
    bullet_box_sphere_penetration,
)
from rivalsim.kernels.bullet_box_triangle import bullet_manifold_replacement
from rivalsim.kernels.vehicle import (
    _authority_input_quaternion_matrix,
    _bullet_apply_impulse,
    _bullet_integrate_external_velocities,
    _bullet_integrate_position,
    _bullet_integrate_quaternion,
    _bullet_inverse_transform_point,
    _bullet_quaternion_matrix,
    _bullet_solve_split_row,
    _bullet_solve_velocity_row,
    _bullet_transform_point,
)
from rivalsim.vehicle_state import MAX_CONTACTS_PER_CAR

DT = 0.008333333767950535
CAR_INV_MASS = 0.0055555556900799274
BALL_INV_MASS = 0.03333333507180214
BALL_INV_INERTIA = 0.025020327419042587
CAR_MAX_SPEED_BT = 46.0
BALL_MAX_SPEED_BT = 120.0
CAR_MAX_ANGULAR_SPEED = 5.5
BALL_MAX_ANGULAR_SPEED = 6.0
BALL_DAMPING = 0.03
# btCollisionDispatcher creates the child-box/sphere manifold from the two
# collision objects (Octane compound and sphere). Relative contact breaking
# chooses the sphere's smaller threshold, observed from the pinned source as
# exactly this float32 value.
CONTACT_BREAKING_THRESHOLD_BT = 0.0380999967
BOX_MARGIN_BT = 0.0386590995
CONTACT_ERP2 = 0.8
CONTACT_SPLIT_TURN_ERP = 0.1
CONTACT_FRICTION = 2.0
CAR_STATIC_FRICTION = 0.3
BALL_STATIC_FRICTION = 0.35
SOLVER_ITERATIONS = 10
WARMSTART_FACTOR = 0.85
CHILD_OFFSET_BT = wp.vec3(0.277513981, 0.0, 0.415099978)
BOX_HALF_WITH_MARGIN_BT = wp.vec3(1.20372915, 0.865653098, 0.385250092)
RS_BROADPHASE_MIN_BT = wp.vec3(-112.0, -120.0, 0.0)
RS_BROADPHASE_CELL_SIZE_BT = 7.4
RS_BROADPHASE_CELLS_Y = 33
RS_BROADPHASE_CELLS_Z = 6
RS_STATIC_BODY_COUNT = 20
RS_MAX_ISLAND_MANIFOLDS = 41
RS_EQUAL_ISLAND_PERMUTATION_WIDTH = 41
RS_CONTACT_AABB_EXPANSION_BT = 0.08
# btCollisionWorld::updateSingleAabb expands the compound car proxy by
# gContactBreakingThreshold.  The separate 0.08 RocketSim change applies only
# to btSphereShape::getAabb and must not enlarge the car proxy.
RS_CAR_PROXY_AABB_EXPANSION_BT = 0.02


_BULLET_SPECIAL_DISTANCE_ACCUMULATE = r"""
    // convertContact accumulates rel_pos.length() into btSpecialResolveInfo.
    // btVector3's pinned SSE path multiplies each lane, adds X/Y before Z,
    // takes the scalar square root, and only then rounds the running sum.
    auto add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    const float x = relative_position_bt[0];
    const float y = relative_position_bt[1];
    const float z = relative_position_bt[2];
    const float length_squared = add(add(mul(x, x), mul(y, y)), mul(z, z));
    float length;
    #if defined(__CUDA_ARCH__)
        length = __fsqrt_rn(length_squared);
    #else
        volatile float result = ::sqrtf(length_squared);
        length = result;
    #endif
    distance_sum = add(distance_sum, length);
"""


@wp.func_native(_BULLET_SPECIAL_DISTANCE_ACCUMULATE)
def _bullet_special_distance_accumulate(
    relative_position_bt: wp.vec3,
    distance_sum: wp.ref[wp.float32],
): ...


_BULLET_SPECIAL_AVERAGE = r"""
    float inverse_count;
    #if defined(__CUDA_ARCH__)
        inverse_count = __fdiv_rn(1.0f, static_cast<float>(contact_count));
        average_normal = wp::vec_t<3,wp::float32>(
            __fmul_rn(normal_sum[0], inverse_count),
            __fmul_rn(normal_sum[1], inverse_count),
            __fmul_rn(normal_sum[2], inverse_count));
        // m_totalNormal uses btVector3::operator/ (reciprocal then multiply),
        // while scalar m_totalDist / count remains a direct DIVSS.
        average_distance = __fdiv_rn(
            distance_sum, static_cast<float>(contact_count));
    #else
        volatile float count_value = static_cast<float>(contact_count);
        volatile float reciprocal = 1.0f / count_value;
        inverse_count = reciprocal;
        volatile float nx = normal_sum[0] * inverse_count;
        volatile float ny = normal_sum[1] * inverse_count;
        volatile float nz = normal_sum[2] * inverse_count;
        volatile float distance = distance_sum / count_value;
        average_normal = wp::vec_t<3,wp::float32>(nx, ny, nz);
        average_distance = distance;
    #endif
"""


@wp.func_native(_BULLET_SPECIAL_AVERAGE)
def _bullet_special_average(
    normal_sum: wp.vec3,
    distance_sum: float,
    contact_count: int,
    average_normal: wp.ref[wp.vec3],
    average_distance: wp.ref[wp.float32],
): ...


@wp.func
def _matrix_vector(matrix: wp.mat33, value: wp.vec3) -> wp.vec3:
    return wp.vec3(
        matrix[0, 0] * value[0]
        + matrix[0, 1] * value[1]
        + matrix[0, 2] * value[2],
        matrix[1, 0] * value[0]
        + matrix[1, 1] * value[1]
        + matrix[1, 2] * value[2],
        matrix[2, 0] * value[0]
        + matrix[2, 1] * value[1]
        + matrix[2, 2] * value[2],
    )


@wp.func
def _pair_denominator(
    car_inverse_inertia: wp.mat33,
    relative_car: wp.vec3,
    relative_ball: wp.vec3,
    axis: wp.vec3,
) -> float:
    car_torque = wp.cross(relative_car, axis)
    car_angular = _matrix_vector(car_inverse_inertia, car_torque)
    ball_axis = -axis
    ball_torque = wp.cross(relative_ball, ball_axis)
    ball_angular = ball_torque * BALL_INV_INERTIA
    car_term = wp.dot(axis, wp.cross(car_angular, relative_car))
    ball_term = wp.dot(ball_axis, wp.cross(ball_angular, relative_ball))
    return CAR_INV_MASS + BALL_INV_MASS + car_term + ball_term


@wp.func
def _pair_tangent(normal: wp.vec3, relative_velocity: wp.vec3) -> wp.vec3:
    tangent = relative_velocity - normal * wp.dot(normal, relative_velocity)
    length_squared = wp.dot(tangent, tangent)
    if length_squared > 1.1920928955078125e-7:
        return tangent / wp.sqrt(length_squared)
    if wp.abs(normal[2]) > 0.7071067811865476:
        inverse = 1.0 / wp.sqrt(normal[1] * normal[1] + normal[2] * normal[2])
        return wp.vec3(0.0, -normal[2] * inverse, normal[1] * inverse)
    inverse = 1.0 / wp.sqrt(normal[0] * normal[0] + normal[1] * normal[1])
    return wp.vec3(-normal[1] * inverse, normal[0] * inverse, 0.0)


_BULLET_PAIR_CONTACT_ROW = r"""
    // setupContactConstraint, convertContactInner's velocity-dependent
    // friction direction, and setupFrictionConstraint for the pinned
    // Octane/standard-ball pair. Both inverse inertia tensors are
    // materialized exactly as btRigidBody::updateInertiaTensor does; the
    // ball's locally isotropic inertia must not be algebraically collapsed.
    auto op_add=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a,b);
    #else
        volatile float v=a+b;return v;
    #endif
    };
    auto op_sub=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a,b);
    #else
        volatile float v=a-b;return v;
    #endif
    };
    auto op_mul=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a,b);
    #else
        volatile float v=a*b;return v;
    #endif
    };
    auto op_div=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a,b);
    #else
        volatile float v=a/b;return v;
    #endif
    };
    struct PairV3{float x;float y;float z;};
    auto make=[](float x,float y,float z)->PairV3{PairV3 v={x,y,z};return v;};
    auto add=[&](PairV3 a,PairV3 b)->PairV3{return make(op_add(a.x,b.x),op_add(a.y,b.y),op_add(a.z,b.z));};
    auto sub=[&](PairV3 a,PairV3 b)->PairV3{return make(op_sub(a.x,b.x),op_sub(a.y,b.y),op_sub(a.z,b.z));};
    auto neg=[&](PairV3 a)->PairV3{return make(-a.x,-a.y,-a.z);};
    auto scale=[&](PairV3 a,float s)->PairV3{return make(op_mul(a.x,s),op_mul(a.y,s),op_mul(a.z,s));};
    auto cross=[&](PairV3 a,PairV3 b)->PairV3{return make(
        op_sub(op_mul(a.y,b.z),op_mul(a.z,b.y)),
        op_sub(op_mul(a.z,b.x),op_mul(a.x,b.z)),
        op_sub(op_mul(a.x,b.y),op_mul(a.y,b.x)));};
    auto dot=[&](PairV3 a,PairV3 b)->float{return op_add(op_add(op_mul(a.x,b.x),op_mul(a.y,b.y)),op_mul(a.z,b.z));};
    const float car_inverse_local[3]={0.0185644571f,0.0104337428f,0.0075815497f};
    const float ball_inverse_local[3]={0.0250203293f,0.0250203293f,0.0250203293f};
    float car_scaled[3][3];float car_tensor[3][3];
    float ball_scaled[3][3];float ball_tensor[3][3];
    for(int r=0;r<3;++r)for(int c=0;c<3;++c){
        car_scaled[r][c]=op_mul(car_basis.data[r][c],car_inverse_local[c]);
        ball_scaled[r][c]=op_mul(ball_basis.data[r][c],ball_inverse_local[c]);
    }
    for(int r=0;r<3;++r)for(int c=0;c<3;++c){
        car_tensor[r][c]=op_add(op_add(op_mul(car_scaled[r][0],car_basis.data[c][0]),op_mul(car_scaled[r][1],car_basis.data[c][1])),op_mul(car_scaled[r][2],car_basis.data[c][2]));
        ball_tensor[r][c]=op_add(op_add(op_mul(ball_scaled[r][0],ball_basis.data[c][0]),op_mul(ball_scaled[r][1],ball_basis.data[c][1])),op_mul(ball_scaled[r][2],ball_basis.data[c][2]));
    }
    auto car_matrix=[&](PairV3 v)->PairV3{return make(
        op_add(op_add(op_mul(car_tensor[0][0],v.x),op_mul(car_tensor[0][1],v.y)),op_mul(car_tensor[0][2],v.z)),
        op_add(op_add(op_mul(car_tensor[1][0],v.x),op_mul(car_tensor[1][1],v.y)),op_mul(car_tensor[1][2],v.z)),
        op_add(op_add(op_mul(car_tensor[2][0],v.x),op_mul(car_tensor[2][1],v.y)),op_mul(car_tensor[2][2],v.z)));};
    auto ball_matrix=[&](PairV3 v)->PairV3{return make(
        op_add(op_add(op_mul(ball_tensor[0][0],v.x),op_mul(ball_tensor[0][1],v.y)),op_mul(ball_tensor[0][2],v.z)),
        op_add(op_add(op_mul(ball_tensor[1][0],v.x),op_mul(ball_tensor[1][1],v.y)),op_mul(ball_tensor[1][2],v.z)),
        op_add(op_add(op_mul(ball_tensor[2][0],v.x),op_mul(ball_tensor[2][1],v.y)),op_mul(ball_tensor[2][2],v.z)));};
    const PairV3 rel_a=make(relative_car_bt[0],relative_car_bt[1],relative_car_bt[2]);
    const PairV3 rel_b=make(relative_ball_bt[0],relative_ball_bt[1],relative_ball_bt[2]);
    const PairV3 n=make(normal[0],normal[1],normal[2]);
    const PairV3 car_pre_l=make(car_pre_linear_bt[0],car_pre_linear_bt[1],car_pre_linear_bt[2]);
    const PairV3 car_pre_a=make(car_pre_angular_world[0],car_pre_angular_world[1],car_pre_angular_world[2]);
    const PairV3 car_force_l=make(car_force_linear_bt[0],car_force_linear_bt[1],car_force_linear_bt[2]);
    const PairV3 car_force_a=make(car_force_angular_world[0],car_force_angular_world[1],car_force_angular_world[2]);
    const PairV3 ball_pre_l=make(ball_pre_linear_bt[0],ball_pre_linear_bt[1],ball_pre_linear_bt[2]);
    const PairV3 ball_pre_a=make(ball_pre_angular_world[0],ball_pre_angular_world[1],ball_pre_angular_world[2]);
    const PairV3 ball_force_l=make(ball_force_linear_bt[0],ball_force_linear_bt[1],ball_force_linear_bt[2]);
    const PairV3 ball_force_a=make(ball_force_angular_world[0],ball_force_angular_world[1],ball_force_angular_world[2]);

    const PairV3 normal_torque_a=cross(rel_a,n);
    const PairV3 normal_torque_b=cross(rel_b,n);
    const PairV3 normal_angular_a=car_matrix(normal_torque_a);
    const PairV3 normal_angular_b=ball_matrix(neg(normal_torque_b));
    const float normal_denom_a=op_add(0.00555555569f,dot(n,cross(normal_angular_a,rel_a)));
    const float normal_denom_b=op_add(0.0333333351f,dot(n,cross(neg(normal_angular_b),rel_b)));
    const float normal_inverse=op_div(1.0f,op_add(normal_denom_a,normal_denom_b));
    const PairV3 normal_two=neg(n);
    const PairV3 normal_cross_two=neg(normal_torque_b);
    const float normal_speed_a=op_add(dot(n,car_force_l),dot(normal_torque_a,car_force_a));
    const float normal_speed_b=op_add(dot(normal_two,ball_force_l),dot(normal_cross_two,ball_force_a));
    const float relative_normal_speed=op_add(normal_speed_a,normal_speed_b);
    normal_jacobian=normal_inverse;
    normal_rhs=op_mul(op_sub(0.0f,relative_normal_speed),normal_inverse);

    // getVelocityInLocalPointNoDelta includes external torque while choosing
    // the velocity-dependent direction.
    const PairV3 car_point=add(car_force_l,cross(car_force_a,rel_a));
    const PairV3 ball_point=add(ball_force_l,cross(ball_force_a,rel_b));
    const PairV3 relative_point_velocity=sub(car_point,ball_point);
    const float projected=dot(n,relative_point_velocity);
    PairV3 friction=sub(relative_point_velocity,scale(n,projected));
    const float friction_length_squared=dot(friction,friction);
    if(friction_length_squared>1.1920928955078125e-7f){
        friction=scale(friction,op_div(1.0f,sqrtf(friction_length_squared)));
    }else if(fabsf(n.z)>0.7071067811865476f){
        const float amount=op_add(op_mul(n.y,n.y),op_mul(n.z,n.z));
        const float inverse=op_div(1.0f,sqrtf(amount));
        friction=make(0.0f,op_mul(-n.z,inverse),op_mul(n.y,inverse));
    }else{
        const float amount=op_add(op_mul(n.x,n.x),op_mul(n.y,n.y));
        const float inverse=op_div(1.0f,sqrtf(amount));
        friction=make(op_mul(-n.y,inverse),op_mul(n.x,inverse),0.0f);
    }
    const PairV3 friction_two=neg(friction);
    const PairV3 friction_torque_a=cross(rel_a,friction);
    const PairV3 friction_torque_b=cross(rel_b,friction_two);
    const PairV3 friction_angular_a=car_matrix(friction_torque_a);
    const PairV3 friction_angular_b=ball_matrix(friction_torque_b);
    const float friction_denom_a=op_add(0.00555555569f,dot(friction,cross(friction_angular_a,rel_a)));
    const float friction_denom_b=op_add(0.0333333351f,dot(friction,cross(neg(friction_angular_b),rel_b)));
    const float friction_inverse=op_div(1.0f,op_add(friction_denom_a,friction_denom_b));
    // setupFrictionConstraint includes each external linear impulse but omits
    // each external torque impulse in its RHS.
    const float friction_speed_a=op_add(dot(friction,car_force_l),dot(friction_torque_a,car_pre_a));
    const float friction_speed_b=op_add(dot(friction_two,ball_force_l),dot(friction_torque_b,ball_pre_a));
    const float relative_friction_speed=op_add(friction_speed_a,friction_speed_b);
    tangent_jacobian=friction_inverse;
    tangent_rhs=op_mul(op_sub(0.0f,relative_friction_speed),friction_inverse);
    tangent=wp::vec_t<3,wp::float32>(friction.x,friction.y,friction.z);

    const float penetration=op_add(distance_bt,0.0f);
    const float inverse_time_step=op_div(1.0f,time_step);
    push_rhs=0.0f;
    if(penetration<=0.0f){
        float positional_error=op_mul(-penetration,0.8f);
        positional_error=op_mul(positional_error,inverse_time_step);
        push_rhs=op_mul(positional_error,normal_inverse);
    }
"""


@wp.func_native(_BULLET_PAIR_CONTACT_ROW)
def _bullet_pair_contact_row(
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


_BULLET_PAIR_SOLVE_VELOCITY_ROW = r"""
    // gResolveSingleConstraintRowGeneric_sse4_1_fma3 for two dynamic bodies.
    auto op_add=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a,b);
    #else
        volatile float v=a+b;return v;
    #endif
    };
    auto op_sub=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a,b);
    #else
        volatile float v=a-b;return v;
    #endif
    };
    auto op_mul=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a,b);
    #else
        volatile float v=a*b;return v;
    #endif
    };
    auto op_fma=[](float a,float b,float c)->float{
    #if defined(__CUDA_ARCH__)
        return __fmaf_rn(a,b,c);
    #else
        volatile float v=fmaf(a,b,c);return v;
    #endif
    };
    struct PairVelV3{float x;float y;float z;};
    auto make=[](float x,float y,float z)->PairVelV3{PairVelV3 v={x,y,z};return v;};
    auto neg=[&](PairVelV3 a)->PairVelV3{return make(-a.x,-a.y,-a.z);};
    auto cross=[&](PairVelV3 a,PairVelV3 b)->PairVelV3{return make(
        op_sub(op_mul(a.y,b.z),op_mul(a.z,b.y)),
        op_sub(op_mul(a.z,b.x),op_mul(a.x,b.z)),
        op_sub(op_mul(a.x,b.y),op_mul(a.y,b.x)));};
    auto dot=[&](PairVelV3 a,PairVelV3 b)->float{return op_add(op_add(op_mul(a.x,b.x),op_mul(a.y,b.y)),op_mul(a.z,b.z));};
    const float car_inverse_local[3]={0.0185644571f,0.0104337428f,0.0075815497f};
    const float ball_inverse_local[3]={0.0250203293f,0.0250203293f,0.0250203293f};
    float car_scaled[3][3];float car_tensor[3][3];float ball_scaled[3][3];float ball_tensor[3][3];
    for(int r=0;r<3;++r)for(int c=0;c<3;++c){car_scaled[r][c]=op_mul(car_basis.data[r][c],car_inverse_local[c]);ball_scaled[r][c]=op_mul(ball_basis.data[r][c],ball_inverse_local[c]);}
    for(int r=0;r<3;++r)for(int c=0;c<3;++c){
        car_tensor[r][c]=op_add(op_add(op_mul(car_scaled[r][0],car_basis.data[c][0]),op_mul(car_scaled[r][1],car_basis.data[c][1])),op_mul(car_scaled[r][2],car_basis.data[c][2]));
        ball_tensor[r][c]=op_add(op_add(op_mul(ball_scaled[r][0],ball_basis.data[c][0]),op_mul(ball_scaled[r][1],ball_basis.data[c][1])),op_mul(ball_scaled[r][2],ball_basis.data[c][2]));
    }
    auto car_matrix=[&](PairVelV3 v)->PairVelV3{return make(
        op_add(op_add(op_mul(car_tensor[0][0],v.x),op_mul(car_tensor[0][1],v.y)),op_mul(car_tensor[0][2],v.z)),
        op_add(op_add(op_mul(car_tensor[1][0],v.x),op_mul(car_tensor[1][1],v.y)),op_mul(car_tensor[1][2],v.z)),
        op_add(op_add(op_mul(car_tensor[2][0],v.x),op_mul(car_tensor[2][1],v.y)),op_mul(car_tensor[2][2],v.z)));};
    auto ball_matrix=[&](PairVelV3 v)->PairVelV3{return make(
        op_add(op_add(op_mul(ball_tensor[0][0],v.x),op_mul(ball_tensor[0][1],v.y)),op_mul(ball_tensor[0][2],v.z)),
        op_add(op_add(op_mul(ball_tensor[1][0],v.x),op_mul(ball_tensor[1][1],v.y)),op_mul(ball_tensor[1][2],v.z)),
        op_add(op_add(op_mul(ball_tensor[2][0],v.x),op_mul(ball_tensor[2][1],v.y)),op_mul(ball_tensor[2][2],v.z)));};
    const PairVelV3 n=make(direction[0],direction[1],direction[2]);
    const PairVelV3 n_two=neg(n);
    const PairVelV3 rel_a=make(relative_car_bt[0],relative_car_bt[1],relative_car_bt[2]);
    const PairVelV3 rel_b=make(relative_ball_bt[0],relative_ball_bt[1],relative_ball_bt[2]);
    const PairVelV3 torque_a=cross(rel_a,n);
    const PairVelV3 torque_b=neg(cross(rel_b,n));
    const PairVelV3 angular_a=car_matrix(torque_a);
    const PairVelV3 angular_b=ball_matrix(torque_b);
    PairVelV3 linear_a=make(car_delta_linear_bt[0],car_delta_linear_bt[1],car_delta_linear_bt[2]);
    PairVelV3 delta_angular_a=make(car_delta_angular_world[0],car_delta_angular_world[1],car_delta_angular_world[2]);
    PairVelV3 linear_b=make(ball_delta_linear_bt[0],ball_delta_linear_bt[1],ball_delta_linear_bt[2]);
    PairVelV3 delta_angular_b=make(ball_delta_angular_world[0],ball_delta_angular_world[1],ball_delta_angular_world[2]);
    float delta=op_sub(rhs,op_mul(applied_impulse,0.0f));
    const float speed_a=op_add(dot(n,linear_a),dot(torque_a,delta_angular_a));
    const float speed_b=op_add(dot(n_two,linear_b),dot(torque_b,delta_angular_b));
    delta=op_fma(-speed_a,jacobian,delta);
    delta=op_fma(-speed_b,jacobian,delta);
    float sum=op_add(applied_impulse,delta);
    if(sum<lower_limit){delta=op_sub(lower_limit,applied_impulse);sum=lower_limit;}
    else if(sum>upper_limit){delta=op_sub(upper_limit,applied_impulse);sum=upper_limit;}
    applied_impulse=sum;
    linear_a.x=op_fma(op_mul(n.x,0.00555555569f),delta,linear_a.x);
    linear_a.y=op_fma(op_mul(n.y,0.00555555569f),delta,linear_a.y);
    linear_a.z=op_fma(op_mul(n.z,0.00555555569f),delta,linear_a.z);
    delta_angular_a.x=op_fma(angular_a.x,delta,delta_angular_a.x);
    delta_angular_a.y=op_fma(angular_a.y,delta,delta_angular_a.y);
    delta_angular_a.z=op_fma(angular_a.z,delta,delta_angular_a.z);
    linear_b.x=op_fma(op_mul(n_two.x,0.0333333351f),delta,linear_b.x);
    linear_b.y=op_fma(op_mul(n_two.y,0.0333333351f),delta,linear_b.y);
    linear_b.z=op_fma(op_mul(n_two.z,0.0333333351f),delta,linear_b.z);
    delta_angular_b.x=op_fma(angular_b.x,delta,delta_angular_b.x);
    delta_angular_b.y=op_fma(angular_b.y,delta,delta_angular_b.y);
    delta_angular_b.z=op_fma(angular_b.z,delta,delta_angular_b.z);
    car_delta_linear_bt=wp::vec_t<3,wp::float32>(linear_a.x,linear_a.y,linear_a.z);
    car_delta_angular_world=wp::vec_t<3,wp::float32>(delta_angular_a.x,delta_angular_a.y,delta_angular_a.z);
    ball_delta_linear_bt=wp::vec_t<3,wp::float32>(linear_b.x,linear_b.y,linear_b.z);
    ball_delta_angular_world=wp::vec_t<3,wp::float32>(delta_angular_b.x,delta_angular_b.y,delta_angular_b.z);
"""


@wp.func_native(_BULLET_PAIR_SOLVE_VELOCITY_ROW)
def _bullet_pair_solve_velocity_row(
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


_BULLET_PAIR_SOLVE_SPLIT_ROW = (
    _BULLET_PAIR_SOLVE_VELOCITY_ROW
    .replace(
        "// gResolveSingleConstraintRowGeneric_sse4_1_fma3 for two dynamic bodies.",
        "// gResolveSplitPenetrationImpulse_sse2 for two dynamic bodies.\n    if(rhs==0.0f)return;",
    )
    .replace(
        "auto op_fma=[](float a,float b,float c)->float{\n    #if defined(__CUDA_ARCH__)\n        return __fmaf_rn(a,b,c);\n    #else\n        volatile float v=fmaf(a,b,c);return v;\n    #endif\n    };",
        "",
    )
    .replace(
        "auto dot=[&](PairVelV3 a,PairVelV3 b)->float{return op_add(op_add(op_mul(a.x,b.x),op_mul(a.y,b.y)),op_mul(a.z,b.z));};",
        "auto dot=[&](PairVelV3 a,PairVelV3 b)->float{return op_add(op_mul(a.x,b.x),op_add(op_mul(a.y,b.y),op_mul(a.z,b.z)));};",
    )
    .replace(
        "float delta=op_sub(rhs,op_mul(applied_impulse,0.0f));",
        "float delta=op_sub(rhs,op_mul(applied_impulse,0.0f));",
    )
    .replace(
        "delta=op_fma(-speed_a,jacobian,delta);\n    delta=op_fma(-speed_b,jacobian,delta);",
        "delta=op_sub(delta,op_mul(speed_a,jacobian));\n    delta=op_sub(delta,op_mul(speed_b,jacobian));",
    )
    .replace(
        "else if(sum>upper_limit){delta=op_sub(upper_limit,applied_impulse);sum=upper_limit;}",
        "",
    )
    .replace("=op_fma(op_mul(n.x,0.00555555569f),delta,linear_a.x);", "=op_add(linear_a.x,op_mul(op_mul(n.x,0.00555555569f),delta));")
    .replace("=op_fma(op_mul(n.y,0.00555555569f),delta,linear_a.y);", "=op_add(linear_a.y,op_mul(op_mul(n.y,0.00555555569f),delta));")
    .replace("=op_fma(op_mul(n.z,0.00555555569f),delta,linear_a.z);", "=op_add(linear_a.z,op_mul(op_mul(n.z,0.00555555569f),delta));")
    .replace("=op_fma(angular_a.x,delta,delta_angular_a.x);", "=op_add(delta_angular_a.x,op_mul(angular_a.x,delta));")
    .replace("=op_fma(angular_a.y,delta,delta_angular_a.y);", "=op_add(delta_angular_a.y,op_mul(angular_a.y,delta));")
    .replace("=op_fma(angular_a.z,delta,delta_angular_a.z);", "=op_add(delta_angular_a.z,op_mul(angular_a.z,delta));")
    .replace("=op_fma(op_mul(n_two.x,0.0333333351f),delta,linear_b.x);", "=op_add(linear_b.x,op_mul(op_mul(n_two.x,0.0333333351f),delta));")
    .replace("=op_fma(op_mul(n_two.y,0.0333333351f),delta,linear_b.y);", "=op_add(linear_b.y,op_mul(op_mul(n_two.y,0.0333333351f),delta));")
    .replace("=op_fma(op_mul(n_two.z,0.0333333351f),delta,linear_b.z);", "=op_add(linear_b.z,op_mul(op_mul(n_two.z,0.0333333351f),delta));")
    .replace("=op_fma(angular_b.x,delta,delta_angular_b.x);", "=op_add(delta_angular_b.x,op_mul(angular_b.x,delta));")
    .replace("=op_fma(angular_b.y,delta,delta_angular_b.y);", "=op_add(delta_angular_b.y,op_mul(angular_b.y,delta));")
    .replace("=op_fma(angular_b.z,delta,delta_angular_b.z);", "=op_add(delta_angular_b.z,op_mul(angular_b.z,delta));")
)


@wp.func_native(_BULLET_PAIR_SOLVE_SPLIT_ROW)
def _bullet_pair_solve_split_row(
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


_ROCKETSIM_EXTRA_HIT_VELOCITY = r"""
    // Ball::_OnHit in RocketSim MathTypes::Vec operation order. GetState
    // converts each body to UU before subtraction; factoring that conversion
    // across the subtraction changes the cached impulse by observable ULPs.
    auto op_add=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a,b);
    #else
        volatile float v=a+b;return v;
    #endif
    };
    auto op_sub=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a,b);
    #else
        volatile float v=a-b;return v;
    #endif
    };
    auto op_mul=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a,b);
    #else
        volatile float v=a*b;return v;
    #endif
    };
    auto op_div=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a,b);
    #else
        volatile float v=a/b;return v;
    #endif
    };
    struct HitV3{float x;float y;float z;};
    auto make=[](float x,float y,float z)->HitV3{HitV3 v={x,y,z};return v;};
    auto add=[&](HitV3 a,HitV3 b)->HitV3{return make(op_add(a.x,b.x),op_add(a.y,b.y),op_add(a.z,b.z));};
    auto sub=[&](HitV3 a,HitV3 b)->HitV3{return make(op_sub(a.x,b.x),op_sub(a.y,b.y),op_sub(a.z,b.z));};
    auto mul=[&](HitV3 a,HitV3 b)->HitV3{return make(op_mul(a.x,b.x),op_mul(a.y,b.y),op_mul(a.z,b.z));};
    auto scale=[&](HitV3 a,float s)->HitV3{return make(op_mul(a.x,s),op_mul(a.y,s),op_mul(a.z,s));};
    auto divide=[&](HitV3 a,float s)->HitV3{return make(op_div(a.x,s),op_div(a.y,s),op_div(a.z,s));};
    auto dot4=[&](HitV3 a,HitV3 b)->float{return op_add(op_add(op_add(op_mul(a.x,b.x),op_mul(a.y,b.y)),op_mul(a.z,b.z)),0.0f);};
    auto normalized=[&](HitV3 a)->HitV3{
        const float length_squared=dot4(a,a);
        if(length_squared>0.0f){
            const float length=sqrtf(length_squared);
            if(length>1.4210854715202004e-14f)return divide(a,length);
        }
        return make(0.0f,0.0f,0.0f);
    };
    const HitV3 ball_pos_uu=scale(make(ball_position_bt[0],ball_position_bt[1],ball_position_bt[2]),50.0f);
    const HitV3 car_pos_uu=scale(make(car_position_bt[0],car_position_bt[1],car_position_bt[2]),50.0f);
    const HitV3 ball_vel_uu=scale(make(ball_velocity_bt[0],ball_velocity_bt[1],ball_velocity_bt[2]),50.0f);
    const HitV3 car_vel_uu=scale(make(car_velocity_bt[0],car_velocity_bt[1],car_velocity_bt[2]),50.0f);
    const HitV3 relative_position=sub(ball_pos_uu,car_pos_uu);
    const HitV3 relative_velocity=sub(ball_vel_uu,car_vel_uu);
    float relative_speed=sqrtf(dot4(relative_velocity,relative_velocity));
    if(relative_speed>4600.0f)relative_speed=4600.0f;
    HitV3 added=make(0.0f,0.0f,0.0f);
    if(relative_speed>0.0f){
        HitV3 hit_direction=normalized(mul(relative_position,make(1.0f,1.0f,0.35f)));
        const HitV3 forward=make(car_forward[0],car_forward[1],car_forward[2]);
        const float forward_dot=dot4(hit_direction,forward);
        const HitV3 forward_adjustment=scale(scale(forward,forward_dot),op_sub(1.0f,0.65f));
        hit_direction=normalized(sub(hit_direction,forward_adjustment));
        float factor=0.65f;
        if(relative_speed>500.0f && relative_speed<2300.0f){
            const float range_between=op_sub(2300.0f,500.0f);
            const float value_difference=op_sub(0.55f,0.65f);
            const float linear_factor=op_div(op_sub(relative_speed,500.0f),range_between);
            factor=op_add(0.65f,op_mul(value_difference,linear_factor));
        }else if(relative_speed>=2300.0f && relative_speed<4600.0f){
            const float range_between=op_sub(4600.0f,2300.0f);
            const float value_difference=op_sub(0.30f,0.55f);
            const float linear_factor=op_div(op_sub(relative_speed,2300.0f),range_between);
            factor=op_add(0.55f,op_mul(value_difference,linear_factor));
        }else if(relative_speed>=4600.0f){
            factor=0.30f;
        }
        added=scale(scale(hit_direction,relative_speed),factor);
    }
    extra_velocity_uu=wp::vec_t<3,wp::float32>(added.x,added.y,added.z);
    const HitV3 solved=make(solved_ball_velocity_bt[0],solved_ball_velocity_bt[1],solved_ball_velocity_bt[2]);
    const HitV3 cache_impulse=scale(added,0.02f);
    const HitV3 finished=add(solved,cache_impulse);
    solved_ball_velocity_bt=wp::vec_t<3,wp::float32>(finished.x,finished.y,finished.z);
"""


@wp.func_native(_ROCKETSIM_EXTRA_HIT_VELOCITY)
def _rocketsim_extra_hit_velocity(
    ball_position_bt: wp.vec3,
    car_position_bt: wp.vec3,
    ball_velocity_bt: wp.vec3,
    car_velocity_bt: wp.vec3,
    car_forward: wp.vec3,
    extra_velocity_uu: wp.ref[wp.vec3],
    solved_ball_velocity_bt: wp.ref[wp.vec3],
): ...


_BULLET_CAP_AMD = r"""
    // Car::_FinishPhysicsTick / Ball::_FinishPhysicsTick velocity caps use
    // btVector3::normalized: SSE dot, AMD RSQRTSS estimate, one Newton step,
    // component scale, then the separate maximum-speed scale.
    auto op_add=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a,b);
    #else
        volatile float v=a+b;return v;
    #endif
    };
    auto op_sub=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a,b);
    #else
        volatile float v=a-b;return v;
    #endif
    };
    auto op_mul=[](float a,float b)->float{
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a,b);
    #else
        volatile float v=a*b;return v;
    #endif
    };
    const float x=value[0],y=value[1],z=value[2];
    const float length_squared=op_add(op_add(op_mul(x,x),op_mul(y,y)),op_mul(z,z));
    const float maximum_squared=op_mul(maximum,maximum);
    if(length_squared>maximum_squared){
        float inverse_length;
    #if defined(__CUDA_ARCH__)
        const unsigned input_bits=__float_as_uint(length_squared);
        const unsigned exponent=(input_bits>>23)&0xffu;
        const unsigned result_exponent=((380u-exponent)>>1)<<23;
        const unsigned index=(input_bits>>11)&0x1fffu;
        const unsigned estimate_mantissa=static_cast<unsigned>(rsqrtss_mantissa.data[index])<<11;
        inverse_length=__uint_as_float(result_exponent|estimate_mantissa);
    #elif defined(__clang__) && (defined(__x86_64__) || defined(_M_X64))
        using CapM128=float __attribute__((__vector_size__(16)));
        const CapM128 input={length_squared,0.0f,0.0f,0.0f};
        inverse_length=__builtin_ia32_rsqrtss(input)[0];
    #elif defined(_MSC_VER)
        const __m128 input=_mm_set_ss(length_squared);
        inverse_length=_mm_cvtss_f32(_mm_rsqrt_ss(input));
    #else
        inverse_length=1.0f/sqrtf(length_squared);
    #endif
        float correction=op_mul(op_mul(length_squared,0.5f),inverse_length);
        correction=op_mul(correction,inverse_length);
        correction=op_sub(1.5f,correction);
        inverse_length=op_mul(inverse_length,correction);
        value=wp::vec_t<3,wp::float32>(
            op_mul(op_mul(x,inverse_length),maximum),
            op_mul(op_mul(y,inverse_length),maximum),
            op_mul(op_mul(z,inverse_length),maximum));
    }
"""


@wp.func_native(_BULLET_CAP_AMD)
def _bullet_cap_amd(
    value: wp.ref[wp.vec3],
    maximum: float,
    rsqrtss_mantissa: wp.array(dtype=wp.uint16),
): ...


@wp.func
def _extra_hit_factor(speed_uu: float) -> float:
    if speed_uu <= 500.0:
        return 0.65
    if speed_uu < 2300.0:
        fraction = (speed_uu - 500.0) / 1800.0
        return 0.65 + (0.55 - 0.65) * fraction
    if speed_uu < 4600.0:
        fraction = (speed_uu - 2300.0) / 2300.0
        return 0.55 + (0.30 - 0.55) * fraction
    return 0.30


@wp.func
def _copy_pair_contact(
    source: int,
    destination: int,
    local_a: wp.array(dtype=wp.vec3),
    local_b: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    tangent: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=wp.float32),
    lifetime: wp.array(dtype=wp.int32),
    normal_impulse: wp.array(dtype=wp.float32),
    tangent_impulse: wp.array(dtype=wp.float32),
    push_impulse: wp.array(dtype=wp.float32),
):
    local_a[destination] = local_a[source]
    local_b[destination] = local_b[source]
    normal[destination] = normal[source]
    tangent[destination] = tangent[source]
    distance[destination] = distance[source]
    lifetime[destination] = lifetime[source]
    normal_impulse[destination] = normal_impulse[source]
    tangent_impulse[destination] = tangent_impulse[source]
    push_impulse[destination] = push_impulse[source]


_BULLET_CAR_BALL_AABB_OVERLAP = r"""
    // Literal bounded translation of:
    //   btCompoundShape::getAabb
    //   btSphereShape::getAabb (including RocketSim's +0.08f change)
    //   btCollisionWorld::updateSingleAabb
    //   TestAabbAgainstAabb2
    // for the fixed one-child Octane compound and standard sphere. Bullet
    // unions the current and interpolation-transform bounds after expanding
    // each by gContactBreakingThreshold, then tests axes in X/Z/Y order.
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto op_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
        return value;
    #endif
    };
    auto op_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    auto op_abs = [](float value) -> float {
    #if defined(__CUDA_ARCH__)
        return fabsf(value);
    #else
        volatile float result = std::fabs(value);
        return result;
    #endif
    };
    struct AabbV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> AabbV3 {
        AabbV3 value = {x, y, z};
        return value;
    };
    auto compound_bounds = [&](const wp::vec_t<3, wp::float32>& position,
                               const wp::mat_t<3, 3, wp::float32>& basis,
                               AabbV3& minimum,
                               AabbV3& maximum) {
        const float local_center[3] = {0.277513981f, 0.0f, 0.415099978f};
        const float local_half[3] = {1.20372915f, 0.865653098f, 0.385250092f};
        float center[3];
        float extent[3];
        for (int row = 0; row < 3; ++row) {
            const float projected_center = op_add(
                op_add(op_mul(basis.data[row][0], local_center[0]),
                       op_mul(basis.data[row][1], local_center[1])),
                op_mul(basis.data[row][2], local_center[2]));
            center[row] = op_add(projected_center, position[row]);
            extent[row] = op_add(
                op_add(op_mul(local_half[0], op_abs(basis.data[row][0])),
                       op_mul(local_half[1], op_abs(basis.data[row][1]))),
                op_mul(local_half[2], op_abs(basis.data[row][2])));
        }
        const float threshold = 0.02f;
        minimum = make(
            op_sub(op_sub(center[0], extent[0]), threshold),
            op_sub(op_sub(center[1], extent[1]), threshold),
            op_sub(op_sub(center[2], extent[2]), threshold));
        maximum = make(
            op_add(op_add(center[0], extent[0]), threshold),
            op_add(op_add(center[1], extent[1]), threshold),
            op_add(op_add(center[2], extent[2]), threshold));
    };
    auto sphere_bounds = [&](const wp::vec_t<3, wp::float32>& position,
                             AabbV3& minimum,
                             AabbV3& maximum) {
        const float extent = op_add(1.8249999284744263f, 0.08f);
        const float threshold = 0.02f;
        minimum = make(
            op_sub(op_sub(position[0], extent), threshold),
            op_sub(op_sub(position[1], extent), threshold),
            op_sub(op_sub(position[2], extent), threshold));
        maximum = make(
            op_add(op_add(position[0], extent), threshold),
            op_add(op_add(position[1], extent), threshold),
            op_add(op_add(position[2], extent), threshold));
    };
    AabbV3 car_min, car_max, car_predicted_min, car_predicted_max;
    AabbV3 ball_min, ball_max, ball_predicted_min, ball_predicted_max;
    compound_bounds(car_position, car_basis, car_min, car_max);
    compound_bounds(
        predicted_car_position, predicted_car_basis,
        car_predicted_min, car_predicted_max);
    sphere_bounds(ball_position, ball_min, ball_max);
    sphere_bounds(
        predicted_ball_position, ball_predicted_min, ball_predicted_max);
    car_min.x = car_min.x < car_predicted_min.x ? car_min.x : car_predicted_min.x;
    car_min.y = car_min.y < car_predicted_min.y ? car_min.y : car_predicted_min.y;
    car_min.z = car_min.z < car_predicted_min.z ? car_min.z : car_predicted_min.z;
    car_max.x = car_max.x > car_predicted_max.x ? car_max.x : car_predicted_max.x;
    car_max.y = car_max.y > car_predicted_max.y ? car_max.y : car_predicted_max.y;
    car_max.z = car_max.z > car_predicted_max.z ? car_max.z : car_predicted_max.z;
    ball_min.x = ball_min.x < ball_predicted_min.x ? ball_min.x : ball_predicted_min.x;
    ball_min.y = ball_min.y < ball_predicted_min.y ? ball_min.y : ball_predicted_min.y;
    ball_min.z = ball_min.z < ball_predicted_min.z ? ball_min.z : ball_predicted_min.z;
    ball_max.x = ball_max.x > ball_predicted_max.x ? ball_max.x : ball_predicted_max.x;
    ball_max.y = ball_max.y > ball_predicted_max.y ? ball_max.y : ball_predicted_max.y;
    ball_max.z = ball_max.z > ball_predicted_max.z ? ball_max.z : ball_predicted_max.z;
    bool overlap = true;
    overlap = (car_min.x > ball_max.x || car_max.x < ball_min.x) ? false : overlap;
    overlap = (car_min.z > ball_max.z || car_max.z < ball_min.z) ? false : overlap;
    overlap = (car_min.y > ball_max.y || car_max.y < ball_min.y) ? false : overlap;
    return overlap ? 1 : 0;
"""


@wp.func_native(_BULLET_CAR_BALL_AABB_OVERLAP)
def _bullet_car_ball_aabb_overlap(
    car_position: wp.vec3,
    car_basis: wp.mat33,
    predicted_car_position: wp.vec3,
    predicted_car_basis: wp.mat33,
    ball_position: wp.vec3,
    predicted_ball_position: wp.vec3,
) -> int: ...


@wp.func
def _rs_cell_index(aabb_min: wp.vec3) -> int:
    # btRSBroadphase::GetCellIndices casts to int (truncation toward zero)
    # before clamping. Dynamic proxies are assigned solely from AABB minimum.
    i = wp.int32(
        (aabb_min[0] - RS_BROADPHASE_MIN_BT[0])
        / RS_BROADPHASE_CELL_SIZE_BT
    )
    j = wp.int32(
        (aabb_min[1] - RS_BROADPHASE_MIN_BT[1])
        / RS_BROADPHASE_CELL_SIZE_BT
    )
    k = wp.int32(
        (aabb_min[2] - RS_BROADPHASE_MIN_BT[2])
        / RS_BROADPHASE_CELL_SIZE_BT
    )
    i = wp.clamp(i, 0, 30)
    j = wp.clamp(j, 0, 32)
    k = wp.clamp(k, 0, 5)
    return (i * RS_BROADPHASE_CELLS_Y + j) * RS_BROADPHASE_CELLS_Z + k


@wp.func
def _rs_static_overlap_mask(
    aabb_min: wp.vec3,
    aabb_max: wp.vec3,
    static_cell_mask: wp.array(dtype=wp.uint32),
    static_aabb_min: wp.array(dtype=wp.vec3),
    static_aabb_max: wp.array(dtype=wp.vec3),
) -> wp.uint32:
    candidates = static_cell_mask[_rs_cell_index(aabb_min)]
    result = wp.uint32(0)
    for body in range(RS_STATIC_BODY_COUNT):
        bit = wp.uint32(1 << body)
        if (candidates & bit) != wp.uint32(0):
            body_min = static_aabb_min[body]
            body_max = static_aabb_max[body]
            separated = (
                aabb_min[0] > body_max[0]
                or aabb_max[0] < body_min[0]
                or aabb_min[1] > body_max[1]
                or aabb_max[1] < body_min[1]
                or aabb_min[2] > body_max[2]
                or aabb_max[2] < body_min[2]
            )
            if not separated:
                result = result | bit
    return result


@wp.func
def _rs_mask_count(mask: wp.uint32) -> int:
    count = wp.int32(0)
    for body in range(RS_STATIC_BODY_COUNT):
        if (mask & wp.uint32(1 << body)) != wp.uint32(0):
            count = count + 1
    return count


@wp.func
def _rs_nth_mask_body(mask: wp.uint32, ordinal: int) -> int:
    found = wp.int32(-1)
    current = wp.int32(0)
    for body in range(RS_STATIC_BODY_COUNT):
        if found < 0 and (mask & wp.uint32(1 << body)) != wp.uint32(0):
            if current == ordinal:
                found = body
            current = current + 1
    return found


@wp.func
def _rs_sorted_manifold_code(
    sorted_position: int,
    ball_mask: wp.uint32,
    car_mask: wp.uint32,
    ball_manifold_count: int,
    total_manifold_count: int,
    equal_island_permutation: wp.array(dtype=wp.int32),
) -> int:
    """Encode one globally sorted slot: ball body, pair=32, car body=64+."""

    source = equal_island_permutation[
        total_manifold_count * RS_EQUAL_ISLAND_PERMUTATION_WIDTH
        + sorted_position
    ]
    if source < ball_manifold_count:
        return _rs_nth_mask_body(ball_mask, source)
    if source == ball_manifold_count:
        return 32
    return 64 + _rs_nth_mask_body(
        car_mask, source - ball_manifold_count - 1
    )


_BULLET_PAIR_REFRESH_CONTACT = r"""
    // btPersistentManifold::refreshContactPoints for one retained dynamic
    // car/ball point.  Transform evaluation is already source-ordered by
    // _bullet_transform_point; preserve the remaining btVector3 multiply,
    // subtract, dot, and threshold operation boundaries here as well.
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto op_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
        return value;
    #endif
    };
    auto op_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    auto dot = [&](float ax, float ay, float az,
                   float bx, float by, float bz) -> float {
        return op_add(
            op_add(op_mul(ax, bx), op_mul(ay, by)),
            op_mul(az, bz));
    };

    const float dx = op_sub(point_a[0], point_b[0]);
    const float dy = op_sub(point_a[1], point_b[1]);
    const float dz = op_sub(point_a[2], point_b[2]);
    const float signed_distance = dot(
        dx, dy, dz, normal[0], normal[1], normal[2]);
    refreshed_distance = signed_distance;

    if (signed_distance > breaking_threshold) {
        invalid = 1;
        return;
    }

    const float projected_x = op_sub(
        point_a[0], op_mul(normal[0], signed_distance));
    const float projected_y = op_sub(
        point_a[1], op_mul(normal[1], signed_distance));
    const float projected_z = op_sub(
        point_a[2], op_mul(normal[2], signed_distance));
    const float lateral_x = op_sub(point_b[0], projected_x);
    const float lateral_y = op_sub(point_b[1], projected_y);
    const float lateral_z = op_sub(point_b[2], projected_z);
    const float lateral_distance_squared = dot(
        lateral_x, lateral_y, lateral_z,
        lateral_x, lateral_y, lateral_z);
    invalid = lateral_distance_squared
        > op_mul(breaking_threshold, breaking_threshold);
"""


@wp.func_native(_BULLET_PAIR_REFRESH_CONTACT)
def _bullet_pair_refresh_contact(
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal: wp.vec3,
    breaking_threshold: float,
    refreshed_distance: wp.ref[wp.float32],
    invalid: wp.ref[wp.int32],
): ...


@wp.func
def _refresh_pair_manifold(
    car_position: wp.vec3,
    car_basis: wp.mat33,
    ball_position: wp.vec3,
    ball_basis: wp.mat33,
    base: int,
    contacts_in: int,
    local_a: wp.array(dtype=wp.vec3),
    local_b: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    tangent: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=wp.float32),
    lifetime: wp.array(dtype=wp.int32),
    normal_impulse: wp.array(dtype=wp.float32),
    tangent_impulse: wp.array(dtype=wp.float32),
    push_impulse: wp.array(dtype=wp.float32),
) -> int:
    contacts = contacts_in
    # btPersistentManifold refreshes in reverse twice: first all world points
    # and lifetimes, then invalidation/removal with swap-from-last compaction.
    for reverse in range(MAX_CAR_BALL_CONTACTS):
        relative = contacts_in - 1 - reverse
        if relative >= 0:
            index = base + relative
            point_a = _bullet_transform_point(
                car_position, car_basis, local_a[index]
            )
            point_b = _bullet_transform_point(
                ball_position, ball_basis, local_b[index]
            )
            refreshed_distance = wp.float32(0.0)
            invalid = wp.int32(0)
            _bullet_pair_refresh_contact(
                point_a,
                point_b,
                normal[index],
                CONTACT_BREAKING_THRESHOLD_BT,
                refreshed_distance,
                invalid,
            )
            distance[index] = refreshed_distance
            lifetime[index] = lifetime[index] + 1
    for reverse in range(MAX_CAR_BALL_CONTACTS):
        relative = contacts_in - 1 - reverse
        if relative >= 0 and relative < contacts:
            index = base + relative
            point_a = _bullet_transform_point(
                car_position, car_basis, local_a[index]
            )
            point_b = _bullet_transform_point(
                ball_position, ball_basis, local_b[index]
            )
            contact_normal = normal[index]
            refreshed_distance = wp.float32(0.0)
            invalid = wp.int32(0)
            _bullet_pair_refresh_contact(
                point_a,
                point_b,
                contact_normal,
                CONTACT_BREAKING_THRESHOLD_BT,
                refreshed_distance,
                invalid,
            )
            if invalid != 0:
                last = base + contacts - 1
                if index != last:
                    _copy_pair_contact(
                        last,
                        index,
                        local_a,
                        local_b,
                        normal,
                        tangent,
                        distance,
                        lifetime,
                        normal_impulse,
                        tangent_impulse,
                        push_impulse,
                    )
                contacts = contacts - 1
    return contacts


@wp.kernel(enable_backward=False)
def capture_car_ball_inputs(
    pair_car_local: int,
    car_position_bt: wp.array(dtype=wp.vec3),
    car_velocity_bt: wp.array(dtype=wp.vec3),
    car_quaternion: wp.array(dtype=wp.quat),
    car_angular_velocity: wp.array(dtype=wp.vec3),
    ball_position_bt: wp.array(dtype=wp.vec3),
    ball_velocity_bt: wp.array(dtype=wp.vec3),
    ball_quaternion: wp.array(dtype=wp.quat),
    ball_angular_velocity: wp.array(dtype=wp.vec3),
    pre_car_position_bt: wp.array(dtype=wp.vec3),
    pre_car_velocity_bt: wp.array(dtype=wp.vec3),
    pre_car_quaternion: wp.array(dtype=wp.quat),
    pre_car_angular_velocity: wp.array(dtype=wp.vec3),
    pre_ball_position_bt: wp.array(dtype=wp.vec3),
    pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    pre_ball_quaternion: wp.array(dtype=wp.quat),
    pre_ball_angular_velocity: wp.array(dtype=wp.vec3),
):
    env = wp.tid()
    car = env * 2 + pair_car_local
    pre_car_position_bt[env] = car_position_bt[car]
    pre_car_velocity_bt[env] = car_velocity_bt[car]
    pre_car_quaternion[env] = car_quaternion[car]
    pre_car_angular_velocity[env] = car_angular_velocity[car]
    pre_ball_position_bt[env] = ball_position_bt[env]
    pre_ball_velocity_bt[env] = ball_velocity_bt[env]
    pre_ball_quaternion[env] = ball_quaternion[env]
    pre_ball_angular_velocity[env] = ball_angular_velocity[env]


@wp.kernel(
    enable_backward=False,
    module="unique",
    module_options={"max_unroll": 4},
)
def car_ball_tick(
    tick_counter: wp.array(dtype=wp.int32),
    pair_car_local: int,
    amd_rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    rs_static_cell_mask: wp.array(dtype=wp.uint32),
    rs_static_aabb_min_bt: wp.array(dtype=wp.vec3),
    rs_static_aabb_max_bt: wp.array(dtype=wp.vec3),
    rs_equal_island_permutation: wp.array(dtype=wp.int32),
    total_car_force_bt: wp.array(dtype=wp.vec3),
    total_car_torque_bt: wp.array(dtype=wp.vec3),
    car_pos: wp.array(dtype=wp.vec3),
    car_vel: wp.array(dtype=wp.vec3),
    car_quat: wp.array(dtype=wp.quat),
    car_ang_vel: wp.array(dtype=wp.vec3),
    car_rigid_position_bt: wp.array(dtype=wp.vec3),
    car_rigid_velocity_bt: wp.array(dtype=wp.vec3),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_quat: wp.array(dtype=wp.quat),
    ball_ang_vel: wp.array(dtype=wp.vec3),
    ball_resident_position_bt: wp.array(dtype=wp.vec3),
    ball_resident_velocity_bt: wp.array(dtype=wp.vec3),
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
    pre_ball_position_bt: wp.array(dtype=wp.vec3),
    pre_ball_velocity_bt: wp.array(dtype=wp.vec3),
    pre_ball_quaternion: wp.array(dtype=wp.quat),
    pre_ball_angular_velocity: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
    hit_this_tick: wp.array(dtype=wp.int32),
    algorithm_active: wp.array(dtype=wp.int32),
    contact_point_a_bt: wp.array(dtype=wp.vec3),
    contact_point_b_bt: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_tangent: wp.array(dtype=wp.vec3),
    contact_distance_bt: wp.array(dtype=wp.float32),
    normal_impulse: wp.array(dtype=wp.float32),
    tangent_impulse: wp.array(dtype=wp.float32),
    push_impulse: wp.array(dtype=wp.float32),
    extra_hit_velocity_uu: wp.array(dtype=wp.vec3),
    relative_pos_on_ball_uu: wp.array(dtype=wp.vec3),
    last_extra_impulse_tick: wp.array(dtype=wp.int32),
    manifold_local_a_bt: wp.array(dtype=wp.vec3),
    manifold_local_b_bt: wp.array(dtype=wp.vec3),
    manifold_normal: wp.array(dtype=wp.vec3),
    manifold_tangent: wp.array(dtype=wp.vec3),
    manifold_distance_bt: wp.array(dtype=wp.float32),
    manifold_lifetime: wp.array(dtype=wp.int32),
    manifold_normal_jacobian: wp.array(dtype=wp.float32),
    manifold_tangent_jacobian: wp.array(dtype=wp.float32),
    manifold_normal_rhs: wp.array(dtype=wp.float32),
    manifold_tangent_rhs: wp.array(dtype=wp.float32),
    manifold_push_rhs: wp.array(dtype=wp.float32),
    manifold_normal_impulse: wp.array(dtype=wp.float32),
    manifold_tangent_impulse: wp.array(dtype=wp.float32),
    manifold_push_impulse: wp.array(dtype=wp.float32),
):
    env = wp.tid()
    car = env * 2 + pair_car_local
    car_position = pre_car_position_bt[env]
    car_quaternion = pre_car_quaternion[env]
    car_basis = _bullet_quaternion_matrix(car_quaternion)
    if tick_counter[0] == 1:
        car_basis = _authority_input_quaternion_matrix(car_quaternion)
    ball_position = pre_ball_position_bt[env]
    ball_quaternion = pre_ball_quaternion[env]
    ball_basis = _bullet_quaternion_matrix(ball_quaternion)
    car_pre_velocity = pre_car_velocity_bt[env]
    car_pre_angular = pre_car_angular_velocity[env]
    ball_source_velocity = pre_ball_velocity_bt[env]
    ball_pre_velocity = ball_source_velocity * wp.pow(1.0 - BALL_DAMPING, DT)
    ball_pre_angular = pre_ball_angular_velocity[env]
    manifold_base = env * MAX_CAR_BALL_CONTACTS
    # btRSBroadphase::calculateOverlappingPairs removes every pair attached to
    # an active dynamic handle before rebuilding the current cell overlaps.
    # The car/ball broadphase pair therefore receives a fresh compound child
    # algorithm and persistent manifold on every tick; the prior tick's child
    # manifold is visible to RocketSim's pre-Bullet hook, but is destroyed
    # before dispatch. Static-body pairs follow a different lifetime path.
    contacts = wp.int32(0)
    contact_count[env] = 0
    hit_this_tick[env] = 0
    extra_hit_velocity_uu[env] = wp.vec3(0.0, 0.0, 0.0)
    normal_impulse[env] = 0.0
    tangent_impulse[env] = 0.0
    push_impulse[env] = 0.0

    # predictUnconstraintMotion applies damping, writes the interpolation
    # transform, and only then does updateSingleAabb build the continuous
    # current-plus-interpolation proxy.  The pair algorithm exists this tick
    # only when btRSBroadphase rebuilds an overlap from those proxy bounds.
    predicted_car_position = car_position
    _bullet_integrate_position(
        car_position,
        wp.vec3(0.0, 0.0, 0.0),
        car_pre_velocity,
        DT,
        0,
        predicted_car_position,
    )
    predicted_car_quaternion = car_quaternion
    _bullet_integrate_quaternion(
        car_basis,
        car_pre_angular,
        DT,
        predicted_car_quaternion,
    )
    predicted_car_basis = _bullet_quaternion_matrix(predicted_car_quaternion)
    predicted_ball_position = ball_position
    _bullet_integrate_position(
        ball_position,
        wp.vec3(0.0, 0.0, 0.0),
        ball_pre_velocity,
        DT,
        0,
        predicted_ball_position,
    )
    pair_active = _bullet_car_ball_aabb_overlap(
        car_position,
        car_basis,
        predicted_car_position,
        predicted_car_basis,
        ball_position,
        predicted_ball_position,
    )
    algorithm_active[env] = pair_active
    if pair_active == 0:
        # car/world and ball/world have already executed as independent
        # islands.  Leaving their outputs in place is the exact no-pair path.
        return

    # The outer dynamic broadphase selected a newly-created pair. The frozen
    # authority includes shallow witnesses just beyond the child's unpredicted
    # AABB because the moving proxy is continuous; do not re-cull narrowphase
    # using the source transform here.

    point_a = wp.vec3(0.0, 0.0, 0.0)
    point_b = wp.vec3(0.0, 0.0, 0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    distance = wp.float32(0.0)
    valid = wp.int32(0)
    degenerate = wp.int32(0)
    bullet_box_sphere_closest(
        car_position,
        car_basis,
        ball_position,
        point_a,
        point_b,
        normal,
        distance,
        valid,
        degenerate,
    )
    catch_degenerate = degenerate != 0 and distance + BOX_MARGIN_BT + BALL_RADIUS_BT < 0.01
    if valid == 0 or catch_degenerate:
        epa_point_a = wp.vec3(0.0, 0.0, 0.0)
        epa_point_b = wp.vec3(0.0, 0.0, 0.0)
        epa_normal = wp.vec3(0.0, 0.0, 0.0)
        epa_distance = wp.float32(0.0)
        epa_valid = wp.int32(0)
        bullet_box_sphere_penetration(
            car_position,
            car_basis,
            ball_position,
            epa_point_a,
            epa_point_b,
            epa_normal,
            epa_distance,
            epa_valid,
        )
        if epa_valid != 0 and (valid == 0 or epa_distance < distance):
            point_a = epa_point_a
            point_b = epa_point_b
            normal = epa_normal
            distance = epa_distance
            valid = 1

    reports_contact = valid != 0 and (
        distance < 0.0
        or distance * distance
        < (BOX_MARGIN_BT + BALL_RADIUS_BT + CONTACT_BREAKING_THRESHOLD_BT)
        * (BOX_MARGIN_BT + BALL_RADIUS_BT + CONTACT_BREAKING_THRESHOLD_BT)
    )
    # pair distance already includes both margins; the callback threshold is
    # the manifold breaking threshold, not the sum used by the raw GJK core.
    if distance > CONTACT_BREAKING_THRESHOLD_BT:
        reports_contact = False
    if reports_contact:
        hit_this_tick[env] = 1
        contact_point_a_bt[env] = point_a
        contact_point_b_bt[env] = point_b
        contact_normal[env] = normal
        contact_distance_bt[env] = distance
        candidate_local_a = _bullet_inverse_transform_point(
            car_position, car_basis, point_a
        )
        candidate_local_b = _bullet_inverse_transform_point(
            ball_position, ball_basis, point_b
        )
        relative_pos_on_ball_uu[env] = candidate_local_b * 50.0
        destination = contacts
        if contacts == MAX_CAR_BALL_CONTACTS:
            destination = bullet_manifold_replacement(
                candidate_local_a,
                manifold_local_a_bt[manifold_base],
                manifold_local_a_bt[manifold_base + 1],
                manifold_local_a_bt[manifold_base + 2],
                manifold_local_a_bt[manifold_base + 3],
                distance,
                manifold_distance_bt[manifold_base],
                manifold_distance_bt[manifold_base + 1],
                manifold_distance_bt[manifold_base + 2],
                manifold_distance_bt[manifold_base + 3],
            )
        else:
            contacts = contacts + 1
        destination = manifold_base + destination
        manifold_local_a_bt[destination] = candidate_local_a
        manifold_local_b_bt[destination] = candidate_local_b
        manifold_normal[destination] = normal
        manifold_tangent[destination] = wp.vec3(0.0, 0.0, 0.0)
        manifold_distance_bt[destination] = distance
        manifold_lifetime[destination] = 0
        manifold_normal_impulse[destination] = 0.0
        manifold_tangent_impulse[destination] = 0.0
        manifold_push_impulse[destination] = 0.0

    if contacts > 0:
        # btConvexConvexAlgorithm owns the child manifold and refreshes it
        # after the new GJK/EPA witness has been appended.
        contacts = _refresh_pair_manifold(
            car_position,
            car_basis,
            ball_position,
            ball_basis,
            manifold_base,
            contacts,
            manifold_local_a_bt,
            manifold_local_b_bt,
            manifold_normal,
            manifold_tangent,
            manifold_distance_bt,
            manifold_lifetime,
            manifold_normal_impulse,
            manifold_tangent_impulse,
            manifold_push_impulse,
        )
    contact_count[env] = contacts

    # btSimulationIslandManager merges the two dynamic bodies while their
    # broadphase pair/algorithm is active, even when the persistent manifold
    # currently contains no contact points.  That zero-contact manifold still
    # participates in the island-manifold quicksort and therefore changes the
    # ordering of the car/static rows.  Continue through the unified island
    # solver here; returning on an empty pair manifold incorrectly leaves the
    # independently solved car/world result in place.

    car_external_linear = wp.vec3(0.0, 0.0, 0.0)
    car_external_angular = wp.vec3(0.0, 0.0, 0.0)
    _bullet_integrate_external_velocities(
        car_basis,
        total_car_force_bt[car] + wp.vec3(0.0, 0.0, -2340.0),
        total_car_torque_bt[car],
        car_external_linear,
        car_external_angular,
    )
    car_force_velocity = car_pre_velocity + car_external_linear
    car_force_angular = car_pre_angular + car_external_angular

    # Arena::Step marks an exactly motionless ball sleeping before Bullet's
    # applyGravity pass. A contact can wake it later in this tick, but that
    # does not retroactively add the skipped external-force impulse.
    ball_external_linear = wp.vec3(0.0, 0.0, 0.0)
    if wp.dot(ball_source_velocity, ball_source_velocity) != 0.0 or wp.dot(
        ball_pre_angular, ball_pre_angular
    ) != 0.0:
        ball_external_linear = wp.vec3(0.0, 0.0, -13.0 * DT)
    ball_force_velocity = ball_pre_velocity + ball_external_linear
    ball_force_angular = ball_pre_angular

    car_delta_velocity = wp.vec3(0.0, 0.0, 0.0)
    car_delta_angular = wp.vec3(0.0, 0.0, 0.0)
    ball_delta_velocity = wp.vec3(0.0, 0.0, 0.0)
    ball_delta_angular = wp.vec3(0.0, 0.0, 0.0)
    car_push = wp.vec3(0.0, 0.0, 0.0)
    car_turn = wp.vec3(0.0, 0.0, 0.0)
    ball_push = wp.vec3(0.0, 0.0, 0.0)
    ball_turn = wp.vec3(0.0, 0.0, 0.0)
    car_static_contacts = car_static_contact_count[car]
    ball_static_contacts = ball_static_contact_count[env]

    # calculateOverlappingPairs walks the ball proxy first, then the car
    # proxy. Each visits source static handles in creation order, and the
    # car/ball pair is first created while walking the ball's dynamic cell.
    # Reconstruct every dispatcher manifold slot, including zero-contact
    # slots, before applying the island's all-equal quicksort permutation.
    # btSphereShape::getAabb uses radius + RocketSim's 0.08 addition, then
    # updateSingleAabb adds gContactBreakingThreshold (0.02).
    ball_proxy_half = wp.vec3(1.925, 1.925, 1.925)
    ball_proxy_min = ball_position - ball_proxy_half
    ball_proxy_max = ball_position + ball_proxy_half
    ball_static_mask = _rs_static_overlap_mask(
        ball_proxy_min,
        ball_proxy_max,
        rs_static_cell_mask,
        rs_static_aabb_min_bt,
        rs_static_aabb_max_bt,
    )
    if wp.dot(ball_source_velocity, ball_source_velocity) == 0.0 and wp.dot(
        ball_pre_angular, ball_pre_angular
    ) == 0.0:
        # needsCollision rejects sleeping-ball/static pairs before an
        # algorithm or persistent manifold is allocated. The active car pair
        # can wake the ball later without recreating those earlier slots.
        ball_static_mask = wp.uint32(0)

    child_center = car_position + _matrix_vector(car_basis, CHILD_OFFSET_BT)
    box_half = BOX_HALF_WITH_MARGIN_BT
    car_extent = wp.vec3(
        wp.abs(car_basis[0, 0]) * box_half[0]
        + wp.abs(car_basis[0, 1]) * box_half[1]
        + wp.abs(car_basis[0, 2]) * box_half[2],
        wp.abs(car_basis[1, 0]) * box_half[0]
        + wp.abs(car_basis[1, 1]) * box_half[1]
        + wp.abs(car_basis[1, 2]) * box_half[2],
        wp.abs(car_basis[2, 0]) * box_half[0]
        + wp.abs(car_basis[2, 1]) * box_half[1]
        + wp.abs(car_basis[2, 2]) * box_half[2],
    ) + wp.vec3(
        RS_CAR_PROXY_AABB_EXPANSION_BT,
        RS_CAR_PROXY_AABB_EXPANSION_BT,
        RS_CAR_PROXY_AABB_EXPANSION_BT,
    )
    car_proxy_min = child_center - car_extent
    car_proxy_max = child_center + car_extent
    car_static_mask = _rs_static_overlap_mask(
        car_proxy_min,
        car_proxy_max,
        rs_static_cell_mask,
        rs_static_aabb_min_bt,
        rs_static_aabb_max_bt,
    )
    ball_static_manifolds = _rs_mask_count(ball_static_mask)
    car_static_manifolds = _rs_mask_count(car_static_mask)
    island_manifolds = ball_static_manifolds + 1 + car_static_manifolds
    for relative in range(MAX_CONTACTS_PER_CAR):
        if relative < car_static_contacts:
            index = car * MAX_CONTACTS_PER_CAR + relative
            car_static_normal_impulse[index] = 0.0
            car_static_tangent_impulse[index] = 0.0
            car_static_push_impulse[index] = 0.0
    for relative in range(MAX_BALL_CONTACTS):
        if relative < ball_static_contacts:
            index = env * MAX_BALL_CONTACTS + relative
            ball_static_normal_impulse[index] = 0.0
            ball_static_tangent_impulse[index] = 0.0
            ball_static_push_impulse[index] = 0.0

    # convertContactSpecial runs after all ordinary manifold conversion.  The
    # ordinary ball/world rows remain in the split-impulse stream with
    # m_isSpecial set, while the appended aggregate row is the only ball/world
    # normal/friction pair resolved by the non-interleaved velocity solver.
    ball_special_normal_sum = wp.vec3(0.0, 0.0, 0.0)
    ball_special_distance_sum = wp.float32(0.0)
    for sorted_slot in range(RS_MAX_ISLAND_MANIFOLDS):
        if sorted_slot < island_manifolds:
            manifold_code = _rs_sorted_manifold_code(
                sorted_slot,
                ball_static_mask,
                car_static_mask,
                ball_static_manifolds,
                island_manifolds,
                rs_equal_island_permutation,
            )
            if manifold_code < 32:
                for relative in range(MAX_BALL_CONTACTS):
                    if relative < ball_static_contacts:
                        index = env * MAX_BALL_CONTACTS + relative
                        if ball_static_mesh[index] == manifold_code:
                            point_a = _bullet_transform_point(
                                ball_position,
                                ball_basis,
                                ball_static_local_a_bt[index],
                            )
                            rel = point_a - ball_position
                            ball_special_normal_sum = (
                                ball_special_normal_sum
                                + ball_static_normal[index]
                            )
                            _bullet_special_distance_accumulate(
                                rel, ball_special_distance_sum
                            )
    ball_special_normal = wp.vec3(0.0, 0.0, 0.0)
    ball_special_rel = wp.vec3(0.0, 0.0, 0.0)
    ball_special_tangent = wp.vec3(0.0, 0.0, 0.0)
    ball_special_normal_jacobian = wp.float32(0.0)
    ball_special_tangent_jacobian = wp.float32(0.0)
    ball_special_normal_rhs = wp.float32(0.0)
    ball_special_tangent_rhs = wp.float32(0.0)
    ball_special_normal_impulse = wp.float32(0.0)
    ball_special_tangent_impulse = wp.float32(0.0)
    if ball_static_contacts > 0:
        ball_special_distance = wp.float32(0.0)
        _bullet_special_average(
            ball_special_normal_sum,
            ball_special_distance_sum,
            ball_static_contacts,
            ball_special_normal,
            ball_special_distance,
        )
        ball_special_rel = ball_special_normal * -ball_special_distance
        ball_special_push_rhs = wp.float32(0.0)
        _bullet_ball_special_contact_row(
            wp.vec3(0.0, 0.0, 0.0),
            ball_basis,
            ball_special_rel,
            wp.vec3(0.0, 0.0, 0.0),
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

    # convertContact walks manifold points in cache order. setupContactConstraint
    # applies each retained normal warm-start immediately; friction warm-start
    # is explicitly zeroed by the pinned RocketSim solver.
    for relative in range(MAX_CAR_BALL_CONTACTS):
        if relative < contacts:
            index = manifold_base + relative
            normal = manifold_normal[index]
            point_a = _bullet_transform_point(
                car_position, car_basis, manifold_local_a_bt[index]
            )
            point_b = _bullet_transform_point(
                ball_position, ball_basis, manifold_local_b_bt[index]
            )
            relative_car = point_a - car_position
            relative_ball = point_b - ball_position
            tangent = wp.vec3(0.0, 0.0, 0.0)
            normal_jacobian = wp.float32(0.0)
            tangent_jacobian = wp.float32(0.0)
            normal_rhs_value = wp.float32(0.0)
            tangent_rhs_value = wp.float32(0.0)
            push_rhs_value = wp.float32(0.0)
            _bullet_pair_contact_row(
                car_basis,
                ball_basis,
                relative_car,
                relative_ball,
                manifold_distance_bt[index],
                DT,
                normal,
                car_pre_velocity,
                car_pre_angular,
                car_force_velocity,
                car_force_angular,
                ball_pre_velocity,
                ball_pre_angular,
                ball_force_velocity,
                ball_force_angular,
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
            manifold_push_impulse[index] = 0.0
            applied = manifold_normal_impulse[index] * WARMSTART_FACTOR
            manifold_normal_impulse[index] = applied
            manifold_tangent_impulse[index] = 0.0
            if applied != 0.0:
                impulse = normal * applied
                _bullet_apply_impulse(
                    car_basis,
                    impulse,
                    relative_car,
                    car_delta_velocity,
                    car_delta_angular,
                )
                ball_impulse = -impulse
                ball_delta_velocity = (
                    ball_delta_velocity + ball_impulse * BALL_INV_MASS
                )
                ball_delta_angular = ball_delta_angular + (
                    wp.cross(relative_ball, ball_impulse) * BALL_INV_INERTIA
                )

    # solveGroupCacheFriendlySplitImpulseIterations walks every normal row in
    # globally sorted manifold order on each iteration. Zero-contact manifold
    # slots change the quicksort permutation even though they emit no rows.
    for _iteration in range(SOLVER_ITERATIONS):
        for sorted_slot in range(RS_MAX_ISLAND_MANIFOLDS):
            if sorted_slot < island_manifolds:
                manifold_code = _rs_sorted_manifold_code(
                    sorted_slot,
                    ball_static_mask,
                    car_static_mask,
                    ball_static_manifolds,
                    island_manifolds,
                    rs_equal_island_permutation,
                )
                if manifold_code >= 64:
                    static_body = manifold_code - 64
                    for relative in range(MAX_CONTACTS_PER_CAR):
                        if relative < car_static_contacts:
                            index = car * MAX_CONTACTS_PER_CAR + relative
                            if car_static_mesh[index] == static_body:
                                point_a = _bullet_transform_point(
                                    car_position,
                                    car_basis,
                                    car_static_local_a_bt[index],
                                )
                                applied_push = wp.float32(
                                    car_static_push_impulse[index]
                                )
                                _bullet_solve_split_row(
                                    car_basis,
                                    car_static_normal[index],
                                    point_a - car_position,
                                    car_static_normal_jacobian[index],
                                    car_static_push_rhs[index],
                                    car_push,
                                    car_turn,
                                    applied_push,
                                )
                                car_static_push_impulse[index] = applied_push
                elif manifold_code == 32:
                    for relative in range(MAX_CAR_BALL_CONTACTS):
                        if relative < contacts:
                            index = manifold_base + relative
                            normal = manifold_normal[index]
                            point_a = _bullet_transform_point(
                                car_position,
                                car_basis,
                                manifold_local_a_bt[index],
                            )
                            point_b = _bullet_transform_point(
                                ball_position,
                                ball_basis,
                                manifold_local_b_bt[index],
                            )
                            relative_car = point_a - car_position
                            relative_ball = point_b - ball_position
                            applied_push = wp.float32(
                                manifold_push_impulse[index]
                            )
                            _bullet_pair_solve_split_row(
                                car_basis,
                                ball_basis,
                                normal,
                                relative_car,
                                relative_ball,
                                manifold_normal_jacobian[index],
                                manifold_push_rhs[index],
                                0.0,
                                1.0e10,
                                car_push,
                                car_turn,
                                ball_push,
                                ball_turn,
                                applied_push,
                            )
                            manifold_push_impulse[index] = applied_push
                else:
                    static_body = manifold_code
                    for relative in range(MAX_BALL_CONTACTS):
                        if relative < ball_static_contacts:
                            index = env * MAX_BALL_CONTACTS + relative
                            if ball_static_mesh[index] == static_body:
                                point_a = _bullet_transform_point(
                                    ball_position,
                                    ball_basis,
                                    ball_static_local_a_bt[index],
                                )
                                applied_push = wp.float32(
                                    ball_static_push_impulse[index]
                                )
                                _bullet_ball_solve_split_row(
                                    ball_basis,
                                    ball_static_normal[index],
                                    point_a - ball_position,
                                    ball_static_normal_jacobian[index],
                                    ball_static_push_rhs[index],
                                    ball_push,
                                    ball_turn,
                                    applied_push,
                                )
                                ball_static_push_impulse[index] = applied_push

    # The default non-interleaved solver resolves all contact normals, then
    # all friction rows, in cache order on every iteration.
    for _iteration in range(SOLVER_ITERATIONS):
        for sorted_slot in range(RS_MAX_ISLAND_MANIFOLDS):
            if sorted_slot < island_manifolds:
                manifold_code = _rs_sorted_manifold_code(
                    sorted_slot,
                    ball_static_mask,
                    car_static_mask,
                    ball_static_manifolds,
                    island_manifolds,
                    rs_equal_island_permutation,
                )
                if manifold_code >= 64:
                    static_body = manifold_code - 64
                    for relative in range(MAX_CONTACTS_PER_CAR):
                        if relative < car_static_contacts:
                            index = car * MAX_CONTACTS_PER_CAR + relative
                            if car_static_mesh[index] == static_body:
                                point_a = _bullet_transform_point(
                                    car_position,
                                    car_basis,
                                    car_static_local_a_bt[index],
                                )
                                applied_normal = wp.float32(
                                    car_static_normal_impulse[index]
                                )
                                _bullet_solve_velocity_row(
                                    car_basis,
                                    car_static_normal[index],
                                    point_a - car_position,
                                    car_static_normal_jacobian[index],
                                    car_static_normal_rhs[index],
                                    0.0,
                                    1.0e10,
                                    car_delta_velocity,
                                    car_delta_angular,
                                    applied_normal,
                                )
                                car_static_normal_impulse[index] = (
                                    applied_normal
                                )
                elif manifold_code == 32:
                    for relative in range(MAX_CAR_BALL_CONTACTS):
                        if relative < contacts:
                            index = manifold_base + relative
                            normal = manifold_normal[index]
                            point_a = _bullet_transform_point(
                                car_position,
                                car_basis,
                                manifold_local_a_bt[index],
                            )
                            point_b = _bullet_transform_point(
                                ball_position,
                                ball_basis,
                                manifold_local_b_bt[index],
                            )
                            relative_car = point_a - car_position
                            relative_ball = point_b - ball_position
                            applied_normal = wp.float32(
                                manifold_normal_impulse[index]
                            )
                            _bullet_pair_solve_velocity_row(
                                car_basis,
                                ball_basis,
                                normal,
                                relative_car,
                                relative_ball,
                                manifold_normal_jacobian[index],
                                manifold_normal_rhs[index],
                                0.0,
                                1.0e10,
                                car_delta_velocity,
                                car_delta_angular,
                                ball_delta_velocity,
                                ball_delta_angular,
                                applied_normal,
                            )
                            manifold_normal_impulse[index] = (
                                applied_normal
                            )
        if ball_static_contacts > 0:
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
        for sorted_slot in range(RS_MAX_ISLAND_MANIFOLDS):
            if sorted_slot < island_manifolds:
                manifold_code = _rs_sorted_manifold_code(
                    sorted_slot,
                    ball_static_mask,
                    car_static_mask,
                    ball_static_manifolds,
                    island_manifolds,
                    rs_equal_island_permutation,
                )
                if manifold_code >= 64:
                    static_body = manifold_code - 64
                    for relative in range(MAX_CONTACTS_PER_CAR):
                        if relative < car_static_contacts:
                            index = car * MAX_CONTACTS_PER_CAR + relative
                            if car_static_mesh[index] == static_body:
                                applied_normal = car_static_normal_impulse[
                                    index
                                ]
                                if applied_normal > 0.0:
                                    point_a = _bullet_transform_point(
                                        car_position,
                                        car_basis,
                                        car_static_local_a_bt[index],
                                    )
                                    applied_tangent = wp.float32(
                                        car_static_tangent_impulse[index]
                                    )
                                    limit = (
                                        CAR_STATIC_FRICTION
                                        * applied_normal
                                    )
                                    _bullet_solve_velocity_row(
                                        car_basis,
                                        car_static_tangent[index],
                                        point_a - car_position,
                                        car_static_tangent_jacobian[index],
                                        car_static_tangent_rhs[index],
                                        -limit,
                                        limit,
                                        car_delta_velocity,
                                        car_delta_angular,
                                        applied_tangent,
                                    )
                                    car_static_tangent_impulse[index] = (
                                        applied_tangent
                                    )
                elif manifold_code == 32:
                    for relative in range(MAX_CAR_BALL_CONTACTS):
                        if relative < contacts:
                            index = manifold_base + relative
                            applied_normal = manifold_normal_impulse[index]
                            if applied_normal > 0.0:
                                normal = manifold_normal[index]
                                tangent = manifold_tangent[index]
                                point_a = _bullet_transform_point(
                                    car_position,
                                    car_basis,
                                    manifold_local_a_bt[index],
                                )
                                point_b = _bullet_transform_point(
                                    ball_position,
                                    ball_basis,
                                    manifold_local_b_bt[index],
                                )
                                relative_car = point_a - car_position
                                relative_ball = point_b - ball_position
                                applied_tangent = wp.float32(
                                    manifold_tangent_impulse[index]
                                )
                                limit = CONTACT_FRICTION * applied_normal
                                _bullet_pair_solve_velocity_row(
                                    car_basis,
                                    ball_basis,
                                    tangent,
                                    relative_car,
                                    relative_ball,
                                    manifold_tangent_jacobian[index],
                                    manifold_tangent_rhs[index],
                                    -limit,
                                    limit,
                                    car_delta_velocity,
                                    car_delta_angular,
                                    ball_delta_velocity,
                                    ball_delta_angular,
                                    applied_tangent,
                                )
                                manifold_tangent_impulse[index] = (
                                    applied_tangent
                                )
        if ball_static_contacts > 0 and ball_special_normal_impulse > 0.0:
            ball_limit = BALL_STATIC_FRICTION * ball_special_normal_impulse
            _bullet_ball_solve_velocity_row(
                ball_basis,
                ball_special_tangent,
                ball_special_rel,
                ball_special_tangent_jacobian,
                ball_special_tangent_rhs,
                -ball_limit,
                ball_limit,
                ball_delta_velocity,
                ball_delta_angular,
                ball_special_tangent_impulse,
            )

    total_normal_impulse = wp.float32(0.0)
    total_tangent_impulse = wp.float32(0.0)
    total_push_impulse = wp.float32(0.0)
    for relative in range(MAX_CAR_BALL_CONTACTS):
        if relative < contacts:
            index = manifold_base + relative
            total_normal_impulse = (
                total_normal_impulse + manifold_normal_impulse[index]
            )
            total_tangent_impulse = (
                total_tangent_impulse + manifold_tangent_impulse[index]
            )
            total_push_impulse = (
                total_push_impulse + manifold_push_impulse[index]
            )
    normal_impulse[env] = total_normal_impulse
    tangent_impulse[env] = total_tangent_impulse
    push_impulse[env] = total_push_impulse
    car_solved_velocity = (car_pre_velocity + car_delta_velocity) + car_external_linear
    car_solved_angular = (car_pre_angular + car_delta_angular) + car_external_angular
    ball_solved_velocity = (
        ball_pre_velocity + ball_delta_velocity
    ) + ball_external_linear
    ball_solved_angular = ball_pre_angular + ball_delta_angular

    car_has_split = wp.int32(
        wp.dot(car_push, car_push) > 0.0 or wp.dot(car_turn, car_turn) > 0.0
    )
    ball_has_split = wp.int32(
        wp.dot(ball_push, ball_push) > 0.0 or wp.dot(ball_turn, ball_turn) > 0.0
    )
    car_split_quat = car_quaternion
    car_integration_basis = car_basis
    if car_has_split != 0:
        _bullet_integrate_quaternion(
            car_basis, car_turn * CONTACT_SPLIT_TURN_ERP, DT, car_split_quat
        )
        # writebackVelocityAndTransform stores the split-corrected quaternion
        # as a basis before predictIntegratedTransform reads it again.
        car_integration_basis = _bullet_quaternion_matrix(car_split_quat)
    ball_split_quat = ball_quaternion
    ball_integration_basis = ball_basis
    if ball_has_split != 0:
        _bullet_integrate_quaternion(
            ball_basis, ball_turn * CONTACT_SPLIT_TURN_ERP, DT, ball_split_quat
        )
        ball_integration_basis = _bullet_quaternion_matrix(ball_split_quat)
    car_integrated_position = car_position
    _bullet_integrate_position(
        car_position,
        car_push,
        car_solved_velocity,
        DT,
        car_has_split,
        car_integrated_position,
    )
    ball_integrated_position = ball_position
    _bullet_integrate_position(
        ball_position,
        ball_push,
        ball_solved_velocity,
        DT,
        ball_has_split,
        ball_integrated_position,
    )
    car_integrated_quat = car_split_quat
    _bullet_integrate_quaternion(
        car_integration_basis,
        car_solved_angular,
        DT,
        car_integrated_quat,
    )
    ball_integrated_quat = ball_split_quat
    _bullet_integrate_quaternion(
        ball_integration_basis,
        ball_solved_angular,
        DT,
        ball_integrated_quat,
    )

    # Arena callback queues this source-defined velocity for Ball::_Finish,
    # after Bullet has integrated the transform but before the final cap.
    extra_velocity = wp.vec3(0.0, 0.0, 0.0)
    callback_tick = tick_counter[0] - 1
    last_extra = last_extra_impulse_tick[env]
    if reports_contact and (
        last_extra < 0
        or callback_tick > last_extra + 1
        or last_extra > callback_tick
    ):
        last_extra_impulse_tick[env] = callback_tick
        # Ball::_OnHit reads the RocketSim CarState/BallState snapshots that
        # were refreshed by the pre-tick hooks, before Bullet damping/gravity
        # and solver-body external impulses are applied.
        forward = wp.vec3(car_basis[0, 0], car_basis[1, 0], car_basis[2, 0])
        _rocketsim_extra_hit_velocity(
            ball_position,
            car_position,
            # Bullet applies the sphere body's linear damping in
            # predictUnconstraintMotion before narrowphase invokes the
            # contact-added callback. Ball::_OnHit therefore observes the
            # damped rigid-body velocity here, while the undamped car path is
            # still represented by car_pre_velocity.
            ball_pre_velocity,
            car_pre_velocity,
            forward,
            extra_velocity,
            ball_solved_velocity,
        )
    extra_hit_velocity_uu[env] = extra_velocity

    _bullet_cap_amd(
        car_solved_velocity, CAR_MAX_SPEED_BT, amd_rsqrtss_mantissa
    )
    _bullet_cap_amd(
        car_solved_angular, CAR_MAX_ANGULAR_SPEED, amd_rsqrtss_mantissa
    )
    _bullet_cap_amd(
        ball_solved_velocity, BALL_MAX_SPEED_BT, amd_rsqrtss_mantissa
    )
    _bullet_cap_amd(
        ball_solved_angular, BALL_MAX_ANGULAR_SPEED, amd_rsqrtss_mantissa
    )
    car_rigid_position_bt[car] = car_integrated_position
    car_rigid_velocity_bt[car] = car_solved_velocity
    car_pos[car] = car_integrated_position * 50.0
    car_vel[car] = car_solved_velocity * 50.0
    car_quat[car] = car_integrated_quat
    car_ang_vel[car] = car_solved_angular
    ball_resident_position_bt[env] = ball_integrated_position
    ball_resident_velocity_bt[env] = ball_solved_velocity
    ball_pos[env] = ball_integrated_position * 50.0
    ball_vel[env] = ball_solved_velocity * 50.0
    ball_quat[env] = ball_integrated_quat
    ball_ang_vel[env] = ball_solved_angular
