"""RocketSim-derived Octane wheel forces and static-world chassis contacts."""

import warp as wp

from rivalsim.kernels.bullet_box_triangle import (
    bullet_box_triangle_closest,
    bullet_box_triangle_penetration,
    bullet_internal_edge_best,
    bullet_manifold_replacement,
)
from rivalsim.vehicle_state import (
    MAX_CONTACTS_PER_CAR,
    MAX_MESH_CANDIDATES_PER_CAR,
)

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


_BULLET_QUATERNION_MATRIX = r"""
    // btMatrix3x3::setRotation in the pinned Windows Bullet build takes its
    // quaternion length through the SSE dot reduction (x*x + z*z) +
    // (y*y + w*w), then forms each row as two products, one add, the scale
    // multiply, and the identity add.  Keep those exact float32 boundaries on
    // CUDA instead of allowing Warp/NVCC to contract or reassociate them.
    auto bt_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto bt_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    auto bt_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };

    const float x = quat.x;
    const float y = quat.y;
    const float z = quat.z;
    const float w = quat.w;
    const float xx = bt_mul(x, x);
    const float yy = bt_mul(y, y);
    const float zz = bt_mul(z, z);
    const float ww = bt_mul(w, w);
    const float length_sq = bt_add(bt_add(xx, zz), bt_add(yy, ww));
    const float scale = bt_div(2.0f, length_sq);

    const float m00 = bt_add(
        1.0f, bt_mul(bt_add(bt_mul(-y, y), bt_mul(-z, z)), scale));
    const float m01 = bt_mul(bt_add(bt_mul(x, y), bt_mul(-w, z)), scale);
    const float m02 = bt_mul(bt_add(bt_mul(x, z), bt_mul(w, y)), scale);
    const float m10 = bt_mul(bt_add(bt_mul(x, y), bt_mul(w, z)), scale);
    const float m11 = bt_add(
        1.0f, bt_mul(bt_add(bt_mul(-x, x), bt_mul(-z, z)), scale));
    const float m12 = bt_mul(bt_add(bt_mul(y, z), bt_mul(-w, x)), scale);
    const float m20 = bt_mul(bt_add(bt_mul(x, z), bt_mul(-w, y)), scale);
    const float m21 = bt_mul(bt_add(bt_mul(y, z), bt_mul(w, x)), scale);
    const float m22 = bt_add(
        1.0f, bt_mul(bt_add(bt_mul(-x, x), bt_mul(-y, y)), scale));

    return wp::mat_t<3, 3, wp::float32>(
        m00, m01, m02,
        m10, m11, m12,
        m20, m21, m22);
"""


@wp.func_native(_BULLET_QUATERNION_MATRIX)
def _bullet_quaternion_matrix(quat: wp.quat) -> wp.mat33: ...


_BULLET_INVERSE_INERTIA_WORLD = r"""
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
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
    const float inverse_local[3] = {
        0.0185644571f, 0.0104337428f, 0.0075815497f};
    float scaled[3][3];
    float tensor[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            scaled[row][column] = op_mul(
                basis.data[row][column], inverse_local[column]);
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            tensor[row][column] = op_add(
                op_add(
                    op_mul(scaled[row][0], basis.data[column][0]),
                    op_mul(scaled[row][1], basis.data[column][1])),
                op_mul(scaled[row][2], basis.data[column][2]));
        }
    }
    return wp::mat_t<3, 3, wp::float32>(
        tensor[0][0], tensor[0][1], tensor[0][2],
        tensor[1][0], tensor[1][1], tensor[1][2],
        tensor[2][0], tensor[2][1], tensor[2][2]);
"""


@wp.func_native(_BULLET_INVERSE_INERTIA_WORLD)
def _bullet_inverse_inertia_world(basis: wp.mat33) -> wp.mat33: ...


_BULLET_VECTOR_SCALE_ADD = r"""
    // btVector3 scalar multiplication and addition are distinct SSE
    // instructions in the pinned Windows Bullet build.  Preserve the
    // intermediate rounded product instead of allowing NVCC to contract the
    // expression into an FMA.
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
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
    return wp::vec_t<3, wp::float32>(
        op_add(origin[0], op_mul(direction[0], amount)),
        op_add(origin[1], op_mul(direction[1], amount)),
        op_add(origin[2], op_mul(direction[2], amount)));
"""


@wp.func_native(_BULLET_VECTOR_SCALE_ADD)
def _bullet_vector_scale_add(
    origin: wp.vec3,
    direction: wp.vec3,
    amount: float,
) -> wp.vec3: ...


_AUTHORITY_INPUT_QUATERNION_MATRIX = r"""
    // The RocketSim authority adapter supplies CarState::rotMat directly. Its
    // matrix is produced by rivalsim.math.quat_to_matrix, not by Bullet's
    // btMatrix3x3(quaternion) constructor. Preserve that exact float32
    // multiply/add order for the first collision dispatch after SetState.
    auto input_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto input_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };

    const float x = quat.x;
    const float y = quat.y;
    const float z = quat.z;
    const float w = quat.w;
    const float two = 2.0f;
    const float m00 = input_add(
        1.0f, -input_mul(two, input_add(input_mul(y, y), input_mul(z, z))));
    const float m01 = input_mul(
        two, input_add(input_mul(x, y), -input_mul(z, w)));
    const float m02 = input_mul(
        two, input_add(input_mul(x, z), input_mul(y, w)));
    const float m10 = input_mul(
        two, input_add(input_mul(x, y), input_mul(z, w)));
    const float m11 = input_add(
        1.0f, -input_mul(two, input_add(input_mul(x, x), input_mul(z, z))));
    const float m12 = input_mul(
        two, input_add(input_mul(y, z), -input_mul(x, w)));
    const float m20 = input_mul(
        two, input_add(input_mul(x, z), -input_mul(y, w)));
    const float m21 = input_mul(
        two, input_add(input_mul(y, z), input_mul(x, w)));
    const float m22 = input_add(
        1.0f, -input_mul(two, input_add(input_mul(x, x), input_mul(y, y))));

    return wp::mat_t<3, 3, wp::float32>(
        m00, m01, m02,
        m10, m11, m12,
        m20, m21, m22);
"""


@wp.func_native(_AUTHORITY_INPUT_QUATERNION_MATRIX)
def _authority_input_quaternion_matrix(quat: wp.quat) -> wp.mat33: ...


_BULLET_QUATERNION_PRODUCT_NORMALIZED = r"""
    // btQuaternion::operator* in the pinned Windows build uses four-wide SSE
    // lanes.  Its grouping is ((w*b - cross_b) + (self*bw + cross_a)), not
    // the scalar expression emitted by Warp's generic quaternion multiply.
    // Follow that grouping and btQuaternion::safeNormalize/normalize exactly.
    auto product_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto product_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
        return value;
    #endif
    };
    auto product_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    auto product_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    auto product_sqrt = [](float value) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsqrt_rn(value);
    #else
        volatile float result = ::sqrtf(value);
        return result;
    #endif
    };

    const float ax = lhs.x;
    const float ay = lhs.y;
    const float az = lhs.z;
    const float aw = lhs.w;
    const float bx = rhs.x;
    const float by = rhs.y;
    const float bz = rhs.z;
    const float bw = rhs.w;

    float x = product_add(
        product_sub(product_mul(aw, bx), product_mul(az, by)),
        product_add(product_mul(ax, bw), product_mul(ay, bz)));
    float y = product_add(
        product_sub(product_mul(aw, by), product_mul(ax, bz)),
        product_add(product_mul(ay, bw), product_mul(az, bx)));
    float z = product_add(
        product_sub(product_mul(aw, bz), product_mul(ay, bx)),
        product_add(product_mul(az, bw), product_mul(ax, by)));
    float w = product_add(
        product_sub(product_mul(aw, bw), product_mul(az, bz)),
        -product_add(product_mul(ax, bx), product_mul(ay, by)));

    // SSE quaternion dot reduces (x*x + z*z) + (y*y + w*w).
    const float length_sq = product_add(
        product_add(product_mul(x, x), product_mul(z, z)),
        product_add(product_mul(y, y), product_mul(w, w)));
    if (length_sq > 1.1920928955078125e-7f) {
        const float inverse_length = product_div(1.0f, product_sqrt(length_sq));
        x = product_mul(x, inverse_length);
        y = product_mul(y, inverse_length);
        z = product_mul(z, inverse_length);
        w = product_mul(w, inverse_length);
    }
    return wp::quat_t<wp::float32>(x, y, z, w);
"""


@wp.func_native(_BULLET_QUATERNION_PRODUCT_NORMALIZED)
def _bullet_quaternion_product_normalized(lhs: wp.quat, rhs: wp.quat) -> wp.quat: ...


_BULLET_INVERSE_TRANSFORM_POINT = r"""
    // btTransform::invXform subtracts the origin, then multiplies by the
    // transpose of the basis. Explicit RN operations prevent NVCC from using
    // a different FMA/reassociation policy in the large contact kernel than
    // in the standalone witness probe. btTransform::invXform reaches
    // btMatrix3x3 * btVector3, whose pinned Windows SSE dot3 implementation
    // reduces X/Y first and then adds Z.
    auto inverse_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto inverse_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
        return value;
    #endif
    };
    auto inverse_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    const float dx = inverse_sub(point_bt[0], body_origin_bt[0]);
    const float dy = inverse_sub(point_bt[1], body_origin_bt[1]);
    const float dz = inverse_sub(point_bt[2], body_origin_bt[2]);
    auto column = [&](int i) -> float {
        return inverse_add(
            inverse_add(
                inverse_mul(basis.data[0][i], dx),
                inverse_mul(basis.data[1][i], dy)),
            inverse_mul(basis.data[2][i], dz));
    };
    return wp::vec_t<3, wp::float32>(column(0), column(1), column(2));
"""


@wp.func_native(_BULLET_INVERSE_TRANSFORM_POINT)
def _bullet_inverse_transform_point(
    body_origin_bt: wp.vec3,
    basis: wp.mat33,
    point_bt: wp.vec3,
) -> wp.vec3: ...


_BULLET_TRANSFORM_POINT = r"""
    // btTransform::operator()(btVector3) evaluates basis * point + origin.
    // Keep the pinned scalar/SSE-visible float32 boundaries explicit so
    // manifold refresh sees the same signed and lateral distances as Bullet.
    auto transform_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto transform_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    auto row = [&](int i) -> float {
        const float partial = transform_add(
            transform_mul(basis.data[i][0], point[0]),
            transform_mul(basis.data[i][1], point[1]));
        return transform_add(
            transform_add(partial, transform_mul(basis.data[i][2], point[2])),
            origin[i]);
    };
    return wp::vec_t<3, wp::float32>(row(0), row(1), row(2));
"""


@wp.func_native(_BULLET_TRANSFORM_POINT)
def _bullet_transform_point(
    origin: wp.vec3,
    basis: wp.mat33,
    point: wp.vec3,
) -> wp.vec3: ...


_BULLET_PLANE_CONTACT_WITNESS = r"""
    // btConvexPlaneCollisionAlgorithm::processCollision evaluates the box
    // support in plane-local space, projects it there, transforms the point
    // on B back to world space, and only then does btManifoldResult reconstruct
    // point A.  Collapsing the two identity-basis plane transforms is not
    // float32-equivalent because the plane origin is subtracted and added
    // around the support transform.
    auto plane_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto plane_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
        return value;
    #endif
    };
    auto plane_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    struct PlaneWitnessV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> PlaneWitnessV3 {
        PlaneWitnessV3 value = {x, y, z};
        return value;
    };
    auto add = [&](PlaneWitnessV3 a, PlaneWitnessV3 b) -> PlaneWitnessV3 {
        return make(
            plane_add(a.x, b.x),
            plane_add(a.y, b.y),
            plane_add(a.z, b.z));
    };
    auto sub = [&](PlaneWitnessV3 a, PlaneWitnessV3 b) -> PlaneWitnessV3 {
        return make(
            plane_sub(a.x, b.x),
            plane_sub(a.y, b.y),
            plane_sub(a.z, b.z));
    };
    auto scale = [&](PlaneWitnessV3 value, float amount) -> PlaneWitnessV3 {
        return make(
            plane_mul(value.x, amount),
            plane_mul(value.y, amount),
            plane_mul(value.z, amount));
    };
    auto dot = [&](PlaneWitnessV3 a, PlaneWitnessV3 b) -> float {
        return plane_add(
            plane_add(
                plane_mul(a.x, b.x),
                plane_mul(a.y, b.y)),
            plane_mul(a.z, b.z));
    };
    auto transform = [&](PlaneWitnessV3 origin, PlaneWitnessV3 point)
        -> PlaneWitnessV3 {
        return make(
            plane_add(
                plane_add(
                    plane_add(
                        plane_mul(basis.data[0][0], point.x),
                        plane_mul(basis.data[0][1], point.y)),
                    plane_mul(basis.data[0][2], point.z)),
                origin.x),
            plane_add(
                plane_add(
                    plane_add(
                        plane_mul(basis.data[1][0], point.x),
                        plane_mul(basis.data[1][1], point.y)),
                    plane_mul(basis.data[1][2], point.z)),
                origin.y),
            plane_add(
                plane_add(
                    plane_add(
                        plane_mul(basis.data[2][0], point.x),
                        plane_mul(basis.data[2][1], point.y)),
                    plane_mul(basis.data[2][2], point.z)),
                origin.z));
    };

    const PlaneWitnessV3 body_origin = make(
        body_origin_bt[0], body_origin_bt[1], body_origin_bt[2]);
    const PlaneWitnessV3 plane_origin = make(
        plane_origin_bt[0], plane_origin_bt[1], plane_origin_bt[2]);
    const PlaneWitnessV3 child_origin = make(
        child_origin_bt[0], child_origin_bt[1], child_origin_bt[2]);
    const PlaneWitnessV3 support = make(
        local_support_bt[0], local_support_bt[1], local_support_bt[2]);
    const PlaneWitnessV3 normal = make(
        plane_normal[0], plane_normal[1], plane_normal[2]);

    // bodyWorld * childTransform, then planeWorld.inverse() * convexWorld.
    const PlaneWitnessV3 convex_world_origin =
        transform(body_origin, child_origin);
    const PlaneWitnessV3 convex_in_plane_origin =
        sub(convex_world_origin, plane_origin);
    const PlaneWitnessV3 support_in_plane =
        transform(convex_in_plane_origin, support);
    const float depth = dot(normal, support_in_plane);

    // The four RocketSim arena planes have identity bases and zero local
    // constants. Preserve Bullet's vector multiply/subtract/add boundaries.
    const PlaneWitnessV3 projected_in_plane =
        sub(support_in_plane, scale(normal, depth));
    const PlaneWitnessV3 point_b = add(projected_in_plane, plane_origin);
    const PlaneWitnessV3 point_a = add(point_b, scale(normal, depth));

    point_a_bt = wp::vec_t<3, wp::float32>(
        point_a.x, point_a.y, point_a.z);
    point_b_bt = wp::vec_t<3, wp::float32>(
        point_b.x, point_b.y, point_b.z);
    distance_bt = depth;
"""


@wp.func_native(_BULLET_PLANE_CONTACT_WITNESS)
def _bullet_plane_contact_witness(
    body_origin_bt: wp.vec3,
    basis: wp.mat33,
    plane_origin_bt: wp.vec3,
    plane_normal: wp.vec3,
    child_origin_bt: wp.vec3,
    local_support_bt: wp.vec3,
    point_a_bt: wp.ref[wp.vec3],
    point_b_bt: wp.ref[wp.vec3],
    distance_bt: wp.ref[wp.float32],
): ...


_BULLET_SELECTED_TRIANGLE_RAYCAST = r"""
    // btTriangleRaycastCallback::processTriangle specialized to the face
    // selected by Warp's acceleration structure. Warp finds the candidate;
    // pinned Bullet arithmetic reconstructs the authoritative hit.
    auto ray_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto ray_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
        return value;
    #endif
    };
    auto ray_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    auto ray_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    struct RayV3 {
        float x;
        float y;
        float z;
    };
    auto make = [](float x, float y, float z) -> RayV3 {
        RayV3 value = {x, y, z};
        return value;
    };
    auto sub = [&](RayV3 a, RayV3 b) -> RayV3 {
        return make(ray_sub(a.x, b.x), ray_sub(a.y, b.y), ray_sub(a.z, b.z));
    };
    auto cross = [&](RayV3 a, RayV3 b) -> RayV3 {
        return make(
            ray_sub(ray_mul(a.y, b.z), ray_mul(a.z, b.y)),
            ray_sub(ray_mul(a.z, b.x), ray_mul(a.x, b.z)),
            ray_sub(ray_mul(a.x, b.y), ray_mul(a.y, b.x)));
    };
    auto dot = [&](RayV3 a, RayV3 b) -> float {
        return ray_add(
            ray_add(ray_mul(a.x, b.x), ray_mul(a.y, b.y)),
            ray_mul(a.z, b.z));
    };
    auto scale = [&](RayV3 value, float amount) -> RayV3 {
        return make(
            ray_mul(value.x, amount),
            ray_mul(value.y, amount),
            ray_mul(value.z, amount));
    };
    auto interpolate = [&](RayV3 from, RayV3 to, float fraction) -> RayV3 {
        const float from_fraction = ray_sub(1.0f, fraction);
        return make(
            ray_add(ray_mul(from.x, from_fraction), ray_mul(to.x, fraction)),
            ray_add(ray_mul(from.y, from_fraction), ray_mul(to.y, fraction)),
            ray_add(ray_mul(from.z, from_fraction), ray_mul(to.z, fraction)));
    };
    auto normalize = [&](RayV3 value) -> RayV3 {
        const float length_squared = dot(value, value);
        // btVector3::normalize uses the pinned SSE _mm_rsqrt_ss estimate and
        // one Newton-Raphson refinement, not a scalar sqrt/divide.  CUDA's
        // rsqrtf is the corresponding hardware estimate; retain every
        // multiply/subtract boundary from Bullet's implementation.
    #if defined(__CUDA_ARCH__)
        float inverse_length = rsqrtf(length_squared);
    #else
        float inverse_length = ray_div(1.0f, sqrtf(length_squared));
    #endif
        float half_length_squared = ray_mul(length_squared, 0.5f);
        float correction = ray_mul(half_length_squared, inverse_length);
        correction = ray_mul(correction, inverse_length);
        correction = ray_sub(1.5f, correction);
        inverse_length = ray_mul(inverse_length, correction);
        return scale(value, inverse_length);
    };

    const RayV3 from = make(source_bt[0], source_bt[1], source_bt[2]);
    const RayV3 to = make(target_bt[0], target_bt[1], target_bt[2]);
    const RayV3 vertex0 = make(v0_bt[0], v0_bt[1], v0_bt[2]);
    const RayV3 vertex1 = make(v1_bt[0], v1_bt[1], v1_bt[2]);
    const RayV3 vertex2 = make(v2_bt[0], v2_bt[1], v2_bt[2]);
    const RayV3 source_face_normal = make(
        face_normal[0], face_normal[1], face_normal[2]);
    const RayV3 v10 = sub(vertex1, vertex0);
    const RayV3 v20 = sub(vertex2, vertex0);
    RayV3 triangle_normal = cross(v10, v20);
    const float plane_distance = dot(vertex0, triangle_normal);
    const float distance_a = ray_sub(dot(triangle_normal, from), plane_distance);
    const float distance_b = ray_sub(dot(triangle_normal, to), plane_distance);
    valid = 0;
    if (ray_mul(distance_a, distance_b) < 0.0f) {
        const float projection_length = ray_sub(distance_a, distance_b);
        const float fraction = ray_div(distance_a, projection_length);
        if (fraction >= 0.0f && fraction < 1.0f) {
            const float edge_tolerance = ray_mul(
                dot(triangle_normal, triangle_normal), -0.0001f);
            const RayV3 point = interpolate(from, to, fraction);
            const RayV3 v0p = sub(vertex0, point);
            const RayV3 v1p = sub(vertex1, point);
            const RayV3 v2p = sub(vertex2, point);
            const bool inside0 = dot(cross(v0p, v1p), triangle_normal) >= edge_tolerance;
            const bool inside1 = dot(cross(v1p, v2p), triangle_normal) >= edge_tolerance;
            const bool inside2 = dot(cross(v2p, v0p), triangle_normal) >= edge_tolerance;
            if (inside0 && inside1 && inside2) {
                // The two pinned btVector3::normalize operations depend on
                // the authority CPU's rsqrt estimate. The immutable result is
                // precomputed once per static face with those exact intrinsics.
                triangle_normal = source_face_normal;
                if (distance_a <= 0.0f) {
                    triangle_normal = scale(triangle_normal, -1.0f);
                }
                hit_fraction = fraction;
                hit_point_bt = wp::vec_t<3, wp::float32>(point.x, point.y, point.z);
                hit_normal = wp::vec_t<3, wp::float32>(
                    triangle_normal.x, triangle_normal.y, triangle_normal.z);
                valid = 1;
            }
        }
    }
"""


@wp.func_native(_BULLET_SELECTED_TRIANGLE_RAYCAST)
def _bullet_selected_triangle_raycast(
    source_bt: wp.vec3,
    target_bt: wp.vec3,
    v0_bt: wp.vec3,
    v1_bt: wp.vec3,
    v2_bt: wp.vec3,
    face_normal: wp.vec3,
    hit_fraction: wp.ref[wp.float32],
    hit_point_bt: wp.ref[wp.vec3],
    hit_normal: wp.ref[wp.vec3],
    valid: wp.ref[wp.int32],
): ...


_BULLET_STATIC_PLANE_RAYCAST = r"""
    struct PlaneRayV3 {
        float x;
        float y;
        float z;
    };
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
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    auto make = [](float x, float y, float z) -> PlaneRayV3 {
        PlaneRayV3 value = {x, y, z};
        return value;
    };
    auto add = [&](PlaneRayV3 a, PlaneRayV3 b) -> PlaneRayV3 {
        return make(op_add(a.x, b.x), op_add(a.y, b.y), op_add(a.z, b.z));
    };
    auto sub = [&](PlaneRayV3 a, PlaneRayV3 b) -> PlaneRayV3 {
        return make(op_sub(a.x, b.x), op_sub(a.y, b.y), op_sub(a.z, b.z));
    };
    auto scale = [&](PlaneRayV3 value, float amount) -> PlaneRayV3 {
        return make(
            op_mul(value.x, amount),
            op_mul(value.y, amount),
            op_mul(value.z, amount));
    };
    auto neg = [&](PlaneRayV3 value) -> PlaneRayV3 {
        return make(-value.x, -value.y, -value.z);
    };
    auto dot = [&](PlaneRayV3 a, PlaneRayV3 b) -> float {
        return op_add(
            op_add(op_mul(a.x, b.x), op_mul(a.y, b.y)),
            op_mul(a.z, b.z));
    };
    auto cross = [&](PlaneRayV3 a, PlaneRayV3 b) -> PlaneRayV3 {
        return make(
            op_sub(op_mul(a.y, b.z), op_mul(a.z, b.y)),
            op_sub(op_mul(a.z, b.x), op_mul(a.x, b.z)),
            op_sub(op_mul(a.x, b.y), op_mul(a.y, b.x)));
    };
    auto sse_normalize = [&](PlaneRayV3 value) -> PlaneRayV3 {
        const float length_squared = dot(value, value);
        float inverse_length;
    #if defined(__CUDA_ARCH__)
        const unsigned length_bits = __float_as_uint(length_squared);
        if (length_bits >= 0x3f7ff000u && length_bits < 0x3f800000u) {
            inverse_length = __uint_as_float(0x3f800000u);
        } else if (length_bits >= 0x3f800000u && length_bits < 0x3f800800u) {
            inverse_length = __uint_as_float(0x3f7ff800u);
        } else {
            inverse_length = rsqrtf(length_squared);
        }
    #elif defined(__clang__) && (defined(__x86_64__) || defined(_M_X64))
        typedef float PlaneFloat4 __attribute__((__vector_size__(16)));
        const PlaneFloat4 estimate_input = {length_squared, 0.0f, 0.0f, 0.0f};
        const PlaneFloat4 estimate = __builtin_ia32_rsqrtss(estimate_input);
        inverse_length = estimate[0];
    #elif defined(_MSC_VER)
        const __m128 estimate_input = _mm_set_ss(length_squared);
        inverse_length = _mm_cvtss_f32(_mm_rsqrt_ss(estimate_input));
    #else
        inverse_length = op_div(1.0f, sqrtf(length_squared));
    #endif
        float correction = op_mul(op_mul(length_squared, 0.5f), inverse_length);
        correction = op_mul(correction, inverse_length);
        correction = op_sub(1.5f, correction);
        inverse_length = op_mul(inverse_length, correction);
        return scale(value, inverse_length);
    };
    auto interpolate = [&](PlaneRayV3 from, PlaneRayV3 to, float fraction) -> PlaneRayV3 {
        const float from_fraction = op_sub(1.0f, fraction);
        return make(
            op_add(op_mul(from.x, from_fraction), op_mul(to.x, fraction)),
            op_add(op_mul(from.y, from_fraction), op_mul(to.y, fraction)),
            op_add(op_mul(from.z, from_fraction), op_mul(to.z, fraction)));
    };

    const PlaneRayV3 source_world = make(source_bt[0], source_bt[1], source_bt[2]);
    const PlaneRayV3 target_world = make(target_bt[0], target_bt[1], target_bt[2]);
    const PlaneRayV3 origin = make(
        plane_origin_bt[0], plane_origin_bt[1], plane_origin_bt[2]);
    const PlaneRayV3 from = sub(source_world, origin);
    const PlaneRayV3 to = sub(target_world, origin);
    PlaneRayV3 plane_normal = sse_normalize(make(
        plane_normal_input[0], plane_normal_input[1], plane_normal_input[2]));

    const PlaneRayV3 aabb_min = make(
        fminf(from.x, to.x), fminf(from.y, to.y), fminf(from.z, to.z));
    const PlaneRayV3 aabb_max = make(
        fmaxf(from.x, to.x), fmaxf(from.y, to.y), fmaxf(from.z, to.z));
    const PlaneRayV3 half_extents = scale(sub(aabb_max, aabb_min), 0.5f);
    const float radius = sqrtf(dot(half_extents, half_extents));
    const PlaneRayV3 center = scale(add(aabb_max, aabb_min), 0.5f);

    PlaneRayV3 tangent0;
    PlaneRayV3 tangent1;
    if (fabsf(plane_normal.z) > 0.70710678118654752440f) {
        const float a = op_add(
            op_mul(plane_normal.y, plane_normal.y),
            op_mul(plane_normal.z, plane_normal.z));
        const float k = op_div(1.0f, sqrtf(a));
        tangent0 = make(
            0.0f, op_mul(-plane_normal.z, k), op_mul(plane_normal.y, k));
        tangent1 = make(
            op_mul(a, k),
            op_mul(-plane_normal.x, tangent0.z),
            op_mul(plane_normal.x, tangent0.y));
    } else {
        const float a = op_add(
            op_mul(plane_normal.x, plane_normal.x),
            op_mul(plane_normal.y, plane_normal.y));
        const float k = op_div(1.0f, sqrtf(a));
        tangent0 = make(
            op_mul(-plane_normal.y, k), op_mul(plane_normal.x, k), 0.0f);
        tangent1 = make(
            op_mul(-plane_normal.z, tangent0.y),
            op_mul(plane_normal.z, tangent0.x),
            op_mul(a, k));
    }

    const float projected_distance = dot(plane_normal, center);
    const PlaneRayV3 projected_center = sub(
        center, scale(plane_normal, projected_distance));
    const PlaneRayV3 tangent0_radius = scale(tangent0, radius);
    const PlaneRayV3 tangent1_radius = scale(tangent1, radius);

    float selected_fraction = current_fraction;
    PlaneRayV3 selected_normal = make(0.0f, 0.0f, 0.0f);
    bool found = false;
    auto process_triangle = [&](PlaneRayV3 vertex0,
                                PlaneRayV3 vertex1,
                                PlaneRayV3 vertex2) {
        const PlaneRayV3 v10 = sub(vertex1, vertex0);
        const PlaneRayV3 v20 = sub(vertex2, vertex0);
        PlaneRayV3 triangle_normal = cross(v10, v20);
        const float plane_distance = dot(vertex0, triangle_normal);
        const float distance_a = op_sub(dot(triangle_normal, from), plane_distance);
        const float distance_b = op_sub(dot(triangle_normal, to), plane_distance);
        if (op_mul(distance_a, distance_b) >= 0.0f) {
            return;
        }
        const float projection_length = op_sub(distance_a, distance_b);
        const float fraction = op_div(distance_a, projection_length);
        if (!(fraction < selected_fraction)) {
            return;
        }
        const float edge_tolerance = op_mul(dot(triangle_normal, triangle_normal), -0.0001f);
        const PlaneRayV3 point = interpolate(from, to, fraction);
        const PlaneRayV3 v0p = sub(vertex0, point);
        const PlaneRayV3 v1p = sub(vertex1, point);
        if (dot(cross(v0p, v1p), triangle_normal) < edge_tolerance) {
            return;
        }
        const PlaneRayV3 v2p = sub(vertex2, point);
        if (dot(cross(v1p, v2p), triangle_normal) < edge_tolerance) {
            return;
        }
        if (dot(cross(v2p, v0p), triangle_normal) < edge_tolerance) {
            return;
        }
        triangle_normal = sse_normalize(triangle_normal);
        if (distance_a <= 0.0f) {
            triangle_normal = neg(triangle_normal);
        }
        selected_fraction = fraction;
        selected_normal = triangle_normal;
        found = true;
    };

    const PlaneRayV3 triangle0 = add(
        add(projected_center, tangent0_radius), tangent1_radius);
    const PlaneRayV3 triangle1 = sub(
        add(projected_center, tangent0_radius), tangent1_radius);
    const PlaneRayV3 triangle2 = sub(
        sub(projected_center, tangent0_radius), tangent1_radius);
    process_triangle(triangle0, triangle1, triangle2);

    const PlaneRayV3 triangle3 = add(
        sub(projected_center, tangent0_radius), tangent1_radius);
    const PlaneRayV3 triangle4 = add(
        add(projected_center, tangent0_radius), tangent1_radius);
    process_triangle(triangle2, triangle3, triangle4);

    valid = found ? 1 : 0;
    if (found) {
        // btDefaultVehicleRaycaster normalizes the callback normal once more.
        selected_normal = sse_normalize(selected_normal);
        const PlaneRayV3 point_world = interpolate(
            source_world, target_world, selected_fraction);
        hit_fraction = selected_fraction;
        hit_point_bt = wp::vec_t<3, wp::float32>(
            point_world.x, point_world.y, point_world.z);
        hit_normal = wp::vec_t<3, wp::float32>(
            selected_normal.x, selected_normal.y, selected_normal.z);
    }
"""


@wp.func_native(_BULLET_STATIC_PLANE_RAYCAST)
def _bullet_static_plane_raycast(
    source_bt: wp.vec3,
    target_bt: wp.vec3,
    plane_origin_bt: wp.vec3,
    plane_normal_input: wp.vec3,
    current_fraction: float,
    hit_fraction: wp.ref[wp.float32],
    hit_point_bt: wp.ref[wp.vec3],
    hit_normal: wp.ref[wp.vec3],
    valid: wp.ref[wp.int32],
): ...


_BULLET_WHEEL_SUSPENSION = r"""
    // btVehicleRL::rayCast, resolveSingleCollision, and updateSuspension for
    // one static-world wheel, preserving the pinned Bullet-unit arithmetic.
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
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    struct WheelV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> WheelV3 {
        WheelV3 value = {x, y, z};
        return value;
    };
    auto sub = [&](WheelV3 a, WheelV3 b) -> WheelV3 {
        return make(op_sub(a.x, b.x), op_sub(a.y, b.y), op_sub(a.z, b.z));
    };
    auto add = [&](WheelV3 a, WheelV3 b) -> WheelV3 {
        return make(op_add(a.x, b.x), op_add(a.y, b.y), op_add(a.z, b.z));
    };
    auto cross = [&](WheelV3 a, WheelV3 b) -> WheelV3 {
        return make(
            op_sub(op_mul(a.y, b.z), op_mul(a.z, b.y)),
            op_sub(op_mul(a.z, b.x), op_mul(a.x, b.z)),
            op_sub(op_mul(a.x, b.y), op_mul(a.y, b.x)));
    };
    auto dot = [&](WheelV3 a, WheelV3 b) -> float {
        return op_add(
            op_add(op_mul(a.x, b.x), op_mul(a.y, b.y)),
            op_mul(a.z, b.z));
    };
    const WheelV3 source = make(source_bt[0], source_bt[1], source_bt[2]);
    const WheelV3 point = make(hit_point_bt[0], hit_point_bt[1], hit_point_bt[2]);
    const WheelV3 contact_normal = make(normal[0], normal[1], normal[2]);
    const WheelV3 chassis_up = make(up[0], up[1], up[2]);
    const WheelV3 body_position = make(body_position_bt[0], body_position_bt[1], body_position_bt[2]);
    const WheelV3 linear_velocity = make(linear_velocity_bt[0], linear_velocity_bt[1], linear_velocity_bt[2]);
    const WheelV3 angular_velocity = make(angular_velocity_world[0], angular_velocity_world[1], angular_velocity_world[2]);

    const float trace_distance = dot(sub(source, point), chassis_up);
    float length = op_sub(trace_distance, radius_bt);
    const float minimum_length = op_sub(rest_length_bt, suspension_travel_bt);
    const float maximum_length = op_add(rest_length_bt, suspension_travel_bt);
    if (length < minimum_length) length = minimum_length;
    if (length > maximum_length) length = maximum_length;

    const float denominator = dot(contact_normal, chassis_up);
    const WheelV3 relative_position = sub(point, body_position);
    const WheelV3 point_velocity = add(linear_velocity, cross(angular_velocity, relative_position));
    const float projected_velocity = dot(contact_normal, point_velocity);
    float relative_velocity = 0.0f;
    float clipped_inverse = 10.0f;
    if (denominator > 0.1f) {
        clipped_inverse = op_div(1.0f, denominator);
        relative_velocity = op_mul(projected_velocity, clipped_inverse);
    }

    float pushback = prior_pushback_bt;
    const float push_threshold = op_sub(op_add(rest_length_bt, radius_bt), 0.05f);
    if (dynamic_ground == 0 && trace_distance < push_threshold) {
        const float distance = op_sub(trace_distance, push_threshold);
        const WheelV3 torque_axis = cross(relative_position, contact_normal);
        const float inverse_local[3] = {0.0185644571f, 0.0104337428f, 0.0075815497f};
        float scaled[3][3];
        float tensor[3][3];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                scaled[row][column] = op_mul(basis.data[row][column], inverse_local[column]);
            }
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                tensor[row][column] = op_add(
                    op_add(
                        op_mul(scaled[row][0], basis.data[column][0]),
                        op_mul(scaled[row][1], basis.data[column][1])),
                    op_mul(scaled[row][2], basis.data[column][2]));
            }
        }
        const WheelV3 angular_component = make(
            op_add(op_add(op_mul(torque_axis.x, tensor[0][0]), op_mul(torque_axis.y, tensor[1][0])), op_mul(torque_axis.z, tensor[2][0])),
            op_add(op_add(op_mul(torque_axis.x, tensor[0][1]), op_mul(torque_axis.y, tensor[1][1])), op_mul(torque_axis.z, tensor[2][1])),
            op_add(op_add(op_mul(torque_axis.x, tensor[0][2]), op_mul(torque_axis.y, tensor[1][2])), op_mul(torque_axis.z, tensor[2][2])));
        const WheelV3 denominator_vector = cross(angular_component, relative_position);
        const float impulse_denominator = op_add(0.00555555569f, dot(contact_normal, denominator_vector));
        const float jacobian_inverse = op_div(1.0f, impulse_denominator);
        const float positional_error = op_div(op_mul(0.2f, -distance), solver_time_step);
        const float velocity_error = -projected_velocity;
        const float penetration_impulse = op_mul(positional_error, jacobian_inverse);
        const float velocity_impulse = op_mul(velocity_error, jacobian_inverse);
        pushback = op_add(penetration_impulse, velocity_impulse);
        if (pushback < 0.0f) pushback = 0.0f;
        pushback = op_div(pushback, 4.0f);
    }

    float force = op_mul(
        op_mul(op_sub(rest_length_bt, length), 500.0f),
        clipped_inverse);
    const float damping = relative_velocity < 0.0f ? 25.0f : 40.0f;
    force = op_sub(force, op_mul(damping, relative_velocity));
    force = op_mul(force, force_scale);
    if (force < 0.0f) force = 0.0f;
    suspension_length_bt = length;
    suspension_relative_velocity_bt = relative_velocity;
    suspension_clipped_inverse = clipped_inverse;
    suspension_force_bt = force;
    extra_pushback_bt = pushback;
"""


@wp.func_native(_BULLET_WHEEL_SUSPENSION)
def _bullet_wheel_suspension(
    source_bt: wp.vec3,
    hit_point_bt: wp.vec3,
    normal: wp.vec3,
    up: wp.vec3,
    body_position_bt: wp.vec3,
    linear_velocity_bt: wp.vec3,
    angular_velocity_world: wp.vec3,
    basis: wp.mat33,
    rest_length_bt: float,
    suspension_travel_bt: float,
    radius_bt: float,
    force_scale: float,
    solver_time_step: float,
    prior_pushback_bt: float,
    dynamic_ground: int,
    suspension_length_bt: wp.ref[wp.float32],
    suspension_relative_velocity_bt: wp.ref[wp.float32],
    suspension_clipped_inverse: wp.ref[wp.float32],
    suspension_force_bt: wp.ref[wp.float32],
    extra_pushback_bt: wp.ref[wp.float32],
): ...


_BULLET_RAY_SPHERE = r"""
    struct RayV3 {
        float x;
        float y;
        float z;
    };
    struct RayClosest {
        RayV3 closest;
        float weights[4];
        int used;
        int valid;
        int degenerate;
    };

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
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    auto make = [](float x, float y, float z) -> RayV3 {
        RayV3 value = {x, y, z};
        return value;
    };
    auto vadd = [&](RayV3 a, RayV3 b) -> RayV3 {
        return make(op_add(a.x, b.x), op_add(a.y, b.y), op_add(a.z, b.z));
    };
    auto vsub = [&](RayV3 a, RayV3 b) -> RayV3 {
        return make(op_sub(a.x, b.x), op_sub(a.y, b.y), op_sub(a.z, b.z));
    };
    auto vneg = [&](RayV3 a) -> RayV3 {
        return make(-a.x, -a.y, -a.z);
    };
    auto scale = [&](RayV3 a, float amount) -> RayV3 {
        return make(op_mul(a.x, amount), op_mul(a.y, amount), op_mul(a.z, amount));
    };
    auto dot = [&](RayV3 a, RayV3 b) -> float {
        return op_add(
            op_add(op_mul(a.x, b.x), op_mul(a.y, b.y)),
            op_mul(a.z, b.z));
    };
    auto length2 = [&](RayV3 a) -> float { return dot(a, a); };
    auto cross = [&](RayV3 a, RayV3 b) -> RayV3 {
        return make(
            op_sub(op_mul(a.y, b.z), op_mul(a.z, b.y)),
            op_sub(op_mul(a.z, b.x), op_mul(a.x, b.z)),
            op_sub(op_mul(a.x, b.y), op_mul(a.y, b.x)));
    };
    auto sse_normalize = [&](RayV3 value) -> RayV3 {
        const float squared = length2(value);
        float inverse;
    #if defined(__CUDA_ARCH__)
        const unsigned input_bits = __float_as_uint(squared);
        const unsigned exponent = (input_bits >> 23) & 0xffu;
        const unsigned result_exponent = ((380u - exponent) >> 1) << 23;
        const unsigned index = (input_bits >> 11) & 0x1fffu;
        const unsigned estimate_mantissa =
            static_cast<unsigned>(rsqrtss_mantissa.data[index]) << 11;
        inverse = __uint_as_float(result_exponent | estimate_mantissa);
    #elif defined(__clang__) && (defined(__x86_64__) || defined(_M_X64))
        typedef float RayFloat4 __attribute__((__vector_size__(16)));
        const RayFloat4 estimate_input = {squared, 0.0f, 0.0f, 0.0f};
        const RayFloat4 estimate = __builtin_ia32_rsqrtss(estimate_input);
        inverse = estimate[0];
    #elif defined(_MSC_VER)
        const __m128 estimate_input = _mm_set_ss(squared);
        inverse = _mm_cvtss_f32(_mm_rsqrt_ss(estimate_input));
    #else
        inverse = op_div(1.0f, sqrtf(squared));
    #endif
        float correction = op_mul(op_mul(squared, 0.5f), inverse);
        correction = op_mul(correction, inverse);
        correction = op_sub(1.5f, correction);
        inverse = op_mul(inverse, correction);
        return scale(value, inverse);
    };

    const RayV3 ray_from = make(source[0], source[1], source[2]);
    const RayV3 ray_to = make(target[0], target[1], target[2]);
    const RayV3 sphere_center = make(center[0], center[1], center[2]);
    const RayV3 r = vsub(ray_to, ray_from);
    auto sphere_basis_transpose_mul = [&](RayV3 value) -> RayV3 {
        return make(
            op_add(op_add(
                op_mul(value.x, sphere_basis.data[0][0]),
                op_mul(value.y, sphere_basis.data[1][0])),
                op_mul(value.z, sphere_basis.data[2][0])),
            op_add(op_add(
                op_mul(value.x, sphere_basis.data[0][1]),
                op_mul(value.y, sphere_basis.data[1][1])),
                op_mul(value.z, sphere_basis.data[2][1])),
            op_add(op_add(
                op_mul(value.x, sphere_basis.data[0][2]),
                op_mul(value.y, sphere_basis.data[1][2])),
                op_mul(value.z, sphere_basis.data[2][2])));
    };
    auto sphere_basis_mul = [&](RayV3 value) -> RayV3 {
        return make(
            op_add(op_add(
                op_mul(sphere_basis.data[0][0], value.x),
                op_mul(sphere_basis.data[0][1], value.y)),
                op_mul(sphere_basis.data[0][2], value.z)),
            op_add(op_add(
                op_mul(sphere_basis.data[1][0], value.x),
                op_mul(sphere_basis.data[1][1], value.y)),
                op_mul(sphere_basis.data[1][2], value.z)),
            op_add(op_add(
                op_mul(sphere_basis.data[2][0], value.x),
                op_mul(sphere_basis.data[2][1], value.y)),
                op_mul(sphere_basis.data[2][2], value.z)));
    };
    auto sphere_support = [&](RayV3 direction, RayV3 sphere_origin) -> RayV3 {
        const RayV3 local_direction = sphere_basis_transpose_mul(direction);
        RayV3 normalized;
        if (length2(local_direction) < op_mul(1.1920929e-7f, 1.1920929e-7f)) {
            normalized = sse_normalize(make(-1.0f, -1.0f, -1.0f));
        } else {
            normalized = sse_normalize(local_direction);
        }
        return vadd(sphere_origin, sphere_basis_mul(scale(normalized, radius)));
    };

    RayV3 simplex_w[5];
    RayV3 simplex_p[5];
    RayV3 simplex_q[5];
    int simplex_count = 0;
    RayV3 last_w = make(1.0e18f, 1.0e18f, 1.0e18f);
    RayV3 cached_p1 = make(0.0f, 0.0f, 0.0f);
    RayV3 cached_p2 = make(0.0f, 0.0f, 0.0f);
    RayV3 cached_v = make(0.0f, 0.0f, 0.0f);

    auto result_reset = [&](RayClosest& result) {
        result.closest = make(0.0f, 0.0f, 0.0f);
        result.weights[0] = 0.0f;
        result.weights[1] = 0.0f;
        result.weights[2] = 0.0f;
        result.weights[3] = 0.0f;
        result.used = 0;
        result.valid = 0;
        result.degenerate = 0;
    };
    auto result_valid = [&](RayClosest& result) -> int {
        return result.weights[0] >= 0.0f && result.weights[1] >= 0.0f
            && result.weights[2] >= 0.0f && result.weights[3] >= 0.0f;
    };
    auto triangle_closest = [&](RayV3 a, RayV3 b, RayV3 c, RayClosest& result) {
        result_reset(result);
        const RayV3 ab = vsub(b, a);
        const RayV3 ac = vsub(c, a);
        const RayV3 ap = vneg(a);
        const float d1 = dot(ab, ap);
        const float d2 = dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) {
            result.closest = a;
            result.weights[0] = 1.0f;
            result.used = 1;
            result.valid = 1;
            return;
        }
        const RayV3 bp = vneg(b);
        const float d3 = dot(ab, bp);
        const float d4 = dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) {
            result.closest = b;
            result.weights[1] = 1.0f;
            result.used = 2;
            result.valid = 1;
            return;
        }
        const float vc = op_sub(op_mul(d1, d4), op_mul(d3, d2));
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float amount = op_div(d1, op_sub(d1, d3));
            result.closest = vadd(a, scale(ab, amount));
            result.weights[0] = op_sub(1.0f, amount);
            result.weights[1] = amount;
            result.used = 3;
            result.valid = 1;
            return;
        }
        const RayV3 cp = vneg(c);
        const float d5 = dot(ab, cp);
        const float d6 = dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) {
            result.closest = c;
            result.weights[2] = 1.0f;
            result.used = 4;
            result.valid = 1;
            return;
        }
        const float vb = op_sub(op_mul(d5, d2), op_mul(d1, d6));
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float amount = op_div(d2, op_sub(d2, d6));
            result.closest = vadd(a, scale(ac, amount));
            result.weights[0] = op_sub(1.0f, amount);
            result.weights[2] = amount;
            result.used = 5;
            result.valid = 1;
            return;
        }
        const float va = op_sub(op_mul(d3, d6), op_mul(d5, d4));
        const float d43 = op_sub(d4, d3);
        const float d56 = op_sub(d5, d6);
        if (va <= 0.0f && d43 >= 0.0f && d56 >= 0.0f) {
            const float amount = op_div(d43, op_add(d43, d56));
            result.closest = vadd(b, scale(vsub(c, b), amount));
            result.weights[1] = op_sub(1.0f, amount);
            result.weights[2] = amount;
            result.used = 6;
            result.valid = 1;
            return;
        }
        const float inverse = op_div(1.0f, op_add(op_add(va, vb), vc));
        const float amount_v = op_mul(vb, inverse);
        const float amount_w = op_mul(vc, inverse);
        result.closest = vadd(vadd(a, scale(ab, amount_v)), scale(ac, amount_w));
        result.weights[0] = op_sub(op_sub(1.0f, amount_v), amount_w);
        result.weights[1] = amount_v;
        result.weights[2] = amount_w;
        result.used = 7;
        result.valid = 1;
    };
    auto outside_plane = [&](RayV3 a, RayV3 b, RayV3 c, RayV3 d) -> int {
        const RayV3 normal = cross(vsub(b, a), vsub(c, a));
        const float sign_p = dot(vneg(a), normal);
        const float sign_d = dot(vsub(d, a), normal);
        if (op_mul(sign_d, sign_d) < op_mul(1.0e-4f, 1.0e-4f)) return -1;
        return op_mul(sign_p, sign_d) < 0.0f ? 1 : 0;
    };
    auto tetrahedron_closest = [&](RayV3 a, RayV3 b, RayV3 c, RayV3 d,
                                    RayClosest& result) -> int {
        result_reset(result);
        result.used = 15;
        const int abc = outside_plane(a, b, c, d);
        const int acd = outside_plane(a, c, d, b);
        const int adb = outside_plane(a, d, b, c);
        const int bdc = outside_plane(b, d, c, a);
        if (abc < 0 || acd < 0 || adb < 0 || bdc < 0) {
            result.degenerate = 1;
            return 0;
        }
        if (!abc && !acd && !adb && !bdc) return 0;
        float best = 3.402823466e38f;
        RayClosest temporary;
        if (abc) {
            triangle_closest(a, b, c, temporary);
            const float candidate = length2(temporary.closest);
            if (candidate < best) {
                best = candidate;
                result.closest = temporary.closest;
                result.used = temporary.used;
                result.weights[0] = temporary.weights[0];
                result.weights[1] = temporary.weights[1];
                result.weights[2] = temporary.weights[2];
                result.weights[3] = 0.0f;
            }
        }
        if (acd) {
            triangle_closest(a, c, d, temporary);
            const float candidate = length2(temporary.closest);
            if (candidate < best) {
                best = candidate;
                result.closest = temporary.closest;
                result.used = 0;
                if (temporary.used & 1) result.used |= 1;
                if (temporary.used & 2) result.used |= 4;
                if (temporary.used & 4) result.used |= 8;
                result.weights[0] = temporary.weights[0];
                result.weights[1] = 0.0f;
                result.weights[2] = temporary.weights[1];
                result.weights[3] = temporary.weights[2];
            }
        }
        if (adb) {
            triangle_closest(a, d, b, temporary);
            const float candidate = length2(temporary.closest);
            if (candidate < best) {
                best = candidate;
                result.closest = temporary.closest;
                result.used = 0;
                if (temporary.used & 1) result.used |= 1;
                if (temporary.used & 2) result.used |= 8;
                if (temporary.used & 4) result.used |= 2;
                result.weights[0] = temporary.weights[0];
                result.weights[1] = temporary.weights[2];
                result.weights[2] = 0.0f;
                result.weights[3] = temporary.weights[1];
            }
        }
        if (bdc) {
            triangle_closest(b, d, c, temporary);
            const float candidate = length2(temporary.closest);
            if (candidate < best) {
                result.closest = temporary.closest;
                result.used = 0;
                if (temporary.used & 1) result.used |= 2;
                if (temporary.used & 2) result.used |= 8;
                if (temporary.used & 4) result.used |= 4;
                result.weights[0] = 0.0f;
                result.weights[1] = temporary.weights[0];
                result.weights[2] = temporary.weights[2];
                result.weights[3] = temporary.weights[1];
            }
        }
        result.valid = result_valid(result);
        return 1;
    };
    auto remove_vertex = [&](int index) {
        --simplex_count;
        simplex_w[index] = simplex_w[simplex_count];
        simplex_p[index] = simplex_p[simplex_count];
        simplex_q[index] = simplex_q[simplex_count];
    };
    auto reduce_vertices = [&](int used) {
        if (simplex_count >= 4 && !(used & 8)) remove_vertex(3);
        if (simplex_count >= 3 && !(used & 4)) remove_vertex(2);
        if (simplex_count >= 2 && !(used & 2)) remove_vertex(1);
        if (simplex_count >= 1 && !(used & 1)) remove_vertex(0);
    };
    auto simplex_closest = [&]() -> int {
        RayClosest closest;
        result_reset(closest);
        if (simplex_count == 1) {
            cached_p1 = simplex_p[0];
            cached_p2 = simplex_q[0];
            cached_v = vsub(cached_p1, cached_p2);
            return 1;
        }
        if (simplex_count == 2) {
            const RayV3 from = simplex_w[0];
            const RayV3 to = simplex_w[1];
            const RayV3 difference = vsub(to, from);
            float parameter = dot(difference, vneg(from));
            int used = 0;
            if (parameter > 0.0f) {
                const float dot_vv = dot(difference, difference);
                if (parameter < dot_vv) {
                    parameter = op_div(parameter, dot_vv);
                    used = 3;
                } else {
                    parameter = 1.0f;
                    used = 2;
                }
            } else {
                parameter = 0.0f;
                used = 1;
            }
            closest.weights[0] = op_sub(1.0f, parameter);
            closest.weights[1] = parameter;
            cached_p1 = vadd(simplex_p[0], scale(vsub(simplex_p[1], simplex_p[0]), parameter));
            cached_p2 = vadd(simplex_q[0], scale(vsub(simplex_q[1], simplex_q[0]), parameter));
            cached_v = vsub(cached_p1, cached_p2);
            reduce_vertices(used);
            return result_valid(closest);
        }
        if (simplex_count == 3) {
            triangle_closest(simplex_w[0], simplex_w[1], simplex_w[2], closest);
            cached_p1 = vadd(
                vadd(scale(simplex_p[0], closest.weights[0]), scale(simplex_p[1], closest.weights[1])),
                scale(simplex_p[2], closest.weights[2]));
            cached_p2 = vadd(
                vadd(scale(simplex_q[0], closest.weights[0]), scale(simplex_q[1], closest.weights[1])),
                scale(simplex_q[2], closest.weights[2]));
            cached_v = vsub(cached_p1, cached_p2);
            reduce_vertices(closest.used);
            return result_valid(closest);
        }
        const int separated = tetrahedron_closest(
            simplex_w[0], simplex_w[1], simplex_w[2], simplex_w[3], closest);
        if (separated) {
            cached_p1 = vadd(vadd(vadd(
                scale(simplex_p[0], closest.weights[0]),
                scale(simplex_p[1], closest.weights[1])),
                scale(simplex_p[2], closest.weights[2])),
                scale(simplex_p[3], closest.weights[3]));
            cached_p2 = vadd(vadd(vadd(
                scale(simplex_q[0], closest.weights[0]),
                scale(simplex_q[1], closest.weights[1])),
                scale(simplex_q[2], closest.weights[2])),
                scale(simplex_q[3], closest.weights[3]));
            cached_v = vsub(cached_p1, cached_p2);
            reduce_vertices(closest.used);
        } else if (closest.degenerate) {
            return 0;
        } else {
            cached_v = make(0.0f, 0.0f, 0.0f);
            return 1;
        }
        return result_valid(closest);
    };
    auto in_simplex = [&](RayV3 value) -> int {
        for (int index = 0; index < simplex_count; ++index) {
            if (length2(vsub(simplex_w[index], value)) <= 0.0001f) return 1;
        }
        return value.x == last_w.x && value.y == last_w.y && value.z == last_w.z;
    };

    float lambda = 0.0f;
    RayV3 interpolated_a = ray_from;
    RayV3 interpolated_b = sphere_center;
    const RayV3 initial_b = sphere_support(r, sphere_center);
    RayV3 v = vsub(ray_from, initial_b);
    RayV3 n = make(0.0f, 0.0f, 0.0f);
    float dist2 = length2(v);
    int remaining = 32;
    int cast_valid = 1;
    while (dist2 > 0.0001f && remaining--) {
        const RayV3 support_a = interpolated_a;
        const RayV3 support_b = sphere_support(v, interpolated_b);
        RayV3 w = vsub(support_a, support_b);
        const float v_dot_w = dot(v, w);
        if (lambda > 1.0f) {
            cast_valid = 0;
            break;
        }
        if (v_dot_w > 0.0f) {
            const float v_dot_r = dot(v, r);
            if (v_dot_r >= -op_mul(1.1920929e-7f, 1.1920929e-7f)) {
                cast_valid = 0;
                break;
            }
            lambda = op_sub(lambda, op_div(v_dot_w, v_dot_r));
            // btVector3::setInterpolate3 uses (1-lambda)*from + lambda*to
            // in the pinned SSE build, rather than from + lambda*(to-from).
            interpolated_a = vadd(
                scale(ray_from, op_sub(1.0f, lambda)),
                scale(ray_to, lambda));
            // The pinned btSubsimplexConvexCast interpolates both transform
            // origins after every lambda advance. fromB and toB are the same
            // rigid sphere transform, but setInterpolate3 still evaluates
            // (1-lambda)*fromB + lambda*toB and its float32 roundoff feeds the
            // next support query.
            interpolated_b = vadd(
                scale(sphere_center, op_sub(1.0f, lambda)),
                scale(sphere_center, lambda));
            w = vsub(support_a, support_b);
            n = v;
        }
        if (!in_simplex(w)) {
            last_w = w;
            simplex_w[simplex_count] = w;
            simplex_p[simplex_count] = support_a;
            simplex_q[simplex_count] = support_b;
            ++simplex_count;
        }
        if (simplex_closest()) {
            v = cached_v;
            dist2 = length2(v);
        } else {
            dist2 = 0.0f;
        }
    }

    if (cast_valid) {
        RayV3 normal = make(0.0f, 0.0f, 0.0f);
        if (length2(n) >= op_mul(1.1920929e-7f, 1.1920929e-7f)) {
            normal = sse_normalize(n);
        }
        if (dot(normal, r) >= 0.0f) cast_valid = 0;
        if (cast_valid && length2(normal) > 0.0001f && lambda < maximum_fraction) {
            // rayTestSingleInternal normalizes the cast result a second time,
            // then btDefaultVehicleRaycaster normalizes the callback normal.
            normal = sse_normalize(normal);
            normal = sse_normalize(normal);
            fraction_out = lambda;
            const RayV3 point = vadd(
                scale(ray_from, op_sub(1.0f, lambda)),
                scale(ray_to, lambda));
            point_out = wp::vec_t<3, wp::float32>(point.x, point.y, point.z);
            normal_out = wp::vec_t<3, wp::float32>(normal.x, normal.y, normal.z);
            valid = 1;
        } else {
            valid = 0;
        }
    } else {
        valid = 0;
    }
"""


@wp.func_native(_BULLET_RAY_SPHERE)
def _bullet_ray_sphere(
    source: wp.vec3,
    target: wp.vec3,
    center: wp.vec3,
    sphere_basis: wp.mat33,
    radius: float,
    maximum_fraction: float,
    rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    fraction_out: wp.ref[wp.float32],
    point_out: wp.ref[wp.vec3],
    normal_out: wp.ref[wp.vec3],
    valid: wp.ref[wp.int32],
): ...


_BULLET_APPLY_IMPULSE = r"""
    // btRigidBody::applyImpulse for the Octane rigid body. The inverse inertia
    // tensor is materialized from the exact stored Bullet basis before the
    // impulse, matching btRigidBody::updateInertiaTensor.
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
    struct ImpulseV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> ImpulseV3 {
        ImpulseV3 value = {x, y, z};
        return value;
    };
    auto cross = [&](ImpulseV3 a, ImpulseV3 b) -> ImpulseV3 {
        return make(
            op_sub(op_mul(a.y, b.z), op_mul(a.z, b.y)),
            op_sub(op_mul(a.z, b.x), op_mul(a.x, b.z)),
            op_sub(op_mul(a.x, b.y), op_mul(a.y, b.x)));
    };
    const ImpulseV3 applied = make(impulse_bt[0], impulse_bt[1], impulse_bt[2]);
    const ImpulseV3 relative = make(relative_position_bt[0], relative_position_bt[1], relative_position_bt[2]);
    ImpulseV3 linear = make(linear_velocity_bt[0], linear_velocity_bt[1], linear_velocity_bt[2]);
    ImpulseV3 angular = make(angular_velocity_world[0], angular_velocity_world[1], angular_velocity_world[2]);
    linear.x = op_add(linear.x, op_mul(applied.x, 0.00555555569f));
    linear.y = op_add(linear.y, op_mul(applied.y, 0.00555555569f));
    linear.z = op_add(linear.z, op_mul(applied.z, 0.00555555569f));
    const ImpulseV3 torque = cross(relative, applied);
    const float inverse_local[3] = {0.0185644571f, 0.0104337428f, 0.0075815497f};
    float scaled[3][3];
    float tensor[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            scaled[row][column] = op_mul(basis.data[row][column], inverse_local[column]);
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            tensor[row][column] = op_add(
                op_add(
                    op_mul(scaled[row][0], basis.data[column][0]),
                    op_mul(scaled[row][1], basis.data[column][1])),
                op_mul(scaled[row][2], basis.data[column][2]));
        }
    }
    const ImpulseV3 angular_delta = make(
        op_add(op_add(op_mul(tensor[0][0], torque.x), op_mul(tensor[0][1], torque.y)), op_mul(tensor[0][2], torque.z)),
        op_add(op_add(op_mul(tensor[1][0], torque.x), op_mul(tensor[1][1], torque.y)), op_mul(tensor[1][2], torque.z)),
        op_add(op_add(op_mul(tensor[2][0], torque.x), op_mul(tensor[2][1], torque.y)), op_mul(tensor[2][2], torque.z)));
    angular.x = op_add(angular.x, angular_delta.x);
    angular.y = op_add(angular.y, angular_delta.y);
    angular.z = op_add(angular.z, angular_delta.z);
    linear_velocity_bt = wp::vec_t<3, wp::float32>(linear.x, linear.y, linear.z);
    angular_velocity_world = wp::vec_t<3, wp::float32>(angular.x, angular.y, angular.z);
"""


@wp.func_native(_BULLET_APPLY_IMPULSE)
def _bullet_apply_impulse(
    basis: wp.mat33,
    impulse_bt: wp.vec3,
    relative_position_bt: wp.vec3,
    linear_velocity_bt: wp.ref[wp.vec3],
    angular_velocity_world: wp.ref[wp.vec3],
): ...


_BULLET_SUSPENSION_IMPULSE = r"""
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
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
    const float scale = op_add(op_mul(suspension_force_bt, time_step), extra_pushback_bt);
    return wp::vec_t<3, wp::float32>(
        op_mul(normal[0], scale),
        op_mul(normal[1], scale),
        op_mul(normal[2], scale));
"""


@wp.func_native(_BULLET_SUSPENSION_IMPULSE)
def _bullet_suspension_impulse(
    normal: wp.vec3,
    suspension_force_bt: float,
    time_step: float,
    extra_pushback_bt: float,
) -> wp.vec3: ...


_BULLET_WHEEL_FRICTION = r"""
    // btVehicleRL::calcFrictionImpulses plus the static-ground portion of
    // applyFrictionImpulses. This is the fixed Octane/Soccar path: the ground
    // body has zero mass and inertia, while the chassis uses the pinned local
    // inverse inertia and Bullet-unit velocities.
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b; return value;
    #endif
    };
    auto op_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b; return value;
    #endif
    };
    auto op_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b; return value;
    #endif
    };
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b; return value;
    #endif
    };
    auto op_sqrt = [](float value) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsqrt_rn(value);
    #else
        volatile float result = ::sqrtf(value); return result;
    #endif
    };
    struct WheelV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> WheelV3 {
        WheelV3 value = {x, y, z}; return value;
    };
    auto add = [&](WheelV3 a, WheelV3 b) -> WheelV3 {
        return make(op_add(a.x,b.x),op_add(a.y,b.y),op_add(a.z,b.z));
    };
    auto sub = [&](WheelV3 a, WheelV3 b) -> WheelV3 {
        return make(op_sub(a.x,b.x),op_sub(a.y,b.y),op_sub(a.z,b.z));
    };
    auto scale = [&](WheelV3 a, float s) -> WheelV3 {
        return make(op_mul(a.x,s),op_mul(a.y,s),op_mul(a.z,s));
    };
    auto cross = [&](WheelV3 a, WheelV3 b) -> WheelV3 {
        return make(
            op_sub(op_mul(a.y,b.z),op_mul(a.z,b.y)),
            op_sub(op_mul(a.z,b.x),op_mul(a.x,b.z)),
            op_sub(op_mul(a.x,b.y),op_mul(a.y,b.x)));
    };
    auto dot = [&](WheelV3 a, WheelV3 b) -> float {
        return op_add(op_add(op_mul(a.x,b.x),op_mul(a.y,b.y)),op_mul(a.z,b.z));
    };
    auto safe_normalize = [&](WheelV3 value) -> WheelV3 {
        const float length_squared = dot(value, value);
        if (length_squared >= 1.4210854715202004e-14f) {
            const float inverse = op_div(1.0f, op_sqrt(length_squared));
            return scale(value, inverse);
        }
        return make(1.0f, 0.0f, 0.0f);
    };

    const WheelV3 origin = make(body_origin_bt[0],body_origin_bt[1],body_origin_bt[2]);
    const WheelV3 point = make(contact_point_bt[0],contact_point_bt[1],contact_point_bt[2]);
    const WheelV3 normal = make(surface_normal[0],surface_normal[1],surface_normal[2]);
    const WheelV3 linear = make(linear_velocity_bt[0],linear_velocity_bt[1],linear_velocity_bt[2]);
    const WheelV3 angular = make(angular_velocity_world[0],angular_velocity_world[1],angular_velocity_world[2]);
    const WheelV3 ground_origin = make(ground_origin_bt[0],ground_origin_bt[1],ground_origin_bt[2]);
    const WheelV3 ground_linear = make(ground_linear_velocity_bt[0],ground_linear_velocity_bt[1],ground_linear_velocity_bt[2]);
    const WheelV3 ground_angular = make(ground_angular_velocity_world[0],ground_angular_velocity_world[1],ground_angular_velocity_world[2]);

    WheelV3 axle = make(raw_axle[0],raw_axle[1],raw_axle[2]);
    const float projection = dot(axle, normal);
    axle = safe_normalize(sub(axle, scale(normal, projection)));
    WheelV3 forward = safe_normalize(cross(normal, axle));

    const WheelV3 relative = sub(point, origin);
    const WheelV3 velocity = add(linear, cross(angular, relative));
    const WheelV3 ground_relative = sub(point, ground_origin);
    WheelV3 ground_velocity = make(0.0f,0.0f,0.0f);
    if (dynamic_ground != 0)
        ground_velocity = add(ground_linear,cross(ground_angular,ground_relative));

    // btJacobianEntry transforms rel_pos.cross(axis) into chassis-local space,
    // applies the diagonal local inverse inertia, then forms its diagonal.
    const WheelV3 torque = cross(relative, axle);
    const WheelV3 local_torque = make(
        op_add(op_add(op_mul(basis.data[0][0],torque.x),op_mul(basis.data[1][0],torque.y)),op_mul(basis.data[2][0],torque.z)),
        op_add(op_add(op_mul(basis.data[0][1],torque.x),op_mul(basis.data[1][1],torque.y)),op_mul(basis.data[2][1],torque.z)),
        op_add(op_add(op_mul(basis.data[0][2],torque.x),op_mul(basis.data[1][2],torque.y)),op_mul(basis.data[2][2],torque.z)));
    const WheelV3 inverse_mass_torque = make(
        op_mul(0.0185644571f,local_torque.x),
        op_mul(0.0104337428f,local_torque.y),
        op_mul(0.0075815497f,local_torque.z));
    float diagonal = op_add(0.00555555569f,dot(inverse_mass_torque,local_torque));
    if (dynamic_ground != 0) {
        // Two-body btJacobianEntry does not collapse the sphere's isotropic
        // inertia in world space. It transforms rel_pos.cross(-axis) into the
        // ball's local basis, multiplies by the stored local inverse inertia,
        // and accumulates massA + angularA + massB + angularB left-to-right.
        const WheelV3 ground_torque_world = cross(ground_relative,scale(axle,-1.0f));
        const WheelV3 ground_torque_local = make(
            op_add(op_add(op_mul(ground_basis.data[0][0],ground_torque_world.x),op_mul(ground_basis.data[1][0],ground_torque_world.y)),op_mul(ground_basis.data[2][0],ground_torque_world.z)),
            op_add(op_add(op_mul(ground_basis.data[0][1],ground_torque_world.x),op_mul(ground_basis.data[1][1],ground_torque_world.y)),op_mul(ground_basis.data[2][1],ground_torque_world.z)),
            op_add(op_add(op_mul(ground_basis.data[0][2],ground_torque_world.x),op_mul(ground_basis.data[1][2],ground_torque_world.y)),op_mul(ground_basis.data[2][2],ground_torque_world.z)));
        const WheelV3 ground_inverse_mass_torque = make(
            op_mul(0.0250203293f,ground_torque_local.x),
            op_mul(0.0250203293f,ground_torque_local.y),
            op_mul(0.0250203293f,ground_torque_local.z));
        diagonal = op_add(
            op_add(diagonal,0.0333333351f),
            dot(ground_inverse_mass_torque,ground_torque_local));
    }
    const float inverse_diagonal = op_div(1.0f,diagonal);
    const float relative_lateral_velocity = dot(axle,sub(velocity,ground_velocity));
    const float side = op_mul(
        op_mul(-0.2f,relative_lateral_velocity),inverse_diagonal);

    float rolling = 0.0f;
    if (engine_force_bt == 0.0f) {
        if (brake_force_bt != 0.0f) {
            // RocketSim passes carRelContactPoint to the ground body's
            // getVelocityInLocalPoint call in this legacy rolling branch.
            WheelV3 rolling_ground_velocity = make(0.0f,0.0f,0.0f);
            if (dynamic_ground != 0)
                rolling_ground_velocity = add(ground_linear,cross(ground_angular,relative));
            float relative_forward_velocity = dot(sub(velocity,rolling_ground_velocity),forward);
            if (time_step > 0.0125000002f) {
                const float threshold = op_add(
                    -op_div(1.0f,op_mul(time_step,150.0f)),0.8f);
                if (::fabsf(relative_forward_velocity) < threshold)
                    relative_forward_velocity = 0.0f;
            }
            rolling = op_mul(-relative_forward_velocity,113.73963f);
            if (rolling < -brake_force_bt) rolling = -brake_force_bt;
            if (rolling > brake_force_bt) rolling = brake_force_bt;
        }
    } else {
        rolling = op_div(-engine_force_bt,60.0f);
    }

    const WheelV3 forward_force = scale(scale(forward,rolling),longitudinal_friction);
    const WheelV3 lateral_force = scale(scale(axle,side),lateral_friction);
    const WheelV3 wheel_impulse = scale(add(forward_force,lateral_force),60.0f);
    const WheelV3 applied = scale(wheel_impulse,time_step);

    // btVehicleRL::applyFrictionImpulses applies ROLLING_INFLUENCE_FIX to the
    // lever arm after calculating the impulse.
    const WheelV3 up = make(
        basis.data[0][2],basis.data[1][2],basis.data[2][2]);
    const float contact_up_dot = dot(up,relative);
    const WheelV3 projected_relative = sub(relative,scale(up,contact_up_dot));

    axle_direction = wp::vec_t<3,wp::float32>(axle.x,axle.y,axle.z);
    forward_direction = wp::vec_t<3,wp::float32>(forward.x,forward.y,forward.z);
    relative_position_bt = wp::vec_t<3,wp::float32>(projected_relative.x,projected_relative.y,projected_relative.z);
    applied_impulse_bt = wp::vec_t<3,wp::float32>(applied.x,applied.y,applied.z);
    side_impulse_bt = side;
    rolling_friction_bt = rolling;
"""


@wp.func_native(_BULLET_WHEEL_FRICTION)
def _bullet_wheel_friction(
    basis: wp.mat33,
    ground_basis: wp.mat33,
    body_origin_bt: wp.vec3,
    linear_velocity_bt: wp.vec3,
    angular_velocity_world: wp.vec3,
    dynamic_ground: int,
    ground_origin_bt: wp.vec3,
    ground_linear_velocity_bt: wp.vec3,
    ground_angular_velocity_world: wp.vec3,
    contact_point_bt: wp.vec3,
    surface_normal: wp.vec3,
    raw_axle: wp.vec3,
    engine_force_bt: float,
    brake_force_bt: float,
    lateral_friction: float,
    longitudinal_friction: float,
    time_step: float,
    axle_direction: wp.ref[wp.vec3],
    forward_direction: wp.ref[wp.vec3],
    relative_position_bt: wp.ref[wp.vec3],
    applied_impulse_bt: wp.ref[wp.vec3],
    side_impulse_bt: wp.ref[wp.float32],
    rolling_friction_bt: wp.ref[wp.float32],
): ...


_BULLET_WHEEL_FRICTION_COEFFICIENTS = r"""
    // Car::_UpdateWheels friction cache. Keep the calculation in Bullet
    // units until the source's explicit BT_TO_UU multiply; forming the wheel
    // lever arm from already-scaled UU coordinates changes reachable bits.
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b; return value;
    #endif
    };
    auto op_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b; return value;
    #endif
    };
    auto op_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b; return value;
    #endif
    };
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b; return value;
    #endif
    };
    struct CoeffV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> CoeffV3 {
        CoeffV3 value = {x, y, z}; return value;
    };
    auto add = [&](CoeffV3 a, CoeffV3 b) -> CoeffV3 {
        return make(op_add(a.x,b.x),op_add(a.y,b.y),op_add(a.z,b.z));
    };
    auto sub = [&](CoeffV3 a, CoeffV3 b) -> CoeffV3 {
        return make(op_sub(a.x,b.x),op_sub(a.y,b.y),op_sub(a.z,b.z));
    };
    auto scale = [&](CoeffV3 value, float amount) -> CoeffV3 {
        return make(
            op_mul(value.x,amount),
            op_mul(value.y,amount),
            op_mul(value.z,amount));
    };
    auto dot = [&](CoeffV3 a, CoeffV3 b) -> float {
        return op_add(
            op_add(op_mul(a.x,b.x),op_mul(a.y,b.y)),
            op_mul(a.z,b.z));
    };
    auto cross = [&](CoeffV3 a, CoeffV3 b) -> CoeffV3 {
        return make(
            op_sub(op_mul(a.y,b.z),op_mul(a.z,b.y)),
            op_sub(op_mul(a.z,b.x),op_mul(a.x,b.z)),
            op_sub(op_mul(a.x,b.y),op_mul(a.y,b.x)));
    };

    const CoeffV3 origin = make(
        body_origin_bt[0], body_origin_bt[1], body_origin_bt[2]);
    const CoeffV3 hard_point = make(
        hard_point_bt[0], hard_point_bt[1], hard_point_bt[2]);
    const CoeffV3 linear = make(
        linear_velocity_bt[0], linear_velocity_bt[1], linear_velocity_bt[2]);
    const CoeffV3 angular = make(
        angular_velocity_world[0], angular_velocity_world[1], angular_velocity_world[2]);
    const CoeffV3 lateral = make(
        lateral_direction[0], lateral_direction[1], lateral_direction[2]);
    const CoeffV3 normal = make(
        surface_normal[0], surface_normal[1], surface_normal[2]);
    const CoeffV3 longitudinal = cross(lateral, normal);
    const CoeffV3 wheel_delta = sub(hard_point, origin);
    const CoeffV3 velocity_uu = scale(
        add(cross(angular, wheel_delta), linear), 50.0f);

    const float base_friction = fabsf(dot(velocity_uu, lateral));
    float curve_input = 0.0f;
    if (base_friction > 5.0f) {
        const float longitudinal_speed = fabsf(dot(velocity_uu, longitudinal));
        curve_input = op_div(
            base_friction, op_add(longitudinal_speed, base_friction));
    }

    float lateral_value;
    if (curve_input <= 0.0f) {
        lateral_value = 1.0f;
    } else if (curve_input < 1.0f) {
        const float range_between = op_sub(1.0f, 0.0f);
        const float value_difference = op_sub(0.2f, 1.0f);
        const float factor = op_div(op_sub(curve_input, 0.0f), range_between);
        lateral_value = op_add(1.0f, op_mul(value_difference, factor));
    } else {
        lateral_value = 0.2f;
    }
    float longitudinal_value = 1.0f;

    if (handbrake_value != 0.0f) {
        const float lateral_curve = 0.1f;
        const float lateral_factor = op_add(
            op_mul(op_sub(lateral_curve, 1.0f), handbrake_value), 1.0f);
        lateral_value = op_mul(lateral_value, lateral_factor);

        float longitudinal_curve;
        if (curve_input <= 0.0f) {
            longitudinal_curve = 0.5f;
        } else if (curve_input < 1.0f) {
            const float factor = op_div(
                op_sub(curve_input, 0.0f), op_sub(1.0f, 0.0f));
            longitudinal_curve = op_add(
                0.5f, op_mul(op_sub(0.9f, 0.5f), factor));
        } else {
            longitudinal_curve = 0.9f;
        }
        const float longitudinal_factor = op_add(
            op_mul(op_sub(longitudinal_curve, 1.0f), handbrake_value), 1.0f);
        longitudinal_value = op_mul(longitudinal_value, longitudinal_factor);
    }

    if (real_throttle == 0.0f) {
        float non_sticky;
        if (normal.z <= 0.0f) {
            non_sticky = 0.1f;
        } else if (normal.z < 0.7075f) {
            const float factor = op_div(
                op_sub(normal.z, 0.0f), op_sub(0.7075f, 0.0f));
            non_sticky = op_add(
                0.1f, op_mul(op_sub(0.5f, 0.1f), factor));
        } else if (normal.z < 1.0f) {
            const float factor = op_div(
                op_sub(normal.z, 0.7075f), op_sub(1.0f, 0.7075f));
            non_sticky = op_add(
                0.5f, op_mul(op_sub(1.0f, 0.5f), factor));
        } else {
            non_sticky = 1.0f;
        }
        lateral_value = op_mul(lateral_value, non_sticky);
        longitudinal_value = op_mul(longitudinal_value, non_sticky);
    }

    lateral_friction = lateral_value;
    longitudinal_friction = longitudinal_value;
"""


@wp.func_native(_BULLET_WHEEL_FRICTION_COEFFICIENTS)
def _bullet_wheel_friction_coefficients(
    body_origin_bt: wp.vec3,
    linear_velocity_bt: wp.vec3,
    angular_velocity_world: wp.vec3,
    hard_point_bt: wp.vec3,
    lateral_direction: wp.vec3,
    surface_normal: wp.vec3,
    handbrake_value: float,
    real_throttle: float,
    lateral_friction: wp.ref[wp.float32],
    longitudinal_friction: wp.ref[wp.float32],
): ...


_BULLET_STICKY_FORCE = r"""
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
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    auto op_sqrt = [](float value) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsqrt_rn(value);
    #else
        volatile float result = ::sqrtf(value);
        return result;
    #endif
    };
    const float x = normal_sum[0];
    const float y = normal_sum[1];
    const float z = normal_sum[2];
    const float length_squared = op_add(
        op_add(op_mul(x, x), op_mul(y, y)), op_mul(z, z));
    const float inverse_length = op_div(1.0f, op_sqrt(length_squared));
    const float nx = op_mul(x, inverse_length);
    const float ny = op_mul(y, inverse_length);
    const float nz = op_mul(z, inverse_length);
    float sticky_scale = 0.5f;
    if (full_stick != 0) {
        sticky_scale = op_add(sticky_scale, op_sub(1.0f, ::fabsf(nz)));
    }
    // Source expression is evaluated left-to-right through btVector3 scalar
    // operators: upwards * scale * (gravityUU * UU_TO_BT) * mass.
    const float gravity_bt = op_mul(-650.0f, 0.02f);
    const float sx = op_mul(op_mul(op_mul(nx, sticky_scale), gravity_bt), 180.0f);
    const float sy = op_mul(op_mul(op_mul(ny, sticky_scale), gravity_bt), 180.0f);
    const float sz = op_mul(op_mul(op_mul(nz, sticky_scale), gravity_bt), 180.0f);
    return wp::vec_t<3, wp::float32>(sx, sy, sz);
"""


@wp.func_native(_BULLET_STICKY_FORCE)
def _bullet_sticky_force(normal_sum: wp.vec3, full_stick: int) -> wp.vec3: ...


_BULLET_INTEGRATE_EXTERNAL_VELOCITIES = r"""
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
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
    const float inverse_local[3] = {
        0.0185644571f, 0.0104337428f, 0.0075815497f};
    float scaled[3][3];
    float tensor[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            scaled[row][column] = op_mul(
                basis.data[row][column], inverse_local[column]);
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            tensor[row][column] = op_add(
                op_add(
                    op_mul(scaled[row][0], basis.data[column][0]),
                    op_mul(scaled[row][1], basis.data[column][1])),
                op_mul(scaled[row][2], basis.data[column][2]));
        }
    }
    // btSequentialImpulseConstraintSolver::initSolverBody uses the
    // btVector3 * btMatrix3x3 overload here, not matrix * vector.  Its SSE
    // implementation scales the three matrix rows and adds them in x/y/z
    // order before applying the time step.
    const float external_torque_impulse[3] = {
        op_mul(op_add(op_add(
            op_mul(total_torque_bt[0], tensor[0][0]),
            op_mul(total_torque_bt[1], tensor[1][0])),
            op_mul(total_torque_bt[2], tensor[2][0])), 0.00833333377f),
        op_mul(op_add(op_add(
            op_mul(total_torque_bt[0], tensor[0][1]),
            op_mul(total_torque_bt[1], tensor[1][1])),
            op_mul(total_torque_bt[2], tensor[2][1])), 0.00833333377f),
        op_mul(op_add(op_add(
            op_mul(total_torque_bt[0], tensor[0][2]),
            op_mul(total_torque_bt[1], tensor[1][2])),
            op_mul(total_torque_bt[2], tensor[2][2])), 0.00833333377f)};
    // btSequentialImpulseConstraintSolver::initSolverBody evaluates
    // totalForce * invMass * timeStep from left to right. Factoring the two
    // scalar terms changes the rounded external impulse for some components.
    linear_velocity_bt = wp::vec_t<3, wp::float32>(
        op_add(linear_velocity_bt[0], op_mul(op_mul(total_force_bt[0], 0.00555555569f), 0.00833333377f)),
        op_add(linear_velocity_bt[1], op_mul(op_mul(total_force_bt[1], 0.00555555569f), 0.00833333377f)),
        op_add(linear_velocity_bt[2], op_mul(op_mul(total_force_bt[2], 0.00555555569f), 0.00833333377f)));
    angular_velocity_world = wp::vec_t<3, wp::float32>(
        op_add(angular_velocity_world[0], external_torque_impulse[0]),
        op_add(angular_velocity_world[1], external_torque_impulse[1]),
        op_add(angular_velocity_world[2], external_torque_impulse[2]));
"""


@wp.func_native(_BULLET_INTEGRATE_EXTERNAL_VELOCITIES)
def _bullet_integrate_external_velocities(
    basis: wp.mat33,
    total_force_bt: wp.vec3,
    total_torque_bt: wp.vec3,
    linear_velocity_bt: wp.ref[wp.vec3],
    angular_velocity_world: wp.ref[wp.vec3],
): ...


_BULLET_AIR_DAMPING_TORQUE = r"""
    // Zero-input, four-wheel-miss branch of RocketSim Car::_UpdateAirTorque.
    // This is deliberately the fixed Octane/static-world source path only:
    // dirPitch=-right, dirYaw=up, dirRoll=-forward, followed by the pinned
    // Bullet inverse-tensor inverse and matrix/vector operation order.
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
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    struct AirV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> AirV3 {
        AirV3 value = {x, y, z};
        return value;
    };
    auto add = [&](AirV3 a, AirV3 b) -> AirV3 {
        return make(op_add(a.x, b.x), op_add(a.y, b.y), op_add(a.z, b.z));
    };
    auto sub = [&](AirV3 a, AirV3 b) -> AirV3 {
        return make(op_sub(a.x, b.x), op_sub(a.y, b.y), op_sub(a.z, b.z));
    };
    auto scale = [&](AirV3 value, float scalar) -> AirV3 {
        return make(
            op_mul(value.x, scalar),
            op_mul(value.y, scalar),
            op_mul(value.z, scalar));
    };
    auto dot = [&](AirV3 a, AirV3 b) -> float {
        return op_add(
            op_add(op_mul(a.x, b.x), op_mul(a.y, b.y)),
            op_mul(a.z, b.z));
    };
    const AirV3 forward = make(
        basis.data[0][0], basis.data[1][0], basis.data[2][0]);
    const AirV3 right = make(
        basis.data[0][1], basis.data[1][1], basis.data[2][1]);
    const AirV3 up = make(
        basis.data[0][2], basis.data[1][2], basis.data[2][2]);
    const AirV3 dir_pitch = make(-right.x, -right.y, -right.z);
    const AirV3 dir_yaw = up;
    const AirV3 dir_roll = make(-forward.x, -forward.y, -forward.z);
    const AirV3 omega = make(
        angular_velocity_world[0],
        angular_velocity_world[1],
        angular_velocity_world[2]);
    const float damp_pitch = op_mul(dot(dir_pitch, omega), 30.0f);
    const float damp_yaw = op_mul(dot(dir_yaw, omega), 20.0f);
    const float damp_roll = op_mul(dot(dir_roll, omega), 50.0f);
    const AirV3 damping = add(
        add(scale(dir_yaw, damp_yaw), scale(dir_pitch, damp_pitch)),
        scale(dir_roll, damp_roll));
    const AirV3 requested = sub(make(0.0f, 0.0f, 0.0f), damping);

    // btRigidBody::updateInertiaTensor:
    // basis.scaled(invInertiaLocal) * basis.transpose().
    const float inverse_local[3] = {
        0.0185644571f, 0.0104337428f, 0.0075815497f};
    float inverse_world[3][3];
    float scaled_basis[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            scaled_basis[row][column] = op_mul(
                basis.data[row][column], inverse_local[column]);
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            inverse_world[row][column] = op_add(
                op_add(
                    op_mul(scaled_basis[row][0], basis.data[column][0]),
                    op_mul(scaled_basis[row][1], basis.data[column][1])),
                op_mul(scaled_basis[row][2], basis.data[column][2]));
        }
    }

    // btMatrix3x3::inverse, including its exact cofactor and determinant order.
    auto cofac = [&](int r1, int c1, int r2, int c2) -> float {
        return op_sub(
            op_mul(inverse_world[r1][c1], inverse_world[r2][c2]),
            op_mul(inverse_world[r1][c2], inverse_world[r2][c1]));
    };
    const float co[3] = {
        cofac(1, 1, 2, 2),
        cofac(1, 2, 2, 0),
        cofac(1, 0, 2, 1)};
    const float determinant = op_add(
        op_add(
            op_mul(inverse_world[0][0], co[0]),
            op_mul(inverse_world[0][1], co[1])),
        op_mul(inverse_world[0][2], co[2]));
    const float inverse_determinant = op_div(1.0f, determinant);
    float inertia_world[3][3];
    inertia_world[0][0] = op_mul(co[0], inverse_determinant);
    inertia_world[0][1] = op_mul(cofac(0, 2, 2, 1), inverse_determinant);
    inertia_world[0][2] = op_mul(cofac(0, 1, 1, 2), inverse_determinant);
    inertia_world[1][0] = op_mul(co[1], inverse_determinant);
    inertia_world[1][1] = op_mul(cofac(0, 0, 2, 2), inverse_determinant);
    inertia_world[1][2] = op_mul(cofac(0, 2, 1, 0), inverse_determinant);
    inertia_world[2][0] = op_mul(co[2], inverse_determinant);
    inertia_world[2][1] = op_mul(cofac(0, 1, 2, 0), inverse_determinant);
    inertia_world[2][2] = op_mul(cofac(0, 0, 1, 1), inverse_determinant);

    // btMatrix3x3 * btVector3 (dot3): each row is reduced as (x+y)+z.
    const AirV3 angular_acceleration = make(
        op_add(op_add(
            op_mul(inertia_world[0][0], requested.x),
            op_mul(inertia_world[0][1], requested.y)),
            op_mul(inertia_world[0][2], requested.z)),
        op_add(op_add(
            op_mul(inertia_world[1][0], requested.x),
            op_mul(inertia_world[1][1], requested.y)),
            op_mul(inertia_world[1][2], requested.z)),
        op_add(op_add(
            op_mul(inertia_world[2][0], requested.x),
            op_mul(inertia_world[2][1], requested.y)),
            op_mul(inertia_world[2][2], requested.z)));
    // C++ left-associativity makes the source expression
    // (inertiaWorld * requested) * CAR_TORQUE_SCALE.
    return wp::vec_t<3, wp::float32>(
        op_mul(angular_acceleration.x, 0.0958738029f),
        op_mul(angular_acceleration.y, 0.0958738029f),
        op_mul(angular_acceleration.z, 0.0958738029f));
"""


@wp.func_native(_BULLET_AIR_DAMPING_TORQUE)
def _bullet_air_damping_torque(
    basis: wp.mat33,
    angular_velocity_world: wp.vec3,
) -> wp.vec3: ...


_BULLET_INTEGRATE_POSITION = r"""
    // btSolverBody::writebackVelocityAndTransform first applies the split
    // push velocity to the stored Bullet transform. The dynamics world then
    // calls btRigidBody::predictIntegratedTransform and applies the solved
    // linear velocity to that updated transform. Keep both multiply/add
    // boundaries explicit so CUDA cannot contract either source expression.
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
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
    float x = current_position_bt[0];
    float y = current_position_bt[1];
    float z = current_position_bt[2];
    if (has_split_correction != 0) {
        x = op_add(x, op_mul(push_velocity_bt[0], time_step));
        y = op_add(y, op_mul(push_velocity_bt[1], time_step));
        z = op_add(z, op_mul(push_velocity_bt[2], time_step));
    }
    x = op_add(x, op_mul(linear_velocity_bt[0], time_step));
    y = op_add(y, op_mul(linear_velocity_bt[1], time_step));
    z = op_add(z, op_mul(linear_velocity_bt[2], time_step));
    integrated_position_bt = wp::vec_t<3, wp::float32>(x, y, z);
"""


@wp.func_native(_BULLET_INTEGRATE_POSITION)
def _bullet_integrate_position(
    current_position_bt: wp.vec3,
    push_velocity_bt: wp.vec3,
    linear_velocity_bt: wp.vec3,
    time_step: float,
    has_split_correction: int,
    integrated_position_bt: wp.ref[wp.vec3],
): ...


_BULLET_CONTACT_ROW = r"""
    // setupContactConstraint + velocity-dependent setupFrictionConstraint for
    // one Octane/static row in the pinned Windows Bullet operation order.
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
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    struct RowV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> RowV3 {
        RowV3 value = {x, y, z};
        return value;
    };
    auto add = [&](RowV3 a, RowV3 b) -> RowV3 {
        return make(op_add(a.x, b.x), op_add(a.y, b.y), op_add(a.z, b.z));
    };
    auto sub = [&](RowV3 a, RowV3 b) -> RowV3 {
        return make(op_sub(a.x, b.x), op_sub(a.y, b.y), op_sub(a.z, b.z));
    };
    auto scale = [&](RowV3 value, float amount) -> RowV3 {
        return make(op_mul(value.x, amount), op_mul(value.y, amount), op_mul(value.z, amount));
    };
    auto cross = [&](RowV3 a, RowV3 b) -> RowV3 {
        return make(
            op_sub(op_mul(a.y, b.z), op_mul(a.z, b.y)),
            op_sub(op_mul(a.z, b.x), op_mul(a.x, b.z)),
            op_sub(op_mul(a.x, b.y), op_mul(a.y, b.x)));
    };
    auto dot = [&](RowV3 a, RowV3 b) -> float {
        return op_add(op_add(op_mul(a.x, b.x), op_mul(a.y, b.y)), op_mul(a.z, b.z));
    };
    const float inverse_local[3] = {0.0185644571f, 0.0104337428f, 0.0075815497f};
    float scaled_basis[3][3];
    float tensor[3][3];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            scaled_basis[row][column] = op_mul(basis.data[row][column], inverse_local[column]);
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            tensor[row][column] = op_add(
                op_add(
                    op_mul(scaled_basis[row][0], basis.data[column][0]),
                    op_mul(scaled_basis[row][1], basis.data[column][1])),
                op_mul(scaled_basis[row][2], basis.data[column][2]));
        }
    }
    auto matrix_vector = [&](RowV3 value) -> RowV3 {
        return make(
            op_add(op_add(op_mul(tensor[0][0], value.x), op_mul(tensor[0][1], value.y)), op_mul(tensor[0][2], value.z)),
            op_add(op_add(op_mul(tensor[1][0], value.x), op_mul(tensor[1][1], value.y)), op_mul(tensor[1][2], value.z)),
            op_add(op_add(op_mul(tensor[2][0], value.x), op_mul(tensor[2][1], value.y)), op_mul(tensor[2][2], value.z)));
    };
    const RowV3 point = make(point_a_bt[0], point_a_bt[1], point_a_bt[2]);
    const RowV3 origin = make(body_origin_bt[0], body_origin_bt[1], body_origin_bt[2]);
    const RowV3 point_b = make(point_b_bt[0], point_b_bt[1], point_b_bt[2]);
    const RowV3 contact_normal = make(normal[0], normal[1], normal[2]);
    const RowV3 pre_linear = make(pre_linear_bt[0], pre_linear_bt[1], pre_linear_bt[2]);
    const RowV3 pre_angular = make(pre_angular_world[0], pre_angular_world[1], pre_angular_world[2]);
    const RowV3 force_linear = make(force_linear_bt[0], force_linear_bt[1], force_linear_bt[2]);
    const RowV3 force_angular = make(force_angular_world[0], force_angular_world[1], force_angular_world[2]);
    const RowV3 relative = sub(point, origin);
    const RowV3 normal_torque = cross(relative, contact_normal);
    const RowV3 normal_angular = matrix_vector(normal_torque);
    const float normal_denominator = op_add(
        0.00555555569f, dot(contact_normal, cross(normal_angular, relative)));
    const float normal_inverse = op_div(1.0f, normal_denominator);
    // Restitution is evaluated through btRigidBody::getVelocityInLocalPoint:
    // linearVelocity + angularVelocity.cross(rel_pos), followed by the
    // contact-normal dot.  Collapsing this to the scalar triple-product form
    // is algebraically equivalent but changes the pinned float32 reductions.
    const RowV3 pre_point_velocity = add(pre_linear, cross(pre_angular, relative));
    const float pre_speed = dot(contact_normal, pre_point_velocity);
    const float force_speed = op_add(dot(contact_normal, force_linear), dot(normal_torque, force_angular));
    float restitution = 0.0f;
    if (fabsf(pre_speed) >= 0.2f) {
        restitution = op_mul(-0.3f, pre_speed);
        if (restitution < 0.0f) restitution = 0.0f;
    }
    normal_jacobian = normal_inverse;
    normal_rhs = op_mul(op_sub(restitution, force_speed), normal_inverse);

    const RowV3 point_velocity = add(force_linear, cross(force_angular, relative));
    const float projected_speed = dot(contact_normal, point_velocity);
    RowV3 friction = sub(point_velocity, scale(contact_normal, projected_speed));
    const float friction_length_squared = dot(friction, friction);
    if (friction_length_squared > 1.1920928955078125e-7f) {
        friction = scale(friction, op_div(1.0f, sqrtf(friction_length_squared)));
    } else if (fabsf(contact_normal.z) > 0.7071067811865476f) {
        const float a = op_add(op_mul(contact_normal.y, contact_normal.y), op_mul(contact_normal.z, contact_normal.z));
        const float inverse = op_div(1.0f, sqrtf(a));
        friction = make(0.0f, op_mul(-contact_normal.z, inverse), op_mul(contact_normal.y, inverse));
    } else {
        const float a = op_add(op_mul(contact_normal.x, contact_normal.x), op_mul(contact_normal.y, contact_normal.y));
        const float inverse = op_div(1.0f, sqrtf(a));
        friction = make(op_mul(-contact_normal.y, inverse), op_mul(contact_normal.x, inverse), 0.0f);
    }
    const RowV3 friction_torque = cross(relative, friction);
    const RowV3 friction_angular = matrix_vector(friction_torque);
    const float friction_denominator = op_add(
        0.00555555569f, dot(friction, cross(friction_angular, relative)));
    const float friction_inverse = op_div(1.0f, friction_denominator);
    const float friction_speed = op_add(dot(friction, force_linear), dot(friction_torque, pre_angular));
    tangent_rhs = op_mul(-1.0f, op_mul(friction_speed, friction_inverse));
    tangent_jacobian = friction_inverse;
    tangent = wp::vec_t<3, wp::float32>(friction.x, friction.y, friction.z);

    // setupContactConstraint consumes btManifoldPoint::getDistance() rather
    // than reconstructing the penetration from the refreshed world points.
    // It also computes invTimeStep from the stored float time step; using the
    // mathematical constant 120 changes the split RHS by one float32 ULP.
    const float penetration = op_add(distance_bt, 0.0f);
    const float inverse_time_step = op_div(1.0f, time_step);
    push_rhs = 0.0f;
    if (penetration <= 0.0f) {
        float positional_error = op_mul(-penetration, 0.8f);
        positional_error = op_mul(positional_error, inverse_time_step);
        push_rhs = op_mul(positional_error, normal_inverse);
    }
"""


@wp.func_native(_BULLET_CONTACT_ROW)
def _bullet_contact_row(
    body_origin_bt: wp.vec3,
    basis: wp.mat33,
    point_a_bt: wp.vec3,
    point_b_bt: wp.vec3,
    distance_bt: float,
    time_step: float,
    normal: wp.vec3,
    pre_linear_bt: wp.vec3,
    pre_angular_world: wp.vec3,
    force_linear_bt: wp.vec3,
    force_angular_world: wp.vec3,
    tangent: wp.ref[wp.vec3],
    normal_jacobian: wp.ref[wp.float32],
    tangent_jacobian: wp.ref[wp.float32],
    normal_rhs: wp.ref[wp.float32],
    tangent_rhs: wp.ref[wp.float32],
    push_rhs: wp.ref[wp.float32],
): ...


_BULLET_SOLVE_SPLIT_ROW = r"""
    // gResolveSplitPenetrationImpulse_sse2 returns before reading or
    // modifying either solver body when m_rhsPenetration is exactly zero.
    if (rhs == 0.0f) return;
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
    struct SplitV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> SplitV3 { SplitV3 value = {x,y,z}; return value; };
    auto cross = [&](SplitV3 a, SplitV3 b) -> SplitV3 {
        return make(
            op_sub(op_mul(a.y,b.z),op_mul(a.z,b.y)),
            op_sub(op_mul(a.z,b.x),op_mul(a.x,b.z)),
            op_sub(op_mul(a.x,b.y),op_mul(a.y,b.x)));
    };
    auto dot_sse2 = [&](SplitV3 a, SplitV3 b) -> float {
        return op_add(op_mul(a.x,b.x), op_add(op_mul(a.y,b.y),op_mul(a.z,b.z)));
    };
    const SplitV3 n = make(direction[0],direction[1],direction[2]);
    const SplitV3 rel = make(relative_position_bt[0],relative_position_bt[1],relative_position_bt[2]);
    const SplitV3 torque = cross(rel,n);
    const float inverse_local[3] = {0.0185644571f,0.0104337428f,0.0075815497f};
    float scaled[3][3]; float tensor[3][3];
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) scaled[r][c]=op_mul(basis.data[r][c],inverse_local[c]);
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) tensor[r][c]=op_add(op_add(op_mul(scaled[r][0],basis.data[c][0]),op_mul(scaled[r][1],basis.data[c][1])),op_mul(scaled[r][2],basis.data[c][2]));
    const SplitV3 angular_component = make(
        op_add(op_add(op_mul(tensor[0][0],torque.x),op_mul(tensor[0][1],torque.y)),op_mul(tensor[0][2],torque.z)),
        op_add(op_add(op_mul(tensor[1][0],torque.x),op_mul(tensor[1][1],torque.y)),op_mul(tensor[1][2],torque.z)),
        op_add(op_add(op_mul(tensor[2][0],torque.x),op_mul(tensor[2][1],torque.y)),op_mul(tensor[2][2],torque.z)));
    SplitV3 push = make(push_velocity_bt[0],push_velocity_bt[1],push_velocity_bt[2]);
    SplitV3 turn = make(turn_velocity_world[0],turn_velocity_world[1],turn_velocity_world[2]);
    const float speed = op_add(dot_sse2(n,push),dot_sse2(torque,turn));
    float delta = op_sub(rhs,op_mul(speed,jacobian));
    float sum = op_add(applied_push_impulse,delta);
    if(sum<0.0f){delta=-applied_push_impulse;sum=0.0f;}
    applied_push_impulse=sum;
    push.x=op_add(push.x,op_mul(op_mul(n.x,0.00555555569f),delta));
    push.y=op_add(push.y,op_mul(op_mul(n.y,0.00555555569f),delta));
    push.z=op_add(push.z,op_mul(op_mul(n.z,0.00555555569f),delta));
    turn.x=op_add(turn.x,op_mul(angular_component.x,delta));
    turn.y=op_add(turn.y,op_mul(angular_component.y,delta));
    turn.z=op_add(turn.z,op_mul(angular_component.z,delta));
    push_velocity_bt=wp::vec_t<3,wp::float32>(push.x,push.y,push.z);
    turn_velocity_world=wp::vec_t<3,wp::float32>(turn.x,turn.y,turn.z);
"""


@wp.func_native(_BULLET_SOLVE_SPLIT_ROW)
def _bullet_solve_split_row(
    basis: wp.mat33,
    direction: wp.vec3,
    relative_position_bt: wp.vec3,
    jacobian: float,
    rhs: float,
    push_velocity_bt: wp.ref[wp.vec3],
    turn_velocity_world: wp.ref[wp.vec3],
    applied_push_impulse: wp.ref[wp.float32],
): ...


_BULLET_SOLVE_VELOCITY_ROW = r"""
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
    struct VelV3{float x;float y;float z;};
    auto make=[](float x,float y,float z)->VelV3{VelV3 v={x,y,z};return v;};
    auto cross=[&](VelV3 a,VelV3 b)->VelV3{return make(op_sub(op_mul(a.y,b.z),op_mul(a.z,b.y)),op_sub(op_mul(a.z,b.x),op_mul(a.x,b.z)),op_sub(op_mul(a.x,b.y),op_mul(a.y,b.x)));};
    auto dot=[&](VelV3 a,VelV3 b)->float{return op_add(op_add(op_mul(a.x,b.x),op_mul(a.y,b.y)),op_mul(a.z,b.z));};
    const VelV3 n=make(direction[0],direction[1],direction[2]);
    const VelV3 rel=make(relative_position_bt[0],relative_position_bt[1],relative_position_bt[2]);
    const VelV3 torque=cross(rel,n);
    const float inverse_local[3]={0.0185644571f,0.0104337428f,0.0075815497f};
    float scaled[3][3];float tensor[3][3];
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)scaled[r][c]=op_mul(basis.data[r][c],inverse_local[c]);
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)tensor[r][c]=op_add(op_add(op_mul(scaled[r][0],basis.data[c][0]),op_mul(scaled[r][1],basis.data[c][1])),op_mul(scaled[r][2],basis.data[c][2]));
    const VelV3 angular_component=make(
        op_add(op_add(op_mul(tensor[0][0],torque.x),op_mul(tensor[0][1],torque.y)),op_mul(tensor[0][2],torque.z)),
        op_add(op_add(op_mul(tensor[1][0],torque.x),op_mul(tensor[1][1],torque.y)),op_mul(tensor[1][2],torque.z)),
        op_add(op_add(op_mul(tensor[2][0],torque.x),op_mul(tensor[2][1],torque.y)),op_mul(tensor[2][2],torque.z)));
    VelV3 linear=make(delta_linear_bt[0],delta_linear_bt[1],delta_linear_bt[2]);
    VelV3 angular=make(delta_angular_world[0],delta_angular_world[1],delta_angular_world[2]);
    const float speed=op_add(dot(n,linear),dot(torque,angular));
    float delta=op_fma(-speed,jacobian,rhs);
    float sum=op_add(applied_impulse,delta);
    if(sum<lower_limit){delta=op_sub(lower_limit,applied_impulse);sum=lower_limit;}
    else if(sum>upper_limit){delta=op_sub(upper_limit,applied_impulse);sum=upper_limit;}
    applied_impulse=sum;
    linear.x=op_fma(op_mul(n.x,0.00555555569f),delta,linear.x);
    linear.y=op_fma(op_mul(n.y,0.00555555569f),delta,linear.y);
    linear.z=op_fma(op_mul(n.z,0.00555555569f),delta,linear.z);
    angular.x=op_fma(angular_component.x,delta,angular.x);
    angular.y=op_fma(angular_component.y,delta,angular.y);
    angular.z=op_fma(angular_component.z,delta,angular.z);
    delta_linear_bt=wp::vec_t<3,wp::float32>(linear.x,linear.y,linear.z);
    delta_angular_world=wp::vec_t<3,wp::float32>(angular.x,angular.y,angular.z);
"""


@wp.func_native(_BULLET_SOLVE_VELOCITY_ROW)
def _bullet_solve_velocity_row(
    basis: wp.mat33,
    direction: wp.vec3,
    relative_position_bt: wp.vec3,
    jacobian: float,
    rhs: float,
    lower_limit: float,
    upper_limit: float,
    delta_linear_bt: wp.ref[wp.vec3],
    delta_angular_world: wp.ref[wp.vec3],
    applied_impulse: wp.ref[wp.float32],
): ...



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
        (row0[0] * matrix[0, 0] + row0[1] * matrix[0, 1]) + row0[2] * matrix[0, 2],
        (row0[0] * matrix[1, 0] + row0[1] * matrix[1, 1]) + row0[2] * matrix[1, 2],
        (row0[0] * matrix[2, 0] + row0[1] * matrix[2, 1]) + row0[2] * matrix[2, 2],
        (row1[0] * matrix[0, 0] + row1[1] * matrix[0, 1]) + row1[2] * matrix[0, 2],
        (row1[0] * matrix[1, 0] + row1[1] * matrix[1, 1]) + row1[2] * matrix[1, 2],
        (row1[0] * matrix[2, 0] + row1[1] * matrix[2, 1]) + row1[2] * matrix[2, 2],
        (row2[0] * matrix[0, 0] + row2[1] * matrix[0, 1]) + row2[2] * matrix[0, 2],
        (row2[0] * matrix[1, 0] + row2[1] * matrix[1, 1]) + row2[2] * matrix[1, 2],
        (row2[0] * matrix[2, 0] + row2[1] * matrix[2, 1]) + row2[2] * matrix[2, 2],
    )
    transpose_result = wp.vec3(
        (value[0] * tensor[0, 0] + value[1] * tensor[1, 0]) + value[2] * tensor[2, 0],
        (value[0] * tensor[0, 1] + value[1] * tensor[1, 1]) + value[2] * tensor[2, 1],
        (value[0] * tensor[0, 2] + value[1] * tensor[1, 2]) + value[2] * tensor[2, 2],
    )
    direct_result = wp.vec3(
        (tensor[0, 0] * value[0] + tensor[0, 1] * value[1]) + tensor[0, 2] * value[2],
        (tensor[1, 0] * value[0] + tensor[1, 1] * value[1]) + tensor[1, 2] * value[2],
        (tensor[2, 0] * value[0] + tensor[2, 1] * value[1]) + tensor[2, 2] * value[2],
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


@wp.kernel(enable_backward=False)
def wheel_pre_tick(
    tick_counter: wp.array(dtype=wp.int32),
    ray_mesh_id: wp.uint64,
    vertices_bt: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    bullet_face_normals: wp.array(dtype=wp.vec3),
    bullet_bvh_rank: wp.array(dtype=wp.int32),
    face_mesh_index: wp.array(dtype=wp.int32),
    enable_forces: int,
    enable_ball_rays: int,
    amd_rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    ball_position_bt: wp.array(dtype=wp.vec3),
    ball_quat: wp.array(dtype=wp.quat),
    ball_velocity_bt: wp.array(dtype=wp.vec3),
    ball_ang_vel: wp.array(dtype=wp.vec3),
    ball_broadphase_proxy_min_bt: wp.array(dtype=wp.vec3),
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
    total_force_bt: wp.array(dtype=wp.vec3),
    total_torque_bt: wp.array(dtype=wp.vec3),
    inverse_inertia_world: wp.array(dtype=wp.mat33),
    previous_contact_count: wp.array(dtype=wp.int32),
    previous_world_contact_normal: wp.array(dtype=wp.vec3),
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
    wheel_hit_point_bt: wp.array(dtype=wp.vec3),
    wheel_hit_normal: wp.array(dtype=wp.vec3),
    wheel_hit_distance: wp.array(dtype=wp.float32),
    wheel_hit_face: wp.array(dtype=wp.int32),
    suspension_length: wp.array(dtype=wp.float32),
    suspension_velocity: wp.array(dtype=wp.float32),
    suspension_clipped_factor: wp.array(dtype=wp.float32),
    suspension_force: wp.array(dtype=wp.float32),
    suspension_pushback: wp.array(dtype=wp.float32),
    suspension_force_bt: wp.array(dtype=wp.float32),
    suspension_pushback_bt: wp.array(dtype=wp.float32),
    debug_wheel_ray_from_bt: wp.array(dtype=wp.vec3),
    debug_wheel_ray_to_bt: wp.array(dtype=wp.vec3),
    debug_wheel_ray_fraction: wp.array(dtype=wp.float32),
    debug_wheel_linear_bt: wp.array(dtype=wp.vec3),
    debug_wheel_angular: wp.array(dtype=wp.vec3),
    wheel_axle: wp.array(dtype=wp.vec3),
    wheel_forward: wp.array(dtype=wp.vec3),
    wheel_friction_impulse: wp.array(dtype=wp.vec3),
    wheel_friction_impulse_bt: wp.array(dtype=wp.vec3),
    wheel_friction_relative_bt: wp.array(dtype=wp.vec3),
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
    base_vel_bt = rigid_velocity_bt[car]
    solver_orientation[car] = quat
    auto_roll_acceleration[car] = wp.vec3(0.0, 0.0, 0.0)
    auto_roll_angular_acceleration[car] = wp.vec3(0.0, 0.0, 0.0)
    total_force_bt[car] = wp.vec3(0.0, 0.0, 0.0)
    total_torque_bt[car] = wp.vec3(0.0, 0.0, 0.0)
    bullet_basis = _bullet_quaternion_matrix(quat)
    if tick_counter[0] == 0:
        bullet_basis = _authority_input_quaternion_matrix(quat)
    inverse_inertia_world[car] = _bullet_inverse_inertia_world(bullet_basis)
    up = _bullet_transform_point(
        wp.vec3(0.0, 0.0, 0.0), bullet_basis, wp.vec3(0.0, 0.0, 1.0)
    )
    forward = _bullet_transform_point(
        wp.vec3(0.0, 0.0, 0.0), bullet_basis, wp.vec3(1.0, 0.0, 0.0)
    )
    right = _bullet_transform_point(
        wp.vec3(0.0, 0.0, 0.0), bullet_basis, wp.vec3(0.0, 1.0, 0.0)
    )
    forward_speed = wp.dot(base_vel, forward)
    abs_speed = wp.abs(forward_speed)
    contact_count = 0
    normal_sum = wp.vec3(0.0, 0.0, 0.0)
    world_contact_count = 0
    world_normal_sum = wp.vec3(0.0, 0.0, 0.0)
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
        rest_bt = rest * 0.02
        radius_bt = radius * 0.02
        travel_bt = (MAX_SUSPENSION_TRAVEL * 0.02 * 100.0) / 100.0
        ray_length_bt = rest_bt + travel_bt + radius_bt - 0.05
        source_bt = _bullet_transform_point(
            rigid_position_bt[car], bullet_basis, connection * 0.02
        )
        direction = -up
        target_bt = _bullet_vector_scale_add(source_bt, direction, ray_length_bt)
        distance_bt = ray_length_bt + 0.02
        normal = wp.vec3(0.0, 0.0, 0.0)
        hit_face = wp.int32(-1)
        source_fraction = wp.float32(0.0)
        source_point_bt = wp.vec3(0.0, 0.0, 0.0)
        source_normal = wp.vec3(0.0, 0.0, 0.0)
        source_valid = wp.int32(0)
        hit_dynamic = wp.int32(0)
        dynamic_ground_basis = bullet_basis
        # Warp's closest-hit ray test can select a different face before the
        # retained face is reconstructed with Bullet arithmetic.  Gather the
        # short ray's AABB candidates, run the pinned
        # btTriangleRaycastCallback operation stream on every face, and retain
        # the exact lowest fraction.  Equal fractions retain Bullet's source
        # collision-body/BVH visit order; no geometric tolerance is involved.
        ray_min = wp.vec3(
            wp.min(source_bt[0], target_bt[0]),
            wp.min(source_bt[1], target_bt[1]),
            wp.min(source_bt[2], target_bt[2]),
        )
        ray_max = wp.vec3(
            wp.max(source_bt[0], target_bt[0]),
            wp.max(source_bt[1], target_bt[1]),
            wp.max(source_bt[2], target_bt[2]),
        )
        closest_fraction = wp.float32(1.0)
        closest_mesh = wp.int32(0x7FFFFFFF)
        closest_rank = wp.int32(0x7FFFFFFF)
        ray_candidates = wp.mesh_query_aabb(ray_mesh_id, ray_min, ray_max)
        for candidate_face in ray_candidates:
            triangle_offset = candidate_face * 3
            candidate_fraction = wp.float32(0.0)
            candidate_point_bt = wp.vec3(0.0, 0.0, 0.0)
            candidate_normal = wp.vec3(0.0, 0.0, 0.0)
            candidate_valid = wp.int32(0)
            _bullet_selected_triangle_raycast(
                source_bt,
                target_bt,
                vertices_bt[triangle_indices[triangle_offset]],
                vertices_bt[triangle_indices[triangle_offset + 1]],
                vertices_bt[triangle_indices[triangle_offset + 2]],
                bullet_face_normals[candidate_face],
                candidate_fraction,
                candidate_point_bt,
                candidate_normal,
                candidate_valid,
            )
            candidate_mesh = face_mesh_index[candidate_face]
            candidate_rank = bullet_bvh_rank[candidate_face]
            if candidate_valid != 0 and (
                candidate_fraction < closest_fraction
                or (
                    candidate_fraction == closest_fraction
                    and (
                        candidate_mesh < closest_mesh
                        or (
                            candidate_mesh == closest_mesh
                            and candidate_rank < closest_rank
                        )
                    )
                )
            ):
                closest_fraction = candidate_fraction
                closest_mesh = candidate_mesh
                closest_rank = candidate_rank
                hit_face = candidate_face
                source_fraction = candidate_fraction
                source_point_bt = candidate_point_bt
                source_normal = candidate_normal
                source_valid = wp.int32(1)
        if source_valid != 0:
            distance_bt = ray_length_bt * source_fraction
            normal = source_normal
        # RocketSim's four static planes are concave shapes. Bullet creates two
        # ray-AABB-sized triangles for each plane and runs the ordinary
        # btTriangleRaycastCallback; an algebraic plane intersection changes
        # both the retained fraction and normal by reachable float32 ULPs.
        for plane in range(4):
            plane_origin_bt = wp.vec3(0.0, 0.0, 0.0)
            plane_normal_input = wp.vec3(0.0, 0.0, 1.0)
            plane_face = wp.int32(-2)
            if plane == 1:
                plane_origin_bt = wp.vec3(0.0, 0.0, SOCCAR_HEIGHT * 0.02)
                plane_normal_input = wp.vec3(0.0, 0.0, -1.0)
                plane_face = wp.int32(-3)
            elif plane == 2:
                plane_origin_bt = wp.vec3(
                    -SOCCAR_EXTENT_X * 0.02,
                    0.0,
                    SOCCAR_HEIGHT * 0.01,
                )
                plane_normal_input = wp.vec3(1.0, 0.0, 0.0)
                plane_face = wp.int32(-4)
            elif plane == 3:
                plane_origin_bt = wp.vec3(
                    SOCCAR_EXTENT_X * 0.02,
                    0.0,
                    SOCCAR_HEIGHT * 0.01,
                )
                plane_normal_input = wp.vec3(-1.0, 0.0, 0.0)
                plane_face = wp.int32(-5)
            plane_fraction = wp.float32(0.0)
            plane_hit_point_bt = wp.vec3(0.0, 0.0, 0.0)
            plane_hit_normal = wp.vec3(0.0, 0.0, 0.0)
            plane_valid = wp.int32(0)
            _bullet_static_plane_raycast(
                source_bt,
                target_bt,
                plane_origin_bt,
                plane_normal_input,
                closest_fraction,
                plane_fraction,
                plane_hit_point_bt,
                plane_hit_normal,
                plane_valid,
            )
            if plane_valid != 0:
                closest_fraction = plane_fraction
                source_fraction = plane_fraction
                source_point_bt = plane_hit_point_bt
                source_normal = plane_hit_normal
                source_valid = wp.int32(1)
                hit_face = plane_face
        # btRSBroadphase's short-ray path visits every dynamic handle resident
        # in the ray-origin cell, without a per-proxy AABB test. Dynamic
        # handles occupy the 3x3x3 neighborhood around the cell selected from
        # their cached AABB minimum. The cache is intentionally one Bullet
        # updateAabbs phase behind this vehicle pre-tick.
        dynamic_ball_visible = wp.int32(0)
        if enable_ball_rays != 0 and car % 2 == 0:
            env = car // 2
            proxy_min_bt = ball_broadphase_proxy_min_bt[env]
            ray_cell_i = wp.int32((source_bt[0] + 112.0) / 7.4)
            ray_cell_j = wp.int32((source_bt[1] + 120.0) / 7.4)
            ray_cell_k = wp.int32(source_bt[2] / 7.4)
            proxy_cell_i = wp.int32((proxy_min_bt[0] + 112.0) / 7.4)
            proxy_cell_j = wp.int32((proxy_min_bt[1] + 120.0) / 7.4)
            proxy_cell_k = wp.int32(proxy_min_bt[2] / 7.4)
            ray_cell_i = wp.max(0, wp.min(30, ray_cell_i))
            ray_cell_j = wp.max(0, wp.min(32, ray_cell_j))
            ray_cell_k = wp.max(0, wp.min(5, ray_cell_k))
            proxy_cell_i = wp.max(0, wp.min(30, proxy_cell_i))
            proxy_cell_j = wp.max(0, wp.min(32, proxy_cell_j))
            proxy_cell_k = wp.max(0, wp.min(5, proxy_cell_k))
            if (
                wp.abs(ray_cell_i - proxy_cell_i) <= 1
                and wp.abs(ray_cell_j - proxy_cell_j) <= 1
                and wp.abs(ray_cell_k - proxy_cell_k) <= 1
            ):
                dynamic_ball_visible = wp.int32(1)
        if dynamic_ball_visible != 0:
            env = car // 2
            ball_basis = _bullet_quaternion_matrix(ball_quat[env])
            dynamic_ground_basis = ball_basis
            ball_fraction = wp.float32(0.0)
            ball_hit_point_bt = wp.vec3(0.0, 0.0, 0.0)
            ball_hit_normal = wp.vec3(0.0, 0.0, 0.0)
            ball_valid = wp.int32(0)
            _bullet_ray_sphere(
                source_bt,
                target_bt,
                ball_position_bt[env],
                ball_basis,
                1.8249999284744263,
                closest_fraction,
                amd_rsqrtss_mantissa,
                ball_fraction,
                ball_hit_point_bt,
                ball_hit_normal,
                ball_valid,
            )
            if ball_valid != 0:
                closest_fraction = ball_fraction
                source_fraction = ball_fraction
                source_point_bt = ball_hit_point_bt
                source_normal = ball_hit_normal
                source_valid = wp.int32(1)
                hit_dynamic = wp.int32(1)
                hit_face = wp.int32(-6)
        if source_valid != 0:
            distance_bt = ray_length_bt * source_fraction
            normal = source_normal
        hit = source_valid != 0
        hit_point_bt = target_bt
        if source_valid != 0:
            hit_point_bt = source_point_bt
        source = source_bt * 50.0
        distance = distance_bt * 50.0
        hit_point = hit_point_bt * 50.0
        sus_length = wp.float32(rest + MAX_SUSPENSION_TRAVEL)
        sus_velocity = wp.float32(0.0)
        clipped = wp.float32(1.0)
        # btVehicleRL only clears m_extraPushback when the ray misses.  While a
        # wheel remains on a static object, a value produced by an earlier
        # below-threshold trace persists until another below-threshold solve
        # replaces it.
        pushback = wp.float32(suspension_pushback[wheel_index])
        prior_pushback_bt = wp.float32(suspension_pushback_bt[wheel_index])
        force_value = wp.float32(0.0)
        exact_force_bt = wp.float32(0.0)
        exact_pushback_bt = wp.float32(0.0)

        if hit:
            contact_count = contact_count + 1
            normal_sum = normal_sum + normal
            if hit_dynamic == 0:
                world_contact_count = world_contact_count + 1
                world_normal_sum = world_normal_sum + normal
            sus_length_bt = wp.float32(0.0)
            sus_velocity_bt = wp.float32(0.0)
            force_value_bt = wp.float32(0.0)
            pushback_bt = wp.float32(0.0)
            _bullet_wheel_suspension(
                source_bt,
                hit_point_bt,
                normal,
                up,
                rigid_position_bt[car],
                base_vel_bt,
                base_ang_vel,
                bullet_basis,
                rest_bt,
                travel_bt,
                radius_bt,
                force_scale,
                solver_dt,
                prior_pushback_bt,
                hit_dynamic,
                sus_length_bt,
                sus_velocity_bt,
                clipped,
                force_value_bt,
                pushback_bt,
            )
            sus_length = sus_length_bt * 50.0
            sus_velocity = sus_velocity_bt * 50.0
            force_value = force_value_bt * 50.0
            pushback = pushback_bt * 50.0
            exact_force_bt = force_value_bt
            exact_pushback_bt = pushback_bt
            contact_offset = hit_point - pos

            old_steer = steer_angle[wheel_index]
            raw_axle = right * wp.cos(old_steer) - forward * wp.sin(old_steer)
            axle = wp.vec3(0.0, 0.0, 0.0)
            forward_at_wheel = wp.vec3(0.0, 0.0, 0.0)
            friction_relative_bt = wp.vec3(0.0, 0.0, 0.0)
            friction_impulse_bt = wp.vec3(0.0, 0.0, 0.0)
            exact_side_bt = wp.float32(0.0)
            exact_rolling_bt = wp.float32(0.0)
            relative = base_vel + wp.cross(base_ang_vel, contact_offset)

            # Bullet resolveSingleBilateral uses a 0.2 contact damping term and
            # the complete angular effective mass at the wheel contact.
            lateral_speed = wp.dot(relative, axle)
            side_denominator = _impulse_denominator(quat, contact_offset, axle, 1.0, 0)
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
            # engine/brake values.  Engine force is converted from the legacy
            # UU cache; brake force is already cached in Bullet units so the
            # float32 multiplication by 52.5 happens in RocketSim's order.
            old_engine = engine_acceleration[wheel_index]
            old_brake = brake_acceleration[wheel_index]
            _bullet_wheel_friction(
                bullet_basis,
                dynamic_ground_basis,
                rigid_position_bt[car],
                base_vel_bt,
                base_ang_vel,
                hit_dynamic,
                ball_position_bt[car // 2],
                ball_velocity_bt[car // 2],
                ball_ang_vel[car // 2],
                hit_point_bt,
                normal,
                raw_axle,
                old_engine * 3.6,
                old_brake,
                lateral_friction[wheel_index],
                longitudinal_friction[wheel_index],
                DT,
                axle,
                forward_at_wheel,
                friction_relative_bt,
                friction_impulse_bt,
                exact_side_bt,
                exact_rolling_bt,
            )
            side_value = exact_side_bt * 25.0
            rolling_value = exact_rolling_bt * 25.0
            cached_impulse = friction_impulse_bt * 50.0
            wheel_axle[wheel_index] = axle
            wheel_forward[wheel_index] = forward_at_wheel
            wheel_friction_impulse[wheel_index] = cached_impulse
            wheel_friction_impulse_bt[wheel_index] = friction_impulse_bt
            wheel_friction_relative_bt[wheel_index] = friction_relative_bt
            side_impulse[wheel_index] = side_value
            rolling_impulse[wheel_index] = rolling_value
        else:
            pushback = 0.0
            exact_pushback_bt = 0.0
            wheel_axle[wheel_index] = wp.vec3(0.0, 0.0, 0.0)
            wheel_forward[wheel_index] = wp.vec3(0.0, 0.0, 0.0)
            wheel_friction_impulse[wheel_index] = wp.vec3(0.0, 0.0, 0.0)
            wheel_friction_impulse_bt[wheel_index] = wp.vec3(0.0, 0.0, 0.0)
            wheel_friction_relative_bt[wheel_index] = wp.vec3(0.0, 0.0, 0.0)
            side_impulse[wheel_index] = 0.0
            rolling_impulse[wheel_index] = 0.0

        wheel_ray_start[wheel_index] = source
        wheel_direction[wheel_index] = direction
        wheel_hit_point[wheel_index] = hit_point
        wheel_hit_point_bt[wheel_index] = hit_point_bt
        wheel_hit_normal[wheel_index] = normal
        wheel_hit_distance[wheel_index] = distance
        wheel_hit_face[wheel_index] = hit_face
        suspension_length[wheel_index] = sus_length
        suspension_velocity[wheel_index] = sus_velocity
        suspension_clipped_factor[wheel_index] = clipped
        suspension_force[wheel_index] = force_value
        suspension_pushback[wheel_index] = pushback
        suspension_force_bt[wheel_index] = exact_force_bt
        suspension_pushback_bt[wheel_index] = exact_pushback_bt
        debug_wheel_ray_from_bt[wheel_index] = source_bt
        debug_wheel_ray_to_bt[wheel_index] = target_bt
        debug_fraction = distance_bt / ray_length_bt
        if source_valid != 0:
            debug_fraction = source_fraction
        debug_wheel_ray_fraction[wheel_index] = debug_fraction
        wheel_contact[wheel_index] = wp.int32(hit)
        wheel_world_contact[wheel_index] = wp.int32(hit and hit_dynamic == 0)

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
        # RocketSim computes
        #   realBrake * ((180.f * (14.25f + 1.f / 3.f)) * .02f)
        # whose parenthesized float32 result is exactly 52.5f.  Multiplying
        # directly here matters: the former equivalent-looking UU conversion
        # rounded the coasting force one ULP lower.
        new_brake = real_brake * 52.5
        new_steer = _steer_curve(abs_speed)
        if handbrake > 0.0:
            new_steer = new_steer + (_powerslide_steer_curve(abs_speed) - new_steer) * handbrake
        new_steer = new_steer * wp.clamp(control_steer[car], -1.0, 1.0)

        for wheel in range(4):
            wheel_index = car * 4 + wheel
            old_steer = steer_angle[wheel_index]
            if wheel_contact[wheel_index] != 0:
                # Car::_UpdateWheels evaluates its slip curve with the raw
                # wheel-transform right axis.  calcFrictionImpulses separately
                # projects that axis onto the contact plane; reusing the
                # projected solver axle here overstates grip on a tilted car.
                lateral_direction = right * wp.cos(old_steer) - forward * wp.sin(old_steer)
                lat = wp.float32(0.0)
                long_factor = wp.float32(0.0)
                _bullet_wheel_friction_coefficients(
                    rigid_position_bt[car],
                    base_vel_bt,
                    base_ang_vel,
                    debug_wheel_ray_from_bt[wheel_index],
                    lateral_direction,
                    wheel_hit_normal[wheel_index],
                    handbrake,
                    real_throttle,
                    lat,
                    long_factor,
                )
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
        vel_bt = base_vel_bt
        ang_vel = base_ang_vel
        for wheel in range(4):
            wheel_index = car * 4 + wheel
            if wheel_contact[wheel_index] != 0 and suspension_force[wheel_index] != 0.0:
                suspension_impulse_bt = _bullet_suspension_impulse(
                    wheel_hit_normal[wheel_index],
                    suspension_force_bt[wheel_index],
                    DT,
                    suspension_pushback_bt[wheel_index],
                )
                suspension_offset_bt = (
                    wheel_hit_point_bt[wheel_index] - rigid_position_bt[car]
                )
                _bullet_apply_impulse(
                    bullet_basis,
                    suspension_impulse_bt,
                    suspension_offset_bt,
                    vel_bt,
                    ang_vel,
                )
            debug_wheel_linear_bt[wheel_index] = vel_bt
            debug_wheel_angular[wheel_index] = ang_vel

        for wheel in range(4):
            wheel_index = car * 4 + wheel
            friction = wheel_friction_impulse_bt[wheel_index]
            if wp.dot(friction, friction) > 0.0:
                _bullet_apply_impulse(
                    bullet_basis,
                    friction,
                    wheel_friction_relative_bt[wheel_index],
                    vel_bt,
                    ang_vel,
                )

        vel = vel_bt * 50.0

        # Car::_UpdateAutoRoll consumes the previous tick's chassis contact (or
        # the current wheel-normal average), then queues a central force and a
        # torque for Bullet's external-force integration.
        previous_world_contact = (
            wp.dot(
                previous_world_contact_normal[car],
                previous_world_contact_normal[car],
            )
            > 0.5
        )
        if control_throttle[car] != 0.0 and (
            (contact_count > 0 and contact_count < 4) or previous_world_contact
        ):
            ground_up = wp.vec3(0.0, 0.0, 0.0)
            if contact_count > 0:
                ground_up = wp.normalize(normal_sum)
            else:
                ground_up = previous_world_contact_normal[car]
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
        rigid_velocity_bt[car] = vel_bt
        queued_force_bt = wp.vec3(0.0, 0.0, 0.0)
        if world_contact_count > 0:
            # Car::_UpdateWheels gates sticky force on
            # m_isInContactWithWorld. Once that gate passes,
            # btVehicleRL::getUpwardsDirFromWheelContacts averages every
            # m_isInContact wheel normal, including dynamic ball hits.
            # Dynamic hits alone do not queue sticky force, but they do
            # participate in its direction when any world wheel is present.
            upwards = wp.normalize(normal_sum)
            sticky_scale = 0.5
            if real_throttle != 0.0 or abs_speed > STOPPING_FORWARD_VEL:
                sticky_scale = sticky_scale + 1.0 - wp.abs(upwards[2])
            vel = vel + upwards * (-650.0 * sticky_scale * DT)
            queued_force_bt = _bullet_sticky_force(
                normal_sum,
                wp.int32(
                    real_throttle != 0.0 or abs_speed > STOPPING_FORWARD_VEL
                ),
            )
        if (
            auto_roll_acceleration[car][0] != 0.0
            or auto_roll_acceleration[car][1] != 0.0
            or auto_roll_acceleration[car][2] != 0.0
        ):
            queued_force_bt = (
                queued_force_bt + auto_roll_acceleration[car] * (0.02 * CAR_MASS)
            )
        total_force_bt[car] = queued_force_bt
        if contact_count == 0:
            # Car::_UpdateAirTorque is called with updateAirControl=true only
            # when all four suspension rays miss. The frozen static-world
            # corpus has zero pitch/yaw/roll input, so the source branch is
            # exactly angular damping with no control torque.
            total_torque_bt[car] = _bullet_air_damping_torque(
                bullet_basis, ang_vel
            )
        on_ground[car] = wp.int32(contact_count >= 3)
        car_vel[car] = vel
        car_ang_vel[car] = ang_vel
    else:
        air_control_disabled[car] = 0
        solver_velocity[car] = base_vel
        solver_angular_velocity[car] = base_ang_vel
    wheels_with_contact[car] = contact_count


@wp.func
def _contact_core_support_point(pos: wp.vec3, quat: wp.quat, normal: wp.vec3) -> wp.vec3:
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


@wp.struct
class _SatPenetrationWitness:
    point_a: wp.vec3
    point_b: wp.vec3
    normal: wp.vec3
    distance: wp.float32


@wp.func
def _deep_sat_penetration_witness(
    pos: wp.vec3,
    quat: wp.quat,
    normal: wp.vec3,
    penetration: float,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
) -> _SatPenetrationWitness:
    """Recover Bullet's rounded-box EPA witness from the SAT direction.

    Move the core box by the outer-box SAT depth, find the closest core/triangle
    feature pair, then subtract the spherical box margin.  The residual
    core/triangle gap corrects cross-axis SAT depth to the rounded btBoxShape
    penetration used by Bullet's EPA solver.
    """

    result = _SatPenetrationWitness()
    center = pos + wp.quat_rotate(quat, HITBOX_OFFSET)
    shifted_center = center + normal * penetration
    best_distance_sq = float(1.0e30)  # noqa: UP018 - Warp mutable function local
    best_box = shifted_center
    best_triangle = v0

    # Triangle vertices against the shifted core box.
    for triangle_vertex in range(3):
        vertex = v0
        if triangle_vertex == 1:
            vertex = v1
        elif triangle_vertex == 2:
            vertex = v2
        local = wp.quat_rotate_inv(quat, vertex - shifted_center)
        local_box = wp.vec3(
            wp.clamp(local[0], -HITBOX_CORE_HALF[0], HITBOX_CORE_HALF[0]),
            wp.clamp(local[1], -HITBOX_CORE_HALF[1], HITBOX_CORE_HALF[1]),
            wp.clamp(local[2], -HITBOX_CORE_HALF[2], HITBOX_CORE_HALF[2]),
        )
        box_point = shifted_center + wp.quat_rotate(quat, local_box)
        delta = box_point - vertex
        distance_sq = wp.dot(delta, delta)
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_box = box_point
            best_triangle = vertex

    # Core-box vertices against the triangle interior and edges.
    for vertex_index in range(8):
        local_box = wp.vec3(
            HITBOX_CORE_HALF[0],
            HITBOX_CORE_HALF[1],
            HITBOX_CORE_HALF[2],
        )
        if (vertex_index & 1) == 0:
            local_box[0] = -local_box[0]
        if (vertex_index & 2) == 0:
            local_box[1] = -local_box[1]
        if (vertex_index & 4) == 0:
            local_box[2] = -local_box[2]
        box_point = shifted_center + wp.quat_rotate(quat, local_box)
        triangle_point = _closest_point_triangle(box_point, v0, v1, v2)
        delta = box_point - triangle_point
        distance_sq = wp.dot(delta, delta)
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_box = box_point
            best_triangle = triangle_point

    # Every core-box edge against every triangle edge.
    for box_axis in range(3):
        for sign_case in range(4):
            local_start = wp.vec3(
                HITBOX_CORE_HALF[0],
                HITBOX_CORE_HALF[1],
                HITBOX_CORE_HALF[2],
            )
            local_end = local_start
            other_bit = 0
            for axis in range(3):
                if axis == box_axis:
                    local_start[axis] = -HITBOX_CORE_HALF[axis]
                    local_end[axis] = HITBOX_CORE_HALF[axis]
                else:
                    sign = 1.0
                    if (sign_case & (1 << other_bit)) == 0:
                        sign = -1.0
                    local_start[axis] = HITBOX_CORE_HALF[axis] * sign
                    local_end[axis] = HITBOX_CORE_HALF[axis] * sign
                    other_bit = other_bit + 1
            box_start = shifted_center + wp.quat_rotate(quat, local_start)
            box_end = shifted_center + wp.quat_rotate(quat, local_end)
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
                box_point = box_start + (box_end - box_start) * parameters[0]
                triangle_point = triangle_start + (triangle_end - triangle_start) * parameters[1]
                delta = box_point - triangle_point
                distance_sq = wp.dot(delta, delta)
                if distance_sq < best_distance_sq:
                    best_distance_sq = distance_sq
                    best_box = box_point
                    best_triangle = triangle_point

    gap = wp.sqrt(wp.max(0.0, best_distance_sq))
    corrected_penetration = wp.max(
        0.0,
        penetration + HITBOX_MARGIN - gap,
    )
    result.point_a = best_box - normal * penetration - normal * HITBOX_MARGIN
    result.point_b = best_triangle
    result.normal = normal
    result.distance = -corrected_penetration
    return result


@wp.func
def _deep_rounded_penetration_witness(
    pos: wp.vec3,
    quat: wp.quat,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
) -> _SatPenetrationWitness:
    """Choose the minimum rounded-box depth across box/triangle SAT axes."""

    result = _SatPenetrationWitness()
    result.point_a = pos
    result.point_b = v0
    result.normal = wp.vec3(0.0, 0.0, 1.0)
    result.distance = -1.0e30
    center = pos + wp.quat_rotate(quat, HITBOX_OFFSET)
    local_v0 = wp.quat_rotate_inv(quat, v0 - center)
    local_v1 = wp.quat_rotate_inv(quat, v1 - center)
    local_v2 = wp.quat_rotate_inv(quat, v2 - center)
    edge0 = local_v1 - local_v0
    edge1 = local_v2 - local_v1
    edge2 = local_v0 - local_v2
    # btGjkPairDetector::m_fixContactNormalDirection compares the centers of
    # the two world AABBs, not the triangle centroid.  For long skewed goal
    # triangles the centroid and AABB center can lie on opposite sides of the
    # minimum SAT axis, so this distinction determines the EPA witness side.
    triangle_aabb_center = wp.vec3(
        (
            wp.min(local_v0[0], wp.min(local_v1[0], local_v2[0]))
            + wp.max(local_v0[0], wp.max(local_v1[0], local_v2[0]))
        )
        * 0.5,
        (
            wp.min(local_v0[1], wp.min(local_v1[1], local_v2[1]))
            + wp.max(local_v0[1], wp.max(local_v1[1], local_v2[1]))
        )
        * 0.5,
        (
            wp.min(local_v0[2], wp.min(local_v1[2], local_v2[2]))
            + wp.max(local_v0[2], wp.max(local_v1[2], local_v2[2]))
        )
        * 0.5,
    )

    for axis_index in range(13):
        axis = wp.vec3(1.0, 0.0, 0.0)
        if axis_index == 1:
            axis = wp.vec3(0.0, 1.0, 0.0)
        elif axis_index == 2:
            axis = wp.vec3(0.0, 0.0, 1.0)
        elif axis_index == 3:
            axis = wp.cross(edge0, edge1)
        elif axis_index == 4:
            axis = wp.cross(edge0, wp.vec3(1.0, 0.0, 0.0))
        elif axis_index == 5:
            axis = wp.cross(edge0, wp.vec3(0.0, 1.0, 0.0))
        elif axis_index == 6:
            axis = wp.cross(edge0, wp.vec3(0.0, 0.0, 1.0))
        elif axis_index == 7:
            axis = wp.cross(edge1, wp.vec3(1.0, 0.0, 0.0))
        elif axis_index == 8:
            axis = wp.cross(edge1, wp.vec3(0.0, 1.0, 0.0))
        elif axis_index == 9:
            axis = wp.cross(edge1, wp.vec3(0.0, 0.0, 1.0))
        elif axis_index == 10:
            axis = wp.cross(edge2, wp.vec3(1.0, 0.0, 0.0))
        elif axis_index == 11:
            axis = wp.cross(edge2, wp.vec3(0.0, 1.0, 0.0))
        elif axis_index == 12:
            axis = wp.cross(edge2, wp.vec3(0.0, 0.0, 1.0))
        axis_length_sq = wp.dot(axis, axis)
        if axis_length_sq > 1.0e-12:
            axis = axis / wp.sqrt(axis_length_sq)
            penetration = _axis_penetration(
                axis,
                local_v0,
                local_v1,
                local_v2,
                HITBOX_COLLISION_HALF,
            )
            world_normal = wp.quat_rotate(quat, axis)
            if wp.dot(axis, triangle_aabb_center) > 0.0:
                world_normal = -world_normal
            candidate = _deep_sat_penetration_witness(
                pos,
                quat,
                world_normal,
                penetration,
                v0,
                v1,
                v2,
            )
            if candidate.distance > result.distance:
                result = candidate
    return result


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
    contact_point_bt: wp.vec3
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
    body_origin: wp.vec3,
    basis: wp.mat33,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
) -> _GjkClosest:
    """Positive-distance Bullet-style GJK in native BT coordinates."""

    result = _GjkClosest()
    result.valid = 0
    basis_transpose = wp.transpose(basis)
    transform_origin = body_origin + basis * GJK_OFFSET_BT
    midpoint = transform_origin * 0.5
    local_origin = transform_origin - midpoint
    v0 = v0 - midpoint
    v1 = v1 - midpoint
    v2 = v2 - midpoint

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
        local_direction = basis_transpose * -axis
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
        point_a = local_origin + basis * local_support

        dot0 = wp.dot(axis, v0)
        dot1 = wp.dot(axis, v1)
        dot2 = wp.dot(axis, v2)
        point_b = v0
        if dot0 < dot1:
            if dot1 < dot2:
                point_b = v2
            else:
                point_b = v1
        elif dot0 < dot2:
            point_b = v2
        w = point_a - point_b
        repeated = _gjk_repeated_vertex(count, w, w0, w1, w2, w3, last_w)
        delta = wp.dot(axis, w)
        if delta > 0.0 and (
            delta * delta > squared_distance * GJK_MAXIMUM_DISTANCE_BT * GJK_MAXIMUM_DISTANCE_BT
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
        result.contact_point_bt = (
            cached_core - axis * (GJK_MARGIN_BT / simplex_length) + midpoint
        )
        result.contact_point = result.contact_point_bt * 50.0
        result.normal = normal
        result.distance = (axis_length - GJK_MARGIN_BT) * 50.0
        result.valid = 1
    return result


@wp.func
def _contact_tangent(normal: wp.vec3, point_velocity: wp.vec3, bt_units: int) -> wp.vec3:
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


_BULLET_INTEGRATE_QUATERNION = r"""
    // Literal btTransformUtil::integrateTransform rotation path, including
    // btMatrix3x3::getRotation, exponential-map construction, the SSE
    // quaternion product, and safeNormalize reduction order.
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
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    auto op_sqrt = [](float value) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsqrt_rn(value);
    #else
        volatile float result = ::sqrtf(value);
        return result;
    #endif
    };
    auto op_sin = [](float value) -> float {
    #if defined(__CUDA_ARCH__)
        // Literal small-positive-domain path from the pinned ucrtbase!sinf
        // export. Bullet supplies |angle * dt / 2| <= pi/8 here. Preserve
        // its double operations and FMA order; CUDA sinf and CUDA's general
        // double sin do not round identically to the authority runtime.
        const unsigned bits = __float_as_uint(value) & 0x7fffffffu;
        if (bits < 0x39000000u) return value;
        const double x = static_cast<double>(value);
        const double x2 = __dmul_rn(x, x);
        const double x3 = __dmul_rn(x2, x);
        if (bits < 0x3c000000u) {
            return static_cast<float>(__fma_rn(
                -x3, __longlong_as_double(0x3fc5555555555555ULL), x));
        }
        double polynomial = __longlong_as_double(0xbf2a01a01a01a01aULL);
        polynomial = __fma_rn(
            x2, __longlong_as_double(0x3ec71de3a556c734ULL), polynomial);
        polynomial = __fma_rn(
            polynomial, x2, __longlong_as_double(0x3f81111111111111ULL));
        polynomial = __fma_rn(
            polynomial, x2, __longlong_as_double(0xbfc5555555555555ULL));
        return static_cast<float>(__fma_rn(polynomial, x3, x));
    #else
        volatile float result = ::sinf(value);
        return result;
    #endif
    };
    auto op_cos = [](float value) -> float {
    #if defined(__CUDA_ARCH__)
        // Matching small-positive-domain path from ucrtbase!cosf.
        const unsigned bits = __float_as_uint(value) & 0x7fffffffu;
        if (bits < 0x39000000u) return 1.0f;
        const double x = static_cast<double>(value);
        if (bits < 0x3c000000u) {
            const double half_x = __dmul_rn(
                x, __longlong_as_double(0x3fe0000000000000ULL));
            return static_cast<float>(__fma_rn(-x, half_x, 1.0));
        }
        const double x2 = __dmul_rn(x, x);
        const double half_x2 = __dmul_rn(
            x2, __longlong_as_double(0x3fe0000000000000ULL));
        const double base = __dsub_rn(1.0, half_x2);
        double polynomial = __longlong_as_double(0x3efa01a01a01a019ULL);
        polynomial = __fma_rn(
            x2, __longlong_as_double(0xbe927e4fb7789f5cULL), polynomial);
        polynomial = __fma_rn(
            polynomial, x2, __longlong_as_double(0xbf56c16c16c16c16ULL));
        polynomial = __fma_rn(
            polynomial, x2, __longlong_as_double(0x3fa5555555555555ULL));
        const double x4 = __dmul_rn(x2, x2);
        return static_cast<float>(__fma_rn(polynomial, x4, base));
    #else
        volatile float result = ::cosf(value);
        return result;
    #endif
    };

    float root;
    float qx;
    float qy;
    float qz;
    float qw;
    const float trace = op_add(
        op_add(basis.data[0][0], basis.data[1][1]),
        basis.data[2][2]);
    if (trace > 0.0f) {
        root = op_add(trace, 1.0f);
        qx = op_sub(basis.data[2][1], basis.data[1][2]);
        qy = op_sub(basis.data[0][2], basis.data[2][0]);
        qz = op_sub(basis.data[1][0], basis.data[0][1]);
        qw = root;
    } else if (basis.data[0][0] < basis.data[1][1]) {
        if (basis.data[1][1] < basis.data[2][2]) {
            root = op_add(
                op_sub(op_sub(basis.data[2][2], basis.data[0][0]), basis.data[1][1]),
                1.0f);
            qx = op_add(basis.data[0][2], basis.data[2][0]);
            qy = op_add(basis.data[1][2], basis.data[2][1]);
            qz = root;
            qw = op_sub(basis.data[1][0], basis.data[0][1]);
        } else {
            root = op_add(
                op_sub(op_sub(basis.data[1][1], basis.data[2][2]), basis.data[0][0]),
                1.0f);
            qx = op_add(basis.data[0][1], basis.data[1][0]);
            qy = root;
            qz = op_add(basis.data[2][1], basis.data[1][2]);
            qw = op_sub(basis.data[0][2], basis.data[2][0]);
        }
    } else if (basis.data[0][0] < basis.data[2][2]) {
        root = op_add(
            op_sub(op_sub(basis.data[2][2], basis.data[0][0]), basis.data[1][1]),
            1.0f);
        qx = op_add(basis.data[0][2], basis.data[2][0]);
        qy = op_add(basis.data[1][2], basis.data[2][1]);
        qz = root;
        qw = op_sub(basis.data[1][0], basis.data[0][1]);
    } else {
        root = op_add(
            op_sub(op_sub(basis.data[0][0], basis.data[1][1]), basis.data[2][2]),
            1.0f);
        qx = root;
        qy = op_add(basis.data[1][0], basis.data[0][1]);
        qz = op_add(basis.data[2][0], basis.data[0][2]);
        qw = op_sub(basis.data[2][1], basis.data[1][2]);
    }
    const float matrix_scale = op_div(0.5f, op_sqrt(root));
    qx = op_mul(qx, matrix_scale);
    qy = op_mul(qy, matrix_scale);
    qz = op_mul(qz, matrix_scale);
    qw = op_mul(qw, matrix_scale);

    const float ax = angular_velocity_world[0];
    const float ay = angular_velocity_world[1];
    const float az = angular_velocity_world[2];
    const float angle_squared = op_add(
        op_add(op_mul(ax, ax), op_mul(ay, ay)), op_mul(az, az));
    float angle = 0.0f;
    if (angle_squared > 1.1920928955078125e-7f) {
        angle = op_sqrt(angle_squared);
    }
    if (op_mul(angle, time_step) > 0.7853981852531433f) {
        angle = op_div(0.7853981852531433f, time_step);
    }

    float axis_scale;
    if (angle < 0.001f) {
        const float time_step_cubed = op_mul(op_mul(time_step, time_step), time_step);
        float correction = op_mul(time_step_cubed, 0.020833333333f);
        correction = op_mul(correction, angle);
        correction = op_mul(correction, angle);
        axis_scale = op_sub(op_mul(0.5f, time_step), correction);
    } else {
        const float half_angle_step = op_mul(op_mul(0.5f, angle), time_step);
        axis_scale = op_div(op_sin(half_angle_step), angle);
    }
    const float dx = op_mul(ax, axis_scale);
    const float dy = op_mul(ay, axis_scale);
    const float dz = op_mul(az, axis_scale);
    const float dw = op_cos(op_mul(op_mul(angle, time_step), 0.5f));

    // dorn * orn0, following the pinned four-lane SSE grouping.
    float x = op_add(
        op_sub(op_mul(dw, qx), op_mul(dz, qy)),
        op_add(op_mul(dx, qw), op_mul(dy, qz)));
    float y = op_add(
        op_sub(op_mul(dw, qy), op_mul(dx, qz)),
        op_add(op_mul(dy, qw), op_mul(dz, qx)));
    float z = op_add(
        op_sub(op_mul(dw, qz), op_mul(dy, qx)),
        op_add(op_mul(dz, qw), op_mul(dx, qy)));
    float w = op_add(
        op_sub(op_mul(dw, qw), op_mul(dz, qz)),
        -op_add(op_mul(dx, qx), op_mul(dy, qy)));

    // btQuaternion::normalize sums (x*x + z*z) + (y*y + w*w).
    const float quaternion_length_squared = op_add(
        op_add(op_mul(x, x), op_mul(z, z)),
        op_add(op_mul(y, y), op_mul(w, w)));
    if (quaternion_length_squared > 1.1920928955078125e-7f) {
        const float inverse_length = op_div(1.0f, op_sqrt(quaternion_length_squared));
        x = op_mul(x, inverse_length);
        y = op_mul(y, inverse_length);
        z = op_mul(z, inverse_length);
        w = op_mul(w, inverse_length);
    }
    integrated_orientation = wp::quat_t<wp::float32>(x, y, z, w);
"""


@wp.func_native(_BULLET_INTEGRATE_QUATERNION)
def _bullet_integrate_quaternion(
    basis: wp.mat33,
    angular_velocity_world: wp.vec3,
    time_step: float,
    integrated_orientation: wp.ref[wp.quat],
): ...


@wp.func
def _contact_integrate_quaternion(basis: wp.mat33, ang_vel: wp.vec3) -> wp.quat:
    # btTransform stores a basis, not a quaternion. Every integration first
    # recovers orn0 from that stored matrix, then applies the exponential map.
    integrated = wp.quat(0.0, 0.0, 0.0, 1.0)
    _bullet_integrate_quaternion(basis, ang_vel, DT, integrated)
    return integrated


@wp.func
def _contact_cap(value: wp.vec3, maximum: float) -> wp.vec3:
    length_sq = wp.dot(value, value)
    if length_sq > maximum * maximum:
        return value * (maximum / wp.sqrt(length_sq))
    return value


_BULLET_MATRIX_AXIS_ANGLE_ROTATE = r"""
    // btClampNormal constructs btQuaternion(edge, diffAngle), materializes
    // btMatrix3x3(rotation), then multiplies that matrix by the contact normal.
    // Preserve the pinned SSE-visible scalar boundaries instead of replacing
    // the source sequence with a mathematically equivalent Rodrigues formula.
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float result = a + b;
        return result;
    #endif
    };
    auto op_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float result = a * b;
        return result;
    #endif
    };
    auto op_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float result = a / b;
        return result;
    #endif
    };
    const float ax = axis[0];
    const float ay = axis[1];
    const float az = axis[2];
    const float axis_length_sq = op_add(
        op_add(op_mul(ax, ax), op_mul(ay, ay)), op_mul(az, az));
    if (axis_length_sq <= 0.0f) {
        return value;
    }
    const float half_angle = op_mul(angle, 0.5f);
    #if defined(__CUDA_ARCH__)
    const float half_sine = static_cast<float>(sin(static_cast<double>(half_angle)));
    const float half_cosine = static_cast<float>(cos(static_cast<double>(half_angle)));
    #else
    const float half_sine = sinf(half_angle);
    const float half_cosine = cosf(half_angle);
    #endif
    const float quaternion_scale = op_div(half_sine, sqrtf(axis_length_sq));
    const float x = op_mul(ax, quaternion_scale);
    const float y = op_mul(ay, quaternion_scale);
    const float z = op_mul(az, quaternion_scale);
    const float w = half_cosine;

    // btQuaternion::length2 groups the four-wide SSE reduction as
    // (x*x + z*z) + (y*y + w*w).
    const float xx = op_mul(x, x);
    const float yy = op_mul(y, y);
    const float zz = op_mul(z, z);
    const float ww = op_mul(w, w);
    const float quaternion_length_sq = op_add(op_add(xx, zz), op_add(yy, ww));
    const float matrix_scale = op_div(2.0f, quaternion_length_sq);

    const float m00 = op_add(
        1.0f, op_mul(op_add(op_mul(-y, y), op_mul(-z, z)), matrix_scale));
    const float m01 = op_mul(
        op_add(op_mul(x, y), op_mul(-w, z)), matrix_scale);
    const float m02 = op_mul(
        op_add(op_mul(x, z), op_mul(w, y)), matrix_scale);
    const float m10 = op_mul(
        op_add(op_mul(x, y), op_mul(w, z)), matrix_scale);
    const float m11 = op_add(
        1.0f, op_mul(op_add(op_mul(-x, x), op_mul(-z, z)), matrix_scale));
    const float m12 = op_mul(
        op_add(op_mul(y, z), op_mul(-w, x)), matrix_scale);
    const float m20 = op_mul(
        op_add(op_mul(x, z), op_mul(-w, y)), matrix_scale);
    const float m21 = op_mul(
        op_add(op_mul(y, z), op_mul(w, x)), matrix_scale);
    const float m22 = op_add(
        1.0f, op_mul(op_add(op_mul(-x, x), op_mul(-y, y)), matrix_scale));

    auto row_dot = [&](float m0, float m1, float m2) -> float {
        return op_add(
            op_add(op_mul(m0, value[0]), op_mul(m1, value[1])),
            op_mul(m2, value[2]));
    };
    return wp::vec_t<3, wp::float32>(
        row_dot(m00, m01, m02),
        row_dot(m10, m11, m12),
        row_dot(m20, m21, m22));
"""


@wp.func_native(_BULLET_MATRIX_AXIS_ANGLE_ROTATE)
def _bullet_matrix_axis_angle_rotate(
    value: wp.vec3, axis: wp.vec3, angle: float
) -> wp.vec3: ...


_BULLET_INTERNAL_EDGE_DYNAMIC = r"""
    auto op_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float result = a + b;
        return result;
    #endif
    };
    auto op_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float result = a - b;
        return result;
    #endif
    };
    auto op_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float result = a * b;
        return result;
    #endif
    };
    auto dot = [&](const wp::vec_t<3, wp::float32>& a,
                   const wp::vec_t<3, wp::float32>& b) -> float {
        return op_add(
            op_add(op_mul(a[0], b[0]), op_mul(a[1], b[1])),
            op_mul(a[2], b[2]));
    };
    if (mode == 0) {
        // btVector3::normalize in the pinned Windows build: RSQRTSS estimate
        // plus one Newton step. Raw GJK/EPA normals reach this callback in the
        // same one-ULP neighborhood as the already-ported support directions.
        const float length_squared = dot(lhs, lhs);
        float inverse_length;
    #if defined(__CUDA_ARCH__)
        const unsigned length_bits = __float_as_uint(length_squared);
        if (length_bits >= 0x3f7ff000u && length_bits < 0x3f800000u) {
            inverse_length = __uint_as_float(0x3f800000u);
        } else if (length_bits >= 0x3f800000u && length_bits < 0x3f800800u) {
            inverse_length = __uint_as_float(0x3f7ff800u);
        } else {
            inverse_length = rsqrtf(length_squared);
        }
    #elif defined(__clang__) && (defined(__x86_64__) || defined(_M_X64))
        using DynamicM128 = float __attribute__((__vector_size__(16)));
        DynamicM128 input = {length_squared, 0.0f, 0.0f, 0.0f};
        DynamicM128 estimate = __builtin_ia32_rsqrtss(input);
        inverse_length = estimate[0];
    #else
        inverse_length = 1.0f / sqrtf(length_squared);
    #endif
        float correction = op_mul(op_mul(length_squared, 0.5f), inverse_length);
        correction = op_mul(correction, inverse_length);
        correction = op_sub(1.5f, correction);
        inverse_length = op_mul(inverse_length, correction);
        return wp::vec_t<3, wp::float32>(
            op_mul(lhs[0], inverse_length),
            op_mul(lhs[1], inverse_length),
            op_mul(lhs[2], inverse_length));
    }
    if (mode == 1) {
        const float value = dot(lhs, rhs);
        return wp::vec_t<3, wp::float32>(value, 0.0f, 0.0f);
    }
    const float numerator = dot(lhs, rhs);
    const float denominator = dot(lhs, third);
    #if defined(__CUDA_ARCH__)
    // CUDA's fast float atan2 approximation differs from the pinned Windows
    // UCRT atan2f by one ULP for observed internal-edge angles. Evaluating the
    // same float32 arguments in correctly rounded double and narrowing once
    // reproduces the authority result without changing either input.
    const float angle = static_cast<float>(atan2(
        static_cast<double>(numerator), static_cast<double>(denominator)));
    #else
    const float angle = atan2f(numerator, denominator);
    #endif
    return wp::vec_t<3, wp::float32>(angle, 0.0f, 0.0f);
"""


@wp.func_native(_BULLET_INTERNAL_EDGE_DYNAMIC)
def _bullet_internal_edge_dynamic(
    lhs: wp.vec3, rhs: wp.vec3, third: wp.vec3, mode: int
) -> wp.vec3: ...


@wp.func
def _bullet_sse_normalize(value: wp.vec3) -> wp.vec3:
    return _bullet_internal_edge_dynamic(
        value, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 0
    )


@wp.func
def _bullet_internal_edge_dot(lhs: wp.vec3, rhs: wp.vec3) -> float:
    return _bullet_internal_edge_dynamic(lhs, rhs, wp.vec3(0.0, 0.0, 0.0), 1)[0]


@wp.func
def _bullet_internal_edge_angle(
    local_normal: wp.vec3, edge_cross: wp.vec3, triangle_normal: wp.vec3
) -> float:
    return _bullet_internal_edge_dynamic(
        local_normal, edge_cross, triangle_normal, 2
    )[0]


@wp.func
def _rotate_axis_angle(value: wp.vec3, axis: wp.vec3, angle: float) -> wp.vec3:
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


@wp.func
def _manifold_replacement_index(
    candidate_local_a: wp.vec3,
    candidate_distance: float,
    manifold_start: int,
    contact_local_a: wp.array(dtype=wp.vec3),
    contact_distance: wp.array(dtype=wp.float32),
) -> int:
    """Mirror btPersistentManifold::sortCachedPoints for a full cache.

    Operate on Bullet-unit rigid-body-local point-A values, which makes the
    reduction invariant to the current car transform and preserves float32
    behavior around near-tied cache areas.
    """

    p0 = contact_local_a[manifold_start]
    p1 = contact_local_a[manifold_start + 1]
    p2 = contact_local_a[manifold_start + 2]
    p3 = contact_local_a[manifold_start + 3]
    return bullet_manifold_replacement(
        candidate_local_a,
        p0,
        p1,
        p2,
        p3,
        candidate_distance,
        contact_distance[manifold_start],
        contact_distance[manifold_start + 1],
        contact_distance[manifold_start + 2],
        contact_distance[manifold_start + 3],
    )


@wp.func
def _refresh_mesh_manifold(
    body_pos_bt: wp.vec3,
    basis: wp.mat33,
    manifold_start: int,
    manifold_contacts: int,
    contact_point: wp.array(dtype=wp.vec3),
    contact_local_a: wp.array(dtype=wp.vec3),
    contact_point_b: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_face: wp.array(dtype=wp.int32),
    contact_mesh: wp.array(dtype=wp.int32),
    contact_distance: wp.array(dtype=wp.float32),
    contact_distance_bt: wp.array(dtype=wp.float32),
    contact_penetration: wp.array(dtype=wp.float32),
    contact_lifetime: wp.array(dtype=wp.int32),
) -> int:
    """Mirror the post-triangle manifold refresh for one CMF dispatch."""

    retained = manifold_contacts
    for reverse_offset in range(4):
        relative_index = manifold_contacts - 1 - reverse_offset
        if relative_index >= 0 and relative_index < retained:
            index = manifold_start + relative_index
            point_a_bt = _bullet_transform_point(
                body_pos_bt,
                basis,
                contact_local_a[index],
            )
            point_b_bt = contact_point_b[index]
            normal = contact_normal[index]
            distance_bt = _bullet_internal_edge_dot(
                point_a_bt - point_b_bt, normal
            )
            projected_point_bt = _bullet_vector_scale_add(
                point_a_bt, normal, -distance_bt
            )
            projected_difference_bt = point_b_bt - projected_point_bt
            lateral_distance_sq = _bullet_internal_edge_dot(
                projected_difference_bt, projected_difference_bt
            )
            breaking_threshold_bt = CONTACT_BREAKING_THRESHOLD * 0.02
            invalid = distance_bt > breaking_threshold_bt or (
                lateral_distance_sq > breaking_threshold_bt * breaking_threshold_bt
            )
            if invalid:
                last = manifold_start + retained - 1
                if index != last:
                    contact_point[index] = contact_point[last]
                    contact_local_a[index] = contact_local_a[last]
                    contact_point_b[index] = contact_point_b[last]
                    contact_normal[index] = contact_normal[last]
                    contact_face[index] = contact_face[last]
                    contact_mesh[index] = contact_mesh[last]
                    contact_distance[index] = contact_distance[last]
                    contact_distance_bt[index] = contact_distance_bt[last]
                    contact_penetration[index] = contact_penetration[last]
                    contact_lifetime[index] = contact_lifetime[last]
                retained = retained - 1
            else:
                contact_lifetime[index] = contact_lifetime[index] + 1
                contact_point[index] = point_a_bt * 50.0
                contact_distance[index] = distance_bt * 50.0
                contact_distance_bt[index] = distance_bt
                contact_penetration[index] = wp.max(0.0, -distance_bt * 50.0)
    return retained


@wp.func
def _bullet_solver_contact_index(
    car: int,
    solver_index: int,
    contacts: int,
    contact_mesh: wp.array(dtype=wp.int32),
) -> int:
    """Map a flattened contact ordinal through Bullet's manifold quicksort.

    ``btSimulationIslandManager::processIslands`` sorts the car's static
    manifolds by island id before ``btSequentialImpulseConstraintSolver``
    converts their points to solver rows.  Every manifold connected to this
    single dynamic body has the same island id, so the pinned
    ``btPersistentManifoldSortPredicate`` returns false for every comparison.
    ``btAlignedObjectArray::quickSortInternal`` nevertheless swaps the equal
    entries during each partition.  The packed permutations below are the
    literal result of that source algorithm for the bounded 1..12 manifold
    counts supported by ``MAX_CONTACTS_PER_CAR``.  Contacts within a manifold
    retain their persistent-manifold order.
    """

    group_count = wp.int32(0)
    previous_mesh = wp.int32(-1)
    for index in range(MAX_CONTACTS_PER_CAR):
        if index < contacts:
            mesh = contact_mesh[car * MAX_CONTACTS_PER_CAR + index]
            if index == 0 or mesh != previous_mesh:
                group_count = group_count + 1
            previous_mesh = mesh

    # Four-bit source-group ordinals, least-significant nibble first.
    permutation = wp.uint64(0x0)
    if group_count == 2:
        permutation = wp.uint64(0x1)
    elif group_count == 3:
        permutation = wp.uint64(0x12)
    elif group_count == 4:
        permutation = wp.uint64(0x1032)
    elif group_count == 5:
        permutation = wp.uint64(0x10243)
    elif group_count == 6:
        permutation = wp.uint64(0x210543)
    elif group_count == 7:
        permutation = wp.uint64(0x2103654)
    elif group_count == 8:
        permutation = wp.uint64(0x23016745)
    elif group_count == 9:
        permutation = wp.uint64(0x230147856)
    elif group_count == 10:
        permutation = wp.uint64(0x3420189756)
    elif group_count == 11:
        permutation = wp.uint64(0x3420159A867)
    elif group_count == 12:
        permutation = wp.uint64(0x3450129AB678)

    remaining = solver_index
    mapped_index = solver_index
    mapped = wp.int32(0)
    for solver_group in range(MAX_CONTACTS_PER_CAR):
        if solver_group < group_count and mapped == 0:
            shift = wp.uint64(solver_group * 4)
            source_group = wp.int32((permutation >> shift) & wp.uint64(0xF))
            group_ordinal = wp.int32(-1)
            group_start = wp.int32(0)
            group_size = wp.int32(0)
            source_previous_mesh = wp.int32(-1)
            for source_index in range(MAX_CONTACTS_PER_CAR):
                if source_index < contacts:
                    source_mesh = contact_mesh[
                        car * MAX_CONTACTS_PER_CAR + source_index
                    ]
                    if source_index == 0 or source_mesh != source_previous_mesh:
                        group_ordinal = group_ordinal + 1
                        if group_ordinal == source_group:
                            group_start = source_index
                    if group_ordinal == source_group:
                        group_size = group_size + 1
                    source_previous_mesh = source_mesh
            if remaining < group_size:
                mapped_index = group_start + remaining
                mapped = 1
            else:
                remaining = remaining - group_size
    return mapped_index


@wp.kernel(
    enable_backward=False,
    module="unique",
    module_options={"max_unroll": 4},
)
def chassis_contacts_v021(
    tick_counter: wp.array(dtype=wp.int32),
    aabb_mesh_id: wp.uint64,
    vertices_bt: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    internal_edge_face_normals: wp.array(dtype=wp.vec3),
    internal_edge_crosses: wp.array(dtype=wp.vec3),
    internal_edge_normal_bs: wp.array(dtype=wp.vec3),
    internal_edge_angles: wp.array(dtype=wp.vec3),
    internal_edge_flags: wp.array(dtype=wp.int32),
    bullet_bvh_rank: wp.array(dtype=wp.int32),
    face_mesh_index: wp.array(dtype=wp.int32),
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
    total_force_bt: wp.array(dtype=wp.vec3),
    total_torque_bt: wp.array(dtype=wp.vec3),
    candidate_count: wp.array(dtype=wp.int32),
    mesh_candidate_count: wp.array(dtype=wp.int32),
    mesh_candidate_overflow: wp.array(dtype=wp.int32),
    contact_overflow: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    world_contact_normal: wp.array(dtype=wp.vec3),
    candidate_total: wp.array(dtype=wp.float32),
    contact_total: wp.array(dtype=wp.float32),
    candidate_max: wp.array(dtype=wp.int32),
    contact_max: wp.array(dtype=wp.int32),
    penetration_max: wp.array(dtype=wp.float32),
    contact_point: wp.array(dtype=wp.vec3),
    contact_local_a: wp.array(dtype=wp.vec3),
    contact_point_b: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_tangent: wp.array(dtype=wp.vec3),
    contact_face: wp.array(dtype=wp.int32),
    contact_mesh: wp.array(dtype=wp.int32),
    contact_distance: wp.array(dtype=wp.float32),
    contact_distance_bt: wp.array(dtype=wp.float32),
    contact_penetration: wp.array(dtype=wp.float32),
    contact_normal_jacobian: wp.array(dtype=wp.float32),
    contact_tangent_jacobian: wp.array(dtype=wp.float32),
    contact_normal_rhs: wp.array(dtype=wp.float32),
    contact_tangent_rhs: wp.array(dtype=wp.float32),
    contact_push_rhs: wp.array(dtype=wp.float32),
    contact_normal_impulse: wp.array(dtype=wp.float32),
    contact_tangent_impulse: wp.array(dtype=wp.float32),
    contact_push_impulse: wp.array(dtype=wp.float32),
    contact_lifetime: wp.array(dtype=wp.int32),
    mesh_candidate_face: wp.array(dtype=wp.int32),
    plane_support_direction: wp.array(dtype=wp.float32),
):
    """Minimal static Bullet contact generation and ten-iteration PGS solve."""

    car = wp.tid()
    pos = solver_position[car]
    quat = solver_orientation[car]
    bullet_basis = _bullet_quaternion_matrix(quat)
    if tick_counter[0] == 0:
        bullet_basis = _authority_input_quaternion_matrix(quat)
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
            force_vel = (pre_force_vel * 0.02 + wp.vec3(0.0, 0.0, -13.0) * DT) * 50.0
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
    previous_contacts = contact_count[car]
    candidates = wp.int32(0)
    retained_mesh_candidates = wp.int32(0)
    mesh_candidate_excess = wp.int32(0)
    contact_excess = wp.int32(0)
    contacts = wp.int32(0)
    maximum_penetration = float(0.0)  # noqa: UP018 - Warp mutable kernel local
    callback_normal = wp.vec3(0.0, 0.0, 0.0)
    previous_plane_mask = wp.int32(0)
    for previous_index in range(MAX_CONTACTS_PER_CAR):
        if previous_index < previous_contacts:
            previous_offset = car * MAX_CONTACTS_PER_CAR + previous_index
            previous_face = contact_face[previous_offset]
            if previous_face <= -10 and previous_face >= -13:
                previous_plane_mask = previous_plane_mask | (1 << (-10 - previous_face))

    # Warp's combined mesh is an acceleration structure only. Gather every
    # SAT-overlapping face, sort it into RocketSim's per-CMF Bullet BVH visit
    # order, and then feed that deterministic stream to the native four-point
    # manifold reduction below.
    query = wp.mesh_query_aabb(aabb_mesh_id, center - aabb_half, center + aabb_half)
    for face in query:
        candidates = candidates + 1
        v0 = wp.mesh_eval_position(aabb_mesh_id, face, 1.0, 0.0)
        v1 = wp.mesh_eval_position(aabb_mesh_id, face, 0.0, 1.0)
        v2 = wp.mesh_eval_position(aabb_mesh_id, face, 0.0, 0.0)
        sat = _triangle_obb_sat(v0, v1, v2, center, quat)
        penetration = sat[3]
        if penetration >= -CONTACT_BREAKING_THRESHOLD and (
            sat[0] * sat[0] + sat[1] * sat[1] + sat[2] * sat[2] > 0.5
        ):
            insert_index = retained_mesh_candidates
            candidate_rank = bullet_bvh_rank[face]
            candidate_mesh = face_mesh_index[face]
            for existing in range(retained_mesh_candidates):
                existing_face = mesh_candidate_face[car * MAX_MESH_CANDIDATES_PER_CAR + existing]
                existing_mesh = face_mesh_index[existing_face]
                if insert_index == retained_mesh_candidates and (
                    candidate_mesh < existing_mesh
                    or (
                        candidate_mesh == existing_mesh
                        and candidate_rank < bullet_bvh_rank[existing_face]
                    )
                ):
                    insert_index = existing
            retained_count = wp.min(
                retained_mesh_candidates + 1,
                MAX_MESH_CANDIDATES_PER_CAR,
            )
            for shift_offset in range(retained_count):
                destination = retained_count - 1 - shift_offset
                if destination > insert_index:
                    mesh_candidate_face[car * MAX_MESH_CANDIDATES_PER_CAR + destination] = (
                        mesh_candidate_face[car * MAX_MESH_CANDIDATES_PER_CAR + destination - 1]
                    )
            if insert_index < MAX_MESH_CANDIDATES_PER_CAR:
                mesh_candidate_face[car * MAX_MESH_CANDIDATES_PER_CAR + insert_index] = face
            if retained_mesh_candidates >= MAX_MESH_CANDIDATES_PER_CAR:
                mesh_candidate_excess = mesh_candidate_excess + 1
            retained_mesh_candidates = retained_count

    active_mesh = wp.int32(-1)
    manifold_start = wp.int32(0)
    manifold_contacts = wp.int32(0)
    for candidate_index in range(retained_mesh_candidates):
        face = mesh_candidate_face[car * MAX_MESH_CANDIDATES_PER_CAR + candidate_index]
        mesh_index = face_mesh_index[face]
        if mesh_index >= 0:
            if mesh_index != active_mesh:
                if active_mesh >= 0:
                    manifold_contacts = _refresh_mesh_manifold(
                        rigid_position_bt[car],
                        bullet_basis,
                        car * MAX_CONTACTS_PER_CAR + manifold_start,
                        manifold_contacts,
                        contact_point,
                        contact_local_a,
                        contact_point_b,
                        contact_normal,
                        contact_face,
                        contact_mesh,
                        contact_distance,
                        contact_distance_bt,
                        contact_penetration,
                        contact_lifetime,
                    )
                    contacts = manifold_start + manifold_contacts
                active_mesh = mesh_index
                manifold_start = contacts
                manifold_contacts = wp.int32(0)
            v0 = wp.mesh_eval_position(aabb_mesh_id, face, 1.0, 0.0)
            v1 = wp.mesh_eval_position(aabb_mesh_id, face, 0.0, 1.0)
            v2 = wp.mesh_eval_position(aabb_mesh_id, face, 0.0, 0.0)
            triangle_offset = face * 3
            v0_bt = vertices_bt[triangle_indices[triangle_offset]]
            v1_bt = vertices_bt[triangle_indices[triangle_offset + 1]]
            v2_bt = vertices_bt[triangle_indices[triangle_offset + 2]]
            sat = _triangle_obb_sat(v0, v1, v2, center, quat)
            penetration = sat[3]
            if penetration >= -CONTACT_BREAKING_THRESHOLD and (
                sat[0] * sat[0] + sat[1] * sat[1] + sat[2] * sat[2] > 0.5
            ):
                local_normal = wp.vec3(sat[0], sat[1], sat[2])
                normal = wp.quat_rotate(quat, local_normal)
                callback_candidate_normal = normal
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

                    box_start = pos + wp.quat_rotate(quat, HITBOX_OFFSET + local_start)
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
                        candidate_triangle = (
                            triangle_start + (triangle_end - triangle_start) * parameters[1]
                        )
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
                candidate_point_a_bt = point * 0.02
                candidate_point_b_bt = triangle_point * 0.02
                distance = -HITBOX_MARGIN - wp.dot(
                    triangle_point - core_point,
                    normal,
                )
                selected_distance_bt = distance * 0.02

                # Bullet runs btGjkPairDetector even after the outer-margin
                # SAT overlaps.  Its finite-precision repeat threshold
                # intentionally preserves both a shallow simplex witness and
                # the B-to-A orientation consumed by the penetration solver.
                pair_point_a_bt = wp.vec3(0.0, 0.0, 0.0)
                pair_point_b_bt = wp.vec3(0.0, 0.0, 0.0)
                pair_normal = wp.vec3(0.0, 0.0, 0.0)
                pair_distance_bt = wp.float32(0.0)
                pair_valid = wp.int32(0)
                pair_degenerate = wp.int32(0)
                bullet_box_triangle_closest(
                    rigid_position_bt[car],
                    bullet_basis,
                    v0_bt,
                    v1_bt,
                    v2_bt,
                    pair_point_a_bt,
                    pair_point_b_bt,
                    pair_normal,
                    pair_distance_bt,
                    pair_valid,
                    pair_degenerate,
                )

                # getClosestPointsNonVirtual invokes calcPenDepth for an
                # invalid GJK result or its exact catch-degenerate condition,
                # then replaces an already-valid GJK witness only when the
                # penetration result is strictly deeper.
                contact_valid = pair_valid
                if pair_valid != 0:
                    point = pair_point_a_bt * 50.0
                    candidate_point_a_bt = pair_point_a_bt
                    candidate_point_b_bt = pair_point_b_bt
                    triangle_point = pair_point_b_bt * 50.0
                    normal = pair_normal
                    callback_candidate_normal = pair_normal
                    distance = pair_distance_bt * 50.0
                    selected_distance_bt = pair_distance_bt
                catch_degenerate = pair_degenerate != 0 and (
                    pair_distance_bt + GJK_MARGIN_BT < 0.01
                )
                if pair_valid == 0 or catch_degenerate:
                    point_a_bt = wp.vec3(0.0, 0.0, 0.0)
                    point_b_bt = wp.vec3(0.0, 0.0, 0.0)
                    epa_normal = wp.vec3(0.0, 0.0, 0.0)
                    epa_distance_bt = wp.float32(0.0)
                    epa_valid = wp.int32(0)
                    bullet_box_triangle_penetration(
                        rigid_position_bt[car],
                        bullet_basis,
                        v0_bt,
                        v1_bt,
                        v2_bt,
                        point_a_bt,
                        point_b_bt,
                        epa_normal,
                        epa_distance_bt,
                        epa_valid,
                    )
                    if epa_valid != 0 and (
                        pair_valid == 0 or epa_distance_bt < pair_distance_bt
                    ):
                        point = point_a_bt * 50.0
                        candidate_point_a_bt = point_a_bt
                        candidate_point_b_bt = point_b_bt
                        triangle_point = point_b_bt * 50.0
                        normal = epa_normal
                        distance = epa_distance_bt * 50.0
                        selected_distance_bt = epa_distance_bt
                        callback_candidate_normal = epa_normal
                        contact_valid = 1
                # RocketSim invokes btAdjustInternalEdgeContacts for every CMF
                # manifold point. Reproduce its closest-edge selection,
                # planar/concave normal replacement, and convex angle clamp.
                edge_angles = internal_edge_angles[face]
                edge_flags = internal_edge_flags[face]
                best_edge_distance_bt = wp.float32(0.0)
                best_edge = bullet_internal_edge_best(
                    candidate_point_b_bt,
                    v0_bt,
                    v1_bt,
                    v2_bt,
                    edge_angles,
                    best_edge_distance_bt,
                )

                internal_edge_reprojected = wp.int32(0)
                if best_edge >= 0 and best_edge_distance_bt < 0.1:
                    # Keep btAdjustInternalEdgeContacts on the original CMF
                    # Bullet-unit vertices used to select ``best_edge``.  A
                    # UU round-trip changes the normalized edge/normal bits at
                    # shared-vertex ties.
                    triangle_normal = internal_edge_face_normals[face]
                    local_contact_normal = _bullet_sse_normalize(normal)
                    edge_angle = edge_angles[best_edge]
                    if edge_angle == 0.0:
                        if (
                            _bullet_internal_edge_dot(
                                triangle_normal, local_contact_normal
                            )
                            >= 0.0
                        ):
                            normal = triangle_normal
                            internal_edge_reprojected = 1
                    else:
                        edge_start = v0_bt
                        edge_end = v1_bt
                        if best_edge == 1:
                            edge_start = v1_bt
                            edge_end = v2_bt
                        elif best_edge == 2:
                            edge_start = v2_bt
                            edge_end = v0_bt
                        edge_vector = edge_start - edge_end
                        is_convex = edge_flags & (1 << best_edge) != 0
                        swap_factor = -1.0
                        if is_convex:
                            swap_factor = 1.0
                        normal_a = triangle_normal * swap_factor
                        static_edge_index = face * 3 + best_edge
                        normal_b = internal_edge_normal_bs[static_edge_index]
                        back_facing = (
                            _bullet_internal_edge_dot(local_contact_normal, normal_a)
                            < 0.0
                            and _bullet_internal_edge_dot(
                                local_contact_normal, normal_b
                            )
                            < 0.0
                        )
                        if back_facing:
                            if (
                                _bullet_internal_edge_dot(
                                    triangle_normal, local_contact_normal
                                )
                                >= 0.0
                            ):
                                normal = triangle_normal
                                internal_edge_reprojected = 1
                        else:
                            edge_cross = internal_edge_crosses[static_edge_index]
                            clamp_contact_normal = local_contact_normal
                            # The pinned source shadows localContactNormalOnB
                            # without renormalizing it on edge 1 and edge 2.
                            if best_edge > 0:
                                clamp_contact_normal = normal
                            current_angle = _bullet_internal_edge_angle(
                                clamp_contact_normal,
                                edge_cross,
                                normal_a,
                            )
                            clamp = 0
                            if edge_angle < 0.0 and current_angle < edge_angle:
                                clamp = 1
                            elif edge_angle >= 0.0 and current_angle > edge_angle:
                                clamp = 1
                            if clamp != 0:
                                clamped_normal = _bullet_matrix_axis_angle_rotate(
                                    clamp_contact_normal,
                                    edge_vector,
                                    edge_angle - current_angle,
                                )
                                if (
                                    _bullet_internal_edge_dot(
                                        clamped_normal, triangle_normal
                                    )
                                    > 0.0
                                ):
                                    normal = clamped_normal
                                    internal_edge_reprojected = 1
                if contact_valid != 0 and distance < CONTACT_BREAKING_THRESHOLD:
                    # Arena::_BulletContactAddedCallback records the raw
                    # GJK/EPA normal before btAdjustInternalEdgeContacts and
                    # before later manifold reduction can evict this point.
                    callback_normal = callback_candidate_normal
                    candidate_local_a = _bullet_inverse_transform_point(
                        rigid_position_bt[car],
                        bullet_basis,
                        candidate_point_a_bt,
                    )
                    output_index = -1
                    if manifold_contacts < 4:
                        if contacts < MAX_CONTACTS_PER_CAR:
                            output_index = car * MAX_CONTACTS_PER_CAR + contacts
                            contacts = contacts + 1
                            manifold_contacts = manifold_contacts + 1
                        else:
                            contact_excess = contact_excess + 1
                    else:
                        replacement = _manifold_replacement_index(
                            candidate_local_a,
                            distance,
                            car * MAX_CONTACTS_PER_CAR + manifold_start,
                            contact_local_a,
                            contact_distance,
                        )
                        output_index = car * MAX_CONTACTS_PER_CAR + manifold_start + replacement
                    if output_index >= 0:
                        contact_point[output_index] = point
                        contact_local_a[output_index] = candidate_local_a
                        # btAdjustInternalEdgeContacts reprojects point B only
                        # when it actually replaces or clamps the normal. An
                        # untouched normal retains the raw GJK/EPA witness.
                        stored_point_b_bt = candidate_point_b_bt
                        if internal_edge_reprojected != 0:
                            stored_point_b_bt = _bullet_vector_scale_add(
                                candidate_point_a_bt,
                                normal,
                                -selected_distance_bt,
                            )
                        contact_point_b[output_index] = stored_point_b_bt
                        contact_normal[output_index] = normal
                        contact_face[output_index] = face
                        contact_mesh[output_index] = mesh_index
                        contact_distance[output_index] = distance
                        contact_distance_bt[output_index] = selected_distance_bt
                        contact_penetration[output_index] = wp.max(0.0, -distance)
                        contact_lifetime[output_index] = 0
                        maximum_penetration = wp.max(maximum_penetration, -distance)

    if active_mesh >= 0:
        manifold_contacts = _refresh_mesh_manifold(
            rigid_position_bt[car],
            bullet_basis,
            car * MAX_CONTACTS_PER_CAR + manifold_start,
            manifold_contacts,
            contact_point,
            contact_local_a,
            contact_point_b,
            contact_normal,
            contact_face,
            contact_mesh,
            contact_distance,
            contact_distance_bt,
            contact_penetration,
            contact_lifetime,
        )
        contacts = manifold_start + manifold_contacts

    # RocketSim adds its four analytic arena planes after the sixteen CMF
    # rigid bodies. Each plane owns a separate one-point manifold, so append
    # them in source-body order after all reduced mesh manifolds.
    for plane in range(4):
        normal = wp.vec3(0.0, 0.0, 1.0)
        plane_point = wp.vec3(0.0, 0.0, 0.0)
        if plane == 1:
            normal = wp.vec3(0.0, 0.0, -1.0)
            plane_point = wp.vec3(0.0, 0.0, SOCCAR_HEIGHT)
        elif plane == 2:
            normal = wp.vec3(1.0, 0.0, 0.0)
            plane_point = wp.vec3(
                -SOCCAR_EXTENT_X, 0.0, SOCCAR_HEIGHT * 0.5
            )
        elif plane == 3:
            normal = wp.vec3(-1.0, 0.0, 0.0)
            plane_point = wp.vec3(
                SOCCAR_EXTENT_X, 0.0, SOCCAR_HEIGHT * 0.5
            )
        plane_point_bt = plane_point * 0.02
        local_direction = _bullet_inverse_transform_point(
            wp.vec3(0.0, 0.0, 0.0), bullet_basis, -normal
        )
        local_support_bt = GJK_CORE_HALF_BT + wp.vec3(
            GJK_MARGIN_BT, GJK_MARGIN_BT, GJK_MARGIN_BT
        )
        for axis in range(3):
            direction = local_direction[axis]
            if direction < 0.0:
                local_support_bt[axis] = -local_support_bt[axis]
            plane_support_direction[car * 12 + plane * 3 + axis] = direction
        point_bt = wp.vec3(0.0, 0.0, 0.0)
        source_point_b_bt = wp.vec3(0.0, 0.0, 0.0)
        distance_bt = wp.float32(0.0)
        _bullet_plane_contact_witness(
            rigid_position_bt[car],
            bullet_basis,
            plane_point_bt,
            normal,
            GJK_OFFSET_BT,
            local_support_bt,
            point_bt,
            source_point_b_bt,
            distance_bt,
        )
        point = point_bt * 50.0
        distance = distance_bt * 50.0
        breaking_threshold = CONTACT_BREAKING_THRESHOLD
        if plane >= 2:
            breaking_threshold = breaking_threshold - CONTACT_BREAKING_ROUNDING_GUARD
        if distance < breaking_threshold:
            candidates = candidates + 1
            callback_normal = normal
            if contacts < MAX_CONTACTS_PER_CAR:
                # btConvexPlaneCollisionAlgorithm adds the raw support point,
                # then btCompoundCollisionAlgorithm calls refreshContactPoints
                # before solver setup. Preserve that local-point round trip:
                # it can change the retained depth by one float32 ULP even
                # though the raw support witness itself is already exact.
                plane_local_a = _bullet_inverse_transform_point(
                    rigid_position_bt[car],
                    bullet_basis,
                    point_bt,
                )
                plane_local_b = source_point_b_bt - plane_point_bt
                refreshed_point_a_bt = _bullet_transform_point(
                    rigid_position_bt[car], bullet_basis, plane_local_a
                )
                refreshed_point_b_bt = plane_local_b + plane_point_bt
                refreshed_distance_bt = _bullet_internal_edge_dot(
                    refreshed_point_a_bt - refreshed_point_b_bt, normal
                )
                projected_point_bt = _bullet_vector_scale_add(
                    refreshed_point_a_bt,
                    normal,
                    -refreshed_distance_bt,
                )
                projected_difference_bt = (
                    refreshed_point_b_bt - projected_point_bt
                )
                refreshed_lateral_sq = _bullet_internal_edge_dot(
                    projected_difference_bt, projected_difference_bt
                )
                breaking_threshold_bt = CONTACT_BREAKING_THRESHOLD * 0.02
                if (
                    refreshed_distance_bt <= breaking_threshold_bt
                    and refreshed_lateral_sq
                    <= breaking_threshold_bt * breaking_threshold_bt
                ):
                    output_index = car * MAX_CONTACTS_PER_CAR + contacts
                    refreshed_distance = refreshed_distance_bt * 50.0
                    contact_point[output_index] = refreshed_point_a_bt * 50.0
                    contact_local_a[output_index] = plane_local_a
                    contact_point_b[output_index] = refreshed_point_b_bt
                    contact_normal[output_index] = normal
                    contact_face[output_index] = -10 - plane
                    contact_mesh[output_index] = 16 + plane
                    contact_distance[output_index] = refreshed_distance
                    contact_distance_bt[output_index] = refreshed_distance_bt
                    contact_penetration[output_index] = wp.max(
                        0.0, -refreshed_distance
                    )
                    contact_lifetime[output_index] = 1
                    maximum_penetration = wp.max(
                        maximum_penetration, -refreshed_distance
                    )
                    contacts = contacts + 1
            else:
                contact_excess = contact_excess + 1

    solve_bt = 0
    if (plane_bt_mode & 4) != 0 and contacts > 0:
        # Bullet integrates every rigid body and contact row in Bullet units,
        # not only the analytic floor manifold.  Keeping mesh contacts in UU
        # introduces a scale round-trip before the next narrowphase transform.
        solve_bt = 1
    solver_pos_units = pos
    solver_pre_force_vel = pre_force_vel
    solver_force_vel = force_vel
    external_force_impulse = wp.vec3(0.0, 0.0, 0.0)
    external_torque_impulse = wp.vec3(0.0, 0.0, 0.0)
    if solve_bt != 0:
        solver_pos_units = rigid_position_bt[car]
        # wheel_pre_tick mutates the same Bullet rigid-body velocity that the
        # constraint solver subsequently reads. Preserve that BT value rather
        # than round-tripping it through the public UU state.
        solver_pre_force_vel = rigid_velocity_bt[car]
        applied_force_bt = total_force_bt[car] + wp.vec3(0.0, 0.0, -2340.0)
        _bullet_integrate_external_velocities(
            bullet_basis,
            applied_force_bt,
            total_torque_bt[car],
            external_force_impulse,
            external_torque_impulse,
        )
        # Solver-row setup reads base + external impulse, but Bullet writeback
        # first adds the accumulated solver delta to the base velocity and only
        # then applies the external impulse in the dynamics-world phase.
        solver_force_vel = solver_pre_force_vel + external_force_impulse
        force_ang_vel = pre_force_ang_vel + external_torque_impulse
        # Auto-roll torque is retained through the legacy state path until its
        # inverse-inertia/inertia construction is translated below. It is zero
        # throughout the frozen no-input static-world corpus.
        force_ang_vel = (
            force_ang_vel + auto_roll_angular_acceleration[car] * DT
        )

    # Set up every constraint from the same unchanged rigid-body state, as
    # Bullet does before applying warmstart or iteration deltas.
    for index in range(MAX_CONTACTS_PER_CAR):
        output_index = car * MAX_CONTACTS_PER_CAR + index
        if index < contacts:
            point = contact_point[output_index]
            if solve_bt != 0:
                point = _bullet_transform_point(
                    rigid_position_bt[car],
                    bullet_basis,
                    contact_local_a[output_index],
                )
            normal = contact_normal[output_index]
            tangent = wp.vec3(0.0, 0.0, 0.0)
            normal_jacobian = wp.float32(0.0)
            tangent_jacobian = wp.float32(0.0)
            normal_rhs = wp.float32(0.0)
            tangent_rhs = wp.float32(0.0)
            push_rhs = wp.float32(0.0)
            if solve_bt != 0:
                _bullet_contact_row(
                    rigid_position_bt[car],
                    bullet_basis,
                    point,
                    contact_point_b[output_index],
                    contact_distance_bt[output_index],
                    DT,
                    normal,
                    solver_pre_force_vel,
                    pre_force_ang_vel,
                    solver_force_vel,
                    force_ang_vel,
                    tangent,
                    normal_jacobian,
                    tangent_jacobian,
                    normal_rhs,
                    tangent_rhs,
                    push_rhs,
                )
            else:
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
                if normal_denominator > 1.0e-9:
                    normal_jacobian = 1.0 / normal_denominator
                pre_force_normal_speed = wp.dot(pre_force_point_velocity, normal)
                restitution = 0.0
                if wp.abs(pre_force_normal_speed) >= 10.0:
                    restitution = wp.max(
                        0.0, -CONTACT_RESTITUTION * pre_force_normal_speed
                    )
                normal_rhs = (
                    restitution - wp.dot(force_point_velocity, normal)
                ) * normal_jacobian
                tangent = _contact_tangent(normal, force_point_velocity, solve_bt)
                tangent_denominator = _impulse_denominator(
                    quat, offset, tangent, inertia_transpose_mix, solve_bt
                )
                if tangent_denominator > 1.0e-9:
                    tangent_jacobian = 1.0 / tangent_denominator
                tangent_rhs = (
                    -wp.dot(friction_rhs_point_velocity, tangent) * tangent_jacobian
                )
            contact_tangent[output_index] = tangent
            contact_normal_jacobian[output_index] = normal_jacobian
            contact_tangent_jacobian[output_index] = tangent_jacobian
            contact_normal_rhs[output_index] = normal_rhs
            contact_tangent_rhs[output_index] = tangent_rhs
            contact_push_rhs[output_index] = push_rhs
            contact_normal_impulse[output_index] = 0.0
            contact_tangent_impulse[output_index] = 0.0
            contact_push_impulse[output_index] = 0.0
            if contact_lifetime[output_index] <= 0:
                contact_lifetime[output_index] = 1
        else:
            contact_point[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_local_a[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_point_b[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_normal[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_tangent[output_index] = wp.vec3(0.0, 0.0, 0.0)
            contact_face[output_index] = -1
            contact_mesh[output_index] = -1
            contact_distance[output_index] = 0.0
            contact_distance_bt[output_index] = 0.0
            contact_penetration[output_index] = 0.0
            contact_normal_jacobian[output_index] = 0.0
            contact_tangent_jacobian[output_index] = 0.0
            contact_normal_rhs[output_index] = 0.0
            contact_tangent_rhs[output_index] = 0.0
            contact_push_rhs[output_index] = 0.0
            contact_normal_impulse[output_index] = 0.0
            contact_tangent_impulse[output_index] = 0.0
            contact_push_impulse[output_index] = 0.0
            contact_lifetime[output_index] = 0

    # RocketSim sets the split threshold to +1e30, so every penetrating static
    # contact is recovered through push/turn velocity, independently of bounce.
    push_vel = wp.vec3(0.0, 0.0, 0.0)
    turn_vel = wp.vec3(0.0, 0.0, 0.0)
    for _iteration in range(CONTACT_SOLVER_ITERATIONS):
        for index in range(MAX_CONTACTS_PER_CAR):
            if index < contacts:
                solver_contact = _bullet_solver_contact_index(
                    car, index, contacts, contact_mesh
                )
                output_index = car * MAX_CONTACTS_PER_CAR + solver_contact
                if solve_bt != 0:
                    point_a_bt = _bullet_transform_point(
                        rigid_position_bt[car],
                        bullet_basis,
                        contact_local_a[output_index],
                    )
                    if contact_push_rhs[output_index] != 0.0:
                        applied_push = wp.float32(contact_push_impulse[output_index])
                        _bullet_solve_split_row(
                            bullet_basis,
                            contact_normal[output_index],
                            point_a_bt - rigid_position_bt[car],
                            contact_normal_jacobian[output_index],
                            contact_push_rhs[output_index],
                            push_vel,
                            turn_vel,
                            applied_push,
                        )
                        contact_push_impulse[output_index] = applied_push
                else:
                    distance = contact_distance[output_index] + CONTACT_LINEAR_SLOP
                    if distance < 0.0:
                        normal = contact_normal[output_index]
                        point = contact_point[output_index]
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
        for index in range(MAX_CONTACTS_PER_CAR):
            if index < contacts:
                solver_contact = _bullet_solver_contact_index(
                    car, index, contacts, contact_mesh
                )
                output_index = car * MAX_CONTACTS_PER_CAR + solver_contact
                normal = contact_normal[output_index]
                if solve_bt != 0:
                    point_bt = _bullet_transform_point(
                        rigid_position_bt[car],
                        bullet_basis,
                        contact_local_a[output_index],
                    )
                    applied_normal = wp.float32(contact_normal_impulse[output_index])
                    _bullet_solve_velocity_row(
                        bullet_basis,
                        normal,
                        point_bt - rigid_position_bt[car],
                        contact_normal_jacobian[output_index],
                        contact_normal_rhs[output_index],
                        0.0,
                        1.0e10,
                        delta_vel,
                        delta_ang_vel,
                        applied_normal,
                    )
                    contact_normal_impulse[output_index] = applied_normal
                else:
                    point = contact_point[output_index]
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
        for index in range(MAX_CONTACTS_PER_CAR):
            if index < contacts:
                solver_contact = _bullet_solver_contact_index(
                    car, index, contacts, contact_mesh
                )
                output_index = car * MAX_CONTACTS_PER_CAR + solver_contact
                # Bullet skips a friction row when its paired normal impulse is
                # zero.  It does not clamp an impulse accumulated by an earlier
                # iteration back to zero, which matters as coupled contacts
                # transfer load during the PGS sweep.
                if contact_normal_impulse[output_index] > 0.0:
                    tangent = contact_tangent[output_index]
                    if solve_bt != 0:
                        point_bt = _bullet_transform_point(
                            rigid_position_bt[car],
                            bullet_basis,
                            contact_local_a[output_index],
                        )
                        limit = CONTACT_FRICTION * contact_normal_impulse[output_index]
                        applied_tangent = wp.float32(
                            contact_tangent_impulse[output_index]
                        )
                        _bullet_solve_velocity_row(
                            bullet_basis,
                            tangent,
                            point_bt - rigid_position_bt[car],
                            contact_tangent_jacobian[output_index],
                            contact_tangent_rhs[output_index],
                            -limit,
                            limit,
                            delta_vel,
                            delta_ang_vel,
                            applied_tangent,
                        )
                        contact_tangent_impulse[output_index] = applied_tangent
                    else:
                        point = contact_point[output_index]
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
    if solve_bt != 0:
        solved_vel = (solver_pre_force_vel + delta_vel) + external_force_impulse
        solved_ang_vel = (
            pre_force_ang_vel + delta_ang_vel
        ) + external_torque_impulse
    split_quat = quat
    has_split_correction = wp.int32(
        push_vel[0] != 0.0
        or push_vel[1] != 0.0
        or push_vel[2] != 0.0
        or turn_vel[0] != 0.0
        or turn_vel[1] != 0.0
        or turn_vel[2] != 0.0
    )
    if has_split_correction != 0:
        split_quat = _contact_integrate_quaternion(
            bullet_basis, turn_vel * CONTACT_SPLIT_TURN_ERP
        )
    # Bullet first writes the split-impulse transform back to the rigid body,
    # then integrates the final velocity from that corrected transform.
    split_pos = solver_pos_units + push_vel * DT
    solved_pos = split_pos + solved_vel * DT
    if solve_bt != 0:
        solved_pos_bt = wp.vec3(0.0, 0.0, 0.0)
        _bullet_integrate_position(
            rigid_position_bt[car],
            push_vel,
            solved_vel,
            DT,
            has_split_correction,
            solved_pos_bt,
        )
        solved_pos = solved_pos_bt
    if (
        solve_bt == 0
        and (plane_bt_mode & 8) != 0
        and contacts > 0
        and contact_face[car * MAX_CONTACTS_PER_CAR] < 0
    ):
        split_pos_bt = pos * 0.02 + push_vel * 0.02 * DT
        solved_pos = (split_pos_bt + solved_vel * 0.02 * DT) * 50.0
    solved_quat = _contact_integrate_quaternion(
        _bullet_quaternion_matrix(split_quat), solved_ang_vel
    )

    # RocketSim caps after Bullet has integrated the transform.
    public_pos = solved_pos
    public_vel = _contact_cap(solved_vel, 2300.0)
    if solve_bt != 0:
        capped_vel_bt = _contact_cap(solved_vel, 46.0)
        rigid_position_bt[car] = solved_pos
        rigid_velocity_bt[car] = capped_vel_bt
        public_pos = solved_pos * 50.0
        public_vel = capped_vel_bt * 50.0
        for index in range(MAX_CONTACTS_PER_CAR):
            if index < contacts:
                output_index = car * MAX_CONTACTS_PER_CAR + index
                contact_normal_impulse[output_index] = contact_normal_impulse[output_index] * 50.0
                contact_tangent_impulse[output_index] = contact_tangent_impulse[output_index] * 50.0
                contact_push_impulse[output_index] = contact_push_impulse[output_index] * 50.0
    else:
        rigid_position_bt[car] = public_pos * 0.02
        rigid_velocity_bt[car] = public_vel * 0.02
    car_pos[car] = public_pos
    car_quat[car] = solved_quat
    car_vel[car] = public_vel
    car_ang_vel[car] = _contact_cap(solved_ang_vel, 5.5)
    candidate_count[car] = candidates
    mesh_candidate_count[car] = retained_mesh_candidates
    mesh_candidate_overflow[car] = mesh_candidate_overflow[car] + mesh_candidate_excess
    contact_overflow[car] = contact_overflow[car] + contact_excess
    contact_count[car] = contacts
    world_contact_normal[car] = callback_normal
    candidate_total[car] = candidate_total[car] + float(candidates)
    contact_total[car] = contact_total[car] + float(contacts)
    candidate_max[car] = wp.max(candidate_max[car], candidates)
    contact_max[car] = wp.max(contact_max[car], contacts)
    penetration_max[car] = wp.max(penetration_max[car], maximum_penetration)
