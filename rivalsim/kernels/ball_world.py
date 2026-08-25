"""RocketSim/Bullet source-port for a Soccar sphere against the static world."""

from __future__ import annotations

import warp as wp

from rivalsim.ball_world_state import MAX_BALL_CONTACTS, MAX_BALL_MESH_CANDIDATES
from rivalsim.kernels.bullet_box_triangle import (
    bullet_internal_edge_best,
    bullet_manifold_replacement,
)
from rivalsim.kernels.vehicle import (
    _BULLET_CONTACT_ROW,
    _BULLET_SOLVE_SPLIT_ROW,
    _BULLET_SOLVE_VELOCITY_ROW,
    _bullet_integrate_position,
    _bullet_integrate_quaternion,
    _bullet_internal_edge_angle,
    _bullet_internal_edge_dot,
    _bullet_inverse_transform_point,
    _bullet_matrix_axis_angle_rotate,
    _bullet_quaternion_matrix,
    _bullet_sse_normalize,
    _bullet_transform_point,
    _bullet_vector_scale_add,
)

DT = 1.0 / 120.0
BALL_RADIUS_BT = 1.8249999284744263
BALL_QUERY_RADIUS_UU = (BALL_RADIUS_BT + 0.04) * 50.0
BALL_INV_MASS = 0.03333333507180214
BALL_INV_INERTIA_BT = 0.025020329281687737
BALL_DAMPING = 0.03
BALL_MAX_SPEED_BT = 120.0
BALL_MAX_ANGULAR_SPEED = 6.0
GRAVITY_BT = -13.0
CONTACT_BREAKING_BT = (BALL_RADIUS_BT + 0.08) * 0.02
CONTACT_FRICTION = 0.35
CONTACT_RESTITUTION = 0.6
CONTACT_ERP2 = 0.8
CONTACT_SPLIT_TURN_ERP = 0.1
CONTACT_SOLVER_ITERATIONS = 10
SIMD_EPSILON = 1.1920928955078125e-7


def _ball_solver_source(source: str) -> str:
    """Parameterize the already validated Bullet row port for the Soccar ball."""

    return (
        source.replace(
            "0.0185644571f, 0.0104337428f, 0.0075815497f",
            "0.0250203293f, 0.0250203293f, 0.0250203293f",
        )
        .replace(
            "0.0185644571f,0.0104337428f,0.0075815497f",
            "0.0250203293f,0.0250203293f,0.0250203293f",
        )
        .replace("0.00555555569f", "0.0333333351f")
        .replace("-0.3f", "-0.6f")
    )


_BULLET_BALL_CONTACT_ROW = _ball_solver_source(_BULLET_CONTACT_ROW)
_BULLET_BALL_SPECIAL_CONTACT_ROW = _BULLET_BALL_CONTACT_ROW.replace(
    "const RowV3 relative = sub(point, origin);",
    "const RowV3 relative = point;",
)
_BULLET_BALL_SOLVE_SPLIT_ROW = _ball_solver_source(_BULLET_SOLVE_SPLIT_ROW)
_BULLET_BALL_SOLVE_VELOCITY_ROW = _ball_solver_source(
    _BULLET_SOLVE_VELOCITY_ROW
)


@wp.func_native(_BULLET_BALL_CONTACT_ROW)
def _bullet_ball_contact_row(
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


@wp.func_native(_BULLET_BALL_SPECIAL_CONTACT_ROW)
def _bullet_ball_special_contact_row(
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


@wp.func_native(_BULLET_BALL_SOLVE_SPLIT_ROW)
def _bullet_ball_solve_split_row(
    basis: wp.mat33,
    direction: wp.vec3,
    relative_position_bt: wp.vec3,
    jacobian: float,
    rhs: float,
    push_velocity_bt: wp.ref[wp.vec3],
    turn_velocity_world: wp.ref[wp.vec3],
    applied_push_impulse: wp.ref[wp.float32],
): ...


@wp.func_native(_BULLET_BALL_SOLVE_VELOCITY_ROW)
def _bullet_ball_solve_velocity_row(
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


_BULLET_SPHERE_PLANE_CONTACT = r"""
    // Literal fixed-sphere translation of
    // btConvexPlaneCollisionAlgorithm::processCollision.  Even for an
    // identity-basis arena plane, Bullet rotates -planeNormal through
    // planeInConvex, normalizes the sphere support direction with the pinned
    // host's RSQRTSS path, transforms that support through convexInPlane, and
    // only then projects the point on B and reconstructs point A.
    auto add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
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
    struct PlaneSphereV3 { float x; float y; float z; };
    auto make = [](float x, float y, float z) -> PlaneSphereV3 {
        PlaneSphereV3 value = {x, y, z};
        return value;
    };
    auto vadd = [&](PlaneSphereV3 a, PlaneSphereV3 b) -> PlaneSphereV3 {
        return make(add(a.x, b.x), add(a.y, b.y), add(a.z, b.z));
    };
    auto vsub = [&](PlaneSphereV3 a, PlaneSphereV3 b) -> PlaneSphereV3 {
        return make(sub(a.x, b.x), sub(a.y, b.y), sub(a.z, b.z));
    };
    auto scale = [&](PlaneSphereV3 value, float amount) -> PlaneSphereV3 {
        return make(
            mul(value.x, amount),
            mul(value.y, amount),
            mul(value.z, amount));
    };
    auto dot = [&](PlaneSphereV3 a, PlaneSphereV3 b) -> float {
        // btVector3 SSE dot adds X/Y first, then Z.
        return add(add(mul(a.x, b.x), mul(a.y, b.y)), mul(a.z, b.z));
    };
    auto matrix_vector = [&](PlaneSphereV3 value) -> PlaneSphereV3 {
        return make(
            add(add(mul(basis.data[0][0], value.x),
                    mul(basis.data[0][1], value.y)),
                mul(basis.data[0][2], value.z)),
            add(add(mul(basis.data[1][0], value.x),
                    mul(basis.data[1][1], value.y)),
                mul(basis.data[1][2], value.z)),
            add(add(mul(basis.data[2][0], value.x),
                    mul(basis.data[2][1], value.y)),
                mul(basis.data[2][2], value.z)));
    };
    auto transpose_matrix_vector = [&](PlaneSphereV3 value) -> PlaneSphereV3 {
        return make(
            add(add(mul(basis.data[0][0], value.x),
                    mul(basis.data[1][0], value.y)),
                mul(basis.data[2][0], value.z)),
            add(add(mul(basis.data[0][1], value.x),
                    mul(basis.data[1][1], value.y)),
                mul(basis.data[2][1], value.z)),
            add(add(mul(basis.data[0][2], value.x),
                    mul(basis.data[1][2], value.y)),
                mul(basis.data[2][2], value.z)));
    };
    auto sse_normalize = [&](PlaneSphereV3 value) -> PlaneSphereV3 {
        const float length_squared = dot(value, value);
        float inverse_length;
    #if defined(__CUDA_ARCH__)
        const unsigned input_bits = __float_as_uint(length_squared);
        const unsigned exponent = (input_bits >> 23) & 0xffu;
        const unsigned result_exponent = ((380u - exponent) >> 1) << 23;
        const unsigned index = (input_bits >> 11) & 0x1fffu;
        const unsigned estimate_mantissa =
            static_cast<unsigned>(rsqrtss_mantissa.data[index]) << 11;
        inverse_length = __uint_as_float(result_exponent | estimate_mantissa);
    #elif defined(__clang__) && (defined(__x86_64__) || defined(_M_X64))
        typedef float PlaneSphereFloat4 __attribute__((__vector_size__(16)));
        const PlaneSphereFloat4 estimate_input = {
            length_squared, 0.0f, 0.0f, 0.0f};
        const PlaneSphereFloat4 estimate = __builtin_ia32_rsqrtss(estimate_input);
        inverse_length = estimate[0];
    #elif defined(_MSC_VER)
        const __m128 estimate_input = _mm_set_ss(length_squared);
        inverse_length = _mm_cvtss_f32(_mm_rsqrt_ss(estimate_input));
    #else
        inverse_length = 1.0f / sqrtf(length_squared);
    #endif
        float correction = mul(mul(length_squared, 0.5f), inverse_length);
        correction = mul(correction, inverse_length);
        correction = sub(1.5f, correction);
        inverse_length = mul(inverse_length, correction);
        return scale(value, inverse_length);
    };

    const PlaneSphereV3 center = make(
        center_bt[0], center_bt[1], center_bt[2]);
    const PlaneSphereV3 plane_origin = make(
        plane_origin_bt[0], plane_origin_bt[1], plane_origin_bt[2]);
    const PlaneSphereV3 normal = make(
        plane_normal[0], plane_normal[1], plane_normal[2]);
    const PlaneSphereV3 negative_normal = make(
        -normal.x, -normal.y, -normal.z);
    const PlaneSphereV3 local_direction =
        transpose_matrix_vector(negative_normal);
    const PlaneSphereV3 local_support = scale(
        sse_normalize(local_direction), 1.8249999284744263f);

    // planeWorld has identity basis and zero local plane constant for all four
    // fixed Soccar analytic planes.
    const PlaneSphereV3 convex_in_plane_origin = vsub(center, plane_origin);
    const PlaneSphereV3 support_in_plane = vadd(
        matrix_vector(local_support), convex_in_plane_origin);
    const float depth = dot(normal, support_in_plane);
    const PlaneSphereV3 projected_in_plane =
        vsub(support_in_plane, scale(normal, depth));
    const PlaneSphereV3 point_b = vadd(projected_in_plane, plane_origin);
    const PlaneSphereV3 point_a = vadd(point_b, scale(normal, depth));

    point_a_bt = wp::vec_t<3, wp::float32>(point_a.x, point_a.y, point_a.z);
    point_b_bt = wp::vec_t<3, wp::float32>(point_b.x, point_b.y, point_b.z);
    distance_bt = depth;
"""


@wp.func_native(_BULLET_SPHERE_PLANE_CONTACT)
def _bullet_sphere_plane_contact(
    center_bt: wp.vec3,
    basis: wp.mat33,
    rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    plane_origin_bt: wp.vec3,
    plane_normal: wp.vec3,
    point_a_bt: wp.ref[wp.vec3],
    point_b_bt: wp.ref[wp.vec3],
    distance_bt: wp.ref[wp.float32],
): ...


@wp.func
def _sphere_closest_point_triangle(
    point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3
) -> wp.vec3:
    """Literal branch order from RocketSim's closestPointTriangle."""

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

    cp = point - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        return a + v * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + v * (c - b)

    inverse = 1.0 / (va + vb + vc)
    v = vb * inverse
    w = vc * inverse
    return a + v * ab + w * ac


@wp.func
def _sphere_face_contains(
    point: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3
) -> bool:
    u = b - a
    v = c - a
    normal = wp.cross(u, v)
    normal_length_sq = wp.dot(normal, normal)
    relative = point - a
    gamma = wp.dot(wp.cross(u, relative), normal) / normal_length_sq
    beta = wp.dot(wp.cross(relative, v), normal) / normal_length_sq
    alpha = 1.0 - gamma - beta
    return (
        0.0 <= alpha
        and alpha <= 1.0
        and 0.0 <= beta
        and beta <= 1.0
        and 0.0 <= gamma
        and gamma <= 1.0
    )


_BULLET_SPHERE_TRIANGLE_CONTACT = r"""
    struct SphereV3 {
        float x;
        float y;
        float z;
    };
    auto add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
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
    auto div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    auto make = [](const wp::vec_t<3, wp::float32>& value) -> SphereV3 {
        return SphereV3{value[0], value[1], value[2]};
    };
    auto vadd = [&](SphereV3 a, SphereV3 b) -> SphereV3 {
        return SphereV3{add(a.x, b.x), add(a.y, b.y), add(a.z, b.z)};
    };
    auto vsub = [&](SphereV3 a, SphereV3 b) -> SphereV3 {
        return SphereV3{sub(a.x, b.x), sub(a.y, b.y), sub(a.z, b.z)};
    };
    auto scale = [&](SphereV3 value, float amount) -> SphereV3 {
        return SphereV3{
            mul(value.x, amount), mul(value.y, amount), mul(value.z, amount)};
    };
    auto dot = [&](SphereV3 a, SphereV3 b) -> float {
        // Pinned btVector3 SSE dot: add X/Y first, then Z.
        return add(add(mul(a.x, b.x), mul(a.y, b.y)), mul(a.z, b.z));
    };
    auto cross = [&](SphereV3 a, SphereV3 b) -> SphereV3 {
        return SphereV3{
            sub(mul(a.y, b.z), mul(a.z, b.y)),
            sub(mul(a.z, b.x), mul(a.x, b.z)),
            sub(mul(a.x, b.y), mul(a.y, b.x))};
    };
    auto closest_point_triangle = [&](SphereV3 p, SphereV3 a, SphereV3 b,
                                      SphereV3 c) -> SphereV3 {
        const SphereV3 ab = vsub(b, a);
        const SphereV3 ac = vsub(c, a);
        const SphereV3 ap = vsub(p, a);
        const float d1 = dot(ab, ap);
        const float d2 = dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) return a;

        const SphereV3 bp = vsub(p, b);
        const float d3 = dot(ab, bp);
        const float d4 = dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) return b;

        const SphereV3 cp = vsub(p, c);
        const float d5 = dot(ab, cp);
        const float d6 = dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) return c;

        const float vc = sub(mul(d1, d4), mul(d3, d2));
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float amount = div(d1, sub(d1, d3));
            return vadd(a, scale(ab, amount));
        }

        const float vb = sub(mul(d5, d2), mul(d1, d6));
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float amount = div(d2, sub(d2, d6));
            return vadd(a, scale(ac, amount));
        }

        const float va = sub(mul(d3, d6), mul(d5, d4));
        const float d43 = sub(d4, d3);
        const float d56 = sub(d5, d6);
        if (va <= 0.0f && d43 >= 0.0f && d56 >= 0.0f) {
            const float amount = div(d43, add(d43, d56));
            return vadd(b, scale(vsub(c, b), amount));
        }

        const float inverse = div(1.0f, add(add(va, vb), vc));
        const float amount_v = mul(vb, inverse);
        const float amount_w = mul(vc, inverse);
        return vadd(vadd(a, scale(ab, amount_v)), scale(ac, amount_w));
    };
    auto face_contains = [&](SphereV3 p, SphereV3 a, SphereV3 b,
                             SphereV3 c) -> bool {
        const SphereV3 u = vsub(b, a);
        const SphereV3 v = vsub(c, a);
        const SphereV3 n = cross(u, v);
        const float n_length_squared = dot(n, n);
        const SphereV3 w = vsub(p, a);
        const float gamma = div(dot(cross(u, w), n), n_length_squared);
        const float beta = div(dot(cross(w, v), n), n_length_squared);
        const float alpha = sub(sub(1.0f, gamma), beta);
        return 0.0f <= alpha && alpha <= 1.0f
            && 0.0f <= beta && beta <= 1.0f
            && 0.0f <= gamma && gamma <= 1.0f;
    };
    auto sse_normalize = [&](SphereV3 value) -> SphereV3 {
        const float length_squared = dot(value, value);
        float inverse_length;
    #if defined(__CUDA_ARCH__)
        // btVector3::normalize executes RSQRTSS on the pinned RocketSim
        // authority host. CUDA's rsqrtf estimate is different, so reproduce
        // the authority CPU's mantissa lookup before Bullet's Newton step.
        const unsigned input_bits = __float_as_uint(length_squared);
        const unsigned exponent = (input_bits >> 23) & 0xffu;
        const unsigned result_exponent = ((380u - exponent) >> 1) << 23;
        const unsigned index = (input_bits >> 11) & 0x1fffu;
        const unsigned estimate_mantissa =
            static_cast<unsigned>(rsqrtss_mantissa.data[index]) << 11;
        inverse_length = __uint_as_float(result_exponent | estimate_mantissa);
    #elif defined(__clang__) && (defined(__x86_64__) || defined(_M_X64))
        typedef float SphereFloat4 __attribute__((__vector_size__(16)));
        const SphereFloat4 estimate_input = {
            length_squared, 0.0f, 0.0f, 0.0f};
        const SphereFloat4 estimate = __builtin_ia32_rsqrtss(estimate_input);
        inverse_length = estimate[0];
    #elif defined(_MSC_VER)
        const __m128 estimate_input = _mm_set_ss(length_squared);
        inverse_length = _mm_cvtss_f32(_mm_rsqrt_ss(estimate_input));
    #else
        inverse_length = div(1.0f, sqrtf(length_squared));
    #endif
        float correction = mul(mul(length_squared, 0.5f), inverse_length);
        correction = mul(correction, inverse_length);
        correction = sub(1.5f, correction);
        inverse_length = mul(inverse_length, correction);
        return scale(value, inverse_length);
    };

    const SphereV3 center = make(center_bt);
    const SphereV3 vertex_a = make(a_bt);
    const SphereV3 vertex_b = make(b_bt);
    const SphereV3 vertex_c = make(c_bt);
    const float radius = 1.8249999284744263f;
    const float radius_with_threshold = add(radius, 0.038099996745586395f);
    auto cofactor = [&](int row_a, int column_a, int row_b,
                        int column_b) -> float {
        return sub(
            mul(basis.data[row_a][column_a], basis.data[row_b][column_b]),
            mul(basis.data[row_a][column_b], basis.data[row_b][column_a]));
    };
    const SphereV3 inverse_co = SphereV3{
        cofactor(1, 1, 2, 2),
        cofactor(1, 2, 2, 0),
        cofactor(1, 0, 2, 1)};
    const SphereV3 basis_row_zero = SphereV3{
        basis.data[0][0], basis.data[0][1], basis.data[0][2]};
    const float inverse_scale = div(1.0f, dot(basis_row_zero, inverse_co));
    float inverse_basis[3][3] = {
        {mul(inverse_co.x, inverse_scale),
         mul(cofactor(0, 2, 2, 1), inverse_scale),
         mul(cofactor(0, 1, 1, 2), inverse_scale)},
        {mul(inverse_co.y, inverse_scale),
         mul(cofactor(0, 0, 2, 2), inverse_scale),
         mul(cofactor(0, 2, 1, 0), inverse_scale)},
        {mul(inverse_co.z, inverse_scale),
         mul(cofactor(0, 1, 2, 0), inverse_scale),
         mul(cofactor(0, 0, 1, 1), inverse_scale)}};
    auto matrix_vector = [&](const float matrix[3][3], SphereV3 value) -> SphereV3 {
        return SphereV3{
            add(add(mul(matrix[0][0], value.x), mul(matrix[0][1], value.y)),
                mul(matrix[0][2], value.z)),
            add(add(mul(matrix[1][0], value.x), mul(matrix[1][1], value.y)),
                mul(matrix[1][2], value.z)),
            add(add(mul(matrix[2][0], value.x), mul(matrix[2][1], value.y)),
                mul(matrix[2][2], value.z))};
    };
    float forward_basis[3][3] = {
        {basis.data[0][0], basis.data[0][1], basis.data[0][2]},
        {basis.data[1][0], basis.data[1][1], basis.data[1][2]},
        {basis.data[2][0], basis.data[2][1], basis.data[2][2]}};
    auto early_separated = [&](SphereV3 triangle_normal) -> bool {
        const SphereV3 local_direction = matrix_vector(
            inverse_basis, triangle_normal);
        const SphereV3 local_support = scale(
            sse_normalize(local_direction), radius);
        const SphereV3 world_support = vadd(
            matrix_vector(forward_basis, local_support), center);
        const float projected_support = dot(triangle_normal, world_support);
        const float projected_triangle = dot(triangle_normal, vertex_a);
        return sub(projected_triangle, projected_support)
            > 0.038099996745586395f;
    };
    SphereV3 early_normal = sse_normalize(
        cross(vsub(vertex_b, vertex_a), vsub(vertex_c, vertex_a)));
    const bool separated_forward = early_separated(early_normal);
    early_normal = scale(early_normal, -1.0f);
    const bool separated_backward = early_separated(early_normal);
    SphereV3 normal = cross(vsub(vertex_b, vertex_a), vsub(vertex_c, vertex_a));
    const float normal_length_squared = dot(normal, normal);
    valid = 0;
    if (!separated_forward && !separated_backward
        && normal_length_squared >= 1.4210854715202004e-14f) {
        const float inverse_length = div(1.0f, sqrtf(normal_length_squared));
        normal = scale(normal, inverse_length);
        const SphereV3 point_to_center = vsub(center, vertex_a);
        float plane_distance = dot(point_to_center, normal);
        if (plane_distance < 0.0f) {
            plane_distance = mul(plane_distance, -1.0f);
            normal = scale(normal, -1.0f);
        }
        if (plane_distance < radius_with_threshold) {
            bool has_contact = false;
            SphereV3 contact_point{0.0f, 0.0f, 0.0f};
            if (face_contains(center, vertex_a, vertex_b, vertex_c)) {
                has_contact = true;
                contact_point = vsub(center, scale(normal, plane_distance));
            } else {
                contact_point = closest_point_triangle(
                    center, vertex_a, vertex_b, vertex_c);
                const SphereV3 difference = vsub(contact_point, center);
                const float capsule_distance_squared = dot(difference, difference);
                if (capsule_distance_squared
                    < mul(radius_with_threshold, radius_with_threshold)) {
                    has_contact = true;
                }
            }
            if (has_contact) {
                const SphereV3 contact_to_center = vsub(center, contact_point);
                const float distance_squared = dot(contact_to_center, contact_to_center);
                if (distance_squared
                    < mul(radius_with_threshold, radius_with_threshold)) {
                    float depth = -radius;
                    SphereV3 result_normal = normal;
                    if (distance_squared > 1.1920928955078125e-7f) {
                        const float distance = sqrtf(distance_squared);
                        result_normal = sse_normalize(contact_to_center);
                        depth = -sub(radius, distance);
                    }
                    point_b = wp::vec_t<3, wp::float32>(
                        contact_point.x, contact_point.y, contact_point.z);
                    normal_out = wp::vec_t<3, wp::float32>(
                        result_normal.x, result_normal.y, result_normal.z);
                    distance_out = depth;
                    valid = 1;
                }
            }
        }
    }
"""


@wp.func_native(_BULLET_SPHERE_TRIANGLE_CONTACT)
def _bullet_sphere_triangle_contact(
    center_bt: wp.vec3,
    basis: wp.mat33,
    rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    a_bt: wp.vec3,
    b_bt: wp.vec3,
    c_bt: wp.vec3,
    point_b: wp.ref[wp.vec3],
    normal_out: wp.ref[wp.vec3],
    distance_out: wp.ref[wp.float32],
    valid: wp.ref[wp.int32],
): ...


@wp.func
def _sphere_triangle_contact(
    center_bt: wp.vec3,
    basis: wp.mat33,
    rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    point_b: wp.ref[wp.vec3],
    normal_out: wp.ref[wp.vec3],
    distance_out: wp.ref[wp.float32],
) -> int:
    valid = wp.int32(0)
    _bullet_sphere_triangle_contact(
        center_bt,
        basis,
        rsqrtss_mantissa,
        a,
        b,
        c,
        point_b,
        normal_out,
        distance_out,
        valid,
    )
    return valid


@wp.func
def _copy_contact(
    source: int,
    destination: int,
    local_a: wp.array(dtype=wp.vec3),
    point_b: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=wp.float32),
    face: wp.array(dtype=wp.int32),
    mesh: wp.array(dtype=wp.int32),
    lifetime: wp.array(dtype=wp.int32),
    normal_impulse: wp.array(dtype=wp.float32),
    tangent_impulse: wp.array(dtype=wp.float32),
    tangent: wp.array(dtype=wp.vec3),
):
    local_a[destination] = local_a[source]
    point_b[destination] = point_b[source]
    normal[destination] = normal[source]
    distance[destination] = distance[source]
    face[destination] = face[source]
    mesh[destination] = mesh[source]
    lifetime[destination] = lifetime[source]
    normal_impulse[destination] = normal_impulse[source]
    tangent_impulse[destination] = tangent_impulse[source]
    tangent[destination] = tangent[source]


@wp.func
def _manifold_replacement(
    candidate: wp.vec3,
    candidate_distance: float,
    start: int,
    local_a: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=wp.float32),
) -> int:
    return bullet_manifold_replacement(
        candidate,
        local_a[start],
        local_a[start + 1],
        local_a[start + 2],
        local_a[start + 3],
        candidate_distance,
        distance[start],
        distance[start + 1],
        distance[start + 2],
        distance[start + 3],
    )


@wp.func
def _plane_space_first(normal: wp.vec3) -> wp.vec3:
    if wp.abs(normal[2]) > 0.7071067811865476:
        inverse = 1.0 / wp.sqrt(normal[1] * normal[1] + normal[2] * normal[2])
        return wp.vec3(0.0, -normal[2] * inverse, normal[1] * inverse)
    inverse = 1.0 / wp.sqrt(normal[0] * normal[0] + normal[1] * normal[1])
    return wp.vec3(-normal[1] * inverse, normal[0] * inverse, 0.0)


@wp.func
def _cap(value: wp.vec3, maximum: float) -> wp.vec3:
    length_sq = wp.dot(value, value)
    if length_sq > maximum * maximum:
        return value * (maximum / wp.sqrt(length_sq))
    return value


@wp.func
def _ball_solver_contact_index(
    env: int,
    solver_index: int,
    contacts: int,
    contact_mesh: wp.array(dtype=wp.int32),
) -> int:
    """Map a contact ordinal through Bullet's equal-island manifold sort."""

    base = env * MAX_BALL_CONTACTS
    group_count = wp.int32(0)
    previous_mesh = wp.int32(-1)
    for index in range(MAX_BALL_CONTACTS):
        if index < contacts:
            mesh = contact_mesh[base + index]
            if index == 0 or mesh != previous_mesh:
                group_count = group_count + 1
            previous_mesh = mesh

    # Four-bit source-manifold ordinals, least-significant nibble first.
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
    elif group_count == 13:
        permutation = wp.uint64(0x3450126ABC789)
    elif group_count == 14:
        permutation = wp.uint64(0x4563012BCDA789)
    elif group_count == 15:
        permutation = wp.uint64(0x45630127CDEB89A)
    elif group_count == 16:
        permutation = wp.uint64(0x54761032DCFE98BA)

    remaining = solver_index
    mapped_index = solver_index
    mapped = wp.int32(0)
    for solver_group in range(MAX_BALL_CONTACTS):
        if solver_group < group_count and mapped == 0:
            shift = wp.uint64(solver_group * 4)
            source_group = wp.int32((permutation >> shift) & wp.uint64(0xF))
            group_ordinal = wp.int32(-1)
            group_start = wp.int32(0)
            group_size = wp.int32(0)
            source_previous_mesh = wp.int32(-1)
            for source_index in range(MAX_BALL_CONTACTS):
                if source_index < contacts:
                    source_mesh = contact_mesh[base + source_index]
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


@wp.kernel(enable_backward=False)
def debug_ball_candidate_stream(
    vertices_bt: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    source_position_bt: wp.array(dtype=wp.vec3),
    source_quaternion: wp.array(dtype=wp.quat),
    candidate_count: wp.array(dtype=wp.int32),
    candidate_face: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    distance_bt_out: wp.array(dtype=wp.float32),
    normal_out: wp.array(dtype=wp.vec3),
    point_b_out: wp.array(dtype=wp.vec3),
    point_a_out: wp.array(dtype=wp.vec3),
    local_a_out: wp.array(dtype=wp.vec3),
):
    """Diagnostic view of the already ordered Phase A candidate stream."""

    index = wp.tid()
    env = index // MAX_BALL_MESH_CANDIDATES
    relative = index - env * MAX_BALL_MESH_CANDIDATES
    valid[index] = 0
    if relative < candidate_count[env]:
        face = candidate_face[index]
        triangle_offset = face * 3
        a = vertices_bt[triangle_indices[triangle_offset]]
        b = vertices_bt[triangle_indices[triangle_offset + 1]]
        c = vertices_bt[triangle_indices[triangle_offset + 2]]
        point_b_bt = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 0.0)
        distance_bt = wp.float32(0.0)
        is_valid = _sphere_triangle_contact(
            source_position_bt[env],
            _bullet_quaternion_matrix(source_quaternion[env]),
            rsqrtss_mantissa,
            a,
            b,
            c,
            point_b_bt,
            normal,
            distance_bt,
        )
        valid[index] = is_valid
        distance_bt_out[index] = distance_bt
        normal_out[index] = normal
        point_b_out[index] = point_b_bt
        point_a_bt = _bullet_vector_scale_add(point_b_bt, normal, distance_bt)
        point_a_out[index] = point_a_bt
        local_a_out[index] = _bullet_inverse_transform_point(
            source_position_bt[env],
            _bullet_quaternion_matrix(source_quaternion[env]),
            point_a_bt,
        )


@wp.kernel(enable_backward=False)
def debug_ball_manifold_replay(
    candidate_count: wp.array(dtype=wp.int32),
    candidate_local_a: wp.array(dtype=wp.vec3),
    candidate_distance: wp.array(dtype=wp.float32),
    candidate_face: wp.array(dtype=wp.int32),
    retained_face_after_step: wp.array(dtype=wp.int32),
):
    """Replay Bullet's four-point reduction from an authority witness stream."""

    if wp.tid() == 0:
        retained_local_a = wp.vec3(0.0, 0.0, 0.0)
        retained_local_b = wp.vec3(0.0, 0.0, 0.0)
        retained_local_c = wp.vec3(0.0, 0.0, 0.0)
        retained_local_d = wp.vec3(0.0, 0.0, 0.0)
        retained_distance_a = wp.float32(0.0)
        retained_distance_b = wp.float32(0.0)
        retained_distance_c = wp.float32(0.0)
        retained_distance_d = wp.float32(0.0)
        retained_face_a = wp.int32(-1)
        retained_face_b = wp.int32(-1)
        retained_face_c = wp.int32(-1)
        retained_face_d = wp.int32(-1)
        retained_count = wp.int32(0)
        for step in range(MAX_BALL_MESH_CANDIDATES):
            if step < candidate_count[0]:
                point = candidate_local_a[step]
                distance = candidate_distance[step]
                face = candidate_face[step]
                if retained_count == 0:
                    retained_local_a = point
                    retained_distance_a = distance
                    retained_face_a = face
                    retained_count = 1
                elif retained_count == 1:
                    retained_local_b = point
                    retained_distance_b = distance
                    retained_face_b = face
                    retained_count = 2
                elif retained_count == 2:
                    retained_local_c = point
                    retained_distance_c = distance
                    retained_face_c = face
                    retained_count = 3
                elif retained_count == 3:
                    retained_local_d = point
                    retained_distance_d = distance
                    retained_face_d = face
                    retained_count = 4
                else:
                    replacement = bullet_manifold_replacement(
                        point,
                        retained_local_a,
                        retained_local_b,
                        retained_local_c,
                        retained_local_d,
                        distance,
                        retained_distance_a,
                        retained_distance_b,
                        retained_distance_c,
                        retained_distance_d,
                    )
                    if replacement == 0:
                        retained_local_a = point
                        retained_distance_a = distance
                        retained_face_a = face
                    elif replacement == 1:
                        retained_local_b = point
                        retained_distance_b = distance
                        retained_face_b = face
                    elif replacement == 2:
                        retained_local_c = point
                        retained_distance_c = distance
                        retained_face_c = face
                    else:
                        retained_local_d = point
                        retained_distance_d = distance
                        retained_face_d = face
                output = step * 4
                retained_face_after_step[output] = retained_face_a
                retained_face_after_step[output + 1] = retained_face_b
                retained_face_after_step[output + 2] = retained_face_c
                retained_face_after_step[output + 3] = retained_face_d


@wp.kernel(enable_backward=False)
def initialize_ball_world_internal(
    ball_pos_uu: wp.array(dtype=wp.vec3),
    ball_vel_uu: wp.array(dtype=wp.vec3),
    position_bt: wp.array(dtype=wp.vec3),
    velocity_bt: wp.array(dtype=wp.vec3),
    broadphase_proxy_min_bt: wp.array(dtype=wp.vec3),
):
    env = wp.tid()
    position_bt[env] = ball_pos_uu[env] * 0.02
    velocity_bt[env] = ball_vel_uu[env] * 0.02
    # Ball::_BulletSetup creates the proxy before SetState relocates the ball.
    # The initial sphere center is (0, 0, 1.825) BT and updateSingleAabb adds
    # the pinned 0.08 BT contact threshold to the 1.825 BT sphere radius.
    broadphase_proxy_min_bt[env] = wp.vec3(
        -1.9049999713897705, -1.9049999713897705, -0.08000004291534424
    )


@wp.kernel(enable_backward=False)
def ball_world_tick(
    mesh_id: wp.uint64,
    vertices_bt: wp.array(dtype=wp.vec3),
    triangle_indices: wp.array(dtype=wp.int32),
    rsqrtss_mantissa: wp.array(dtype=wp.uint16),
    internal_edge_face_normals: wp.array(dtype=wp.vec3),
    internal_edge_crosses: wp.array(dtype=wp.vec3),
    internal_edge_normal_bs: wp.array(dtype=wp.vec3),
    internal_edge_angles: wp.array(dtype=wp.vec3),
    internal_edge_flags: wp.array(dtype=wp.int32),
    bullet_bvh_rank: wp.array(dtype=wp.int32),
    face_mesh_index: wp.array(dtype=wp.int32),
    ball_pos: wp.array(dtype=wp.vec3),
    ball_vel: wp.array(dtype=wp.vec3),
    ball_quat: wp.array(dtype=wp.quat),
    ball_ang_vel: wp.array(dtype=wp.vec3),
    resident_position_bt: wp.array(dtype=wp.vec3),
    resident_velocity_bt: wp.array(dtype=wp.vec3),
    broadphase_proxy_min_bt: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
    candidate_count: wp.array(dtype=wp.int32),
    candidate_overflow: wp.array(dtype=wp.int32),
    contact_overflow: wp.array(dtype=wp.int32),
    contact_local_a_bt: wp.array(dtype=wp.vec3),
    contact_point_b_bt: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_distance_bt: wp.array(dtype=wp.float32),
    contact_face: wp.array(dtype=wp.int32),
    contact_mesh: wp.array(dtype=wp.int32),
    contact_lifetime: wp.array(dtype=wp.int32),
    contact_normal_impulse: wp.array(dtype=wp.float32),
    contact_tangent_impulse: wp.array(dtype=wp.float32),
    contact_tangent: wp.array(dtype=wp.vec3),
    contact_normal_jacobian: wp.array(dtype=wp.float32),
    contact_tangent_jacobian: wp.array(dtype=wp.float32),
    contact_normal_rhs: wp.array(dtype=wp.float32),
    contact_tangent_rhs: wp.array(dtype=wp.float32),
    contact_push_rhs: wp.array(dtype=wp.float32),
    contact_push_impulse: wp.array(dtype=wp.float32),
    candidate_face: wp.array(dtype=wp.int32),
):
    env = wp.tid()
    contact_base = env * MAX_BALL_CONTACTS
    candidate_base = env * MAX_BALL_MESH_CANDIDATES
    position_bt = resident_position_bt[env]
    # btDiscreteDynamicsWorld::stepSimulation calls updateAabbs before
    # collision detection and before transform integration. Retain that
    # starting-transform proxy minimum for the following tick's vehicle rays.
    # The extent is sphere radius 1.825 + RocketSim's 0.08 sphere change +
    # updateSingleAabb's 0.02 gContactBreakingThreshold expansion.
    broadphase_proxy_min_bt[env] = position_bt - wp.vec3(
        1.9249999523162842, 1.9249999523162842, 1.9249999523162842
    )
    quaternion = ball_quat[env]
    basis = _bullet_quaternion_matrix(quaternion)
    pre_velocity_bt = resident_velocity_bt[env]
    pre_angular = ball_ang_vel[env]

    # Arena::Step marks an exactly motionless ball ISLAND_SLEEPING before
    # Bullet begins the step.  The static arena bodies are inactive too, so
    # btCollisionDispatcher::needsCollision rejects every ball/world pair.
    # An active car/ball pair can join the bodies into one awake island later
    # in collision dispatch; car_ball_tick then performs that island's solver
    # writeback and transform integration without retroactively adding gravity.
    if wp.dot(pre_velocity_bt, pre_velocity_bt) == 0.0 and wp.dot(
        pre_angular, pre_angular
    ) == 0.0:
        contact_count[env] = 0
        candidate_count[env] = 0
        candidate_overflow[env] = 0
        contact_overflow[env] = 0
        ball_pos[env] = position_bt * 50.0
        ball_vel[env] = pre_velocity_bt * 50.0
        return

    pre_velocity_bt = pre_velocity_bt * wp.pow(1.0 - BALL_DAMPING, DT)
    external_force_impulse_bt = wp.vec3(0.0, 0.0, GRAVITY_BT * DT)
    force_velocity_bt = pre_velocity_bt + external_force_impulse_bt

    # RocketSim's btRSBroadphase removes every active broadphase pair before
    # rebuilding the current tick's overlap set.  Pair removal destroys the
    # collision algorithm, whose btConvexTriangleCallback destructor clears
    # its persistent manifold.  Ball/world points therefore persist across
    # the triangle witness stream within one tick, but never across ticks.
    contacts = wp.int32(0)
    query_center_uu = position_bt * 50.0
    query_half = wp.vec3(
        BALL_QUERY_RADIUS_UU, BALL_QUERY_RADIUS_UU, BALL_QUERY_RADIUS_UU
    )
    retained_candidates = wp.int32(0)
    total_candidates = wp.int32(0)
    query = wp.mesh_query_aabb(
        mesh_id, query_center_uu - query_half, query_center_uu + query_half
    )
    for face in query:
        total_candidates = total_candidates + 1
        mesh = face_mesh_index[face]
        rank = bullet_bvh_rank[face]
        insert = retained_candidates
        duplicate = wp.int32(0)
        for existing in range(MAX_BALL_MESH_CANDIDATES):
            if existing < retained_candidates:
                existing_face = candidate_face[candidate_base + existing]
                if existing_face == face:
                    duplicate = 1
                existing_mesh = face_mesh_index[existing_face]
                existing_rank = bullet_bvh_rank[existing_face]
                if insert == retained_candidates and (
                    mesh < existing_mesh
                    or (mesh == existing_mesh and rank < existing_rank)
                ):
                    insert = existing
        if duplicate == 0:
            destination_limit = wp.min(
                retained_candidates, MAX_BALL_MESH_CANDIDATES - 1
            )
            for shift in range(MAX_BALL_MESH_CANDIDATES - 1):
                destination = destination_limit - shift
                if destination > insert:
                    candidate_face[candidate_base + destination] = candidate_face[
                        candidate_base + destination - 1
                    ]
            if insert < MAX_BALL_MESH_CANDIDATES:
                candidate_face[candidate_base + insert] = face
            if retained_candidates < MAX_BALL_MESH_CANDIDATES:
                retained_candidates = retained_candidates + 1

    candidate_count[env] = total_candidates
    candidate_overflow[env] = wp.max(
        0, total_candidates - MAX_BALL_MESH_CANDIDATES
    )

    for candidate_index in range(MAX_BALL_MESH_CANDIDATES):
        if candidate_index < retained_candidates:
            face = candidate_face[candidate_base + candidate_index]
            mesh = face_mesh_index[face]
            triangle_offset = face * 3
            a = vertices_bt[triangle_indices[triangle_offset]]
            b = vertices_bt[triangle_indices[triangle_offset + 1]]
            c = vertices_bt[triangle_indices[triangle_offset + 2]]
            point_b_bt = wp.vec3(0.0, 0.0, 0.0)
            normal = wp.vec3(0.0, 0.0, 0.0)
            distance_bt = wp.float32(0.0)
            valid = _sphere_triangle_contact(
                position_bt,
                basis,
                rsqrtss_mantissa,
                a,
                b,
                c,
                point_b_bt,
                normal,
                distance_bt,
            )
            if valid != 0:
                point_a_bt = _bullet_vector_scale_add(
                    point_b_bt, normal, distance_bt
                )
                edge_angles = internal_edge_angles[face]
                edge_flags = internal_edge_flags[face]
                edge_distance_bt = wp.float32(0.0)
                best_edge = bullet_internal_edge_best(
                    point_b_bt, a, b, c, edge_angles, edge_distance_bt
                )
                reprojected = wp.int32(0)
                if best_edge >= 0 and edge_distance_bt < 0.1:
                    triangle_normal = internal_edge_face_normals[face]
                    local_normal = _bullet_sse_normalize(normal)
                    edge_angle = edge_angles[best_edge]
                    if edge_angle == 0.0:
                        if _bullet_internal_edge_dot(triangle_normal, local_normal) >= 0.0:
                            normal = triangle_normal
                            reprojected = 1
                    else:
                        edge_start = a
                        edge_end = b
                        if best_edge == 1:
                            edge_start = b
                            edge_end = c
                        elif best_edge == 2:
                            edge_start = c
                            edge_end = a
                        edge_vector = edge_start - edge_end
                        is_convex = edge_flags & (1 << best_edge) != 0
                        swap_factor = -1.0
                        if is_convex:
                            swap_factor = 1.0
                        normal_a = triangle_normal * swap_factor
                        edge_index = face * 3 + best_edge
                        normal_b = internal_edge_normal_bs[edge_index]
                        back_facing = (
                            _bullet_internal_edge_dot(local_normal, normal_a) < 0.0
                            and _bullet_internal_edge_dot(local_normal, normal_b) < 0.0
                        )
                        if back_facing:
                            if _bullet_internal_edge_dot(triangle_normal, local_normal) >= 0.0:
                                normal = triangle_normal
                                reprojected = 1
                        else:
                            edge_cross = internal_edge_crosses[edge_index]
                            clamp_normal = local_normal
                            if best_edge > 0:
                                clamp_normal = normal
                            current_angle = _bullet_internal_edge_angle(
                                clamp_normal, edge_cross, normal_a
                            )
                            clamp = wp.int32(0)
                            if edge_angle < 0.0 and current_angle < edge_angle:
                                clamp = 1
                            elif edge_angle >= 0.0 and current_angle > edge_angle:
                                clamp = 1
                            if clamp != 0:
                                clamped = _bullet_matrix_axis_angle_rotate(
                                    clamp_normal,
                                    edge_vector,
                                    edge_angle - current_angle,
                                )
                                if _bullet_internal_edge_dot(clamped, triangle_normal) > 0.0:
                                    normal = clamped
                                    reprojected = 1
                if reprojected != 0:
                    point_b_bt = _bullet_vector_scale_add(
                        point_a_bt, normal, -distance_bt
                    )
                local_a = _bullet_inverse_transform_point(
                    position_bt, basis, point_a_bt
                )

                manifold_start = wp.int32(-1)
                manifold_count = wp.int32(0)
                for existing in range(MAX_BALL_CONTACTS):
                    if existing < contacts and contact_mesh[contact_base + existing] == mesh:
                        if manifold_start < 0:
                            manifold_start = existing
                        manifold_count = manifold_count + 1
                destination = wp.int32(-1)
                if manifold_count < 4:
                    insert = contacts
                    if manifold_start >= 0:
                        insert = manifold_start + manifold_count
                    if contacts < MAX_BALL_CONTACTS:
                        for shift in range(MAX_BALL_CONTACTS - 1):
                            source_relative = contacts - shift - 1
                            if source_relative >= insert:
                                _copy_contact(
                                    contact_base + source_relative,
                                    contact_base + source_relative + 1,
                                    contact_local_a_bt,
                                    contact_point_b_bt,
                                    contact_normal,
                                    contact_distance_bt,
                                    contact_face,
                                    contact_mesh,
                                    contact_lifetime,
                                    contact_normal_impulse,
                                    contact_tangent_impulse,
                                    contact_tangent,
                                )
                        destination = contact_base + insert
                        contacts = contacts + 1
                    else:
                        contact_overflow[env] = contact_overflow[env] + 1
                else:
                    replacement = _manifold_replacement(
                        local_a,
                        distance_bt,
                        contact_base + manifold_start,
                        contact_local_a_bt,
                        contact_distance_bt,
                    )
                    destination = contact_base + manifold_start + replacement
                if destination >= 0:
                    contact_local_a_bt[destination] = local_a
                    contact_point_b_bt[destination] = point_b_bt
                    contact_normal[destination] = normal
                    contact_distance_bt[destination] = distance_bt
                    contact_face[destination] = face
                    contact_mesh[destination] = mesh
                    if manifold_count < 4:
                        contact_lifetime[destination] = 0
                        contact_normal_impulse[destination] = 0.0
                        contact_tangent_impulse[destination] = 0.0
                    contact_tangent[destination] = wp.vec3(0.0, 0.0, 0.0)

    # Analytic plane bodies follow all sixteen mesh bodies in Arena.cpp.
    for plane in range(4):
        normal = wp.vec3(0.0, 0.0, 1.0)
        plane_point_bt = wp.vec3(0.0, 0.0, 0.0)
        if plane == 1:
            normal = wp.vec3(0.0, 0.0, -1.0)
            plane_point_bt = wp.vec3(0.0, 0.0, 40.96)
        elif plane == 2:
            normal = wp.vec3(1.0, 0.0, 0.0)
            plane_point_bt = wp.vec3(-81.92, 0.0, 20.48)
        elif plane == 3:
            normal = wp.vec3(-1.0, 0.0, 0.0)
            plane_point_bt = wp.vec3(81.92, 0.0, 20.48)
        point_a_bt = wp.vec3(0.0, 0.0, 0.0)
        point_b_bt = wp.vec3(0.0, 0.0, 0.0)
        distance_bt = wp.float32(0.0)
        _bullet_sphere_plane_contact(
            position_bt,
            basis,
            rsqrtss_mantissa,
            plane_point_bt,
            normal,
            point_a_bt,
            point_b_bt,
            distance_bt,
        )
        if distance_bt < CONTACT_BREAKING_BT:
            local_a = _bullet_inverse_transform_point(
                position_bt, basis, point_a_bt
            )
            mesh = 16 + plane
            manifold_start = wp.int32(-1)
            manifold_count = wp.int32(0)
            for existing in range(MAX_BALL_CONTACTS):
                if existing < contacts and contact_mesh[contact_base + existing] == mesh:
                    if manifold_start < 0:
                        manifold_start = existing
                    manifold_count = manifold_count + 1
            destination = wp.int32(-1)
            if manifold_count < 4:
                insert = contacts
                if manifold_start >= 0:
                    insert = manifold_start + manifold_count
                if contacts < MAX_BALL_CONTACTS:
                    for shift in range(MAX_BALL_CONTACTS - 1):
                        source_relative = contacts - shift - 1
                        if source_relative >= insert:
                            _copy_contact(
                                contact_base + source_relative,
                                contact_base + source_relative + 1,
                                contact_local_a_bt,
                                contact_point_b_bt,
                                contact_normal,
                                contact_distance_bt,
                                contact_face,
                                contact_mesh,
                                contact_lifetime,
                                contact_normal_impulse,
                                contact_tangent_impulse,
                                contact_tangent,
                            )
                    destination = contact_base + insert
                    contacts = contacts + 1
                else:
                    contact_overflow[env] = contact_overflow[env] + 1
            else:
                replacement = _manifold_replacement(
                    local_a,
                    distance_bt,
                    contact_base + manifold_start,
                    contact_local_a_bt,
                    contact_distance_bt,
                )
                destination = contact_base + manifold_start + replacement
            if destination >= 0:
                contact_local_a_bt[destination] = local_a
                contact_point_b_bt[destination] = point_b_bt
                contact_normal[destination] = normal
                contact_distance_bt[destination] = distance_bt
                contact_face[destination] = -10 - plane
                contact_mesh[destination] = mesh
                if manifold_count < 4:
                    contact_lifetime[destination] = 0
                    contact_normal_impulse[destination] = 0.0
                    contact_tangent_impulse[destination] = 0.0
                contact_tangent[destination] = wp.vec3(0.0, 0.0, 0.0)

    # btConvexConcaveCollisionAlgorithm refreshes after the complete witness
    # stream.  Existing points participate in reduction before this removal;
    # every retained point, including a new point, gains one lifetime tick.
    for reverse in range(MAX_BALL_CONTACTS):
        relative = contacts - 1 - reverse
        if relative >= 0 and relative < contacts:
            index = contact_base + relative
            point_a_bt = _bullet_transform_point(
                position_bt, basis, contact_local_a_bt[index]
            )
            normal = contact_normal[index]
            point_b_bt = contact_point_b_bt[index]
            distance_bt = _bullet_internal_edge_dot(point_a_bt - point_b_bt, normal)
            projected = _bullet_vector_scale_add(point_a_bt, normal, -distance_bt)
            lateral = point_b_bt - projected
            lateral_sq = _bullet_internal_edge_dot(lateral, lateral)
            invalid = distance_bt > CONTACT_BREAKING_BT or (
                lateral_sq > CONTACT_BREAKING_BT * CONTACT_BREAKING_BT
            )
            if invalid:
                for shift in range(MAX_BALL_CONTACTS - 1):
                    source_relative = relative + 1 + shift
                    if source_relative < contacts:
                        _copy_contact(
                            contact_base + source_relative,
                            contact_base + source_relative - 1,
                            contact_local_a_bt,
                            contact_point_b_bt,
                            contact_normal,
                            contact_distance_bt,
                            contact_face,
                            contact_mesh,
                            contact_lifetime,
                            contact_normal_impulse,
                            contact_tangent_impulse,
                            contact_tangent,
                        )
                contacts = contacts - 1
            else:
                contact_distance_bt[index] = distance_bt
                contact_lifetime[index] = contact_lifetime[index] + 1

    delta_velocity = wp.vec3(0.0, 0.0, 0.0)
    delta_angular = wp.vec3(0.0, 0.0, 0.0)
    push_velocity = wp.vec3(0.0, 0.0, 0.0)
    turn_velocity = wp.vec3(0.0, 0.0, 0.0)
    special_normal_sum = wp.vec3(0.0, 0.0, 0.0)
    special_distance_sum = wp.float32(0.0)

    # Bullet converts manifolds after its equal-island quicksort.  The ordinary
    # rows remain present for split impulse and special-row accumulation, but
    # RocketSim's interleaved velocity solver resolves only the special row.
    for solver_relative in range(MAX_BALL_CONTACTS):
        if solver_relative < contacts:
            relative = _ball_solver_contact_index(
                env, solver_relative, contacts, contact_mesh
            )
            index = contact_base + relative
            normal = contact_normal[index]
            point_a_bt = _bullet_transform_point(
                position_bt, basis, contact_local_a_bt[index]
            )
            rel = point_a_bt - position_bt
            tangent = wp.vec3(0.0, 0.0, 0.0)
            normal_jacobian = wp.float32(0.0)
            tangent_jacobian = wp.float32(0.0)
            normal_rhs = wp.float32(0.0)
            tangent_rhs = wp.float32(0.0)
            push_rhs = wp.float32(0.0)
            distance_bt = contact_distance_bt[index]
            _bullet_ball_contact_row(
                position_bt,
                basis,
                point_a_bt,
                contact_point_b_bt[index],
                distance_bt,
                DT,
                normal,
                pre_velocity_bt,
                pre_angular,
                force_velocity_bt,
                pre_angular,
                tangent,
                normal_jacobian,
                tangent_jacobian,
                normal_rhs,
                tangent_rhs,
                push_rhs,
            )
            contact_tangent[index] = tangent
            contact_normal_jacobian[index] = normal_jacobian
            contact_tangent_jacobian[index] = tangent_jacobian
            contact_normal_rhs[index] = normal_rhs
            contact_tangent_rhs[index] = tangent_rhs
            contact_push_rhs[index] = push_rhs
            contact_push_impulse[index] = 0.0

            contact_normal_impulse[index] = 0.0
            contact_tangent_impulse[index] = 0.0
            special_normal_sum = special_normal_sum + normal
            special_distance_sum = special_distance_sum + wp.sqrt(
                _bullet_internal_edge_dot(rel, rel)
            )

    # RocketSim appends one aggregated special row for every ball/world island.
    special_normal = wp.vec3(0.0, 0.0, 0.0)
    special_rel = wp.vec3(0.0, 0.0, 0.0)
    special_tangent = wp.vec3(0.0, 0.0, 0.0)
    special_normal_jacobian = wp.float32(0.0)
    special_tangent_jacobian = wp.float32(0.0)
    special_normal_rhs = wp.float32(0.0)
    special_tangent_rhs = wp.float32(0.0)
    special_normal_impulse = wp.float32(0.0)
    special_tangent_impulse = wp.float32(0.0)
    if contacts > 0:
        inverse_contacts = 1.0 / wp.float32(contacts)
        special_normal = _bullet_vector_scale_add(
            wp.vec3(0.0, 0.0, 0.0), special_normal_sum, inverse_contacts
        )
        special_distance = special_distance_sum * inverse_contacts
        special_rel = _bullet_vector_scale_add(
            wp.vec3(0.0, 0.0, 0.0), special_normal, -special_distance
        )
        special_push_rhs = wp.float32(0.0)
        # convertContactSpecial passes rel_pos1 = normal * -distance directly
        # into setupContactConstraint.  Reconstructing a world point and then
        # subtracting the body origin discards low bits at arena coordinates.
        _bullet_ball_special_contact_row(
            wp.vec3(0.0, 0.0, 0.0),
            basis,
            special_rel,
            wp.vec3(0.0, 0.0, 0.0),
            special_distance,
            DT,
            special_normal,
            pre_velocity_bt,
            pre_angular,
            force_velocity_bt,
            pre_angular,
            special_tangent,
            special_normal_jacobian,
            special_tangent_jacobian,
            special_normal_rhs,
            special_tangent_rhs,
            special_push_rhs,
        )

    # Bullet runs split impulse first and includes every ordinary special-marked
    # row.  Its appended aggregate row has positive synthetic distance and is
    # therefore a split no-op.
    for _iteration in range(CONTACT_SOLVER_ITERATIONS):
        for solver_relative in range(MAX_BALL_CONTACTS):
            if solver_relative < contacts:
                relative = _ball_solver_contact_index(
                    env, solver_relative, contacts, contact_mesh
                )
                index = contact_base + relative
                rhs = contact_push_rhs[index]
                if rhs != 0.0:
                    normal = contact_normal[index]
                    point_a_bt = _bullet_transform_point(
                        position_bt, basis, contact_local_a_bt[index]
                    )
                    rel = point_a_bt - position_bt
                    applied_push = contact_push_impulse[index]
                    _bullet_ball_solve_split_row(
                        basis,
                        normal,
                        rel,
                        contact_normal_jacobian[index],
                        rhs,
                        push_velocity,
                        turn_velocity,
                        applied_push,
                    )
                    contact_push_impulse[index] = applied_push

    # RocketSim's non-interleaved path skips every ordinary m_isSpecial normal
    # row.  Their friction rows retain a zero normal impulse, leaving only the
    # appended non-special aggregate normal and its friction row to resolve.
    for _iteration in range(CONTACT_SOLVER_ITERATIONS):
        if contacts > 0:
            _bullet_ball_solve_velocity_row(
                basis,
                special_normal,
                special_rel,
                special_normal_jacobian,
                special_normal_rhs,
                0.0,
                10000000000.0,
                delta_velocity,
                delta_angular,
                special_normal_impulse,
            )

            limit = CONTACT_FRICTION * special_normal_impulse
            _bullet_ball_solve_velocity_row(
                basis,
                special_tangent,
                special_rel,
                special_tangent_jacobian,
                special_tangent_rhs,
                -limit,
                limit,
                delta_velocity,
                delta_angular,
                special_tangent_impulse,
            )

    # btSolverBody::writebackVelocity first adds the solver delta to its base
    # velocity. writeBackBodies then adds m_externalForceImpulse when setting
    # the rigid body's velocity. The alternate grouping
    # (base + external) + delta differs by one ULP on reachable contacts.
    solved_velocity_bt = (pre_velocity_bt + delta_velocity) + external_force_impulse_bt
    solved_angular = pre_angular + delta_angular
    split_quaternion = quaternion
    has_split = wp.int32(0)
    if wp.dot(push_velocity, push_velocity) > 0.0 or wp.dot(
        turn_velocity, turn_velocity
    ) > 0.0:
        has_split = 1
        _bullet_integrate_quaternion(
            basis,
            turn_velocity * CONTACT_SPLIT_TURN_ERP,
            DT,
            split_quaternion,
        )
    integrated_position_bt = position_bt
    _bullet_integrate_position(
        position_bt,
        push_velocity,
        solved_velocity_bt,
        DT,
        has_split,
        integrated_position_bt,
    )
    integrated_quaternion = split_quaternion
    _bullet_integrate_quaternion(
        _bullet_quaternion_matrix(split_quaternion),
        solved_angular,
        DT,
        integrated_quaternion,
    )

    solved_velocity_bt = _cap(solved_velocity_bt, BALL_MAX_SPEED_BT)
    solved_angular = _cap(solved_angular, BALL_MAX_ANGULAR_SPEED)
    ball_pos[env] = integrated_position_bt * 50.0
    ball_vel[env] = solved_velocity_bt * 50.0
    resident_position_bt[env] = integrated_position_bt
    resident_velocity_bt[env] = solved_velocity_bt
    ball_quat[env] = integrated_quaternion
    ball_ang_vel[env] = solved_angular
    contact_count[env] = contacts
