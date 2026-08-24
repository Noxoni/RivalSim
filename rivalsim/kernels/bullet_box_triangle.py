"""Pinned Bullet 3.24 box/triangle narrowphase primitives.

This module is an intentionally bounded source translation.  It specializes
Bullet's ``btGjkEpaSolver2`` support mapping to the only pair used by the
v0.2.x static-world chassis path: one marginless oriented box core against one
static triangle.  The GJK/EPA iteration limits, projection formulas, hull
ordering, strict comparisons, and witness reconstruction follow the pinned
RocketSim Bullet source under ``.reference/RocketSimPython/libsrc``.

It is not a generic Bullet shape dispatcher and must not grow into one.
"""

import warp as wp

# Adapted from Bullet 3.24's btGjkEpa2.cpp and btPersistentManifold.cpp.
# Bullet is distributed under the zlib license; see THIRD_PARTY_NOTICES.md.
_BULLET_BOX_TRIANGLE_PENETRATION = r"""
    struct BtV3 {
        float x;
        float y;
        float z;
    };

    // Bullet's pinned Windows build uses scalar float32 results from SSE
    // multiply/add/subtract instructions.  Explicit CUDA round-to-nearest
    // operations prevent contraction into FMAs and preserve that operation
    // order on the GPU path.
    auto bt_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto bt_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
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
    auto bt_make = [](float x, float y, float z) -> BtV3 {
        BtV3 value = {x, y, z};
        return value;
    };
    auto bt_vadd = [&](BtV3 a, BtV3 b) -> BtV3 {
        return bt_make(bt_add(a.x, b.x), bt_add(a.y, b.y), bt_add(a.z, b.z));
    };
    auto bt_vsub = [&](BtV3 a, BtV3 b) -> BtV3 {
        return bt_make(bt_sub(a.x, b.x), bt_sub(a.y, b.y), bt_sub(a.z, b.z));
    };
    auto bt_vneg = [&](BtV3 value) -> BtV3 {
        return bt_make(-value.x, -value.y, -value.z);
    };
    auto bt_vmul = [&](BtV3 value, float scale) -> BtV3 {
        return bt_make(
            bt_mul(value.x, scale),
            bt_mul(value.y, scale),
            bt_mul(value.z, scale));
    };
    auto bt_dot = [&](BtV3 a, BtV3 b) -> float {
        const float x = bt_mul(a.x, b.x);
        const float y = bt_mul(a.y, b.y);
        const float z = bt_mul(a.z, b.z);
        return bt_add(bt_add(x, y), z);
    };
    auto bt_length2 = [&](BtV3 value) -> float {
        return bt_dot(value, value);
    };
    auto bt_length = [&](BtV3 value) -> float {
        return sqrtf(bt_length2(value));
    };
    auto bt_cross = [&](BtV3 a, BtV3 b) -> BtV3 {
        return bt_make(
            bt_sub(bt_mul(a.y, b.z), bt_mul(a.z, b.y)),
            bt_sub(bt_mul(a.z, b.x), bt_mul(a.x, b.z)),
            bt_sub(bt_mul(a.x, b.y), bt_mul(a.y, b.x)));
    };
    auto bt_normalized = [&](BtV3 value) -> BtV3 {
        const float inverse = bt_div(1.0f, bt_length(value));
        return bt_vmul(value, inverse);
    };
    auto bt_sse_normalized = [&](BtV3 value) -> BtV3 {
        const float length_squared = bt_length2(value);
        float inverse;
    #if defined(__CUDA_ARCH__)
        // localGetSupportVertexNonVirtual receives the direction already
        // normalized by GJK::getsupport.  In that reachable one-ULP
        // neighborhood, pinned Windows Bullet's RSQRTSS estimate has these
        // two exact buckets.  CUDA rsqrtf is much more accurate and therefore
        // does not preserve Bullet's result after its single Newton step.
        const unsigned length_bits = __float_as_uint(length_squared);
        if (length_bits >= 0x3f7ff000u && length_bits < 0x3f800000u) {
            inverse = __uint_as_float(0x3f800000u);
        } else if (length_bits >= 0x3f800000u && length_bits < 0x3f800800u) {
            inverse = __uint_as_float(0x3f7ff800u);
        } else {
            inverse = rsqrtf(length_squared);
        }
    #elif defined(__clang__) && (defined(__x86_64__) || defined(_M_X64))
        typedef float BtFloat4 __attribute__((__vector_size__(16)));
        const BtFloat4 estimate_input = {length_squared, 0.0f, 0.0f, 0.0f};
        const BtFloat4 estimate = __builtin_ia32_rsqrtss(estimate_input);
        inverse = estimate[0];
    #elif defined(_MSC_VER)
        const __m128 estimate_input = _mm_set_ss(length_squared);
        inverse = _mm_cvtss_f32(_mm_rsqrt_ss(estimate_input));
    #else
        inverse = bt_div(1.0f, sqrtf(length_squared));
    #endif
        float half_length = bt_mul(length_squared, 0.5f);
        half_length = bt_mul(half_length, inverse);
        half_length = bt_mul(half_length, inverse);
        const float correction = bt_sub(1.5f, half_length);
        inverse = bt_mul(inverse, correction);
        return bt_vmul(value, inverse);
    };
    auto bt_det = [&](BtV3 a, BtV3 b, BtV3 c) -> float {
        float value = bt_mul(bt_mul(a.y, b.z), c.x);
        value = bt_add(value, bt_mul(bt_mul(a.z, b.x), c.y));
        value = bt_sub(value, bt_mul(bt_mul(a.x, b.z), c.y));
        value = bt_sub(value, bt_mul(bt_mul(a.y, b.x), c.z));
        value = bt_add(value, bt_mul(bt_mul(a.x, b.y), c.z));
        value = bt_sub(value, bt_mul(bt_mul(a.z, b.y), c.x));
        return value;
    };

    const BtV3 body = bt_make(body_origin_bt[0], body_origin_bt[1], body_origin_bt[2]);
    const BtV3 triangle_world[3] = {
        bt_make(v0_bt[0], v0_bt[1], v0_bt[2]),
        bt_make(v1_bt[0], v1_bt[1], v1_bt[2]),
        bt_make(v2_bt[0], v2_bt[1], v2_bt[2]),
    };
    auto bt_basis_mul = [&](BtV3 value) -> BtV3 {
        const float x = bt_add(
            bt_add(bt_mul(basis.data[0][0], value.x), bt_mul(basis.data[0][1], value.y)),
            bt_mul(basis.data[0][2], value.z));
        const float y = bt_add(
            bt_add(bt_mul(basis.data[1][0], value.x), bt_mul(basis.data[1][1], value.y)),
            bt_mul(basis.data[1][2], value.z));
        const float z = bt_add(
            bt_add(bt_mul(basis.data[2][0], value.x), bt_mul(basis.data[2][1], value.y)),
            bt_mul(basis.data[2][2], value.z));
        return bt_make(x, y, z);
    };
    auto bt_basis_transpose_mul = [&](BtV3 value) -> BtV3 {
        const float x = bt_add(
            bt_add(bt_mul(basis.data[0][0], value.x), bt_mul(basis.data[1][0], value.y)),
            bt_mul(basis.data[2][0], value.z));
        const float y = bt_add(
            bt_add(bt_mul(basis.data[0][1], value.x), bt_mul(basis.data[1][1], value.y)),
            bt_mul(basis.data[2][1], value.z));
        const float z = bt_add(
            bt_add(bt_mul(basis.data[0][2], value.x), bt_mul(basis.data[1][2], value.y)),
            bt_mul(basis.data[2][2], value.z));
        return bt_make(x, y, z);
    };

    const BtV3 child_offset = bt_make(0.277513981f, 0.0f, 0.415099978f);
    const BtV3 center = bt_vadd(body, bt_basis_mul(child_offset));
    // getClosestPointsNonVirtual recenters both transforms before invoking the
    // penetration solver.  btGjkEpaSolver2 then caches
    // transformA.inverseTimes(transformB), so triangle support is evaluated as
    // transpose(basis) * vertex + relative_origin.  Do not collapse that into
    // transpose(basis) * (vertex - center): the two source expressions have
    // different float32 rounding on large arena coordinates.
    const BtV3 position_offset = bt_vmul(center, 0.5f);
    const BtV3 local_origin_a = bt_vsub(center, position_offset);
    const BtV3 local_origin_b = bt_vsub(bt_make(0.0f, 0.0f, 0.0f), position_offset);
    const BtV3 triangle_to_box_origin = bt_basis_transpose_mul(
        bt_vsub(local_origin_b, local_origin_a));
    const BtV3 box_half = bt_make(1.16507006f, 0.826994002f, 0.346590996f);
    const float box_margin = 0.0386590995f;

    auto bt_box_support_core = [&](BtV3 direction) -> BtV3 {
        return bt_make(
            direction.x >= 0.0f ? box_half.x : -box_half.x,
            direction.y >= 0.0f ? box_half.y : -box_half.y,
            direction.z >= 0.0f ? box_half.z : -box_half.z);
    };
    auto bt_box_support = [&](BtV3 direction) -> BtV3 {
        // localGetSupportVertexNonVirtual normalizes again, selects the
        // marginless core vertex, then adds the spherical convex margin.
        const BtV3 normalized_direction = bt_sse_normalized(direction);
        const BtV3 core = bt_box_support_core(normalized_direction);
        return bt_vadd(core, bt_vmul(normalized_direction, box_margin));
    };
    auto bt_triangle_support_world_core = [&](BtV3 direction) -> BtV3 {
        const float dot0 = bt_dot(direction, triangle_world[0]);
        const float dot1 = bt_dot(direction, triangle_world[1]);
        const float dot2 = bt_dot(direction, triangle_world[2]);
        const int axis = dot0 < dot1 ? (dot1 < dot2 ? 2 : 1) : (dot0 < dot2 ? 2 : 0);
        return triangle_world[axis];
    };
    auto bt_triangle_support_world = [&](BtV3 direction) -> BtV3 {
        // The triangle's margin is zero, but the non-virtual support wrapper
        // still normalizes before its dot3/maxAxis selection.
        const BtV3 normalized_direction = bt_sse_normalized(direction);
        return bt_triangle_support_world_core(normalized_direction);
    };
    auto bt_support0 = [&](BtV3 direction, bool with_margin) -> BtV3 {
        return with_margin ? bt_box_support(direction) : bt_box_support_core(direction);
    };
    auto bt_support1 = [&](BtV3 direction, bool with_margin) -> BtV3 {
        const BtV3 direction_world = bt_basis_mul(direction);
        const BtV3 point_b_world = with_margin
            ? bt_triangle_support_world(direction_world)
            : bt_triangle_support_world_core(direction_world);
        return bt_vadd(
            bt_basis_transpose_mul(point_b_world), triangle_to_box_origin);
    };
    auto bt_shape_support = [&](BtV3 direction,
                                bool with_margin,
                                BtV3& unit_direction) -> BtV3 {
        unit_direction = bt_normalized(direction);
        const BtV3 point_a_local = bt_support0(unit_direction, with_margin);
        const BtV3 point_b_local = bt_support1(bt_vneg(unit_direction), with_margin);
        return bt_vsub(point_a_local, point_b_local);
    };

    auto bt_project2 = [&](BtV3 a, BtV3 b, float* weights, int& mask) -> float {
        const BtV3 d = bt_vsub(b, a);
        const float length_squared = bt_length2(d);
        if (length_squared > 0.0f) {
            const float t = bt_div(-bt_dot(a, d), length_squared);
            if (t >= 1.0f) {
                weights[0] = 0.0f;
                weights[1] = 1.0f;
                mask = 2;
                return bt_length2(b);
            }
            if (t <= 0.0f) {
                weights[0] = 1.0f;
                weights[1] = 0.0f;
                mask = 1;
                return bt_length2(a);
            }
            weights[1] = t;
            weights[0] = bt_sub(1.0f, t);
            mask = 3;
            return bt_length2(bt_vadd(a, bt_vmul(d, t)));
        }
        return -1.0f;
    };

    auto bt_project3 = [&](BtV3 a, BtV3 b, BtV3 c, float* weights, int& mask) -> float {
        const BtV3 vertices[3] = {a, b, c};
        const BtV3 edges[3] = {bt_vsub(a, b), bt_vsub(b, c), bt_vsub(c, a)};
        const BtV3 normal = bt_cross(edges[0], edges[1]);
        const float normal_length_squared = bt_length2(normal);
        if (normal_length_squared > 0.0f) {
            float minimum_distance = -1.0f;
            for (int index = 0; index < 3; ++index) {
                if (bt_dot(vertices[index], bt_cross(edges[index], normal)) > 0.0f) {
                    const int adjacent = index == 0 ? 1 : (index == 1 ? 2 : 0);
                    float sub_weights[2] = {0.0f, 0.0f};
                    int sub_mask = 0;
                    const float distance = bt_project2(
                        vertices[index], vertices[adjacent], sub_weights, sub_mask);
                    if (minimum_distance < 0.0f || distance < minimum_distance) {
                        minimum_distance = distance;
                        mask = ((sub_mask & 1) ? (1 << index) : 0)
                            + ((sub_mask & 2) ? (1 << adjacent) : 0);
                        weights[index] = sub_weights[0];
                        weights[adjacent] = sub_weights[1];
                        const int remaining = adjacent == 0 ? 1 : (adjacent == 1 ? 2 : 0);
                        weights[remaining] = 0.0f;
                    }
                }
            }
            if (minimum_distance < 0.0f) {
                const float plane_dot = bt_dot(a, normal);
                const float normal_length = sqrtf(normal_length_squared);
                const BtV3 projection = bt_vmul(
                    normal, bt_div(plane_dot, normal_length_squared));
                minimum_distance = bt_length2(projection);
                mask = 7;
                weights[0] = bt_div(
                    bt_length(bt_cross(edges[1], bt_vsub(b, projection))),
                    normal_length);
                weights[1] = bt_div(
                    bt_length(bt_cross(edges[2], bt_vsub(c, projection))),
                    normal_length);
                weights[2] = bt_sub(1.0f, bt_add(weights[0], weights[1]));
            }
            return minimum_distance;
        }
        return -1.0f;
    };

    auto bt_project4 = [&](BtV3 a, BtV3 b, BtV3 c, BtV3 d, float* weights, int& mask) -> float {
        const BtV3 vertices[4] = {a, b, c, d};
        const BtV3 delta[3] = {bt_vsub(a, d), bt_vsub(b, d), bt_vsub(c, d)};
        const float volume = bt_det(delta[0], delta[1], delta[2]);
        const float orientation = bt_mul(
            volume, bt_dot(a, bt_cross(bt_vsub(b, c), bt_vsub(a, b))));
        if (orientation <= 0.0f && fabsf(volume) > 0.0f) {
            float minimum_distance = -1.0f;
            for (int index = 0; index < 3; ++index) {
                const int adjacent = index == 0 ? 1 : (index == 1 ? 2 : 0);
                const float side = bt_mul(
                    volume, bt_dot(d, bt_cross(delta[index], delta[adjacent])));
                if (side > 0.0f) {
                    float sub_weights[3] = {0.0f, 0.0f, 0.0f};
                    int sub_mask = 0;
                    const float distance = bt_project3(
                        vertices[index], vertices[adjacent], d, sub_weights, sub_mask);
                    if (minimum_distance < 0.0f || distance < minimum_distance) {
                        minimum_distance = distance;
                        mask = ((sub_mask & 1) ? (1 << index) : 0)
                            + ((sub_mask & 2) ? (1 << adjacent) : 0)
                            + ((sub_mask & 4) ? 8 : 0);
                        weights[index] = sub_weights[0];
                        weights[adjacent] = sub_weights[1];
                        const int remaining = adjacent == 0 ? 1 : (adjacent == 1 ? 2 : 0);
                        weights[remaining] = 0.0f;
                        weights[3] = sub_weights[2];
                    }
                }
            }
            if (minimum_distance < 0.0f) {
                minimum_distance = 0.0f;
                mask = 15;
                weights[0] = bt_div(bt_det(c, b, d), volume);
                weights[1] = bt_div(bt_det(a, c, d), volume);
                weights[2] = bt_div(bt_det(b, a, d), volume);
                weights[3] = bt_sub(
                    1.0f, bt_add(bt_add(weights[0], weights[1]), weights[2]));
            }
            return minimum_distance;
        }
        return -1.0f;
    };

    // Status values mirror gjkepa2_impl::GJK::eStatus.
    auto bt_gjk_evaluate = [&](BtV3 guess,
                               bool with_margin,
                               BtV3* output_directions,
                               BtV3* output_points,
                               float* output_weights,
                               int& output_rank,
                               BtV3& output_ray) -> int {
        BtV3 simplex_d[4];
        BtV3 simplex_w[4];
        float simplex_p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int rank = 0;
        const float squared_ray_length = bt_length2(guess);
        const BtV3 initial_direction = squared_ray_length > 0.0f
            ? bt_vneg(guess)
            : bt_make(1.0f, 0.0f, 0.0f);
        simplex_w[0] = bt_shape_support(
            initial_direction, with_margin, simplex_d[0]);
        simplex_p[0] = 1.0f;
        rank = 1;
        BtV3 ray = simplex_w[0];
        BtV3 last_w[4] = {ray, ray, ray, ray};
        int last_w_cursor = 0;
        unsigned iterations = 0;
        float alpha = 0.0f;
        int status = 0;

        do {
            const float ray_length = bt_length(ray);
            if (ray_length < 0.0001f) {
                status = 1;
                break;
            }

            simplex_w[rank] = bt_shape_support(
                bt_vneg(ray), with_margin, simplex_d[rank]);
            const BtV3 newest = simplex_w[rank];
            ++rank;
            bool duplicate = false;
            for (int index = 0; index < 4; ++index) {
                if (bt_length2(bt_vsub(newest, last_w[index])) < 0.0001f) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                --rank;
                break;
            }
            last_w_cursor = (last_w_cursor + 1) & 3;
            last_w[last_w_cursor] = newest;

            const float omega = bt_div(bt_dot(ray, newest), ray_length);
            if (omega > alpha) {
                alpha = omega;
            }
            if (bt_sub(bt_sub(ray_length, alpha), bt_mul(0.0001f, ray_length)) <= 0.0f) {
                --rank;
                break;
            }

            float projected_weights[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int mask = 0;
            float squared_distance = -1.0f;
            if (rank == 2) {
                squared_distance = bt_project2(
                    simplex_w[0], simplex_w[1], projected_weights, mask);
            } else if (rank == 3) {
                squared_distance = bt_project3(
                    simplex_w[0], simplex_w[1], simplex_w[2], projected_weights, mask);
            } else {
                squared_distance = bt_project4(
                    simplex_w[0], simplex_w[1], simplex_w[2], simplex_w[3],
                    projected_weights, mask);
            }

            if (squared_distance >= 0.0f) {
                BtV3 next_d[4];
                BtV3 next_w[4];
                float next_p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                int next_rank = 0;
                ray = bt_make(0.0f, 0.0f, 0.0f);
                for (int index = 0; index < rank; ++index) {
                    if ((mask & (1 << index)) != 0) {
                        next_d[next_rank] = simplex_d[index];
                        next_w[next_rank] = simplex_w[index];
                        next_p[next_rank] = projected_weights[index];
                        ray = bt_vadd(
                            ray, bt_vmul(simplex_w[index], projected_weights[index]));
                        ++next_rank;
                    }
                }
                rank = next_rank;
                for (int index = 0; index < rank; ++index) {
                    simplex_d[index] = next_d[index];
                    simplex_w[index] = next_w[index];
                    simplex_p[index] = next_p[index];
                }
                if (mask == 15) {
                    status = 1;
                }
            } else {
                --rank;
                break;
            }
            ++iterations;
            if (iterations >= 128) {
                status = 2;
            }
        } while (status == 0);

        output_rank = rank;
        output_ray = ray;
        for (int index = 0; index < rank; ++index) {
            output_directions[index] = simplex_d[index];
            output_points[index] = simplex_w[index];
            output_weights[index] = simplex_p[index];
        }
        return status;
    };

    auto bt_enclose_origin = [&](BtV3* simplex_d, BtV3* simplex_w, int& rank) -> bool {
        auto append = [&](BtV3 direction) {
            simplex_w[rank] = bt_shape_support(
                direction, true, simplex_d[rank]);
            ++rank;
        };
        auto rank4_encloses = [&]() -> bool {
            return fabsf(bt_det(
                bt_vsub(simplex_w[0], simplex_w[3]),
                bt_vsub(simplex_w[1], simplex_w[3]),
                bt_vsub(simplex_w[2], simplex_w[3]))) > 0.0f;
        };
        auto try_rank3 = [&]() -> bool {
            const BtV3 normal = bt_cross(
                bt_vsub(simplex_w[1], simplex_w[0]),
                bt_vsub(simplex_w[2], simplex_w[0]));
            if (bt_length2(normal) > 0.0f) {
                append(normal);
                if (rank4_encloses()) {
                    return true;
                }
                --rank;
                append(bt_vneg(normal));
                if (rank4_encloses()) {
                    return true;
                }
                --rank;
            }
            return false;
        };
        auto try_rank2 = [&]() -> bool {
            const BtV3 delta = bt_vsub(simplex_w[1], simplex_w[0]);
            for (int axis_index = 0; axis_index < 3; ++axis_index) {
                BtV3 axis = bt_make(0.0f, 0.0f, 0.0f);
                if (axis_index == 0) axis.x = 1.0f;
                if (axis_index == 1) axis.y = 1.0f;
                if (axis_index == 2) axis.z = 1.0f;
                const BtV3 perpendicular = bt_cross(delta, axis);
                if (bt_length2(perpendicular) > 0.0f) {
                    append(perpendicular);
                    if (try_rank3()) {
                        return true;
                    }
                    --rank;
                    append(bt_vneg(perpendicular));
                    if (try_rank3()) {
                        return true;
                    }
                    --rank;
                }
            }
            return false;
        };

        if (rank == 1) {
            for (int axis_index = 0; axis_index < 3; ++axis_index) {
                BtV3 axis = bt_make(0.0f, 0.0f, 0.0f);
                if (axis_index == 0) axis.x = 1.0f;
                if (axis_index == 1) axis.y = 1.0f;
                if (axis_index == 2) axis.z = 1.0f;
                append(axis);
                if (try_rank2()) {
                    return true;
                }
                --rank;
                append(bt_vneg(axis));
                if (try_rank2()) {
                    return true;
                }
                --rank;
            }
            return false;
        }
        if (rank == 2) {
            return try_rank2();
        }
        if (rank == 3) {
            return try_rank3();
        }
        if (rank == 4) {
            return rank4_encloses();
        }
        return false;
    };

    auto bt_epa_evaluate = [&](BtV3* simplex_d,
                               BtV3* simplex_w,
                               float* simplex_p,
                               int& simplex_rank,
                               BtV3 guess,
                               BtV3& result_normal,
                               float& result_depth,
                               BtV3* result_directions,
                               float* result_weights,
                               int& result_rank) -> bool {
        const int EPA_MAX_VERTICES = 128;
        const int EPA_MAX_FACES = 256;
        BtV3 vertex_d[132];
        BtV3 vertex_w[132];
        for (int index = 0; index < simplex_rank; ++index) {
            vertex_d[index] = simplex_d[index];
            vertex_w[index] = simplex_w[index];
        }

        if (simplex_rank > 1 && bt_enclose_origin(vertex_d, vertex_w, simplex_rank)) {
            if (bt_det(
                    bt_vsub(vertex_w[0], vertex_w[3]),
                    bt_vsub(vertex_w[1], vertex_w[3]),
                    bt_vsub(vertex_w[2], vertex_w[3])) < 0.0f) {
                const BtV3 swap_d = vertex_d[0];
                const BtV3 swap_w = vertex_w[0];
                vertex_d[0] = vertex_d[1];
                vertex_w[0] = vertex_w[1];
                vertex_d[1] = swap_d;
                vertex_w[1] = swap_w;
                const float swap_p = simplex_p[0];
                simplex_p[0] = simplex_p[1];
                simplex_p[1] = swap_p;
            }

            BtV3 face_normal[EPA_MAX_FACES];
            float face_distance[EPA_MAX_FACES];
            int face_vertex[EPA_MAX_FACES][3];
            int face_neighbor[EPA_MAX_FACES][3];
            unsigned char face_neighbor_edge[EPA_MAX_FACES][3];
            unsigned char face_pass[EPA_MAX_FACES];
            int face_previous[EPA_MAX_FACES];
            int face_next[EPA_MAX_FACES];
            int stock_root = 0;
            int stock_count = EPA_MAX_FACES;
            int hull_root = -1;
            int hull_count = 0;
            for (int index = 0; index < EPA_MAX_FACES; ++index) {
                face_previous[index] = index == 0 ? -1 : index - 1;
                face_next[index] = index + 1 < EPA_MAX_FACES ? index + 1 : -1;
                face_pass[index] = 0;
                face_neighbor[index][0] = -1;
                face_neighbor[index][1] = -1;
                face_neighbor[index][2] = -1;
            }

            auto list_remove = [&](int& root, int& count, int face) {
                const int previous = face_previous[face];
                const int next = face_next[face];
                if (next >= 0) face_previous[next] = previous;
                if (previous >= 0) face_next[previous] = next;
                if (root == face) root = next;
                --count;
            };
            auto list_append = [&](int& root, int& count, int face) {
                face_previous[face] = -1;
                face_next[face] = root;
                if (root >= 0) face_previous[root] = face;
                root = face;
                ++count;
            };
            auto bind_faces = [&](int face_a, int edge_a, int face_b, int edge_b) {
                face_neighbor_edge[face_a][edge_a] = static_cast<unsigned char>(edge_b);
                face_neighbor[face_a][edge_a] = face_b;
                face_neighbor_edge[face_b][edge_b] = static_cast<unsigned char>(edge_a);
                face_neighbor[face_b][edge_b] = face_a;
            };
            auto edge_distance = [&](int face,
                                     int vertex_a,
                                     int vertex_b,
                                     float& distance) -> bool {
                const BtV3 edge = bt_vsub(vertex_w[vertex_b], vertex_w[vertex_a]);
                const BtV3 outward = bt_cross(edge, face_normal[face]);
                if (bt_dot(vertex_w[vertex_a], outward) < 0.0f) {
                    const float edge_length_squared = bt_length2(edge);
                    const float a_dot_edge = bt_dot(vertex_w[vertex_a], edge);
                    const float b_dot_edge = bt_dot(vertex_w[vertex_b], edge);
                    if (a_dot_edge > 0.0f) {
                        distance = bt_length(vertex_w[vertex_a]);
                    } else if (b_dot_edge < 0.0f) {
                        distance = bt_length(vertex_w[vertex_b]);
                    } else {
                        const float a_dot_b = bt_dot(
                            vertex_w[vertex_a], vertex_w[vertex_b]);
                        const float numerator = bt_sub(
                            bt_mul(bt_length2(vertex_w[vertex_a]), bt_length2(vertex_w[vertex_b])),
                            bt_mul(a_dot_b, a_dot_b));
                        float quotient = bt_div(numerator, edge_length_squared);
                        if (quotient < 0.0f) quotient = 0.0f;
                        distance = sqrtf(quotient);
                    }
                    return true;
                }
                return false;
            };
            auto new_face = [&](int vertex_a, int vertex_b, int vertex_c, bool forced) -> int {
                if (stock_root < 0) return -1;
                const int face = stock_root;
                list_remove(stock_root, stock_count, face);
                list_append(hull_root, hull_count, face);
                face_pass[face] = 0;
                face_vertex[face][0] = vertex_a;
                face_vertex[face][1] = vertex_b;
                face_vertex[face][2] = vertex_c;
                face_normal[face] = bt_cross(
                    bt_vsub(vertex_w[vertex_b], vertex_w[vertex_a]),
                    bt_vsub(vertex_w[vertex_c], vertex_w[vertex_a]));
                const float normal_length = bt_length(face_normal[face]);
                if (normal_length > 0.0001f) {
                    if (!(edge_distance(face, vertex_a, vertex_b, face_distance[face])
                          || edge_distance(face, vertex_b, vertex_c, face_distance[face])
                          || edge_distance(face, vertex_c, vertex_a, face_distance[face]))) {
                        face_distance[face] = bt_div(
                            bt_dot(vertex_w[vertex_a], face_normal[face]), normal_length);
                    }
                    face_normal[face] = bt_vmul(
                        face_normal[face], bt_div(1.0f, normal_length));
                    if (forced || face_distance[face] >= -0.00001f) {
                        return face;
                    }
                }
                list_remove(hull_root, hull_count, face);
                list_append(stock_root, stock_count, face);
                return -1;
            };
            auto find_best = [&]() -> int {
                int best = hull_root;
                float minimum = bt_mul(face_distance[best], face_distance[best]);
                for (int face = face_next[best]; face >= 0; face = face_next[face]) {
                    const float squared = bt_mul(face_distance[face], face_distance[face]);
                    if (squared < minimum) {
                        best = face;
                        minimum = squared;
                    }
                }
                return best;
            };

            const int tetra[4] = {
                new_face(0, 1, 2, true),
                new_face(1, 0, 3, true),
                new_face(2, 1, 3, true),
                new_face(0, 2, 3, true),
            };
            if (hull_count == 4) {
                bind_faces(tetra[0], 0, tetra[1], 0);
                bind_faces(tetra[0], 1, tetra[2], 0);
                bind_faces(tetra[0], 2, tetra[3], 0);
                bind_faces(tetra[1], 1, tetra[3], 2);
                bind_faces(tetra[1], 2, tetra[2], 1);
                bind_faces(tetra[2], 2, tetra[3], 1);

                auto expand = [&](unsigned pass,
                                  int new_vertex,
                                  int initial_face,
                                  int initial_edge,
                                  int& horizon_current,
                                  int& horizon_first,
                                  unsigned& horizon_faces) -> bool {
                    int stack_face[EPA_MAX_FACES + 4];
                    int stack_edge[EPA_MAX_FACES + 4];
                    unsigned char stack_state[EPA_MAX_FACES + 4];
                    int top = 0;
                    stack_face[0] = initial_face;
                    stack_edge[0] = initial_edge;
                    stack_state[0] = 0;
                    bool child_result = false;
                    while (top >= 0) {
                        const int face = stack_face[top];
                        const int edge = stack_edge[top];
                        if (stack_state[top] == 0) {
                            if (face < 0 || face_pass[face] == pass) {
                                child_result = false;
                                --top;
                                continue;
                            }
                            const int edge1 = edge == 0 ? 1 : (edge == 1 ? 2 : 0);
                            if (bt_sub(
                                    bt_dot(face_normal[face], vertex_w[new_vertex]),
                                    face_distance[face]) < -0.00001f) {
                                const int horizon_face = new_face(
                                    face_vertex[face][edge1],
                                    face_vertex[face][edge],
                                    new_vertex,
                                    false);
                                if (horizon_face >= 0) {
                                    bind_faces(horizon_face, 0, face, edge);
                                    if (horizon_current >= 0) {
                                        bind_faces(horizon_current, 1, horizon_face, 2);
                                    } else {
                                        horizon_first = horizon_face;
                                    }
                                    horizon_current = horizon_face;
                                    ++horizon_faces;
                                    child_result = true;
                                } else {
                                    child_result = false;
                                }
                                --top;
                                continue;
                            }

                            face_pass[face] = static_cast<unsigned char>(pass);
                            stack_state[top] = 1;
                            if (top + 1 >= EPA_MAX_FACES + 4) {
                                child_result = false;
                                --top;
                                continue;
                            }
                            ++top;
                            stack_face[top] = face_neighbor[face][edge1];
                            stack_edge[top] = face_neighbor_edge[face][edge1];
                            stack_state[top] = 0;
                            continue;
                        }
                        if (stack_state[top] == 1) {
                            if (!child_result) {
                                child_result = false;
                                --top;
                                continue;
                            }
                            const int edge2 = edge == 0 ? 2 : (edge == 1 ? 0 : 1);
                            stack_state[top] = 2;
                            if (top + 1 >= EPA_MAX_FACES + 4) {
                                child_result = false;
                                --top;
                                continue;
                            }
                            ++top;
                            stack_face[top] = face_neighbor[face][edge2];
                            stack_edge[top] = face_neighbor_edge[face][edge2];
                            stack_state[top] = 0;
                            continue;
                        }
                        if (child_result) {
                            list_remove(hull_root, hull_count, face);
                            list_append(stock_root, stock_count, face);
                            child_result = true;
                        }
                        --top;
                    }
                    return child_result;
                };

                int best = find_best();
                BtV3 outer_normal = face_normal[best];
                float outer_distance = face_distance[best];
                int outer_vertex[3] = {
                    face_vertex[best][0], face_vertex[best][1], face_vertex[best][2]};
                unsigned pass = 0;
                int next_vertex = 0;
                for (unsigned iteration = 0; iteration < 255; ++iteration) {
                    if (next_vertex >= EPA_MAX_VERTICES) {
                        break;
                    }
                    const int vertex = 4 + next_vertex;
                    ++next_vertex;
                    vertex_w[vertex] = bt_shape_support(
                        face_normal[best], true, vertex_d[vertex]);
                    const float support_distance = bt_sub(
                        bt_dot(face_normal[best], vertex_w[vertex]), face_distance[best]);
                    if (support_distance > 0.0001f) {
                        int horizon_current = -1;
                        int horizon_first = -1;
                        unsigned horizon_faces = 0;
                        bool expansion_valid = true;
                        ++pass;
                        face_pass[best] = static_cast<unsigned char>(pass);
                        for (int edge = 0; edge < 3 && expansion_valid; ++edge) {
                            expansion_valid = expand(
                                pass,
                                vertex,
                                face_neighbor[best][edge],
                                face_neighbor_edge[best][edge],
                                horizon_current,
                                horizon_first,
                                horizon_faces);
                        }
                        if (expansion_valid && horizon_faces >= 3) {
                            bind_faces(horizon_current, 1, horizon_first, 2);
                            list_remove(hull_root, hull_count, best);
                            list_append(stock_root, stock_count, best);
                            best = find_best();
                            outer_normal = face_normal[best];
                            outer_distance = face_distance[best];
                            outer_vertex[0] = face_vertex[best][0];
                            outer_vertex[1] = face_vertex[best][1];
                            outer_vertex[2] = face_vertex[best][2];
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                const BtV3 projection = bt_vmul(outer_normal, outer_distance);
                float weights[3];
                weights[0] = bt_length(bt_cross(
                    bt_vsub(vertex_w[outer_vertex[1]], projection),
                    bt_vsub(vertex_w[outer_vertex[2]], projection)));
                weights[1] = bt_length(bt_cross(
                    bt_vsub(vertex_w[outer_vertex[2]], projection),
                    bt_vsub(vertex_w[outer_vertex[0]], projection)));
                weights[2] = bt_length(bt_cross(
                    bt_vsub(vertex_w[outer_vertex[0]], projection),
                    bt_vsub(vertex_w[outer_vertex[1]], projection)));
                const float sum = bt_add(bt_add(weights[0], weights[1]), weights[2]);
                result_normal = outer_normal;
                result_depth = outer_distance;
                result_rank = 3;
                for (int index = 0; index < 3; ++index) {
                    result_directions[index] = vertex_d[outer_vertex[index]];
                    result_weights[index] = bt_div(weights[index], sum);
                }
                return true;
            }
        }

        // btGjkEpa2::EPA::Evaluate fallback.  The caller still applies the
        // pair-detector zero-witness normal fallback exactly.
        result_normal = bt_vneg(guess);
        const float normal_length = bt_length(result_normal);
        if (normal_length > 0.0f) {
            result_normal = bt_vmul(result_normal, bt_div(1.0f, normal_length));
        } else {
            result_normal = bt_make(1.0f, 0.0f, 0.0f);
        }
        result_depth = 0.0f;
        result_rank = 1;
        result_directions[0] = simplex_d[0];
        result_weights[0] = 1.0f;
        return true;
    };

    point_a_bt = wp::vec_t<3, wp::float32>(0.0f, 0.0f, 0.0f);
    point_b_bt = wp::vec_t<3, wp::float32>(0.0f, 0.0f, 0.0f);
    normal_world = wp::vec_t<3, wp::float32>(0.0f, 0.0f, 0.0f);
    distance_bt = 0.0f;
    valid = 0;

    auto safe_normalized = [&](BtV3 value) -> BtV3 {
        const float length_squared = bt_length2(value);
        if (length_squared >= 1.1920929e-7f * 1.1920929e-7f) {
            return bt_vmul(value, bt_div(1.0f, sqrtf(length_squared)));
        }
        return bt_make(1.0f, 0.0f, 0.0f);
    };
    BtV3 guesses[9] = {
        safe_normalized(bt_vneg(center)),
        safe_normalized(center),
        bt_make(0.0f, 0.0f, 1.0f),
        bt_make(0.0f, 1.0f, 0.0f),
        bt_make(1.0f, 0.0f, 0.0f),
        bt_make(1.0f, 1.0f, 0.0f),
        bt_make(1.0f, 1.0f, 1.0f),
        bt_make(0.0f, 1.0f, 1.0f),
        bt_make(1.0f, 0.0f, 1.0f),
    };

    bool finished = false;
    for (int guess_index = 0; guess_index < 9 && !finished; ++guess_index) {
        BtV3 simplex_d[4];
        BtV3 simplex_w[4];
        float simplex_p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int simplex_rank = 0;
        BtV3 ray;
        const BtV3 penetration_guess = bt_vneg(guesses[guess_index]);
        const int status = bt_gjk_evaluate(
            penetration_guess,
            true,
            simplex_d,
            simplex_w,
            simplex_p,
            simplex_rank,
            ray);
        if (status == 1) {
            BtV3 epa_normal;
            float epa_depth = 0.0f;
            BtV3 epa_directions[3];
            float epa_weights[3] = {0.0f, 0.0f, 0.0f};
            int epa_rank = 0;
            if (bt_epa_evaluate(
                    simplex_d,
                    simplex_w,
                    simplex_p,
                    simplex_rank,
                    penetration_guess,
                    epa_normal,
                    epa_depth,
                    epa_directions,
                    epa_weights,
                    epa_rank)) {
                BtV3 witness_a_local = bt_make(0.0f, 0.0f, 0.0f);
                for (int index = 0; index < epa_rank; ++index) {
                    witness_a_local = bt_vadd(
                        witness_a_local,
                        bt_vmul(bt_box_support(epa_directions[index]), epa_weights[index]));
                }
                const BtV3 witness_b_local = bt_vsub(
                    witness_a_local, bt_vmul(epa_normal, epa_depth));
                // Penetration returns witnesses through the recentered
                // localTransA.  getClosestPointsNonVirtual only adds the
                // position offset back after deriving the pair normal and
                // depth, so retain both rounding boundaries here.
                const BtV3 witness_a_offset_world = bt_vadd(
                    bt_basis_mul(witness_a_local), local_origin_a);
                const BtV3 witness_b_offset_world = bt_vadd(
                    bt_basis_mul(witness_b_local), local_origin_a);
                BtV3 pair_normal = bt_vsub(
                    witness_b_offset_world, witness_a_offset_world);
                float pair_normal_length_squared = bt_length2(pair_normal);
                if (pair_normal_length_squared <= 1.1920929e-7f * 1.1920929e-7f) {
                    pair_normal = bt_vneg(epa_normal);
                    pair_normal_length_squared = bt_length2(pair_normal);
                }
                if (pair_normal_length_squared > 1.1920929e-7f * 1.1920929e-7f) {
                    pair_normal = bt_vmul(
                        pair_normal, bt_div(1.0f, sqrtf(pair_normal_length_squared)));
                    const float pair_distance = -bt_length(
                        bt_vsub(witness_a_offset_world, witness_b_offset_world));

                    const BtV3 triangle_aabb[3] = {
                        bt_vadd(triangle_world[0], local_origin_b),
                        bt_vadd(triangle_world[1], local_origin_b),
                        bt_vadd(triangle_world[2], local_origin_b),
                    };
                    const BtV3 triangle_min = bt_make(
                        triangle_aabb[0].x < triangle_aabb[1].x
                            ? (triangle_aabb[0].x < triangle_aabb[2].x
                                ? triangle_aabb[0].x : triangle_aabb[2].x)
                            : (triangle_aabb[1].x < triangle_aabb[2].x
                                ? triangle_aabb[1].x : triangle_aabb[2].x),
                        triangle_aabb[0].y < triangle_aabb[1].y
                            ? (triangle_aabb[0].y < triangle_aabb[2].y
                                ? triangle_aabb[0].y : triangle_aabb[2].y)
                            : (triangle_aabb[1].y < triangle_aabb[2].y
                                ? triangle_aabb[1].y : triangle_aabb[2].y),
                        triangle_aabb[0].z < triangle_aabb[1].z
                            ? (triangle_aabb[0].z < triangle_aabb[2].z
                                ? triangle_aabb[0].z : triangle_aabb[2].z)
                            : (triangle_aabb[1].z < triangle_aabb[2].z
                                ? triangle_aabb[1].z : triangle_aabb[2].z));
                    const BtV3 triangle_max = bt_make(
                        triangle_aabb[0].x > triangle_aabb[1].x
                            ? (triangle_aabb[0].x > triangle_aabb[2].x
                                ? triangle_aabb[0].x : triangle_aabb[2].x)
                            : (triangle_aabb[1].x > triangle_aabb[2].x
                                ? triangle_aabb[1].x : triangle_aabb[2].x),
                        triangle_aabb[0].y > triangle_aabb[1].y
                            ? (triangle_aabb[0].y > triangle_aabb[2].y
                                ? triangle_aabb[0].y : triangle_aabb[2].y)
                            : (triangle_aabb[1].y > triangle_aabb[2].y
                                ? triangle_aabb[1].y : triangle_aabb[2].y),
                        triangle_aabb[0].z > triangle_aabb[1].z
                            ? (triangle_aabb[0].z > triangle_aabb[2].z
                                ? triangle_aabb[0].z : triangle_aabb[2].z)
                            : (triangle_aabb[1].z > triangle_aabb[2].z
                                ? triangle_aabb[1].z : triangle_aabb[2].z));
                    const BtV3 triangle_center = bt_vmul(
                        bt_vadd(triangle_min, triangle_max), 0.5f);
                    const BtV3 center_difference = bt_vsub(
                        local_origin_a, triangle_center);
                    if (bt_dot(center_difference, pair_normal) < 0.0f) {
                        pair_normal = bt_vneg(pair_normal);
                    }
                    const BtV3 manifold_point_b = bt_vadd(
                        witness_b_offset_world, position_offset);
                    const BtV3 manifold_point_a = bt_vadd(
                        manifold_point_b, bt_vmul(pair_normal, pair_distance));
                    point_a_bt = wp::vec_t<3, wp::float32>(
                        manifold_point_a.x, manifold_point_a.y, manifold_point_a.z);
                    point_b_bt = wp::vec_t<3, wp::float32>(
                        manifold_point_b.x, manifold_point_b.y, manifold_point_b.z);
                    normal_world = wp::vec_t<3, wp::float32>(
                        pair_normal.x, pair_normal.y, pair_normal.z);
                    distance_bt = pair_distance;
                    valid = 1;
                }
                finished = true;
            }
        } else {
            BtV3 distance_d[4];
            BtV3 distance_w[4];
            float distance_p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int distance_rank = 0;
            BtV3 distance_ray;
            const int distance_status = bt_gjk_evaluate(
                guesses[guess_index],
                false,
                distance_d,
                distance_w,
                distance_p,
                distance_rank,
                distance_ray);
            if (distance_status == 0) {
                // btGjkEpaPenetrationDepthSolver returns the separated result
                // immediately instead of trying the remaining guess vectors.
                // calcPenDepth itself returns false, but it leaves the
                // marginless Distance witnesses and normal in the output
                // references. btGjkPairDetector method 6 consumes them.
                BtV3 witness_a_local = bt_make(0.0f, 0.0f, 0.0f);
                BtV3 witness_b_local = bt_make(0.0f, 0.0f, 0.0f);
                for (int index = 0; index < distance_rank; ++index) {
                    const float weight = distance_p[index];
                    witness_a_local = bt_vadd(
                        witness_a_local,
                        bt_vmul(bt_support0(distance_d[index], false), weight));
                    witness_b_local = bt_vadd(
                        witness_b_local,
                        bt_vmul(bt_support1(bt_vneg(distance_d[index]), false), weight));
                }

                const BtV3 witness_a_offset_world = bt_vadd(
                    bt_basis_mul(witness_a_local), local_origin_a);
                const BtV3 witness_b_offset_world = bt_vadd(
                    bt_basis_mul(witness_b_local), local_origin_a);
                BtV3 fallback_axis = bt_vsub(witness_a_local, witness_b_local);
                const float fallback_axis_length = bt_length(fallback_axis);
                if (fallback_axis_length > 0.0001f) {
                    fallback_axis = bt_vmul(
                        fallback_axis, bt_div(1.0f, fallback_axis_length));
                }

                if (bt_length2(fallback_axis) > 0.0f) {
                    const float pair_distance = bt_sub(
                        bt_length(bt_vsub(
                            witness_a_offset_world, witness_b_offset_world)),
                        box_margin);
                    BtV3 pair_normal = bt_sse_normalized(fallback_axis);

                    const BtV3 triangle_aabb[3] = {
                        bt_vadd(triangle_world[0], local_origin_b),
                        bt_vadd(triangle_world[1], local_origin_b),
                        bt_vadd(triangle_world[2], local_origin_b),
                    };
                    const BtV3 triangle_min = bt_make(
                        triangle_aabb[0].x < triangle_aabb[1].x
                            ? (triangle_aabb[0].x < triangle_aabb[2].x
                                ? triangle_aabb[0].x : triangle_aabb[2].x)
                            : (triangle_aabb[1].x < triangle_aabb[2].x
                                ? triangle_aabb[1].x : triangle_aabb[2].x),
                        triangle_aabb[0].y < triangle_aabb[1].y
                            ? (triangle_aabb[0].y < triangle_aabb[2].y
                                ? triangle_aabb[0].y : triangle_aabb[2].y)
                            : (triangle_aabb[1].y < triangle_aabb[2].y
                                ? triangle_aabb[1].y : triangle_aabb[2].y),
                        triangle_aabb[0].z < triangle_aabb[1].z
                            ? (triangle_aabb[0].z < triangle_aabb[2].z
                                ? triangle_aabb[0].z : triangle_aabb[2].z)
                            : (triangle_aabb[1].z < triangle_aabb[2].z
                                ? triangle_aabb[1].z : triangle_aabb[2].z));
                    const BtV3 triangle_max = bt_make(
                        triangle_aabb[0].x > triangle_aabb[1].x
                            ? (triangle_aabb[0].x > triangle_aabb[2].x
                                ? triangle_aabb[0].x : triangle_aabb[2].x)
                            : (triangle_aabb[1].x > triangle_aabb[2].x
                                ? triangle_aabb[1].x : triangle_aabb[2].x),
                        triangle_aabb[0].y > triangle_aabb[1].y
                            ? (triangle_aabb[0].y > triangle_aabb[2].y
                                ? triangle_aabb[0].y : triangle_aabb[2].y)
                            : (triangle_aabb[1].y > triangle_aabb[2].y
                                ? triangle_aabb[1].y : triangle_aabb[2].y),
                        triangle_aabb[0].z > triangle_aabb[1].z
                            ? (triangle_aabb[0].z > triangle_aabb[2].z
                                ? triangle_aabb[0].z : triangle_aabb[2].z)
                            : (triangle_aabb[1].z > triangle_aabb[2].z
                                ? triangle_aabb[1].z : triangle_aabb[2].z));
                    const BtV3 triangle_center = bt_vmul(
                        bt_vadd(triangle_min, triangle_max), 0.5f);
                    const BtV3 center_difference = bt_vsub(
                        local_origin_a, triangle_center);
                    if (bt_dot(center_difference, pair_normal) < 0.0f) {
                        pair_normal = bt_vneg(pair_normal);
                    }

                    const BtV3 manifold_point_b = bt_vadd(
                        witness_b_offset_world, position_offset);
                    const BtV3 manifold_point_a = bt_vadd(
                        manifold_point_b, bt_vmul(pair_normal, pair_distance));
                    point_a_bt = wp::vec_t<3, wp::float32>(
                        manifold_point_a.x, manifold_point_a.y, manifold_point_a.z);
                    point_b_bt = wp::vec_t<3, wp::float32>(
                        manifold_point_b.x, manifold_point_b.y, manifold_point_b.z);
                    normal_world = wp::vec_t<3, wp::float32>(
                        pair_normal.x, pair_normal.y, pair_normal.z);
                    distance_bt = pair_distance;
                    valid = 1;
                }
                finished = true;
            }
        }
    }
"""


@wp.func_native(_BULLET_BOX_TRIANGLE_PENETRATION)
def bullet_box_triangle_penetration(
    body_origin_bt: wp.vec3,
    basis: wp.mat33,
    v0_bt: wp.vec3,
    v1_bt: wp.vec3,
    v2_bt: wp.vec3,
    point_a_bt: wp.ref[wp.vec3],
    point_b_bt: wp.ref[wp.vec3],
    normal_world: wp.ref[wp.vec3],
    distance_bt: wp.ref[wp.float32],
    valid: wp.ref[wp.int32],
): ...


_BULLET_BOX_TRIANGLE_CLOSEST = r"""
    struct BtPairV3 {
        float x;
        float y;
        float z;
    };
    struct BtPairClosest {
        BtPairV3 closest;
        float weights[4];
        int used;
        int valid;
        int degenerate;
    };

    // Scalarized SSE float32 operation order from the pinned Windows Bullet
    // build. CUDA round-to-nearest intrinsics prevent FMA contraction.
    auto pair_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto pair_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
        return value;
    #endif
    };
    auto pair_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    auto pair_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    auto pair_make = [](float x, float y, float z) -> BtPairV3 {
        BtPairV3 value = {x, y, z};
        return value;
    };
    auto pair_vadd = [&](BtPairV3 a, BtPairV3 b) -> BtPairV3 {
        return pair_make(
            pair_add(a.x, b.x), pair_add(a.y, b.y), pair_add(a.z, b.z));
    };
    auto pair_vsub = [&](BtPairV3 a, BtPairV3 b) -> BtPairV3 {
        return pair_make(
            pair_sub(a.x, b.x), pair_sub(a.y, b.y), pair_sub(a.z, b.z));
    };
    auto pair_vneg = [&](BtPairV3 value) -> BtPairV3 {
        return pair_make(-value.x, -value.y, -value.z);
    };
    auto pair_vmul = [&](BtPairV3 value, float scale) -> BtPairV3 {
        return pair_make(
            pair_mul(value.x, scale),
            pair_mul(value.y, scale),
            pair_mul(value.z, scale));
    };
    auto pair_dot = [&](BtPairV3 a, BtPairV3 b) -> float {
        const float x = pair_mul(a.x, b.x);
        const float y = pair_mul(a.y, b.y);
        const float z = pair_mul(a.z, b.z);
        return pair_add(pair_add(x, y), z);
    };
    auto pair_length2 = [&](BtPairV3 value) -> float {
        return pair_dot(value, value);
    };
    auto pair_cross = [&](BtPairV3 a, BtPairV3 b) -> BtPairV3 {
        return pair_make(
            pair_sub(pair_mul(a.y, b.z), pair_mul(a.z, b.y)),
            pair_sub(pair_mul(a.z, b.x), pair_mul(a.x, b.z)),
            pair_sub(pair_mul(a.x, b.y), pair_mul(a.y, b.x)));
    };
    auto pair_basis_mul = [&](BtPairV3 value) -> BtPairV3 {
        return pair_make(
            pair_add(
                pair_add(
                    pair_mul(basis.data[0][0], value.x),
                    pair_mul(basis.data[0][1], value.y)),
                pair_mul(basis.data[0][2], value.z)),
            pair_add(
                pair_add(
                    pair_mul(basis.data[1][0], value.x),
                    pair_mul(basis.data[1][1], value.y)),
                pair_mul(basis.data[1][2], value.z)),
            pair_add(
                pair_add(
                    pair_mul(basis.data[2][0], value.x),
                    pair_mul(basis.data[2][1], value.y)),
                pair_mul(basis.data[2][2], value.z)));
    };
    auto pair_basis_transpose_mul = [&](BtPairV3 value) -> BtPairV3 {
        return pair_make(
            pair_add(
                pair_add(
                    pair_mul(basis.data[0][0], value.x),
                    pair_mul(basis.data[1][0], value.y)),
                pair_mul(basis.data[2][0], value.z)),
            pair_add(
                pair_add(
                    pair_mul(basis.data[0][1], value.x),
                    pair_mul(basis.data[1][1], value.y)),
                pair_mul(basis.data[2][1], value.z)),
            pair_add(
                pair_add(
                    pair_mul(basis.data[0][2], value.x),
                    pair_mul(basis.data[1][2], value.y)),
                pair_mul(basis.data[2][2], value.z)));
    };

    const BtPairV3 body = pair_make(
        body_origin_bt[0], body_origin_bt[1], body_origin_bt[2]);
    const BtPairV3 triangle[3] = {
        pair_make(v0_bt[0], v0_bt[1], v0_bt[2]),
        pair_make(v1_bt[0], v1_bt[1], v1_bt[2]),
        pair_make(v2_bt[0], v2_bt[1], v2_bt[2]),
    };
    const BtPairV3 child_offset = pair_make(0.277513981f, 0.0f, 0.415099978f);
    const BtPairV3 center = pair_vadd(body, pair_basis_mul(child_offset));
    const BtPairV3 position_offset = pair_vmul(center, 0.5f);
    const BtPairV3 local_origin_a = pair_vsub(center, position_offset);
    const BtPairV3 local_origin_b = pair_vneg(position_offset);
    const BtPairV3 box_half = pair_make(1.16507006f, 0.826994002f, 0.346590996f);
    const float margin_a = 0.0386590995f;
    const float margin_b = 0.0f;
    const float margin = pair_add(margin_a, margin_b);
    const float maximum_distance = pair_add(margin, 0.0406245552f);
    const float maximum_distance_squared = pair_mul(maximum_distance, maximum_distance);

    auto pair_box_support = [&](BtPairV3 direction) -> BtPairV3 {
        return pair_make(
            direction.x >= 0.0f ? box_half.x : -box_half.x,
            direction.y >= 0.0f ? box_half.y : -box_half.y,
            direction.z >= 0.0f ? box_half.z : -box_half.z);
    };
    auto pair_triangle_support = [&](BtPairV3 direction) -> BtPairV3 {
        const float dot0 = pair_dot(direction, triangle[0]);
        const float dot1 = pair_dot(direction, triangle[1]);
        const float dot2 = pair_dot(direction, triangle[2]);
        const int axis = dot0 < dot1
            ? (dot1 < dot2 ? 2 : 1)
            : (dot0 < dot2 ? 2 : 0);
        return triangle[axis];
    };
    auto pair_transform_a = [&](BtPairV3 local) -> BtPairV3 {
        return pair_vadd(pair_basis_mul(local), local_origin_a);
    };
    auto pair_transform_b = [&](BtPairV3 local) -> BtPairV3 {
        return pair_vadd(local, local_origin_b);
    };

    BtPairV3 simplex_w[5];
    BtPairV3 simplex_p[5];
    BtPairV3 simplex_q[5];
    int simplex_count = 0;
    BtPairV3 last_w = pair_make(1.0e18f, 1.0e18f, 1.0e18f);
    BtPairV3 cached_p1 = pair_make(0.0f, 0.0f, 0.0f);
    BtPairV3 cached_p2 = pair_make(0.0f, 0.0f, 0.0f);
    BtPairV3 cached_v = pair_make(0.0f, 0.0f, 0.0f);
    int cached_valid = 0;

    auto pair_result_reset = [&](BtPairClosest& result) {
        result.closest = pair_make(0.0f, 0.0f, 0.0f);
        result.weights[0] = 0.0f;
        result.weights[1] = 0.0f;
        result.weights[2] = 0.0f;
        result.weights[3] = 0.0f;
        result.used = 0;
        result.valid = 0;
        result.degenerate = 0;
    };
    auto pair_result_validate = [&](BtPairClosest& result) -> int {
        return result.weights[0] >= 0.0f
            && result.weights[1] >= 0.0f
            && result.weights[2] >= 0.0f
            && result.weights[3] >= 0.0f;
    };
    auto pair_triangle_closest = [&](BtPairV3 a,
                                     BtPairV3 b,
                                     BtPairV3 c,
                                     BtPairClosest& result) {
        pair_result_reset(result);
        const BtPairV3 ab = pair_vsub(b, a);
        const BtPairV3 ac = pair_vsub(c, a);
        const BtPairV3 ap = pair_vneg(a);
        const float d1 = pair_dot(ab, ap);
        const float d2 = pair_dot(ac, ap);
        if (d1 <= 0.0f && d2 <= 0.0f) {
            result.closest = a;
            result.weights[0] = 1.0f;
            result.used = 1;
            result.valid = 1;
            return;
        }
        const BtPairV3 bp = pair_vneg(b);
        const float d3 = pair_dot(ab, bp);
        const float d4 = pair_dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3) {
            result.closest = b;
            result.weights[1] = 1.0f;
            result.used = 2;
            result.valid = 1;
            return;
        }
        const float vc = pair_sub(pair_mul(d1, d4), pair_mul(d3, d2));
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float value = pair_div(d1, pair_sub(d1, d3));
            result.closest = pair_vadd(a, pair_vmul(ab, value));
            result.weights[0] = pair_sub(1.0f, value);
            result.weights[1] = value;
            result.used = 3;
            result.valid = 1;
            return;
        }
        const BtPairV3 cp = pair_vneg(c);
        const float d5 = pair_dot(ab, cp);
        const float d6 = pair_dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6) {
            result.closest = c;
            result.weights[2] = 1.0f;
            result.used = 4;
            result.valid = 1;
            return;
        }
        const float vb = pair_sub(pair_mul(d5, d2), pair_mul(d1, d6));
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float value = pair_div(d2, pair_sub(d2, d6));
            result.closest = pair_vadd(a, pair_vmul(ac, value));
            result.weights[0] = pair_sub(1.0f, value);
            result.weights[2] = value;
            result.used = 5;
            result.valid = 1;
            return;
        }
        const float va = pair_sub(pair_mul(d3, d6), pair_mul(d5, d4));
        const float d4_minus_d3 = pair_sub(d4, d3);
        const float d5_minus_d6 = pair_sub(d5, d6);
        if (va <= 0.0f && d4_minus_d3 >= 0.0f && d5_minus_d6 >= 0.0f) {
            const float value = pair_div(
                d4_minus_d3, pair_add(d4_minus_d3, d5_minus_d6));
            result.closest = pair_vadd(b, pair_vmul(pair_vsub(c, b), value));
            result.weights[1] = pair_sub(1.0f, value);
            result.weights[2] = value;
            result.used = 6;
            result.valid = 1;
            return;
        }
        const float denominator = pair_div(1.0f, pair_add(pair_add(va, vb), vc));
        const float value_v = pair_mul(vb, denominator);
        const float value_w = pair_mul(vc, denominator);
        result.closest = pair_vadd(
            pair_vadd(a, pair_vmul(ab, value_v)), pair_vmul(ac, value_w));
        result.weights[0] = pair_sub(pair_sub(1.0f, value_v), value_w);
        result.weights[1] = value_v;
        result.weights[2] = value_w;
        result.used = 7;
        result.valid = 1;
    };
    auto pair_outside_plane = [&](BtPairV3 a,
                                  BtPairV3 b,
                                  BtPairV3 c,
                                  BtPairV3 d) -> int {
        const BtPairV3 normal = pair_cross(pair_vsub(b, a), pair_vsub(c, a));
        const float sign_p = pair_dot(pair_vneg(a), normal);
        const float sign_d = pair_dot(pair_vsub(d, a), normal);
        if (pair_mul(sign_d, sign_d) < pair_mul(1.0e-4f, 1.0e-4f)) {
            return -1;
        }
        return pair_mul(sign_p, sign_d) < 0.0f ? 1 : 0;
    };
    auto pair_tetrahedron_closest = [&](BtPairV3 a,
                                        BtPairV3 b,
                                        BtPairV3 c,
                                        BtPairV3 d,
                                        BtPairClosest& result) -> int {
        pair_result_reset(result);
        result.used = 15;
        const int outside_abc = pair_outside_plane(a, b, c, d);
        const int outside_acd = pair_outside_plane(a, c, d, b);
        const int outside_adb = pair_outside_plane(a, d, b, c);
        const int outside_bdc = pair_outside_plane(b, d, c, a);
        if (outside_abc < 0 || outside_acd < 0 || outside_adb < 0 || outside_bdc < 0) {
            result.degenerate = 1;
            return 0;
        }
        if (!outside_abc && !outside_acd && !outside_adb && !outside_bdc) {
            return 0;
        }
        float best_distance_squared = 3.402823466e38f;
        BtPairClosest temporary;
        if (outside_abc) {
            pair_triangle_closest(a, b, c, temporary);
            const float candidate = pair_length2(temporary.closest);
            if (candidate < best_distance_squared) {
                best_distance_squared = candidate;
                result.closest = temporary.closest;
                result.used = temporary.used;
                result.weights[0] = temporary.weights[0];
                result.weights[1] = temporary.weights[1];
                result.weights[2] = temporary.weights[2];
                result.weights[3] = 0.0f;
            }
        }
        if (outside_acd) {
            pair_triangle_closest(a, c, d, temporary);
            const float candidate = pair_length2(temporary.closest);
            if (candidate < best_distance_squared) {
                best_distance_squared = candidate;
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
        if (outside_adb) {
            pair_triangle_closest(a, d, b, temporary);
            const float candidate = pair_length2(temporary.closest);
            if (candidate < best_distance_squared) {
                best_distance_squared = candidate;
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
        if (outside_bdc) {
            pair_triangle_closest(b, d, c, temporary);
            const float candidate = pair_length2(temporary.closest);
            if (candidate < best_distance_squared) {
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
        result.valid = pair_result_validate(result);
        return 1;
    };
    auto pair_remove_vertex = [&](int index) {
        --simplex_count;
        simplex_w[index] = simplex_w[simplex_count];
        simplex_p[index] = simplex_p[simplex_count];
        simplex_q[index] = simplex_q[simplex_count];
    };
    auto pair_reduce_vertices = [&](int used) {
        if (simplex_count >= 4 && !(used & 8)) pair_remove_vertex(3);
        if (simplex_count >= 3 && !(used & 4)) pair_remove_vertex(2);
        if (simplex_count >= 2 && !(used & 2)) pair_remove_vertex(1);
        if (simplex_count >= 1 && !(used & 1)) pair_remove_vertex(0);
    };
    auto pair_simplex_closest = [&]() -> int {
        BtPairClosest closest;
        pair_result_reset(closest);
        if (simplex_count == 1) {
            cached_p1 = simplex_p[0];
            cached_p2 = simplex_q[0];
            cached_v = pair_vsub(cached_p1, cached_p2);
            cached_valid = 1;
            return 1;
        }
        if (simplex_count == 2) {
            const BtPairV3 from = simplex_w[0];
            const BtPairV3 to = simplex_w[1];
            const BtPairV3 difference = pair_vsub(to, from);
            float parameter = pair_dot(difference, pair_vneg(from));
            int used = 0;
            if (parameter > 0.0f) {
                const float dot_vv = pair_dot(difference, difference);
                if (parameter < dot_vv) {
                    parameter = pair_div(parameter, dot_vv);
                    used = 3;
                } else {
                    parameter = 1.0f;
                    used = 2;
                }
            } else {
                parameter = 0.0f;
                used = 1;
            }
            closest.weights[0] = pair_sub(1.0f, parameter);
            closest.weights[1] = parameter;
            cached_p1 = pair_vadd(
                simplex_p[0],
                pair_vmul(pair_vsub(simplex_p[1], simplex_p[0]), parameter));
            cached_p2 = pair_vadd(
                simplex_q[0],
                pair_vmul(pair_vsub(simplex_q[1], simplex_q[0]), parameter));
            cached_v = pair_vsub(cached_p1, cached_p2);
            pair_reduce_vertices(used);
            cached_valid = pair_result_validate(closest);
            return cached_valid;
        }
        if (simplex_count == 3) {
            pair_triangle_closest(simplex_w[0], simplex_w[1], simplex_w[2], closest);
            cached_p1 = pair_vadd(
                pair_vadd(
                    pair_vmul(simplex_p[0], closest.weights[0]),
                    pair_vmul(simplex_p[1], closest.weights[1])),
                pair_vmul(simplex_p[2], closest.weights[2]));
            cached_p2 = pair_vadd(
                pair_vadd(
                    pair_vmul(simplex_q[0], closest.weights[0]),
                    pair_vmul(simplex_q[1], closest.weights[1])),
                pair_vmul(simplex_q[2], closest.weights[2]));
            cached_v = pair_vsub(cached_p1, cached_p2);
            pair_reduce_vertices(closest.used);
            cached_valid = pair_result_validate(closest);
            return cached_valid;
        }
        const int has_separation = pair_tetrahedron_closest(
            simplex_w[0], simplex_w[1], simplex_w[2], simplex_w[3], closest);
        if (has_separation) {
            cached_p1 = pair_vadd(
                pair_vadd(
                    pair_vadd(
                        pair_vmul(simplex_p[0], closest.weights[0]),
                        pair_vmul(simplex_p[1], closest.weights[1])),
                    pair_vmul(simplex_p[2], closest.weights[2])),
                pair_vmul(simplex_p[3], closest.weights[3]));
            cached_p2 = pair_vadd(
                pair_vadd(
                    pair_vadd(
                        pair_vmul(simplex_q[0], closest.weights[0]),
                        pair_vmul(simplex_q[1], closest.weights[1])),
                    pair_vmul(simplex_q[2], closest.weights[2])),
                pair_vmul(simplex_q[3], closest.weights[3]));
            cached_v = pair_vsub(cached_p1, cached_p2);
            pair_reduce_vertices(closest.used);
        } else if (closest.degenerate) {
            cached_valid = 0;
            return 0;
        } else {
            cached_v = pair_make(0.0f, 0.0f, 0.0f);
            cached_valid = 1;
            return 1;
        }
        cached_valid = pair_result_validate(closest);
        return cached_valid;
    };
    auto pair_in_simplex = [&](BtPairV3 value) -> int {
        for (int index = 0; index < simplex_count; ++index) {
            if (pair_length2(pair_vsub(simplex_w[index], value)) <= 0.0001f) {
                return 1;
            }
        }
        return value.x == last_w.x && value.y == last_w.y && value.z == last_w.z;
    };

    BtPairV3 axis = pair_make(0.0f, 1.0f, 0.0f);
    float squared_distance = 1.0e18f;
    float pair_distance = 0.0f;
    BtPairV3 pair_normal = pair_make(0.0f, 0.0f, 0.0f);
    int pair_valid = 0;
    int check_simplex = 0;
    int current_iteration = 0;
    int degenerate_simplex = 0;

    for (int guard = 0; guard < 1003; ++guard) {
        const BtPairV3 direction_a = pair_basis_transpose_mul(pair_vneg(axis));
        const BtPairV3 direction_b = axis;
        const BtPairV3 point_a = pair_transform_a(pair_box_support(direction_a));
        const BtPairV3 point_b = pair_transform_b(pair_triangle_support(direction_b));
        const BtPairV3 value = pair_vsub(point_a, point_b);
        const float delta = pair_dot(axis, value);
        if (delta > 0.0f
            && pair_mul(delta, delta)
                > pair_mul(squared_distance, maximum_distance_squared)) {
            degenerate_simplex = 10;
            check_simplex = 1;
            break;
        }
        if (pair_in_simplex(value)) {
            degenerate_simplex = 1;
            check_simplex = 1;
            break;
        }
        const float f0 = pair_sub(squared_distance, delta);
        const float f1 = pair_mul(squared_distance, 1.0e-6f);
        if (f0 <= f1) {
            degenerate_simplex = f0 <= 0.0f ? 2 : 11;
            check_simplex = 1;
            break;
        }
        last_w = value;
        simplex_w[simplex_count] = value;
        simplex_p[simplex_count] = point_a;
        simplex_q[simplex_count] = point_b;
        ++simplex_count;
        if (!pair_simplex_closest()) {
            degenerate_simplex = 3;
            check_simplex = 1;
            break;
        }
        const BtPairV3 new_axis = cached_v;
        const float new_length_squared = pair_length2(new_axis);
        if (new_length_squared < 1.0e-6f) {
            axis = new_axis;
            degenerate_simplex = 6;
            check_simplex = 1;
            break;
        }
        const float previous_squared_distance = squared_distance;
        squared_distance = new_length_squared;
        if (pair_sub(previous_squared_distance, squared_distance)
            <= pair_mul(1.1920929e-7f, previous_squared_distance)) {
            degenerate_simplex = 12;
            check_simplex = 1;
            break;
        }
        axis = new_axis;
        if (current_iteration++ > 1000) {
            break;
        }
        if (simplex_count == 4) {
            degenerate_simplex = 13;
            break;
        }
    }

    if (check_simplex) {
        BtPairV3 point_on_a = cached_p1;
        BtPairV3 point_on_b = cached_p2;
        pair_normal = axis;
        const float length_squared = pair_length2(axis);
        if (length_squared < 0.0001f) degenerate_simplex = 5;
        if (length_squared > pair_mul(1.1920929e-7f, 1.1920929e-7f)) {
            const float reciprocal_length = pair_div(1.0f, sqrtf(length_squared));
            pair_normal = pair_vmul(pair_normal, reciprocal_length);
            const float simplex_length = sqrtf(squared_distance);
            point_on_a = pair_vsub(
                point_on_a,
                pair_vmul(axis, pair_div(margin_a, simplex_length)));
            point_on_b = pair_vadd(
                point_on_b,
                pair_vmul(axis, pair_div(margin_b, simplex_length)));
            pair_distance = pair_sub(pair_div(1.0f, reciprocal_length), margin);
            pair_valid = 1;
        }
        cached_p1 = point_on_a;
        cached_p2 = point_on_b;
    }

    if (pair_valid
        && (pair_distance < 0.0f
            || pair_mul(pair_distance, pair_distance) < maximum_distance_squared)) {
        // getAabbSlow() for the shifted triangle followed by the pinned
        // centroid-direction workaround in getClosestPointsNonVirtual().
        const BtPairV3 shifted0 = pair_transform_b(triangle[0]);
        const BtPairV3 shifted1 = pair_transform_b(triangle[1]);
        const BtPairV3 shifted2 = pair_transform_b(triangle[2]);
        auto pair_min = [](float a, float b) -> float { return a < b ? a : b; };
        auto pair_max = [](float a, float b) -> float { return a > b ? a : b; };
        const BtPairV3 triangle_min = pair_make(
            pair_min(shifted0.x, pair_min(shifted1.x, shifted2.x)),
            pair_min(shifted0.y, pair_min(shifted1.y, shifted2.y)),
            pair_min(shifted0.z, pair_min(shifted1.z, shifted2.z)));
        const BtPairV3 triangle_max = pair_make(
            pair_max(shifted0.x, pair_max(shifted1.x, shifted2.x)),
            pair_max(shifted0.y, pair_max(shifted1.y, shifted2.y)),
            pair_max(shifted0.z, pair_max(shifted1.z, shifted2.z)));
        const BtPairV3 triangle_center = pair_vmul(
            pair_vadd(triangle_min, triangle_max), 0.5f);
        const BtPairV3 center_difference = pair_vsub(local_origin_a, triangle_center);
        if (pair_dot(center_difference, pair_normal) < 0.0f) {
            pair_normal = pair_vneg(pair_normal);
        }
        const BtPairV3 world_point_b = pair_vadd(cached_p2, position_offset);
        const BtPairV3 world_point_a = pair_vadd(
            world_point_b, pair_vmul(pair_normal, pair_distance));
        point_a_bt = wp::vec_t<3, wp::float32>(
            world_point_a.x, world_point_a.y, world_point_a.z);
        point_b_bt = wp::vec_t<3, wp::float32>(
            world_point_b.x, world_point_b.y, world_point_b.z);
        normal_world = wp::vec_t<3, wp::float32>(
            pair_normal.x, pair_normal.y, pair_normal.z);
        distance_bt = pair_distance;
        valid = 1;
    } else {
        // ``isValid`` and whether getClosestPointsNonVirtual reports a
        // callback are distinct source states.  calcPenDepth is selected from
        // the former, before the maximum-distance callback gate below it.
        // Preserve a valid shallow GJK witness even when it is too far away to
        // report; otherwise the caller incorrectly treats that witness as an
        // invalid GJK solve and invokes EPA.
        if (pair_valid) {
            const BtPairV3 world_point_b = pair_vadd(cached_p2, position_offset);
            const BtPairV3 world_point_a = pair_vadd(
                world_point_b, pair_vmul(pair_normal, pair_distance));
            point_a_bt = wp::vec_t<3, wp::float32>(
                world_point_a.x, world_point_a.y, world_point_a.z);
            point_b_bt = wp::vec_t<3, wp::float32>(
                world_point_b.x, world_point_b.y, world_point_b.z);
            normal_world = wp::vec_t<3, wp::float32>(
                pair_normal.x, pair_normal.y, pair_normal.z);
            distance_bt = pair_distance;
            valid = 1;
        } else {
            valid = 0;
        }
    }
    degenerate_status = degenerate_simplex;
"""


@wp.func_native(_BULLET_BOX_TRIANGLE_CLOSEST)
def bullet_box_triangle_closest(
    body_origin_bt: wp.vec3,
    basis: wp.mat33,
    v0_bt: wp.vec3,
    v1_bt: wp.vec3,
    v2_bt: wp.vec3,
    point_a_bt: wp.ref[wp.vec3],
    point_b_bt: wp.ref[wp.vec3],
    normal_world: wp.ref[wp.vec3],
    distance_bt: wp.ref[wp.float32],
    valid: wp.ref[wp.int32],
    degenerate_status: wp.ref[wp.int32],
): ...


_BULLET_INTERNAL_EDGE_BEST = r"""
    struct BtEdgeV3 {
        float x;
        float y;
        float z;
    };
    auto edge_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto edge_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
        return value;
    #endif
    };
    auto edge_mul = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fmul_rn(a, b);
    #else
        volatile float value = a * b;
        return value;
    #endif
    };
    auto edge_div = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fdiv_rn(a, b);
    #else
        volatile float value = a / b;
        return value;
    #endif
    };
    auto edge_make = [](float x, float y, float z) -> BtEdgeV3 {
        BtEdgeV3 result = {x, y, z};
        return result;
    };
    auto edge_vsub = [&](BtEdgeV3 a, BtEdgeV3 b) -> BtEdgeV3 {
        return edge_make(
            edge_sub(a.x, b.x),
            edge_sub(a.y, b.y),
            edge_sub(a.z, b.z));
    };
    auto edge_vadd = [&](BtEdgeV3 a, BtEdgeV3 b) -> BtEdgeV3 {
        return edge_make(
            edge_add(a.x, b.x),
            edge_add(a.y, b.y),
            edge_add(a.z, b.z));
    };
    auto edge_vmul = [&](BtEdgeV3 value, float scale) -> BtEdgeV3 {
        return edge_make(
            edge_mul(value.x, scale),
            edge_mul(value.y, scale),
            edge_mul(value.z, scale));
    };
    auto edge_dot = [&](BtEdgeV3 a, BtEdgeV3 b) -> float {
        return edge_add(
            edge_add(edge_mul(a.x, b.x), edge_mul(a.y, b.y)),
            edge_mul(a.z, b.z));
    };
    auto nearest_distance = [&](BtEdgeV3 point,
                                BtEdgeV3 line0,
                                BtEdgeV3 line1) -> float {
        const BtEdgeV3 line_delta = edge_vsub(line1, line0);
        BtEdgeV3 nearest = line0;
        if (edge_dot(line_delta, line_delta)
            >= 1.1920929e-7f * 1.1920929e-7f) {
            float delta = edge_div(
                edge_dot(edge_vsub(point, line0), line_delta),
                edge_dot(line_delta, line_delta));
            if (delta < 0.0f) {
                delta = 0.0f;
            } else if (delta > 1.0f) {
                delta = 1.0f;
            }
            nearest = edge_vadd(line0, edge_vmul(line_delta, delta));
        }
        const BtEdgeV3 separation = edge_vsub(point, nearest);
        return sqrtf(edge_dot(separation, separation));
    };

    const BtEdgeV3 point = edge_make(point_b_bt[0], point_b_bt[1], point_b_bt[2]);
    const BtEdgeV3 vertices[3] = {
        edge_make(v0_bt[0], v0_bt[1], v0_bt[2]),
        edge_make(v1_bt[0], v1_bt[1], v1_bt[2]),
        edge_make(v2_bt[0], v2_bt[1], v2_bt[2]),
    };
    int best = -1;
    float best_distance = 3.402823466e+38f;
    for (int edge = 0; edge < 3; ++edge) {
        const float angle = edge_angles[edge];
        const float absolute_angle = angle < 0.0f ? -angle : angle;
        if (absolute_angle < 6.28318530717958647692f) {
            const int end = edge == 2 ? 0 : edge + 1;
            const float distance = nearest_distance(
                point, vertices[edge], vertices[end]);
            if (distance < best_distance) {
                best = edge;
                best_distance = distance;
            }
        }
    }
    best_distance_bt = best_distance;
    return best;
"""


@wp.func_native(_BULLET_INTERNAL_EDGE_BEST)
def bullet_internal_edge_best(
    point_b_bt: wp.vec3,
    v0_bt: wp.vec3,
    v1_bt: wp.vec3,
    v2_bt: wp.vec3,
    edge_angles: wp.vec3,
    best_distance_bt: wp.ref[wp.float32],
) -> wp.int32: ...


_BULLET_MANIFOLD_REPLACEMENT = r"""
    auto bt_add = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fadd_rn(a, b);
    #else
        volatile float value = a + b;
        return value;
    #endif
    };
    auto bt_sub = [](float a, float b) -> float {
    #if defined(__CUDA_ARCH__)
        return __fsub_rn(a, b);
    #else
        volatile float value = a - b;
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
    auto score = [&](const wp::vec_t<3, wp::float32>& a,
                     const wp::vec_t<3, wp::float32>& b,
                     const wp::vec_t<3, wp::float32>& c,
                     const wp::vec_t<3, wp::float32>& d) -> float {
        const float ax = bt_sub(a[0], b[0]);
        const float ay = bt_sub(a[1], b[1]);
        const float az = bt_sub(a[2], b[2]);
        const float bx = bt_sub(c[0], d[0]);
        const float by = bt_sub(c[1], d[1]);
        const float bz = bt_sub(c[2], d[2]);
        const float x = bt_sub(bt_mul(ay, bz), bt_mul(az, by));
        const float y = bt_sub(bt_mul(az, bx), bt_mul(ax, bz));
        const float z = bt_sub(bt_mul(ax, by), bt_mul(ay, bx));
        return bt_add(bt_add(bt_mul(x, x), bt_mul(y, y)), bt_mul(z, z));
    };

    int deepest = -1;
    float deepest_distance = candidate_distance;
    if (distance0 < deepest_distance) {
        deepest = 0;
        deepest_distance = distance0;
    }
    if (distance1 < deepest_distance) {
        deepest = 1;
        deepest_distance = distance1;
    }
    if (distance2 < deepest_distance) {
        deepest = 2;
        deepest_distance = distance2;
    }
    if (distance3 < deepest_distance) {
        deepest = 3;
    }

    float result0 = 0.0f;
    float result1 = 0.0f;
    float result2 = 0.0f;
    float result3 = 0.0f;
    if (deepest != 0) result0 = score(candidate, point1, point3, point2);
    if (deepest != 1) result1 = score(candidate, point0, point3, point2);
    if (deepest != 2) result2 = score(candidate, point0, point3, point1);
    if (deepest != 3) result3 = score(candidate, point0, point2, point1);

    // btVector4::closestAxis4() is absolute4().maxAxis4(); scores are already
    // non-negative.  maxAxis4 uses strict comparisons in lane order, so exact
    // ties retain the earliest lane.
    int result = 0;
    float maximum = result0;
    if (result1 > maximum) {
        result = 1;
        maximum = result1;
    }
    if (result2 > maximum) {
        result = 2;
        maximum = result2;
    }
    if (result3 > maximum) {
        result = 3;
    }
    return result;
"""


@wp.func_native(_BULLET_MANIFOLD_REPLACEMENT)
def bullet_manifold_replacement(
    candidate: wp.vec3,
    point0: wp.vec3,
    point1: wp.vec3,
    point2: wp.vec3,
    point3: wp.vec3,
    candidate_distance: wp.float32,
    distance0: wp.float32,
    distance1: wp.float32,
    distance2: wp.float32,
    distance3: wp.float32,
) -> wp.int32: ...
