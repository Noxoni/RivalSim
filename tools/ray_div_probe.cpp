#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include <immintrin.h>

#include "bullet3-3.24/LinearMath/btTransform.h"

int main() {
    const float lambda = 0.110101461f;
    const float v[3] = {-0.385259151f, 0.292722702f, 0.243446469f};
    const float w[3] = {0.159785271f, -0.121406555f, 0.549686849f};
    const float r[3] = {0.0f, 0.0f, -0.991100013f};
    const float dot_w = (v[0] * w[0] + v[1] * w[1]) + v[2] * w[2];
    const float dot_r = (v[0] * r[0] + v[1] * r[1]) + v[2] * r[2];
    const float result = lambda - dot_w / dot_r;
    std::cout << std::setprecision(9)
              << "dot_w=" << dot_w << " dot_r=" << dot_r
              << " result=" << result
              << " bits=" << std::bit_cast<std::uint32_t>(result) << '\n';

    const __m128 vv = _mm_setr_ps(v[0], v[1], v[2], 0.0f);
    const __m128 ww = _mm_setr_ps(w[0], w[1], w[2], 0.0f);
    const __m128 rr = _mm_setr_ps(r[0], r[1], r[2], 0.0f);
    const auto dot = [](__m128 a, __m128 b) {
        __m128 value = _mm_mul_ps(a, b);
        const __m128 z = _mm_movehl_ps(value, value);
        const __m128 y = _mm_shuffle_ps(value, value, 0x55);
        value = _mm_add_ss(value, y);
        value = _mm_add_ss(value, z);
        return _mm_cvtss_f32(value);
    };
    const float sse_w = dot(vv, ww);
    const float sse_r = dot(vv, rr);
    const float sse_result = lambda - sse_w / sse_r;
    std::cout << "sse_w=" << sse_w << " sse_r=" << sse_r
              << " result=" << sse_result
              << " bits=" << std::bit_cast<std::uint32_t>(sse_result) << '\n';

    btVector3 direction(-0.561258316f, 0.426447868f, -0.195868194f);
    direction.normalize();
    const btTransform sphere(
        btQuaternion::getIdentity(),
        btVector3(5.94047451f, -11.739295f, -0.723971069f));
    const btVector3 support = sphere(1.8249999284744263f * direction);
    std::cout << "normalized=" << direction.x() << ',' << direction.y() << ','
              << direction.z() << " support=" << support.x() << ','
              << support.y() << ',' << support.z()
              << " support_z_bits="
              << std::bit_cast<std::uint32_t>(support.z()) << '\n';
}
