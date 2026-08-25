#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>

namespace {

std::uint32_t RsqrtBits(std::uint32_t bits) {
    __m128 input = _mm_castsi128_ps(_mm_cvtsi32_si128(bits));
    __m128 output = _mm_rsqrt_ss(input);
    return static_cast<std::uint32_t>(
        _mm_cvtsi128_si32(_mm_castps_si128(output)));
}

}  // namespace

int main() {
    std::uint16_t mantissa[8192] = {};
    for (std::uint32_t index = 0; index < 8192; ++index) {
        const std::uint32_t input = 0x3f000000u + (index << 11);
        mantissa[index] = static_cast<std::uint16_t>(
            (RsqrtBits(input) & 0x007fffffu) >> 11);
    }

    std::uint32_t state = 0x20260824u;
    for (std::uint32_t iteration = 0; iteration < 10000000u; ++iteration) {
        state = state * 1664525u + 1013904223u;
        const std::uint32_t exponent = 1u + ((state >> 23) % 254u);
        const std::uint32_t input = (exponent << 23) | (state & 0x007fffffu);
        const std::uint32_t output_exponent = ((380u - exponent) >> 1) << 23;
        const std::uint32_t predicted =
            output_exponent
            | (static_cast<std::uint32_t>(
                   mantissa[(input >> 11) & 0x1fffu])
               << 11);
        const std::uint32_t observed = RsqrtBits(input);
        if (predicted != observed) {
            std::printf(
                "mismatch input=%08X predicted=%08X observed=%08X\n",
                input,
                predicted,
                observed);
            return 1;
        }
    }

    std::printf("AMD_RSQRTSS_MANTISSA_HEX=");
    for (std::uint32_t index = 0; index < 8192; ++index) {
        std::printf("%03X", mantissa[index]);
    }
    std::printf("\n");
    return 0;
}
