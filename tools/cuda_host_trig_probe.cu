#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr std::size_t kSampleCount = 2'000'000;

__device__ float UcrtSmallSin(float value) {
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
}

__device__ float UcrtSmallCos(float value) {
    const unsigned bits = __float_as_uint(value) & 0x7fffffffu;
    if (bits < 0x39000000u) return 1.0f;
    const double x = static_cast<double>(value);
    if (bits < 0x3c000000u) {
        const double halfX = __dmul_rn(
            x, __longlong_as_double(0x3fe0000000000000ULL));
        return static_cast<float>(__fma_rn(-x, halfX, 1.0));
    }
    const double x2 = __dmul_rn(x, x);
    const double halfX2 = __dmul_rn(
        x2, __longlong_as_double(0x3fe0000000000000ULL));
    const double base = __dsub_rn(1.0, halfX2);
    double polynomial = __longlong_as_double(0x3efa01a01a01a019ULL);
    polynomial = __fma_rn(
        x2, __longlong_as_double(0xbe927e4fb7789f5cULL), polynomial);
    polynomial = __fma_rn(
        polynomial, x2, __longlong_as_double(0xbf56c16c16c16c16ULL));
    polynomial = __fma_rn(
        polynomial, x2, __longlong_as_double(0x3fa5555555555555ULL));
    const double x4 = __dmul_rn(x2, x2);
    return static_cast<float>(__fma_rn(polynomial, x4, base));
}

__global__ void EvaluateTrig(
    const float* inputs,
    std::uint32_t* floatSin,
    std::uint32_t* floatCos,
    std::uint32_t* doubleSin,
    std::uint32_t* doubleCos,
    std::uint32_t* ucrtSin,
    std::uint32_t* ucrtCos,
    std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float value = inputs[index];
    floatSin[index] = __float_as_uint(::sinf(value));
    floatCos[index] = __float_as_uint(::cosf(value));
    doubleSin[index] = __float_as_uint(
        static_cast<float>(::sin(static_cast<double>(value))));
    doubleCos[index] = __float_as_uint(
        static_cast<float>(::cos(static_cast<double>(value))));
    ucrtSin[index] = __float_as_uint(UcrtSmallSin(value));
    ucrtCos[index] = __float_as_uint(UcrtSmallCos(value));
}

std::uint32_t FloatBits(float value) {
    union {
        float number;
        std::uint32_t bits;
    } converted = {value};
    return converted.bits;
}

void Check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
        std::exit(2);
    }
}

}  // namespace

int main() {
    std::vector<float> inputs(kSampleCount);
    std::uint32_t state = 0xC0DEC0DEu;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        const std::uint32_t exponent = 1u + ((state >> 24) % 125u);
        state = state * 1664525u + 1013904223u;
        std::uint32_t mantissa = state & 0x007fffffu;
        if (exponent == 125u) {
            mantissa %= 0x004ccccdu;
        }
        union {
            std::uint32_t bits;
            float number;
        } converted = {(exponent << 23) | mantissa};
        inputs[index] = converted.number;
    }
    inputs[0] = 0.0f;
    inputs[1] = 0.0246152766f;
    inputs[2] = 0.3926990926f;
    inputs[3] = 0.4f;

    float* deviceInputs = nullptr;
    std::uint32_t* deviceOutputs = nullptr;
    Check(cudaMalloc(&deviceInputs, inputs.size() * sizeof(float)), "cudaMalloc inputs");
    Check(
        cudaMalloc(&deviceOutputs, inputs.size() * sizeof(std::uint32_t) * 6),
        "cudaMalloc outputs");
    Check(
        cudaMemcpy(
            deviceInputs,
            inputs.data(),
            inputs.size() * sizeof(float),
            cudaMemcpyHostToDevice),
        "cudaMemcpy inputs");

    const int threads = 256;
    const int blocks = static_cast<int>((inputs.size() + threads - 1) / threads);
    EvaluateTrig<<<blocks, threads>>>(
        deviceInputs,
        deviceOutputs,
        deviceOutputs + inputs.size(),
        deviceOutputs + inputs.size() * 2,
        deviceOutputs + inputs.size() * 3,
        deviceOutputs + inputs.size() * 4,
        deviceOutputs + inputs.size() * 5,
        inputs.size());
    Check(cudaGetLastError(), "EvaluateTrig launch");

    std::vector<std::uint32_t> outputs(inputs.size() * 6);
    Check(
        cudaMemcpy(
            outputs.data(),
            deviceOutputs,
            outputs.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy outputs");
    Check(cudaFree(deviceOutputs), "cudaFree outputs");
    Check(cudaFree(deviceInputs), "cudaFree inputs");

    std::size_t floatSinMismatches = 0;
    std::size_t floatCosMismatches = 0;
    std::size_t doubleSinMismatches = 0;
    std::size_t doubleCosMismatches = 0;
    std::size_t ucrtSinMismatches = 0;
    std::size_t ucrtCosMismatches = 0;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const std::uint32_t hostSin = FloatBits(::sinf(inputs[index]));
        const std::uint32_t hostCos = FloatBits(::cosf(inputs[index]));
        floatSinMismatches += outputs[index] != hostSin;
        floatCosMismatches += outputs[inputs.size() + index] != hostCos;
        doubleSinMismatches += outputs[inputs.size() * 2 + index] != hostSin;
        doubleCosMismatches += outputs[inputs.size() * 3 + index] != hostCos;
        ucrtSinMismatches += outputs[inputs.size() * 4 + index] != hostSin;
        ucrtCosMismatches += outputs[inputs.size() * 5 + index] != hostCos;
    }
    std::printf(
        "samples=%zu domain=[0,0.4] cuda_sinf_mismatches=%zu "
        "cuda_cosf_mismatches=%zu cuda_double_sin_cast_mismatches=%zu "
        "cuda_double_cos_cast_mismatches=%zu ucrt_sin_port_mismatches=%zu "
        "ucrt_cos_port_mismatches=%zu\n",
        inputs.size(),
        floatSinMismatches,
        floatCosMismatches,
        doubleSinMismatches,
        doubleCosMismatches,
        ucrtSinMismatches,
        ucrtCosMismatches);
    return ucrtSinMismatches == 0 && ucrtCosMismatches == 0 ? 0 : 1;
}
