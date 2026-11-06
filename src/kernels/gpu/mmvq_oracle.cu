// mmvq_oracle.cu - plain one-thread-per-output reference kernel for Q8_0 / Q4_0 weights times Q8_1 activations
// (see include/baloon/mmvq_oracle.hpp). Accumulates in float in block order (the device has no dependable fp64 path: every production kernel here is fp32); the tuned kernels must agree with it to
// float rounding of their own summation order.
#include "baloon/mmvq_oracle.hpp"

#include "baloon/f16_bits.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int kBlockElems = 32;
constexpr int kQ81Bytes = 36;

// integer dot of one 32-value weight block with one Q8_1 block; type 8 = Q8_0, type 2 = Q4_0
__device__ __forceinline__ int block_int_dot(int type, const uint8_t* wb, const uint8_t* xb) {
    int acc = 0;
    for (int i = 0; i < kBlockElems; ++i) {
        int wv;
        if (type == 8) wv = (int)(int8_t)wb[2 + i];
        else wv = (int)((i < 16 ? (wb[2 + i] & 15) : (wb[2 + i - 16] >> 4)) - 8);
        acc += wv * (int)(int8_t)xb[4 + i];
    }
    return acc;
}
}  // namespace

// External linkage and a plain (non-template) name on purpose: launch tooling identifies kernels by symbol.
__global__ void mmvq_oracle_kernel(int type, const uint8_t* __restrict__ w, const uint8_t* __restrict__ x, float* __restrict__ y, int n_in, int n_out, int ncols) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n_out * ncols) return;
    const int row = (int)(idx % n_out), col = (int)(idx / n_out);
    const int nb = n_in / kBlockElems;
    const int bytes = type == 8 ? 34 : 18;
    const uint8_t* wr = w + (size_t)row * nb * bytes;
    const uint8_t* xc = x + (size_t)col * nb * kQ81Bytes;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = wr + (size_t)b * bytes;
        const uint8_t* xb = xc + (size_t)b * kQ81Bytes;
        const uint16_t dw = (uint16_t)(wb[0] | (wb[1] << 8)), dx = (uint16_t)(xb[0] | (xb[1] << 8));
        acc += (f16bits::f32_from_f16(dw) * f16bits::f32_from_f16(dx)) * (float)block_int_dot(type, wb, xb);
    }
    y[idx] = acc;
}

bool mmvq_oracle_supported(int type) { return type == 8 || type == 2; }

cudaError_t mmvq_oracle(int type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    if (!mmvq_oracle_supported(type) || w == nullptr || x_q8_1 == nullptr || y == nullptr) return cudaErrorInvalidValue;
    if (ncols < 1 || n_in <= 0 || n_out <= 0 || n_in % kBlockElems != 0) return cudaErrorInvalidValue;
    const long long total = (long long)n_out * ncols;
    const unsigned blocks = (unsigned)((total + 127) / 128);
    mmvq_oracle_kernel<<<blocks, 128, 0, s>>>(type, static_cast<const uint8_t*>(w), static_cast<const uint8_t*>(x_q8_1), y, n_in, n_out, ncols);
    return cudaGetLastError();
}

}  // namespace kg
}  // namespace baloon
