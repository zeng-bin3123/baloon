// f16_bits_rounding.cu - device-side conversions: one thread per input word runs the same integer rounding path as the
// host so the GPU result can be compared bit-for-bit.
#include <cuda_runtime.h>
#include <cstdint>
#include "baloon/f16_bits_rounding.hpp"
#include "f16_bits_rounding_dev.h"

namespace f16dev {
using namespace baloon::f16bits;
using namespace baloon::f16;

__global__ void f16_round_kernel(const uint32_t* in, uint16_t* half_out, uint16_t* bf_out, int32_t* top_out, uint32_t* rne_out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t x = in[i];
    half_out[i] = round_f32_to_f16_bits(baloon::f16::bits_f32(x));
    bf_out[i] = round_f32_to_bf16_bits(baloon::f16::bits_f32(x));
    top_out[i] = encode_top((x >> 16) & 0x8000u, (x >> 23) & 0xFFu, x & 0x7FFFFFu);
    rne_out[i] = rne_step(x >> 13 & 0x3FFFFu, x & 0x1FFFu);
}

int launch_round(const uint32_t* in, uint16_t* half_out, uint16_t* bf_out, int32_t* top_out, uint32_t* rne_out, int n) {
    if (n <= 0) return -1;
    f16_round_kernel<<<(n + 255) / 256, 256>>>(in, half_out, bf_out, top_out, rne_out, n);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -2;
}
}  // namespace f16dev
