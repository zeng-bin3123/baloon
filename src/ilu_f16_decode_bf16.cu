// ilu_f16_decode_bf16.cu - device evaluation of the fp16 decoder's normalisation / re-bias model and of the bf16
// rounding model, next to the real device conversions, so tests can compare them bit for bit.
#include <cuda_runtime.h>
#include <cstdint>
#include "baloon/f16_decode_bf16.hpp"

namespace baloon {
namespace f16_decode_dev {

static __global__ void f16_decode_kernel(const uint16_t* h, int32_t* model_field, uint32_t* norm_m, uint32_t* norm_s,
                                         uint32_t* real_bits, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t v = h[i];
    const int32_t ex = (int32_t)((v >> 10) & 0x1Fu);
    const uint32_t m = v & 0x3FFu;
    const f16bits::NormShift ns = f16bits::norm_shift(10u, m, 0u);
    norm_m[i] = ns.m; norm_s[i] = ns.s;
    int32_t field;
    if (ex == 0) field = (m == 0u) ? 0 : f16bits::subnormal_exp_field(ns.s);
    else if (ex == 31) field = 255;
    else field = f16bits::normal_exp_field(ex);
    model_field[i] = field;
    real_bits[i] = f16bits::bits_of(f16bits::f32_from_f16((uint16_t)v));
}

static __global__ void bf16_round_kernel(const uint32_t* bits, uint32_t* model, uint16_t* real, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    model[i] = (uint32_t)f16bits::bf16_round((uint64_t)bits[i]);
    real[i] = f16bits::round_f32_to_bf16_bits(f16bits::float_of(bits[i]));
}

static int finish(cudaStream_t s) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    e = cudaStreamSynchronize(s);
    return e == cudaSuccess ? 0 : (int)e;
}

int f16_decode_step(const uint16_t* h, int32_t* model_field, uint32_t* norm_m, uint32_t* norm_s, uint32_t* real_bits,
                    int64_t n, void* stream) {
    if (n < 0 || (n > 0 && (!h || !model_field || !norm_m || !norm_s || !real_bits))) return -1;
    if (n == 0) return 0;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    if (blocks > 0x7FFFFFFF) return -1;
    f16_decode_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(h, model_field, norm_m, norm_s, real_bits, n);
    return finish((cudaStream_t)stream);
}

int bf16_round_step(const uint32_t* bits, uint32_t* model, uint16_t* real, int64_t n, void* stream) {
    if (n < 0 || (n > 0 && (!bits || !model || !real))) return -1;
    if (n == 0) return 0;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    if (blocks > 0x7FFFFFFF) return -1;
    bf16_round_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(bits, model, real, n);
    return finish((cudaStream_t)stream);
}

}  // namespace f16_decode_dev
}  // namespace baloon
