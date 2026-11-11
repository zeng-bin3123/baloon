// dequant_bf16.cu - dequantize ggml/engine block formats (and plain F32/F16/BF16) to bf16, f16 or f32 on the GPU.
// One thread per 32-element group (group32); every group lies inside one block because all block sizes of the
// supported quantized types are multiples of 32.
#include <cuda_runtime.h>
#include "baloon/isa_selfprobe.hpp"

#include <cstdint>

#include "baloon/kern_cuda_misc.hpp"
#include "baloon/kg_kernels.hpp"
#include "kern_cu_blocks.cuh"

namespace baloon {
namespace kg {
// The i-quant route of dequant_f16/f32 calls iq_kernels.cu. Weak references: a binary that links dequant_bf16.cu without
// iq_kernels.cu still links, and the route then reports cudaErrorInvalidValue (the i-quants are not decodable here).
__attribute__((weak)) cudaError_t iq_dequant_f32(Fmt f, const uint8_t* src, float* dst, long long n, const IqGrids& g, cudaStream_t s);
__attribute__((weak)) cudaError_t iq_dequant_f16(Fmt f, const uint8_t* src, uint16_t* dst, long long n, const IqGrids& g, cudaStream_t s);
}  // namespace kg
}  // namespace baloon

namespace baloon {
namespace kern {

#define REF_DQ 1   // the dequantization rule is the reference rule of include/baloon/scalar_dequantizers.hpp

// Output element holders: a 16-bit pattern (bf16 / f16) or a float.
struct H16 { uint16_t bits; };

constexpr int8_t kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

namespace {

enum Out { OUT_BF16 = 0, OUT_F16 = 1, OUT_F32 = 2 };

// f2bf: (u + 0x7FFF + ((u >> 16) & 1)) >> 16 on the f32 bits, wrapping mod 2^32, nearest/ties-to-even, NO NaN guard.
__device__ __forceinline__ uint16_t f2bf(float f) {
    const uint32_t u = (uint32_t)__float_as_uint(f) + 0x7FFFu + (((uint32_t)__float_as_uint(f) >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float h2f(uint16_t h) { return f16bits::f32_from_f16(h); }

__device__ __forceinline__ int scale_min_k4(int j, const uint8_t* q, int& d, int& m) { scale_min_k4_dev(j, q, d, m); return 0; }

template <int O> __device__ __forceinline__ void put(void* dst, int64_t i, float v);
template <> __device__ __forceinline__ void put<OUT_BF16>(void* dst, int64_t i, float v) { ((uint16_t*)dst)[i] = f2bf(v); }
template <> __device__ __forceinline__ void put<OUT_F16>(void* dst, int64_t i, float v) { ((uint16_t*)dst)[i] = f16bits::f16_from_f32(v); }
template <> __device__ __forceinline__ void put<OUT_F32>(void* dst, int64_t i, float v) { ((float*)dst)[i] = v; }

// Value of element e of a flat tensor of the given type.
__device__ __forceinline__ float elem_at(int type, const uint8_t* src, int64_t e) {
    switch (type) {
        case 0: return ((const float*)src)[e];
        case 1: return h2f(((const uint16_t*)src)[e]);
        case 30: return f16bits::bf16_bits_to_f32(((const uint16_t*)src)[e]);
        default: {
            const int be = block_elems_of(type), bb = block_bytes_of(type);
            return dq_elem(type, src + (e / be) * (int64_t)bb, (int)(e % be));
        }
    }
}

// The 32 elements [32 g, 32 g + 32) of the tensor.
template <int O>
__device__ __forceinline__ void group32(int type, const uint8_t* src, void* dst, int64_t g, int64_t n) {
    for (int j = 0; j < 32; ++j) {
        const int64_t e = g * 32 + j;
        if (e < n) put<O>(dst, e, elem_at(type, src, e));
    }
}

template <int O>
__global__ void dequant_kernel(int type, const uint8_t* src, void* dst, int64_t n) {
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g * 32 >= n) return;
    group32<O>(type, src, dst, g, n);
}

// i-quant (grid-codebook) types served by the iq_kernels path, not by this block-format table:
// IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22, IQ1_M 29. Disjoint from dequant_bf16_supported().
bool iq_only(int type) {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 29;
}
void geometry(int64_t n, dim3& grid, dim3& block) {
    const int64_t groups = (n + 31) / 32;
    block = dim3(256);
    grid = dim3((unsigned)((groups + 255) / 256));
}

template <int O>
cudaError_t launch(int type, const void* src, void* dst, int64_t n, cudaStream_t s) {
    if (!dequant_bf16_supported(type) || src == nullptr || dst == nullptr || n < 0) return cudaErrorInvalidValue;
    const int be = block_elems_of(type);
    if (be > 0 && n % be != 0) return cudaErrorInvalidValue;   // a quantized tensor is a whole number of blocks
    if (n == 0) return cudaSuccess;
    dim3 grid, block;
    geometry(n, grid, block);
    dequant_kernel<O><<<grid, block, 0, s>>>(type, (const uint8_t*)src, dst, n);
    return cudaGetLastError();
}

}  // namespace

bool dequant_bf16_supported(int type) {
    switch (type) {
        case 0: case 1: case 30:
        case T_Q4_0: case T_Q5_0: case T_Q8_0: case T_IQ4_NL: case T_Q3_K: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ4_XS: case T_Q2_0: return true;
        default: return false;
    }
}

cudaError_t dequant_bf16(int type, const void* src, uint16_t* dst, int64_t n, cudaStream_t s) { return launch<OUT_BF16>(type, src, dst, n, s); }
cudaError_t dequant_f16(int type, const void* src, uint16_t* dst, int64_t n, cudaStream_t s) { return launch<OUT_F16>(type, src, dst, n, s); }
cudaError_t dequant_f32(int type, const void* src, float* dst, int64_t n, cudaStream_t s) { return launch<OUT_F32>(type, src, dst, n, s); }

// dequant_f16 / dequant_f32 route the i-quant types {16,17,18,21,22,29} to iq_kernels (kg::iq_dequant_*, which needs the
// device codebooks in `iq`); every other type takes the block-format table above. dequant_bf16 has no such route.
cudaError_t dequant_f16(int type, const void* src, uint16_t* dst, int64_t n, cudaStream_t s, const kg::IqGrids* iq) {
    if (!iq_only(type)) return launch<OUT_F16>(type, src, dst, n, s);
    if (iq == nullptr || !kg::iq_dequant_f16 || src == nullptr || dst == nullptr || n < 0) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    return kg::iq_dequant_f16((kg::Fmt)type, (const uint8_t*)src, dst, (long long)n, *iq, s);
}
cudaError_t dequant_f32(int type, const void* src, float* dst, int64_t n, cudaStream_t s, const kg::IqGrids* iq) {
    if (!iq_only(type)) return launch<OUT_F32>(type, src, dst, n, s);
    if (iq == nullptr || !kg::iq_dequant_f32 || src == nullptr || dst == nullptr || n < 0) return cudaErrorInvalidValue;
    if (n == 0) return cudaSuccess;
    return kg::iq_dequant_f32((kg::Fmt)type, (const uint8_t*)src, dst, (long long)n, *iq, s);
}

}  // namespace kern
}  // namespace baloon
