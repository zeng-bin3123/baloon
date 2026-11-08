// flash_short.cu - host-validated entry of the short-context flash decode step (Q 24x256, KV 2x256 fp16, width <= 256).
// Unlike native_flash_attn_step (whose kernel answers a bad step with NaN + *status = 1), native_flash_attn_short_step rejects
// a malformed step on the host before any launch: shape facts, 8-byte-aligned non-empty buffers, and no overlap between the
// output / status buffers and any input or each other. validate_spans is the pure address-range check it uses.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/decode_attn_kernels.hpp"

namespace baloon {
namespace attn_kernels {

namespace {
// 64-lane wave reductions (the BI-V100 wave is 64 wide).
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}
// One wave per row of `cols` floats: out[row] = max (mode 0) or sum (mode 1) of the row.
__global__ void wave_reduce_kernel(const float* __restrict__ x, float* __restrict__ out, int cols, int mode) {
    const float* r = x + (size_t)blockIdx.x * cols;
    float v = mode == 0 ? -INFINITY : 0.f;
    for (int c = threadIdx.x; c < cols; c += 64) v = mode == 0 ? fmaxf(v, r[c]) : v + r[c];
    v = mode == 0 ? warp_max(v) : warp_sum(v);
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}
bool span_overlap(const Span& a, const Span& b) { return a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes; }
Span span_of(const void* p, uint64_t bytes) { return Span{(uint64_t)(uintptr_t)p, bytes}; }
}  // namespace

// ins: read-only buffers; outs: written buffers. Valid iff every span is non-empty, non-null and aligned to `align`, and no
// out span overlaps another out span or any in span (input spans may alias each other).
bool validate_spans(const Span* ins, int n_in, const Span* outs, int n_out, uint64_t align) {
    if (align == 0 || (align & (align - 1))) return false;
    for (int i = 0; i < n_in; ++i)
        if (!ins[i].addr || !ins[i].bytes || (ins[i].addr & (align - 1))) return false;
    for (int i = 0; i < n_out; ++i) {
        if (!outs[i].addr || !outs[i].bytes || (outs[i].addr & (align - 1))) return false;
        for (int j = 0; j < n_in; ++j) if (span_overlap(outs[i], ins[j])) return false;
        for (int j = 0; j < i; ++j) if (span_overlap(outs[i], outs[j])) return false;
    }
    return true;
}

// Row-wise max / sum of rows x [rows][cols] by one 64-lane wave per row (the reductions the flash kernels build on).
// Returns 0 ok, -1 bad arguments, >0 a cudaError_t value.
int flash_wave_reduce(const float* x, float* out, int rows, int cols, int mode, void* stream) {
    if (!x || !out || rows <= 0 || cols <= 0 || (mode != 0 && mode != 1)) return -1;
    wave_reduce_kernel<<<(unsigned)rows, 64, 0, (cudaStream_t)stream>>>(x, out, cols, mode);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

// Returns 0 ok, -1 rejected by host validation, >0 a cudaError_t value.
int native_flash_attn_short_step(const FlashArgs& a, void* stream) {
    if (!a.q || !a.k || !a.v || !a.out || !a.status) return -1;
    if (a.width < 1 || a.width > a.max_ctx || a.width > kFlashMaxCtx) return -1;
    if (a.n_kv != a.width || a.pos != a.width - 1 || a.n_bid != a.width / 4) return -1;
    const uint64_t qb = (uint64_t)kFlashHeads * kHD * 4, kvb = (uint64_t)a.width * kFlashKvHeads * kHD * 2;
    const Span ins[3] = {span_of(a.q, qb), span_of(a.k, kvb), span_of(a.v, kvb)};
    const Span outs[2] = {span_of(a.out, qb), span_of(a.status, 4)};
    if (!validate_spans(ins, 3, outs, 2, 4)) return -1;
    return native_flash_attn_step(a, stream);
}

}  // namespace attn_kernels
}  // namespace baloon
