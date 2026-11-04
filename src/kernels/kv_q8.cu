// kv_q8.cu - int8 KV cache for BI-V100: per-64-value group scale = amax / 127 (float), code = clamp(rint(x / scale), -127, 127),
// value = code * scale. Rows are 256 values = 4 groups; row mapping as in the fp16 cache.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "baloon/qsa_native_kernels.hpp"

namespace baloon {
namespace qsa_dev {
namespace {

constexpr int HD = 256;
constexpr int GROUP = 64;
constexpr int NT = 256;

int check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return kOk;
    std::fprintf(stderr, "kv_q8: %s failed (%d)\n", what, (int)e);
    return (int)e;
}
bool validate(const Shapes& sh) {
    return sh.head_dim == HD && sh.n_head_kv > 0 && sh.page_size > 0 && sh.max_cells > 0;
}

__device__ __forceinline__ float shfl32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ int64_t row_of(const int32_t* table, int n_kv_heads, int h, int page_size, int cell) {
    const int64_t page = table ? table[cell / page_size] : cell / page_size;
    return (page * n_kv_heads + h) * (int64_t)page_size + (cell % page_size);
}

__global__ void __launch_bounds__(NT) kv_append_q8_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                                          int8_t* __restrict__ kpool, int8_t* __restrict__ vpool,
                                                          float* __restrict__ kscale, float* __restrict__ vscale,
                                                          const int32_t* __restrict__ table, const Step* __restrict__ step,
                                                          int n_kv_heads, int page_size) {
    __shared__ float red[2][8];  // [tensor][warp]
    const int h = blockIdx.x, t = threadIdx.x, warp = t >> 5, g = t / GROUP;
    const int64_t row = row_of(table, n_kv_heads, h, page_size, step->pos);
    const float kx = k[h * HD + t], vx = v[h * HD + t];
    float ka = fabsf(kx), va = fabsf(vx);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { ka = fmaxf(ka, shfl32(ka, o)); va = fmaxf(va, shfl32(va, o)); }
    if ((t & 31) == 0) { red[0][warp] = ka; red[1][warp] = va; }
    __syncthreads();
    const float kamax = fmaxf(red[0][2 * g], red[0][2 * g + 1]);  // the two warps of group g
    const float vamax = fmaxf(red[1][2 * g], red[1][2 * g + 1]);
    const float ksc = kamax / 127.f, vsc = vamax / 127.f;
    const float kid = ksc != 0.f ? 1.f / ksc : 0.f, vid = vsc != 0.f ? 1.f / vsc : 0.f;
    float kq = rintf(kx * kid), vq = rintf(vx * vid);
    kq = fminf(127.f, fmaxf(-127.f, kq));
    vq = fminf(127.f, fmaxf(-127.f, vq));
    kpool[row * HD + t] = (int8_t)kq;
    vpool[row * HD + t] = (int8_t)vq;
    if ((t % GROUP) == 0) { kscale[row * 4 + g] = ksc; vscale[row * 4 + g] = vsc; }
}

__global__ void __launch_bounds__(NT) kv_gather_q8_kernel(const int8_t* __restrict__ pool, const float* __restrict__ scale,
                                                          const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                                          const Step* __restrict__ step, int n_kv_heads, int page_size,
                                                          float* __restrict__ out) {
    const int i = blockIdx.x;
    if (i >= step->width) return;
    const int cell = ids[i];
    for (int h = 0; h < n_kv_heads; ++h) {
        const int64_t row = row_of(table, n_kv_heads, h, page_size, cell);
        for (int d = threadIdx.x; d < HD; d += NT)
            out[((int64_t)i * n_kv_heads + h) * HD + d] = (float)pool[row * HD + d] * scale[row * 4 + d / GROUP];
    }
}

}  // namespace

int kv_append_q8_step(const float* k, const float* v, int8_t* kpool, int8_t* vpool, float* kscale, float* vscale,
                      const int32_t* table, const Step* step, const Shapes& sh, void* stream) {
    if (!k || !v || !kpool || !vpool || !kscale || !vscale || !step || !validate(sh)) return kInvalid;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    kv_append_q8_kernel<<<sh.n_head_kv, NT, 0, st>>>(k, v, kpool, vpool, kscale, vscale, table, step, sh.n_head_kv, sh.page_size);
    return check("kv_append_q8_step");
}

int kv_gather_q8_step(const int8_t* pool, const float* scale, const int32_t* table, const int32_t* ids, const Step* step,
                      const Shapes& sh, float* out, void* stream) {
    if (!pool || !scale || !ids || !step || !out || !validate(sh)) return kInvalid;
    const int cap = qsa_step::selection_width(sh.max_cells, sh.top_k, sh.idx_block);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    kv_gather_q8_kernel<<<cap, NT, 0, st>>>(pool, scale, table, ids, step, sh.n_head_kv, sh.page_size, out);
    return check("kv_gather_q8_step");
}

}  // namespace qsa_dev
}  // namespace baloon
