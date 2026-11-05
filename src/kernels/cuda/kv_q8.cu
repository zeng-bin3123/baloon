// kv_q8.cu - Q8 KV-cache append and gather on the device. Layout identical to baloon/kv_q8_quantization.hpp: a 64-value
// group stores fp16(amax/127) and int8 codes rint(x*127/amax) clamped to [-127,127]; rows are addressed through a page table.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "baloon/f16_conversion.hpp"
#include "baloon/kv_q8_quantization.hpp"
#include "act_kv_qsa_dev.h"

namespace baloon {
namespace kvq8dev {
namespace {
constexpr int kGroup = kvq8::kGroupSize;  // 64

int check(cudaError_t e) {
    if (e != cudaSuccess) std::fprintf(stderr, "kv_q8: CUDA error '%s'\n", cudaGetErrorString(e));
    return (int)e;
}

// Host-side argument validation: positive geometry, head_dim a multiple of 64, page index inside the table.
bool validate(int64_t pos, int kv_heads, int head_dim, int page_size, int table_len, int64_t n_pages) {
    if (kv_heads <= 0 || head_dim <= 0 || head_dim % kGroup != 0 || page_size <= 0 || table_len <= 0 || n_pages <= 0)
        return false;
    if (pos < 0 || pos / page_size >= table_len) return false;
    return true;
}

// One block (64 threads) per (kv head, group): max-reduce, scale, quantise.
__global__ void kv_append_q8_kernel(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table,
                                    int64_t pos, int kv_heads, int head_dim, int page_size) {
    __shared__ float red[kGroup];
    const int groups = head_dim / kGroup;
    const int h = blockIdx.x / groups, g = blockIdx.x % groups, t = threadIdx.x;
    const float x = src[(size_t)h * head_dim + (size_t)g * kGroup + t];
    red[t] = fabsf(x);
    __syncthreads();
    for (int s = kGroup / 2; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    const float amax = red[0];
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * kv_heads + h) * page_size + pos % page_size;
    const float inv = amax > 0.f ? 127.f / amax : 0.f;
    int q = (int)rintf(x * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    codes[(size_t)row * head_dim + (size_t)g * kGroup + t] = (int8_t)q;
    if (t == 0) scales[(size_t)row * groups + g] = baloon::f16::float_to_half(amax / 127.f);
}

// One thread per (id, kv head, quad of 4 values).
__global__ void kv_gather_q8_kernel(const int8_t* codes, const uint16_t* scales, const int32_t* table,
                                    const int32_t* ids, int n_ids, int kv_heads, int head_dim, int page_size,
                                    float* out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = head_dim / 4, m = (int64_t)kv_heads * per;
    if (i >= (int64_t)n_ids * m) return;
    const int64_t id = i / m, rem = i % m, h = rem / per, quad = rem % per;
    const int32_t cell = ids[id];
    const int groups = head_dim / kGroup;
    float* o = out + ((size_t)id * kv_heads + h) * head_dim + quad * 4;
    if (cell < 0) { o[0] = o[1] = o[2] = o[3] = 0.f; return; }
    const int64_t page = table[cell / page_size];
    const int64_t row = (page * kv_heads + h) * page_size + cell % page_size;
    const int g = (int)(quad * 4) / kGroup;
    const float sc = baloon::f16::half_to_float(scales[(size_t)row * groups + g]);
    const int8_t* c = codes + (size_t)row * head_dim + quad * 4;
    for (int j = 0; j < 4; ++j) o[j] = (float)c[j] * sc;
}
}  // namespace

int kv_append_q8_step(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table, int64_t pos,
                      int kv_heads, int head_dim, int page_size, int table_len, int64_t n_pages) {
    if (!validate(pos, kv_heads, head_dim, page_size, table_len, n_pages) || !src || !codes || !scales || !table)
        return -1;
    kv_append_q8_kernel<<<kv_heads * (head_dim / kGroup), kGroup>>>(src, codes, scales, table, pos, kv_heads,
                                                                    head_dim, page_size);
    if (int e = check(cudaGetLastError())) return e;
    return check(cudaDeviceSynchronize());
}

int kv_gather_q8_step(const int8_t* codes, const uint16_t* scales, const int32_t* table, const int32_t* ids,
                      int n_ids, int kv_heads, int head_dim, int page_size, int table_len, int64_t n_pages,
                      float* out) {
    if (n_ids < 0 || !validate(0, kv_heads, head_dim, page_size, table_len, n_pages) || !codes || !scales || !table ||
        !ids || !out)
        return -1;
    if (n_ids == 0) return 0;
    const int64_t total = (int64_t)n_ids * kv_heads * (head_dim / 4);
    kv_gather_q8_kernel<<<(unsigned)((total + 255) / 256), 256>>>(codes, scales, table, ids, n_ids, kv_heads,
                                                                  head_dim, page_size, out);
    if (int e = check(cudaGetLastError())) return e;
    return check(cudaDeviceSynchronize());
}
}  // namespace kvq8dev
}  // namespace baloon
