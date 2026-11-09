// ucc7_geometry.cu - one-thread-per-item device kernels that evaluate the closed-form facts proved in the Lean modules
// QsaCacheIndexAttend / QsaBlockSelect / NativeFlashAttnShort, so the device arithmetic can be compared with the host model.
#include <cuda_runtime.h>
#include <cstdint>
#include "baloon/ucc7_geometry.hpp"

namespace baloon {
namespace ucc7 {
namespace {

constexpr int kThreads = 256;

__global__ void geometry_kernel(const int32_t* __restrict__ pos, int n, int top_k, int r, int32_t* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = pos[i], nkv = p + 1, nbid = nkv / r, cap = top_k + r - 1;
    const int tail = nkv - r * nbid;
    int32_t* o = out + (size_t)i * kStepFields;
    o[0] = nkv; o[1] = nbid; o[2] = nkv < cap ? nkv : cap; o[3] = tail; o[4] = tail != 0; o[5] = (p % 24) / 12;
}

__global__ void order_key_kernel(const uint32_t* __restrict__ bits, int n, uint32_t* __restrict__ keys) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t b = bits[i];
    keys[i] = b < 0x80000000u ? b + 0x80000000u : 0xFFFFFFFFu - b;
}

__global__ void quota_kernel(const int32_t* __restrict__ budget, const int32_t* __restrict__ before, const int32_t* __restrict__ eq,
                             int n, int32_t* __restrict__ quota) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int q = budget[i] - before[i];
    q = q < 0 ? 0 : q;
    quota[i] = q < eq[i] ? q : eq[i];
}

inline unsigned grid_for(int n) { return (unsigned)((n + kThreads - 1) / kThreads); }

}  // namespace

int geometry_eval(const int32_t* pos, int n, int top_k, int r, int32_t* out, void* stream) {
    if (!pos || !out || n <= 0 || top_k < 1 || r < 1) return -1;
    geometry_kernel<<<grid_for(n), kThreads, 0, (cudaStream_t)stream>>>(pos, n, top_k, r, out);
    return (int)cudaGetLastError();
}
int order_key_eval(const uint32_t* bits, int n, uint32_t* keys, void* stream) {
    if (!bits || !keys || n <= 0) return -1;
    order_key_kernel<<<grid_for(n), kThreads, 0, (cudaStream_t)stream>>>(bits, n, keys);
    return (int)cudaGetLastError();
}
int quota_eval(const int32_t* budget, const int32_t* before, const int32_t* eq, int n, int32_t* quota, void* stream) {
    if (!budget || !before || !eq || !quota || n <= 0) return -1;
    quota_kernel<<<grid_for(n), kThreads, 0, (cudaStream_t)stream>>>(budget, before, eq, n, quota);
    return (int)cudaGetLastError();
}

}  // namespace ucc7
}  // namespace baloon
