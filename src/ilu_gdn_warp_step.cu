// ilu_gdn_warp_step.cu - warp-per-(head,column) gated-delta-net step for BI-V100.
// Block (head, 1, z) of 32x4 threads; col = z*4 + ty; lane owns rows r*32+lane for r<4, kept in registers.
#include <cuda_runtime.h>
#include "baloon/ilu_gdn_kernels.hpp"
#include "ilu_gdn_shfl.cuh"

namespace baloon {
namespace ilu_gdn {
namespace {

__global__ void __launch_bounds__(128)
step_kernel(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
            const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
            float* __restrict__ out, int h_k, int h_v, float scale) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    const int qh = head % h_k;
    float s[4], kk[4], qq[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + lane;
        s[r] = state[((size_t)i * h_v + head) * 128 + col];
        kk[r] = k[qh * 128 + i];
        qq[r] = q[qh * 128 + i];
    }
    const float decay = expf(gate[head]);
    float kv = 0.f;
#pragma unroll
    for (int r = 0; r < 4; ++r) { s[r] *= decay; kv += s[r] * kk[r]; }
    kv = lane_reduce(kv);
    const float d = (v[head * 128 + col] - kv) * beta[head];
    float o = 0.f;
#pragma unroll
    for (int r = 0; r < 4; ++r) { s[r] += kk[r] * d; o += s[r] * qq[r]; }
    o = lane_reduce(o);
#pragma unroll
    for (int r = 0; r < 4; ++r) state[((size_t)(r * 32 + lane) * h_v + head) * 128 + col] = s[r];
    if (lane == 0) out[head * 128 + col] = o * scale;
}

}  // namespace

int step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
         float* out, int S, int h_k, int h_v, float scale, void* stream) {
    if (S != native_gdn::kS || h_k <= 0 || h_v <= 0 || h_v % h_k != 0 || h_v > native_gdn::kMaxGridY) return kInvalid;
    const uint64_t hk = (uint64_t)h_k, hv = (uint64_t)h_v;
    const Region rg[] = {
        region(state, 128ull * 128ull * hv * 4ull, true),
        region(out, 128ull * hv * 4ull, true),
        region(q, 128ull * hk * 4ull, false),
        region(k, 128ull * hk * 4ull, false),
        region(v, 128ull * hv * 4ull, false),
        region(gate, hv * 4ull, false),
        region(beta, hv * 4ull, false),
    };
    if (!regions_ok(rg, sizeof(rg) / sizeof(rg[0]))) return kInvalid;
    const dim3 grid((unsigned)h_v, 1, (unsigned)native_gdn::grid_z(S));
    const dim3 block(native_gdn::kLanes, native_gdn::kBlockY, 1);
    step_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(state, q, k, v, gate, beta, out, h_k, h_v, scale);
    return (int)cudaGetLastError();
}

}  // namespace ilu_gdn
}  // namespace baloon
