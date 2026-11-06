// prefill_kernels.cu - device-side index decode, routing lane coverage, ternary expert dequantisation and INT8 KV
// quantisation of the prefill path. Launch grids are blocks_for(n) blocks of 256 threads; every kernel guards i < n.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/prefill_device_api.hpp"
#include "baloon/prefill_kernel_geometry.hpp"

namespace baloon {
namespace pfk_dev {
namespace {

__global__ void hc_decode_kernel(int64_t n, int32_t* out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const pfk::Thc p = pfk::hc_decode(i);
    out[i] = (int32_t)(p.t * pfk::kHcD + (int64_t)p.c * pfk::kHcN + p.d);
}

// One block of 32 logical lanes (threadIdx.x < 32; surplus hardware lanes idle) - independent of wave width.
__global__ void route_cover_kernel(int32_t* counts) {
    const int lane = threadIdx.x;
    if (lane >= pfk::kLanes) return;
    for (int i = 0; i < pfk::kPerLane; ++i) atomicAdd(&counts[pfk::lane_expert(lane, i)], 1);
}

__global__ void expert_dequant_kernel(const uint8_t* blob, float* gu_out, float* d_out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pfk::kExpertThreads) return;
    const uint8_t* gu_sc = blob + pfk::O_GU_SC;
    const uint8_t* d_sc = blob + pfk::O_D_SC;
    if (i < pfk::kGuRows * pfk::kGuBytes) {
        const int64_t row = i / pfk::kGuBytes, byte = i % pfk::kGuBytes;
        const uint8_t c = blob[pfk::O_GU_CODES + i];
        const int64_t si = pfk::gu_scale_idx(row, byte);
        const float s = f16::half_to_float((uint16_t)(gu_sc[2 * si] | (gu_sc[2 * si + 1] << 8)));
        for (int k = 0; k < 4; ++k) gu_out[pfk::gu_out_idx(row, byte, k)] = s * (float)(pfk::code_field(c, k) - 1);
    } else {
        const int64_t j = i - pfk::kGuRows * pfk::kGuBytes;
        const int64_t row = j / pfk::kDBytes, byte = j % pfk::kDBytes;
        const uint8_t c = blob[pfk::O_D_CODES + j];
        const int64_t si = pfk::d_scale_idx(row, byte);
        const float s = f16::half_to_float((uint16_t)(d_sc[2 * si] | (d_sc[2 * si + 1] << 8)));
        for (int k = 0; k < 4; ++k) d_out[pfk::d_out_idx(row, byte, k)] = s * (float)(pfk::code_field(c, k) - 1);
    }
}

__global__ void kv_quant_kernel(const float* x, int8_t* q, int64_t n, float inv) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q[i] = (int8_t)pfk::clamp_q((int64_t)rintf(x[i] * inv));
}

int finish() {
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

}  // namespace

int launch_hc_decode(int T, int32_t* out) {
    if (T <= 0 || !out) return -1;
    const int64_t n = (int64_t)T * pfk::kHcD;
    hc_decode_kernel<<<(unsigned)pfk::blocks_for(n), pfk::kThreads>>>(n, out);
    return finish();
}
int launch_route_cover(int32_t* counts) {
    if (!counts) return -1;
    route_cover_kernel<<<1, 64>>>(counts);
    return finish();
}
int launch_expert_dequant(const uint8_t* blob, float* gu_out, float* d_out) {
    if (!blob || !gu_out || !d_out) return -1;
    expert_dequant_kernel<<<(unsigned)pfk::blocks_for(pfk::kExpertThreads), pfk::kThreads>>>(blob, gu_out, d_out);
    return finish();
}
int launch_kv_quant(const float* x, int8_t* q, int64_t n, float inv) {
    if (!x || !q || n <= 0) return -1;
    kv_quant_kernel<<<(unsigned)pfk::blocks_for(n), pfk::kThreads>>>(x, q, n, inv);
    return finish();
}

}  // namespace pfk_dev
}  // namespace baloon
