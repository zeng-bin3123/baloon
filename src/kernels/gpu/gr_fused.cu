// gr_fused.cu - fused gated-residual read at the fixed geometry n=2560, hc=4, lr=320 (hc*n = 10240). Same math as gr_read
// (per-stream RMS norm, bf16-rounded operands, down+silu, up+sigmoid gate, stream mean, injection dots) in three launches:
//   down  : DOWN_BLOCKS blocks of 8 hidden rows; every block recomputes the 4 stream RMS values itself (no xn round trip)
//   up    : UP_BLOCKS blocks, each owning 32 columns x 4 streams = 128 rows, so the stream mean closes inside the block
//   inject: one block per stream
// The multi-token variant (T <= 32) uses UPM_BLOCKS blocks of 16 columns and grid.y = token. Weight pointers must be 16-byte aligned.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int N = 2560;
constexpr int HC = 4;
constexpr int D = N * HC;                    // 10240
constexpr int LR = 320;
constexpr int THREADS = 256;
constexpr int WARPS = 8;                     // logical rows per down block (hardware: 4 waves x 2 rows)
constexpr int DOWN_BLOCKS = LR / WARPS;      // 40
constexpr int UP_COLS = 32;
constexpr int UP_BLOCKS = N / UP_COLS;       // 80
constexpr int UPM_COLS = 16;
constexpr int UPM_BLOCKS = N / UPM_COLS;    // 160
constexpr int TILE = N;
constexpr int TQ = TILE / 8 / 32;            // 10 chunks of 8 per lane in the 32-lane model
static_assert(DOWN_BLOCKS * WARPS == LR && UP_BLOCKS * UP_COLS == N && UPM_BLOCKS * UPM_COLS == N, "fused geometry");

__device__ __forceinline__ float sigmoidf_(float x) { return kgc::sigmoid(x); }
__device__ __forceinline__ float warp_sum(float v) { return kgc::wave_sum(v); }

// dot of 8 bf16 weights (one 16-byte chunk) with 8 floats
__device__ __forceinline__ float dot8(const uint4& w, const float* x) {
    const uint32_t u[4] = {w.x, w.y, w.z, w.w};
    float a = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        a += kgc::bf16_to_f((uint16_t)(u[k] & 0xffffu)) * x[2 * k];
        a += kgc::bf16_to_f((uint16_t)(u[k] >> 16)) * x[2 * k + 1];
    }
    return a;
}

// inverse RMS of the four streams, computed by the whole block (sh: 4 floats scratch + HC outputs)
__device__ __forceinline__ void stream_inv(const float* __restrict__ R, float eps, float* inv, float* sh) {
    for (int c = 0; c < HC; ++c) {
        float acc = 0.f;
        for (int d = threadIdx.x; d < N; d += THREADS) acc += R[c * N + d] * R[c * N + d];
        acc = warp_sum(acc);
        if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = acc;
        __syncthreads();
        if (threadIdx.x == 0) inv[c] = rsqrtf((sh[0] + sh[1] + sh[2] + sh[3]) / (float)N + eps);
        __syncthreads();
    }
}

__global__ void __launch_bounds__(256) fused_down_kernel(const float* __restrict__ Rall, const uint16_t* __restrict__ down,
                                                         float* __restrict__ lo_all, float eps) {
    __shared__ float sh[4], inv[HC];
    const float* R = Rall + (size_t)blockIdx.y * D;
    stream_inv(R, eps, inv, sh);
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    for (int rr = wave; rr < WARPS; rr += 4) {
        const int row = blockIdx.x * WARPS + rr;
        const uint4* wv = (const uint4*)(down + (size_t)row * D);
        float acc = 0.f;
        for (int ch = lane; ch < D / 8; ch += 64) {
            float x[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int i = ch * 8 + k;
                x[k] = kgc::bf16_to_f(kgc::f_to_bf16(R[i] * inv[i / N]));
            }
            acc += dot8(wv[ch], x);
        }
        acc = warp_sum(acc);
        if (lane == 0) lo_all[(size_t)blockIdx.y * LR + row] = kgc::silu(acc);
    }
}

template <int COLS>
__global__ void __launch_bounds__(256) fused_up_kernel(const float* __restrict__ Rall, const uint16_t* __restrict__ up,
                                                       const float* __restrict__ lo_all, float* __restrict__ xall, float eps) {
    __shared__ float sh[4], inv[HC], hid[LR], g[HC * COLS];
    const float* R = Rall + (size_t)blockIdx.y * D;
    stream_inv(R, eps, inv, sh);
    for (int k = threadIdx.x; k < LR; k += THREADS) hid[k] = kgc::bf16_to_f(kgc::f_to_bf16(lo_all[(size_t)blockIdx.y * LR + k]));
    __syncthreads();
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    for (int r = wave; r < HC * COLS; r += 4) {
        const int c = r / COLS, col = blockIdx.x * COLS + r % COLS;
        const uint4* wv = (const uint4*)(up + (size_t)(c * N + col) * LR);
        float acc = 0.f;
        if (lane < LR / 8) acc = dot8(wv[lane], hid + lane * 8);
        acc = warp_sum(acc);
        if (lane == 0) g[r] = R[c * N + col] * inv[c] * sigmoidf_(acc);
    }
    __syncthreads();
    if (threadIdx.x < COLS) {
        float s = 0.f;
        for (int c = 0; c < HC; ++c) s += g[c * COLS + threadIdx.x];
        xall[(size_t)blockIdx.y * N + blockIdx.x * COLS + threadIdx.x] = s / (float)HC;
    }
}

__global__ void __launch_bounds__(256) fused_inject_kernel(const uint16_t* __restrict__ inj, const float* __restrict__ xall,
                                                           float* __restrict__ wall) {
    __shared__ float sh[4];
    const float* x = xall + (size_t)blockIdx.y * N;
    float acc = 0.f;
    for (int d = threadIdx.x; d < N; d += THREADS) acc += kgc::bf16_to_f(inj[(size_t)blockIdx.x * N + d]) * x[d];
    acc = warp_sum(acc);
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = acc;
    __syncthreads();
    if (threadIdx.x == 0) wall[(size_t)blockIdx.y * HC + blockIdx.x] = sh[0] + sh[1] + sh[2] + sh[3];
}
}  // namespace

bool fused_gr_supported(int n, int hc, int lr) { return n == N && hc == HC && lr == LR; }

cudaError_t fused_gr_read_multi(const GrMulti& m, cudaStream_t s) {
    if (m.T < 1 || m.T > 32 || !m.R || !m.down || !m.up || !m.inject || !m.lo || !m.x_out || !m.w_out) return cudaErrorInvalidValue;
    if (((uintptr_t)m.down | (uintptr_t)m.up) & 15) return cudaErrorInvalidValue;
    fused_down_kernel<<<dim3(DOWN_BLOCKS, m.T), THREADS, 0, s>>>(m.R, m.down, m.lo, m.eps);
    if (m.T == 1)
        fused_up_kernel<UP_COLS><<<dim3(UP_BLOCKS, 1), THREADS, 0, s>>>(m.R, m.up, m.lo, m.x_out, m.eps);
    else
        fused_up_kernel<UPM_COLS><<<dim3(UPM_BLOCKS, m.T), THREADS, 0, s>>>(m.R, m.up, m.lo, m.x_out, m.eps);
    fused_inject_kernel<<<dim3(HC, m.T), THREADS, 0, s>>>(m.inject, m.x_out, m.w_out);
    return cudaGetLastError();
}
cudaError_t fused_gr_read(const float* R, const uint16_t* down, const uint16_t* up, const uint16_t* inject, float* lo,
                          float* x_out, float* w_out, float eps, cudaStream_t s) {
    GrMulti m{};
    m.R = R; m.down = down; m.up = up; m.inject = inject; m.lo = lo; m.x_out = x_out; m.w_out = w_out; m.eps = eps; m.T = 1;
    return fused_gr_read_multi(m, s);
}
}  // namespace kg
}  // namespace baloon
