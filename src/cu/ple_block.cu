// ple_block.cu - GPU half of the PLE block (geometry: PleBlockGeometry theory).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "baloon/kern_geometry.hpp"
#include "baloon/kern_ple.hpp"
#include "ix_compat.h"
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace kern {

namespace {

constexpr int THREADS = 256;
constexpr uint32_t NH = plegeo::NG_HIST, NE = plegeo::NG_EMBD, NHC = plegeo::NG_HC, HCD = plegeo::NG_HC_DIM;
bool native_bf16 = false;
bool native_postops = false;

__device__ __forceinline__ uint16_t bf16_bits(float f) { return f16bits::round_f32_to_bf16_bits(f); }
__device__ __forceinline__ float bf16_float(uint16_t h) { return f16bits::bf16_bits_to_f32(h); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }

// Block-wide sum (shared-memory halving tree; correct for any wavefront size). All threads get the total.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

// Depthwise causal dilated convolution: tap k reads row tapRow(9, 4, 3, k) = 0, 3, 6, 9 of [history | new]; only the last
// tap reads the new row. History is row-fastest: row + 9 * channel.
__global__ void conv_kernel(const float* __restrict__ w, const float* __restrict__ hist, const float* __restrict__ new_row, float* __restrict__ out) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= HCD) return;
    float acc = 0.f;
    for (uint32_t k = 0; k < plegeo::CONV_KERNEL; ++k) {
        const uint32_t row = plegeo::tap_row(NH, plegeo::CONV_KERNEL, plegeo::CONV_DIL, k);
        const float v = row == NH ? new_row[c] : hist[plegeo::state_index(row, c)];
        acc += w[plegeo::kernel_weight_index(k, c)] * v;
    }
    out[c] = acc;
}

// Per-stream RMS norm: one block per stream.
__global__ void gnorm_kernel(const float* __restrict__ x, const float* __restrict__ nw, float* __restrict__ out) {
    __shared__ float sh[THREADS];
    const uint32_t s = blockIdx.x;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < NE; i += blockDim.x) { const float v = x[s * NE + i]; ss += v * v; }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / (float)NE + 1e-6f);
    for (uint32_t i = threadIdx.x; i < NE; i += blockDim.x) out[s * NE + i] = x[s * NE + i] * r * nw[i];
}

// gate[s] = sigmoid(dot(gate_w[s], hidden)); one block per stream.
__global__ void gate_kernel(const uint16_t* __restrict__ gw, const float* __restrict__ hidden, float* __restrict__ gate) {
    __shared__ float sh[THREADS];
    const uint32_t s = blockIdx.x;
    float acc = 0.f;
    for (uint32_t j = threadIdx.x; j < NE; j += blockDim.x) acc += bf16_float(gw[(size_t)s * NE + j]) * hidden[j];
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) gate[s] = 1.f / (1.f + expf(-acc));
}

// Broadcast: element i < hc_dim reads hidden[i % n_embd] and gate[i / n_embd].
__global__ void bcast_kernel(const float* __restrict__ hidden, const float* __restrict__ gate, float* __restrict__ out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HCD) return;
    out[i] = hidden[i % NE] * gate[i / NE];
}

__global__ void add3_kernel(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, float* __restrict__ out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (a[i] + b[i]) + c[i];
}

// r[j] = silu(sum over streams of x[s * n_embd + j]); writes the float vector and its bf16 copy.
__global__ void to_bf16_kernel(const float* __restrict__ x, float* __restrict__ r, uint16_t* __restrict__ rb) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NE) return;
    float s = 0.f;
    for (uint32_t k = 0; k < NHC; ++k) s += x[k * NE + j];
    const float v = silu_f(s);
    r[j] = v;
    rb[j] = bf16_bits(v);
}

// out[o] = [hidden[o] +] sum_j bf16(proj[o][j]) * act[j]; one block per output row.
__global__ void bf16_gemv_kernel(const uint16_t* __restrict__ pw, const float* __restrict__ act, const uint16_t* __restrict__ act_bf, int use_bf,
                                 const float* __restrict__ hidden, int add_hidden, float* __restrict__ out) {
    __shared__ float sh[THREADS];
    const uint32_t o = blockIdx.x;
    float acc = 0.f;
    for (uint32_t j = threadIdx.x; j < NE; j += blockDim.x) {
        const float a = use_bf ? bf16_float(act_bf[j]) : act[j];
        acc += bf16_float(pw[(size_t)o * NE + j]) * a;
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) out[o] = add_hidden ? hidden[o] + acc : acc;
}

__global__ void history_advance_kernel(float* __restrict__ hist, const float* __restrict__ new_row) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= HCD) return;
    float* col = hist + plegeo::state_index(0, c);
    for (uint32_t r = 0; r + 1 < NH; ++r) col[r] = col[r + 1];
    col[NH - 1] = new_row[c];
}

cudaError_t ck(cudaError_t e) { return e; }

bool overlap(const void* a, size_t ab, const void* b, size_t bb) {
    return plegeo::overlap((uint64_t)(uintptr_t)a, ab, (uint64_t)(uintptr_t)b, bb);
}

}  // namespace

bool ple_block_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

size_t ple_block_scratch_bytes() { return (size_t)plegeo::scratch_bytes(); }

void ple_set_native_bf16(bool on) { native_bf16 = on; }
void ple_set_native_postops(bool on) { native_postops = on; }
bool ple_native_postops_enabled() { return native_postops; }

cudaError_t ple_history_advance(float* history, const float* new_row, cudaStream_t s) {
    if (history == nullptr || new_row == nullptr) return cudaErrorInvalidValue;
    if (overlap(history, (size_t)NH * HCD * 4, new_row, (size_t)HCD * 4)) return cudaErrorInvalidValue;
    history_advance_kernel<<<(HCD + THREADS - 1) / THREADS, THREADS, 0, s>>>(history, new_row);
    return ck(cudaGetLastError());
}

cudaError_t ple_block(const PleWeights& w, const float* new_row, const float* history, const float* hidden, const PleOut& out, void* scratch, cudaStream_t s) {
    if (!w.conv_w || !w.norm_w || !w.gate_w || !w.proj_w || !new_row || !history || !hidden || !out.out || !scratch) return cudaErrorInvalidValue;
    const size_t sb = ple_block_scratch_bytes();
    if (overlap(scratch, sb, new_row, (size_t)HCD * 4) || overlap(scratch, sb, history, (size_t)NH * HCD * 4) || overlap(scratch, sb, hidden, (size_t)NE * 4) ||
        overlap(scratch, sb, out.out, (size_t)NE * 4) || (out.gate && overlap(scratch, sb, out.gate, (size_t)NHC * 4)))
        return cudaErrorInvalidValue;
    if (overlap(out.out, (size_t)NE * 4, hidden, (size_t)NE * 4) && out.out != hidden) return cudaErrorInvalidValue;   // exact alias is fine
    // carve: float workspace [conv | nrm | bcast | sum3 | stream-sum | r | gate], activation image, bf16 copy, slack
    float* f = static_cast<float*>(scratch);
    float* conv = f;
    float* nrm = f + HCD;
    float* bc = f + 2 * HCD;
    float* s3 = f + 3 * HCD;
    float* tmp = f + 4 * HCD;           // reserved stream workspace (5th hc_dim region)
    float* rvec = f + 5 * HCD;
    float* gate = f + 5 * HCD + NE;     // hc floats, the end of the float workspace
    (void)tmp;
    uint8_t* base = static_cast<uint8_t*>(scratch);
    const size_t f_bytes = plegeo::r16(plegeo::float_ws_floats() * 4);
    const size_t q_bytes = plegeo::r16((uint64_t)NE / 32 * 34);
    uint16_t* rbf = reinterpret_cast<uint16_t*>(base + f_bytes + q_bytes);   // bf16 copy follows the (reserved) Q8_0 image
    const unsigned gb = (HCD + THREADS - 1) / THREADS;
    conv_kernel<<<gb, THREADS, 0, s>>>(w.conv_w, history, new_row, conv);
    gnorm_kernel<<<NHC, THREADS, 0, s>>>(conv, w.norm_w, nrm);
    gate_kernel<<<NHC, THREADS, 0, s>>>(w.gate_w, hidden, gate);
    bcast_kernel<<<gb, THREADS, 0, s>>>(hidden, gate, bc);
    add3_kernel<<<gb, THREADS, 0, s>>>(bc, nrm, new_row, s3, HCD);
    to_bf16_kernel<<<(NE + THREADS - 1) / THREADS, THREADS, 0, s>>>(s3, rvec, rbf);
    bf16_gemv_kernel<<<NE, THREADS, 0, s>>>(w.proj_w, rvec, rbf, native_bf16 ? 1 : 0, hidden, native_postops ? 1 : 0, out.out);
    if (out.gate) cudaMemcpyAsync(out.gate, gate, NHC * sizeof(float), cudaMemcpyDeviceToDevice, s);
    return ck(cudaGetLastError());
}

}  // namespace kern
}  // namespace baloon
