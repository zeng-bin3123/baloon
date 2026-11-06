// native_flash_attn.cu - short-context (<= 256 cells) flash-attention decode kernel specialised to Q 24x256, KV 2x256 fp16,
// one block per query head, 128 threads, online softmax with a streaming maximum over two 128-cell passes.
// K/V buffers are [cell][2 kv heads][256] fp16. Query head h reads kv head h / 12. An unsupported step writes a NaN output
// and *status = 1 (valid: 1 <= width <= max_ctx, n_kv == width, pos == width-1, n_bid == width/4).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/decode_attn_kernels.hpp"

namespace baloon {
namespace attn_kernels {
namespace {

__global__ void __launch_bounds__(128)
flash_kernel(const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
             float* __restrict__ out, int32_t* status, int width, int nkv, int pos, int nbid, int max_ctx) {
    __shared__ float sq[256], red[128], es[128];
    const int head = blockIdx.x, tid = threadIdx.x;
    const bool valid = width >= 1 && width <= max_ctx && width <= kFlashMaxCtx && nkv == width && pos == width - 1 &&
                       nbid == width / 4;
    if (!valid) {
        out[head * 256 + tid] = __int_as_float(0x7fc00000);
        out[head * 256 + tid + 128] = __int_as_float(0x7fc00000);
        if (head == 0 && tid == 0) *status = 1;
        return;
    }
    if (head == 0 && tid == 0) *status = 0;
    const int kvh = head / (kFlashHeads / kFlashKvHeads);
    sq[tid] = q[head * 256 + tid] * 0.0625f;
    sq[tid + 128] = q[head * 256 + tid + 128] * 0.0625f;
    __syncthreads();

    float m = -INFINITY, l = 0.f, acc0 = 0.f, acc1 = 0.f;
    for (int pass = 0; pass < 2; ++pass) {
        const int cell = pass * 128 + tid;
        float s = -INFINITY;
        if (cell < width) {
            const __half* kp = k + ((size_t)cell * 2 + kvh) * 256;
            float d = 0.f;
            for (int i = 0; i < 256; ++i) d += sq[i] * __half2float(kp[i]);
            s = d;
        }
        red[tid] = s;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
            __syncthreads();
        }
        const float mnew = fmaxf(m, red[0]);
        __syncthreads();
        const float rescale = expf(m - mnew);   // m - mnew <= 0
        const float e = (cell < width) ? expf(s - mnew) : 0.f;
        es[tid] = e;
        red[tid] = e;
        __syncthreads();
        for (int off = 64; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        l = l * rescale + red[0];
        acc0 *= rescale;
        acc1 *= rescale;
        for (int c = 0; c < 128; ++c) {
            if (pass * 128 + c >= width) break;
            const __half* vp = v + ((size_t)(pass * 128 + c) * 2 + kvh) * 256;
            acc0 += es[c] * __half2float(vp[tid]);
            acc1 += es[c] * __half2float(vp[tid + 128]);
        }
        m = mnew;
        __syncthreads();
    }
    out[head * 256 + tid] = acc0 / l;
    out[head * 256 + tid + 128] = acc1 / l;
}

}  // namespace

int native_flash_attn_step(const FlashArgs& a, void* stream) {
    if (!a.q || !a.k || !a.v || !a.out || !a.status) return -1;
    flash_kernel<<<kFlashHeads, kFlashThreads, 0, (cudaStream_t)stream>>>(
        a.q, static_cast<const __half*>(a.k), static_cast<const __half*>(a.v), a.out, a.status, a.width, a.n_kv, a.pos,
        a.n_bid, a.max_ctx);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int)e;
}

}  // namespace attn_kernels
}  // namespace baloon
