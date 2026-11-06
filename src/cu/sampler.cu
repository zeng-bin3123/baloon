// sampler.cu - GPU token sampler: repeat / presence / frequency penalties over the tail history window, greedy argmax
// (ties -> lowest index), and top-k / top-p / temperature sampling with a Philox4x32-10 uniform.
// One CUDA block per row. The (value, index) combine is the shuffle-reduction rule of the router: the challenger wins
// iff larger, or equal with a smaller index; it is commutative and associative, so any tree gives the same winner.
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "baloon/kern_cuda_misc.hpp"

namespace baloon {
namespace kern {

namespace {

constexpr int SAMPLER_THREADS = 256;

struct Cand { float v; int i; };
__device__ __forceinline__ Cand comb(Cand a, Cand b) {
    return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

// Occurrences of `tok` in the history window [start, len) of one row.
__device__ __forceinline__ int history_count(const int32_t* hist, int start, int len, int tok) {
    int c = 0;
    for (int k = start; k < len; ++k) c += (hist[k] == tok);
    return c;
}

// repeat penalty (multiply non-positive logits, divide positive ones) when the token occurred, then the
// frequency term count * f and the presence term (a boolean, not the count) are subtracted.
__device__ __forceinline__ float apply_penalties(float l, int count, float repeat, float presence, float freq) {
    if (count > 0) l = l <= 0.f ? l * repeat : l / repeat;
    return l - ((float)count * freq + (count > 0 ? presence : 0.f));
}

// ---- Philox4x32-10 -------------------------------------------------------------------------------------------------
__device__ __forceinline__ void philox4x32_round(uint32_t c[4], const uint32_t k[2]) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c[0]), lo0 = 0xD2511F53u * c[0];
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c[2]), lo1 = 0xCD9E8D57u * c[2];
    const uint32_t n0 = hi1 ^ c[1] ^ k[0], n2 = hi0 ^ c[3] ^ k[1];
    c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
}
// Uniform in [0, 1): the 24-bit mantissa draw (c0 >> 8) * 2^-24.
__device__ __forceinline__ float philox_uniform(uint64_t seed, uint32_t row, uint64_t counter) {
    uint32_t c[4] = {row, (uint32_t)counter, (uint32_t)(counter >> 32), 0u};
    uint32_t k[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
    for (int r = 0; r < 10; ++r) {
        philox4x32_round(c, k);
        k[0] += 0x9E3779B9u;
        k[1] += 0xBB67AE85u;
    }
    return (float)(c[0] >> 8) * (1.0f / 16777216.0f);
}

// Penalized copy of the row's logits into ws (NaN becomes -inf).
__device__ __forceinline__ void penalize_row(const float* __restrict__ logits, float* __restrict__ ws, int n_vocab, const int32_t* hist, int hstart, int hlen,
                                             float repeat, float presence, float freq) {
    const bool pen = hlen > hstart && (repeat != 1.f || presence != 0.f || freq != 0.f);
    for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) {
        float l = logits[i];
        if (!(l == l)) l = -INFINITY;
        if (pen) l = apply_penalties(l, history_count(hist, hstart, hlen, i), repeat, presence, freq);
        ws[i] = l;
    }
}

// Block-wide argmax of ws (ties -> lowest index). All threads get the winner.
__device__ __forceinline__ Cand block_argmax(const float* ws, int n_vocab, Cand* sh) {
    Cand best{-INFINITY, n_vocab};
    for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) best = comb(best, Cand{ws[i], i});
    sh[threadIdx.x] = best;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = comb(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const Cand r = sh[0];
    __syncthreads();
    return r;
}

__global__ void sampler_greedy_kernel(const float* __restrict__ logits, float* __restrict__ ws, int n_vocab, const int32_t* history, int hist_stride,
                                      const int32_t* hist_len, int last_n, float repeat, float presence, float freq, int32_t* __restrict__ out) {
    __shared__ Cand sh[SAMPLER_THREADS];
    const int row = blockIdx.x;
    const float* lg = logits + (size_t)row * n_vocab;
    float* w = ws + (size_t)row * n_vocab;
    const int len = hist_len ? hist_len[row] : 0;
    const int hl = last_n > 0 ? (len < last_n ? len : last_n) : 0;
    penalize_row(lg, w, n_vocab, history ? history + (size_t)row * hist_stride : nullptr, len - hl, len, repeat, presence, freq);
    __syncthreads();
    const Cand b = block_argmax(w, n_vocab, sh);
    if (threadIdx.x == 0) out[row] = b.i < n_vocab ? b.i : 0;   // every candidate -inf: token 0
}

__global__ void sampler_kernel(const float* __restrict__ logits, float* __restrict__ ws, int n_vocab, const int32_t* history, int hist_stride,
                               const int32_t* hist_len, int last_n, float repeat, float presence, float freq, float temperature, int top_k,
                               float top_p, int min_keep, uint64_t seed, uint64_t counter, int32_t* __restrict__ out) {
    __shared__ Cand sh[SAMPLER_THREADS];
    __shared__ int keep_ids[kSamplerMaxKeep];
    __shared__ float keep_logit[kSamplerMaxKeep];
    __shared__ int n_keep;
    const int row = blockIdx.x;
    const float* lg = logits + (size_t)row * n_vocab;
    float* w = ws + (size_t)row * n_vocab;
    const int len = hist_len ? hist_len[row] : 0;
    const int hl = last_n > 0 ? (len < last_n ? len : last_n) : 0;
    penalize_row(lg, w, n_vocab, history ? history + (size_t)row * hist_stride : nullptr, len - hl, len, repeat, presence, freq);
    __syncthreads();
    const int kk = (top_k > 0 && top_k < kSamplerMaxKeep) ? top_k : kSamplerMaxKeep;
    if (threadIdx.x == 0) n_keep = 0;
    __syncthreads();
    for (int j = 0; j < kk; ++j) {   // survivors in descending order (stable: lowest index first on ties)
        const Cand b = block_argmax(w, n_vocab, sh);
        if (!(b.v > -INFINITY) || b.i >= n_vocab) break;
        if (threadIdx.x == 0) { keep_ids[j] = b.i; keep_logit[j] = b.v; n_keep = j + 1; w[b.i] = -INFINITY; }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    const int nk = n_keep;
    if (nk == 0) { out[row] = 0; return; }
    float wt[kSamplerMaxKeep];
    float z = 0.f;
    for (int j = 0; j < nk; ++j) { wt[j] = expf((keep_logit[j] - keep_logit[0]) / temperature); z += wt[j]; }
    int cut = nk;
    if (top_p < 1.f) {
        float cum = 0.f;
        cut = nk;
        for (int j = 0; j < nk; ++j) { cum += wt[j] / z; if (cum >= top_p) { cut = j + 1; break; } }
    }
    if (cut < min_keep) cut = min_keep < nk ? min_keep : nk;   // top-p cut adjustment
    if (cut < 1) cut = 1;
    float zc = 0.f;
    for (int j = 0; j < cut; ++j) zc += wt[j];
    const float target = philox_uniform(seed, (uint32_t)row, counter) * zc;
    float cum = 0.f;
    int pick = cut - 1;
    for (int j = 0; j < cut; ++j) { cum += wt[j]; if (cum > target) { pick = j; break; } }
    out[row] = keep_ids[pick];
}

}  // namespace

cudaError_t sample_tokens(const float* logits, int n_rows, int n_vocab, const int32_t* history, int hist_stride, const int32_t* hist_len,
                          const SamplerParams& p, uint64_t seed, uint64_t counter, float* workspace, int32_t* out_tokens, cudaStream_t stream) {
    if (logits == nullptr || workspace == nullptr || out_tokens == nullptr || n_rows <= 0 || n_vocab <= 0) return cudaErrorInvalidValue;
    if (p.repeat_penalty <= 0.f) return cudaErrorInvalidValue;
    if (p.temperature <= 0.f) {
        sampler_greedy_kernel<<<n_rows, SAMPLER_THREADS, 0, stream>>>(logits, workspace, n_vocab, history, hist_stride, hist_len, p.last_n,
                                                                      p.repeat_penalty, p.presence_penalty, p.frequency_penalty, out_tokens);
    } else {
        sampler_kernel<<<n_rows, SAMPLER_THREADS, 0, stream>>>(logits, workspace, n_vocab, history, hist_stride, hist_len, p.last_n, p.repeat_penalty,
                                                               p.presence_penalty, p.frequency_penalty, p.temperature, p.top_k, p.top_p, p.min_keep,
                                                               seed, counter, out_tokens);
    }
    return cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon
