// ilu_gdn_kernels.hpp - host API of the BI-V100 gated-delta-net kernels (warp-per-column step + preprocessing).
// Pure C++ (no CUDA headers) so it compiles standalone; the implementations live in src/ilu_gdn_*.cu.
// Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
// All shuffles use 32-lane segments, so the result does not depend on the hardware wave width (64 on ivcore10).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "baloon/native_gdn_step.hpp"
#include "baloon/native_gdn_preprocess.hpp"

namespace baloon {
namespace ilu_gdn {

constexpr int kOk = 0;
constexpr int kInvalid = -1;
constexpr float kSoftplusThreshold = 20.f;

// One span of device memory touched by a launch; `write` spans are outputs.
struct Region { uint64_t addr; uint64_t bytes; bool write; };

inline Region region(const void* p, uint64_t bytes, bool write) {
    return Region{(uint64_t)(uintptr_t)p, bytes, write};
}

// Host validation shared by every launch: non-null, no address wrap, 4-byte aligned, and no write region
// overlapping any other region (read regions may alias each other). Zero-byte regions are not exempt.
inline bool regions_ok(const Region* r, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (r[i].addr == 0) return false;
        const native_gdn::Span s{r[i].addr, r[i].bytes};
        if (!native_gdn::span_valid(s) || !native_gdn::span_aligned4(s)) return false;
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if ((r[i].write || r[j].write) && native_gdn::overlap(r[i].addr, r[i].bytes, r[j].addr, r[j].bytes))
                return false;
    return true;
}

// Overflow-safe softplus: above 20 the input is returned unchanged.
inline float softplus_sel(float v) { return v > kSoftplusThreshold ? v : std::log1p(std::exp(v)); }

// Warp-per-(head,column) recurrence step. S must be 128, h_v % h_k == 0, h_v <= 65535.
//   state: [128][h_v][128] floats, updated in place;  q,k: [h_k][128];  v,out: [h_v][128];  gate,beta: [h_v].
//   per (head, col): s_i *= exp(gate); d = (v - sum_i s_i k_i) * beta; s_i += k_i d; out = scale * sum_i s_i q_i.
int step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
         float* out, int S, int h_k, int h_v, float scale, void* stream);

// 4-tap conv + SiLU with a 3-deep history; hist[c*3..c*3+2] = [h0,h1,h2] becomes [h1,h2,x]; w[c*4..c*4+3].
int conv_silu(float* hist, const float* x, const float* w, float* out, int channels, void* stream);

// In-place row norm over rows of width 128. rms=false: x*rsqrt(sum x^2 + eps); rms=true: x*rsqrt(sum x^2/128 + eps).
int row_norm(float* x, int rows, float eps, bool rms, void* stream);

// out = sigmoid(raw).
int beta_sigmoid(const float* raw, float* out, int n, void* stream);

// out = softplus_sel(alpha + bias) * a_neg.
int gate_from_alpha(const float* alpha, const float* bias, const float* a_neg, float* out, int n, void* stream);

// y[r][c] = o[r][c] * rsqrt(mean_c o^2 + eps) * w[c] * sigmoid(z[r][c]); rows of width 128.
int out_rms_gate(const float* o, const float* z, const float* w, float* y, int rows, float eps, void* stream);

}  // namespace ilu_gdn
}  // namespace baloon
