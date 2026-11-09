// ilu_native_gdn.hpp - host API of the BI-V100 native gated-delta-net step and its preprocessing kernels.
// Pure C++ (no CUDA headers); implementations: src/kernels/cuda/ilu_native_gdn.cu (step) and
// src/kernels/cuda/ilu_native_gdn_pre.cu (conv / norms / gates).
// Return codes: 0 ok, -1 rejected by host validation (nothing launched, no memory touched), >0 a cudaError_t value.
// Streams must be non-null. Every writable span must be disjoint from every other span (read spans may alias).
// A span is valid iff its address is non-null, 4-aligned and addr + bytes does not wrap the address space.
#pragma once
#include <cstdint>

namespace baloon {
namespace ilu_nat {

constexpr int kReject = -1;

// ---- step (src/kernels/cuda/ilu_native_gdn.cu) ----
namespace step_impl {
bool valid_span(uint64_t addr, uint64_t bytes);                       // nonnull, 4-aligned, no wrap
bool overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn);       // a < b+bn && b < a+an (no emptiness guard)
}  // namespace step_impl

bool native_gdn_enabled();
void native_gdn_set_enabled(bool on);

// state [128][h_v][128] updated in place; q,k [h_k][128]; v,o [h_v][128]; gate,beta [h_v]. S must be 128,
// h_v % h_k == 0, h_v <= 65535. Per (head, col): s_i *= exp(gate); d = (v - sum_i s_i k_i) * beta;
// s_i += k_i d; o = sum_i s_i q_i / sqrt(128).  q/k head of value head h is h % h_k.
int native_gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                    float* o, int S, int h_k, int h_v, void* stream);

// ---- preprocessing (src/kernels/cuda/ilu_native_gdn_pre.cu) ----
namespace pre_impl {
bool valid(uint64_t addr, uint64_t bytes);                                   // nonnull, 4-aligned, no wrap
bool disjoint(uint64_t a, uint64_t an, uint64_t b, uint64_t bn);             // intervals do not intersect
bool count_and_stream(int count, void* stream);    // count in [1, 65535] and stream non-null
bool norm_geometry(int rows, int width, int threads, unsigned* grid);   // width 128, threads 256, rows ok
}  // namespace pre_impl

// hist[c*3..c*3+2]=[h0,h1,h2] becomes [h1,h2,x]; acc = h0*w0+h1*w1+h2*w2+x*w3 (+0 bias); out = acc*sigmoid(acc).
int native_gdn_conv_silu(float* hist, const float* x, const float* w, float* out, int channels, void* stream);
// out[r][c] = in[r][c] * rsqrt(sum_c in^2 + eps) * scale, rows of width 128 (out disjoint from in).
int native_gdn_l2_norm(const float* in, float* out, int rows, float eps, float scale, void* stream);
// out = sigmoid(raw).
int native_gdn_beta_gate(const float* raw, float* out, int n, void* stream);
// out = softplus(alpha + bias) * a_neg, softplus(v) = v for v > 20, else log1p(exp(v)).
int native_gdn_gate(const float* alpha, const float* bias, const float* a_neg, float* out, int n, void* stream);
// y[r][c] = o * rsqrt(mean_c o^2 + eps) * gamma[c] * sigmoid(z[r][c]); rows of width 128; gamma has 128 floats.
int native_gdn_out_norm(const float* o, const float* z, const float* gamma, float* y, int rows, float eps, void* stream);

}  // namespace ilu_nat
}  // namespace baloon
