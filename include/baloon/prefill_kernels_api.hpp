// prefill_kernels_api.hpp - host entry points of the batched prompt-path CUDA kernels (src/prefill/kernels.cu).
// All pointers are device pointers; every wrapper launches on the default stream and returns 0 on success, a CUDA error
// code otherwise, and -1 on invalid arguments (nothing launched).
#pragma once
#include <cstdint>

namespace baloon {
namespace pk {

#ifndef BALOON_PK_GEOMETRY_IN_TU
constexpr int N = 2560;
#endif
constexpr int HC = 4;
constexpr int D = N * HC;
constexpr int LR = 320;
#ifndef BALOON_PK_GEOMETRY_IN_TU
constexpr int S = 128;
#endif
constexpr int HK = 16;
constexpr int HV = 48;
constexpr int C = 10240;
#ifndef BALOON_PK_GEOMETRY_IN_TU
constexpr int RG = 4;
#endif
constexpr int kRoutedK = 10, kExperts = 512;
constexpr int64_t kBlobBytes = 1382400;
constexpr int64_t kGuOff = 0, kDownOff = 1280 * 640, kGuScaleOff = kDownOff + 2560 * 160,
                  kDownScaleOff = kGuScaleOff + 1280 * 40 * 2;

#ifndef BALOON_PK_GEOMETRY_IN_TU
inline int64_t blocks_for(int64_t n, int64_t t) { return (n + t - 1) / t; }
#endif

// hyper-connection (residual [T][4][2560])
int gr_norm(const float* x, float* out, int T, float eps);                                  // rms over the 4 streams of a token
int gr_mix(const float* x, const float* coef, int ld, float* out, int T);                   // out[t][d] = sum_c coef[t*ld+c]*x[t][c][d]
int gr_write(const float* res, const float* y, const float* inj, int ld, float* out, int T);// out = res + inj[t*ld+c]*y[t][d]
int gr_broadcast(const float* e, float* out, int T);                                        // out[t][c][d] = e[t][d]
int gr_silu(const float* x, float* out, int64_t n);

// gated delta net
int gdn_gates(const float* a, const float* b, const float* a_log, const float* dt_bias, float* g, float* beta, int T);
int gdn_conv(const float* x, const float* w, float* state, float* out, int T);              // causal width-4 depthwise conv + silu
int gdn_l2(float* qkv, int T, float eps);                                                   // l2-normalise the 2*HK q/k heads in place
int gdn_recurrence(const float* qkv, const float* g, const float* beta, float* state, float* out, int T);

// MoE
int route(const float* logits, int32_t* ids, float* wts, int T);
int blob_dequant(const uint8_t* blob, float* gu_out, float* d_out);
int blob_dequant_f16(const uint8_t* blob, uint16_t* gu_out, uint16_t* d_out);
int swiglu_pair(const float* gate, const float* up, float* out, int64_t n);
int swiglu_interleaved(const float* in, float* out, int rows, int width);                   // in: [rows][2*width] (g,u interleaved)
int gather_rows16(const uint16_t* src, const int32_t* idx, uint16_t* dst, int rows, int width);
int moe_combine(const float* expert_out, const float* wts, const float* shared, float* out, int T);

// QSA helpers
int rms_rows(const float* x, const float* w, float* out, int rows, int width, float eps);
int rope(float* x, const int32_t* pos, int T, int heads, int head_dim, float theta);
int split_q(const float* qfull, float* q, float* gate, int T);
int gate_attn(const float* attn, const float* gate, float* out, int T);
int kv_append(const float* k, int8_t* kq, float* ks, int64_t base_row, int T);              // int8 rows + 4 scales per row

int to_bf16(const float* x, uint16_t* out, int64_t n);
int to_f16(const float* x, uint16_t* out, int64_t n);

}  // namespace pk
}  // namespace baloon
