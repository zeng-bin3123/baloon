// prefill_batch_api.hpp - constants and host API of the batched-prompt CUDA kernels in src/prefill_batch_kernels.cu.
// Every pointer is a device pointer unless noted. Return: 0 ok, -1 rejected by host validation (nothing launched),
// >0 a cudaError_t value. Buffers are flat row-major; kernels decompose a linear thread index into (token, stream, dim).
#pragma once
#include <cstdint>

namespace baloon {
namespace pfb {

constexpr int kEmbN = 2560;                 // embedding width
constexpr int kStreams = 4;                 // hyper-connection streams
constexpr int kHcWidth = kEmbN * kStreams;  // 10240
constexpr int kStateS = 128;                // GDN head dim
constexpr int kKeyHeads = 16, kValHeads = 48;
constexpr int kConvC = (2 * kKeyHeads + kValHeads) * kStateS;  // 10240 = q | k | v channels
constexpr int kValWidth = kValHeads * kStateS;                 // 6144
constexpr int kConvTaps = 4;                // causal depthwise conv taps (3 stored + the new input)
constexpr int kRowGroups = 4;               // recurrence: 4 groups x 32 rows = 128 state rows
constexpr int kRowsPerGroup = kStateS / kRowGroups;
constexpr int kExperts = 512, kLanes = 32, kRegs = kExperts / kLanes, kTopK = 10;
constexpr int kQHeads = 24, kQHeadDim = 256, kQFullPerHead = 2 * kQHeadDim;
constexpr int kKvChannels = 256, kKvGroups = 4, kKvGroupWidth = kKvChannels / kKvGroups;

// expert blob (bytes): gate/up codes | down codes | gate/up scales (f16) | down scales (f16)
constexpr int64_t kGuRows = 1280, kGuCols = 2560, kDnRows = 2560, kDnCols = 640;
constexpr int64_t kGuOff = 0;
constexpr int64_t kDownOff = kGuRows * (kGuCols / 4);                 // 1280 * 640
constexpr int64_t kGuScaleOff = kDownOff + kDnRows * (kDnCols / 4);   // + 2560 * 160
constexpr int64_t kDownScaleOff = kGuScaleOff + kGuRows * 40 * 2;     // + 1280 * 40 f16
constexpr int64_t kBlobBytes = kDownScaleOff + kDnRows * 10 * 2;      // + 2560 * 10 f16
static_assert(kBlobBytes == 1382400, "expert blob is 1,382,400 bytes");
static_assert(kConvC == 10240 && kValWidth == 6144 && kHcWidth == 10240, "channel layout");
static_assert(kRowGroups * kRowsPerGroup == kStateS, "row groups cover the state rows");
static_assert(kLanes * kRegs == kExperts, "32 lanes x 16 registers cover the experts");

inline int64_t blocks_for(int64_t n, int64_t t) { return (n + t - 1) / t; }

int to_bf16(const float* x, uint16_t* y, int64_t n);                       // y = bf16 bits, round-to-nearest-even
int to_f16(const float* x, uint16_t* y, int64_t n);                        // y = f16 bits, round-to-nearest-even
int rms_rows(const float* x, const float* w, float* y, int rows, int dim, float eps);
int gr_norm(const float* res, float* y, int T, float eps);                 // per-token RMS over the 4 streams (width 10240)
int gr_mix(const float* res, const float* coef, int ld, float* y, int T);  // y[t][d] = sum_c coef[t*ld+c] * res[t][c][d]
int gr_write(const float* res, const float* inj, int ld, const float* blk, float* out, int T);  // out = res + inj[t*ld+c]*blk[t][d]
int gr_broadcast(const float* e, float* out, int T);                       // out[t][c][d] = e[t][d]
int gr_silu(const float* x, float* y, int64_t n);
int gdn_gates(const float* a, const float* b, const float* a_log, const float* dt_bias, float* g, float* beta, int T);
int gdn_conv(const float* x, const float* w, float* state, float* out, int T);   // state [C][3], w [C][4]; silu applied
int gdn_l2(float* qkv, int T, float eps);                                  // L2-normalise the 32 q/k heads of every token
int gdn_recurrence(const float* qkv, const float* g, const float* beta, float* state, float* out, int T, float scale);
int route(const float* logits, int32_t* ids, float* wts, int T);           // top-10 of 512 logits, softmax over the chosen
int blob_dequant(const uint8_t* blob, float* gu, float* dn);               // value = (code-1) * f16 scale of its 64-group
int blob_dequant_f16(const uint8_t* blob, uint16_t* gu, uint16_t* dn);
int swiglu_interleaved(const float* x, float* y, int R);                   // x [R][1280] gate/up interleaved -> y [R][640]
int swiglu_pair(const float* gate, const float* up, float* y, int64_t n);
int gather_rows16(const void* src, const int32_t* idx, void* dst, int R, int units_per_row);  // 16-byte units
int moe_combine(const float* shared, const float* expert, const float* w, float* out, int T);
int split_q(const float* qfull, float* q, int T);                          // query half of every (query|gate) head
int gate_attn(const float* attn, const float* qfull, float* out, int T);   // out = attn * sigmoid(gate half)
int rope(float* x, const int32_t* pos, int T, int heads, int head_dim, float base);  // rotates pairs (p, p+32) of 64 dims
int kv_append(const float* x, const int32_t* rows, int8_t* kq, float* sc, int T);    // int8 rows + one scale per 64 channels

}  // namespace pfb
}  // namespace baloon
