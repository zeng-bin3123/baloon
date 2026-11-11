// kern_cuda_misc.hpp - public launchers of the unit-14 CUDA sources: dequantization to bf16/f16/f32, S-family GEMV,
// Q2_0 grouped expert (S2) kernels, sampler. Host-side declarations only (include <cuda_runtime.h> first).
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "baloon/kern_geometry.hpp"

namespace baloon {
namespace kern {

// ---------------------------------------------------------------------------------------- dequant_bf16.cu
// Types: 0 F32, 1 F16, 30 BF16 (plain conversion) and the ggml/engine quantized types
// 2 Q4_0, 6 Q5_0, 8 Q8_0, 20 IQ4_NL, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K, 23 IQ4_XS, 42 Q2_0.
// n is the element count (a whole number of blocks for the quantized types). dst may be uint16_t bits (bf16/f16) or float.
bool dequant_bf16_supported(int type);
cudaError_t dequant_bf16(int type, const void* src, uint16_t* dst, int64_t n, cudaStream_t stream);
cudaError_t dequant_f16(int type, const void* src, uint16_t* dst, int64_t n, cudaStream_t stream);
cudaError_t dequant_f32(int type, const void* src, float* dst, int64_t n, cudaStream_t stream);
}  // namespace kern
namespace kg { struct IqGrids; }
namespace kern {
// Same, plus the i-quant route: types 16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS, 21 IQ3_S, 22 IQ2_S, 29 IQ1_M go to iq_kernels
// (kg::iq_dequant_f16/f32) with the device codebooks `iq`; iq == nullptr or iq_kernels not linked -> cudaErrorInvalidValue.
// dequant_bf16 has no i-quant route.
cudaError_t dequant_f16(int type, const void* src, uint16_t* dst, int64_t n, cudaStream_t stream, const kg::IqGrids* iq);
cudaError_t dequant_f32(int type, const void* src, float* dst, int64_t n, cudaStream_t stream, const kg::IqGrids* iq);

// ---------------------------------------------------------------------------------------- s_gemv.cu
// y[row] = sum_i w[row][i] * x[i], w = decode(code) * group scale. bits = 2 (code-1), 4 (IQ4_NL codebook), 8 (int8).
// codes: row-major, n_in*bits/8 bytes per row. scales: fp16 bits, n_in >> group_shift per row. group = 1 << group_shift
// elements (16, 32 or 64). n_in % 16 == 0.
cudaError_t s_gemv(int bits, const uint8_t* codes, const uint16_t* scales, const float* x, float* y, int n_in, int n_out,
                   int group_shift, cudaStream_t stream);
// Activation as Q8_K blocks (292 bytes / 256 elements); n_in % 256 == 0. One warp per row.
cudaError_t s_gemv_q8k(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* x_q8k, float* y, int n_in,
                       int n_out, int group_shift, cudaStream_t stream);
// Split-K: the lane iterations are cut into `splits` ranges whose partial sums land in `partials` (splits * n_out floats)
// and are added in range order by a finishing kernel (deterministic).
cudaError_t s_gemv_q8k_split(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* x_q8k, float* y,
                             float* partials, int n_in, int n_out, int group_shift, int splits, cudaStream_t stream);
// Activation as Q8_0 blocks (34 bytes / 32 elements); n_in % 32 == 0.
cudaError_t s_gemv_q8_0_split(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* x_q8_0, float* y,
                              float* partials, int n_in, int n_out, int group_shift, int splits, cudaStream_t stream);
// Dispatch by activation form: q8k != 0 -> Q8_K, otherwise Q8_0. The _async form never synchronizes.
cudaError_t s_gemv_split(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* x_q, int q8k, float* y,
                         float* partials, int n_in, int n_out, int group_shift, int splits, cudaStream_t stream);
cudaError_t s_gemv_split_async(int bits, const uint8_t* codes, const uint16_t* scales, const uint8_t* x_q, int q8k, float* y,
                               float* partials, int n_in, int n_out, int group_shift, int splits, cudaStream_t stream);
bool q8k_form_ok(int n_in);

// ---------------------------------------------------------------------------------------- s2_expert_grouped.cu
// Resident Q2_0 experts: blob of slot s at pool + s * slot_bytes (slot_bytes >= q2geo::BLOB). All launchers are
// asynchronous on `stream`; `scratch` must hold moe_hit_grouped_scratch_bytes(n_tok, K) bytes.
size_t moe_hit_grouped_scratch_bytes(int n_tok, int K);

// Resident-hit selection by ballot-prefix compaction. ids: n_tok*K selected experts; slot_of_expert: NE entries (-1 = not
// resident). Outputs per token t: hit_slot[t*K + pos], hit_dst[t*K + pos] (router slot k) for pos < n_hit[t]; unused
// entries have hit_slot = -1. K <= 64.
cudaError_t moe_hit_select(const int32_t* ids, const int32_t* slot_of_expert, int K, int32_t* hit_slot, int32_t* hit_dst,
                           int32_t* n_hit, cudaStream_t stream);
cudaError_t moe_hit_select_multi(const int32_t* ids, const int32_t* slot_of_expert, int n_tok, int K, int32_t* hit_slot,
                                 int32_t* hit_dst, int32_t* n_hit, cudaStream_t stream);
// Grouping by resident expert (stable: ascending expert, then token, then k): pair arrays of capacity n_tok*K,
// unused entries slot = -1.
cudaError_t moe_group_resident(const int32_t* ids, const int32_t* slot_of_expert, int n_tok, int K, int32_t* pair_slot,
                               int32_t* pair_tok, int32_t* pair_dst, cudaStream_t stream);

// out[t*H + i] += sum over router slots k (ascending) of topk_w[t*K + k] * expert(ids[t*K + k])(x[t])[i], resident experts only.
// Single token, hit list known on the host (n_hit entries of hit_slot / hit_dst).
cudaError_t moe_hit_grouped_s2(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* hit_slot,
                               const int32_t* hit_dst, int n_hit, const float* topk_w, int K, float* out, void* scratch,
                               cudaStream_t stream);
// Same, hit count read from the device (n_hit_dev), at most max_hits entries.
cudaError_t moe_hit_grouped_s2_dev(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* hit_slot,
                                   const int32_t* hit_dst, const int32_t* n_hit_dev, int max_hits, const float* topk_w, int K,
                                   float* out, void* scratch, cudaStream_t stream);
// n_tok tokens; hit arrays as produced by moe_hit_select_multi.
cudaError_t moe_hit_grouped_s2_multi(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* hit_slot,
                                     const int32_t* hit_dst, int n_tok, int K, const float* topk_w, float* out, void* scratch,
                                     cudaStream_t stream);
// Same arithmetic order as the CPU kernel (serial float accumulation per row, round-to-nearest-even quantization).
cudaError_t moe_hit_grouped_s2_cpu_order(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* hit_slot,
                                         const int32_t* hit_dst, int n_tok, int K, const float* topk_w, float* out,
                                         void* scratch, cudaStream_t stream);
// Pair-list form (as produced by moe_group_resident), tokens grouped per resident expert.
cudaError_t moe_grouped_s2(const uint8_t* pool, uint64_t slot_bytes, const float* x, const int32_t* pair_slot,
                           const int32_t* pair_tok, const int32_t* pair_dst, int n_tok, int K, const float* topk_w, float* out,
                           void* scratch, cudaStream_t stream);
// Combine step on its own: out += weighted sum of the per-router-slot expert outputs held in scratch.
cudaError_t moe_hit_add(const float* topk_w, int n_tok, int K, float* out, void* scratch, cudaStream_t stream);

// ---------------------------------------------------------------------------------------- sampler.cu
struct SamplerParams {
    float temperature = 1.f;       // <= 0: greedy
    int top_k = 40;                // <= 0: 64 (capacity)
    float top_p = 1.f;
    int min_keep = 1;
    float repeat_penalty = 1.f;
    float presence_penalty = 0.f;
    float frequency_penalty = 0.f;
    int last_n = 64;               // history window (tail of each row)
};
constexpr int kSamplerMaxKeep = 64;
// logits: n_rows * n_vocab; history: n_rows rows of hist_stride ints with hist_len[row] valid entries.
// workspace: n_rows * n_vocab floats (penalized logits). out_tokens: n_rows.
cudaError_t sample_tokens(const float* logits, int n_rows, int n_vocab, const int32_t* history, int hist_stride,
                          const int32_t* hist_len, const SamplerParams& p, uint64_t seed, uint64_t counter, float* workspace,
                          int32_t* out_tokens, cudaStream_t stream);

// ---------------------------------------------------------------------------------------- quantize_act.cu
// Q8_0 block = fp16 d + 32 int8 (34 B); Q8_K block = f32 d + 256 int8 + 16 int16 bsums (292 B). nb = block count.
cudaError_t quantize_q8_0_ggml_rule(const float* x, uint8_t* out, int64_t nb, cudaStream_t stream);   // clamp [-128,127]
cudaError_t quantize_q8_0_cpu_rule(const float* x, uint8_t* out, int64_t nb, cudaStream_t stream);    // clamp [-127,127]
cudaError_t quantize_q8_k(const float* x, uint8_t* out, int64_t nb, cudaStream_t stream);

}  // namespace kern
}  // namespace baloon
