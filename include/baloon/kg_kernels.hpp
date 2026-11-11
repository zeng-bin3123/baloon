// kg_kernels.hpp - host launchers of the src/kernels/gpu sources (wave64 / ivcore10). Every launcher is asynchronous on `s`
// (unless stated) and returns cudaSuccess or the first error; argument validation failures return cudaErrorInvalidValue.
// Matrices named *_w / w / down / up / gate are bf16 BITS unless a block format is stated; activations are f32.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace baloon {
namespace kg {

// ---- block formats of the native expert path (ggml type ids) ---------------------------------------------------------------
enum class Fmt : int { Q3_K = 11, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, IQ1_M = 29, Q2_0 = 42 };
// Codebook grids (device pointers; null = format unsupported): iq2xxs 256 x u64, iq2xs 512 x u64, iq2s 1024 x u64,
// iq3xxs 256 x u32, iq3s 512 x u32, iq1s 2048 x u64 (eight int8 lanes).
struct IqGrids {
    const uint64_t* iq2xxs = nullptr; const uint64_t* iq2xs = nullptr; const uint64_t* iq2s = nullptr;
    const uint32_t* iq3xxs = nullptr; const uint32_t* iq3s = nullptr; const uint64_t* iq1s = nullptr;
};
struct ExpertLayout { size_t gate_off = 0, up_off = 0, down_off = 0, total = 0; };

// ---- gated residual ----------------------------------------------------------------------------------------------------------
struct GrDims { int n = 0, hc = 0, lr = 0; };                         // n_embd, streams, low rank
struct GrWs { float* xn = nullptr; uint16_t* xq = nullptr; uint16_t* lq = nullptr; float* gated = nullptr; float* lo = nullptr; size_t bytes = 0; };
// fused multi-token read at n=2560, hc=4, lr=320 (1 <= T <= 32); R: T*10240 f32, lo: T*320 f32, x_out: T*2560, w_out: T*4.
struct GrMulti {
    const float* R = nullptr; const uint16_t* down = nullptr; const uint16_t* up = nullptr; const uint16_t* inject = nullptr;
    float* lo = nullptr; float* x_out = nullptr; float* w_out = nullptr; float eps = 1e-6f; int T = 1;
};

// ---- dense / elementwise -------------------------------------------------------------------------------------------------------
cudaError_t bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, long long n_in, long long n_out, cudaStream_t s = 0);
cudaError_t bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, long long n_in, long long n_out, int tpr, cudaStream_t s = 0);
cudaError_t bf16_gemv_fp32_mmvf(const uint16_t* w, const float* x, float* y, long long rows, long long cols, cudaStream_t s = 0);
cudaError_t dequant_s2(const uint8_t* blocks, float* out, long long n_blocks, cudaStream_t s = 0);
// Canonical S2 plane (codes n_blocks*16 bytes, one f32 scale per 64-element block) -> f32, one thread per block.
// n_blocks <= 0 launches nothing.
cudaError_t dequant_s2_plane(const uint8_t* codes, const float* scales, float* out, long long n_blocks, cudaStream_t s = 0);
cudaError_t add_inplace(float* y, const float* x, long long n, cudaStream_t s = 0);
cudaError_t scale_inplace(float* y, float a, long long n, cudaStream_t s = 0);
cudaError_t silu_inplace(float* y, long long n, cudaStream_t s = 0);
cudaError_t f32_to_bf16_bulk(const float* x, uint16_t* y, long long n, cudaStream_t s = 0);
cudaError_t f32_to_f16_bulk(const float* x, uint16_t* y, long long n, cudaStream_t s = 0);
cudaError_t copy_from_mapped(const float* src, float* dst, long long n, cudaStream_t s = 0);
cudaError_t copy_i32_from_mapped(const int32_t* src, int32_t* dst, long long n, cudaStream_t s = 0);
cudaError_t embedding_gather(const float* table, long long vocab, long long n, const int32_t* tokens, long long count, float* out, cudaStream_t s = 0);
cudaError_t gdn_gate(const float* alpha, const float* a_log, const float* dt_bias, float* g, int n, cudaStream_t s = 0);
cudaError_t rms_norm_weighted(const float* x, const float* w, float* y, int rows, int cols, float eps, cudaStream_t s = 0);
// doorbell word (device-visible uint32): ring = +1, publish = store, wait = spin until word >= target (<= max_spins reads); status[0] = 1/0
cudaError_t doorbell_ring(uint32_t* bell, cudaStream_t s = 0);
cudaError_t doorbell_publish(uint32_t* bell, uint32_t value, cudaStream_t s = 0);
cudaError_t doorbell_wait(uint32_t* bell, uint32_t target, uint64_t max_spins, uint32_t* status, cudaStream_t s = 0);

// ---- gated delta net -------------------------------------------------------------------------------------------------------------
cudaError_t gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* o, int S, int h_k, int h_v, cudaStream_t s = 0);
cudaError_t gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int channels, int d_conv, cudaStream_t s = 0);
cudaError_t gdn_l2_norm(float* x, int rows, int cols, float eps, cudaStream_t s = 0);
cudaError_t gdn_beta_gate(float* beta, int n, cudaStream_t s = 0);
cudaError_t gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int h_v, int S, float eps, cudaStream_t s = 0);
cudaError_t fused_gdn_ab(const float* a, const float* b, const float* a_log, const float* dt_bias, float* beta, float* g, int n, cudaStream_t s = 0);
// decode_ab_proj: one wave per row of [w_alpha; w_beta] (bf16 bits, h_v rows each, row length n_embd % 8 == 0, 16-byte aligned); g from alpha rows, beta from beta rows.
cudaError_t fused_gdn_ab(const uint16_t* w_alpha, const uint16_t* w_beta, const float* x, const float* a_log, const float* dt_bias, float* beta, float* g, int h_v, int n_embd, cudaStream_t s = 0);
cudaError_t fused_gdn_conv_l2(float* conv_state, const float* x, const float* kW, float* out, int h_k, int h_v, int d_conv, float eps, cudaStream_t s = 0);
cudaError_t fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, const float* z,
                                const float* ssm_norm, float* y, int h_k, int h_v, float eps, cudaStream_t s = 0);
bool native_gdn_enabled();
void native_gdn_set_enabled(bool on);
// native_gdn_*: the stream must be non-null (explicit stream required; the legacy default stream is rejected)
cudaError_t native_gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* o, int S, int h_k, int h_v, cudaStream_t s);
cudaError_t native_gdn_beta_gate(float* beta, int n, cudaStream_t s);
cudaError_t native_gdn_gate(const float* alpha, const float* dt_bias, const float* a_log, float* g, int n, cudaStream_t s);
cudaError_t native_gdn_conv_silu(float* conv_state, const float* x, const float* kW, float* out, int C, int dc, cudaStream_t s);
cudaError_t native_gdn_l2_norm(float* x, int rows, int cols, float eps, cudaStream_t s);
cudaError_t native_gdn_out_norm(const float* o, const float* z, const float* norm_w, float* y, int h_v, float eps, cudaStream_t s);

// ---- gated residual --------------------------------------------------------------------------------------------------------------
size_t gr_workspace_bytes(const GrDims& d);
bool gr_workspace_init(void* base, size_t avail, const GrDims& d, GrWs* ws);
void gr_set_fp32_activations(bool on);
void gr_set_native_mmvf(bool on);
cudaError_t gr_read(const float* R, const uint16_t* down, const uint16_t* up, const uint16_t* inj, const GrWs& ws, const GrDims& d, float eps, float* x_out, float* w_out, cudaStream_t s = 0);
cudaError_t gr_write(const float* R, const float* block_out, const float* w, float* out, int n, int hc, cudaStream_t s = 0);
bool fused_gr_supported(int n, int hc, int lr);
cudaError_t fused_gr_read(const float* R, const uint16_t* down, const uint16_t* up, const uint16_t* inject, float* lo, float* x_out, float* w_out, float eps, cudaStream_t s = 0);
cudaError_t fused_gr_read_multi(const GrMulti& m, cudaStream_t s = 0);
cudaError_t native_gr_rms_norm_weighted(const float* x, const float* w, float* y, int rows, int cols, float eps, cudaStream_t s = 0);
cudaError_t native_gr_pre_gated(const float* xn, const float* logit, float* gated, long long n, cudaStream_t s = 0);
cudaError_t native_gr_down_silu(float* h, long long n, cudaStream_t s = 0);
cudaError_t native_gr_post(const float* R, const float* block_out, const float* w, const float* bias, float scale, float* out, int n, int hc, cudaStream_t s = 0);

// ---- MoE ---------------------------------------------------------------------------------------------------------------------------
cudaError_t router_top10(const float* logits, int n_tokens, int n_expert, int k, int32_t* ids, float* weights, cudaStream_t s = 0);
bool native_router_enabled();
void native_router_set_enabled(bool on);
cudaError_t native_router_top10(const float* logits, int n_tokens, int n_expert, int k, int32_t* ids, float* weights, cudaStream_t s = 0);
void shared_expert_set_native_bf16(bool on);
size_t shared_expert_scratch_bytes(int n_tok, int n_embd, int n_ff);
cudaError_t shared_expert_multi(const float* x, int n_tok, int n_embd, int n_ff, const uint16_t* gate_w, const uint16_t* up_w, const uint16_t* down_w,
                                const uint16_t* gate_scalar_w, float* out, void* scratch, size_t scratch_bytes, cudaStream_t s = 0);
cudaError_t shared_expert(const float* x, int n_embd, int n_ff, const uint16_t* gate_w, const uint16_t* up_w, const uint16_t* down_w,
                          const uint16_t* gate_scalar_w, float* out, void* scratch, size_t scratch_bytes, cudaStream_t s = 0);
cudaError_t moe_combine(const float* parts, const float* weights, const float* shared, float* y, int n_embd, int k, int has_shared, cudaStream_t s = 0);
bool native_moe_combine_enabled();
void native_moe_combine_set_enabled(bool on);
cudaError_t native_moe_combine(const float* parts, const float* weights, const float* shared, float* y, int n_embd, int k, int has_shared, cudaStream_t s = 0);

// ---- Q2_0 (S2) GEMV: w = blocks of 18 bytes / 64 elements, n_in % 64 == 0 ------------------------------------------------------------
cudaError_t s2_gemv(const uint8_t* w, const float* x, float* y, int n_in, int n_out, cudaStream_t s = 0);
cudaError_t s2_gemv_fast(const uint8_t* w, const float* x, float* y, int n_in, int n_out, cudaStream_t s = 0);        // x rounded to f16
cudaError_t s2_gemv_q8(const uint8_t* w, const uint8_t* xq8, float* y, int n_in, int n_out, cudaStream_t s = 0);       // Q8_0 activations
cudaError_t s2_gemv_quads(const uint8_t* w, const float* x, float* y, int n_in, int n_out, cudaStream_t s = 0);

// ---- rotary / attention helpers --------------------------------------------------------------------------------------------------------
cudaError_t build_rope_table(float* table, int n_pos, int rot, float theta, cudaStream_t s = 0);
cudaError_t rope_neox_apply(float* x, const float* table, const int32_t* pos, int n_tok, int n_heads, int head_dim, int rot, cudaStream_t s = 0);
bool native_rope_enabled();
void native_rope_set_enabled(bool on);
bool mrope_table_set(const int sections[3]);            // null = default (all frequencies on component 0); negative section -> false, unchanged
int mrope_table(int i, int half);
cudaError_t native_rope_apply(float* x, const int32_t* pos3, int n_tok, int n_heads, int head_dim, int rot, float theta, cudaStream_t s = 0);   // synchronizes s
bool native_qsa_enabled();
void native_qsa_set_enabled(bool on);
cudaError_t native_qsa_gate_apply(float* out, const float* gate_v, long long n, cudaStream_t s = 0);
cudaError_t native_qsa_rms_norm_weighted(float* x, const float* w, int rows, int cols, float eps, cudaStream_t s = 0);
bool native_qsa_indexer_enabled();
void native_qsa_indexer_set_enabled(bool on);
cudaError_t native_qsa_indexer_append(float* cache, float* norms, const float* key, int pos, int capacity, cudaStream_t s = 0);   // D = 128
cudaError_t native_ple_postops(const float* gate, const float* ple, float* t_out, const float* proj, const float* norm_w, const float* resid,
                               const float* layer_scalar, float* out, int rows, int cols, float eps, cudaStream_t s = 0);

// ---- i-quant / k-quant / Q2_0 block formats --------------------------------------------------------------------------------------------
bool is_iq(Fmt f);
bool iq_supported(Fmt f, const IqGrids& g);
int iq_block_elems(Fmt f);
int iq_block_bytes(Fmt f);
long long iq_row_bytes(Fmt f, long long cols);            // -1 when cols is not a whole number of blocks
cudaError_t iq_dequant_f32(Fmt f, const uint8_t* src, float* dst, long long n, const IqGrids& g, cudaStream_t s = 0);
cudaError_t iq_dequant_f16(Fmt f, const uint8_t* src, uint16_t* dst, long long n, const IqGrids& g, cudaStream_t s = 0);
cudaError_t iq_dequant_gu_f16(Fmt f, const uint8_t* gate, const uint8_t* up, uint16_t* dst, int rows, int cols, const IqGrids& g, cudaStream_t s = 0);
cudaError_t iq_embed_rows(Fmt f, const uint8_t* table, long long n_rows, const int32_t* ids, int n_ids, int cols, float* out, const IqGrids& g, cudaStream_t s = 0);
cudaError_t quantize_q8_1_rows(const float* x, uint8_t* out, int rows, int cols, cudaStream_t s = 0);   // 36-byte blocks, cols % 32 == 0
cudaError_t iq_mmvq(Fmt f, const uint8_t* w, const uint8_t* q8, float* y, int rows, int cols, const IqGrids& g, cudaStream_t s = 0);
bool native_expert_layout(Fmt f, int n_embd, int n_ff, ExpertLayout* out);
size_t native_expert_scratch_bytes(int n_embd, int n_ff);
cudaError_t native_expert_grouped(Fmt f, const uint8_t* blob, const float* x, int n_embd, int n_ff, float* out, void* scratch, size_t scratch_bytes,
                                  const IqGrids& g, cudaStream_t s = 0);

}  // namespace kg
}  // namespace baloon
