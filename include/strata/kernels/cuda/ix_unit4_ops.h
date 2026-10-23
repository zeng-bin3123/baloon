// ix_unit4_ops.h -- Unit 4 CUDA kernel APIs for BI-V100 (ivcore10)
// Fused GDN, fused GR, activation quantize, QSA decode attention, IQ kernels.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif

// Fused GDN step + GroupNorm + sigmoid gate
void ix_fused_gdn_step_norm(
    float* state, const float* q, const float* k, const float* v,
    const float* gate, const float* beta, const float* z,
    const float* gamma, float eps, float* result,
    int n_head, int S, cudaStream_t stream);

// Gated residual (hyper-connection)
void ix_gr_norm(const float* R, const float* norm_w, float eps,
                int n_embd, int hc, float* xn, cudaStream_t stream);
void ix_gr_mean(const float* gated, int n_embd, int hc,
                float* mixed, cudaStream_t stream);
void ix_gr_write(const float* R, const float* block_out, const float* inject,
                 int n_embd, int hc, float* R_out, cudaStream_t stream);

// Activation quantization Q8_0
void ix_quantize_q8_0(const float* x, uint8_t* out, int64_t n, cudaStream_t stream);
void ix_dequant_q8_0(const uint8_t* blocks, float* out, int64_t n, cudaStream_t stream);
void ix_quantize_q8_0_scaled(const float* x, uint8_t* out, float* scales,
                             int64_t n, cudaStream_t stream);

// QSA sparse decode attention
void ix_qsa_decode_attn(
    const float* q, const uint16_t* k_sel, const uint16_t* v_sel,
    int n_heads, int n_kv_heads, int head_dim, int n_sel,
    float* out, cudaStream_t stream);

// IQ kernels
void ix_iq4nl_dequant(const uint8_t* data, float* out, int64_t n, cudaStream_t stream);
void ix_vec_dot_q8k(const uint8_t* a, const uint8_t* b, float* result,
                    int64_t n_blocks, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
