// ix_kernels.h — Public C API for all BI-V100 CUDA kernels
//
// All kernels are compiled with clang++ -x cuda for ivcore10.
// Call from C++ host code or Python ctypes.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════
// infccl — P2P AllReduce for BI-V100 PCIe topology
// ═══════════════════════════════════════════════════════════════════

int  infccl_init(int world_size, const int* device_ids, int64_t max_elems);
void infccl_destroy(void);
int  infccl_allreduce(float** bufs, int64_t n);
int  infccl_allreduce_inplace(float* x, int64_t n, int rank);
int  infccl_allreduce_f16(half** bufs, int64_t n);

// ═══════════════════════════════════════════════════════════════════
// ix_native_mmvq — Dense Q8_0 GEMV (warp64)
// ═══════════════════════════════════════════════════════════════════

// Single-column Q8_0 × Q8_1 GEMV
void ix_mmvq_q8_0(const void* weights, const void* activations,
                   float* output, int64_t n_in, int64_t n_out,
                   cudaStream_t stream);

// Multi-column (2-4 tokens) Q8_0 × Q8_1 GEMV (interleaved)
void ix_mmvq_q8_0_multi(const void* weights, const void** activations,
                         float* output, int64_t n_in, int64_t n_out,
                         int n_cols, cudaStream_t stream);

// Q4_K × Q8_1 GEMV (for expert gate/up)
void ix_mmvq_q4k(const void* weights, const void* activations,
                  float* output, int64_t n_in, int64_t n_out,
                  cudaStream_t stream);

// FP32 → Q8_1 quantize
void ix_quantize_q8_1(const float* input, void* output, int64_t n,
                       cudaStream_t stream);

// Fused SwiGLU + Q8_1 quantize
void ix_swiglu_quantize_q8_1(const float* gate, const float* up,
                              void* output, int64_t n,
                              cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════
// ix_native_router — MoE Router (256 experts, top-k, warp64)
// ═══════════════════════════════════════════════════════════════════

// Softmax → top-k → renormalize
void ix_router_top_k(const float* logits, int n_tokens, int n_expert,
                     int k, int* ids, float* weights,
                     cudaStream_t stream);

// BF16 router GEMV: x_bf16 · W_router → logits
void ix_router_bf16_gemv(const uint16_t* x_bf16, const uint16_t* w_bf16,
                         float* logits, int n_embd, int n_expert,
                         cudaStream_t stream);

// Combined router + shared expert scalar gate
void ix_router_and_shared_gate(const uint16_t* x_bf16,
                                const uint16_t* w_combined,
                                float* logits, float* shared_gate,
                                int n_embd, int n_expert,
                                cudaStream_t stream);

// Expert grouping (dispatch table)
void ix_router_group(const int* ids, int k,
                     int* group_ids, int* group_count, int* group_map,
                     cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════
// ix_moe_elementwise — MoE combine + fused ops
// ═══════════════════════════════════════════════════════════════════

// out = shared + Σ w[j] * parts[j]
void ix_moe_combine(const float* shared_out, const float* parts,
                    const float* w, float* out,
                    int n_embd, int k, cudaStream_t stream);

// Fused: combine + RMSNorm + Q8_1 quantize
void ix_moe_combine_norm_q8_1(const float* residual, const float* shared_out,
                               const float* parts, const float* w,
                               const float* norm_weight, float eps,
                               float* out_f32, void* out_q8,
                               int n_embd, int k, cudaStream_t stream);

// Gated residual: out = residual * gate + addition
void ix_gated_residual(const float* residual, const float* addition,
                       const float* gate, float* out,
                       int n, cudaStream_t stream);

// Fused: add + RMSNorm + Q8_1
void ix_add_norm_q8_1(const float* a, const float* b,
                      const float* norm_w, float eps,
                      float* x_out, void* q8_out,
                      int n, cudaStream_t stream);

// S2 expert grouped GEMV (codebook + SwiGLU fused)
void ix_expert_grouped(const uint8_t* expert_blob, const uint8_t* x_q8_0,
                       float* out, int n_in, int n_ff,
                       cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════
// ix_bf16_gemv — BF16 GEMV + conversions + norms
// ═══════════════════════════════════════════════════════════════════

// BF16 → FP16 batch conversion
void ix_bf16_to_f16(const uint16_t* bf16_in, void* f16_out, int64_t n,
                    cudaStream_t stream);

// BF16 × BF16 GEMV (via FP32 accumulation)
void ix_bf16_gemv(const uint16_t* x_bf16, const uint16_t* w_bf16,
                  float* y, int n_in, int n_out,
                  cudaStream_t stream);

// Paired BF16 GEMV (two projections sharing one activation)
void ix_bf16_gemv_pair(const uint16_t* x_bf16,
                       const uint16_t* w1_bf16, const uint16_t* w2_bf16,
                       float* y1, float* y2,
                       int n_in, int n_out1, int n_out2,
                       cudaStream_t stream);

// FP32 → BF16 conversion
void ix_f32_to_bf16(const float* f32_in, uint16_t* bf16_out, int64_t n,
                    cudaStream_t stream);

// RMSNorm
void ix_rms_norm(const float* x, const float* weight, float* y,
                 float eps, int n, cudaStream_t stream);

// Fused Add + RMSNorm
void ix_add_rms_norm(const float* a, const float* b, const float* weight,
                     float* sum_out, float* norm_out,
                     float eps, int n, cudaStream_t stream);

// Sigmoid
void ix_sigmoid(const float* in, float* out, int n, cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════
// ix_gdn — DeltaNet/SSM kernels
// ═══════════════════════════════════════════════════════════════════

// Short Conv1D + SiLU (causal, width 4)
void ix_gdn_conv_silu(const float* x, float* conv_state,
                      const float* conv_weight, const float* conv_bias,
                      float* out, int C, int d_conv,
                      cudaStream_t stream);

// L2 normalize (per-head, for q and k only)
void ix_l2_norm(float* vec, int n_head, int head_dim,
                cudaStream_t stream);

// DeltaNet recurrence step
void ix_gdn_step(float* state, const float* q, const float* k,
                 const float* v, const float* gate, const float* beta,
                 float* o, int n_head, int S,
                 cudaStream_t stream);

// GDN output: rms_norm(o) * sigmoid(z)
void ix_gdn_output(const float* o, const float* z, const float* norm_w,
                   float* y, float eps, int n,
                   cudaStream_t stream);

// Softplus gate: softplus(alpha + dt) * ssm_a
void ix_softplus_gate(const float* alpha, const float* dt,
                      const float* ssm_a, float* gate,
                      int n_head, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif
