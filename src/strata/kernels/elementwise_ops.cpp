// strata/kernels/elementwise_ops.cpp — GPU/CPU dispatch for elementwise ops
// Target: Iluvatar BI-V100 (ivcore10)
//
// Routes operations to GPU kernels (ix_ew_*) or CPU fallback (elementwise.hpp).

#include "strata/kernels/elementwise_ops.hpp"
#include "strata/kernels/elementwise.hpp"

namespace strata {
namespace kernels {

// Forward declarations for GPU kernel extern "C" functions.
// These are defined in ix_elementwise.cu and linked from libix_kernels.so.
// Declared __attribute__((weak)) so host-only linking succeeds:
// the symbols resolve to nullptr when libix_kernels.so is not linked,
// and g_gpu_kernels_loaded guards all calls at runtime.
#define IX_WEAK __attribute__((weak))

extern "C" {
IX_WEAK void ix_ew_embedding_gather(
    const uint8_t* codes, const float* scales, const float* offsets,
    int64_t n, int code_bits, int code_bias, int group_elems,
    float* out, cudaStream_t stream);

IX_WEAK void ix_ew_silu_inplace(float* x, int64_t n, cudaStream_t stream);
IX_WEAK void ix_ew_scale_inplace(float* x, int64_t n, float s, cudaStream_t stream);
IX_WEAK void ix_ew_add_inplace(float* dst, const float* src, int64_t n, cudaStream_t stream);

IX_WEAK void ix_ew_rms_norm_weighted(
    float* x, const float* w, int64_t rows, int64_t cols,
    float eps, cudaStream_t stream);

IX_WEAK void ix_ew_doorbell_ring(uint32_t* d_seq, cudaStream_t stream);
IX_WEAK void ix_ew_doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, cudaStream_t stream);
IX_WEAK void ix_ew_copy_from_mapped(float* dst, const float* src, int64_t n, cudaStream_t stream);
IX_WEAK void ix_ew_copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, cudaStream_t stream);
IX_WEAK void ix_ew_doorbell_publish(
    const float* x, const int32_t* ids, const float* weights,
    int64_t n, int64_t k,
    float* x_out, int32_t* ids_out, float* weights_out,
    uint32_t* d_seq, cudaStream_t stream);

IX_WEAK void ix_ew_f32_to_f16_bulk(const float* src, uint16_t* dst, int64_t n, cudaStream_t stream);
IX_WEAK void ix_ew_f32_to_bf16_bulk(const float* src, uint16_t* dst, int64_t n, cudaStream_t stream);

IX_WEAK void ix_ew_gdn_gate(
    const float* alpha, const float* dt, const float* ssm_a,
    float* gate, int64_t n_tokens, int64_t h_v, cudaStream_t stream);
}

#undef IX_WEAK

namespace ops {

bool g_gpu_kernels_loaded = false;

void embedding_gather(const uint8_t* codes, const float* scales,
                      const float* offsets, int64_t n, int code_bits,
                      int code_bias, int group_elems, float* out,
                      cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_embedding_gather(codes, scales, offsets, n, code_bits,
                               code_bias, group_elems, out, stream);
    } else {
        strata::kernels::embedding_gather(codes, scales, offsets, n, code_bits,
                                          code_bias, group_elems, out, stream);
    }
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
              float* gate, int64_t n_tokens, int64_t h_v,
              cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_gdn_gate(alpha, dt, ssm_a, gate, n_tokens, h_v, stream);
    } else {
        strata::kernels::gdn_gate(alpha, dt, ssm_a, gate, n_tokens, h_v, stream);
    }
}

void scale_inplace(float* x, int64_t n, float s,
                   cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_scale_inplace(x, n, s, stream);
    } else {
        strata::kernels::scale_inplace(x, n, s, stream);
    }
}

void add_inplace(float* dst, const float* src, int64_t n,
                 cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_add_inplace(dst, src, n, stream);
    } else {
        strata::kernels::add_inplace(dst, src, n, stream);
    }
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n,
                     cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_f32_to_f16_bulk(x, y, n, stream);
    } else {
        strata::kernels::f32_to_f16_bulk(x, y, n, stream);
    }
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n,
                      cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_f32_to_bf16_bulk(x, y, n, stream);
    } else {
        strata::kernels::f32_to_bf16_bulk(x, y, n, stream);
    }
}

void silu_inplace(float* x, int64_t n,
                  cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_silu_inplace(x, n, stream);
    } else {
        strata::kernels::silu_inplace(x, n, stream);
    }
}

void rms_norm_weighted(float* x, const float* w, int64_t rows,
                       int64_t cols, float eps,
                       cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_rms_norm_weighted(x, w, rows, cols, eps, stream);
    } else {
        strata::kernels::rms_norm_weighted(x, w, rows, cols, eps, stream);
    }
}

void doorbell_ring(uint32_t* d_seq,
                   cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_doorbell_ring(d_seq, stream);
    } else {
        strata::kernels::doorbell_ring(d_seq, stream);
    }
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq,
                   cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_doorbell_wait(d_flag, d_seq, stream);
    } else {
        strata::kernels::doorbell_wait(d_flag, d_seq, stream);
    }
}

void copy_from_mapped(float* dst, const float* src, int64_t n,
                      cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_copy_from_mapped(dst, src, n, stream);
    } else {
        strata::kernels::copy_from_mapped(dst, src, n, stream);
    }
}

void doorbell_publish(const float* x, const int32_t* ids,
                      const float* weights, int64_t n, int64_t k,
                      float* x_out, int32_t* ids_out, float* weights_out,
                      uint32_t* d_seq,
                      cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_doorbell_publish(x, ids, weights, n, k,
                               x_out, ids_out, weights_out, d_seq, stream);
    } else {
        strata::kernels::doorbell_publish(x, ids, weights, n, k,
                                          x_out, ids_out, weights_out, d_seq, stream);
    }
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n,
                          cudaStream_t stream, bool use_gpu) {
    if (use_gpu && g_gpu_kernels_loaded) {
        ix_ew_copy_i32_from_mapped(dst, src, n, stream);
    } else {
        strata::kernels::copy_i32_from_mapped(dst, src, n, stream);
    }
}

}  // namespace ops
}  // namespace kernels
}  // namespace strata

