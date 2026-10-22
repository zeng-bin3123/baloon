// strata/kernels/gr.cpp — GR (Grouped Residual) kernel implementations
// Target: Iluvatar BI-V100 (ivcore10)
//
// Host-side reference implementations.

#include "strata/kernels/gr.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace strata {
namespace kernels {

// ─── Global configuration state ────────────────────────────────────
static bool g_gr_fp32_activations = false;
static bool g_gr_native_mmvf = false;

void gr_set_fp32_activations(bool enabled) {
    g_gr_fp32_activations = enabled;
}

void gr_set_native_mmvf(bool enabled) {
    g_gr_native_mmvf = enabled;
    if (enabled) {
        g_gr_fp32_activations = true;  // native MMVF implies FP32
    }
}

// ─── bf16 helpers ──────────────────────────────────────────────────
static inline float bf16_to_f32(uint16_t val) {
    uint32_t out = static_cast<uint32_t>(val) << 16;
    float result;
    std::memcpy(&result, &out, 4);
    return result;
}

static inline uint16_t f32_to_bf16(float val) {
    uint32_t x;
    std::memcpy(&x, &val, 4);
    // Round-to-nearest-even
    uint32_t rounding = 0x00007FFFu + ((x >> 16) & 1u);
    x += rounding;
    return static_cast<uint16_t>(x >> 16);
}

// ─── gr_workspace_init ─────────────────────────────────────────────
// Carve base into workspace regions, 16-byte aligned.
static inline size_t align16(size_t n) {
    return (n + 15) & ~static_cast<size_t>(15);
}

size_t gr_workspace_init(const GrShapes& shapes, void* base,
                         GrWorkspace& out) {
    const int64_t total_elems = shapes.hc * shapes.n_embd;
    const int64_t lr = shapes.hc_lr;

    // Region sizes
    const size_t xn_bytes    = align16(static_cast<size_t>(total_elems) * sizeof(float));
    const size_t xq_bytes    = align16(static_cast<size_t>(total_elems) * sizeof(uint16_t));
    const size_t lq_bytes    = align16(static_cast<size_t>(lr) * sizeof(uint16_t));
    const size_t gated_bytes = align16(static_cast<size_t>(total_elems) * sizeof(float));
    const size_t lo_bytes    = align16(static_cast<size_t>(lr) * sizeof(float));

    const size_t total = xn_bytes + xq_bytes + lq_bytes + gated_bytes + lo_bytes;

    if (!base) {
        // Size-only query
        return total;
    }

    auto* p = static_cast<uint8_t*>(base);

    out.xn    = reinterpret_cast<float*>(p);               p += xn_bytes;
    out.xq    = reinterpret_cast<uint16_t*>(p);             p += xq_bytes;
    out.lq    = reinterpret_cast<uint16_t*>(p);             p += lq_bytes;
    out.gated = reinterpret_cast<float*>(p);               p += gated_bytes;
    out.lo    = reinterpret_cast<float*>(p);               // p += lo_bytes;
    out.bytes = total;

    return total;
}

// ─── gr_read ───────────────────────────────────────────────────────
// build_hc_mix: collapse hc streams to one vector.
void gr_read(const float* R, const float* w_norm,
             const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps,
             const GrShapes& shapes, const GrWorkspace& ws,
             float* mixed, float* inject, cudaStream_t /*stream*/) {
    if (!R || !w_norm || !w_down || !w_up || !mixed) return;
    if (!shapes.valid()) return;
    if (ws.bytes == 0) return;  // workspace not initialized

    const int64_t n_embd = shapes.n_embd;
    const int64_t hc     = shapes.hc;
    const int64_t hc_lr  = shapes.hc_lr;
    const int64_t total  = hc * n_embd;
    const bool use_fp32  = g_gr_fp32_activations || g_gr_native_mmvf;

    // Step 1: xn[c] = RMSNorm(R[c], w_norm[c])
    // w_norm is stored as (1+w), so the multiplication already includes the implicit +1.
    for (int64_t c = 0; c < hc; ++c) {
        const float* r_stream = R + c * n_embd;
        const float* wn = w_norm + c * n_embd;
        float* xn_stream = ws.xn + c * n_embd;

        // RMS norm per stream
        float sq_sum = 0.0f;
        for (int64_t d = 0; d < n_embd; ++d) {
            sq_sum += r_stream[d] * r_stream[d];
        }
        const float mean_sq = sq_sum / static_cast<float>(n_embd);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (int64_t d = 0; d < n_embd; ++d) {
            xn_stream[d] = r_stream[d] * inv_rms * wn[d];
        }
    }

    // Step 1b: xq = bf16(xn) unless FP32 contract
    if (!use_fp32) {
        for (int64_t i = 0; i < total; ++i) {
            ws.xq[i] = f32_to_bf16(ws.xn[i]);
        }
    }

    // Step 2: lo = SiLU(xn_source · w_down^T / hc)
    // w_down: (hc_lr, hc * n_embd), bf16
    // Division INSIDE SiLU
    for (int64_t r = 0; r < hc_lr; ++r) {
        float dot = 0.0f;
        for (int64_t d = 0; d < total; ++d) {
            float xval = use_fp32 ? ws.xn[d] : bf16_to_f32(ws.xq[d]);
            float wval = bf16_to_f32(w_down[r * total + d]);
            dot += xval * wval;
        }
        // Division by hc INSIDE SiLU
        const float x = dot / static_cast<float>(hc);
        ws.lo[r] = x / (1.0f + std::exp(-x));
    }

    // Step 2b: lq = bf16(lo) unless FP32 contract
    if (!use_fp32) {
        for (int64_t i = 0; i < hc_lr; ++i) {
            ws.lq[i] = f32_to_bf16(ws.lo[i]);
        }
    }

    // Step 3: gated[c,d] = xn[c,d] * sigmoid(lo_source · w_up^T[c*n_embd + d])
    // w_up: (hc * n_embd, hc_lr), bf16
    for (int64_t idx = 0; idx < total; ++idx) {
        float dot = 0.0f;
        for (int64_t r = 0; r < hc_lr; ++r) {
            float lo_val = use_fp32 ? ws.lo[r] : bf16_to_f32(ws.lq[r]);
            float wval = bf16_to_f32(w_up[idx * hc_lr + r]);
            dot += lo_val * wval;
        }
        const float sig = 1.0f / (1.0f + std::exp(-dot));
        ws.gated[idx] = ws.xn[idx] * sig;
    }

    // Step 4: mixed = mean_c(gated[c])
    for (int64_t d = 0; d < n_embd; ++d) {
        float sum = 0.0f;
        for (int64_t c = 0; c < hc; ++c) {
            sum += ws.gated[c * n_embd + d];
        }
        mixed[d] = sum / static_cast<float>(hc);
    }

    // Step 5: inject[c] = xn_source · w_inject^T[c]  (null w_inject → untouched)
    if (w_inject && inject) {
        for (int64_t c = 0; c < hc; ++c) {
            float dot = 0.0f;
            for (int64_t d = 0; d < total; ++d) {
                float xval = use_fp32 ? ws.xn[d] : bf16_to_f32(ws.xq[d]);
                float wval = bf16_to_f32(w_inject[c * total + d]);
                dot += xval * wval;
            }
            inject[c] = dot;
        }
    }
}

// ─── gr_write ──────────────────────────────────────────────────────
// build_hc_combine: scatter block output into residual streams.
//
//   R_out[c, d] = R[c, d] + block_out[d] * w[c]
//   w[c] = 2 * sigmoid(inject[c] / hc)
void gr_write(const float* R, const float* block_out,
              const float* inject, const GrShapes& shapes,
              float* R_out, cudaStream_t /*stream*/) {
    if (!R || !block_out || !inject || !R_out) return;
    if (!shapes.valid()) return;

    const int64_t n_embd = shapes.n_embd;
    const int64_t hc     = shapes.hc;

    for (int64_t c = 0; c < hc; ++c) {
        // w[c] = 2 * sigmoid(inject[c] / hc)
        const float w = 2.0f / (1.0f + std::exp(-inject[c] / static_cast<float>(hc)));

        for (int64_t d = 0; d < n_embd; ++d) {
            R_out[c * n_embd + d] = R[c * n_embd + d] + block_out[d] * w;
        }
    }
}

}  // namespace kernels
}  // namespace strata

