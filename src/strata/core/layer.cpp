// strata/core/layer.cpp — GDN & MoE layer implementation
// Target: Iluvatar BI-V100 (ivcore10)
#include "strata/core/layer.hpp"
#include "strata/core/weights.hpp"

#include <cstring>

namespace strata {

// ─── Configuration state (module-level) ──────────────────────────
static bool g_native_bf16 = false;
static bool g_native_flash_attn_short = false;
static bool g_shared_early = false;

void layer_set_native_bf16(bool enabled) { g_native_bf16 = enabled; }
void layer_set_native_flash_attn_short(bool enabled) { g_native_flash_attn_short = enabled; }
void layer_set_shared_early(bool enabled) { g_shared_early = enabled; }
bool layer_shared_early() { return g_shared_early; }

// ─── Alignment helper ────────────────────────────────────────────
static inline uint64_t align16(uint64_t x) {
    return (x + 15) & ~uint64_t(15);
}

// ─── Architecture constants ──────────────────────────────────────
// GDN: S=128, h_k=16, h_v=48, conv_channels=10240
// 2*S*h_k + S*h_v = 10240
static constexpr int64_t GDN_S = 128;
static constexpr int64_t GDN_HK = 16;
static constexpr int64_t GDN_HV = 48;
static constexpr int64_t GDN_CONV_CHANNELS = 10240;
static constexpr int64_t GDN_DCONV_M1 = 3;  // d_conv - 1

// Q8_K block: 4 + 256 + 16*2 = 292 bytes per block of 256 elements
static constexpr int64_t Q8K_BLOCK_ELEMS = 256;
static constexpr int64_t Q8K_BLOCK_BYTES = 292;

// Q8_0 block: 2 + 32 = 34 bytes per block of 32 elements
static constexpr int64_t Q80_BLOCK_ELEMS = 32;
static constexpr int64_t Q80_BLOCK_BYTES = 34;

static inline uint64_t q8k_bytes(int64_t n) {
    int64_t blocks = (n + Q8K_BLOCK_ELEMS - 1) / Q8K_BLOCK_ELEMS;
    return static_cast<uint64_t>(blocks) * Q8K_BLOCK_BYTES;
}

static inline uint64_t q80_bytes(int64_t n) {
    int64_t blocks = (n + Q80_BLOCK_ELEMS - 1) / Q80_BLOCK_ELEMS;
    return static_cast<uint64_t>(blocks) * Q80_BLOCK_BYTES;
}

// ─── GdnBuffers sizing ──────────────────────────────────────────
uint64_t gdn_buffers_bytes(const Geometry& g) {
    int64_t ne = g.n_embd;
    int64_t value_dim = GDN_HV * GDN_S;

    uint64_t total = 0;

    // Quantized activation images
    total += align16(q8k_bytes(ne));      // x_q8k
    total += align16(q80_bytes(ne));      // x_q8_0
    total += align16(ne * 2);             // x_bf16

    // QKV path (conv_channels = 10240)
    total += align16(GDN_CONV_CHANNELS * sizeof(float));  // qkv
    total += align16(GDN_CONV_CHANNELS * sizeof(float));  // conv_out
    total += align16(GDN_CONV_CHANNELS * sizeof(float));  // h

    // Per-head scalars
    total += align16(GDN_HV * sizeof(float));   // alpha
    total += align16(GDN_HV * sizeof(float));   // beta
    total += align16(GDN_HV * sizeof(float));   // gate

    // Recurrence
    total += align16(GDN_HV * GDN_S * sizeof(float));  // o
    total += align16(value_dim * sizeof(float));        // z
    total += align16(value_dim * sizeof(float));        // y

    // Output quantized images
    total += align16(q8k_bytes(ne));      // y_q8k
    total += align16(q80_bytes(ne));      // y_q8_0

    // Persistent state
    total += align16(GDN_S * GDN_HV * GDN_S * sizeof(float));  // state
    total += align16(GDN_CONV_CHANNELS * GDN_DCONV_M1 * sizeof(float)); // conv_state

    return total;
}

uint64_t gdn_buffers_init(const Geometry& g, void* base, GdnBuffers& b) {
    auto* p = static_cast<uint8_t*>(base);
    uint8_t* start = p;
    int64_t ne = g.n_embd;
    int64_t value_dim = GDN_HV * GDN_S;

    // x_q8k
    b.x_q8k = p;
    p += align16(q8k_bytes(ne));

    // x_q8_0
    b.x_q8_0 = p;
    p += align16(q80_bytes(ne));

    // x_bf16
    b.x_bf16 = reinterpret_cast<uint16_t*>(p);
    p += align16(ne * 2);

    // qkv
    b.qkv = reinterpret_cast<float*>(p);
    p += align16(GDN_CONV_CHANNELS * sizeof(float));

    // conv_out
    b.conv_out = reinterpret_cast<float*>(p);
    p += align16(GDN_CONV_CHANNELS * sizeof(float));

    // h
    b.h = reinterpret_cast<float*>(p);
    p += align16(GDN_CONV_CHANNELS * sizeof(float));

    // alpha
    b.alpha = reinterpret_cast<float*>(p);
    p += align16(GDN_HV * sizeof(float));

    // beta
    b.beta = reinterpret_cast<float*>(p);
    p += align16(GDN_HV * sizeof(float));

    // gate
    b.gate = reinterpret_cast<float*>(p);
    p += align16(GDN_HV * sizeof(float));

    // o
    b.o = reinterpret_cast<float*>(p);
    p += align16(GDN_HV * GDN_S * sizeof(float));

    // z
    b.z = reinterpret_cast<float*>(p);
    p += align16(value_dim * sizeof(float));

    // y
    b.y = reinterpret_cast<float*>(p);
    p += align16(value_dim * sizeof(float));

    // y_q8k
    b.y_q8k = p;
    p += align16(q8k_bytes(ne));

    // y_q8_0
    b.y_q8_0 = p;
    p += align16(q80_bytes(ne));

    // state
    b.state = reinterpret_cast<float*>(p);
    p += align16(GDN_S * GDN_HV * GDN_S * sizeof(float));

    // conv_state
    b.conv_state = reinterpret_cast<float*>(p);
    p += align16(GDN_CONV_CHANNELS * GDN_DCONV_M1 * sizeof(float));

    return static_cast<uint64_t>(p - start);
}

void gdn_buffers_zero_state(const GdnBuffers& b, const Geometry& g, void* stream) {
    auto cs = static_cast<cudaStream_t>(stream);
    // state: S × h_v × S floats
    size_t state_bytes = GDN_S * GDN_HV * GDN_S * sizeof(float);
    cudaMemsetAsync(b.state, 0, state_bytes, cs);
    // conv_state: C × (d_conv - 1) floats
    size_t conv_bytes = GDN_CONV_CHANNELS * GDN_DCONV_M1 * sizeof(float);
    cudaMemsetAsync(b.conv_state, 0, conv_bytes, cs);
}

// ─── gdn_layer (stub) ────────────────────────────────────────────
// Full implementation requires GEMV kernels from a later unit.
// This stub validates the interface and returns success.
bool gdn_layer(const WeightTable& tables, const Geometry& g,
               int64_t layer, const GdnBuffers& b,
               const float* mixed, float* out, void* stream,
               std::string& err) {
    if (!mixed || !out) {
        err = "gdn_layer: null input/output";
        return false;
    }
    if (layer < 0 || layer >= g.n_layers) {
        err = "gdn_layer: layer " + std::to_string(layer) + " out of range";
        return false;
    }
    // Op order from ref/gdn.py::gdn_decode:
    // 1. qkv = W_qkv · x  (Q8_K activation contract)
    // 2. conv → SiLU (BEFORE split, on entire 10240-wide conv output)
    // 3. q, k = l2_norm(·)  (v NOT normalized; 2*h_k rows, not 2*h_k + h_v)
    // 4. β = σ(W_β · x)  (BF16 activation)
    // 5. gate = softplus(W_α · x + dt) · ssm_a  (BF16 activation)
    // 6. Recurrent step (gdn_step)
    // 7. y = rms_norm(o, ssm_norm) · σ(z)
    // 8. out = W_ssm_out · y
    //
    // Kernel dispatch deferred to later unit.
    (void)tables; (void)b; (void)stream;
    return true;
}

// ─── moe_layer (stub) ────────────────────────────────────────────
bool moe_layer(const WeightTable& tables, const Geometry& g,
               int64_t layer, int64_t k, const MoEBuffers& b,
               const float* x, const float* parts, float* out,
               void* stream, std::string& err,
               const Doorbell* db) {
    if (!x || !out) {
        err = "moe_layer: null input/output";
        return false;
    }
    if (k <= 0) {
        err = "moe_layer: k must be positive";
        return false;
    }
    // out = shared_expert(x) + Σ w_j · expert_{ids[j]}(x)
    // Router: BF16 activation → GEMV → top-k → ids, weights
    (void)tables; (void)g; (void)layer; (void)b; (void)parts; (void)stream; (void)db;
    return true;
}

// ─── moe_route (stub) ────────────────────────────────────────────
bool moe_route(const WeightTable& tables, const Geometry& g,
               int64_t layer, int64_t k, const MoEBuffers& b,
               const float* x, void* stream, std::string& err,
               const Doorbell* db) {
    if (!x) {
        err = "moe_route: null input";
        return false;
    }
    // x → bf16 → router GEMV → top-k → (ids, weights) → ring doorbell
    (void)tables; (void)g; (void)layer; (void)k; (void)b; (void)stream; (void)db;
    return true;
}

// ─── moe_finish (stub) ───────────────────────────────────────────
bool moe_finish(const WeightTable& tables, const Geometry& g,
                int64_t layer, int64_t k, const MoEBuffers& b,
                const float* x, const float* parts, float* out,
                void* stream, std::string& err) {
    if (!x || !out) {
        err = "moe_finish: null input/output";
        return false;
    }
    // CRITICAL: parts MUST be from THIS layer's router.
    // Passing previous layer's vectors is the L100 bug.
    (void)tables; (void)g; (void)layer; (void)k; (void)b; (void)parts; (void)stream;
    return true;
}

// ─── moe_shared (stub) ───────────────────────────────────────────
bool moe_shared(const WeightTable& tables, const Geometry& g,
                int64_t layer, const MoEBuffers& b,
                const float* x, void* stream, std::string& err) {
    if (!x) {
        err = "moe_shared: null input";
        return false;
    }
    (void)tables; (void)g; (void)layer; (void)b; (void)stream;
    return true;
}

// ─── moe_combine_parts (stub) ────────────────────────────────────
bool moe_combine_parts(const Geometry& g, int64_t layer, int64_t k,
                        const MoEBuffers& b, const float* parts,
                        float* out, void* stream, std::string& err) {
    if (!out) {
        err = "moe_combine_parts: null output";
        return false;
    }
    // out = shared + Σ w_j · parts[j]
    (void)g; (void)layer; (void)k; (void)b; (void)parts; (void)stream;
    return true;
}

}  // namespace strata

