// strata/core/layout.cpp — Per-layer weight shape and engine-form validation
// Target: Iluvatar BI-V100 (ivcore10)
//
// Implements check_layer and check_all from layout.hpp.
//
// GDN layers require: attn_qkv, ssm_alpha, ssm_beta, ssm_out, ssm_norm,
//   conv1d, dt — NO attn_q/k/v separately.
// QSA layers require: attn_q, attn_k, attn_v, attn_output, indexer.*
//   — NO attn_qkv (fused).
//
// The model has 36 GDN + 12 QSA = 48 layers.
// QSA layers are at indices {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}.

#include "strata/core/layout.hpp"
#include <cstdio>
#include <string>
#include <vector>

namespace strata {

namespace {

// ─── Want2: expected 2-D tensor shape ────────────────────────────────
struct Want2 {
    const char* suffix;
    int64_t ne0;       // contiguous axis
    int64_t ne1;       // outer axis
    bool qsa_only;     // skip if layer is GDN
    bool gdn_only;     // skip if layer is QSA
};

// ─── Want1: expected 1-D tensor (element count + kind) ──────────────
struct Want1 {
    const char* suffix;
    int64_t elements;
    WeightKind kind;
    bool qsa_only;
    bool gdn_only;
};

// Build a fully-qualified tensor name for error messages.
std::string tensor_name(int64_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

// Format a shape mismatch error.
bool shape_fail(std::string& err, int64_t layer, const char* suffix,
                const char* dim, int64_t actual, int64_t want) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s: %s = %lld, want %lld",
             tensor_name(layer, suffix).c_str(), dim,
             static_cast<long long>(actual), static_cast<long long>(want));
    err = buf;
    return false;
}

}  // namespace

bool check_layer(const WeightTable& table, const Geometry& g,
                 int64_t layer, std::string& err) {
    const bool qsa = is_qsa_layer(g, layer);

    const int64_t d = g.n_embd;
    const int64_t nh = g.n_head;
    const int64_t nkv = g.n_kv_head;
    const int64_t hd = (nh > 0) ? d / nh : 0;
    const int64_t hc_dim = g.hc;

    // SSM parameters (for GDN layers)
    // Conv channels C = 10240, value_dim V, v_heads, state_size S
    // These come from the Geometry; use reasonable defaults if not set.

    // ── 2-D shape checks ─────────────────────────────────────────────
    // Common tensors (both layer types)
    const Want2 want2_common[] = {
        {"ffn_gate.weight",  d, g.expert_hidden, false, false},
        {"ffn_up.weight",    d, g.expert_hidden, false, false},
        {"ffn_down.weight",  g.expert_hidden, d, false, false},
    };

    // QSA-specific tensors
    const Want2 want2_qsa[] = {
        {"attn_q.weight",      d, nh * hd,        true, false},
        {"attn_k.weight",      d, nkv * hd,       true, false},
        {"attn_v.weight",      d, nkv * hd,       true, false},
        {"attn_output.weight", nh * hd, d,         true, false},
    };

    // GDN-specific tensors
    const Want2 want2_gdn[] = {
        {"attn_qkv.weight",    d, hc_dim,         false, true},
        {"ssm_out.weight",     hc_dim, d,         false, true},
    };

    // Check 2-D shapes
    auto check_2d = [&](const Want2* list, size_t count) -> bool {
        for (size_t i = 0; i < count; ++i) {
            const Want2& w = list[i];
            if (w.qsa_only && !qsa) continue;
            if (w.gdn_only && qsa) continue;

            std::string name = tensor_name(layer, w.suffix);
            const WeightRef* ref = table.find(name);
            if (ref == nullptr) {
                err = "missing: " + name;
                return false;
            }
            if (ref->ne0 != w.ne0) {
                return shape_fail(err, layer, w.suffix, "ne0", ref->ne0, w.ne0);
            }
            if (ref->ne1 != w.ne1) {
                return shape_fail(err, layer, w.suffix, "ne1", ref->ne1, w.ne1);
            }
        }
        return true;
    };

    if (!check_2d(want2_common, sizeof(want2_common) / sizeof(want2_common[0]))) return false;
    if (!check_2d(want2_qsa, sizeof(want2_qsa) / sizeof(want2_qsa[0]))) return false;
    if (!check_2d(want2_gdn, sizeof(want2_gdn) / sizeof(want2_gdn[0]))) return false;

    // ── 1-D kind checks ──────────────────────────────────────────────
    // Bf16InF32 tensors store 2 bytes/element; casting to float* reads 2× length.
    const Want1 want1_gdn[] = {
        {"ssm_norm.weight", d, WeightKind::F32, false, true},
    };

    for (const auto& w : want1_gdn) {
        if (w.qsa_only && !qsa) continue;
        if (w.gdn_only && qsa) continue;

        std::string name = tensor_name(layer, w.suffix);
        const WeightRef* ref = table.find(name);
        if (ref == nullptr) {
            err = "missing: " + name;
            return false;
        }
        if (ref->n_elements != w.elements && w.elements > 0) {
            char buf[256];
            snprintf(buf, sizeof(buf), "%s: n_elements = %lld, want %lld",
                     name.c_str(), static_cast<long long>(ref->n_elements),
                     static_cast<long long>(w.elements));
            err = buf;
            return false;
        }
        if (ref->kind != w.kind) {
            char buf[256];
            snprintf(buf, sizeof(buf), "%s: kind = %d, want %d",
                     name.c_str(), static_cast<int>(ref->kind),
                     static_cast<int>(w.kind));
            err = buf;
            return false;
        }
    }

    // ── Plane layout consistency ─────────────────────────────────────
    // For quantized tensors: codes_bytes + scales_bytes + offset_bytes == total_bytes
    for (const auto& [name, ref] : table.all()) {
        if (ref.quantized() && ref.codes_bytes > 0) {
            uint64_t sum = ref.codes_bytes + ref.scales_bytes + ref.offset_bytes;
            if (sum != ref.total_bytes && ref.total_bytes > 0) {
                char buf[384];
                snprintf(buf, sizeof(buf),
                         "%s: plane sum (%llu + %llu + %llu = %llu) != total_bytes (%llu)",
                         name.c_str(),
                         static_cast<unsigned long long>(ref.codes_bytes),
                         static_cast<unsigned long long>(ref.scales_bytes),
                         static_cast<unsigned long long>(ref.offset_bytes),
                         static_cast<unsigned long long>(sum),
                         static_cast<unsigned long long>(ref.total_bytes));
                err = buf;
                return false;
            }
        }
    }

    return true;
}

bool check_all(const WeightTable& table, const Geometry& g,
               std::string& err) {
    if (g.n_layers <= 0) {
        err = "check_all: n_layers must be positive";
        return false;
    }

    int64_t n_qsa = 0, n_gdn = 0;

    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (!check_layer(table, g, l, err)) {
            return false;
        }
        if (is_qsa_layer(g, l)) {
            ++n_qsa;
        } else {
            ++n_gdn;
        }
    }

    // Validate layer-type split
    // The model should have 12 QSA + 36 GDN = 48 layers
    if (g.n_layers == 48) {
        if (n_qsa != 12 || n_gdn != 36) {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "layer split mismatch: %lld QSA + %lld GDN (want 12 + 36)",
                     static_cast<long long>(n_qsa), static_cast<long long>(n_gdn));
            err = buf;
            return false;
        }
    }

    return true;
}

}  // namespace strata

