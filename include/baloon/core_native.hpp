// core_native.hpp - native dense projection and embedding bindings (host model) over a WeightTable. Engine form 1 is raw
// BF16 (2 bytes/element); a binding checks the row exists, is BF16 and its byte count equals the shape.
#pragma once
#include <cstdint>
#include <string>

#include "baloon/core_layer.hpp"
#include "baloon/core_weights.hpp"

namespace baloon {
namespace core {

struct NativeDense {
    const uint16_t* w = nullptr;
    int64_t rows = 0, cols = 0;
    bool valid() const { return w != nullptr && rows > 0 && cols > 0; }
    bool bind(const WeightTable& t, const std::string& name, const uint8_t* arena, int64_t r, int64_t c) {
        *this = NativeDense{};
        const WIndexRow* row = t.find(name);
        if (!row || !row->placed || row->engine_kind != 1 || r <= 0 || c <= 0) return false;
        if (row->dst_bytes != (uint64_t)r * (uint64_t)c * 2) return false;
        w = reinterpret_cast<const uint16_t*>(arena + row->arena_off); rows = r; cols = c;
        return true;
    }
    bool apply(const float* x, float* out) const { if (!valid()) return false; project_bf16(w, x, rows, cols, out); return true; }
};

struct NativeEmbed {
    const float*    table   = nullptr;  // float32 embedding table (legacy path; first three members kept for compat)
    int64_t vocab = 0, n = 0;
    const uint16_t* w_bf16 = nullptr;  // BF16 embedding table (engine_kind=1); set by bind(); preferred in row()
    bool valid() const { return (w_bf16 || table) && vocab > 0 && n > 0; }
    // Bind BF16 embedding weight (engine_kind==1) from the weight arena.
    bool bind(const WeightTable& t, const std::string& name, const uint8_t* arena, int64_t v, int64_t hidden) {
        *this = NativeEmbed{};
        const WIndexRow* r = t.find(name);
        if (!r || !r->placed || r->engine_kind != 1 || v <= 0 || hidden <= 0) return false;
        if (r->dst_bytes != (uint64_t)v * (uint64_t)hidden * 2) return false;
        w_bf16 = reinterpret_cast<const uint16_t*>(arena + r->arena_off);
        vocab = v; n = hidden;
        return true;
    }
    bool row(int64_t token, float* out) const {
        if (w_bf16) return embed_row_bf16(w_bf16, vocab, n, token, out);
        return table ? embed_row(table, vocab, n, token, out) : false;
    }
};

// Process-wide embedding binding used by the token path; nullptr clears it.
inline const NativeEmbed*& native_embed_slot() { static const NativeEmbed* p = nullptr; return p; }
inline const NativeEmbed* native_embed() { return native_embed_slot(); }
inline void set_native_embed(const NativeEmbed* e) { native_embed_slot() = e; }

}  // namespace core
}  // namespace baloon
