// cpu_expert_layout.hpp - layout of one routed expert stored in the GGUF block formats (Q2_0 or an i-quant) and the
// runtime-dispatched row/activation entry points used by the CPU expert path.
//
// Blob layout (one expert, one quant type): gate rows [n_ff][n_embd] | up rows [n_ff][n_embd] | down rows [n_embd][n_ff].
//   up_off = n_ff * gu_row_bytes, down_off = 2 * up_off, bytes = down_off + n_embd * down_row_bytes.
// Types: Q2_0 = ggml id 42 (64 values / 18 bytes), IQ2_XXS 16 (66 B), IQ2_XS 17 (74 B), IQ3_XXS 18 (98 B),
//        IQ3_S 21 (110 B), IQ2_S 22 (82 B); the i-quants hold 256 values per block.
#pragma once
#include <cstddef>
#include <cstdint>

#include "baloon/kern_cpu_expert.hpp"

namespace baloon {
namespace expert_layout_ns {

constexpr int kTypeQ2_0 = 42;

// Values and bytes per block of a supported type; 0 for an unsupported one.
constexpr int64_t block_values(int type) {
    return type == kTypeQ2_0 ? 64 : (type == 16 || type == 17 || type == 18 || type == 21 || type == 22) ? 256 : 0;
}
constexpr int64_t block_bytes(int type) {
    return type == kTypeQ2_0 ? 18 : type == 16 ? 66 : type == 17 ? 74 : type == 18 ? 98 : type == 21 ? 110 : type == 22 ? 82 : 0;
}
// Bytes of a row of n_in values; 0 when the type is unsupported or n_in is not a positive whole number of blocks.
constexpr int64_t row_bytes(int type, int64_t n_in) {
    return (block_values(type) == 0 || n_in <= 0 || n_in % block_values(type)) ? 0 : n_in / block_values(type) * block_bytes(type);
}

}  // namespace expert_layout_ns

struct ExpertLayout {
    int type = 0;
    int64_t n_embd = 0, n_ff = 0;
    int64_t gu_row_bytes = 0, down_row_bytes = 0;
    int64_t up_off = 0, down_off = 0, bytes = 0;
    bool ok = false;
};

// Layout of an expert of `type` with hidden width n_embd and FFN width n_ff. ok == false (everything else zero) when the
// type is unsupported or a width is not a whole number of blocks.
inline ExpertLayout expert_layout(int type, int64_t n_embd, int64_t n_ff) {
    ExpertLayout l;
    const int64_t gu = expert_layout_ns::row_bytes(type, n_embd), dn = expert_layout_ns::row_bytes(type, n_ff);
    if (gu == 0 || dn == 0) return l;
    l.type = type; l.n_embd = n_embd; l.n_ff = n_ff;
    l.gu_row_bytes = gu; l.down_row_bytes = dn;
    l.up_off = n_ff * gu;
    l.down_off = 2 * l.up_off;
    l.bytes = l.down_off + n_embd * dn;
    l.ok = true;
    return l;
}

// Validates a stored expert blob of `blob_bytes` bytes: succeeds only when the layout is valid and the blob has exactly
// the size the layout tiles (no slack, no truncation). On failure `out` is a default (ok == false) layout.
inline bool expert_layout_load(ExpertLayout& out, int type, int64_t n_embd, int64_t n_ff, uint64_t blob_bytes) {
    const ExpertLayout l = expert_layout(type, n_embd, n_ff);
    if (!l.ok || blob_bytes != (uint64_t)l.bytes) { out = ExpertLayout{}; return false; }
    out = l;
    return true;
}

// The AVX-512 feature set the expert kernels need (F, BW, VL, VNNI, VBMI) is present.
inline bool cpu_avx512_ok() { return kern::cpu_features().usable(); }

// Activation quantizer (per-32 scale, int8 in [-127, 127], chunk sums) on any CPU: AVX2 when available, scalar otherwise;
// both forms are bit-identical. n % 32 == 0.
inline void act_quant_any(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx) {
    if (kern::cpu_features().avx2) kern::act_quant_q8_1_avx2(x, n, q, d, hx);
    else kern::act_quant_q8_1(x, n, q, d, hx);
}

// Rows [row0, row1) of a GGUF-layout Q2_0 matrix against nt quantized activations, on whatever the CPU offers.
// Returns false (writing nothing) when the CPU has neither AVX-512 nor AVX2.
inline bool q2_rows_any(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                        size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out, size_t out_stride) {
    const kern::CpuFeatures f = kern::cpu_features();
    if (f.usable()) {
        kern::q2_0_gguf_rows_multi(w, row0, row1, nb, nt, q, q_stride, xd, hx, ch_stride, out, out_stride);
        return true;
    }
    if (f.avx2) {
        kern::q2_0_gguf_rows_multi_avx2(w, row0, row1, nb, nt, q, q_stride, xd, hx, ch_stride, out, out_stride);
        return true;
    }
    return false;
}

}  // namespace baloon
