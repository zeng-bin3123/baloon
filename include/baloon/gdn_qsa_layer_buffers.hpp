// gdn_qsa_layer_buffers.hpp - buffer carving, quantized-plane layout, GEMV dispatch and dump layout of the
// GDN / QSA / MoE layer. Every carved region is 16-byte aligned; *_bytes and *_init walk the same list.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace baloon {
namespace layer_buffers {

constexpr uint64_t kAlign = 16;
constexpr uint64_t kQ8kBlockElems = 256;
constexpr uint64_t kQ8kBlockBytes = 292;
constexpr int kLayers = 48;

inline uint64_t align16(uint64_t n) { return (n + 15) / 16 * 16; }
// Q8_K activation buffer: n/256 blocks of 292 bytes.
inline uint64_t q8k_bytes(uint64_t n) { return n / kQ8kBlockElems * kQ8kBlockBytes; }

// End offset after carving `parts` starting at `acc`.
inline uint64_t carve_end(uint64_t acc, const std::vector<uint64_t>& parts) {
    for (uint64_t p : parts) acc += align16(p);
    return acc;
}
// Offset of the i-th carved region (beyond the end it saturates at the carved total).
inline uint64_t off16(const std::vector<uint64_t>& l, size_t i) {
    uint64_t o = 0;
    for (size_t j = 0; j < i && j < l.size(); ++j) o += align16(l[j]);
    return o;
}

// Planes of a quantized tensor: codes | scales | offsets.
struct Planes { uint64_t codes, scales, offs; };
inline bool planes_tile(const Planes& p, uint64_t bytes) { return p.codes + p.scales + p.offs == bytes; }
inline uint64_t scales_off(const Planes& p) { return p.codes; }
inline uint64_t offs_off(const Planes& p) { return p.codes + p.scales; }

enum class Kind { native, s2, q8_0, q8k };
inline Kind kernel_of(bool native, int bits, bool wants_q8k) {
    if (native) return Kind::native;
    if (bits == 2) return Kind::s2;
    return wants_q8k ? Kind::q8k : Kind::q8_0;
}

// Layer kinds: layer l is QSA iff l % 4 == 3.
inline bool is_qsa_layer(int l) { return l % 4 == 3; }
inline int count_qsa(int nlayers) { int c = 0; for (int l = 0; l < nlayers; ++l) c += is_qsa_layer(l); return c; }
inline int count_gdn(int nlayers) { return nlayers - count_qsa(nlayers); }

// q/gate split: row h of the [nh x 2*hd] buffer holds q in [0,hd) and gate in [hd,2hd).
inline uint64_t qgate_row_off(uint64_t h, uint64_t hd) { return h * (2 * hd); }
inline uint64_t qgate_gate_off(uint64_t h, uint64_t hd) { return h * (2 * hd) + hd; }
inline uint64_t qgate_total(uint64_t nh, uint64_t hd) { return nh * (2 * hd); }
// RoPE table check: absolute position must lie inside the table.
inline bool rope_pos_ok(uint64_t pos_base, uint64_t pos, uint64_t max_cells) { return pos_base + pos < max_cells; }

// Half-level dump: floats per layer.
inline uint64_t dump_stride(uint64_t n, uint64_t hc, uint64_t nhhd, uint64_t nvk) {
    return 2 * n + 2 * hc + nhhd + 5 * nvk + 8;
}
inline uint64_t dump_slot_base(uint64_t n, uint64_t hc, uint64_t nhhd, uint64_t nvk) {
    return 2 * n + 2 * hc + nhhd + 2 * nvk + 8;
}
inline uint64_t dump_kcur_off(uint64_t n, uint64_t hc, uint64_t nhhd, uint64_t nvk) {
    return dump_slot_base(n, hc, nhhd, nvk) + 2 * nvk;   // kcur starts at base + 2*nvk, length nvk
}
inline uint64_t dump_layer_off(uint64_t layer, uint64_t stride) { return layer * stride; }
inline uint64_t dump_gather_end(uint64_t n, uint64_t hc, uint64_t nhhd, uint64_t nvk) {
    return (2 * n + 2 * hc + nhhd + 2 * nvk) + nvk / 2 + 4 + 4;
}

inline bool verify_compatible(bool bf16, bool gr, bool gdn, bool attn, bool sel, bool idx) {
    return bf16 && gr && gdn && attn && sel && idx;
}

}  // namespace layer_buffers
}  // namespace baloon
