// expert_table.hpp - per-layer native expert table (native_experts.txt): contiguity validation of the pack, total size,
// running maximum blob size, GGUF-offset slots, the fixed-size default layout, and the AVX-512 gating predicates (the OS
// state mask in XCR0 and the CPUID leaf-7 bits) shared with the CPU feature probe.
//
// File format (text, '#' starts a comment):
//   native_experts v1|v2 <n_layers> <n_expert>
//   one line per layer:  <offset> <bytes_per_expert> <gu_type> <d_type>   (v2 adds: <gguf_gate> <gguf_up> <gguf_down>)
// Layer l starts where layer l-1 ended: offset[l] = sum over k < l of bytes[k] * n_expert. A gap or overlap is an error.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "baloon/kern_geometry.hpp"

namespace baloon {
namespace etable {

// ---- gating predicates -----------------------------------------------------------------------------------------
constexpr bool bit_set(uint64_t r, unsigned k) { return ((r >> k) & 1u) == 1u; }
// The OS saves SSE, AVX, opmask, ZMM_Hi256 and Hi16_ZMM state (XCR0 & 0xE6 == 0xE6).
constexpr bool xcr0_ok(uint64_t xcr0) { return (xcr0 & 0xE6u) == 0xE6u; }
// CPUID leaf 7: F, BW, VL in EBX bits 16, 30, 31; VNNI in ECX bit 11, VBMI in ECX bit 1.
constexpr bool avx512_ok(uint32_t ebx, uint32_t ecx) {
    return bit_set(ebx, 16) && bit_set(ebx, 30) && bit_set(ebx, 31) && bit_set(ecx, 11) && bit_set(ecx, 1);
}

// ---- GGUF offsets: three per layer in an array of 3 * n_layers --------------------------------------------------
constexpr uint64_t gguf_slot(uint64_t layer, uint64_t kind) { return 3 * layer + kind; }   // kind: 0 gate, 1 up, 2 down

// ---- default (no table) layout ----------------------------------------------------------------------------------
constexpr uint64_t default_total(uint64_t n_layers, uint64_t n_expert, uint64_t blob) { return n_layers * n_expert * blob; }

struct LayerEntry {
    uint64_t offset = 0;          // start of the layer inside the pack
    uint64_t bytes = 0;           // bytes of one expert blob of this layer
    int gu_type = 0, d_type = 0;  // ggml type ids
};

// Walk the layers: each offset must equal the running total. On success `total` = sum of bytes * n_expert.
bool contiguous(uint64_t n_expert, const std::vector<LayerEntry>& layers, uint64_t* total);
// Running maximum of the blob sizes (0 for no layers).
uint64_t max_blob(const std::vector<LayerEntry>& layers);

struct ExpertTable {
    bool v2 = false;
    uint64_t n_layers = 0, n_expert = 0, total = 0, max_blob = 0;
    std::vector<LayerEntry> layers;
    std::vector<uint64_t> gguf;   // 3 * n_layers entries (v2 only), indexed gguf_slot(l, kind)
    bool by_default = false;      // true when built by default_table()
};

// Parse + validate. On failure `err` names the problem and `out` is left default.
bool parse_expert_table(const std::string& text, ExpertTable& out, std::string& err);
// Read `path`; a missing file is not an error: `out` becomes default_table(...) and the function returns true.
bool load_expert_table(const std::string& path, uint64_t n_layers, uint64_t n_expert, ExpertTable& out, std::string& err);
// Fixed-size layout: every layer holds n_expert Q2_0 blobs (kern BLOB bytes each) back to back.
ExpertTable default_table(uint64_t n_layers, uint64_t n_expert);

}  // namespace etable
}  // namespace baloon
