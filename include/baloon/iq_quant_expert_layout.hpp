// iq_quant_expert_layout.hpp - index/layout arithmetic of the IQ-quant expert kernels: lane-strided call
// decomposition, native expert blob layout (gate | up | down), 256-byte scratch alignment, gate/up interleave,
// Q2_0 code decoding.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace iq_layout {

struct Call { uint64_t block_index, intra_block_offset; };
constexpr Call decompose_call(uint64_t k, uint64_t calls_per_block) {
    return Call{k / calls_per_block, k % calls_per_block};
}

// Calls visited by `lane` (stride 32) below nb*ipb.
inline std::vector<uint64_t> lane_calls(uint64_t lane, uint64_t nb, uint64_t ipb) {
    std::vector<uint64_t> v;
    for (uint64_t k = lane; k < nb * ipb; k += 32) v.push_back(k);
    return v;
}

struct Layout {
    uint64_t n_embd, n_ff, gate_up_row_bytes, down_row_bytes;
    constexpr uint64_t up_offset() const { return n_ff * gate_up_row_bytes; }
    constexpr uint64_t down_offset() const { return 2 * up_offset(); }
    constexpr uint64_t bytes() const { return down_offset() + n_embd * down_row_bytes; }
};

constexpr uint64_t align256(uint64_t x) { return (x + 255) / 256 * 256; }

// Interleaved row of (row, parity).
constexpr uint64_t gu_row(uint64_t r, uint64_t parity) { return 2 * r + parity; }

// Q2_0: a 2-bit code c decodes to c - 1.
constexpr int q2_0_decode(uint32_t c) { return (int)c - 1; }

constexpr uint64_t calls_per_block_iq256 = 256 / 32;  // 8
constexpr uint64_t calls_per_block_iq32 = 32 / 16;    // 2

}  // namespace iq_layout
}  // namespace baloon
