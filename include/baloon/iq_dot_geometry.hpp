// iq_dot_geometry.hpp - geometry of the IQ-family dot-product / dequantizer / grouped native-expert kernels (ggml type
// ids 16,17,18,20,21,22,23,29,42,11): row byte sizes, 7-bit sign index expansion, per-format traits of the warp row
// loop, native expert blob layout (gate rows | up rows | down rows), grouped-expert scratch sizing and the dequantizer
// thread maps. Header only; every function is constexpr so the kernels can static_assert on it.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace iqdot {

constexpr bool supported(uint32_t t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11;
}

// Bytes of a weight row of n values; 0 for unsupported types.
constexpr uint64_t row_bytes(uint32_t t, uint64_t n) {
    switch (t) {
        case 16: return (n / 256) * 66;
        case 17: return (n / 256) * 74;
        case 18: return (n / 256) * 98;
        case 20: return (n / 32) * 18;
        case 21: return (n / 256) * 110;
        case 22: return (n / 256) * 82;
        case 29: return (n / 256) * 56;
        case 23: return (n / 256) * 136;
        case 11: return (n / 256) * 110;
        case 42: return (n / 64) * 18;
        default: return 0;
    }
}

// ---- sign expansion ---------------------------------------------------------------------------------------
constexpr uint32_t popc8(uint32_t v) {
    uint32_t c = 0;
    for (uint32_t i = 0; i < 8; ++i) c += (v >> i) & 1u;
    return c;
}
// Restore the 8th sign bit of a 7-bit sign index as the parity of the 7 bits.
constexpr uint32_t ksigns(uint32_t v) { return v ^ ((popc8(v) & 1u) << 7); }
constexpr uint32_t broadcast4(uint32_t s) { return s * 0x01010101u; }

// ---- warp call mapping ------------------------------------------------------------------------------------
struct Call { uint64_t block, part; };
constexpr Call decompose_call(uint64_t k, uint64_t ipb) { return Call{k / ipb, k % ipb}; }
constexpr uint64_t iqs_of(uint64_t k, uint64_t ipb, uint64_t step) { return step * (k % ipb); }

// ---- format traits ----------------------------------------------------------------------------------------
struct Traits { uint32_t qk, ipb, step; bool valid; };
constexpr Traits traits(uint32_t t) {
    switch (t) {
        case 16: case 17: case 18: case 21: case 22: return {256, 8, 2, true};
        case 20: return {32, 2, 2, true};
        case 29: return {256, 8, 1, true};
        case 42: return {64, 2, 1, true};
        default: return {0, 0, 0, false};
    }
}
constexpr bool traits_ok(uint32_t t) { return traits(t).valid && traits(t).ipb > 0 && traits(t).qk % 32 == 0; }

// ---- native expert blob layout ----------------------------------------------------------------------------
struct ExpertLayout { uint64_t gu_row, d_row, up_off, down_off, bytes; };
constexpr ExpertLayout expert_layout(uint32_t gu_type, uint32_t d_type, uint64_t n_embd, uint64_t n_ff) {
    return ExpertLayout{row_bytes(gu_type, n_embd), row_bytes(d_type, n_ff), n_ff * row_bytes(gu_type, n_embd),
                        2 * (n_ff * row_bytes(gu_type, n_embd)),
                        2 * (n_ff * row_bytes(gu_type, n_embd)) + n_embd * row_bytes(d_type, n_ff)};
}

constexpr uint64_t align256(uint64_t x) { return (x + 255) / 256 * 256; }

// Scratch: three float planes plus the q8_1 image of the intermediate.
constexpr uint64_t grouped_scratch_bytes(uint64_t cap, uint64_t n_ff) {
    return 3 * align256(cap * n_ff * 4) + align256(cap * (n_ff / 32) * 36);
}

// Single-token native expert (cap = 1) as carved by the production launcher: the q8_1 image of the input hidden vector
// (n_embd / 32 blocks of 36 bytes), then the three float planes (gate, up, h) and the q8_1 image of h, i.e. exactly
// grouped_scratch_bytes(1, n_ff). Nothing follows the last region.
constexpr uint64_t native_scratch_bytes(uint64_t n_embd, uint64_t n_ff) {
    return align256((n_embd / 32) * 36) + grouped_scratch_bytes(1, n_ff);
}

// ---- dequantizer thread maps ------------------------------------------------------------------------------
constexpr uint32_t dq8_offset(uint32_t tid) { return 32 * (tid % 8) + 8 * (tid / 8); }       // iq2_xxs / iq2_xs / iq2_s
constexpr uint32_t dq_q2_0_offset(uint32_t tid) { return (tid / 8) * 64 + (tid % 8) * 8; }   // == 8 * tid
constexpr uint32_t dq_iq4_nl_hi(uint32_t tid, uint32_t j) { return 4 * (tid / 8) + 16 + j; }
constexpr uint32_t gu_interleave(uint32_t r, uint32_t parity) { return 2 * r + parity; }

static_assert(row_bytes(16, 256) == 66 && row_bytes(42, 64) == 18 && row_bytes(99, 256) == 0, "row bytes");
static_assert(ksigns(0x7f) == 0xff && ksigns(1) == 0x81 && ksigns(3) == 3 && ksigns(0) == 0, "ksigns");

}  // namespace iqdot
}  // namespace baloon

