// layer_compose.hpp - layer composition arithmetic: 16-byte arena carving, Q8_K sizes, quantized plane layout check,
// activation dispatch, block_layer_pre stage selection, RoPE row guard, dump stride.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace layer_compose {

constexpr uint64_t align_up16(uint64_t n) { return (n + 15) / 16 * 16; }
constexpr uint64_t q8k_bytes(uint64_t n) { return n / 256 * 292; }

struct PlaneSizes {
    uint64_t codes = 0, scales = 0, offset = 0, bytes = 0;
    constexpr bool ok() const { return codes != 0 && scales != 0 && codes + scales + offset == bytes; }
};

enum class Dispatch { Native, S2Q80, LegacyQ80, Q8kSplit };
constexpr Dispatch dispatch(bool native, uint64_t code_bits, bool wants_q8k) {
    return native ? Dispatch::Native
         : code_bits == 2 ? Dispatch::S2Q80
         : wants_q8k ? Dispatch::Q8kSplit
         : Dispatch::LegacyQ80;
}

// run_i of block_layer_pre: pfx mode when pfx>0, else by half.
constexpr bool run_stage(uint64_t i, uint64_t pfx, uint64_t half) {
    return pfx > 0 ? i < pfx : (i <= 2 ? (half == 0 || half == 1) : (half == 0 || half == 2));
}

constexpr bool rope_row_ok(int64_t pos_base, int64_t pos, int64_t max_cells) {
    return pos_base >= 0 && pos >= 0 && pos_base + pos < max_cells;
}

inline uint64_t aligned_total16(const std::vector<uint64_t>& l) {
    uint64_t t = 0;
    for (uint64_t b : l) t += align_up16(b);
    return t;
}
// Cursor::take sequence from `used`.
inline uint64_t take_all16(uint64_t used, const std::vector<uint64_t>& l) {
    for (uint64_t b : l) used = align_up16(used + b);
    return used;
}
inline uint64_t raw_sum(const std::vector<uint64_t>& l) {
    uint64_t t = 0;
    for (uint64_t b : l) t += b;
    return t;
}
// *_bytes = align16(sum) + 256 slack: covers per-take rounding for at most 17 takes.
inline uint64_t state_bytes(const std::vector<uint64_t>& l) { return align_up16(raw_sum(l)) + 256; }

constexpr uint64_t dump_stride(uint64_t n, uint64_t hc, uint64_t nhhd, uint64_t nvk) {
    return 2 * n + 2 * hc + nhhd + 5 * nvk + 8;
}
// Documented finding: the step[0..3] write (vs + nvk/2 + 4, len 4) overlaps the k_pool pfx at vs+8 when nvk>=2.
constexpr bool step_overlaps_kpool(uint64_t vs, uint64_t nvk) { return !(vs + nvk / 2 + 4 + 4 <= vs + 8); }

}  // namespace layer_compose
}  // namespace baloon
