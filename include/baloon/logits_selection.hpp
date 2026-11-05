// logits_selection.hpp - which logits rows of a generation run get dumped when the dump is thinned by a stride.
// Rows are numbered by position 0 .. total-1 (total = n_prompt - 1 + max_new). With stride s, row r is selected iff
// r % s == 0, so row 0 is always kept and stride 1 selects everything. row_count(total, s) is the number of selected
// rows, i.e. ceil(total / s). parse_stride reads the command-line value: a plain decimal integer in [1, 2^32-1].
#pragma once
#include <cstdint>
#include <string_view>

namespace baloon {
namespace logits {

constexpr uint64_t kMaxStride = 0xFFFFFFFFull;

// Strict parse: non-empty, digits only (no sign, space or suffix), no overflow past kMaxStride, value >= 1.
// Returns false and leaves *out untouched on any violation.
inline bool parse_stride(std::string_view s, uint64_t* out) {
    if (s.empty() || !out) return false;
    uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (uint64_t)(c - '0');
        if (v > kMaxStride) return false;
    }
    if (v == 0) return false;
    *out = v;
    return true;
}

// Number of rows in [0, total) that selected() accepts; 0 for an invalid stride.
inline uint64_t row_count(uint64_t total, uint64_t stride) {
    if (stride == 0) return 0;
    return total / stride + (total % stride != 0 ? 1 : 0);
}

// Whether row `row` is dumped under `stride`; nothing is selected for an invalid stride.
inline bool selected(uint64_t row, uint64_t stride) {
    return stride != 0 && row % stride == 0;
}

}  // namespace logits
}  // namespace baloon
