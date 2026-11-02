// GGUF v3 header reader primitives: ggml type geometry, tensor element counts,
// bounds-checked cursor, data-section alignment, array bounds, architecture guard.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace baloon::gguf {

// ---- block geometry --------------------------------------------------------

struct BlockGeometry {
    uint32_t elems;  // elements per block
    uint32_t bytes;  // bytes per block
    bool operator==(const BlockGeometry& o) const { return elems == o.elems && bytes == o.bytes; }
};

// Geometry for a supported ggml type id; nullopt for unsupported ids.
inline std::optional<BlockGeometry> block_geometry(uint32_t t) {
    switch (t) {
        case 0:  return BlockGeometry{1, 4};
        case 1:  return BlockGeometry{1, 2};
        case 30: return BlockGeometry{1, 2};
        case 2:  return BlockGeometry{32, 18};
        case 3:  return BlockGeometry{32, 20};
        case 6:  return BlockGeometry{32, 22};
        case 7:  return BlockGeometry{32, 24};
        case 8:  return BlockGeometry{32, 34};
        case 9:  return BlockGeometry{32, 36};
        case 10: return BlockGeometry{256, 84};
        case 11: return BlockGeometry{256, 110};
        case 12: return BlockGeometry{256, 144};
        case 13: return BlockGeometry{256, 176};
        case 14: return BlockGeometry{256, 210};
        case 16: return BlockGeometry{256, 66};
        case 17: return BlockGeometry{256, 74};
        case 18: return BlockGeometry{256, 98};
        case 20: return BlockGeometry{32, 18};
        case 21: return BlockGeometry{256, 110};
        case 22: return BlockGeometry{256, 82};
        case 23: return BlockGeometry{256, 136};
        case 29: return BlockGeometry{256, 56};
        case 42: return BlockGeometry{64, 18};  // Q2_0: fp16 scale + 16 code bytes
        default: return std::nullopt;
    }
}

// Bytes occupied by n elements of type t (n / elems * bytes, as in the model).
inline std::optional<uint64_t> row_bytes(uint32_t t, uint64_t n) {
    auto g = block_geometry(t);
    if (!g) return std::nullopt;
    return n / g->elems * g->bytes;
}

// ---- tensor info -----------------------------------------------------------

// Product of dimensions (empty shape -> 1; any zero dim -> 0).
inline uint64_t tensor_elements(const std::vector<uint64_t>& shape) {
    uint64_t n = 1;
    for (uint64_t d : shape) n *= d;
    return n;
}

// ---- alignment / data start ------------------------------------------------

// Round pos up to a multiple of align: (pos + align - 1) / align * align.
// align == 0 yields 0 (division by zero is 0 in the model); callers reject it.
inline uint64_t align_up(uint64_t pos, uint64_t align) {
    if (align == 0) return 0;
    return (pos + align - 1) / align * align;
}

// Data-section start; nullopt when align is 0 or the result exceeds the file size.
inline std::optional<uint64_t> data_start(uint64_t pos, uint64_t align, uint64_t file_size) {
    if (align == 0) return std::nullopt;
    uint64_t s = align_up(pos, align);
    if (s > file_size) return std::nullopt;
    return s;
}

// ---- bounds-checked cursor -------------------------------------------------

struct Cursor {
    uint64_t pos = 0;
    uint64_t size = 0;

    // Reading n bytes: nullopt if it would pass the end, else a cursor advanced by exactly n.
    std::optional<Cursor> read(uint64_t n) const {
        if (pos > size || n > size - pos) return std::nullopt;
        return Cursor{pos + n, size};
    }

    // In-place variant: returns false (and leaves *this untouched) on overrun.
    bool advance(uint64_t n) {
        auto c = read(n);
        if (!c) return false;
        *this = *c;
        return true;
    }
};

// ---- array bounds ----------------------------------------------------------

inline constexpr uint64_t kMaxArrayLen = uint64_t{1} << 24;
inline constexpr uint32_t kMaxArrayDepth = 2;

inline bool depth_ok(uint32_t d) { return d <= kMaxArrayDepth; }
inline bool array_len_ok(uint64_t count) { return count <= kMaxArrayLen; }

// ---- architecture guard ----------------------------------------------------

struct Guard {
    uint64_t block_count = 48;
    uint64_t hidden = 2560;
    uint64_t experts = 512;
    uint64_t experts_used = 10;
    uint64_t head_count = 24;
    uint64_t head_count_kv = 2;

    std::vector<std::pair<std::string, uint64_t>> reqs() const {
        return {{"qwen4exp.block_count", block_count},
                {"qwen4exp.embedding_length", hidden},
                {"qwen4exp.expert_count", experts},
                {"qwen4exp.expert_used_count", experts_used},
                {"qwen4exp.attention.head_count", head_count},
                {"qwen4exp.attention.head_count_kv", head_count_kv}};
    }
};

using MetadataLookup = std::function<std::optional<uint64_t>(const std::string&)>;

// Empty string iff the architecture is "qwen4exp" and every required key is present and equal.
inline std::string arch_guard(const Guard& g, const std::optional<std::string>& arch,
                              const MetadataLookup& md) {
    if (!arch) return "missing general.architecture";
    if (*arch != "qwen4exp") return "wrong architecture";
    for (const auto& r : g.reqs()) {
        auto v = md(r.first);
        if (!v || *v != r.second) return "metadata mismatch";
    }
    return "";
}

}  // namespace baloon::gguf

