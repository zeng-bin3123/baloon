// gguf_header_geometry.hpp - GGUF v3 header facts: magic bytes, fixed header sizes, element counts, expected byte counts from block
// geometry, the aligned start of the data section, and the ggml type-name table (ids 31 and 41 share a name but stay distinct).
#pragma once
#include <cstdint>
#include <optional>
#include <vector>

namespace baloon {
namespace gguf_geom {

constexpr uint32_t kMagic = 0x46554747u;
constexpr uint64_t kHeaderBytes = 4 + (4 + 8 + 8);   // magic + version + n_tensors + n_kv
constexpr uint64_t kTensorInfoTail = 4 + 8;          // type + offset
static_assert(kHeaderBytes == 24 && kTensorInfoTail == 12, "fixed header sizes");
static_assert((kMagic & 255) == 71 && ((kMagic >> 8) & 255) == 71 && ((kMagic >> 16) & 255) == 85 && (kMagic >> 24) == 70, "G G U F");

inline uint64_t shape_elems(const std::vector<uint64_t>& shape) { uint64_t n = 1; for (auto d : shape) n *= d; return n; }

// Whole blocks only; a partial block has no byte count.
inline std::optional<uint64_t> expected_bytes(uint64_t elems, uint64_t block_elems, uint64_t block_bytes) {
    if (block_elems == 0 || elems % block_elems != 0) return std::nullopt;
    return elems / block_elems * block_bytes;
}

// Round pos up to a multiple of a (a > 0): the data_start formula.
constexpr uint64_t align_up(uint64_t pos, uint64_t a) { return (pos + a - 1) / a * a; }

inline const char* type_name(uint32_t id) {
    switch (id) {
        case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0"; case 8: return "Q8_0"; case 30: return "BF16";
        case 31: return "Q4_0_8_8"; case 41: return "Q4_0_8_8"; case 42: return "Q2_0";
        default: return "unknown";
    }
}

}  // namespace gguf_geom
}  // namespace baloon
