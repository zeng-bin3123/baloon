// iq_native_pack_index.hpp - integer model of the native pack for i-quant model files (experts.bin per-layer blob offsets,
// 64-byte aligned dense/extra tensors, exact F32->BF16 router narrowing) and of the MTP runtime blob (gate/up row interleave,
// code plane then scale plane, dense.bin padding to 256). Pure integer / byte logic; header only.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace baloon {
namespace iqpack {

constexpr uint64_t kExpertsPerLayer = 512;
constexpr uint64_t kAlign = 64;

constexpr uint64_t pad64(uint64_t n) { return (kAlign - n % kAlign) % kAlign; }
constexpr uint64_t round_row(uint64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

// Offset of layer l in experts.bin given per-layer blob sizes: sum of blob[k]*512 over earlier layers.
inline uint64_t experts_offset(const std::vector<uint64_t>& blob_per_layer, size_t layer) {
    uint64_t off = 0;
    for (size_t k = 0; k < layer && k < blob_per_layer.size(); ++k) off += blob_per_layer[k] * kExpertsPerLayer;
    return off;
}

// Layer count = 1 + max layer index present (0 when no layer index is present).
inline uint64_t layer_count(const std::vector<uint64_t>& layer_indices) {
    if (layer_indices.empty()) return 0;
    uint64_t m = 0;
    for (uint64_t l : layer_indices) if (l > m) m = l;
    return 1 + m;
}

// Index kind codes: BF16 -> 4, F16 -> 5, anything else (F32) -> 2.
inline int kind_of(const std::string& dtype) { return dtype == "BF16" ? 4 : dtype == "F16" ? 5 : 2; }

// Router narrowing F32 -> BF16 is allowed only when the low 16 bits of every word are zero (the high half then recovers the
// word exactly). Returns false and leaves *out untouched when any word would lose information.
inline bool narrow_router_f32_to_bf16(const uint32_t* words, size_t n, std::vector<uint16_t>* out) {
    for (size_t i = 0; i < n; ++i) if ((words[i] & 0xFFFFu) != 0) return false;
    out->resize(n);
    for (size_t i = 0; i < n; ++i) (*out)[i] = (uint16_t)(words[i] >> 16);
    return true;
}

// Offset of the extra tensor that follows the given tensors: sum of each length padded to 64.
inline uint64_t extra_offset(const std::vector<uint64_t>& lens) {
    uint64_t off = 0;
    for (uint64_t b : lens) off += b + pad64(b);
    return off;
}

}  // namespace iqpack

namespace mtprt {

constexpr uint64_t kHidden = 2560, kFF = 640, kExperts = 512;
constexpr uint64_t kBlob = 3 * (kHidden * kFF * 18 / 64);   // 1,382,400
static_assert(kBlob == 1382400, "MTP runtime expert blob size");
constexpr uint64_t kGuCodes = 2 * kFF * (kHidden / 64) * 16;   // 819200
constexpr uint64_t kDownCodes = kHidden * (kFF / 64) * 16;     // 409600
constexpr uint64_t kGuScales = 2 * kFF * (kHidden / 64) * 2;   // 102400
constexpr uint64_t kDownScales = kHidden * (kFF / 64) * 2;     // 51200
static_assert(kGuCodes + kDownCodes + kGuScales + kDownScales == kBlob, "planes tile the blob");
static_assert(kExperts * kBlob == 707788800ull, "expert file size");

constexpr uint64_t gate_slot(uint64_t r) { return 2 * r; }
constexpr uint64_t up_slot(uint64_t r) { return 2 * r + 1; }
constexpr uint64_t pad256(uint64_t n) { return (256 - n % 256) % 256; }

}  // namespace mtprt
}  // namespace baloon

