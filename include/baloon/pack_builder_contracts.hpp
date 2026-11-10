// pack_builder_contracts.hpp - pack-building arithmetic: LSB-first b-bit code packing, 64-byte plane alignment, arena addressing,
// scale plane sizes, the iq-pack arena / index accounting, router narrowing, and the MTP quantiser grid and byte counts.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace baloon {
namespace packb {

// ---- LSB-first packing: codes of `bits` bits (1,2,4,8), 8/bits per byte ----
inline std::vector<uint8_t> pack_lsb(unsigned bits, const std::vector<uint8_t>& codes) {
    const unsigned per = 8 / bits;
    std::vector<uint8_t> out((codes.size() + per - 1) / per, 0);
    for (size_t i = 0; i < codes.size(); ++i) out[i / per] |= (uint8_t)(codes[i] << (bits * (i % per)));
    return out;
}
inline std::vector<uint8_t> unpack_lsb(unsigned bits, const std::vector<uint8_t>& bytes, size_t n) {
    const unsigned per = 8 / bits;
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = (uint8_t)((bytes[i / per] >> (bits * (i % per))) & ((1u << bits) - 1));
    return out;
}
constexpr unsigned code_bits(char form_digit) { return form_digit == '2' ? 2 : form_digit == '4' ? 4 : form_digit == '8' ? 8 : 0; }

// ---- alignment ----
constexpr uint64_t pad_to_64(uint64_t n) { return (64 - n % 64) % 64; }
constexpr uint64_t align_up_64(uint64_t n) { return (n + 63) / 64 * 64; }
inline uint64_t aligned_total(const std::vector<uint64_t>& sizes) { uint64_t t = 0; for (auto n : sizes) t += align_up_64(n); return t; }

// ---- arena addressing ----
constexpr uint64_t kExpertsPerLayer = 512;
constexpr uint64_t expert_index(uint64_t layer, uint64_t e) { return layer * kExpertsPerLayer + e; }
constexpr uint64_t scale_plane_bytes(uint64_t n, bool fp16) { return fp16 ? n * 2 : n * 4; }
// Per-layer blob sizes -> experts arena bytes (512 blobs per layer).
inline uint64_t arena_total(const std::vector<uint64_t>& blob_per_layer) { uint64_t t = 0; for (auto b : blob_per_layer) t += b * kExpertsPerLayer; return t; }

// ---- router narrowing: F32 pattern with zero low half is exactly a BF16 widened by 16 bits ----
constexpr uint32_t router_narrow(uint32_t u) { return u >> 16; }

// ---- MTP quantiser grid and sizes ----
constexpr unsigned scale_numer(unsigned i) { return 16 + i; }   // candidate f_i = (16+i)/32, i = 0..16
constexpr int code_of_symbol(int s) { return s + 1; }
constexpr unsigned nibble_pack(unsigned lo, unsigned hi) { return lo + 16 * hi; }
constexpr uint64_t expert_bytes(uint64_t elems, uint64_t block, uint64_t block_bytes) { return elems / block * block_bytes; }

static_assert(pad_to_64(0) == 0 && pad_to_64(10) == 54 && pad_to_64(64) == 0, "pad");
static_assert(expert_bytes(512ull * 1280 * 2560, 64, 18) + expert_bytes(512ull * 2560 * 640, 64, 18) == 707788800ull, "Q2_0 experts");
static_assert(expert_bytes(512ull * 1280 * 2560, 32, 18) + expert_bytes(512ull * 2560 * 640, 32, 18) == 1415577600ull, "Q4_0 experts");
static_assert(expert_bytes(512ull * 1280 * 2560, 32, 34) + expert_bytes(512ull * 2560 * 640, 32, 34) == 2673868800ull, "Q8_0 experts");

}  // namespace packb
}  // namespace baloon
