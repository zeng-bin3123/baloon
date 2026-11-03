// strata_pack_builder.hpp - pack builder/verifier primitives: LSB-first 2/4/8-bit code packing, 64-byte plane
// alignment, the expert-blob arithmetic (1,382,400 bytes) with role rebuild (gate/up interleave inversion and
// block-index split), scale dtype selection, and the dense/embedding/expert tensor partition.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>
#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace pack {

// ---- code packing -------------------------------------------------------------------------------------------
inline unsigned per_byte(unsigned bits) { return 8 / bits; }
inline bool valid_bits(unsigned bits) { return bits == 2 || bits == 4 || bits == 8; }

// First code lands in the low bits. n must be a multiple of per_byte(bits); each code must fit `bits`.
inline std::optional<std::vector<uint8_t>> pack_codes(const uint8_t* codes, size_t n, unsigned bits) {
    if (!valid_bits(bits)) return std::nullopt;
    const unsigned pb = per_byte(bits);
    if (n % pb != 0) return std::nullopt;
    const unsigned mask = (1u << bits) - 1u;
    std::vector<uint8_t> out(n / pb, 0);
    for (size_t i = 0; i < n; ++i) {
        if (codes[i] > mask) return std::nullopt;
        out[i / pb] |= (uint8_t)(codes[i] << (bits * (i % pb)));
    }
    return out;
}
inline std::optional<std::vector<uint8_t>> unpack_codes(const uint8_t* packed, size_t n, unsigned bits) {
    if (!valid_bits(bits)) return std::nullopt;
    const unsigned pb = per_byte(bits);
    if (n % pb != 0) return std::nullopt;
    const unsigned mask = (1u << bits) - 1u;
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = (uint8_t)((packed[i / pb] >> (bits * (i % pb))) & mask);
    return out;
}

// ---- plane alignment ----------------------------------------------------------------------------------------------
inline uint64_t pad64(uint64_t n) { return (64 - n % 64) % 64; }

// Appends planes so that every plane starts on a 64-byte boundary.
class PlaneWriter {
public:
    size_t append_plane(const uint8_t* p, size_t n) {
        buf_.resize(buf_.size() + (size_t)pad64(buf_.size()), 0);
        const size_t off = buf_.size();
        buf_.insert(buf_.end(), p, p + n);
        return off;
    }
    size_t append_plane(const std::vector<uint8_t>& v) { return append_plane(v.data(), v.size()); }
    const std::vector<uint8_t>& bytes() const { return buf_; }
private:
    std::vector<uint8_t> buf_;
};

// ---- expert blob ---------------------------------------------------------------------------------------------------
constexpr uint64_t kH = 2560, kFF = 640, kQK = 64;
constexpr uint64_t kScGu = kH / kQK;   // blocks per gate/up row (40)
constexpr uint64_t kScD = kFF / kQK;   // blocks per down row (10)
constexpr uint64_t kGuCodes = 2 * kFF * (kScGu * 16);
constexpr uint64_t kGuScales = 2 * kFF * kScGu * 2;
constexpr uint64_t kDCodes = kH * (kScD * 16);
constexpr uint64_t kDScales = kH * kScD * 2;
constexpr uint64_t kBlobBytes = kGuCodes + kGuScales + kDCodes + kDScales;
constexpr uint64_t kRoleBlocks = kH * kFF / kQK;   // 25600 blocks per role
constexpr uint64_t kRoleBytes = kRoleBlocks * 18;  // raw Q2_0 bytes per role
static_assert(kBlobBytes == 1382400, "expert blob size");
static_assert(3 * kRoleBytes == kBlobBytes, "three roles of 18-byte 64-weight blocks");
static_assert(kFF * kScGu == kH * kScD && kFF * kScGu == kRoleBlocks, "equal block counts per role");

// Plane offsets inside a blob (each plane starts 64-aligned, which holds without padding for these sizes).
constexpr uint64_t kGuCodesOff = 0;
constexpr uint64_t kGuScalesOff = kGuCodesOff + kGuCodes;
constexpr uint64_t kDCodesOff = kGuScalesOff + kGuScales;
constexpr uint64_t kDScalesOff = kDCodesOff + kDCodes;
static_assert(kGuScalesOff % 64 == 0 && kDCodesOff % 64 == 0 && kDScalesOff % 64 == 0, "64-aligned planes");

enum class Role { Gate, Up, Down };

// Rebuild one role as raw Q2_0 blocks in block order blk = row*SC + b_in. Gate is slot 2r, up is slot 2r+1.
inline bool rebuild_role(const uint8_t* blob, size_t blob_size, Role role, std::vector<uint8_t>* out) {
    if (blob_size != kBlobBytes) return false;
    const uint64_t rows = (role == Role::Down) ? kH : kFF;
    const uint64_t sc = (role == Role::Down) ? kScD : kScGu;
    const uint8_t* codes = blob + ((role == Role::Down) ? kDCodesOff : kGuCodesOff);
    const uint8_t* scales = blob + ((role == Role::Down) ? kDScalesOff : kGuScalesOff);
    out->assign((size_t)(rows * sc * 18), 0);
    for (uint64_t blk = 0; blk < rows * sc; ++blk) {
        const uint64_t row = blk / sc, b_in = blk % sc;
        const uint64_t slot = role == Role::Gate ? 2 * row : role == Role::Up ? 2 * row + 1 : row;
        const uint64_t idx = slot * sc + b_in;
        uint8_t* o = out->data() + blk * 18;
        std::memcpy(o, scales + idx * 2, 2);
        std::memcpy(o + 2, codes + idx * 16, 16);
    }
    return true;
}

// Inverse of rebuild_role for all three roles.
inline bool build_expert_blob(const std::vector<uint8_t>& gate, const std::vector<uint8_t>& up,
                              const std::vector<uint8_t>& down, std::vector<uint8_t>* blob) {
    if (gate.size() != kRoleBytes || up.size() != kRoleBytes || down.size() != kRoleBytes) return false;
    blob->assign((size_t)kBlobBytes, 0);
    auto put = [&](const std::vector<uint8_t>& src, Role role) {
        const uint64_t rows = (role == Role::Down) ? kH : kFF;
        const uint64_t sc = (role == Role::Down) ? kScD : kScGu;
        uint8_t* codes = blob->data() + ((role == Role::Down) ? kDCodesOff : kGuCodesOff);
        uint8_t* scales = blob->data() + ((role == Role::Down) ? kDScalesOff : kGuScalesOff);
        for (uint64_t blk = 0; blk < rows * sc; ++blk) {
            const uint64_t row = blk / sc, b_in = blk % sc;
            const uint64_t slot = role == Role::Gate ? 2 * row : role == Role::Up ? 2 * row + 1 : row;
            const uint64_t idx = slot * sc + b_in;
            std::memcpy(scales + idx * 2, src.data() + blk * 18, 2);
            std::memcpy(codes + idx * 16, src.data() + blk * 18 + 2, 16);
        }
    };
    put(gate, Role::Gate); put(up, Role::Up); put(down, Role::Down);
    return true;
}

// Pack verification: rebuilt roles must equal the source roles byte for byte.
inline bool verify_expert_blob(const std::vector<uint8_t>& blob, const std::vector<uint8_t>& gate,
                               const std::vector<uint8_t>& up, const std::vector<uint8_t>& down) {
    std::vector<uint8_t> g, u, d;
    if (!rebuild_role(blob.data(), blob.size(), Role::Gate, &g) || !rebuild_role(blob.data(), blob.size(), Role::Up, &u) ||
        !rebuild_role(blob.data(), blob.size(), Role::Down, &d)) return false;
    return g == gate && u == up && d == down;
}

// ---- expert arena addressing (experts.bin: layer*NE + e blobs of kBlobBytes) -------------------------------------
constexpr uint64_t kExpertsPerLayer = 512;
inline uint64_t expert_flat_index(uint64_t layer, uint64_t e) { return layer * kExpertsPerLayer + e; }
inline uint64_t expert_blob_offset(uint64_t layer, uint64_t e) { return expert_flat_index(layer, e) * kBlobBytes; }
// Experts verified with `--expert-every k` out of n: ceil(n / k); each contributes 3 role checks.
inline uint64_t experts_checked(uint64_t n, uint64_t every) { return every == 0 ? 0 : (n + every - 1) / every; }
inline uint64_t role_checks(uint64_t n, uint64_t every) { return 3 * experts_checked(n, every); }

// ---- scale dtype ---------------------------------------------------------------------------------------------------
inline uint64_t scale_bytes(bool fp16) { return fp16 ? 2 : 4; }

// FP16 only if the FP16 round trip is bit-exact for every scale.
inline bool choose_scale_dtype_fp16(const float* s, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (f16::f32_bits(f16::half_to_float(f16::float_to_half(s[i]))) != f16::f32_bits(s[i])) return false;
    return true;
}

// Serialise scales little-endian in the chosen dtype.
inline std::vector<uint8_t> encode_scales(const float* s, size_t n, bool* fp16_out) {
    const bool fp16 = choose_scale_dtype_fp16(s, n);
    if (fp16_out) *fp16_out = fp16;
    std::vector<uint8_t> out;
    out.reserve(n * (size_t)scale_bytes(fp16));
    for (size_t i = 0; i < n; ++i) {
        if (fp16) { const uint16_t h = f16::float_to_half(s[i]); out.push_back((uint8_t)h); out.push_back((uint8_t)(h >> 8)); }
        else { const uint32_t b = f16::f32_bits(s[i]); for (int k = 0; k < 4; ++k) out.push_back((uint8_t)(b >> (8 * k))); }
    }
    return out;
}

// ---- tensor partition ----------------------------------------------------------------------------------------------
inline bool contains(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }
inline bool is_expert(const std::string& name) { return contains(name, ".experts.") || contains(name, "_exps"); }
inline bool is_embd(const std::string& name) { return contains(name, "embed_tokens") || contains(name, "token_embd"); }

struct Partition { std::vector<size_t> dense, embd, expert; };

// Each tensor goes to exactly one of dense.bin, embd.bin or the expert arena.
inline Partition partition_tensors(const std::vector<std::string>& names) {
    Partition p;
    for (size_t i = 0; i < names.size(); ++i) {
        if (is_expert(names[i])) p.expert.push_back(i);
        else if (is_embd(names[i])) p.embd.push_back(i);
        else p.dense.push_back(i);
    }
    return p;
}

inline bool shapes_equal(uint64_t a, uint64_t b) { return a == b; }

}  // namespace pack
}  // namespace baloon
