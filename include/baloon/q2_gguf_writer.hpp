// q2_gguf_writer.hpp - minimal GGUF v3 writer and Q2_0 quantiser: code clamp, LSB-first packing,
// fixed-point requantisation, 18-byte blocks, alignment padding.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <variant>
#include <vector>
#include "baloon/f16_conversion.hpp"
#include "baloon/gguf_lazy_loader.hpp"

namespace baloon {
namespace q2w {

constexpr size_t kBlockElems = 64, kBlockBytes = 18, kCodeBytes = 16;
constexpr uint32_t kDefaultAlignment = 32;

// Code for a rounded symbol s: clamp(s + 1, 0, 3).
inline int code_of_symbol(long s) { long c = s + 1; return (int)(c < 0 ? 0 : (c > 3 ? 3 : c)); }
inline int symbol_of_code(int c) { return c - 1; }  // in {-1,0,1,2}

// Packing position of element j inside a block's code bytes.
inline size_t byte_index(size_t j) { return j / 4; }
inline unsigned bit_offset(size_t j) { return (unsigned)(j % 4) * 2; }

// Padding needed to reach the next multiple of a.
inline uint64_t pad(uint64_t a, uint64_t n) { return (a - n % a) % a; }

// Write the 18-byte block for explicit scale bits and 64 codes (LSB-first, four per byte).
inline void pack_block(uint16_t d_bits, const uint8_t codes[64], uint8_t* out) {
    out[0] = (uint8_t)(d_bits & 0xFF); out[1] = (uint8_t)(d_bits >> 8);
    std::memset(out + 2, 0, kCodeBytes);
    for (size_t j = 0; j < 64; ++j) out[2 + byte_index(j)] |= (uint8_t)((codes[j] & 3u) << bit_offset(j));
}
inline void unpack_codes(const uint8_t* block, uint8_t codes[64]) {
    for (size_t j = 0; j < 64; ++j) codes[j] = (uint8_t)((block[2 + byte_index(j)] >> bit_offset(j)) & 3u);
}

// Quantise one block against a given (already fp16-representable) scale d. d == 0 gives symbol 0 everywhere.
inline void quantize_codes_with_scale(const float* w, float d, uint8_t codes[64]) {
    for (int j = 0; j < 64; ++j) {
        long s = 0;
        if (d != 0.f) s = (long)std::nearbyint((double)w[j] / (double)d);
        codes[j] = (uint8_t)code_of_symbol(s);
    }
}

// d = amax (rounded to fp16 so decode uses exactly the scale the codes were computed against).
inline void quantize_block(const float* w, uint8_t* out18) {
    float amax = 0.f;
    for (int j = 0; j < 64; ++j) amax = std::fmax(amax, std::fabs(w[j]));
    const uint16_t hb = f16::float_to_half(amax);
    const float d = f16::half_to_float(hb);
    uint8_t codes[64];
    quantize_codes_with_scale(w, d, codes);
    pack_block(hb, codes, out18);
}

inline void dequantize_block(const uint8_t* b, float* y) {
    const float d = f16::half_to_float((uint16_t)(b[0] | (b[1] << 8)));
    uint8_t codes[64];
    unpack_codes(b, codes);
    for (int j = 0; j < 64; ++j) y[j] = d * (float)symbol_of_code(codes[j]);
}

// ---- GGUF writer ---------------------------------------------------------------------------------------------
using KvValue = std::variant<std::string, uint32_t, uint64_t, float, bool>;
struct Kv { std::string key; KvValue value; };
struct TensorSpec {
    std::string name;
    std::vector<uint64_t> shape;   // GGUF order (fastest axis first)
    uint32_t type = gguf::kQ2_0;
    std::vector<uint8_t> data;     // raw bytes, exactly tensor_bytes(...)
};

namespace detail {
inline void put32(std::vector<uint8_t>& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back((uint8_t)(v >> (8 * i))); }
inline void put64(std::vector<uint8_t>& o, uint64_t v) { for (int i = 0; i < 8; ++i) o.push_back((uint8_t)(v >> (8 * i))); }
inline void putstr(std::vector<uint8_t>& o, const std::string& s) { put64(o, s.size()); o.insert(o.end(), s.begin(), s.end()); }
}  // namespace detail

// Serialise a GGUF v3 file. Every tensor and the data section start on an `alignment`-byte boundary.
// Returns false (with *err) if a tensor's byte count does not match its shape/type.
inline bool write_file(const std::vector<Kv>& kvs, const std::vector<TensorSpec>& tensors, std::vector<uint8_t>* out,
                       std::string* err = nullptr, uint32_t alignment = kDefaultAlignment) {
    using namespace detail;
    if (alignment == 0) { if (err) *err = "alignment must be positive"; return false; }
    std::vector<uint64_t> offsets;
    uint64_t cur = 0;
    for (const auto& t : tensors) {
        const auto g = gguf::block_geometry(t.type);
        uint64_t n = 1; for (auto d : t.shape) n *= d;
        if (g.el == 0 || t.shape.empty() || !gguf::whole_blocks(n, g.el) || t.data.size() != gguf::tensor_bytes(n, g.el, g.by)) {
            if (err) *err = "tensor '" + t.name + "' has inconsistent size/shape/type";
            return false;
        }
        offsets.push_back(cur);
        cur += t.data.size();
        cur += pad(alignment, cur);
    }
    std::vector<uint8_t> o;
    put32(o, gguf::kMagic); put32(o, 3);
    put64(o, tensors.size());
    const bool has_align = [&] { for (auto& k : kvs) if (k.key == "general.alignment") return true; return false; }();
    put64(o, kvs.size() + ((has_align || alignment == kDefaultAlignment) ? 0 : 1));
    for (const auto& k : kvs) {
        putstr(o, k.key);
        if (auto* s = std::get_if<std::string>(&k.value)) { put32(o, 8); putstr(o, *s); }
        else if (auto* u = std::get_if<uint32_t>(&k.value)) { put32(o, 4); put32(o, *u); }
        else if (auto* u6 = std::get_if<uint64_t>(&k.value)) { put32(o, 10); put64(o, *u6); }
        else if (auto* f = std::get_if<float>(&k.value)) { put32(o, 6); put32(o, f16::f32_bits(*f)); }
        else { put32(o, 7); o.push_back(std::get<bool>(k.value) ? 1 : 0); }
    }
    if (!has_align && alignment != kDefaultAlignment) { putstr(o, "general.alignment"); put32(o, 4); put32(o, alignment); }
    for (size_t i = 0; i < tensors.size(); ++i) {
        putstr(o, tensors[i].name);
        put32(o, (uint32_t)tensors[i].shape.size());
        for (auto d : tensors[i].shape) put64(o, d);
        put32(o, tensors[i].type);
        put64(o, offsets[i]);
    }
    o.resize(o.size() + pad(alignment, o.size()), 0);
    for (size_t i = 0; i < tensors.size(); ++i) {
        o.insert(o.end(), tensors[i].data.begin(), tensors[i].data.end());
        o.resize(o.size() + pad(alignment, tensors[i].data.size()), 0);
    }
    *out = std::move(o);
    return true;
}

}  // namespace q2w
}  // namespace baloon
