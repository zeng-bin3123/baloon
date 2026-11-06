// gguf_lazy_loader.hpp - lazy GGUF tensor access: block geometry, per-expert byte runs of fused
// [in, out, experts] tensors, row offsets, shape reversal, shard bookkeeping and a GGUF v3 header parser.
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>
#include <dirent.h>
#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace gguf {

// ---- tensor types and block geometry (elements per block, bytes per block) ---------------------------
enum GgmlType : uint32_t {
    kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ5_0 = 6, kQ8_0 = 8, kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14,
    kIQ4_NL = 20, kIQ4_XS = 23, kBF16 = 30,
    kQ2_0 = 42  // GGML type id 42: 64 elements / 18 bytes (fp16 scale + 64 two-bit codes)
};

struct BlockGeometry { uint64_t el = 0, by = 0; };  // el == 0 means unknown type

inline BlockGeometry block_geometry(uint32_t type) {
    switch (type) {
        case kF32: return {1, 4};
        case kF16: return {1, 2};
        case kBF16: return {1, 2};
        case kQ2_0: return {64, 18};
        case kQ4_0: return {32, 18};
        case kQ5_0: return {32, 22};
        case kQ8_0: return {32, 34};
        case kQ3_K: return {256, 110};
        case kQ4_K: return {256, 144};
        case kQ5_K: return {256, 176};
        case kQ6_K: return {256, 210};
        case kIQ4_NL: return {32, 18};
        case kIQ4_XS: return {256, 136};
        default: return {0, 0};
    }
}

inline uint64_t tensor_bytes(uint64_t elements, uint64_t el, uint64_t by) { return elements / el * by; }
inline uint64_t row_bytes(uint64_t width, uint64_t el, uint64_t by) { return width / el * by; }
inline uint64_t expert_bytes(uint64_t per, uint64_t el, uint64_t by) { return per / el * by; }
inline uint64_t expert_start(uint64_t e, uint64_t per, uint64_t el, uint64_t by) { return e * expert_bytes(per, el, by); }
// The Python spelling of the expert start, e * (per // el) * by; always equal to expert_start (e * len).
inline uint64_t expert_start_py(uint64_t e, uint64_t per, uint64_t el, uint64_t by) { return e * (per / el) * by; }
inline bool whole_blocks(uint64_t elements, uint64_t el) { return el != 0 && elements % el == 0; }

// ---- PLE table facts (width-160 IQ4_NL rows, 320,001,536 tokens, 16 rows per token) --------------------------
constexpr uint64_t kPleVocab = 320001536ull, kPleWidth = 160, kPleRowsPerToken = 16;
inline uint64_t ple_row_bytes() { auto g = block_geometry(kIQ4_NL); return row_bytes(kPleWidth, g.el, g.by); }          // 90
inline uint64_t ple_table_bytes() { return kPleVocab * ple_row_bytes(); }                                                // 28.8e9
inline uint64_t ple_fp32_bytes() { return kPleVocab * kPleWidth * 4; }                                                   // ~205 GB
inline uint64_t ple_token_floats() { return kPleRowsPerToken * kPleWidth; }                                              // 2560

// ---- shapes ---------------------------------------------------------------------------------------------
inline std::vector<uint64_t> numpy_shape(std::vector<uint64_t> shape) {
    return std::vector<uint64_t>(shape.rbegin(), shape.rend());
}

// ---- shards -----------------------------------------------------------------------------------------------
inline bool shards_complete(uint64_t found, uint64_t total) { return found == total; }
inline bool row_in_range(uint64_t idx, uint64_t n) { return idx < n; }

inline std::string shard_name(const std::string& prefix, unsigned index1, unsigned total) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "-%05u-of-%05u.gguf", index1, total);
    return prefix + buf;
}

// Parse "<prefix>-00002-of-00005.gguf"; returns false if the name does not follow the shard pattern.
inline bool parse_shard_name(const std::string& fname, std::string* prefix, unsigned* index1, unsigned* total) {
    const std::string suffix = ".gguf";
    if (fname.size() < suffix.size() + 15 || fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;
    const size_t end = fname.size() - suffix.size();
    // "-NNNNN-of-MMMMM" is 15 chars
    const size_t p = end - 15;
    if (fname[p] != '-' || fname.compare(p + 6, 4, "-of-") != 0) return false;
    unsigned a = 0, b = 0;
    for (int i = 0; i < 5; ++i) {
        const char ca = fname[p + 1 + i], cb = fname[p + 10 + i];
        if (ca < '0' || ca > '9' || cb < '0' || cb > '9') return false;
        a = a * 10 + (unsigned)(ca - '0');
        b = b * 10 + (unsigned)(cb - '0');
    }
    *prefix = fname.substr(0, p); *index1 = a; *total = b;
    return true;
}

// Given file names in a directory, return the shard set of `first` (ordered by index) if complete.
inline std::optional<std::vector<std::string>> discover_shards_in(const std::vector<std::string>& names,
                                                                  const std::string& first) {
    std::string prefix; unsigned idx = 0, total = 0;
    if (!parse_shard_name(first, &prefix, &idx, &total) || total == 0) return std::nullopt;
    std::vector<std::string> out;
    for (unsigned i = 1; i <= total; ++i) {
        const std::string want = shard_name(prefix, i, total);
        bool ok = false;
        for (const auto& n : names) if (n == want) { ok = true; break; }
        if (!ok) return std::nullopt;  // missing shard -> rejected
        out.push_back(want);
    }
    return out;
}

inline std::optional<std::vector<std::string>> discover_shards(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "." : (slash == 0 ? "/" : path.substr(0, slash));
    const std::string fname = slash == std::string::npos ? path : path.substr(slash + 1);
    std::string prefix; unsigned idx = 0, total = 0;
    if (!parse_shard_name(fname, &prefix, &idx, &total)) return std::vector<std::string>{path};  // single file
    std::vector<std::string> names;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) names.push_back(e->d_name);
        closedir(d);
    }
    auto r = discover_shards_in(names, fname);
    if (!r) return std::nullopt;
    const std::string base = slash == std::string::npos ? "" : (dir == "/" ? "/" : dir + "/");
    for (auto& s : *r) s = base + s;
    return r;
}

// ---- GGUF v3 header parsing ------------------------------------------------------------------------------
constexpr uint32_t kMagic = 0x46554747u;  // bytes 'G','G','U','F' little-endian
constexpr uint32_t kDefaultAlignment = 32;

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape;  // GGUF order: fastest axis first
    uint32_t type = 0;
    uint64_t offset = 0;          // relative to the data section
    uint64_t elements() const { uint64_t n = 1; for (auto d : shape) n *= d; return n; }
    uint64_t nbytes() const { auto g = block_geometry(type); return g.el ? tensor_bytes(elements(), g.el, g.by) : 0; }
};

struct Index {
    uint32_t version = 0;
    uint32_t alignment = kDefaultAlignment;
    uint64_t data_start = 0;  // absolute byte offset of the data section
    std::vector<TensorInfo> tensors;
    const TensorInfo* find(const std::string& n) const {
        for (auto& t : tensors) if (t.name == n) return &t;
        return nullptr;
    }
};

namespace detail {
struct Cursor {
    const uint8_t* p; uint64_t n; uint64_t pos = 0; bool ok = true;
    bool need(uint64_t k) { if (!ok || n - pos < k || pos > n) { ok = false; return false; } return true; }
    uint32_t u32() { if (!need(4)) return 0; uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)p[pos + i] << (8 * i); pos += 4; return v; }
    uint64_t u64() { if (!need(8)) return 0; uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)p[pos + i] << (8 * i); pos += 8; return v; }
    std::string str() { uint64_t l = u64(); if (!need(l)) return {}; std::string s((const char*)p + pos, (size_t)l); pos += l; return s; }
    void skip(uint64_t k) { if (need(k)) pos += k; }
};
inline uint64_t scalar_size(uint32_t t) {
    switch (t) { case 0: case 1: case 7: return 1; case 2: case 3: return 2; case 4: case 5: case 6: return 4;
                 case 10: case 11: case 12: return 8; default: return 0; }
}
inline void skip_value(Cursor& c, uint32_t t) {
    if (t == 8) { (void)c.str(); return; }
    if (t == 9) {
        const uint32_t et = c.u32(); const uint64_t cnt = c.u64();
        if (et == 8 || et == 9) { for (uint64_t i = 0; i < cnt && c.ok; ++i) skip_value(c, et); return; }
        const uint64_t sz = scalar_size(et);
        if (sz == 0) { c.ok = false; return; }
        if (cnt > c.n / sz) { c.ok = false; return; }
        c.skip(cnt * sz); return;
    }
    const uint64_t sz = scalar_size(t);
    if (sz == 0) { c.ok = false; return; }
    c.skip(sz);
}
}  // namespace detail

// Parse the header of an in-memory GGUF image. Returns nullopt on malformed input; tensor ranges are bounds-checked.
inline std::optional<Index> parse_header(const uint8_t* data, uint64_t size) {
    detail::Cursor c{data, size};
    if (c.u32() != kMagic) return std::nullopt;
    Index ix;
    ix.version = c.u32();
    if (ix.version < 2 || ix.version > 3) return std::nullopt;
    const uint64_t nt = c.u64(), nkv = c.u64();
    if (!c.ok || nt > (1u << 24) || nkv > (1u << 24)) return std::nullopt;
    for (uint64_t i = 0; i < nkv && c.ok; ++i) {
        const std::string key = c.str();
        const uint32_t t = c.u32();
        if (key == "general.alignment" && t == 4) { ix.alignment = c.u32(); continue; }
        detail::skip_value(c, t);
    }
    if (!c.ok || ix.alignment == 0) return std::nullopt;
    for (uint64_t i = 0; i < nt && c.ok; ++i) {
        TensorInfo ti;
        ti.name = c.str();
        const uint32_t nd = c.u32();
        if (nd == 0 || nd > 8) return std::nullopt;
        for (uint32_t d = 0; d < nd; ++d) ti.shape.push_back(c.u64());
        ti.type = c.u32();
        ti.offset = c.u64();
        ix.tensors.push_back(std::move(ti));
    }
    if (!c.ok) return std::nullopt;
    const uint64_t a = ix.alignment;
    ix.data_start = (c.pos + a - 1) / a * a;
    for (auto& t : ix.tensors) {
        auto g = block_geometry(t.type);
        if (g.el == 0 || !whole_blocks(t.elements(), g.el)) return std::nullopt;
        if (ix.data_start > size || t.offset > size - ix.data_start || t.nbytes() > size - ix.data_start - t.offset) return std::nullopt;
    }
    return ix;
}

// ---- lazy tensor view over an in-memory / mapped image ----------------------------------------------------
struct Span { const uint8_t* ptr = nullptr; uint64_t len = 0; };

class LazyImage {
public:
    LazyImage(const uint8_t* data, uint64_t size) : data_(data), size_(size) { auto ix = parse_header(data, size); if (ix) { ix_ = *ix; ok_ = true; } }
    bool ok() const { return ok_; }
    uint64_t size() const { return size_; }  // image length in bytes
    const Index& index() const { return ix_; }

    // Whole tensor bytes.
    std::optional<Span> read_tensor(const std::string& name) const {
        const TensorInfo* t = ok_ ? ix_.find(name) : nullptr;
        if (!t) return std::nullopt;
        return Span{data_ + ix_.data_start + t->offset, t->nbytes()};
    }
    // Expert `e` of a fused [in, out, experts] tensor: contiguous run of one expert's bytes.
    std::optional<Span> read_expert(const std::string& name, uint64_t e) const {
        const TensorInfo* t = ok_ ? ix_.find(name) : nullptr;
        if (!t || t->shape.size() < 2) return std::nullopt;
        const uint64_t n = t->shape.back();
        if (!row_in_range(e, n)) return std::nullopt;
        const auto g = block_geometry(t->type);
        const uint64_t per = t->elements() / n;
        if (!whole_blocks(per, g.el)) return std::nullopt;
        return Span{data_ + ix_.data_start + t->offset + expert_start(e, per, g.el, g.by), expert_bytes(per, g.el, g.by)};
    }
    // Rows [r0, r0+count) of a 2-D tensor (width = shape[0], rows = shape[1]).
    std::optional<Span> read_rows(const std::string& name, uint64_t r0, uint64_t count) const {
        const TensorInfo* t = ok_ ? ix_.find(name) : nullptr;
        if (!t || t->shape.size() != 2) return std::nullopt;
        const uint64_t n = t->shape[1];
        if (count == 0 || !row_in_range(r0, n) || count > n - r0) return std::nullopt;
        const auto g = block_geometry(t->type);
        const uint64_t rb = row_bytes(t->shape[0], g.el, g.by);
        return Span{data_ + ix_.data_start + t->offset + r0 * rb, count * rb};
    }
private:
    const uint8_t* data_; uint64_t size_; Index ix_; bool ok_ = false;
};

// BF16 widens to FP32 by a 16-bit left shift (exact).
inline uint32_t bf16_widen_bits(uint16_t h) { return (uint32_t)h << 16; }
inline float bf16_widen(uint16_t h) { return f16::bits_f32(bf16_widen_bits(h)); }

}  // namespace gguf
}  // namespace baloon
