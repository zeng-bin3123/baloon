// gguf_file_loader.hpp - file-backed lazy GGUF access: shard discovery on disk, a read-only mapped multi-shard
// reader (`Gguf`) that hands out whole tensors, single experts of fused [in, out, experts] tensors and gathered rows,
// and flat dequantisation of those byte runs to f32 (bf16 widening is the exact 16-bit shift).
// Builds on gguf_lazy_loader.hpp (header parsing, block geometry) and artifact_dequant.hpp (per-block dequantisers).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "baloon/artifact_dequant.hpp"
#include "baloon/gguf_lazy_loader.hpp"

namespace baloon {
namespace gguf_file {

// ---- bf16 -> f32: the 16 stored bits become the high half of the f32 (exact, NaN payloads and -0 preserved) -----
inline void bf16_to_f32(const uint16_t* src, uint64_t n, float* dst) {
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t u = (uint32_t)src[i] << 16;
        std::memcpy(&dst[i], &u, 4);
    }
}

// ---- shard discovery: the complete, index-ordered shard set of `path` (the path itself when it is not a shard) ----
// A shard set with any missing sibling is rejected; a path whose file does not exist is rejected.
inline std::optional<std::vector<std::string>> discover_shards(const std::string& path) {
    auto r = gguf::discover_shards(path);
    if (!r) return std::nullopt;  // never empty when present
    for (const auto& s : *r) {
        struct stat st;
        if (::stat(s.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return std::nullopt;
    }
    return r;
}

// ---- flat dequantisation of `elements` values of ggml type `type` stored at `src` -> f32 ----------------------
// False for an unknown type or when `elements` is not a whole number of blocks (nothing is written then).
inline bool dequant_flat(uint32_t type, const uint8_t* src, uint64_t elements, float* out) {
    const gguf::BlockGeometry g = gguf::block_geometry(type);
    if (g.el == 0 || !src || !out || elements % g.el != 0) return false;
    switch (type) {
        case gguf::kF32: std::memcpy(out, src, (size_t)elements * 4); return true;
        case gguf::kF16:
            for (uint64_t i = 0; i < elements; ++i) out[i] = artifact::fp16_to_fp32(artifact::read_u16(src + 2 * i));
            return true;
        case gguf::kBF16: {
            for (uint64_t i = 0; i < elements; ++i) {
                const uint32_t u = (uint32_t)artifact::read_u16(src + 2 * i) << 16;
                std::memcpy(&out[i], &u, 4);
            }
            return true;
        }
        default: break;
    }
    void (*fn)(const uint8_t*, float*) = nullptr;
    switch (type) {
        case gguf::kQ2_0: fn = artifact::dequantize_q2_0; break;
        case gguf::kQ4_0: fn = artifact::dequantize_q4_0; break;
        case gguf::kQ5_0: fn = artifact::dequantize_q5_0; break;
        case gguf::kQ8_0: fn = artifact::dequantize_q8_0; break;
        case gguf::kIQ4_NL: fn = artifact::dequantize_iq4_nl; break;
        case gguf::kQ3_K: fn = artifact::dequantize_q3_K; break;
        case gguf::kQ4_K: fn = artifact::dequantize_q4_K; break;
        case gguf::kQ5_K: fn = artifact::dequantize_q5_K; break;
        case gguf::kQ6_K: fn = artifact::dequantize_q6_K; break;
        case gguf::kIQ4_XS: fn = artifact::dequantize_iq4_xs; break;
        default: return false;
    }
    const uint64_t blocks = elements / g.el;
    for (uint64_t b = 0; b < blocks; ++b) fn(src + b * g.by, out + b * g.el);
    return true;
}

// ---- the lazy multi-shard reader -------------------------------------------------------------------------------
struct Gguf {
    Gguf() = default;
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;
    ~Gguf() { close(); }

    // Map every shard of `path` read-only and parse its header. False (and `err`) on a missing shard, an unreadable
    // or malformed file, or a tensor name that appears in two shards.
    bool open(const std::string& path, std::string* err = nullptr) {
        close();
        auto shards = discover_shards(path);
        if (!shards) return fail(err, "incomplete or missing shard set: " + path);
        for (const auto& p : *shards) {
            const int fd = ::open(p.c_str(), O_RDONLY);
            if (fd < 0) { close(); return fail(err, "cannot open " + p); }
            struct stat st;
            if (::fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); close(); return fail(err, "empty or unreadable " + p); }
            void* m = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            ::close(fd);
            if (m == MAP_FAILED) { close(); return fail(err, "mmap failed for " + p); }
            auto s = std::make_unique<Shard>();
            s->path = p; s->map = m; s->size = (uint64_t)st.st_size;
            s->img = std::make_unique<gguf::LazyImage>((const uint8_t*)m, s->size);
            if (!s->img->ok()) { ::munmap(m, (size_t)s->size); close(); return fail(err, "malformed GGUF header in " + p); }
            for (const auto& t : s->img->index().tensors) {
                if (where_.count(t.name)) { ::munmap(m, (size_t)s->size); close(); return fail(err, "duplicate tensor " + t.name); }
                where_[t.name] = shards_.size();
            }
            shards_.push_back(std::move(s));
        }
        return true;
    }
    void close() {
        for (auto& s : shards_) if (s && s->map) ::munmap(s->map, (size_t)s->size);
        shards_.clear(); where_.clear();
    }
    size_t shard_count() const { return shards_.size(); }
    bool has(const std::string& name) const { return where_.count(name) != 0; }
    const gguf::TensorInfo* info(const std::string& name) const {
        auto it = where_.find(name);
        return it == where_.end() ? nullptr : shards_[it->second]->img->index().find(name);
    }

    // Whole tensor bytes.
    std::optional<gguf::Span> tensor(const std::string& name) const {
        auto it = where_.find(name);
        return it == where_.end() ? std::nullopt : shards_[it->second]->img->read_tensor(name);
    }
    // Expert `e` of a fused [in, out, experts] tensor: nbytes/experts contiguous bytes at e * that length.
    // Legal only when one expert holds a whole number of blocks.
    std::optional<gguf::Span> expert(const std::string& name, uint64_t e) const {
        auto it = where_.find(name);
        return it == where_.end() ? std::nullopt : shards_[it->second]->img->read_expert(name, e);
    }
    // Gathered rows of a 2-D [width, rows] tensor: the bytes of each listed row, concatenated in the order given.
    // Legal only when width is a whole number of blocks and every index is below the row count.
    bool rows(const std::string& name, const std::vector<uint64_t>& idx, std::vector<uint8_t>& out) const {
        out.clear();
        const gguf::TensorInfo* t = info(name);
        if (!t || t->shape.size() != 2) return false;
        const gguf::BlockGeometry g = gguf::block_geometry(t->type);
        if (!gguf::whole_blocks(t->shape[0], g.el)) return false;
        for (uint64_t r : idx) if (!gguf::row_in_range(r, t->shape[1])) return false;
        const uint64_t rb = gguf::row_bytes(t->shape[0], g.el, g.by);
        out.reserve(idx.size() * (size_t)rb);
        for (uint64_t r : idx) {
            auto sp = shards_[where_.at(name)]->img->read_rows(name, r, 1);
            if (!sp || sp->len != rb) { out.clear(); return false; }
            out.insert(out.end(), sp->ptr, sp->ptr + sp->len);
        }
        return true;
    }
    // f32 views of the same runs.
    bool expert_f32(const std::string& name, uint64_t e, std::vector<float>& out) const {
        out.clear();
        const gguf::TensorInfo* t = info(name);
        auto sp = expert(name, e);
        if (!t || !sp) return false;
        const gguf::BlockGeometry g = gguf::block_geometry(t->type);
        const uint64_t n = sp->len / g.by * g.el;
        out.resize((size_t)n);
        if (!dequant_flat(t->type, sp->ptr, n, out.data())) { out.clear(); return false; }
        return true;
    }
    bool rows_f32(const std::string& name, const std::vector<uint64_t>& idx, std::vector<float>& out) const {
        out.clear();
        const gguf::TensorInfo* t = info(name);
        std::vector<uint8_t> raw;
        if (!t || !rows(name, idx, raw)) return false;
        const uint64_t n = idx.size() * t->shape[0];
        out.resize((size_t)n);
        if (!dequant_flat(t->type, raw.data(), n, out.data())) { out.clear(); return false; }
        return true;
    }

private:
    struct Shard {
        std::string path; void* map = nullptr; uint64_t size = 0;
        std::unique_ptr<gguf::LazyImage> img;
    };
    static bool fail(std::string* err, const std::string& m) { if (err) *err = m; return false; }
    std::vector<std::unique_ptr<Shard>> shards_;
    std::map<std::string, size_t> where_;
};

}  // namespace gguf_file
}  // namespace baloon
