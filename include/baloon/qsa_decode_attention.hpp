// qsa_decode_attention.hpp - chunked split-softmax decode attention over selected KV cells and its merge.
// Block (chunk, kv_head[, query]); CHUNK = 64 cells, G = 12 query heads per kv head, HD = 256.
// Each chunk stores (acc, m, l); the merge rescales by exp(m - M) and divides by L.
#pragma once
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

namespace baloon {
namespace qsa_decode {

constexpr uint32_t kChunk = 64;
constexpr uint32_t kG = 12;
constexpr uint32_t kHD = 256;
constexpr uint32_t kWarps = 8;
constexpr uint32_t kLanes = 32;
constexpr uint32_t kScaleGroup = 32;               // int8 scale group size
constexpr uint32_t kMaxBatch = 65535;              // grid.z

inline uint32_t chunks(uint32_t cap) { return (cap + kChunk - 1) / kChunk; }
inline uint32_t n_here(uint32_t n_ids, uint32_t c0) { return n_ids > c0 ? std::min(kChunk, n_ids - c0) : 0u; }
inline uint32_t kv_head(uint32_t h) { return h / kG; }
inline uint32_t local_head(uint32_t h) { return h % kG; }
inline uint32_t scale_group(uint32_t d) { return d / kScaleGroup; }
inline bool batch_ok(uint32_t nq) { return nq <= kMaxBatch; }

// warp/lane partitions
inline uint32_t cell_warp(uint32_t c) { return c % kWarps; }
inline uint32_t cell_iter(uint32_t c) { return c / kWarps; }
inline uint32_t dim_lane(uint32_t d) { return d / 8; }
inline uint32_t dim_slot(uint32_t d) { return d % 8; }
inline uint32_t head_warp(uint32_t h) { return h % kWarps; }
inline uint32_t head_iter(uint32_t h) { return h / kWarps; }

// pooled row of (page, kv head, slot)
inline uint64_t pool_row(uint64_t page, uint64_t nkv, uint64_t h, uint64_t ps, uint64_t slot) {
    return (page * nkv + h) * ps + slot;
}

// scratch layout per query: acc [x*HD], m [x], l [x] with x = n_chunks * n_head
inline uint64_t x_of(uint32_t n_chunks, uint32_t n_head) { return (uint64_t)n_chunks * n_head; }
inline uint64_t acc_off(uint32_t chunk, uint32_t head, uint32_t n_head, uint32_t d) {
    return ((uint64_t)chunk * n_head + head) * kHD + d;
}
inline uint64_t m_off(uint32_t n_chunks, uint32_t n_head, uint32_t chunk, uint32_t head) {
    return x_of(n_chunks, n_head) * kHD + (uint64_t)chunk * n_head + head;
}
inline uint64_t l_off(uint32_t n_chunks, uint32_t n_head, uint32_t chunk, uint32_t head) {
    return x_of(n_chunks, n_head) * kHD + x_of(n_chunks, n_head) + (uint64_t)chunk * n_head + head;
}
inline uint64_t query_stride(uint32_t n_chunks, uint32_t n_head) { return x_of(n_chunks, n_head) * (kHD + 2); }
inline uint64_t query_base(uint32_t q, uint32_t n_chunks, uint32_t n_head) { return (uint64_t)q * query_stride(n_chunks, n_head); }

// ---- host reference of the split softmax. scores[i] already scaled; values row i has `hd` entries.
struct Partial { std::vector<float> acc; float m; float l; };
inline Partial empty_partial(size_t hd) { return Partial{std::vector<float>(hd, 0.f), -FLT_MAX, 0.f}; }
inline Partial chunk_partial(const float* scores, const float* values, uint32_t n, size_t hd) {
    Partial p = empty_partial(hd);
    if (n == 0) return p;
    float m = -FLT_MAX;
    for (uint32_t i = 0; i < n; ++i) m = std::max(m, scores[i]);
    float l = 0.f;
    for (uint32_t i = 0; i < n; ++i) {
        const float e = std::exp(scores[i] - m);
        l += e;
        for (size_t d = 0; d < hd; ++d) p.acc[d] += e * values[i * hd + d];
    }
    p.m = m; p.l = l;
    return p;
}
// Merge: M = max m; L = sum l*exp(m-M); out = sum acc*exp(m-M) / L (zero when nothing was attended).
inline std::vector<float> merge(const std::vector<Partial>& ps, size_t hd) {
    std::vector<float> out(hd, 0.f);
    float M = -FLT_MAX;
    for (const auto& p : ps) M = std::max(M, p.m);
    float L = 0.f;
    std::vector<double> acc(hd, 0.0);
    for (const auto& p : ps) {
        if (p.l == 0.f) continue;           // empty chunk is neutral
        const float s = std::exp(p.m - M);
        L += p.l * s;
        for (size_t d = 0; d < hd; ++d) acc[d] += (double)p.acc[d] * s;
    }
    if (L > 0.f) for (size_t d = 0; d < hd; ++d) out[d] = (float)(acc[d] / L);
    return out;
}

}  // namespace qsa_decode
}  // namespace baloon
