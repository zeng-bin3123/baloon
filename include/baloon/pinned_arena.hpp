// pinned_arena.hpp - sliced cudaHostRegister bookkeeping, FNV-1a chunked checksum, per-layer expert load ranges and the
// bandwidth loop. Pure host formulas.
#pragma once
#include <cstdint>
#include <cstddef>

namespace baloon {
namespace parena {

inline uint64_t slice_count(uint64_t bytes, uint64_t slice) { return slice ? bytes / slice : 0; }
inline uint64_t slice_covered(uint64_t bytes, uint64_t slice) { return slice ? bytes / slice * slice : 0; }
inline uint64_t slice_tail(uint64_t bytes, uint64_t slice) { return bytes - slice_covered(bytes, slice); }  // locked remainder

constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
inline uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed = kFnvOffset) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) h = (h ^ p[i]) * kFnvPrime;
    return h;
}
// Hash a buffer by feeding it in `chunk`-byte pieces (left fold: result is chunk-size independent).
inline uint64_t fnv1a64_chunked(const uint8_t* p, uint64_t n, uint64_t chunk, uint64_t seed = kFnvOffset) {
    if (chunk == 0) chunk = 1;
    uint64_t h = seed;
    for (uint64_t pos = 0; pos < n;) {
        const uint64_t m = (n - pos) < chunk ? (n - pos) : chunk;
        h = fnv1a64(p + pos, m, h);
        pos += m;
    }
    return h;
}

inline uint64_t layer_off(uint64_t blobs_per_layer, uint64_t blob_bytes, uint64_t L) { return L * blobs_per_layer * blob_bytes; }
inline uint64_t layer_bytes(uint64_t blobs_per_layer, uint64_t blob_bytes) { return blobs_per_layer * blob_bytes; }

inline uint64_t read_step(uint64_t rem, uint64_t chunk) { return rem < chunk ? rem : chunk; }
inline uint64_t moved_per_pass(uint64_t bytes, uint64_t chunk) { return chunk ? bytes / chunk * chunk : 0; }

}  // namespace parena
}  // namespace baloon
