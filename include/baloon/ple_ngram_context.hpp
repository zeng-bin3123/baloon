// ple_ngram_context.hpp - context construction (cut at negative/EOS predecessor) and incremental 64-bit mixing of the
// PLE n-gram hash.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace plectx {

// ctx[0] is the current token; ctx[s] (s >= 1) is the predecessor s positions back = prev[s-1].
// Once a predecessor is negative or equals eos, that position and every later one is eos.
inline std::vector<int64_t> build_context(int64_t token, const int64_t* prev, uint64_t n, int64_t eos) {
    std::vector<int64_t> ctx(n ? n : 1, eos);
    ctx[0] = token;
    bool cut = false;
    for (uint64_t s = 0; s + 1 < n; ++s) {
        cut = cut || prev[s] < 0 || prev[s] == eos;
        ctx[s + 1] = cut ? eos : prev[s];
    }
    return ctx;
}

// mixed(j) = ctx[0]*m[0] ^ ... ^ ctx[j]*m[j] over wrapping 64-bit arithmetic (order j+1).
inline uint64_t mixed(const int64_t* ctx, const uint64_t* mult, uint64_t j) {
    uint64_t h = (uint64_t)ctx[0] * mult[0];
    for (uint64_t i = 1; i <= j; ++i) h ^= (uint64_t)ctx[i] * mult[i];
    return h;
}
inline uint64_t mixed_step(uint64_t prev_hash, int64_t ctx_j, uint64_t mult_j) { return prev_hash ^ ((uint64_t)ctx_j * mult_j); }

// Head row index stays inside the head's vocabulary window [off, off + v).
inline uint64_t head_row(uint64_t m, uint64_t v, uint64_t off) { return m % v + off; }

}  // namespace plectx
}  // namespace baloon
