// sampler_chain_contracts.hpp - index, window and tie-break contracts of the sampler.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace baloon {
namespace sampler {

constexpr int kMaxTopK = 64;          // KMAX
constexpr int kBlockThreads = 64;     // sampler launch block
constexpr int kGreedyThreads = 1024;  // greedy reduction block
constexpr int kGreedyWarps = kGreedyThreads / 32;  // 32-entry shared arrays

// hlen = clamp(penalty_last_n, 0, history_len); the window is the tail of the history row.
inline size_t window_len(int64_t last_n, size_t history_len) {
    if (last_n < 0) return 0;
    if (last_n < (int64_t)history_len) return (size_t)last_n;
    return history_len;
}
inline size_t window_start(int64_t last_n, size_t history_len) { return history_len - window_len(last_n, history_len); }

// Repetition (multiply when logit <= 0, divide otherwise), then frequency and one-shot presence.
// A token with count 0 is untouched.
inline float apply_penalties(float logit, uint32_t count, float repeat, float freq, float present) {
    if (count == 0) return logit;
    const float l = logit <= 0.f ? logit * repeat : logit / repeat;
    return l - ((float)count * freq + (count > 0 ? 1.f : 0.f) * present);
}

// (value, index) pair; larger value wins, ties go to the smaller index.
struct Pair { float v; uint32_t i; };
inline Pair combine(Pair a, Pair b) { return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a; }

inline Pair sentinel(uint32_t n_vocab) { return Pair{-std::numeric_limits<float>::infinity(), n_vocab}; }

// A sentinel result (index == n_vocab) maps to token 0.
inline uint32_t final_token(uint32_t wi, uint32_t n_vocab) { return wi < n_vocab ? wi : 0u; }

// Greedy argmax; equal values resolve to the smaller index.
inline uint32_t greedy_argmax(const float* logits, uint32_t n_vocab) {
    Pair best = sentinel(n_vocab);
    for (uint32_t i = 0; i < n_vocab; ++i) best = combine(best, Pair{logits[i], i});
    return final_token(best.i, n_vocab);
}

// Launch grid of 64-thread blocks covers all tokens.
inline size_t grid_blocks(size_t n) { return (n + kBlockThreads - 1) / kBlockThreads; }

// Per-token RNG counter is counter + t (pairwise distinct).
inline uint64_t rng_counter(uint64_t counter, uint64_t t) { return counter + t; }

inline int effective_top_k(int k) { return k < kMaxTopK ? k : kMaxTopK; }

// Number of selection passes of the sampled path: the top-k clamp (0 when top_k <= 0, which the sampled path refuses).
inline int k_passes(int top_k) { return top_k > 0 ? (top_k < kMaxTopK ? top_k : kMaxTopK) : 0; }
// Presence is a boolean, not the count.
inline uint32_t presence(uint32_t count) { return count > 0 ? 1u : 0u; }
// Total subtraction of the penalties for a token seen `count` times.
inline double penalty_sub(uint32_t count, double freq, double present) { return (double)count * freq + (double)presence(count) * present; }
// top-p cut after the min_keep floor, capped at the survivor count.
inline size_t cut_after(size_t cut, size_t min_keep, size_t n_keep) { return cut < min_keep ? (min_keep < n_keep ? min_keep : n_keep) : cut; }
// Philox draw: the top 24 bits of a 32-bit word over 2^24, strictly below 1.
inline float philox_unit(uint32_t word) { return (float)(word >> 8) * (1.0f / 16777216.0f); }

}  // namespace sampler
}  // namespace baloon
