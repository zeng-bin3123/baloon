// router_top_k_selection.hpp - selection order, clamp and launch sizing of the MoE router.
#pragma once
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace baloon {
namespace router_topk {

struct Cand { double value; uint64_t index; };

// a beats b: larger value, or equal value and smaller index.
constexpr bool better(const Cand& a, const Cand& b) { return a.value > b.value || (a.value == b.value && a.index < b.index); }
// Shuffle-combine step: keep b only if it beats a.
constexpr Cand pick(const Cand& a, const Cand& b) { return better(b, a) ? b : a; }

// k passes of a block-wide argmax over (value, index); each pass masks the winner. Returns up to k indices.
inline std::vector<uint64_t> top_k(const std::vector<double>& v, uint64_t k) {
    std::vector<bool> taken(v.size(), false);
    std::vector<uint64_t> out;
    for (uint64_t p = 0; p < k && p < v.size(); ++p) {
        bool have = false;
        Cand best{0.0, (uint64_t)v.size()};
        for (uint64_t e = 0; e < v.size(); ++e) {
            if (taken[e]) continue;
            Cand c{v[e], e};
            if (!have || better(c, best)) { best = c; have = true; }
        }
        taken[best.index] = true;
        out.push_back(best.index);
    }
    return out;
}

// Renormalisation divisor max(sum, 2^-14).
inline double clamp_divisor(double sum) { return std::max(sum, 1.0 / 16384.0); }

constexpr uint64_t round32(uint64_t t) { return (t + 31) / 32 * 32; }
constexpr uint64_t round16(uint64_t n) { return (n + 15) / 16 * 16; }
constexpr uint64_t launch_threads(uint64_t n_expert) { return round32(n_expert < 512 ? n_expert : 512); }
// Shared bytes: 16-byte-rounded taken mask, n doubles, n floats.
constexpr uint64_t smem_bytes(uint64_t n_expert) { return round16(n_expert) + n_expert * 8 + n_expert * 4; }

}  // namespace router_topk
}  // namespace baloon
