// cpu_expert_pool.hpp - CPU expert thread pool arithmetic: task splitting, row-range segmentation across experts,
// Q2_0 / native routing by phase mode, expert batching, park accounting and core selection.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace baloon {
namespace cpu_pool {

constexpr uint64_t split_pt(uint64_t rows, uint64_t tasks, uint64_t i) { return rows * i / tasks; }

constexpr uint64_t threads(uint64_t n_workers, bool host_works) { return n_workers + (host_works ? 1 : 0); }
constexpr uint64_t tasks_per_phase(uint64_t n_workers, bool host_works) { return 3 * threads(n_workers, host_works); }
constexpr uint64_t parts_per_expert(uint64_t thr, uint64_t n) { return std::max<uint64_t>(1, (3 * thr + n - 1) / n); }

struct Seg { uint64_t expert, r0, r1; };

// Cut global rows [g0, g1) across experts of `per` rows each. Every step advances and never overshoots.
inline std::vector<Seg> segment(uint64_t per, uint64_t g0, uint64_t g1) {
    std::vector<Seg> out;
    if (per == 0) return out;
    uint64_t r = g0;
    while (r < g1) {
        uint64_t e = r / per, r0 = r % per;
        uint64_t r1 = std::min(per, r0 + (g1 - r));
        out.push_back({e, r0, r1});
        r += r1 - r0;
    }
    return out;
}

enum class Path { q2_gu, native_gu, q2_down, native_down, other };

// Phase mode 5 = gate/up, 6 = down; type id 42 = Q2_0.
constexpr Path route(uint64_t mode, uint64_t gu, uint64_t d) {
    return mode == 5 ? (gu == 42 ? Path::q2_gu : Path::native_gu)
         : mode == 6 ? (d == 42 ? Path::q2_down : Path::native_down)
                     : Path::other;
}

constexpr uint64_t batch_len(uint64_t cap, uint64_t n, uint64_t b0) { return std::min(cap, n - b0); }

// First core is reserved for the host.
inline std::vector<int> skip_first(const std::vector<int>& cores) {
    return cores.empty() ? cores : std::vector<int>(cores.begin() + 1, cores.end());
}
inline int core_for(const std::vector<int>& cores, uint64_t i) { return i < cores.size() ? cores[i] : -1; }
constexpr bool should_pin(int core) { return core >= 0; }

}  // namespace cpu_pool
}  // namespace baloon
