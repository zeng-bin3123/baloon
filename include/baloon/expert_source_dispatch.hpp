// Adapter between the expert blob stores and the CPU expert pool: blob addressing and bounds checks,
// geometry refusal, hit/miss split, cache counters, the latching failure flag, hit-timing counters,
// job compaction and the PCIe share of misses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace baloon::expert_source {

// ---- geometry and addressing -------------------------------------------------

inline constexpr uint64_t kBlob = 1382400;  // bytes per expert blob
inline constexpr uint64_t kH = 2560;        // embedding width the kernel is compiled for
inline constexpr uint64_t kFF = 640;        // expert feed-forward width the kernel is compiled for

// Any other widths are refused.
inline bool geometry_ok(uint64_t n_embd, uint64_t ff) { return n_embd == kH && ff == kFF; }

// Flat blob index, no padding.
inline uint64_t blob_index(uint64_t n_expert, uint64_t l, uint64_t e) { return l * n_expert + e; }
inline uint64_t blob_offset(uint64_t n_expert, uint64_t l, uint64_t e) { return blob_index(n_expert, l, e) * kBlob; }

// File-size check: exact equality with n_layers * n_expert * BLOB.
inline bool size_ok(uint64_t file_size, uint64_t n_layers, uint64_t n_expert) {
    return file_size == n_layers * n_expert * kBlob;
}

// Lookup guard: expert against its own axis first (an id >= n_expert aliases the next layer's
// expert under the flat index alone), then the flat index against the blob count.
inline bool blob_ok(uint64_t n_expert, uint64_t blobs, uint64_t l, uint64_t e) {
    return e < n_expert && blob_index(n_expert, l, e) < blobs;
}

// Activation scales: H / 32 Q8_0 chunks of 34 bytes.
inline constexpr uint64_t kQ8Chunks = kH / 32;            // 80
inline constexpr uint64_t kQ8HitBytes = kQ8Chunks * 34;   // 2720

// ---- hit / miss split --------------------------------------------------------

// CPU row for one router index: zero when the GPU owns it. GPU row: zero when the CPU owns it.
inline int64_t cpu_row(bool hit, int64_t v) { return hit ? 0 : v; }
inline int64_t gpu_row(bool hit, int64_t v) { return hit ? v : 0; }

// Hit decision from a residency lookup.
inline bool is_hit(const std::function<std::optional<uint32_t>(uint32_t)>& res, uint32_t e) {
    return res(e).has_value();
}

// Router weight applied once, by the combine: weighted sum of unweighted parts.
inline int64_t combine(const std::vector<int64_t>& ws, const std::vector<int64_t>& parts) {
    int64_t s = 0;
    size_t n = ws.size() < parts.size() ? ws.size() : parts.size();
    for (size_t i = 0; i < n; ++i) s += ws[i] * parts[i];
    return s;
}

// Hit path pointers present or absent; ready only when all of them are.
struct HitWiring {
    bool cache = false, cache_base = false, parts_out = false, hit_out = false, mixed = false,
         scratch = false, xq = false, xq_scale = false, d_slot = false, d_dst = false;
    bool ready() const {
        return cache && cache_base && parts_out && hit_out && mixed && scratch && xq && xq_scale && d_slot && d_dst;
    }
};

// ---- job compaction ----------------------------------------------------------

// Number of misses in a hit-flag row list (= number of CPU jobs).
inline size_t count_miss(const std::vector<bool>& hits) {
    size_t n = 0;
    for (bool h : hits) n += h ? 0 : 1;
    return n;
}

// Output row index of each CPU job, in order (job j computes output row result[j] >= j).
inline std::vector<size_t> miss_jobs(const std::vector<bool>& hits) {
    std::vector<size_t> out;
    for (size_t i = 0; i < hits.size(); ++i)
        if (!hits[i]) out.push_back(i);
    return out;
}

// ---- counters ----------------------------------------------------------------

struct Counters {
    uint64_t hits = 0, admitted = 0, refused = 0;
    // One lookup: resident is a hit, else admitted if a slot was free, else refused.
    void step(bool resident, bool admitted_ok) {
        if (resident) ++hits;
        else if (admitted_ok) ++admitted;
        else ++refused;
    }
    uint64_t total() const { return hits + admitted + refused; }
};

struct Timing {
    uint64_t ready = 0, late = 0;
    // One layer: done by pool return is ready, else late.
    void record(bool done) {
        if (done) ++ready;
        else ++late;
    }
};

// ---- failure latch -----------------------------------------------------------

struct FailState {
    bool failed = false;
    std::optional<std::string> msg;

    // Record a failure: the latch is set.
    void raise(const std::string& m) {
        failed = true;
        msg = m;
    }
    // Clearing the message never re-arms the adapter.
    void clear_msg() { msg.reset(); }
    // Dispatch short-circuits when latched.
    bool should_run() const { return !failed; }
};

// ---- PCIe share --------------------------------------------------------------

// Missed experts the GPU reads over PCIe: (nmiss * pcie_num) >> 8.
inline uint64_t pcie_share(uint64_t nmiss, uint64_t pcie_num) { return (nmiss * pcie_num) >> 8; }

}  // namespace baloon::expert_source

