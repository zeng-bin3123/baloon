// expert_source_blob_access.hpp - expert blob addressing and dispatch job/output indexing.
#pragma once
#include <cstdint>
#include <optional>
#include <vector>

namespace baloon {
namespace expert_source {

// Flat blob index layer*ne+expert with a separate expert-axis check (else it would alias into the next layer).
constexpr std::optional<uint64_t> blob_index(uint64_t ne, uint64_t blobs, uint64_t layer, uint64_t expert) {
    if (ne <= expert) return std::nullopt;
    if (blobs <= layer * ne + expert) return std::nullopt;
    return layer * ne + expert;
}

// A blob is pinned only if it ends inside the registered bytes.
constexpr bool pinned_ok(uint64_t off, uint64_t len, uint64_t pinned_bytes) { return off + len <= pinned_bytes; }

inline uint64_t misses(const std::vector<bool>& hit) {
    uint64_t n = 0;
    for (bool h : hit) n += h ? 0 : 1;
    return n;
}
inline uint64_t hit_count(const std::vector<bool>& hit) {
    uint64_t n = 0;
    for (bool h : hit) n += h ? 1 : 0;
    return n;
}

// Job table built by the dispatch loop: job array indexed by njobs (misses only), output rows indexed by i.
// Hit rows are skipped (the caller zeroes them). Returns for each miss its output row.
inline std::vector<uint64_t> job_output_rows(const std::vector<bool>& hit) {
    std::vector<uint64_t> rows;
    for (uint64_t i = 0; i < hit.size(); ++i)
        if (!hit[i]) rows.push_back(i);
    return rows;
}

// Missed experts the GPU reads over PCIe: (nmiss * pcie_num) >> 8.
constexpr uint64_t pcie_share(uint64_t nmiss, uint64_t num) { return nmiss * num / 256; }

}  // namespace expert_source
}  // namespace baloon
