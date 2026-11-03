// expert_source_dispatch.hpp - expert pool adapter contracts: pack geometry, blob addressing, hit/miss partition,
// PCIe share of verify-window misses, sticky failure latch, pinned-range test.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace baloon {
namespace source_dispatch {

constexpr uint64_t kBlobBytes = 1382400;
constexpr uint64_t kLayers = 48;
constexpr uint64_t kExperts = 512;
constexpr uint64_t kPackBlobs = kLayers * kExperts;          // 24576
constexpr uint64_t kPackBytes = kPackBlobs * kBlobBytes;     // 33973862400
static_assert(kPackBlobs == 24576, "pack blob count");
static_assert(kPackBytes == 33973862400ull, "pack size");

constexpr uint64_t blob_idx(uint64_t ne, uint64_t l, uint64_t e) { return l * ne + e; }
// Both axes checked: expert < ne is NOT implied by the flat index being inside the file.
constexpr bool blob_ok(uint64_t nl, uint64_t ne, uint64_t l, uint64_t e) { return l < nl && e < ne; }
constexpr uint64_t blob_offset(uint64_t ne, uint64_t l, uint64_t e, uint64_t blob_bytes) {
    return blob_idx(ne, l, e) * blob_bytes;
}

struct Split {
    std::vector<size_t> hits, misses;  // positions into the router index list
};
// Hits and misses partition the router indices.
inline Split split_router(const std::vector<bool>& is_hit) {
    Split s;
    for (size_t j = 0; j < is_hit.size(); ++j) (is_hit[j] ? s.hits : s.misses).push_back(j);
    return s;
}
// Row j seen by the combine: cpu[j] + gpu[j]; hit rows are zero on the CPU side, miss rows zero on the GPU side.
inline void combine_rows(const std::vector<bool>& is_hit, const std::vector<float>& expert_out,
                         std::vector<float>& cpu, std::vector<float>& gpu, std::vector<float>& parts) {
    const size_t k = is_hit.size();
    cpu.assign(k, 0.f); gpu.assign(k, 0.f); parts.assign(k, 0.f);
    for (size_t j = 0; j < k; ++j) {
        (is_hit[j] ? gpu[j] : cpu[j]) = expert_out[j];
        parts[j] = cpu[j] + gpu[j];
    }
}

// Distinct missed experts the GPU reads over PCIe: nmiss*pcie_num/256 (the last m in routing order).
constexpr uint64_t pcie_share(uint64_t nmiss, uint64_t pcie_num) { return (nmiss * pcie_num) >> 8; }
// Ranks [nmiss-m, nmiss) go to the GPU; returns first rank. Requires m <= nmiss.
constexpr uint64_t pcie_first_rank(uint64_t nmiss, uint64_t m) { return nmiss - m; }

// Sticky failure latch.
constexpr bool latch(bool failed, bool ok) { return failed || !ok; }
struct DispatchState {
    bool failed = false;
    std::string msg;
    void step(bool ok, const char* why = "") {
        const bool nf = latch(failed, ok);
        if (nf && !failed) msg = why;
        failed = nf;
    }
    void clear_message() { msg.clear(); }  // does NOT re-arm
};

// Blob is DMA-able only if its whole extent lies inside the registered range (overflow-safe).
constexpr bool pinned_at(uint64_t off, uint64_t bytes, uint64_t pinned_bytes) {
    return off <= pinned_bytes && bytes <= pinned_bytes - off;
}

}  // namespace source_dispatch
}  // namespace baloon
