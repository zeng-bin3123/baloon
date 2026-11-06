// rt_pinned.hpp - page-locked host arena for streaming weights to the device, with checksummed loads.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace baloon {
namespace rt {

constexpr uint64_t kFnvBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
// FNV-1a over n bytes, continuing from `seed` (left fold: fnv(a||b) == fnv(b, seed=fnv(a))).
uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed = kFnvBasis);

// parts+1 boundaries splitting [0,total): bounds[0]=0, bounds[parts]=total, every piece is floor or ceil of total/parts.
// parts == 0 yields an empty vector.
std::vector<uint64_t> uniform_bounds(uint64_t total, uint64_t parts);

// Drains the sticky CUDA error state; returns the number of errors cleared.
int clear_error();

enum class Backing { None, Pageable, Pinned, HugePinned };

class PinArena {
public:
    PinArena();
    ~PinArena();
    PinArena(const PinArena&) = delete;
    PinArena& operator=(const PinArena&) = delete;
    // Maps `bytes` (>0) of zeroed host memory, asks for 2 MiB pages first and falls back to normal pages, then
    // page-locks it in up to `slices` page-aligned, roughly equal registrations (never more than the page count). Pinning failure
    // degrades to Backing::Pageable rather than failing. Returns false only when no memory could be mapped.
    bool reserve(uint64_t bytes, uint64_t slices = 1);
    void release();
    uint8_t* data() const { return p_; }
    uint64_t bytes() const { return bytes_; }
    Backing backing() const { return backing_; }
    uint64_t slices() const { return bounds_.empty() ? 0 : bounds_.size() - 1; }
    uint64_t registered_bytes() const { return registered_; }   // sum of successfully registered slice sizes
private:
    uint8_t* p_ = nullptr;
    uint64_t bytes_ = 0, map_bytes_ = 0, registered_ = 0;
    Backing backing_ = Backing::None;
    std::vector<uint64_t> bounds_;
    std::vector<bool> pinned_;
};

using Reader = std::function<bool(uint64_t off, uint64_t n, uint8_t* dst)>;
struct LoadReport { uint64_t bytes = 0, reads = 0, checksum = kFnvBasis; bool ok = true; };
struct BandwidthReport { uint64_t bytes = 0, passes = 0; double ms = 0, gbps = 0; bool ok = true; };

// Layer L of `blob_bytes`-sized blobs (`blobs` per layer) is read from source offset L*blobs*blob_bytes into the same
// arena offset, in `chunk`-byte reads. The checksum is the left-fold FNV over all bytes in order.
LoadReport load_experts(PinArena& a, const Reader& rd, uint64_t layers, uint64_t blobs, uint64_t blob_bytes, uint64_t chunk);
// Arbitrary (offset,length) source ranges packed back to back from arena offset 0.
LoadReport load_experts_ranges(PinArena& a, const Reader& rd, const std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t chunk);
// Streams the arena to `dst` (device, >= bytes) `passes` times in `chunk`-byte async copies; only whole chunks move.
BandwidthReport stream_bandwidth(PinArena& a, void* dst, uint64_t chunk, uint64_t passes, cudaStream_t stream = nullptr);

}  // namespace rt
}  // namespace baloon
