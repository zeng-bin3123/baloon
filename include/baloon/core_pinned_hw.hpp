// core_pinned_hw.hpp - the real pinned host arena (large pages, cudaHostRegister in whole slices, resident lock for the
// unregistered tail) and the parallel per-layer expert load with an FNV-1a checksum per layer. Implemented in src/core/pinned.cu.
//
// Contracts (PinnedArenaLoader theory):
//  * layers are claimed from an atomic counter, so each layer is loaded by exactly one worker;
//  * layer L occupies [L*bpl*blob, (L+1)*bpl*blob): ranges abut and tile the whole source;
//  * a layer hash is accumulated chunk by chunk with the running value as the next seed (== hash of the concatenation), mod 2^64;
//  * sliced registration pins bytes/slice whole slices; the tail (< one slice) is locked resident; registered prefix + locked
//    range cover the arena exactly.
// Observed quirk kept on purpose: kRefFnvSeed is the standard FNV-1a offset basis with its last digit dropped.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace baloon {
namespace hw {

constexpr uint64_t kRefFnvSeed = 1469598103934665603ull;
constexpr uint64_t kStdFnvBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed = kRefFnvSeed);
// parts+1 boundaries over [0,total): bounds[0]=0, bounds[parts]=total, pieces differ by at most one. parts==0 -> empty.
std::vector<uint64_t> uniform_bounds(uint64_t total, uint64_t parts);
// Drains the sticky CUDA error state; returns how many errors were cleared.
int clear_error();

enum class Backing { None, Pageable, Pinned, HugePinned };

class PinnedArena {
public:
    PinnedArena();
    ~PinnedArena();
    PinnedArena(const PinnedArena&) = delete;
    PinnedArena& operator=(const PinnedArena&) = delete;
    // Maps `bytes` (>0) of zeroed memory (2 MiB pages first, normal pages as fallback), then registers bytes/slice whole slices
    // (slice 0 or >= bytes means one slice; a slice is rounded down to a 4096 multiple, minimum 4096). Registration stops at the
    // first failure; everything past the registered prefix is mlock'ed. Returns false only if no memory could be mapped.
    bool reserve(uint64_t bytes, uint64_t slice = 0);
    void release();
    uint8_t* data() const { return p_; }
    uint64_t bytes() const { return bytes_; }
    Backing backing() const { return backing_; }
    uint64_t slice_bytes() const { return slice_; }
    uint64_t registered_bytes() const { return registered_; }                 // prefix [0, registered)
    uint64_t locked_begin() const { return registered_; }                     // locked range = [registered, bytes)
    uint64_t locked_bytes() const { return bytes_ - registered_; }
    bool locked_ok() const { return locked_ok_; }                              // mlock of the tail succeeded (or tail empty)
private:
    uint8_t* p_ = nullptr;
    uint64_t bytes_ = 0, map_bytes_ = 0, registered_ = 0, slice_ = 0;
    Backing backing_ = Backing::None;
    bool locked_ok_ = true;
};

using Reader = std::function<bool(uint64_t off, uint64_t n, uint8_t* dst)>;   // must be callable from several threads

struct LoadResult {
    bool ok = true;
    uint64_t bytes = 0, reads = 0;
    std::vector<uint64_t> hashes;          // one per layer (load_experts) or per range (load_experts_ranges)
    std::vector<uint32_t> claimed_by;      // worker index that loaded each layer (load_experts)
};

// Loads `layers` layers of `bpl` blobs of `blob` bytes from source offset L*bpl*blob into the same arena offset, in `chunk`-byte
// reads, with `workers` threads claiming layers from an atomic counter. A failed read latches and stops every worker.
LoadResult load_experts(PinnedArena& a, const Reader& rd, uint64_t layers, uint64_t bpl, uint64_t blob, uint64_t chunk,
                        unsigned workers = 1);
// (offset,length) source ranges packed back to back from arena offset 0, one hash per range.
LoadResult load_experts_ranges(PinnedArena& a, const Reader& rd, const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                               uint64_t chunk);

struct Bandwidth { bool ok = true; uint64_t bytes = 0, passes = 0; double ms = 0, gbps = 0; };
// Streams the arena to device memory `dst` (>= bytes) `passes` times in `chunk`-byte async copies; only whole chunks move.
Bandwidth stream_bandwidth(PinnedArena& a, void* dst, uint64_t chunk, uint64_t passes, cudaStream_t stream = nullptr);

// Device-side FNV-1a: one thread hashes one `chunk`-byte piece of device memory `dev` (only whole chunks), each from `seed`, so a
// streamed arena can be verified on the GPU against the host's per-chunk hashes. Launches a __global__ kernel and synchronizes.
bool device_chunk_hashes(const void* dev, uint64_t bytes, uint64_t chunk, std::vector<uint64_t>& out, uint64_t seed = kRefFnvSeed,
                         cudaStream_t stream = nullptr);

}  // namespace hw
}  // namespace baloon
