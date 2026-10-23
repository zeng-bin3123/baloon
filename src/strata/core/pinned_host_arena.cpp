// strata/core/pinned_host_arena.cpp — Page-locked host memory bump allocator
#include "strata/core/pinned_host_arena.hpp"

#include <cstdio>

namespace strata {

bool PinnedHostArena::reserve(uint64_t total_bytes,
                              const std::vector<uint64_t>& slice_boundaries,
                              std::string& err) {
    if (total_bytes == 0) {
        err = "PinnedHostArena::reserve: zero-size reservation";
        return false;
    }
    if (arena_ && arena_->valid()) {
        err = "PinnedHostArena::reserve: already reserved";
        return false;
    }

    // Construct the underlying PinnedArena with slice boundaries.
    // The PinnedArena constructor handles:
    //   1. Large-page allocation (MAP_HUGETLB)
    //   2. Fallback to normal pages
    //   3. CUDA host registration (whole or per-slice)
    //   4. Fallback to cudaMallocHost
    if (slice_boundaries.empty()) {
        arena_ = std::make_unique<PinnedArena>(total_bytes, uint64_t(0));
    } else {
        arena_ = std::make_unique<PinnedArena>(total_bytes, slice_boundaries);
    }

    if (!arena_->valid()) {
        err = "PinnedHostArena::reserve: PinnedArena allocation failed — " + arena_->note;
        arena_.reset();
        return false;
    }

    capacity_ = total_bytes;
    cursor_ = 0;
    return true;
}

uint8_t* PinnedHostArena::alloc(uint64_t bytes, uint64_t alignment) {
    if (!valid() || bytes == 0) return nullptr;

    // Align cursor up to `alignment` (must be power of 2)
    uint64_t mask = alignment - 1;
    uint64_t aligned = (cursor_ + mask) & ~mask;

    if (aligned + bytes > capacity_) return nullptr;

    uint8_t* ptr = arena_->data() + aligned;
    cursor_ = aligned + bytes;
    return ptr;
}

StreamStats PinnedHostArena::measure_bandwidth(uint64_t chunk_size,
                                               int iterations) const {
    if (!valid() || cursor_ == 0) {
        return StreamStats{};
    }
    // Measure over the currently-used portion of the arena
    return stream_bandwidth(arena_->data(), cursor_, chunk_size, iterations);
}

}  // namespace strata

