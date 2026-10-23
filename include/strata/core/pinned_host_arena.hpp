#pragma once
// strata/core/pinned_host_arena.hpp — Page-locked host memory bump allocator
// for expert weight streaming.
//
// Wraps a PinnedArena and provides a bump-allocator interface: callers
// request contiguous byte ranges and receive a host pointer guaranteed to
// be registered with CUDA for DMA (where the underlying arena managed it).
//
// The arena is allocated once at startup (tens of GiB for the expert pool)
// and never freed until shutdown.  Sub-allocations are not individually freed;
// the entire arena is released when the PinnedHostArena is destroyed.

#include "strata/core/pinned.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata {

// ─── PinnedHostArena ─────────────────────────────────────────────────
// Bump allocator over a PinnedArena.  Thread-safety: NOT thread-safe.
// Allocation happens during init (single-threaded); streaming reads
// are const on the arena.
struct PinnedHostArena {
    PinnedHostArena() = default;

    // Reserve `total_bytes` of pinned host memory.
    // `slice_boundaries` guides per-slice CUDA registration when
    // whole-arena registration fails (empty = try whole, then give up).
    // Returns false and fills `err` on failure.
    bool reserve(uint64_t total_bytes,
                 const std::vector<uint64_t>& slice_boundaries,
                 std::string& err);

    // Bump-allocate `bytes` from the arena.
    // `alignment` must be a power of 2 (default 64 for cache-line alignment).
    // Returns nullptr if the arena is exhausted.
    uint8_t* alloc(uint64_t bytes, uint64_t alignment = 64);

    // How many bytes remain.
    uint64_t remaining() const { return capacity_ - cursor_; }

    // Total capacity.
    uint64_t capacity() const { return capacity_; }

    // Current cursor position (bytes allocated so far).
    uint64_t used() const { return cursor_; }

    // Base pointer (for computing offsets into the arena).
    uint8_t* base() const { return arena_ ? arena_->data() : nullptr; }

    // Is the arena usable?
    bool valid() const { return arena_ && arena_->valid(); }

    // How many bytes were successfully CUDA-registered.
    uint64_t registered_bytes() const { return arena_ ? arena_->registered_bytes : 0; }

    // Page backing mode.
    PageBacking backing() const { return arena_ ? arena_->backing : PageBacking::NormalPages; }

    // Reset the bump cursor to the start (for reuse, e.g. after a full reload).
    void reset() { cursor_ = 0; }

    // Stream bandwidth measurement (delegates to strata::stream_bandwidth).
    StreamStats measure_bandwidth(uint64_t chunk_size, int iterations) const;

private:
    std::unique_ptr<PinnedArena> arena_;
    uint64_t capacity_ = 0;
    uint64_t cursor_   = 0;
};

}  // namespace strata

