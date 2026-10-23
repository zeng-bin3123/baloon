#pragma once
// strata/core/device_arena.hpp — GPU VRAM bump allocator
//
// Manages a contiguous VRAM region from which callers bump-allocate
// sub-regions.  Every allocation is NaN-filled so that use of
// uninitialized memory propagates NaN rather than producing a
// plausible wrong answer.
//
// The arena is allocated once during startup.  Sub-allocations are
// not individually freed; call reset() to reclaim all sub-allocations,
// or destroy the arena to free the VRAM.

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata {

// ─── DeviceArena ─────────────────────────────────────────────────────
// Bump allocator on device (GPU) memory.  NOT thread-safe.
struct DeviceArena {
    DeviceArena() = default;
    ~DeviceArena();

    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // Allocate `total_bytes` of device memory, NaN-fill the whole region.
    // Returns false and fills `err` on failure.
    bool open(uint64_t total_bytes, std::string& err);

    // Bump-allocate `bytes` from the arena.
    // `alignment` must be a power of 2 (default 256 for GPU coalescing).
    // The returned pointer points to NaN-filled memory.
    // Returns nullptr if the arena is exhausted.
    void* alloc(uint64_t bytes, uint64_t alignment = 256);

    // How many bytes remain.
    uint64_t remaining() const { return capacity_ - cursor_; }

    // Total capacity.
    uint64_t capacity() const { return capacity_; }

    // Current cursor position (bytes allocated so far).
    uint64_t used() const { return cursor_; }

    // Base pointer.
    void* base() const { return base_; }

    // Is the arena usable?
    bool valid() const { return base_ != nullptr; }

    // Reset the bump cursor to the start (re-NaN-fills the arena).
    // Returns false on NaN-fill failure.
    bool reset(std::string& err);

    // Release the VRAM (called automatically by destructor).
    void close();

private:
    void* base_       = nullptr;
    uint64_t capacity_ = 0;
    uint64_t cursor_   = 0;

    // NaN-fill [base_, base_ + capacity_).
    bool nan_fill(std::string& err);
};

}  // namespace strata

