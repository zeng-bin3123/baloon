#pragma once
// strata/core/device.hpp — CUDA device init and NaN-filled arena allocation
// Target: Iluvatar BI-V100 (ivcore10)
//
// NaN fill rationale: uninitialized GPU memory reads as 0, which sums to a
// finite contribution and produces a plausible wrong answer.  Filling with
// NaN (0x7FC00000) ensures that any use of uninitialized memory propagates
// NaN through the computation, making the bug visible.

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata {

// ─── DeviceInfo ──────────────────────────────────────────────────────
// GPU properties relevant to the runtime's memory planning.
struct DeviceInfo {
    int sm_count = 0;          // multiprocessor count
    size_t vram_total = 0;     // total global memory in bytes
    size_t vram_free = 0;      // free global memory in bytes
    int cc_major = 0;          // compute capability major
    int cc_minor = 0;          // compute capability minor
};

// ─── CudaError ───────────────────────────────────────────────────────
// Exception for CUDA runtime failures.
struct CudaError {
    std::string message;
    int code = 0;
    CudaError(std::string msg, int c) : message(std::move(msg)), code(c) {}
};

// ─── device_init ─────────────────────────────────────────────────────
// Sets the active device, queries properties.
// Returns DeviceInfo on success, false on failure.
bool device_init(int device_id, DeviceInfo& info, std::string& err);

// ─── arena_alloc ─────────────────────────────────────────────────────
// Allocates `bytes` of device memory, fills with NaN pattern.
// Returns device pointer on success, nullptr on failure.
void* arena_alloc(size_t bytes, std::string& err);

// ─── arena_free ──────────────────────────────────────────────────────
// Frees device memory allocated by arena_alloc.
void arena_free(void* ptr);

// ─── device_mem_info ─────────────────────────────────────────────────
// Queries current free and total device memory.
bool device_mem_info(size_t& free_bytes, size_t& total_bytes, std::string& err);

}  // namespace strata

