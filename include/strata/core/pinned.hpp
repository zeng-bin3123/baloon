#pragma once
// strata/core/pinned.hpp — Page-locked host memory arena with large-page support
// Target: Iluvatar BI-V100 (ivcore10)
//
// At 33.97 GB, 4 KB pages → ~8.3M TLB entries (exceeds any TLB).
// Large pages (2 MB) reduce by 512×.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata {

// ─── PageBacking ─────────────────────────────────────────────────────
enum class PageBacking {
    LargePages,    // mmap MAP_HUGETLB — minimal TLB pressure
    NormalPages,   // mmap + cudaHostRegister — fallback
    PinnedByCuda   // cudaMallocHost — simplest, no large pages
};

// ─── PinnedArena ─────────────────────────────────────────────────────
// RAII arena for page-locked host memory.  When large-page allocation or
// whole-arena cudaHostRegister fails, falls back to slice-by-slice
// registration so that at least a prefix is pinned.
struct PinnedArena {
    void* base = nullptr;
    uint64_t capacity = 0;
    PageBacking backing = PageBacking::NormalPages;
    std::string note;
    uint64_t locked_bytes = 0;
    uint64_t registered_bytes = 0;
    uint64_t slice_size = 0;
    int slice_count = 0;
    std::vector<uint64_t> slice_starts;

    PinnedArena() = default;

    // Allocate `bytes` of page-locked host memory.
    // `slice` controls the registration chunk size (0 = register all at once).
    explicit PinnedArena(uint64_t bytes, uint64_t slice = 0);

    // Allocate with explicit per-slice boundaries (for multi-section arenas).
    PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds);

    ~PinnedArena();

    PinnedArena(const PinnedArena&) = delete;
    PinnedArena& operator=(const PinnedArena&) = delete;

    bool valid() const { return base != nullptr; }
    uint8_t* data() const { return static_cast<uint8_t*>(base); }
};

// ─── LoadStats ───────────────────────────────────────────────────────
// Statistics from loading expert weight blobs from disk.
struct LoadStats {
    double seconds = 0.0;
    uint64_t bytes_read = 0;
    uint64_t layers_loaded = 0;
    std::vector<uint64_t> checksums;  // per-layer FNV-1a checksums

    double gib_per_second() const {
        return seconds > 0 ? static_cast<double>(bytes_read) / (1024.0 * 1024.0 * 1024.0) / seconds : 0.0;
    }
};

// Load expert weight blobs from `path` into `dst`.
// Each thread opens its own file handle for portable parallel pread.
LoadStats load_experts(const std::string& path, uint8_t* dst,
                       uint64_t blob_size, uint64_t blobs_per_layer,
                       uint64_t n_layers, int n_threads, uint64_t chunk_size);

// Load arbitrary byte ranges (offset[i], size[i]) from `path` into `dst`.
LoadStats load_experts_ranges(const std::string& path, uint8_t* dst,
                              const std::vector<uint64_t>& offsets,
                              const std::vector<uint64_t>& sizes,
                              int n_threads, uint64_t chunk_size);

// ─── FNV-1a 64-bit ──────────────────────────────────────────────────
// Per-layer checksums enable identification of WHICH layer is corrupt.
// h0 = seed (default: FNV offset basis)
// h[i+1] = (h[i] ^ p[i]) * 1099511628211  mod 2^64
uint64_t fnv1a64(const uint8_t* data, uint64_t n,
                 uint64_t seed = 14695981039346656037ull);

// ─── StreamStats ─────────────────────────────────────────────────────
// Measures host→device streaming bandwidth.
// Central design constant: per token E = 663.6 MB of expert weights.
// Hit rate h ≈ 0.167.  Miss fraction (1-h)*E / bandwidth = ms per token.
struct StreamStats {
    double seconds = 0.0;
    uint64_t bytes_streamed = 0;
    uint64_t chunk_size = 0;

    double gib_per_second() const {
        return seconds > 0 ? static_cast<double>(bytes_streamed) / (1024.0 * 1024.0 * 1024.0) / seconds : 0.0;
    }
};

// Benchmark pinned→device copy bandwidth.
StreamStats stream_bandwidth(const uint8_t* src_pinned, uint64_t total_bytes,
                             uint64_t chunk_size, int iterations);

}  // namespace strata

