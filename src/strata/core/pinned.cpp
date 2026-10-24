// strata/core/pinned.cpp — Page-locked host memory, parallel loading, bandwidth bench
#include "strata/core/pinned.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace strata {

// ─── PinnedArena ─────────────────────────────────────────────────────

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : slice_size(slice) {
    if (bytes == 0) return;
    capacity = bytes;

    // Try large pages first
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) {
        base = p;
        backing = PageBacking::LargePages;
        locked_bytes = bytes;
        note = "large pages (MAP_HUGETLB)";
    } else {
        // Fall back to normal pages
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            // Last resort: cudaMallocHost
            cudaError_t rc = cudaMallocHost(&base, bytes);
            if (rc != cudaSuccess) {
                // Clear sticky error
                (void)cudaGetLastError();
                base = nullptr;
                note = std::string("cudaMallocHost failed: ") + cudaGetErrorString(rc);
                return;
            }
            backing = PageBacking::PinnedByCuda;
            locked_bytes = bytes;
            registered_bytes = bytes;
            note = "cudaMallocHost";
            return;
        }
        base = p;
        backing = PageBacking::NormalPages;
        note = "normal pages";
    }

    // Register with CUDA BEFORE touching pages.
    // cudaHostRegister pins what is RESIDENT, so a pre-faulted region is expensive.
    //
    // Critical: cudaGetLastError() MUST be called after a failed registration
    // to clear the sticky error. Without clearing, the next cudaGetLastError()
    // (e.g. in gr_read) reports "out of memory" for an unrelated kernel.
    if (slice_size == 0 || slice_size >= bytes) {
        // Try whole arena
        cudaError_t rc = cudaHostRegister(base, bytes, cudaHostRegisterDefault);
        if (rc == cudaSuccess) {
            registered_bytes = bytes;
            locked_bytes = bytes;
            slice_count = 1;
        } else {
            // Clear the sticky CUDA error state
            (void)cudaGetLastError();
            note += " (cudaHostRegister whole failed, unregistered)";
        }
    } else {
        // Slice-by-slice registration
        uint64_t off = 0;
        while (off < bytes) {
            uint64_t chunk = std::min(slice_size, bytes - off);
            cudaError_t rc = cudaHostRegister(
                static_cast<uint8_t*>(base) + off, chunk, cudaHostRegisterDefault);
            if (rc != cudaSuccess) {
                // Clear sticky error so subsequent unrelated CUDA calls
                // don't see "out of memory" from this registration failure
                (void)cudaGetLastError();
                break;
            }
            slice_starts.push_back(off);
            registered_bytes = off + chunk;
            ++slice_count;
            off += chunk;
        }
        locked_bytes = registered_bytes;
    }
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds)
    : PinnedArena(bytes, 0) {
    // Re-register using explicit boundaries if the simple path already worked
    // (or use bounds as the slice guide).
    slice_starts = bounds;
}

PinnedArena::~PinnedArena() {
    if (!base) return;

    if (backing == PageBacking::PinnedByCuda) {
        cudaFreeHost(base);
    } else {
        // Unregister slices
        if (registered_bytes > 0) {
            if (slice_count <= 1) {
                cudaHostUnregister(base);
            } else {
                for (int i = 0; i < slice_count; ++i) {
                    uint64_t off = (i < (int)slice_starts.size()) ? slice_starts[i]
                                   : i * slice_size;
                    uint64_t chunk = std::min(slice_size, capacity - off);
                    cudaHostUnregister(static_cast<uint8_t*>(base) + off);
                }
            }
        }
        munmap(base, capacity);
    }
    base = nullptr;
}

// ─── FNV-1a 64-bit ──────────────────────────────────────────────────

uint64_t fnv1a64(const uint8_t* data, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

// ─── Parallel file loading ───────────────────────────────────────────

// Thread worker: opens its OWN file handle (no shared seek lock), claims
// layers via atomic counter — one layer per claim, no contention.
static void load_worker(const std::string& path, uint8_t* base_dst,
                        uint64_t layer_bytes, uint64_t n_layers,
                        uint64_t chunk_size, std::atomic<uint64_t>& next_layer,
                        uint64_t* checksums) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return;

    for (;;) {
        uint64_t layer = next_layer.fetch_add(1);
        if (layer >= n_layers) break;

        uint8_t* dst = base_dst + layer * layer_bytes;
        uint64_t file_offset = layer * layer_bytes;
        uint64_t done = 0;
        uint64_t h = 14695981039346656037ull;  // FNV-1a offset basis

        while (done < layer_bytes) {
            uint64_t want = std::min(chunk_size, layer_bytes - done);
            ssize_t got = pread(fd, dst + done, want,
                                static_cast<off_t>(file_offset + done));
            if (got <= 0) break;
            // Per-layer FNV-1a checksum so a corrupt read names WHICH layer
            for (ssize_t j = 0; j < got; ++j) {
                h ^= dst[done + j];
                h *= 1099511628211ull;
            }
            done += static_cast<uint64_t>(got);
        }

        checksums[layer] = h;
    }

    close(fd);
}

LoadStats load_experts(const std::string& path, uint8_t* dst,
                       uint64_t blob_size, uint64_t blobs_per_layer,
                       uint64_t n_layers, int n_threads, uint64_t chunk_size) {
    LoadStats stats;
    uint64_t layer_bytes = blob_size * blobs_per_layer;
    uint64_t total = layer_bytes * n_layers;
    stats.checksums.resize(n_layers, 0);

    auto t0 = std::chrono::steady_clock::now();

    // T threads, each with its own file handle.
    // Layer assignment: next_layer.fetch_add(1) — one layer per claim, no contention.
    std::atomic<uint64_t> next_layer{0};
    int effective_threads = std::min(n_threads, static_cast<int>(n_layers));
    if (effective_threads < 1) effective_threads = 1;

    std::vector<std::thread> threads;
    threads.reserve(effective_threads);
    for (int t = 0; t < effective_threads; ++t) {
        threads.emplace_back(load_worker, std::cref(path), dst,
                             layer_bytes, n_layers, chunk_size,
                             std::ref(next_layer), stats.checksums.data());
    }
    for (auto& t : threads) t.join();

    auto t1 = std::chrono::steady_clock::now();
    stats.seconds = std::chrono::duration<double>(t1 - t0).count();
    stats.bytes_read = total;
    stats.layers_loaded = n_layers;
    return stats;
}

// Worker for range-based loading: each thread claims ranges via atomic counter.
static void load_ranges_worker(const std::string& path, uint8_t* base_dst,
                                const uint64_t* file_offsets,
                                const uint64_t* range_sizes,
                                const uint64_t* dst_offsets,
                                uint64_t n_ranges, uint64_t chunk_size,
                                std::atomic<uint64_t>& next_range,
                                uint64_t* checksums) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return;

    for (;;) {
        uint64_t idx = next_range.fetch_add(1);
        if (idx >= n_ranges) break;

        uint8_t* dst = base_dst + dst_offsets[idx];
        uint64_t file_offset = file_offsets[idx];
        uint64_t nbytes = range_sizes[idx];
        uint64_t done = 0;
        uint64_t h = 14695981039346656037ull;

        while (done < nbytes) {
            uint64_t want = std::min(chunk_size, nbytes - done);
            ssize_t got = pread(fd, dst + done, want,
                                static_cast<off_t>(file_offset + done));
            if (got <= 0) break;
            for (ssize_t j = 0; j < got; ++j) {
                h ^= dst[done + j];
                h *= 1099511628211ull;
            }
            done += static_cast<uint64_t>(got);
        }

        checksums[idx] = h;
    }

    close(fd);
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst,
                              const std::vector<uint64_t>& offsets,
                              const std::vector<uint64_t>& sizes,
                              int n_threads, uint64_t chunk_size) {
    LoadStats stats;
    size_t n = std::min(offsets.size(), sizes.size());
    stats.checksums.resize(n, 0);

    // Precompute destination offsets (contiguous layout)
    std::vector<uint64_t> dst_offs(n);
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) {
        dst_offs[i] = acc;
        acc += sizes[i];
        stats.bytes_read += sizes[i];
    }

    auto t0 = std::chrono::steady_clock::now();

    std::atomic<uint64_t> next_range{0};
    int effective_threads = std::min(n_threads, static_cast<int>(n));
    if (effective_threads < 1) effective_threads = 1;

    std::vector<std::thread> threads;
    threads.reserve(effective_threads);
    for (int t = 0; t < effective_threads; ++t) {
        threads.emplace_back(load_ranges_worker, std::cref(path), dst,
                             offsets.data(), sizes.data(), dst_offs.data(),
                             static_cast<uint64_t>(n), chunk_size,
                             std::ref(next_range), stats.checksums.data());
    }
    for (auto& t : threads) t.join();

    auto t1 = std::chrono::steady_clock::now();
    stats.seconds = std::chrono::duration<double>(t1 - t0).count();
    stats.layers_loaded = n;
    return stats;
}

// ─── StreamStats bandwidth bench ─────────────────────────────────────

StreamStats stream_bandwidth(const uint8_t* src_pinned, uint64_t total_bytes,
                             uint64_t chunk_size, int iterations) {
    StreamStats stats;
    stats.chunk_size = chunk_size;

    void* d_buf = nullptr;
    cudaError_t rc = cudaMalloc(&d_buf, chunk_size);
    if (rc != cudaSuccess) return stats;

    cudaStream_t cs;
    cudaStreamCreate(&cs);

    auto t0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        uint64_t off = 0;
        while (off < total_bytes) {
            uint64_t n = std::min(chunk_size, total_bytes - off);
            cudaMemcpyAsync(d_buf, src_pinned + off, n,
                            cudaMemcpyHostToDevice, cs);
            off += n;
        }
        cudaStreamSynchronize(cs);
        stats.bytes_streamed += total_bytes;
    }

    auto t1 = std::chrono::steady_clock::now();
    stats.seconds = std::chrono::duration<double>(t1 - t0).count();

    cudaStreamDestroy(cs);
    cudaFree(d_buf);
    return stats;
}

}  // namespace strata

