// strata/core/pinned.cpp — Page-locked host memory, parallel loading, bandwidth bench
#include "strata/core/pinned.hpp"

#include <algorithm>
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

    // Register with CUDA (slice-by-slice fallback)
    if (slice_size == 0 || slice_size >= bytes) {
        // Try whole arena
        cudaError_t rc = cudaHostRegister(base, bytes, cudaHostRegisterDefault);
        if (rc == cudaSuccess) {
            registered_bytes = bytes;
            locked_bytes = bytes;
            slice_count = 1;
        } else {
            note += " (cudaHostRegister whole failed, unregistered)";
        }
    } else {
        // Slice-by-slice
        uint64_t off = 0;
        while (off < bytes) {
            uint64_t chunk = std::min(slice_size, bytes - off);
            cudaError_t rc = cudaHostRegister(
                static_cast<uint8_t*>(base) + off, chunk, cudaHostRegisterDefault);
            if (rc != cudaSuccess) break;
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

// Thread worker: opens its own fd, reads assigned range with pread.
static void load_thread(const std::string& path, uint8_t* dst,
                        uint64_t file_offset, uint64_t nbytes,
                        uint64_t chunk_size, uint64_t* checksum) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return;

    uint64_t done = 0;
    uint64_t h = 14695981039346656037ull;
    while (done < nbytes) {
        uint64_t want = std::min(chunk_size, nbytes - done);
        ssize_t got = pread(fd, dst + done, want, file_offset + done);
        if (got <= 0) break;
        // Accumulate checksum over what we just read
        for (ssize_t j = 0; j < got; ++j) {
            h ^= dst[done + j];
            h *= 1099511628211ull;
        }
        done += static_cast<uint64_t>(got);
    }
    close(fd);
    if (checksum) *checksum = h;
}

LoadStats load_experts(const std::string& path, uint8_t* dst,
                       uint64_t blob_size, uint64_t blobs_per_layer,
                       uint64_t n_layers, int n_threads, uint64_t chunk_size) {
    LoadStats stats;
    uint64_t layer_bytes = blob_size * blobs_per_layer;
    uint64_t total = layer_bytes * n_layers;
    stats.checksums.resize(n_layers, 0);

    auto t0 = std::chrono::steady_clock::now();

    // One thread per layer (up to n_threads in flight)
    std::vector<std::thread> threads;
    for (uint64_t layer = 0; layer < n_layers; ++layer) {
        while ((int)threads.size() >= n_threads) {
            // Join the oldest
            threads.front().join();
            threads.erase(threads.begin());
        }
        uint64_t off = layer * layer_bytes;
        threads.emplace_back(load_thread, std::cref(path), dst + off,
                             off, layer_bytes, chunk_size,
                             &stats.checksums[layer]);
    }
    for (auto& t : threads) t.join();

    auto t1 = std::chrono::steady_clock::now();
    stats.seconds = std::chrono::duration<double>(t1 - t0).count();
    stats.bytes_read = total;
    stats.layers_loaded = n_layers;
    return stats;
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst,
                              const std::vector<uint64_t>& offsets,
                              const std::vector<uint64_t>& sizes,
                              int n_threads, uint64_t chunk_size) {
    LoadStats stats;
    size_t n = std::min(offsets.size(), sizes.size());
    stats.checksums.resize(n, 0);

    auto t0 = std::chrono::steady_clock::now();

    uint64_t dst_off = 0;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < n; ++i) {
        while ((int)threads.size() >= n_threads) {
            threads.front().join();
            threads.erase(threads.begin());
        }
        threads.emplace_back(load_thread, std::cref(path), dst + dst_off,
                             offsets[i], sizes[i], chunk_size,
                             &stats.checksums[i]);
        stats.bytes_read += sizes[i];
        dst_off += sizes[i];
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

