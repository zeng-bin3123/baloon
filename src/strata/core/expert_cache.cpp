// strata/core/expert_cache.cpp — VRAM expert blob cache implementation
#include "strata/core/expert_cache.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstring>
#include <numeric>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata {

// ─── Profile magic ───────────────────────────────────────────────────
static constexpr uint32_t kProfileMagic = 0x50525453;  // 'STRP' little-endian

// Helper: read exactly n bytes from fd, returns false on short/error.
static bool read_exact(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < n) {
        ssize_t got = ::read(fd, p + done, n - done);
        if (got <= 0) return false;
        done += static_cast<size_t>(got);
    }
    return true;
}

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked,
                         int64_t& slots, std::string& err) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "cannot open profile: " + path + ": " + strerror(errno);
        return false;
    }

    // Header: magic, version, n_layers, n_expert, slots, n_ranked
    uint32_t magic = 0;
    int32_t version = 0;
    int32_t hdr_layers = 0, hdr_expert = 0, hdr_slots = 0, hdr_ranked = 0;

    bool ok = read_exact(fd, &magic, 4) &&
              read_exact(fd, &version, 4) &&
              read_exact(fd, &hdr_layers, 4) &&
              read_exact(fd, &hdr_expert, 4) &&
              read_exact(fd, &hdr_slots, 4) &&
              read_exact(fd, &hdr_ranked, 4);

    if (!ok) {
        err = "profile header read failed: " + path;
        ::close(fd);
        return false;
    }
    if (magic != kProfileMagic) {
        err = "bad profile magic";
        ::close(fd);
        return false;
    }
    if (hdr_layers <= 0 || hdr_expert <= 0 || hdr_ranked < 0) {
        err = "invalid profile dimensions";
        ::close(fd);
        return false;
    }

    slots = hdr_slots;

    ranked.resize(static_cast<size_t>(hdr_ranked));
    for (int32_t i = 0; i < hdr_ranked; ++i) {
        int32_t l = 0, e = 0;
        if (!read_exact(fd, &l, 4) || !read_exact(fd, &e, 4)) {
            err = "profile truncated at entry " + std::to_string(i);
            ::close(fd);
            return false;
        }
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "profile entry out of range: (" + std::to_string(l) +
                  ", " + std::to_string(e) + ")";
            ::close(fd);
            return false;
        }
        ranked[i] = {l, e};
    }

    ::close(fd);
    return true;
}

// ─── ExpertCache ─────────────────────────────────────────────────────

ExpertCache::~ExpertCache() {
    close();
}

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert,
                       int64_t blob_bytes, std::string& err) {
    if (n_slots <= 0 || n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "ExpertCache::open: invalid dimensions";
        return false;
    }

    close();

    int64_t total = n_slots * blob_bytes;
    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&base_), static_cast<size_t>(total));
    if (rc != cudaSuccess) {
        err = std::string("ExpertCache::open cudaMalloc failed: ") + cudaGetErrorString(rc);
        base_ = nullptr;
        return false;
    }

    total_bytes_ = total;
    slots_ = n_slots;
    blob_ = blob_bytes;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    next_free_ = 0;
    per_layer_ = false;

    // Initialize residency: all kNotResident
    residency_.assign(static_cast<size_t>(n_layers * n_expert), kNotResident);
    off_.clear();
    layer_next_free_.clear();

    return true;
}

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes,
                             int64_t n_layers, int64_t n_expert, std::string& err) {
    if (slot_bytes.empty() || n_layers <= 0 || n_expert <= 0) {
        err = "ExpertCache::open_sized: invalid dimensions";
        return false;
    }

    close();

    // Build offset table: off_[0] = 0, off_[i+1] = off_[i] + slot_bytes[i]
    int64_t n_slots = static_cast<int64_t>(slot_bytes.size());
    off_.resize(static_cast<size_t>(n_slots + 1));
    off_[0] = 0;
    for (int64_t i = 0; i < n_slots; ++i) {
        if (slot_bytes[i] <= 0) {
            err = "ExpertCache::open_sized: slot_bytes[" + std::to_string(i) + "] <= 0";
            return false;
        }
        off_[i + 1] = off_[i] + static_cast<uint64_t>(slot_bytes[i]);
    }

    int64_t total = static_cast<int64_t>(off_[n_slots]);
    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&base_), static_cast<size_t>(total));
    if (rc != cudaSuccess) {
        err = std::string("ExpertCache::open_sized cudaMalloc failed: ") + cudaGetErrorString(rc);
        base_ = nullptr;
        off_.clear();
        return false;
    }

    total_bytes_ = total;
    slots_ = n_slots;
    blob_ = 0;  // variable sizes
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    next_free_ = 0;
    per_layer_ = false;

    residency_.assign(static_cast<size_t>(n_layers * n_expert), kNotResident);
    layer_next_free_.clear();

    return true;
}

void ExpertCache::close() {
    if (base_) {
        cudaFree(base_);
        base_ = nullptr;
    }
    total_bytes_ = 0;
    slots_ = 0;
    blob_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    next_free_ = 0;
    per_layer_ = false;
    fills_ = 0;
    admits_ = 0;
    residency_.clear();
    off_.clear();
    layer_next_free_.clear();
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) {
        return kNotResident;
    }
    return residency_[static_cast<size_t>(layer * n_expert_ + expert)];
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) {
        return kNotResident;
    }

    size_t idx = static_cast<size_t>(layer * n_expert_ + expert);

    // Already resident?
    if (residency_[idx] != kNotResident) {
        return residency_[idx];
    }

    if (per_layer_) {
        // Per-layer mode: each layer gets its own slot range.
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);

        // Initialize per-layer free counters on first use.
        if (layer_next_free_.empty()) {
            layer_next_free_.resize(static_cast<size_t>(n_layers_), 0);
            for (int64_t l = 0; l < n_layers_; ++l) {
                int64_t llo = 0, lhi = 0;
                layer_slot_range(l, llo, lhi);
                layer_next_free_[l] = llo;
            }
        }

        int64_t& nf = layer_next_free_[static_cast<size_t>(layer)];
        if (nf >= hi) {
            return kNotResident;  // layer partition full
        }

        int32_t slot = static_cast<int32_t>(nf);
        ++nf;
        residency_[idx] = slot;
        ++admits_;
        return slot;
    } else {
        // Global mode
        if (next_free_ >= slots_) {
            return kNotResident;  // cache full
        }

        int32_t slot = static_cast<int32_t>(next_free_);
        ++next_free_;
        residency_[idx] = slot;
        ++admits_;
        return slot;
    }
}

void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    if (n_layers_ <= 0) {
        lo = hi = 0;
        return;
    }
    int64_t q = slots_ / n_layers_;
    lo = layer * q;
    if (layer == n_layers_ - 1) {
        hi = slots_;  // last layer gets remainder
    } else {
        hi = (layer + 1) * q;
    }
}

uint8_t* ExpertCache::device_slot(int32_t slot) {
    if (!off_.empty()) {
        return base_ + off_[static_cast<size_t>(slot)];
    }
    return base_ + static_cast<int64_t>(slot) * blob_;
}

const uint8_t* ExpertCache::device_slot(int32_t slot) const {
    if (!off_.empty()) {
        return base_ + off_[static_cast<size_t>(slot)];
    }
    return base_ + static_cast<int64_t>(slot) * blob_;
}

bool ExpertCache::fill_slot(int32_t slot, const uint8_t* host_blob,
                            void* stream, std::string& err, int64_t bytes) {
    if (slot < 0 || slot >= static_cast<int32_t>(slots_)) {
        err = "fill_slot: invalid slot " + std::to_string(slot);
        return false;
    }

    int64_t nbytes = bytes > 0 ? bytes : blob_;
    if (nbytes <= 0) {
        err = "fill_slot: zero byte count";
        return false;
    }

    uint8_t* dst = device_slot(slot);
    cudaError_t rc = cudaMemcpyAsync(dst, host_blob,
                                      static_cast<size_t>(nbytes),
                                      cudaMemcpyHostToDevice,
                                      static_cast<cudaStream_t>(stream));
    if (rc != cudaSuccess) {
        err = std::string("fill_slot cudaMemcpyAsync: ") + cudaGetErrorString(rc);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t* host_blob,
                                     std::string& err, int64_t bytes) {
    if (slot < 0 || slot >= static_cast<int32_t>(slots_)) {
        err = "fill_slot_blocking: invalid slot " + std::to_string(slot);
        return false;
    }

    int64_t nbytes = bytes > 0 ? bytes : blob_;
    if (nbytes <= 0) {
        err = "fill_slot_blocking: zero byte count";
        return false;
    }

    uint8_t* dst = device_slot(slot);
    cudaError_t rc = cudaMemcpy(dst, host_blob,
                                 static_cast<size_t>(nbytes),
                                 cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) {
        err = std::string("fill_slot_blocking cudaMemcpy: ") + cudaGetErrorString(rc);
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t* host_blob,
                              std::string& err, int64_t bytes) {
    if (slot < 0 || slot >= static_cast<int32_t>(slots_)) {
        err = "verify_slot: invalid slot " + std::to_string(slot);
        return false;
    }

    int64_t nbytes = bytes > 0 ? bytes : blob_;
    if (nbytes <= 0) {
        err = "verify_slot: zero byte count";
        return false;
    }

    std::vector<uint8_t> readback(static_cast<size_t>(nbytes));
    const uint8_t* src = device_slot(slot);
    cudaError_t rc = cudaMemcpy(readback.data(), src,
                                 static_cast<size_t>(nbytes),
                                 cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) {
        err = std::string("verify_slot cudaMemcpy D2H: ") + cudaGetErrorString(rc);
        return false;
    }

    if (std::memcmp(readback.data(), host_blob, static_cast<size_t>(nbytes)) != 0) {
        err = "verify_slot: data mismatch at slot " + std::to_string(slot);
        return false;
    }
    return true;
}

int32_t ExpertCache::batch_lookup(int64_t layer, const int32_t* ids,
                                   int32_t* slots_out, int32_t k) const {
    int32_t hits = 0;
    for (int32_t i = 0; i < k; ++i) {
        int32_t s = slot_of(layer, ids[i]);
        slots_out[i] = s;
        if (s != kNotResident) ++hits;
    }
    return hits;
}

bool ExpertCache::adaptive_swap_layer(int64_t layer,
                                       const int32_t* route_counts,
                                       const uint8_t* (*blob_source)(int64_t, int64_t, void*),
                                       void* ctx,
                                       void* stream,
                                       std::string& err) {
    if (layer < 0 || layer >= n_layers_ || n_expert_ <= 0) return false;
    if (!base_ || !per_layer_) return false;

    int64_t lo = 0, hi = 0;
    layer_slot_range(layer, lo, hi);

    // Find the most-routed missing expert for this layer
    int32_t best_miss_expert = -1;
    int32_t best_miss_count = -1;
    for (int64_t e = 0; e < n_expert_; ++e) {
        size_t idx = static_cast<size_t>(layer * n_expert_ + e);
        if (residency_[idx] == kNotResident) {
            int32_t c = route_counts[idx];
            if (c > best_miss_count) {
                best_miss_count = c;
                best_miss_expert = static_cast<int32_t>(e);
            }
        }
    }
    if (best_miss_expert < 0 || best_miss_count <= 0) return false;

    // Find the least-routed resident expert in this layer's slot range
    int32_t worst_res_expert = -1;
    int32_t worst_res_slot = -1;
    int32_t worst_res_count = INT32_MAX;
    for (int64_t e = 0; e < n_expert_; ++e) {
        size_t idx = static_cast<size_t>(layer * n_expert_ + e);
        int32_t s = residency_[idx];
        if (s != kNotResident && s >= static_cast<int32_t>(lo) && s < static_cast<int32_t>(hi)) {
            int32_t c = route_counts[idx];
            if (c < worst_res_count) {
                worst_res_count = c;
                worst_res_expert = static_cast<int32_t>(e);
                worst_res_slot = s;
            }
        }
    }
    if (worst_res_expert < 0) return false;

    // Only swap if the miss is more frequently routed
    if (best_miss_count <= worst_res_count) return false;

    // Get the blob for the new expert
    const uint8_t* blob = blob_source(layer, best_miss_expert, ctx);
    if (!blob) {
        err = "adaptive_swap: blob_source returned null for (" +
              std::to_string(layer) + ", " + std::to_string(best_miss_expert) + ")";
        return false;
    }

    // Copy into the evicted slot
    if (!fill_slot(worst_res_slot, blob, stream, err)) {
        return false;
    }

    // Update residency
    size_t old_idx = static_cast<size_t>(layer * n_expert_ + worst_res_expert);
    size_t new_idx = static_cast<size_t>(layer * n_expert_ + best_miss_expert);
    residency_[old_idx] = kNotResident;
    residency_[new_idx] = worst_res_slot;

    return true;
}

}  // namespace strata

