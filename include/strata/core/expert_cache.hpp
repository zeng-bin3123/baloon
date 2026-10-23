#pragma once
// strata/core/expert_cache.hpp — VRAM expert blob cache with slot management
// Target: Iluvatar BI-V100 (ivcore10)
//
// Maps (layer, expert) pairs to VRAM slots.  No eviction: once a slot
// is full, new admits are refused until the cache is reset or re-opened.
// Supports both uniform blob sizes and per-slot variable sizes.
//
// Profile format (binary, little-endian):
//   [magic='STRP', version, n_layers, n_expert, slots, n_ranked]
//   followed by n_ranked (layer, expert) pairs ranked by routing frequency.

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata {

// Sentinel: (layer, expert) has no VRAM slot assigned.
inline constexpr int32_t kNotResident = -1;

// ─── read_expert_profile ─────────────────────────────────────────────
// Read a binary expert routing profile from disk.
// On success: ranked filled with (layer, expert) pairs, slots set.
// Returns false on I/O or format error, with err set.
bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked,
                         int64_t& slots, std::string& err);

// ─── ExpertCache ─────────────────────────────────────────────────────
// VRAM cache for expert weight blobs.  Allocates a contiguous device
// buffer and tracks which (layer, expert) occupies which slot.
//
// Two modes:
//   open()       — uniform blob size, one allocation
//   open_sized() — variable per-slot sizes with offset table
//
// admit() assigns the next free slot; no eviction.
// fill_slot() copies host data to the assigned device slot.
class ExpertCache {
public:
    ExpertCache() = default;
    ~ExpertCache();

    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    // ── Uniform-size open ────────────────────────────────────────────
    // Allocates n_slots * blob_bytes bytes of device memory.
    // Post: all residency entries = kNotResident, next_free_ = 0.
    bool open(int64_t n_slots, int64_t n_layers, int64_t n_expert,
              int64_t blob_bytes, std::string& err);

    // ── Variable-size open ───────────────────────────────────────────
    // Allocates sum(slot_bytes[i]) bytes of device memory.
    // off_[i+1] = off_[i] + slot_bytes[i], off_[0] = 0.
    bool open_sized(const std::vector<int64_t>& slot_bytes,
                    int64_t n_layers, int64_t n_expert, std::string& err);

    // ── Close / reset ────────────────────────────────────────────────
    void close();

    // ── Queries ──────────────────────────────────────────────────────
    // Returns the slot index for (layer, expert), or kNotResident.
    int32_t slot_of(int64_t layer, int64_t expert) const;

    bool is_open() const { return base_ != nullptr; }
    int64_t slots() const { return slots_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }
    int64_t blob_bytes() const { return blob_; }
    int64_t next_free() const { return next_free_; }
    bool per_layer() const { return per_layer_; }

    // Device base pointer (for GPU kernels that index by slot).
    const uint8_t* device_base() const { return base_; }

    // Residency table (host-side, size = n_layers * n_expert).
    const int32_t* residency() const { return residency_.data(); }

    // Per-slot offset table (sized mode only, size = slots + 1).
    const uint64_t* slot_offsets() const { return off_.empty() ? nullptr : off_.data(); }

    // ── Slot admission ───────────────────────────────────────────────
    // Assigns the next free slot for (layer, expert).
    // Returns slot index on success, kNotResident if full.
    // No eviction ever occurs.
    int32_t admit(int64_t layer, int64_t expert);

    // ── Per-layer slot partitioning ──────────────────────────────────
    // Enable/disable per-layer mode: each layer gets slots/n_layers
    // contiguous slots (last layer gets remainder).
    void set_per_layer(bool on) { per_layer_ = on; }

    // Returns [lo, hi) slot range for a given layer.
    void layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const;

    // Per-layer free counters (set in per_layer mode).
    const std::vector<int64_t>& layer_next_free() const { return layer_next_free_; }

    // ── Device slot access ───────────────────────────────────────────
    // Returns device pointer for given slot index.
    // Pre: 0 <= slot < slots_
    uint8_t* device_slot(int32_t slot);
    const uint8_t* device_slot(int32_t slot) const;

    // ── Host → Device transfer ───────────────────────────────────────
    // Async copy host_blob into device slot (ordered by stream).
    // bytes = 0 means use blob_ (uniform mode).
    bool fill_slot(int32_t slot, const uint8_t* host_blob,
                   void* stream, std::string& err, int64_t bytes = 0);

    // Blocking copy (synchronizes stream internally).
    bool fill_slot_blocking(int32_t slot, const uint8_t* host_blob,
                            std::string& err, int64_t bytes = 0);

    // Verify: read back device slot and compare with host_blob.
    bool verify_slot(int32_t slot, const uint8_t* host_blob,
                     std::string& err, int64_t bytes = 0);

    // ── Batch lookup ─────────────────────────────────────────────
    // Look up k expert ids for a layer; write slot indices into slots_out.
    // Returns number of hits (slot != kNotResident).
    int32_t batch_lookup(int64_t layer, const int32_t* ids,
                         int32_t* slots_out, int32_t k) const;

    // ── Adaptive swap ────────────────────────────────────────────
    // Swap the most-routed missing expert into the slot of the
    // least-routed resident expert for a given layer.
    // route_counts: [n_layers * n_expert] routing frequency counters.
    // blob_source: callback to get the host blob for a (layer, expert).
    // stream: CUDA stream for the async copy.
    // Returns true if a swap was performed.
    bool adaptive_swap_layer(int64_t layer,
                             const int32_t* route_counts,
                             const uint8_t* (*blob_source)(int64_t layer, int64_t expert, void* ctx),
                             void* ctx,
                             void* stream,
                             std::string& err);

    // ── Statistics ────────────────────────────────────────────────
    int64_t total_fills() const { return fills_; }
    int64_t total_admits() const { return admits_; }

private:
    uint8_t* base_ = nullptr;          // device memory base
    int64_t total_bytes_ = 0;          // total device allocation
    int64_t slots_ = 0;                // number of slots
    int64_t blob_ = 0;                 // bytes per slot (uniform mode)
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    int64_t next_free_ = 0;            // global free pointer
    bool per_layer_ = false;           // per-layer partitioning mode

    int64_t fills_ = 0;               // total fill_slot calls
    int64_t admits_ = 0;              // total successful admissions

    std::vector<int32_t> residency_;   // size = n_layers * n_expert
    std::vector<uint64_t> off_;        // size = slots + 1 (sized mode)
    std::vector<int64_t> layer_next_free_; // per-layer free counters
};

}  // namespace strata

