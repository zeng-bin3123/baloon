// core_expert_cache.hpp - the VRAM-resident expert tier (host model): profile validation, slot arena sizing, the
// (layer, expert) -> slot residency table, per-layer slot ranges for admission, variable-size slot offsets, bounded fills.
// Residency table is flat row-major layer*n_expert+expert; out-of-range queries are "not resident"; layer l owns
// [l*q, (l+1)*q) with q = slots/n_layers (last layer takes the remainder); admission never evicts; sized slots are
// 256-byte aligned back to back; fill_slot copies at most the slot's blob size.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace baloon {
namespace core {

constexpr int32_t kNotResident = -1;

// Header acceptance of a profile file: n_layers x n_expert must match the model, at most `want` pairs ranked.
inline bool profile_header_ok(int64_t nl, int64_t ne, int64_t ranked, int64_t want, int64_t n_layers, int64_t n_expert) {
    return nl == n_layers && ne == n_expert && ranked >= 0 && ranked <= want;
}
// Profile layout (little-endian int32): n_layers, n_expert, n_ranked, then n_ranked x (layer, expert).
bool parse_expert_profile(const uint8_t* data, size_t bytes, int64_t n_layers, int64_t n_expert, int64_t want,
                          std::vector<std::pair<int32_t, int32_t>>& out);
bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t want,
                         std::vector<std::pair<int32_t, int32_t>>& out);
std::vector<uint8_t> encode_expert_profile(int64_t n_layers, int64_t n_expert,
                                           const std::vector<std::pair<int32_t, int32_t>>& pairs);

constexpr int64_t cache_align256(int64_t n) { return (n + 255) / 256 * 256; }
constexpr int64_t fill_len(int64_t bytes, int64_t blob) { return (bytes > 0 && bytes <= blob) ? bytes : blob; }

class ExpertCache {
public:
    ExpertCache() = default;
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;
    ~ExpertCache();

    // Uniform slots: arena = slots * blob bytes.
    bool open(int64_t n_layers, int64_t n_expert, int64_t slots, int64_t blob, bool per_layer = true);
    // Variable-size slots: slot i holds blobs[i] bytes, starts 256-byte aligned, back to back (off_ has slots+1 entries).
    bool open_sized(int64_t n_layers, int64_t n_expert, const std::vector<int64_t>& blobs, bool per_layer = true);
    void close();
    bool is_open() const { return open_; }

    int32_t slot_of(int64_t layer, int64_t expert) const;           // kNotResident when absent or out of range
    int32_t admit(int64_t layer, int64_t expert);                   // slot, or kNotResident (full / out of range); never evicts
    std::pair<int64_t, int64_t> layer_slot_range(int64_t layer) const;  // [lo, hi); empty [0,0) out of range
    uint8_t* device_slot(int64_t slot);                             // start of the slot in the arena, nullptr out of range
    int64_t slot_bytes(int64_t slot) const;
    int64_t slots() const { return slots_; }
    int64_t arena_bytes() const { return (int64_t)arena_.size(); }
    int64_t resident_count() const { return resident_; }
    int64_t pending_fills() const { return pending_; }

    // Copy min(bytes or whole blob) into the slot; fill_slot leaves the copy "in flight" until sync.
    bool fill_slot(int64_t slot, const void* src, int64_t bytes);
    bool fill_slot_blocking(int64_t slot, const void* src, int64_t bytes);
    void sync() { pending_ = 0; }
    bool verify_slot(int64_t slot, const void* src, int64_t bytes) const;

private:
    bool open_ = false, per_layer_ = true;
    int64_t n_layers_ = 0, n_expert_ = 0, slots_ = 0, blob_ = 0, resident_ = 0, pending_ = 0, next_free_ = 0;
    std::vector<int32_t> table_;       // layer * n_expert + expert -> slot
    std::vector<int64_t> next_;        // per-layer admission cursor
    std::vector<int64_t> off_;         // slot offsets (slots + 1 entries)
    std::vector<int64_t> sized_;       // per-slot blob bytes (open_sized only)
    std::vector<uint8_t> arena_;
};

}  // namespace core
}  // namespace baloon
