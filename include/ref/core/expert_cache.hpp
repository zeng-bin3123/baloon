// expert_cache.hpp - reference declarations for the VRAM-resident expert cache.
// ExpertCache: slot table, per-layer admission, 256-aligned sized slots, fill/verify.
// kNotResident: sentinel value returned when a (layer, expert) pair has no slot.
// read_expert_profile: load ranked (layer, expert) pairs from a profile file.
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace baloon {
namespace core {

constexpr int32_t kNotResident = -1;

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         int64_t want, std::vector<std::pair<int32_t, int32_t>>& out);

class ExpertCache {
public:
    ExpertCache() = default;
    ~ExpertCache();
    bool open(int64_t n_layers, int64_t n_expert, int64_t slots, int64_t blob,
              bool per_layer = true);
    bool open_sized(int64_t n_layers, int64_t n_expert,
                    const std::vector<int64_t>& blobs, bool per_layer = true);
    void close();
    bool is_open() const;
    int32_t slot_of(int64_t layer, int64_t expert) const;
    int32_t admit(int64_t layer, int64_t expert);
    std::pair<int64_t, int64_t> layer_slot_range(int64_t layer) const;
    uint8_t* device_slot(int64_t slot);
    int64_t slot_bytes(int64_t slot) const;
    int64_t slots() const;
    int64_t arena_bytes() const;
    int64_t resident_count() const;
    int64_t pending_fills() const;
    bool fill_slot(int64_t slot, const void* src, int64_t bytes);
    bool fill_slot_blocking(int64_t slot, const void* src, int64_t bytes);
    void sync();
    bool verify_slot(int64_t slot, const void* src, int64_t bytes) const;
};

} // namespace core
} // namespace baloon
