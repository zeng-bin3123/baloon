// session_driver.hpp - session arena sizing/carving (256-byte cursor), per-layer state slicing (GDN/QSA = 3:1),
// QSA arena offsets, timing event indices, prefix telescoping.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace session_driver {

constexpr uint64_t align_up256(uint64_t n) { return (n + 255) / 256 * 256; }
inline uint64_t aligned_total256(const std::vector<uint64_t>& l) {
    uint64_t t = 0;
    for (uint64_t b : l) t += align_up256(b);
    return t;
}
inline uint64_t take_all256(uint64_t used, const std::vector<uint64_t>& l) {
    for (uint64_t b : l) used = align_up256(used + b);
    return used;
}
inline uint64_t raw_sum(const std::vector<uint64_t>& l) {
    uint64_t t = 0;
    for (uint64_t b : l) t += b;
    return t;
}
// session_bytes: raw sum aligned once, plus 4096 slack (covers <= 7 takes).
inline uint64_t session_bytes(const std::vector<uint64_t>& l) { return align_up256(raw_sum(l)) + 4096; }

constexpr bool is_qsa(uint64_t l) { return l % 4 == 3; }
// Counters (closed form of "layers before l").
constexpr uint64_t gdn_before(uint64_t l) { return l - l / 4; }
constexpr uint64_t qsa_before(uint64_t l) { return l / 4; }
constexpr uint64_t kLayers = 48, kGdnLayers = 36, kQsaLayers = 12;

// Offset of QSA state i: state 0 at 0, state i>=1 at first + (i-1)*rest.
constexpr uint64_t qsa_off(uint64_t first, uint64_t rest, uint64_t i) { return i == 0 ? 0 : first + (i - 1) * rest; }

constexpr uint64_t prefix_event(uint64_t l, uint64_t k) { return l * 6 + k - 1; }  // k = 1..5
constexpr uint64_t prefix_final_event(uint64_t l) { return l * 6 + 5; }
constexpr uint64_t stage_event(uint64_t l, uint64_t k) { return l * 4 + k; }       // k = 0..3
constexpr uint64_t prefix_graph_index(uint64_t k) { return k - 1; }                // k = 1..5, 5 entries

// Stage times from prefix totals t1..t5: t1, t2-t1, ... telescoping to t5.
inline std::vector<int64_t> stage_times(const std::vector<int64_t>& t) {
    std::vector<int64_t> s;
    int64_t prev = 0;
    for (int64_t x : t) { s.push_back(x - prev); prev = x; }
    return s;
}

}  // namespace session_driver
}  // namespace baloon
