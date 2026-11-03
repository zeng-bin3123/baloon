// session_state_layout.hpp - session arena layout and per-layer state addressing.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace session {

constexpr size_t kArenaAlign = 256;
constexpr int kLayers = 48;
constexpr int kQsaPeriod = 4;      // layer l is a QSA layer iff l % 4 == 3
constexpr int kNumGdn = 36;
constexpr int kNumQsa = 12;

inline size_t align_up_256(size_t n) { return (n + (kArenaAlign - 1)) / kArenaAlign * kArenaAlign; }

inline bool is_qsa_layer(int l) { return l % kQsaPeriod == kQsaPeriod - 1; }

// Index of a GDN layer = number of non-QSA layers before it (closed form l - l/4 on this geometry).
inline int gdn_index(int l) { return l - l / kQsaPeriod; }
// Index of a QSA layer among QSA layers.
inline int qsa_index(int l) { return l / kQsaPeriod; }

// Each GDN layer owns F = state + conv floats; the conv state starts where the recurrent state ends.
inline size_t gdn_state_floats(size_t state_floats, size_t conv_floats) { return state_floats + conv_floats; }
inline size_t gdn_slice_offset(size_t idx, size_t F) { return idx * F; }
inline size_t gdn_conv_offset(size_t idx, size_t state_floats, size_t conv_floats) {
    return idx * (state_floats + conv_floats) + state_floats;
}
inline size_t gdn_arena_floats(size_t n, size_t F) { return n * F; }

// QSA state arena: state 0 owns RoPE (`first` units), the rest borrow (`rest` units each).
inline size_t qsa_off(size_t first, size_t rest, size_t i) { return i == 0 ? 0 : first + (i - 1) * rest; }
inline size_t qsa_size(size_t first, size_t rest, size_t i) { return i == 0 ? first : rest; }
inline size_t qsa_arena(size_t first, size_t rest, size_t n) { return n == 0 ? 0 : first + (n - 1) * rest; }

// Ring sequence: at layer l the loops wait for ring number l + 1 (always positive, strictly monotone).
inline uint64_t ring_wait_value(int l) { return (uint64_t)l + 1u; }

// Five stage times are consecutive differences of prefix times (t0 = 0) and telescope to t5.
inline std::array<int64_t, 5> stage_times(int64_t t1, int64_t t2, int64_t t3, int64_t t4, int64_t t5) {
    return {t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4};
}

}  // namespace session
}  // namespace baloon
