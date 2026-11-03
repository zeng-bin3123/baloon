// ple_block_indexing.hpp - history shift, conv tap rows, stream layout, gate and scratch sizing of the PLE block.
// History is row-fastest: element (row, channel) at row + hist*channel. Depthwise conv (kernel 4, dilation 3,
// history 9) reads rows hist-(kern-1-k)*dil = 0,3,6,9; only the last tap reads the new row (row == hist).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace ple_block {

constexpr int kHistRows = 9;
constexpr int kConvKernel = 4;
constexpr int kConvDilation = 3;
constexpr int kStreamCount = 4;
constexpr int kEmbeddingDim = 2560;
constexpr int kStreamDim = kStreamCount * kEmbeddingDim;  // 10240
constexpr float kGateFloor = 1e-6f;

// Shift one channel column up by one and append x last (column has n entries).
inline void advance_column(float* col, int n, float x) {
    for (int i = 0; i + 1 < n; ++i) col[i] = col[i + 1];
    col[n - 1] = x;
}
// Flat history buffer [channels][hist], row-fastest.
inline int64_t hist_index(int64_t row, int64_t nhist, int64_t c) { return row + nhist * c; }
inline void history_advance(float* hist, int nhist, int channels, const float* x) {
    for (int c = 0; c < channels; ++c) advance_column(hist + hist_index(0, nhist, c), nhist, x[c]);
}

inline int tap_row(int nhist, int kern, int dil, int k) { return nhist - (kern - 1 - k) * dil; }

// Depthwise conv output of one channel: sum_k w[k] * (tap k value). The column holds `nhist` stored rows;
// the row index nhist denotes the new value `x_new`.
inline float conv_channel(const float* col, int nhist, float x_new, const float* w, int kern, int dil) {
    float acc = 0.f;
    for (int k = 0; k < kern; ++k) {
        const int r = tap_row(nhist, kern, dil, k);
        acc += w[k] * (r == nhist ? x_new : col[r]);
    }
    return acc;
}

// Stream c element d of the grouped norm.
inline int64_t stream_index(int64_t c, int64_t d, int64_t n) { return c * n + d; }
inline int64_t gate_stream(int64_t i, int64_t n) { return i / n; }
inline int64_t value_index(int64_t i, int64_t n) { return i % n; }

inline float sgn(float s) { return s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f); }
// gate = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6)))
inline float gate(float s) {
    const float m = std::fabs(s) > kGateFloor ? std::fabs(s) : kGateFloor;
    const float z = sgn(s) * std::sqrt(m);
    return 1.f / (1.f + std::exp(-z));
}

inline uint64_t align16(uint64_t x) { return (x + 15) / 16 * 16; }
// Scratch: float region, Q8 activation image, BF16 embedding copy, 256 bytes of slack; regions rounded to 16.
struct Scratch { uint64_t float_off, q8_off, bf16_off, total; };
inline Scratch scratch_layout(uint64_t float_bytes, uint64_t q8_bytes, uint64_t bf16_bytes) {
    Scratch s;
    s.float_off = 0;
    s.q8_off = align16(float_bytes);
    s.bf16_off = s.q8_off + align16(q8_bytes);
    s.total = s.bf16_off + bf16_bytes + 256;
    return s;
}

}  // namespace ple_block
}  // namespace baloon
