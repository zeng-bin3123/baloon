// qsa_native_kernels.hpp - host API and pure arithmetic of the unit-10 kernels:
//   * ple_native      : native PLE post-projection operators (signed-sqrt stream gate, value broadcast, dilated 4-tap
//                       conv + residual) at the fixed geometry N = 2560, H = 4 streams, D = N*H = 10240, history depth 9.
//   * qsa_native      : single-block indexer key append (tail / block pooling / rotation / spare row), tensor-tile style
//                       block scoring, and the radix-select top-k with the monotone 32-bit order key.
// Return codes of the launchers are cudaError_t (cudaErrorInvalidValue = rejected by validation, nothing launched).
// Everything marked inline below is plain C++ so the invariants can be tested without a device.
#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace baloon {

// ------------------------------------------------------------------------------------------------ PLE post-ops
namespace ple_native {

constexpr int kN = 2560;          // embedding width
constexpr int kH = 4;             // streams
constexpr int kD = kN * kH;       // 10240
constexpr int kHist = 9;          // history rows per channel (row-fastest: row + 9*channel)
constexpr int kTaps = 4;          // conv kernel, dilation 3
constexpr int kDilation = 3;
constexpr int kGateThreads = 512; // threads per stream in the gate kernel
constexpr int kGateSlots = 8;     // partial lanes d = tid + j*512, masked to d < N
constexpr float kGateFloor = 1e-6f;

// History row read by tap k < 3 (tap 3 reads the freshly normalised value instead).
inline int hist_row(int k) { return kDilation * k; }
inline int64_t hist_index(int64_t row, int64_t channel) { return row + (int64_t)kHist * channel; }
inline int64_t weight_index(int64_t channel, int k) { return channel * kTaps + k; }

// IEEE binary16 -> float (exact).
inline float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else {  // subnormal
            e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            bits = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) bits = s | 0x7F800000u | (m << 13);
    else bits = s | ((e + 112u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// gate = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6)))
inline float gate_of(float s) {
    const float m = std::fabs(s) > kGateFloor ? std::fabs(s) : kGateFloor;
    const float sg = s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f);
    return 1.0f / (1.0f + std::exp(-(sg * std::sqrt(m))));
}
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Arguments of the fused post-ops. Inputs are read-only, outputs are written.
struct Args {
    const float* normed;      // [D]   normalised key (the fresh value of conv tap 3)
    const float* gate_w;      // [D]   gate weights
    const float* value;       // [N]   value vector, broadcast to every stream
    const uint16_t* conv_w;   // [4*D] fp16 bits, tap k of channel c at c*4 + k
    const float* hist;        // [9*D] row-fastest history (not modified)
    const float* hidden;      // [D]   residual input
    float* gate;              // [H]   out: stream gates
    float* gated;             // [D]   out: value[i % N] * gate[i / N]
    float* resid;             // [D]   out: hidden + (gated + silu(conv))
};

// Validates the spans and runs gate -> broadcast -> conv/residual on `stream`.
// Accepted aliasing (exactly two): resid == hidden, or resid == gated (exact same pointer); both at once is rejected.
cudaError_t native_ple_postops(const Args& a, cudaStream_t stream);

}  // namespace ple_native

// ------------------------------------------------------------------------------------------------ QSA kernels
namespace qsa_native {

constexpr int kD = 128;           // indexer key width
constexpr int kR = 4;             // cells per pooled block
constexpr int kRows = 32;         // pooled rows per score block
constexpr int kTopK = 2048;
constexpr int kMaxWidth = kTopK + kR - 1;   // 2051
constexpr float kTailBias = 1.0e9f;
constexpr uint32_t kNanKey = 0u;            // order key reserved for NaN

// Monotone 32-bit key on float bit patterns: negative patterns reverse, non-negative ones are lifted above them.
inline uint32_t order_key_bits(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
// NaN is mapped to the reserved key 0, below every finite value.
inline uint32_t order_key_f(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) return kNanKey;
    return order_key_bits(u);
}
// Cells emitted by the threshold selection: all strictly above, plus the equal ones that fit the budget.
inline uint32_t sel_count(uint32_t gt, uint32_t eq, uint32_t width) { return gt + (eq < width - gt ? eq : width - gt); }
// Weight (cells) of pooled block b.
inline uint32_t block_weight(uint32_t n_kv, uint32_t n_bid, uint32_t b) { return b < n_bid ? (uint32_t)kR : n_kv - n_bid * kR; }

// Step acceptance of the score kernel.
inline bool score_step_ok(int n, int max_cells, int pos, int n_bid, int width) {
    const int w = n < kMaxWidth ? n : kMaxWidth;
    return n >= 1 && n <= max_cells && pos == n - 1 && n_bid == n / kR && width == w;
}

struct IndexerArgs {
    const float* key;      // [128] raw key of cell `pos`
    float* tail;           // [3][128] raw keys of slots 0..2
    float* pooled;         // [max_cells/4 + 1][128]
    float* dead;           // [128] spare row
    int pos, max_cells;
    long long pos_base;    // multiple of 4
    float freq_step, eps;
};
// Slot = pos % 4: slots 0..2 are stored raw; slot 3 pools tail + key, RMS-normalises, rotates dims 0..63 (NEOX pairs d, d+32)
// at position pos_base + 4*(pos/4), writes row pos/4 and moves the spare row to pos/4 + 1. pos 0 initialises row 0 and the spare.
bool native_qsa_indexer_enabled();
void native_qsa_indexer_set_enabled(bool on);
cudaError_t native_qsa_indexer_append(const IndexerArgs& a, cudaStream_t stream);

struct ScoreArgs {
    const float* pooled;       // [max_blocks][128]
    const float* iq;           // [4][128]
    const float* hw;           // [4]
    const float* bias;         // optional [nb]
    float* block_scores;       // optional out [nb]
    float* cell_scores;        // out [n]
    int pos, n, n_bid, width, max_cells, max_blocks;
};
// score(row) = sum_h hw[h] * relu(iq_h . pooled[row]) (+ bias[row]) (+1e9 for the tail row when n % 4 != 0);
// row r writes cells [4r, min(4r+4, n)). Accepted iff the step is consistent and max_blocks == max_cells/4 + 1.
bool native_qsa_score_enabled();
void native_qsa_score_set_enabled(bool on);
cudaError_t native_qsa_score(const ScoreArgs& a, cudaStream_t stream);

// Per-block scores for n_kv cells: nb = n_kv/4 + (n_kv % 4 != 0) blocks, same formula as native_qsa_score (including the
// +1e9 bias of an incomplete tail block, so the newest cells are always selected). rows >= nb pooled rows.
cudaError_t qsa_block_scores(const float* pooled, const float* iq, const float* hw, const float* bias, float* scores, int n_kv,
                             int rows, cudaStream_t stream);
// Exact top-`width` cells of the block scores (each block weighs 4 cells, the tail block n_kv - 4*n_bid): cells strictly above
// the threshold key plus the first (by ascending index) threshold-equal cells. out_ids: [width] ascending, padded with -1;
// *n_out = number selected (n_kv when n_kv <= width).
cudaError_t qsa_block_topk(const float* scores, int n_kv, int width, int32_t* out_ids, int32_t* n_out, cudaStream_t stream);

}  // namespace qsa_native
}  // namespace baloon

