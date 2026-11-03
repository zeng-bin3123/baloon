// gr_gated_residual.hpp - workspace layout and index arithmetic of gr_read / gr_write (gated residual).
// gr_workspace_init lays five regions (xn, xq, lq, gated, lo), each rounded up to 16 bytes so none overlaps.
// gr_write: out[i] = R[i] + block_out[d] * w[c] with i = c*n_embd + d.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace gr {

inline uint64_t al(uint64_t n) { return (n + 15) / 16 * 16; }

struct Workspace { uint64_t off[5]; uint64_t size[5]; uint64_t total; };
inline Workspace workspace_layout(const uint64_t sizes[5]) {
    Workspace w{};
    uint64_t p = 0;
    for (int i = 0; i < 5; ++i) { w.off[i] = p; w.size[i] = sizes[i]; p += al(sizes[i]); }
    w.total = p;
    return w;
}

inline int stream_of(int64_t i, int64_t n) { return (int)(i / n); }
inline int col_of(int64_t i, int64_t n) { return (int)(i % n); }
inline void write(const float* R, const float* block_out, const float* w, float* out, int n, int hc) {
    for (int64_t i = 0; i < (int64_t)n * hc; ++i) out[i] = R[i] + block_out[col_of(i, n)] * w[stream_of(i, n)];
}
// Mean over streams: sum_c x[c*n+d] / hc.
inline void mean_streams(const float* x, float* out, int n, int hc) {
    for (int d = 0; d < n; ++d) { float s = 0.f; for (int c = 0; c < hc; ++c) s += x[(int64_t)c * n + d]; out[d] = s / (float)hc; }
}

constexpr int kDownWarps = 8, kGateRowsPerBlock = 8, kMeanThreads = 256;
inline int gate_blocks(int hcdim) { return (hcdim + kGateRowsPerBlock - 1) / kGateRowsPerBlock; }
inline int gate_row(int b, int w) { return b * kGateRowsPerBlock + w; }
inline int mean_blocks(int n) { return (n + kMeanThreads - 1) / kMeanThreads; }
inline int down_index(int w, int lane, int k) { return w * 32 + lane + kDownWarps * 32 * k; }
inline bool even_hc_dim(int n) { return (4 * n) % 2 == 0; }
inline int inject_threads(int hc) { return 32 * hc; }
constexpr size_t kScratchDoubles = 8, kScratchFloats = 8;

}  // namespace gr
}  // namespace baloon
