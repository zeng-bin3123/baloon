// verify_window_kernels.hpp - device helpers of the speculative verify window (multi-token gated-delta-net kernels and small
// gathers). Pure C++ (no CUDA headers); implemented in src/kernels/gpu/verify_window_kernels.cu.
// Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
// Token t's conv window is the last four entries of [hist0, hist1, hist2, x_0..x_t]; only earlier tokens are read for the
// three history taps. The per-token arithmetic is the single-token arithmetic with the same operation order. Cross-lane
// reductions use 32-lane segments, so results do not depend on the wave width (64 on ivcore10).
// n_keep is a DEVICE value (*n_keep_dev) so a recorded graph can be replayed with a different accept count; it is clamped
// to [0, T] on the device.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace vwin {

constexpr int S = 128;            // gated-delta-net state width
constexpr int RG = 4;             // row groups of the fused step (512 threads = RG * S)
constexpr int RPG = 32;           // rows per group (RG * RPG = S)
constexpr uint32_t kMaxT = 8;     // widest verify window
constexpr int kTaps = 4;          // 3 history entries + current input
constexpr int kIdentMax = 1024;   // ident_hits: at most 1024 entries
constexpr int kRebaseMax = 128;   // rebase_ptrs: 128 threads
constexpr int kMapMax = 64;       // map_ids: at most 64 ids
constexpr int kCopyBlocksCap = 64;

// ---- pure index arithmetic (host model of the device formulas) -----------------------------------------------------
inline uint64_t rowmajor(uint64_t t, uint64_t h, uint64_t r) { return t * h + r; }
inline uint32_t seq_src(uint32_t t, uint32_t j) { return t + j; }                // entry of [hist0..2, x0, ..]
inline bool seq_is_hist(uint32_t e) { return e < 3; }
inline uint32_t seq_row(uint32_t e) { return e - 3; }                            // input row of a non-history entry
inline int32_t keep_clamp(int32_t n, int32_t T) { return n < 0 ? 0 : (n > T ? T : n); }
inline int32_t win_start(int32_t n_kv, int32_t window) { return n_kv > window ? n_kv - window : 0; }
inline int32_t win_width(int32_t n_kv, int32_t window) { return n_kv - win_start(n_kv, window); }
inline uint32_t copy_blocks(uint64_t n) { const uint64_t b = (n + 255) / 256; return (uint32_t)(b < kCopyBlocksCap ? b : kCopyBlocksCap); }
inline bool bits_ok(int bits) { return bits == 1 || bits == 2 || bits == 4 || bits == 8; }
inline uint64_t codes_per_byte(int bits) { return 8u / (unsigned)bits; }
inline unsigned unpack_code(const uint8_t* row, uint64_t k, int bits) {
    const uint64_t cpb = codes_per_byte(bits);
    return (row[k / cpb] >> ((k % cpb) * (uint64_t)bits)) & ((1u << bits) - 1u);
}

// ---- device entry points (host launchers) ---------------------------------------------------------------------------
// conv + l2 over a T-token window. x [T][C], hist [3][C] (read only), w [C][4], out [T][C], C == 128*(2*hk + hv); the first
// 2*hk*128 channels (q then k heads) are l2-normalised, x / max(norm, eps); the remaining channels are plain conv outputs.
int gdn_conv_l2_multi(const float* x, const float* hist, const float* w, float* out, int T, int C, int hk, int hv, float eps,
                      void* stream);
// New history = last three entries of [hist | x_0..x_{n-1}] in place with n = clamp(*n_keep_dev, 0, T); n = 0 keeps hist.
int gdn_conv_commit(const float* x, float* hist, const int32_t* n_keep_dev, int T, int C, void* stream);
// alpha/beta for T tokens: xin [T][D], wab [2*hv][D]; row < hv: gate = softplus(acc + dt[row]) * a[row]; row >= hv:
// beta = sigmoid(acc).
int gdn_ab_multi(const float* xin, const float* wab, const float* dt, const float* a, float* gate, float* beta, int T, int D,
                 int hv, void* stream);
// T-token gated-delta-net step + output rms-norm. state (S,hv,S) j fastest; q,k [T][hk][128]; v [T][hv][128]; gate,beta
// [T][hv]; nw [128]; out [T][hv][128] written for t >= t_out_begin (earlier tokens only replay). n_keep_dev == nullptr or
// *n_keep_dev <= 0: state untouched; otherwise the state afterwards equals the state after the first min(n, T) tokens.
int gdn_step_norm_multi(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                        const float* nw, float* out, int T, int hk, int hv, float scale, float eps,
                        const int32_t* n_keep_dev, int t_out_begin, void* stream);
// Packed-code embedding gather: rows of ceil(dim*bits/8) bytes; scales [rows][dim/group_elems]; value = (code - 2^(bits-1)) *
// scale. group_elems | dim. ids outside [0, rows) give a zero row.
int embedding_gather_dev(const uint8_t* table, const float* scales, const int32_t* ids, float* out, int n, int rows, int dim,
                         int bits, int group_elems, void* stream);
// dst[k*blob_bytes + b] = src[ids[k]*blob_bytes + b]; invalid ids give zero bytes. Grid-stride over n*blob_bytes.
int fetch_blobs(const uint8_t* src, const int32_t* ids, uint8_t* dst, int n, int nblobs, int blob_bytes, void* stream);
// dst[k*row16 + o] = src[ids[k]*row16 + o]; invalid ids give zero.
int gather_rows(const uint16_t* src, const int32_t* ids, uint16_t* dst, int n, int rows, int row16, void* stream);
// dst[i] = src[*idx*stride + i] for i < stride; out-of-range *idx gives zeros. At most 64 blocks, grid-stride.
int copy_indexed(const float* src, const int32_t* idx, float* dst, int stride, int nrows, void* stream);
// R[t*n*hc + i] = x[t*n + i % n]   /   R[...] += x[...]
int broadcast_streams(const float* x, float* R, int T, int n, int hc, void* stream);
int add_streams_broadcast(float* R, const float* x, int T, int n, int hc, void* stream);
// ptrs[k] = base + k*bytes, n <= 128
int rebase_ptrs(uint64_t* ptrs, uint64_t base, uint64_t bytes, int n, void* stream);
// steps[4*t..4*t+3] = (c, c+1, (c+1)/4, c+1) for c = c0 + t
int dense_steps(int32_t* steps, int c0, int T, void* stream);
// ids[j] = start + j for j < width, -1 beyond; *width_out = n_kv - start with start = n_kv > window ? n_kv - window : 0.
int window_ids(int32_t* ids, int32_t* width_out, int n_kv, int window, void* stream);
// out[i] = ids[i] in [0, nmap) ? map[ids[i]] : -1, n <= 64
int map_ids(const int32_t* ids, const int32_t* map, int32_t* out, int n, int nmap, void* stream);
// *hits = #{i < n : ids[i] == i}, n <= 1024
int ident_hits(const int32_t* ids, int n, int32_t* hits, void* stream);
// Per row: tok = lowest-index argmax, prob = 1 / sum exp(l - max). 512 threads (16 segments of 32 lanes), block-stride over the vocabulary.
int row_top_prob(const float* logits, int32_t* tok, float* prob, int rows, int vocab, void* stream);
// sel[0] = lowest index of the maximum prob; tok_out[0] = tok[sel[0]].
int mtp_select(const int32_t* tok, const float* prob, int n, int32_t* sel, int32_t* tok_out, void* stream);
// Spin until *flag >= want or `spins` polls elapse; *status = 1 reached, 0 timed out.
int wait_flag_ge(const volatile uint32_t* flag, uint32_t want, uint32_t spins, int32_t* status, void* stream);

}  // namespace vwin
}  // namespace baloon
