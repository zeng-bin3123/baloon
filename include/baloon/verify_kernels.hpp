// verify_kernels.hpp - multi-token (speculative verify window) device helpers and the pure index arithmetic they obey.
// Pure C++ (no CUDA headers). Device entry points are implemented in src/verify_kernels.cu. Return codes: 0 ok,
// -1 rejected by host validation (nothing launched), >0 a cudaError_t value. Shuffles use 32-lane segments, so results do
// not depend on the hardware wave width (64 on ivcore10).
//
// Conv window: token t reads entries t..t+2 of seq = [hist0, hist1, hist2, x0, x1, ...] plus x_t with weight w[3].
// Gated delta-net state layout (S, h_v, S), element (i, h, j) at ((i*h_v + h)*S + j); q/k head of value head h is h % h_k.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace verify_kernels {

constexpr uint32_t kVerifyMaxT = 8;       // widest verify window (tokens)
constexpr int kStateS = 128;              // gated delta-net state width
constexpr int kRowGroups = 4;             // row groups of the fused step
constexpr int kRowsPerGroup = 32;
constexpr int kSelectionBound = 2051;     // top_k (2048) + r (4) - 1
constexpr int kConvTaps = 4;              // 3 history entries + the current input

// ---- pure index arithmetic (host model of the device formulas) ----------------------------------------------------
inline uint64_t st_idx(uint64_t hv, uint64_t S, uint64_t i, uint64_t h, uint64_t j) { return (i * hv + h) * S + j; }
inline uint32_t qk_head(uint32_t h, uint32_t hk) { return h % hk; }
// entry i of [hist0..2, x0, x1, ...] as an index: i < 3 -> history slot i, else input i - 3 (returned as -(i-3)-1 style tag)
inline bool seq_is_hist(uint32_t i) { return i < 3; }
inline uint32_t seq_slot(uint32_t i) { return i < 3 ? i : i - 3; }
struct DenseStep { int32_t pos, n_kv, n_bid, width; };
inline DenseStep dense_step(int32_t c) { return DenseStep{c, c + 1, (c + 1) / 4, c + 1}; }
inline int32_t win_start(int32_t n_kv, int32_t window) { return n_kv > window ? n_kv - window : 0; }
inline int32_t win_width(int32_t n_kv, int32_t window) { return n_kv - win_start(n_kv, window); }
inline bool commits(bool has_keep, int n) { return has_keep && n > 0; }
inline uint64_t gather_src(uint64_t id, uint64_t o, uint64_t row_len) { return id * row_len + o; }
inline uint64_t broadcast_src(uint64_t t, uint64_t i, uint64_t n) { return t * n + i % n; }
inline uint64_t codes_per_byte(int bits) { return 8u / (unsigned)bits; }
inline bool bits_ok(int bits) { return bits == 2 || bits == 4 || bits == 8; }
// Code `k` (0-based) of a packed row: byte k / cpb, shift (k % cpb) * bits, low bits first.
inline unsigned unpack_code(const uint8_t* row, uint64_t k, int bits) {
    const uint64_t cpb = codes_per_byte(bits);
    const unsigned mask = (1u << bits) - 1u;
    return (row[k / cpb] >> ((k % cpb) * (uint64_t)bits)) & mask;
}

// ---- device entry points (host launchers) -------------------------------------------------------------------------
// conv + l2 over a T-token window. x [T][C], hist [3][C] (read only), w [C][4], out [T][C]; the first 2*hk*128 channels
// are q then k heads of 128 dims, each l2-normalised (1/max(norm, eps)); the remaining channels are plain conv outputs.
int gdn_conv_l2_multi(const float* x, const float* hist, const float* w, float* out, int T, int C, int hk, float eps,
                      void* stream);
// New history = last three entries of [hist | x_0..x_{n-1}] in place (hist [3][C]); n <= 0 is a no-op.
int gdn_conv_commit(const float* x, float* hist, int n, int C, void* stream);
// alpha/beta for T tokens. xin [T][D], wab [2*hv][D]; row < hv: gate = softplus(acc + dt[row]) * a[row] -> gate[t][row];
// row >= hv: beta[t][row-hv] = sigmoid(acc). One 32-lane segment per row.
int gdn_ab_multi(const float* xin, const float* wab, const float* dt, const float* a, float* gate, float* beta, int T,
                 int D, int hv, void* stream);
// T-token gated-delta-net step + output rms-norm. state (S,hv,S) j fastest; q,k [T][hk][128]; v [T][hv][128];
// gate,beta [T][hv]; nw [128]; out [T][hv][128] written for tokens t >= t_out_begin (earlier tokens are replay only).
// n_keep <= 0 means "no keep count": the state is never modified. n_keep > 0: the state afterwards equals the state after
// the first min(n_keep, T) tokens, while all T tokens still produce outputs.
int gdn_step_norm_multi(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                        const float* nw, float* out, int T, int hk, int hv, float scale, float eps, int n_keep,
                        int t_out_begin, void* stream);
// Packed-code embedding gather: table rows of ceil(dim*bits/8) bytes, value = (code - 2^(bits-1)) * scale[row].
// ids >= rows or < 0 give a zero row.
int embedding_gather_dev(const uint8_t* table, const float* scales, const int32_t* ids, float* out, int n, int rows,
                         int dim, int bits, void* stream);
// Byte blob gather: dst[k*blob_bytes ..] = src[ids[k]*blob_bytes ..]; invalid ids give zero bytes.
int fetch_blobs(const uint8_t* src, const int32_t* ids, uint8_t* dst, int n, int nblobs, int blob_bytes, void* stream);
// 16-bit row gather: dst[k*row16 + o] = src[ids[k]*row16 + o]; invalid ids give zero.
int gather_rows(const uint16_t* src, const int32_t* ids, uint16_t* dst, int n, int rows, int row16, void* stream);
// dst[i] = src[idx[0]*stride + i], i < stride; idx is read from device memory; an out-of-range idx gives zeros.
int copy_indexed(const float* src, const int32_t* idx, float* dst, int stride, int nrows, void* stream);
// R[t*n*hc + i] = x[t*n + i % n]
int broadcast_streams(const float* x, float* R, int T, int n, int hc, void* stream);
// R[t*n*hc + i] += x[t*n + i % n]
int add_streams_broadcast(float* R, const float* x, int T, int n, int hc, void* stream);
// ptrs[k] = base + k*bytes
int rebase_ptrs(uint64_t* ptrs, uint64_t base, uint64_t bytes, int n, void* stream);
// steps[4*t..4*t+3] = (c, c+1, (c+1)/4, c+1) for c = c0 + t
int dense_steps(int32_t* steps, int c0, int T, void* stream);
// ids[j] = start + j for j < width, -1 beyond (window slots); *width_out = n_kv - start.
int window_ids(int32_t* ids, int32_t* width_out, int n_kv, int window, void* stream);
// out[i] = ids[i] if 0 <= ids[i] < nmap ? map[ids[i]] : -1
int map_ids(const int32_t* ids, const int32_t* map, int32_t* out, int n, int nmap, void* stream);
// *hits = number of i < n with ids[i] == i
int ident_hits(const int32_t* ids, int n, int32_t* hits, void* stream);
// Per-row top probability under softmax: tok[r] = lowest-index argmax, prob[r] = 1 / sum exp(l - max).
int row_top_prob(const float* logits, int32_t* tok, float* prob, int rows, int vocab, void* stream);
// Choose the draft row: sel[0] = lowest index of the maximum prob; tok_out[0] = tok[sel[0]].
int mtp_select(const int32_t* tok, const float* prob, int n, int32_t* sel, int32_t* tok_out, void* stream);
// Wait until *flag >= want or `spins` polls elapse; *status = 1 reached, 0 timed out.
int wait_flag_ge(const volatile uint32_t* flag, uint32_t want, uint32_t spins, int32_t* status, void* stream);

}  // namespace verify_kernels
}  // namespace baloon
