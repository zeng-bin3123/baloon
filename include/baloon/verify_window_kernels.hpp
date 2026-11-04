// verify_window_kernels.hpp - host API of the multi-token (speculative verify window) device helpers: conv + l2 over a token
// window and its history commit, alpha/beta for T tokens, the GDN step + out-norm over T tokens with replay and keep-count,
// device-side gathers, stream broadcasts, step-state builders and the MTP selector. Pure C++ (no CUDA headers).
// Return codes: 0 ok, -1 rejected by host validation, >0 a cudaError_t value.
#pragma once
#include <cstddef>
#include <cstdint>
#include "baloon/qsa_step_state.hpp"

namespace baloon {
namespace verify_dev {

using qsa_step::Step;
constexpr int kOk = 0;
constexpr int kInvalid = -1;
constexpr int kVerifyMaxT = 16;  // largest token window

// R[t][c][i] = x[t][i] for every stream c < hc.
int broadcast_streams(const float* x, float* R, int T, int n, int hc, void* stream);
// R[t][c][i] += delta[t][i].
int add_streams_broadcast(float* R, const float* delta, int T, int n, int hc, void* stream);
// dst[i][:] = src[idx[i]][:] for i < m (device-side row indices).
int copy_indexed(const float* src, const int32_t* idx, float* dst, int m, int row, void* stream);
// steps[t] = (c0 + t, c0 + t + 1, (c0 + t + 1) / 4, c0 + t + 1).
int dense_steps(int c0, int T, Step* steps, void* stream);
// Row i of the packed embedding table (bits in {2,4,8}, 8/bits codes per byte, little-endian within the byte):
// out[i][c] = (code - 2^(bits-1)) * scale[ids[i]].
int embedding_gather_dev(const uint8_t* table, const float* scale, const int32_t* ids, int n, int cols, int bits, float* out,
                         void* stream);
// dst[i][:] = src[ids[i]][:] byte rows of `blob_bytes` bytes.
int fetch_blobs(const uint8_t* src, const int32_t* ids, int n, size_t blob_bytes, uint8_t* dst, void* stream);
// 16-bit rows: dst[i][:] = src[ids[i]][:] (ids outside [0, rows) give zero rows).
int gather_rows(const uint16_t* src, const int32_t* ids, int n, int row16, int rows, uint16_t* dst, void* stream);
// alpha/beta for T tokens. x [T][n_embd] f32, W [2 h_v][n_embd] bf16 bits; row < h_v alpha: gate = softplus(acc + dt) * a;
// row >= h_v beta = sigmoid(acc). gate [T][h_v], beta [T][h_v].
int gdn_ab_multi(const float* x, const uint16_t* W, const float* dt, const float* a, float* gate, float* beta, int T, int n_embd,
                 int h_v, void* stream);
// Channels are laid out [q: h_k * 128 | k: h_k * 128 | v: h_v * 128]. hist [C][3], x [T][C], w [C][4], out [T][C].
// out = silu(sum_{j<3} win_j w_j + x_t w_3), win_j = [hist0, hist1, hist2, x0, x1, ...][t + j]; q and k rows are then l2-normalised.
int gdn_conv_l2_multi(const float* hist, const float* x, const float* w, float* out, int T, int h_k, int h_v, float eps,
                      void* stream);
// hist <- last three entries of [hist | x_0 .. x_{n-1}]; no-op for n <= 0.
int gdn_conv_commit(float* hist, const float* x, int n, int C, void* stream);
// state [128][h_v][128] (in place), q/k [T][h_k][128], v [T][h_v][128], gate/beta [T][h_v], out [T][h_v][128].
// Tokens t < t_out_begin only advance the state (replay); the state is written back after token keep-1 when keep != null and
// *keep > 0 (never without a keep-count). out = rms-normalised step output.
int gdn_step_norm_multi(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                        float* out, int T, int h_k, int h_v, float scale, float eps, const int32_t* keep, int t_out_begin,
                        void* stream);
// *hits = length of the longest prefix with a[i] == b[i] (accepted draft prefix).
int ident_hits(const int32_t* a, const int32_t* b, int n, int32_t* hits, void* stream);
// out[i] = table[in[i]] (-1 when in[i] is outside [0, len)).
int map_ids(const int32_t* in, const int32_t* table, int len, int32_t* out, int n, void* stream);
// dst[:] = rows[*row_idx][:] (zeros when *row_idx is outside [0, n_rows)).
int mtp_select(const float* rows, const int32_t* row_idx, int n_rows, int stride, float* dst, void* stream);
// ptrs[k] = base + k * bytes.
int rebase_ptrs(const void* base, size_t bytes, void** ptrs, int count, void* stream);
// Per row: id = argmax (lowest index on ties), prob = softmax probability of that id.
int row_top_prob(const float* logits, int rows, int V, int32_t* id, float* prob, void* stream);
// Spin until *flag >= target (at most max_spin polls); *status = 0 reached, 1 gave up.
int wait_flag_ge(const volatile int32_t* flag, int32_t target, int64_t max_spin, int32_t* status, void* stream);
// Window of the last `window` cells: start = max(0, n_kv - window); ids[j] = start + j for j < n_kv - start, -1 up to cap;
// step->width is rewritten to n_kv - start.
int window_ids(Step* step, int window, int32_t* ids, int cap, void* stream);

}  // namespace verify_dev
}  // namespace baloon
