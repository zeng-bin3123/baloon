// ref/prefill/kernels.hpp — reference specification: host entry points of the batched prompt-path CUDA kernels.
// All pointers are device pointers; every wrapper launches on the default stream.
// Returns 0 on success, a CUDA error code otherwise, -1 on invalid arguments.
#pragma once
#include <cstdint>

namespace baloon {
namespace pk {

// gr_norm: RMS-normalise the full 4-stream hyper-connection tensor x[T][4][2560] into out[T][4][2560].
// eps guards the rsqrt against a zero-norm row (over all 4*2560 = 10240 elements of one token).
int gr_norm(const float* x, float* out, int T, float eps);

// gr_mix: out[t][d] = sum_{c=0}^{HC-1} coef[t*ld+c] * x[t][c][d].
// ld >= HC allows coef rows to be part of a wider matrix; out is [T][N], x is [T][HC][N].
int gr_mix(const float* x, const float* coef, int ld, float* out, int T);

// gr_write: out[t][c][d] = res[t][c][d] + inj[t*ld+c] * y[t][d].
// Broadcasts the injection weight inj across all HC streams, adding the scaled layer output y back to the residual.
int gr_write(const float* res, const float* y, const float* inj, int ld, float* out, int T);

// gr_broadcast: out[t][c][d] = e[t][d] for all c in [0,HC).
// Tiles a single-stream embedding e[T][N] into all HC streams of the hyper-connection tensor.
int gr_broadcast(const float* e, float* out, int T);

// gr_silu: element-wise SiLU activation: out[i] = x[i] * sigmoid(x[i]).
int gr_silu(const float* x, float* out, int64_t n);

// gdn_gates: g[i] = -exp(a_log[i%HV]) * softplus(a[i] + dt_bias[i%HV]);  beta[i] = sigmoid(b[i]).
// HV=48 value heads; a_log and dt_bias are per-head (length HV); input/output shapes [T][HV].
int gdn_gates(const float* a, const float* b, const float* a_log, const float* dt_bias, float* g, float* beta, int T);

// gdn_conv: causal width-4 depthwise conv over C=10240 channels + SiLU: out[t][c] = acc*sigmoid(acc).
// state[c][3] holds the three most-recent inputs (oldest first) and is updated in-place.
// Chunk-split invariant: sequential calls on consecutive slices equal one full call.
int gdn_conv(const float* x, const float* w, float* state, float* out, int T);

// gdn_recurrence: gated delta-net state update over HV=48 value heads, state layout [row][head][col].
// g[t][h] <= 0 is the per-step decay gate; beta[t][h] is the update gate; qkv layout [T][C], C=2*HK*S+HV*S.
// Chunk-split invariant: sequential calls produce the same output as one full call (state carries in-place).
int gdn_recurrence(const float* qkv, const float* g, const float* beta, float* state, float* out, int T);

// route: top-kRoutedK=10 routing over kExperts=512 experts per token.
// ids[T][10] = selected expert indices (descending logit order, ties broken to the lower expert id).
// wts[T][10] = softmax weights (sum to 1 per token, descending order matching ids).
int route(const float* logits, int32_t* ids, float* wts, int T);

// blob_dequant: dequantise one expert blob (kBlobBytes=1382400 bytes) to f32.
// 2-bit codes per byte (groups of 64 columns share one f16 scale); value = (code-1)*scale.
// gu_out: [1280][2560] gate/up projection; d_out: [2560][640] down projection.
int blob_dequant(const uint8_t* blob, float* gu_out, float* d_out);

// blob_dequant_f16: same layout as blob_dequant but output stored as fp16 (round-to-nearest-even of the f32 value).
int blob_dequant_f16(const uint8_t* blob, uint16_t* gu_out, uint16_t* d_out);

// swiglu_pair: out[i] = gate[i] * sigmoid(gate[i]) * up[i].  Paired (separate gate, up) form.
int swiglu_pair(const float* gate, const float* up, float* out, int64_t n);

// swiglu_interleaved: in[rows][2*width] with (gate,up) interleaved per column; out[rows][width].
// out[r][k] = in[r][2k] * sigmoid(in[r][2k]) * in[r][2k+1]. No separate gate/up buffers needed.
int swiglu_interleaved(const float* in, float* out, int rows, int width);

// gather_rows16: copy rows of src indexed by idx into dst; operates on 16-bit elements, 16-byte-aligned moves.
// dst[r][j] = src[idx[r]][j] for r in [0,rows), j in [0,width). width is counted in uint16_t elements.
int gather_rows16(const uint16_t* src, const int32_t* idx, uint16_t* dst, int rows, int width);

// moe_combine: out[t][d] = shared[t][d] + sum_{k=0}^{kRoutedK-1} wts[t][k] * expert_out[t*kRoutedK+k][d].
// expert_out shape [T*kRoutedK][N]; shared is the shared-expert output; result is the combined residual update.
int moe_combine(const float* expert_out, const float* wts, const float* shared, float* out, int T);

// rms_rows: per-row RMS norm with learned scale: out[r][i] = x[r][i] / rms(x[r]) * w[i].
// eps guards the rsqrt; w is a per-column weight vector of length width.
int rms_rows(const float* x, const float* w, float* out, int rows, int width, float eps);

// rope: apply rotary position embedding in-place on x[T][heads][head_dim].
// Rotates (p, p+32) pairs for p in [0,32) of every head; dims >= 64 are untouched.
// theta = 10000.0 by convention; pos[t] is the absolute position of token t.
int rope(float* x, const int32_t* pos, int T, int heads, int head_dim, float theta);

// split_q: decompose qfull[T][24*512] into q[T][24*256] and gate[T][24*256] per head.
// Head h: qfull[t][h*512 + 0..255] -> q,  qfull[t][h*512 + 256..511] -> gate.
int split_q(const float* qfull, float* q, float* gate, int T);

// gate_attn: out[i] = attn[i] * sigmoid(gate[i]).  Soft-gates the attention output with a learned gate vector.
int gate_attn(const float* attn, const float* gate, float* out, int T);

// kv_append: quantise K to int8 in groups of 64 channels, store starting at base_row.
// scale = fp16(amax/127) per group (4 groups per 256-dim key); codes clamped to +-127; zero group -> code 0, scale 0.
int kv_append(const float* k, int8_t* kq, uint16_t* ks, int64_t base_row, int T);

// to_bf16: convert n float32 values to bfloat16 (round-to-nearest-even) into a uint16 buffer.
// NaN payload is preserved (quiet bit set). Exact bit-for-bit match with the host f32->bf16 rounding rule.
int to_bf16(const float* x, uint16_t* out, int64_t n);

// to_f16: convert n float32 values to float16 (round-to-nearest-even) into a uint16 buffer.
// Handles subnormals, inf and NaN; exact bit-for-bit match with the host f16_from_f32 rounding rule.
int to_f16(const float* x, uint16_t* out, int64_t n);

}  // namespace pk
}  // namespace baloon
