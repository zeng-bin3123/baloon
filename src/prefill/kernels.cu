// kernels.cu - CUDA kernels of the batched prompt path for Iluvatar BI-V100 (ivcore10, wave64): hyper-connection
// norm/mix/write/broadcast, GDN gates/conv/recurrence, MoE routing/dequant/SwiGLU/gather/combine and QSA helpers.
// Every kernel addresses flat buffers by decomposing a linear thread index into (token, stream, dim) coordinates, guards
// i < n, and uses block-level reductions that do not depend on the hardware wave width (32 or 64).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// Geometry owned by this translation unit; the API header skips these four when BALOON_PK_GEOMETRY_IN_TU is set.
namespace baloon {
namespace pk {
constexpr int N = 2560;  // model width
constexpr int S = 128;   // GDN head size
constexpr int RG = 4;    // residual streams
inline int64_t blocks_for(int64_t n, int64_t t) { return (n + t - 1) / t; }
}  // namespace pk
}  // namespace baloon
#define BALOON_PK_GEOMETRY_IN_TU 1
#define BALOON_PK_DEFINE_KERNELS 1
#include "baloon/prefill_kernels_api.hpp"
#include "kernels_api.cuh"

// Device code (helpers and the kernel bodies) lives in baloon/prefill_kernels_api.hpp behind BALOON_PK_DEFINE_KERNELS.
// Helpers injected: block_sum (block halving-tree reduction), warp_sum, warp_max (lane reductions),
// sigm (sigmoid), bf (float->bf16 bits), hf (fp16 bits->float); kernels use __launch_bounds__.
namespace baloon {
namespace pk {


// ---- host wrappers -----------------------------------------------------------------------------------------------
int gr_norm(const float* x, float* out, int T, float eps) {
    if (!x || !out || T <= 0) return -1;
    gr_norm_kernel<<<(unsigned)T, 256>>>(x, out, eps);
    return check();
}
int gr_mix(const float* x, const float* coef, int ld, float* out, int T) {
    if (!x || !coef || !out || T <= 0 || ld < HC) return -1;
    const int64_t n = (int64_t)T * N;
    gr_mix_kernel<<<grid(n), TH>>>(x, coef, ld, out, n);
    return check();
}
int gr_write(const float* res, const float* y, const float* inj, int ld, float* out, int T) {
    if (!res || !y || !inj || !out || T <= 0 || ld < HC) return -1;
    const int64_t n = (int64_t)T * D;
    gr_write_kernel<<<grid(n), TH>>>(res, y, inj, ld, out, n);
    return check();
}
int gr_broadcast(const float* e, float* out, int T) {
    if (!e || !out || T <= 0) return -1;
    const int64_t n = (int64_t)T * D;
    gr_broadcast_kernel<<<grid(n), TH>>>(e, out, n);
    return check();
}
int gr_silu(const float* x, float* out, int64_t n) {
    if (!x || !out || n <= 0) return -1;
    gr_silu_kernel<<<grid(n), TH>>>(x, out, n);
    return check();
}
int gdn_gates(const float* a, const float* b, const float* a_log, const float* dt_bias, float* g, float* beta, int T) {
    if (!a || !b || !a_log || !dt_bias || !g || !beta || T <= 0) return -1;
    const int64_t n = (int64_t)T * HV;
    gdn_gates_kernel<<<grid(n), TH>>>(a, b, a_log, dt_bias, g, beta, n);
    return check();
}
int gdn_conv(const float* x, const float* w, float* state, float* out, int T) {
    if (!x || !w || !state || !out || T <= 0) return -1;
    gdn_conv_kernel<<<C / 128, 128>>>(x, w, state, out, T);
    return check();
}
int gdn_l2(float* qkv, int T, float eps) {
    if (!qkv || T <= 0) return -1;
    gdn_l2_kernel<<<dim3(2 * HK, (unsigned)T), 128>>>(qkv, eps);
    return check();
}
int gdn_recurrence(const float* qkv, const float* g, const float* beta, float* state, float* out, int T) {
    if (!qkv || !g || !beta || !state || !out || T <= 0) return -1;
    static_assert(RG * (S / RG) == S, "row groups cover all rows");
    gdn_rec_kernel<<<HV, S>>>(qkv, g, beta, state, out, T);
    return check();
}
int route(const float* logits, int32_t* ids, float* wts, int T) {
    if (!logits || !ids || !wts || T <= 0) return -1;
    route_kernel<<<(unsigned)T, 64>>>(logits, ids, wts);
    return check();
}
int blob_dequant(const uint8_t* blob, float* gu_out, float* d_out) {
    if (!blob || !gu_out || !d_out) return -1;
    blob_dequant_kernel<false><<<grid(1280 * 640 + 2560 * 160), TH>>>(blob, gu_out, d_out);
    return check();
}
int blob_dequant_f16(const uint8_t* blob, uint16_t* gu_out, uint16_t* d_out) {
    if (!blob || !gu_out || !d_out) return -1;
    blob_dequant_kernel<true><<<grid(1280 * 640 + 2560 * 160), TH>>>(blob, gu_out, d_out);
    return check();
}
int swiglu_pair(const float* gate, const float* up, float* out, int64_t n) {
    if (!gate || !up || !out || n <= 0) return -1;
    swiglu_pair_kernel<<<grid(n), TH>>>(gate, up, out, n);
    return check();
}
int swiglu_interleaved(const float* in, float* out, int rows, int width) {
    if (!in || !out || rows <= 0 || width <= 0) return -1;
    const int64_t n = (int64_t)rows * width;
    swiglu_il_kernel<<<grid(n), TH>>>(in, out, n, width);
    return check();
}
int gather_rows16(const uint16_t* src, const int32_t* idx, uint16_t* dst, int rows, int width) {
    if (!src || !idx || !dst || rows <= 0 || width <= 0 || width % 8 != 0) return -1;
    const int per = width / 8;
    const int64_t n = (int64_t)rows * per;
    gather_rows16_kernel<<<grid(n), TH>>>((const uint4*)src, idx, (uint4*)dst, n, per);
    return check();
}
int moe_combine(const float* expert_out, const float* wts, const float* shared, float* out, int T) {
    if (!expert_out || !wts || !shared || !out || T <= 0) return -1;
    const int64_t n = (int64_t)T * N;
    moe_combine_kernel<<<grid(n), TH>>>(expert_out, wts, shared, out, n);
    return check();
}
int rms_rows(const float* x, const float* w, float* out, int rows, int width, float eps) {
    if (!x || !w || !out || rows <= 0 || width <= 0) return -1;
    rms_rows_kernel<<<(unsigned)rows, 256>>>(x, w, out, width, eps);
    return check();
}
int rope(float* x, const int32_t* pos, int T, int heads, int head_dim, float theta) {
    if (!x || !pos || T <= 0 || heads <= 0 || head_dim < 64) return -1;
    const int64_t n = (int64_t)T * heads * 32;
    rope_kernel<<<grid(n), TH>>>(x, pos, n, heads, head_dim, theta);
    return check();
}
int split_q(const float* qfull, float* q, float* gate, int T) {
    if (!qfull || !q || !gate || T <= 0) return -1;
    const int64_t n = (int64_t)T * 6144;
    split_q_kernel<<<grid(n), TH>>>(qfull, q, gate, n);
    return check();
}
int gate_attn(const float* attn, const float* gate, float* out, int T) {
    if (!attn || !gate || !out || T <= 0) return -1;
    const int64_t n = (int64_t)T * 6144;
    gate_attn_kernel<<<grid(n), TH>>>(attn, gate, out, n);
    return check();
}
int kv_append(const float* k, int8_t* kq, uint16_t* ks, int64_t base_row, int T) {
    if (!k || !kq || !ks || base_row < 0 || T <= 0) return -1;
    kv_append_kernel<<<dim3(4, (unsigned)T), 64>>>(k, kq, ks, base_row);
    return check();
}
int to_bf16(const float* x, uint16_t* out, int64_t n) {
    if (!x || !out || n <= 0) return -1;
    to_bf16_kernel<<<grid(n), TH>>>(x, out, n);
    return check();
}
int to_f16(const float* x, uint16_t* out, int64_t n) {
    if (!x || !out || n <= 0) return -1;
    to_f16_kernel<<<grid(n), TH>>>(x, out, n);
    return check();
}

int route_ids_bad(const int32_t* ids, int64_t n, int* bad_count) {
    if (!ids || !bad_count || n <= 0) return -1;
    route_ids_bad_kernel<<<grid(n), TH>>>(ids, n, bad_count);
    return check();
}

}  // namespace pk
}  // namespace baloon

// R is [T][HC=4][N=2560]; flat index i decomposes as t=i/10240, c=(i%10240)/2560, d=i%2560.
// grIndex(t,c,d) = t*10240 + c*2560 + d; grIndex_injective proves distinct coords → distinct cells.

// Blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16,
// down scales [2560][10] f16 — total 1,382,400 bytes. Each code byte holds four 2-bit codes.
// Weight value = ((c >> 2k) & 3) - 1 ∈ {-1,0,1,2} times block scale of its 64-wide group.

// KV append: row = (page*2 + kvh)*page_size + pos%page_size; int8 codes clamped to [-127,127].
// kvRow_injective: distinct (page,kvh,pos%ps) → distinct rows. clampQ_range: -128 never written.

// GDN conv: each channel keeps last three inputs (v0,v1,v2); after a chunk of any length,
// history is exactly its last three values. (convStep_last_three: xs++[a,b,c] → (a,b,c))

// Router: ten rounds of warp arg-max; pairwise step prefers larger probability; ties go to
// smaller expert id. pick is commutative (pick_comm) → order-independent, all lanes agree.

// Interleaved SwiGLU: gate at column 2k, up at 2k+1 (gateIdx/upIdx). QSA Qf: 256 query values
// then 256 gate values per head. RoPE rotates pair (p, p+32) of a 64-wide rotary span.

// blob_dequant_kernel: weight code = ((byte >> 2k) & 3); dequantized weight = (code-1) * scale.
// code2_lt: code ∈ {0,1,2,3}; weightMult_range: code-1 ∈ {-1,0,1,2}. code 1 → zero.

// kv_append_kernel: int8 KV cache codes clamped to [-127,127] (symmetric); -128 is never
// written. One fp16 scale per 64 values: a 256-wide head row has 4 scale groups.
