// decode_attn_kernels.hpp - host API of the chunked split-K QSA decode attention (qsa_decode_attn_*) and the short-context
// flash decode kernel (native_flash_attn_step), plus the pure geometry they obey. Implementations: src/qsa_decode_attn.cu,
// src/native_flash_attn.cu. Return codes: 0 ok, -1 rejected by host validation, >0 a cudaError_t value.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace attn_kernels {

constexpr int kChunk = 64, kG = 12, kHD = 256;
constexpr int kFlashHeads = 24, kFlashKvHeads = 2, kFlashMaxCtx = 256, kFlashThreads = 128;

inline int n_chunks(int cap) { return (cap + kChunk - 1) / kChunk; }
inline uint64_t qsa_scratch_floats(int cap, int n_head) { return (uint64_t)n_chunks(cap) * n_head * (kHD + 2) + 64; }

// Selected-cell attention. K/V pools hold 256-wide rows (fp16, or int8 + float scales[row*4 + d/64] when q8).
// row(cell c, kv head h) = (page * n_kv + h) * page_size + slot. q [nq][n_kv*12][256] (scaled inside), ids [nq][cap],
// n_ids [nq], scratch >= nq * qsa_scratch_floats floats, out [nq][n_kv*12][256]. ids >= n_cells or < 0 are masked;
// heads with nothing attended output zeros.
struct QsaArgs {
    const float* q; const void* k; const void* v; const float* k_scale; const float* v_scale;
    const int32_t* ids; const int32_t* n_ids; float* scratch; float* out;
    int nq, n_kv, cap, page_size, n_cells; bool q8; float scale;
};
int qsa_decode_attn_batch(const QsaArgs& a, void* stream);
int qsa_decode_attn_step(const QsaArgs& a, void* stream);   // nq must be 1

// One decode step: q [24][256] floats (scaled inside by 1/16), K and V fp16 buffers [cell][2][256], out [24][256].
// Valid iff 1 <= width <= max_ctx, n_kv == width, pos == width - 1, n_bid == width / 4; otherwise the output is NaN and
// *status = 1 (status = 0 on success).
struct FlashArgs {
    const float* q; const void* k; const void* v; float* out; int32_t* status;
    int width, n_kv, pos, n_bid, max_ctx;
};
int native_flash_attn_step(const FlashArgs& a, void* stream);
uint64_t qsa_decode_attn_scratch_floats(int cap, int n_head);

}  // namespace attn_kernels
}  // namespace baloon
