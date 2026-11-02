// bl_gated_residual_read_write.h - workspace layout, reduction mapping and index contracts of the generic gated
// residual / hyper-connection gr_read and gr_write kernels.
#pragma once
#include <vector>
#include "bl_nat.h"

namespace baloon::gated_residual {

constexpr nat alignUp16(nat n) { return (n + 15) / 16 * 16; }

// Five regions xn, xq, lq, gated, lo laid out back to back, each rounded up to 16 bytes.
struct Workspace {
  nat off[5]; nat size[5]; nat total;
};
constexpr Workspace workspace(nat n_embd, nat hc, nat hc_lr) {
  nat hc_dim = hc * n_embd;
  Workspace w{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, 0};
  w.size[0] = alignUp16(hc_dim * 4);  // xn  (float)
  w.size[1] = alignUp16(hc_dim * 2);  // xq  (bf16 quantised streams)
  w.size[2] = alignUp16(hc_lr * 2);   // lq
  w.size[3] = alignUp16(hc_dim * 4);  // gated
  w.size[4] = alignUp16(hc_lr * 4);   // lo
  nat o = 0;
  for (int i = 0; i < 5; ++i) { w.off[i] = o; o += w.size[i]; }
  w.total = o;
  return w;
}
static_assert(workspace(2560, 4, 320).total == 104320, "real geometry total");

// Down kernel: element i of a row decomposes into (iteration k, warp, lane): i = k*256 + warp*32 + lane.
constexpr nat downK(nat i) { return i / 256; }
constexpr nat downWarp(nat i) { return (i % 256) / 32; }
constexpr nat downLane(nat i) { return i % 32; }
constexpr nat downElement(nat k, nat w, nat l) { return w * 32 + l + 256 * k; }

constexpr nat gateGridBlocks(nat hc_dim) { return (hc_dim + 7) / 8; }
constexpr nat meanGridBlocks(nat n) { return (n + 255) / 256; }
constexpr nat meanIndex(nat c, nat d, nat n_embd) { return c * n_embd + d; }
constexpr nat writeStream(nat i, nat n_embd) { return ndiv(i, n_embd); }
constexpr nat writeCol(nat i, nat n_embd) { return nmod(i, n_embd); }
constexpr bool injectBlockOk(nat hc) { return 32 * hc <= 1024; }
constexpr bool nativeEven(nat n_embd, nat hc) { return n_embd % 2 == 0 && (hc * n_embd) % 2 == 0; }

// Reference write kernel: out[c*n_embd+d] = R[...] + block_out[d] * w[c].
inline void gr_write_ref(const float* R, const float* block_out, const float* w, nat n_embd, nat hc, float* out) {
  for (nat i = 0; i < n_embd * hc; ++i) out[i] = R[i] + block_out[writeCol(i, n_embd)] * w[writeStream(i, n_embd)];
}

}  // namespace baloon::gated_residual
