// bl_q2_avx2_fallback.h - Q2_0 rows + Q8_1 activation quantiser for CPUs without AVX-512.
// Same arithmetic as the VNNI kernel, organised the AVX2 way: shift-and-mask planes + interleave unpack,
// groups of 4 tokens.
#pragma once
#include <cstring>
#include <vector>
#include "bl_f16_bits.h"
#include "bl_nat.h"
#include "bl_q2_expert_vnni_kernel.h"

namespace baloon::q2_avx2 {

constexpr nat BLOCK_STRIDE = 18;
using q2_expert::QAct;

// Four shift-and-mask planes: plane k of byte i = (byte >> 2k) & 3.
// Output position p takes plane p%4 of byte p/4 (positions 0..31 from bytes 0..7, 32..63 from bytes 8..15).
inline void unpack64(const std::uint8_t* bytes16, std::uint8_t out[64]) {
  std::uint8_t plane[4][16];
  for (int k = 0; k < 4; ++k)
    for (int i = 0; i < 16; ++i) plane[k][i] = std::uint8_t((bytes16[i] >> (2 * k)) & 3u);
  // byte interleave of planes (0,1) and (2,3), then word interleave of the two results.
  for (int i = 0; i < 16; ++i) {
    out[4 * i + 0] = plane[0][i];
    out[4 * i + 1] = plane[1][i];
    out[4 * i + 2] = plane[2][i];
    out[4 * i + 3] = plane[3][i];
  }
}

constexpr nat tileSize(nat nt, nat t0) { return nmin(4, sat_sub(nt, t0)); }

inline QAct act_quant_q8_1_avx2(const float* x, nat n) { return q2_expert::act_quant_q8_1(x, n); }

// Rows x tokens: out[t * nrows + r] for tokens in groups of 4.
inline void q2_0_gguf_rows_multi_avx2(const std::uint8_t* rows, nat nrows, nat nblocks, const std::vector<QAct>& acts,
                                      float* out) {
  nat nt = acts.size();
  for (nat t0 = 0; t0 < nt; t0 += tileSize(nt, t0)) {
    nat tn = tileSize(nt, t0);
    for (nat r = 0; r < nrows; ++r) {
      const std::uint8_t* row = rows + r * nblocks * BLOCK_STRIDE;
      float acc[4] = {0, 0, 0, 0};
      for (nat b = 0; b < nblocks; ++b) {
        const std::uint8_t* blk = row + b * BLOCK_STRIDE;
        std::uint16_t h = std::uint16_t(blk[0] | (blk[1] << 8));
        std::uint32_t fb = f16_bits::f32BitsFromF16(h);
        float dw; std::memcpy(&dw, &fb, 4);
        std::uint8_t codes[64];
        unpack64(blk + 2, codes);  // decoded once, applied to every token of the tile
        for (nat t = 0; t < tn; ++t) {
          const QAct& a = acts[t0 + t];
          for (nat c = 0; c < 2; ++c) {
            std::int32_t dot = 0;
            for (nat j = 0; j < 32; ++j) dot += std::int32_t(codes[c * 32 + j]) * std::int32_t(a.q[b * 64 + c * 32 + j]);
            nat ci = b * 2 + c;
            acc[t] += dw * a.d[ci] * float(dot - a.s[ci]);
          }
        }
      }
      for (nat t = 0; t < tn; ++t) out[(t0 + t) * nrows + r] = acc[t];
    }
  }
}

}  // namespace baloon::q2_avx2
