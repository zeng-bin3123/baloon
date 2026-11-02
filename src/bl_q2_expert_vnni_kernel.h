// bl_q2_expert_vnni_kernel.h - Q2_0 expert kernel (gate/up -> SwiGLU -> Q8_1 requantise -> down).
// Portable scalar implementation of the arithmetic the AVX-512 VNNI kernel performs: unsigned 2-bit codes
// against int8 activations per 32-value chunk, minus the offset correction (c-1)*x = c*x - x.
#pragma once
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include "bl_f16_bits.h"
#include "bl_nat.h"

namespace baloon::q2_expert {

constexpr nat QK = 64;   // weights per Q2_0 block
constexpr nat QKA = 32;  // values per activation chunk
constexpr nat BLOCK_STRIDE = 18;  // fp16 scale + 16 code bytes
static_assert(QK == 2 * QKA, "a weight block spans two activation chunks");

// 2-bit code j of a 16-bit lane / of a byte array.
constexpr nat code2(nat w, nat j) { return (w >> (2 * j)) & 3; }
inline nat codeAt(const std::uint8_t* bytes, nat j) { return (bytes[j / 4] >> (2 * (j % 4))) & 3u; }

// Unpack the 64 codes of one block (16 bytes) into out[64] (values 0..3).
inline void unpack_q2_0(const std::uint8_t* bytes, std::uint8_t* out) {
  for (nat j = 0; j < QK; ++j) out[j] = std::uint8_t(codeAt(bytes, j));
}

// Mask over the 16 lanes for a batch of nb blocks (two lanes per block).
constexpr nat tailMask(nat nb) { return nb >= 8 ? 0xFFFFu : (nat(1) << (2 * nb)) - 1; }
constexpr nat batchSize(nat nblocks, nat b0) { return nmin(8, sat_sub(nblocks, b0)); }

constexpr std::int64_t clamp127(std::int64_t v) { return v < -127 ? -127 : (v > 127 ? 127 : v); }
// Round half away from zero of a value given scaled by two (u = 2x): (u+1)/2 for u >= 0, mirrored below.
constexpr std::int64_t roundAway(std::int64_t u) { return u >= 0 ? (u + 1) / 2 : -((-u + 1) / 2); }

// Round half away from zero of a float.
inline std::int64_t round_half_away(float v) { return std::int64_t(v >= 0 ? std::floor(v + 0.5f) : -std::floor(-v + 0.5f)); }

// Activation quantiser (Q8_1-like): per 32-value chunk, d = amax/127, q = clamp(round_away(x/d)), stored int8.
struct QAct {
  std::vector<std::int8_t> q;  // n values
  std::vector<float> d;        // per chunk scale
  std::vector<std::int32_t> s; // per chunk sum of q
};
inline QAct act_quant_q8_1(const float* x, nat n) {
  QAct a;
  nat nc = n / QKA;
  a.q.assign(n, 0); a.d.assign(nc, 0.f); a.s.assign(nc, 0);
  for (nat c = 0; c < nc; ++c) {
    float amax = 0.f;
    for (nat j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(x[c * QKA + j]));
    float d = amax / 127.f;
    float id = d != 0.f ? 1.f / d : 0.f;
    std::int32_t sum = 0;
    for (nat j = 0; j < QKA; ++j) {
      std::int64_t v = clamp127(round_half_away(x[c * QKA + j] * id));
      a.q[c * QKA + j] = std::int8_t(v);
      sum += std::int32_t(v);
    }
    a.d[c] = d; a.s[c] = sum;
  }
  return a;
}

// Integer parts of one block against its two chunks: sum(c*xhat) per chunk.
inline void block_code_dots(const std::uint8_t* blk_codes16, const std::int8_t* xq, std::int32_t out[2]) {
  out[0] = out[1] = 0;
  for (nat j = 0; j < QK; ++j) out[j / QKA] += std::int32_t(codeAt(blk_codes16, j)) * std::int32_t(xq[j]);
}

// Row dot of one Q2_0 row (nblocks blocks of stride 18) with a quantised activation vector.
// result = sum_blk d_w * sum_chunk d_x[chunk] * (sum(c*xhat) - sum(xhat))   [offset identity]
inline float row_dot_z(const std::uint8_t* row, nat nblocks, const QAct& a) {
  float acc = 0.f;
  for (nat b = 0; b < nblocks; ++b) {
    const std::uint8_t* blk = row + b * BLOCK_STRIDE;
    std::uint16_t h = std::uint16_t(blk[0] | (blk[1] << 8));
    std::uint32_t fb = f16_bits::f32BitsFromF16(h);
    float dw; std::memcpy(&dw, &fb, 4);
    std::int32_t cd[2];
    block_code_dots(blk + 2, a.q.data() + b * QK, cd);
    for (nat c = 0; c < 2; ++c) {
      nat ci = b * 2 + c;
      acc += dw * a.d[ci] * float(cd[c] - a.s[ci]);
    }
  }
  return acc;
}

// Token-tile dispatch.
constexpr nat ntSel(nat n) { return n >= 8 ? 8 : n; }
constexpr bool useMulti(bool oracle, nat n, nat maxt) { return !oracle && 1 <= n && n <= maxt; }

// Gate row is 2r, up row is 2r+1.
constexpr nat gateRow(nat r) { return 2 * r; }
constexpr nat upRow(nat r) { return 2 * r + 1; }

inline float silu(float x) { return x / (1.f + std::exp(-x)); }

// Full expert for one token: gu = interleaved gate/up Q2_0 rows (2*ff rows, each hq blocks), down = h rows of
// ff/64 blocks. x has hq*64 values. Output y (nh values).
inline void expert_forward(const std::uint8_t* gu, const std::uint8_t* down, nat hdim, nat ff, const float* x,
                           float* y) {
  nat nbH = hdim / QK, nbF = ff / QK;
  QAct ax = act_quant_q8_1(x, hdim);
  std::vector<float> h(ff);
  for (nat r = 0; r < ff; ++r) {
    float g = row_dot_z(gu + gateRow(r) * nbH * BLOCK_STRIDE, nbH, ax);
    float u = row_dot_z(gu + upRow(r) * nbH * BLOCK_STRIDE, nbH, ax);
    h[r] = silu(g) * u;
  }
  QAct ah = act_quant_q8_1(h.data(), ff);
  for (nat o = 0; o < hdim; ++o) y[o] = row_dot_z(down + o * nbF * BLOCK_STRIDE, nbF, ah);
}

}  // namespace baloon::q2_expert
