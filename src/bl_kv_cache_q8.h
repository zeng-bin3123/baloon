// bl_kv_cache_q8.h - addressing and quantisation contracts of the paged int8 KV cache.
// Rows are (page, kv head, slot); values are int8 per head_dim, scales one fp16 per 64-value group.
#pragma once
#include <cmath>
#include <cstdint>
#include "bl_f16_bits.h"
#include "bl_nat.h"

namespace baloon::kv_cache_q8 {

constexpr nat GROUP = 64;  // quantisation group width

constexpr std::int64_t clamp127(std::int64_t q) { return q < -127 ? -127 : (q > 127 ? 127 : q); }

// Cache row of (page, head h, slot s).
constexpr nat cacheRow(nat page, nat h, nat s, nat kv, nat ps) { return (page * kv + h) * ps + s; }
constexpr nat pageOf(nat pos, nat ps) { return ndiv(pos, ps); }
constexpr nat slotOf(nat pos, nat ps) { return nmod(pos, ps); }
constexpr nat valueOffset(nat row, nat g, nat t, nat hd) { return row * hd + g * GROUP + t; }
constexpr nat scaleIndex(nat row, nat g, nat groups) { return row * groups + g; }
constexpr nat groupsOf(nat hd) { return ndiv(hd, GROUP); }

// Gather: thread i -> (id, head, q4), per = head_dim / 4; destination index equals i (dense scratch).
struct GatherIdx { nat id, head, q4; };
constexpr GatherIdx gatherDecompose(nat i, nat kv, nat per) {
  nat span = kv * per;
  nat rem = nmod(i, span);
  return {ndiv(i, span), ndiv(rem, per), nmod(rem, per)};
}
constexpr nat gatherDst(nat id, nat h, nat q4, nat kv, nat per) { return (id * kv + h) * per + q4; }
constexpr nat quadGroup(nat q4) { return (q4 * 4) / GROUP; }

// validate: head_dim multiple of 64, positive heads and page size.
constexpr bool validShapes(nat hd, nat nkv, nat ps) { return hd % GROUP == 0 && nkv > 0 && ps > 0; }
constexpr nat gridBlocks(nat total) { return ndiv(total + 255, 256); }
constexpr nat appendWarps(nat t) { return t / 32; }  // 64 threads = 2 warps

// Reference quantiser of one 64-value group: scale = amax/127 rounded through fp16 (the STORED scale);
// codes are round-half-away(x / stored), clamped to [-127,127]; zero scale gives code 0.
inline float halfToFloat(std::uint16_t h) {
  std::uint32_t b = f16_bits::f32BitsFromF16(h);
  float f;
  __builtin_memcpy(&f, &b, 4);
  return f;
}
inline std::uint16_t floatToHalf(float f) {
  std::uint32_t b;
  __builtin_memcpy(&b, &f, 4);
  return f16_bits::f16FromF32Bits(b);
}
inline std::uint16_t quantGroup(const float* x, std::int8_t* q) {
  float amax = 0.f;
  for (nat i = 0; i < GROUP; ++i) amax = std::fabs(x[i]) > amax ? std::fabs(x[i]) : amax;
  std::uint16_t sh = floatToHalf(amax / 127.f);
  float d = halfToFloat(sh);
  for (nat i = 0; i < GROUP; ++i) {
    std::int64_t c = 0;
    if (d > 0.f) c = clamp127(std::int64_t(std::lround(x[i] / d)));
    q[i] = std::int8_t(c);
  }
  return sh;
}
inline float dequant(std::int8_t q, std::uint16_t sh) { return float(q) * halfToFloat(sh); }

}  // namespace baloon::kv_cache_q8
