// bl_native_quant_mat_vec.h - block layouts, thread-to-block mapping, grid rules and launch contracts of the
// native quantized mat-vec kernels (1..8 columns) plus the q8_1 activation quantiser.
#pragma once
#include <cmath>
#include <cstdint>
#include "bl_nat.h"

namespace baloon::native_mmvq {

constexpr nat THREADS = 128;  // 4 warps x 32 lanes
constexpr nat Q81_BYTES = 36;

struct Geometry { nat type, elems, bytes, T, BPI; };
// type id, values per block, bytes per block, lanes per weight block (T), blocks per iteration (BPI): T*BPI = 128.
constexpr Geometry GEOMETRY[10] = {
    {2, 32, 18, 16, 8},    // Q4_0
    {6, 32, 22, 16, 8},    // Q5_0
    {8, 32, 34, 2, 64},    // Q8_0
    {20, 32, 18, 16, 8},   // IQ4_NL
    {11, 256, 110, 32, 4}, // Q3_K
    {12, 256, 144, 8, 16}, // Q4_K
    {13, 256, 176, 2, 64}, // Q5_K
    {14, 256, 210, 2, 64}, // Q6_K
    {23, 256, 136, 4, 32}, // IQ4_XS
    {42, 64, 18, 2, 64},   // Q2_0
};
constexpr nat SUPPORTED[16] = {2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29};
constexpr nat IQ_ROUTE[6] = {16, 17, 18, 21, 22, 29};
constexpr nat NATIVE_ROUTE[10] = {2, 6, 8, 20, 11, 12, 13, 14, 23, 42};

template <std::size_t N> constexpr bool contains(const nat (&a)[N], nat t) {
  for (nat v : a) if (v == t) return true;
  return false;
}
constexpr const Geometry* geometryOf(nat type) {
  for (const Geometry& g : GEOMETRY) if (g.type == type) return &g;
  return nullptr;
}
constexpr bool supported(nat type) { return contains(SUPPORTED, type); }
constexpr bool isIqRoute(nat type) { return contains(IQ_ROUTE, type); }
constexpr bool isNativeRoute(nat type) { return contains(NATIVE_ROUTE, type); }

// Thread mapping: thread tid handles weight block group tid / T and intra-block lane tid % T.
constexpr nat threadGroup(nat tid, nat T) { return ndiv(tid, T); }
constexpr nat threadLane(nat tid, nat T) { return nmod(tid, T); }
// Rows per block: small K (fewer weight blocks per row than BPI) uses 4, otherwise 1; upstream multi-column 2.
constexpr nat rowsPerBlock(nat blocksPerRow, nat BPI) { return blocksPerRow < BPI ? 4 : 1; }
constexpr nat gridBlocks(nat nOut, nat rows) { return ndiv(nOut + rows - 1, rows); }
constexpr nat weightBlockAddr(nat row, nat kbx, nat bpr) { return row * bpr + kbx; }
constexpr nat actBlockAddr(nat j, nat kby, nat stride) { return j * stride + kby; }
constexpr nat outAddr(nat j, nat row, nat nOut) { return j * nOut + row; }
constexpr nat partialBytesExact(nat ncols) { return 3 * ncols * 4 * 32 * 4; }

constexpr nat q81Bytes(nat nIn, nat ncols) { return ncols * (nIn / 32) * Q81_BYTES; }
constexpr nat weightBytes(nat nIn, nat blockElems, nat blockBytes, nat nOut) { return (nIn / blockElems) * blockBytes * nOut; }
// Overflow guard: row_bytes <= MAX / n_out  =>  row_bytes * n_out <= MAX.
constexpr bool weightBytesGuardOk(nat nIn, nat blockElems, nat blockBytes, nat nOut, nat maxBytes) {
  return nOut > 0 && (nIn / blockElems) * blockBytes <= maxBytes / nOut;
}
constexpr bool shapeOk(nat nIn, nat ncols, nat elems) { return nIn > 0 && elems > 0 && nIn % elems == 0 && 1 <= ncols && ncols <= 8; }
constexpr bool aligned4(nat addr) { return addr % 4 == 0; }
constexpr bool multiColumn(nat ncols) { return 2 <= ncols && ncols <= 8; }

// q8_1 activation block: scale d = amax/127, codes round-half-away(x/d), 0 when amax = 0; s = d * sum(codes).
struct Q81 { float d, s; std::int8_t q[32]; };
inline Q81 quantQ81(const float* x) {
  float amax = 0;
  for (nat i = 0; i < 32; ++i) amax = std::fabs(x[i]) > amax ? std::fabs(x[i]) : amax;
  Q81 b{};
  b.d = amax / 127.f;
  std::int32_t sum = 0;
  for (nat i = 0; i < 32; ++i) {
    std::int32_t c = amax == 0.f ? 0 : std::int32_t(std::lround(x[i] / b.d));
    c = c < -127 ? -127 : (c > 127 ? 127 : c);
    b.q[i] = std::int8_t(c);
    sum += c;
  }
  b.s = b.d * float(sum);
  return b;
}

}  // namespace baloon::native_mmvq
