// Unit 3 invariant tests: each CHECK mirrors a LAYER 1 statement of the unit theory.
// Exit code is non-zero iff any check fails.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <utility>
#include <vector>
#include "bl_kv_cache_q8.h"
#include "bl_native_gdn_preprocessing.h"
#include "bl_native_gdn_recurrence.h"
#include "bl_native_quant_mat_vec.h"
#include "bl_native_router_top10.h"
#include "bl_native_short_flash_attention.h"
#include "bl_ple_block.h"
#include "bl_qsa_block_scoring.h"
#include "bl_qsa_cache_indexer_attention.h"
#include "bl_qsa_indexer_pooling.h"

using baloon::nat;
static long g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail <= 40) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)
static std::mt19937 rng(12345);
static float rnd(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

static void test_kv_cache_q8() {
  using namespace baloon::kv_cache_q8;
  for (std::int64_t q = -1000; q <= 1000; ++q) {
    CHECK(-127 <= clamp127(q) && clamp127(q) <= 127);
    if (-127 <= q && q <= 127) CHECK(clamp127(q) == q);
    CHECK(clamp127(clamp127(q)) == clamp127(q));
    CHECK(std::int64_t(std::int8_t(clamp127(q))) == clamp127(q));  // int8 cast does not wrap
  }
  // append addressing
  for (nat ps = 1; ps <= 32; ps += 3)
    for (nat pos = 0; pos < 300; ++pos) {
      CHECK(ps * pageOf(pos, ps) + slotOf(pos, ps) == pos);
      CHECK(slotOf(pos, ps) < ps);
    }
  const nat NP = 5, KV = 3, PS = 7;
  std::set<nat> rows;
  for (nat p = 0; p < NP; ++p) for (nat h = 0; h < KV; ++h) for (nat s = 0; s < PS; ++s) {
    nat r = cacheRow(p, h, s, KV, PS);
    CHECK(r < NP * KV * PS);
    rows.insert(r);
  }
  CHECK(rows.size() == NP * KV * PS);  // bijective
  for (nat groups = 1; groups <= 4; ++groups)
    for (nat row = 0; row < 6; ++row)
      for (nat g = 0; g < groups; ++g) {
        CHECK(scaleIndex(row, g, groups) < (row + 1) * groups);
        for (nat t = 0; t < 64; ++t) {
          CHECK(valueOffset(row, g, t, groups * 64) < (row + 1) * groups * 64);
          CHECK(valueOffset(row, g, t, groups * 64) >= row * groups * 64);
        }
      }
  for (nat t = 0; t < 64; ++t) CHECK(appendWarps(t) < 2);
  // gather decomposition: dense destination, in range, quads inside one group
  for (nat kv = 1; kv <= 3; ++kv)
    for (nat hd : {64u, 128u, 256u}) {
      nat per = hd / 4, ids = 4;
      for (nat i = 0; i < ids * kv * per; ++i) {
        GatherIdx g = gatherDecompose(i, kv, per);
        CHECK(g.id < ids && g.head < kv && g.q4 < per);
        CHECK(gatherDst(g.id, g.head, g.q4, kv, per) == i);  // scratch is dense
        CHECK(g.q4 * 4 + 3 < hd);
        CHECK((g.q4 * 4 + 3) / 64 == (g.q4 * 4) / 64);
        CHECK(quadGroup(g.q4) < hd / 64);
      }
    }
  CHECK(validShapes(64, 2, 16) && validShapes(256, 1, 1) && !validShapes(100, 2, 16) && !validShapes(64, 0, 16) && !validShapes(64, 2, 0));
  for (nat hd = 0; hd < 1024; ++hd) if (validShapes(hd, 1, 1)) CHECK(hd % 4 == 0);
  for (nat total = 0; total < 2000; ++total) CHECK(gridBlocks(total) * 256 >= total);
  // quantiser: stored (fp16-rounded) scale, round trip error within about half a step
  for (int trial = 0; trial < 200; ++trial) {
    float x[64]; std::int8_t q[64];
    float mag = std::pow(10.f, rnd(-3.f, 3.f));
    for (float& v : x) v = rnd(-mag, mag);
    std::uint16_t sh = quantGroup(x, q);
    float d = halfToFloat(sh);
    for (nat i = 0; i < 64; ++i) {
      CHECK(q[i] >= -127 && q[i] <= 127);
      CHECK(std::fabs(dequant(q[i], sh) - x[i]) <= 0.65f * d + 1e-30f);
    }
  }
  { float z[64] = {}; std::int8_t q[64]; std::uint16_t sh = quantGroup(z, q);
    for (nat i = 0; i < 64; ++i) { CHECK(q[i] == 0); }
    CHECK(halfToFloat(sh) == 0.f); }
}

static void test_flash_attention() {
  using namespace baloon::native_short_flash;
  for (std::int64_t w = -2; w <= 300; ++w) {
    CHECK(validStep(w, w, w - 1, w / 4, 256) == (1 <= w && w <= 256));
    CHECK(!validStep(w, w + 1, w - 1, w / 4, 256));       // nkv mismatch
    CHECK(!validStep(w, w, w, w / 4, 256));               // pos mismatch
    CHECK(!validStep(w, w, w - 1, w / 4 + 1, 256));       // nbid mismatch
    if (validStep(w, w, w - 1, w / 4, 256)) { CHECK(w <= 256); CHECK(w - 1 < w); }
  }
  CHECK(!validStep(10, 10, 9, 2, 8));  // width above max context
  CHECK(N_HEAD / N_KV == HEADS_PER_KV);
  for (nat h = 0; h < 24; ++h) { CHECK(kvHead(h) < 2); CHECK(kvHead(h) * 12 <= h && h < kvHead(h) * 12 + 12); }
  // tile cell mapping: in range, injective, covering [0,128)
  std::set<nat> cells;
  for (nat w = 0; w < WARPS; ++w) for (nat g = 0; g < 4; ++g) for (nat r = 0; r < ROWS; ++r) {
    nat c = tileCell(w, g * 8 + 3, r);  // any lane of the group gives the same cell
    CHECK(c < TILE);
    CHECK(tileCell(w, g * 8, r) == c);
    cells.insert(c);
  }
  CHECK(cells.size() == TILE);
  CHECK(2 * TILE == PAD);
  for (nat base : {0u, 128u}) for (nat l = 0; l < TILE; ++l) CHECK(base + l < PAD);
  // K/V addressing and register slots
  for (nat cap : {256u, 300u}) for (nat c = 0; c < cap; c += 17) for (nat kv = 0; kv < 2; ++kv) for (nat d = 0; d < 256; d += 31)
    CHECK(kvIndex(c, kv, d) < cap * 2 * 256);
  CHECK(2 * 256 * 2 == 1024);
  std::set<nat> dims;
  for (nat m = 0; m < 4; ++m) for (nat l = 0; l < 8; ++l) for (nat j = 0; j < 4; ++j) {
    CHECK(dimPair(m, l, j) + 1 < 256);
    CHECK((32 * m) / 8 + j < 16);
    dims.insert(dimPair(m, l, j));
  }
  CHECK(dims.size() == 128);
  // output reduction
  CHECK(REDUCE_FLOATS == 4096);
  std::set<nat> outs;
  for (nat w = 0; w < 4; ++w) for (nat g = 0; g < 4; ++g) for (nat i0 : {0u, 128u}) for (nat t = 0; t < 128; ++t)
    CHECK(reduceIndex(w, g, i0, t) < 4096);
  for (nat i0 : {0u, 128u}) for (nat t = 0; t < 128; ++t) outs.insert(i0 + t);
  CHECK(outs.size() == 256 && *outs.rbegin() == 255);
  for (nat h = 0; h < 24; ++h) for (nat e = 0; e < 256; e += 5) CHECK(outIndex(h, e) < 24 * 256);
  CHECK(supportedShape(24, 2, 256, 256, 256) && !supportedShape(24, 2, 128, 256, 256) && !supportedShape(24, 2, 256, 255, 256) &&
        !supportedShape(24, 2, 256, 512, 257) && !supportedShape(24, 2, 256, 512, 0));
  for (std::int64_t cap = 256; cap < 300; ++cap) for (std::int64_t m = 1; m <= 256; m += 15) CHECK(supportedShape(24, 2, 256, cap, m) && m <= cap);
  // numerics: tile-ordered online softmax equals the naive softmax for every context length class
  for (nat width : {1u, 2u, 31u, 127u, 128u, 129u, 200u, 256u}) {
    std::vector<float> K(256 * 2 * 256), V(256 * 2 * 256), q(256), a(256), b(256);
    for (float& v : K) v = rnd(-1, 1);
    for (float& v : V) v = rnd(-1, 1);
    for (float& v : q) v = rnd(-1, 1);
    refAttention(q.data(), K.data(), V.data(), width, 1, 1.f / 16.f, a.data());
    tiledAttention(q.data(), K.data(), V.data(), width, 1, 1.f / 16.f, b.data());
    for (nat d = 0; d < 256; ++d) CHECK(std::fabs(a[d] - b[d]) < 2e-4f);
  }
}

static void test_gdn_recurrence() {
  using namespace baloon::native_gdn;
  std::set<nat> cols;
  for (nat z = 0; z < 32; ++z) for (nat y = 0; y < 4; ++y) { CHECK(column(z, y) < 128); cols.insert(column(z, y)); }
  CHECK(cols.size() == 128);
  std::set<nat> rows;
  for (nat r = 0; r < 4; ++r) for (nat l = 0; l < 32; ++l) { CHECK(laneRow(r, l) < 128); rows.insert(laneRow(r, l)); }
  CHECK(rows.size() == 128);
  CHECK(4 * 32 == 128 && 32 * 4 == 128);
  for (nat hv : {1u, 2u, 8u, 32u}) {
    std::set<nat> st;
    for (nat i = 0; i < 128; i += 7) for (nat h = 0; h < hv; ++h) for (nat c = 0; c < 128; c += 5) {
      CHECK(stateElem(i, h, c, hv) < 128 * hv * 128);
      CHECK(voElem(h, c) < hv * 128);
    }
    for (nat i = 0; i < 128; ++i) for (nat h = 0; h < hv; ++h) for (nat c = 0; c < 128; ++c) st.insert(stateElem(i, h, c, hv));
    CHECK(st.size() == 128 * hv * 128);
  }
  for (nat hk : {1u, 2u, 4u}) for (nat h = 0; h < 16; ++h) for (nat i = 0; i < 128; i += 9) {
    CHECK(qkHead(h, hk) < hk);
    CHECK(qkElem(h, i, hk) < hk * 128);
  }
  CHECK(validShape(128, 2, 4) && validShape(128, 4, 4) && validShape(128, 1, 65535) && !validShape(64, 2, 4) && !validShape(128, 0, 4) &&
        !validShape(128, 2, 0) && !validShape(128, 2, 65536) && !validShape(128, 3, 4));
  for (nat hk = 1; hk <= 8; ++hk) for (nat hv = 1; hv <= 64; ++hv) if (validShape(128, hk, hv)) CHECK(hv <= 65535 && hv % hk == 0);
  for (nat hv = 1; hv < 50; ++hv) CHECK(stateBytes(hv) == 128 * voBytes(hv));
  CHECK(qkBytes(2) == 1024 && headBytes(8) == 32);
  CHECK(!overlaps(100, 40, 140, 8) && !overlaps(140, 8, 100, 40) && overlaps(100, 41, 140, 8));
  for (nat a = 0; a < 20; a += 3) for (nat an = 1; an < 8; ++an) for (nat b = 0; b < 20; b += 2) for (nat bn = 1; bn < 8; ++bn)
    CHECK(overlaps(a, an, b, bn) == overlaps(b, bn, a, an));
  // numerics: warp-split reduction equals the direct delta rule
  for (int trial = 0; trial < 20; ++trial) {
    float s[128], s2[128], q[128], k[128];
    for (nat i = 0; i < 128; ++i) { s[i] = s2[i] = rnd(-1, 1); q[i] = rnd(-1, 1); k[i] = rnd(-0.2f, 0.2f); }
    float v = rnd(-1, 1), g = rnd(-1, 0), beta = rnd(0, 1);
    float out = stepColumn(s, q, k, v, g, beta);
    float dec = std::exp(g), kv = 0;
    for (nat i = 0; i < 128; ++i) { s2[i] *= dec; kv += k[i] * s2[i]; }
    float delta = (v - kv) * beta, o2 = 0;
    for (nat i = 0; i < 128; ++i) { s2[i] += k[i] * delta; o2 += q[i] * s2[i]; }
    o2 /= std::sqrt(128.f);
    CHECK(std::fabs(out - o2) < 1e-3f);
    for (nat i = 0; i < 128; ++i) CHECK(std::fabs(s[i] - s2[i]) < 1e-4f);
  }
}

static void test_gdn_preprocessing() {
  using namespace baloon::native_gdn_pre;
  // conv: window [h0,h1,h2,x], new history is the window tail, newest input hits the last tap
  float h[3] = {1, 2, 3}, w[4];
  window(h, 4, w);
  CHECK(w[0] == 1 && w[1] == 2 && w[2] == 3 && w[3] == 4);
  float taps[4] = {10, 20, 30, 40};
  CHECK(dot4(taps, h, 4) == 1 * 10 + 2 * 20 + 3 * 30 + 4 * 40);
  float raw;
  float outv = convStep(h, taps, 4, &raw);
  CHECK(raw == 300 && h[0] == 2 && h[1] == 3 && h[2] == 4);
  CHECK(std::fabs(outv - silu(300.f)) < 1e-3f);
  float h3[3] = {9, 9, 9};
  nextHistory(h3, 1); nextHistory(h3, 2); nextHistory(h3, 3);
  CHECK(h3[0] == 1 && h3[1] == 2 && h3[2] == 3);  // three steps leave the last three inputs
  for (nat c = 0; c < 50; ++c) for (nat t = 0; t < 3; ++t) { CHECK(histAddr(c, t) < 50 * 3); CHECK(tapAddr(c, t) < 50 * 4); CHECK(tapAddr(c, 3) < 50 * 4); }
  for (nat n = 0; n < 3000; ++n) CHECK(gridBlocks256(n) * 256 >= n);
  CHECK(convSpanBytes(10) == 7 * 10 * 4);
  // L2 norm: output is x / sqrt(sum + eps); unit norm as eps -> 0
  for (int trial = 0; trial < 50; ++trial) {
    float x[128], o[128], sum = 0, eps = 1e-6f;
    for (float& v : x) { v = rnd(-3, 3); sum += v * v; }
    l2NormRow(x, eps, o);
    float n2 = 0;
    for (nat i = 0; i < 128; ++i) { CHECK(std::fabs(o[i] - x[i] / std::sqrt(sum + eps)) < 1e-4f); n2 += o[i] * o[i]; }
    CHECK(std::fabs(n2 - 1.f) < 1e-3f);
  }
  for (nat row = 0; row < 5; ++row) for (nat col = 0; col < 128; ++col) CHECK(normRowAddr(row, col) < 5 * 128);
  for (nat t = 0; t < 256; ++t) CHECK(threadActive(t) == (t < 128));
  // gate / sigmoid
  for (double v = -30; v <= 30; v += 0.25) CHECK(softplusBranch(v) == (v > 20));
  CHECK(softplus(25.f) == 25.f);
  for (float v = -10; v <= 20; v += 0.5f) CHECK(std::fabs(softplus(v) - std::log(1.f + std::exp(v))) < 1e-4f);
  CHECK(std::fabs(gate(0.5f, 0.25f, -2.f) - softplus(0.75f) * -2.f) < 1e-6f);
  for (float v = -8; v <= 8; v += 0.5f) CHECK(std::fabs(sigmoid(v) + sigmoid(-v) - 1.f) < 1e-6f);
  // launch contracts
  CHECK(countOk(1) && countOk(65535) && !countOk(0) && !countOk(65536) && !countOk(-1));
  for (nat c = 1; c <= 65535; c += 97) CHECK(1 <= gridBlocks256(c) && gridBlocks256(c) <= 256);
  CHECK(normGeometryOk(1, 128) && normGeometryOk(65535, 128) && !normGeometryOk(0, 128) && !normGeometryOk(1, 64) && !normGeometryOk(65536, 128));
}

static void test_quant_mat_vec() {
  using namespace baloon::native_mmvq;
  // struct sizes
  CHECK(4 + 12 + 32 + 128 == 176 && 4 + 12 + 128 == 144 && 32 + 64 + 12 + 2 == 110 && 128 + 64 + 16 + 2 == 210 &&
        2 + 2 + 4 + 128 == 136 && 2 + 16 == 18 && 2 + 4 + 16 == 22 && 2 + 32 == 34 && 4 + 32 == 36);
  CHECK(Q81_BYTES == 36 && THREADS == 128);
  for (const Geometry& g : GEOMETRY) {
    CHECK(g.T * g.BPI == 128);  // every format covers all 128 threads
    CHECK(isNativeRoute(g.type) && supported(g.type));
    CHECK(g.elems % 32 == 0 && g.bytes > 0);
    for (nat tid = 0; tid < THREADS; ++tid) {
      CHECK(threadGroup(tid, g.T) < g.BPI && threadLane(tid, g.T) < g.T);
      CHECK(g.T * threadGroup(tid, g.T) + threadLane(tid, g.T) == tid);
    }
    std::set<std::pair<nat, nat>> seen;
    for (nat tid = 0; tid < THREADS; ++tid) seen.insert({threadGroup(tid, g.T), threadLane(tid, g.T)});
    CHECK(seen.size() == THREADS);
  }
  CHECK(geometryOf(13)->bytes == 176 && geometryOf(12)->bytes == 144 && geometryOf(11)->bytes == 110 && geometryOf(14)->bytes == 210 &&
        geometryOf(23)->bytes == 136 && geometryOf(42)->elems == 64 && geometryOf(99) == nullptr);
  CHECK(2 * 4 * 32 / 32 == 8 && 4 * 32 / 16 == 8 && 4 * 32 / 2 == 64 && 4 * 4 * 32 / 32 == 16 && 4 * 32 / 32 == 4);
  // dispatch table
  nat n = 0; std::set<nat> uniq;
  for (nat t : SUPPORTED) { ++n; uniq.insert(t); CHECK(isIqRoute(t) != isNativeRoute(t)); }
  CHECK(n == 16 && uniq.size() == 16);
  for (nat t = 0; t < 100; ++t) CHECK(supported(t) == (isIqRoute(t) || isNativeRoute(t)));
  // rows per block and grids
  CHECK(rowsPerBlock(7, 8) == 4 && rowsPerBlock(8, 8) == 1 && rowsPerBlock(100, 8) == 1);
  for (nat nOut = 1; nOut < 300; ++nOut) for (nat rpb : {1u, 2u, 4u}) {
    CHECK(gridBlocks(nOut, rpb) * rpb >= nOut);
    CHECK(gridBlocks(nOut, rpb) * rpb < nOut + rpb);
    for (nat b = 0; b < gridBlocks(nOut, rpb); b += 11) for (nat i = 0; i < rpb; ++i) CHECK(rpb * b + i < gridBlocks(nOut, rpb) * rpb);
  }
  for (nat row = 0; row < 9; ++row) for (nat k = 0; k < 6; ++k) CHECK(weightBlockAddr(row, k, 6) < 9 * 6);
  for (nat j = 0; j < 8; ++j) for (nat k = 0; k < 7; ++k) { CHECK(actBlockAddr(j, k, 7) < 8 * 7); CHECK(outAddr(j, k, 7) < 8 * 7); }
  for (nat nc = 1; nc <= 8; ++nc) CHECK(partialBytesExact(nc) <= 49152);
  CHECK(partialBytesExact(1) == 1536);
  // q8_1 accounting
  for (nat a = 0; a < 5; ++a) for (nat b = 0; b < 5; ++b) CHECK(q81Bytes(256, a + b) == q81Bytes(256, a) + q81Bytes(256, b));
  for (nat c = 0; c < 4; ++c) for (nat m = 1; m < 6; ++m) for (nat k = 0; k < 32 * m; ++k) {
    CHECK((c * (32 * m) + k) / 32 == c * m + k / 32);
    CHECK(c * m + k / 32 < (c + 1) * m);
  }
  // weight bytes
  for (nat o1 = 0; o1 < 5; ++o1) for (nat o2 = 0; o2 < 5; ++o2)
    CHECK(weightBytes(512, 256, 144, o1 + o2) == weightBytes(512, 256, 144, o1) + weightBytes(512, 256, 144, o2));
  CHECK(weightBytes(4096, 256, 144, 10) == 16 * 144 * 10);
  for (nat nOut = 1; nOut < 40; nOut += 3) for (nat M : {1000u, 100000u})
    if (weightBytesGuardOk(512, 32, 18, nOut, M)) CHECK(weightBytes(512, 32, 18, nOut) <= M);
  CHECK(!weightBytesGuardOk(512, 32, 18, 0, 100));
  CHECK(!weightBytesGuardOk(512, 32, 18, 1000, 100000));
  // launch contract
  CHECK(shapeOk(256, 1, 256) && shapeOk(512, 8, 256) && !shapeOk(0, 1, 256) && !shapeOk(100, 1, 256) && !shapeOk(256, 0, 256) && !shapeOk(256, 9, 256));
  for (nat nIn = 1; nIn < 1200; ++nIn) for (const Geometry& g : GEOMETRY) if (shapeOk(nIn, 1, g.elems)) CHECK(nIn % 32 == 0);
  CHECK(aligned4(0) && aligned4(8) && !aligned4(6));
  for (nat a = 0; a < 40; a += 4) for (nat off = 0; off < 40; off += 4) CHECK(aligned4(a + off));
  CHECK(!multiColumn(1) && multiColumn(2) && multiColumn(8) && !multiColumn(9));
  // q8_1 quantiser
  for (int trial = 0; trial < 100; ++trial) {
    float x[32]; for (float& v : x) v = rnd(-5, 5);
    Q81 b = quantQ81(x);
    float s = 0;
    for (nat i = 0; i < 32; ++i) { CHECK(b.q[i] >= -127 && b.q[i] <= 127); CHECK(std::fabs(float(b.q[i]) * b.d - x[i]) <= 0.5f * b.d * 1.001f + 1e-6f); s += float(b.q[i]); }
    CHECK(std::fabs(b.s - b.d * s) < 1e-4f);
  }
  float z[32] = {}; Q81 zb = quantQ81(z);
  CHECK(zb.d == 0.f && zb.s == 0.f);
  for (nat i = 0; i < 32; ++i) CHECK(zb.q[i] == 0);
}

static void test_qsa_indexer() {
  using namespace baloon::qsa_indexer;
  for (nat pos = 0; pos < 400; ++pos) {
    CHECK(R * blockOf(pos) + slotOf(pos) == pos && slotOf(pos) < R);
    CHECK(poolTriggered(pos) == (pos == 0 || pos % 4 == 3));
    if (pos != 0 && poolTriggered(pos)) CHECK(pos == 4 * blockOf(pos) + 3);
    for (nat mc : {pos + 1, pos + 17}) {
      CHECK(blockOf(pos) < pooledRows(mc));
      if (pos % 4 == 3) CHECK(seedRow(pos) < pooledRows(mc));
    }
  }
  for (nat slot = 0; slot < 3; ++slot) for (nat d = 0; d < 128; ++d) CHECK(tailIndex(slot, d) < 3 * 128);
  for (nat mc = 1; mc < 200; ++mc) {
    CHECK(pooledRows(mc) >= 1);
    for (nat row = 0; row < pooledRows(mc); ++row) {
      CHECK(pooledElem(row, 127) < pooledRows(mc) * 128);
      CHECK((row + 1) * 128 * 4 <= pooledBytes(mc));
    }
  }
  for (nat pb = 0; pb < 64; pb += 4) for (nat b = 0; b < 30; ++b) CHECK(ropePos(pb, b) % 4 == 0);
  for (nat pb = 0; pb < 64; pb += 4) for (nat mc = 1; mc < 40; ++mc) for (nat pos = 0; pos < mc; ++pos) CHECK(pb + 4 * (pos / 4) < pb + mc);
  for (nat d = 0; d < 64; ++d) { CHECK(pairOf(d) < 32); CHECK(pairOf(d) + 32 < 64); CHECK(d < 32 ? pairOf(d) == d : pairOf(d) == d - 32); }
  CHECK(appendArgsValid(128, 4, 64, 1000, 8) && !appendArgsValid(64, 4, 64, 1000, 8) && !appendArgsValid(128, 8, 64, 1000, 8) &&
        !appendArgsValid(128, 4, 32, 1000, 8) && !appendArgsValid(128, 4, 64, 0, 8) && !appendArgsValid(128, 4, 64, 1000, 6) &&
        !appendArgsValid(128, 4, 64, 1000, -4) && !appendArgsValid(128, 4, 64, 2147483647, 4) && appendArgsValid(128, 4, 64, 2147483647, 0));
  // streaming reference equals direct pooling of the four keys; rotation keeps norm; dims >= 64 are not rotated
  const nat MC = 40;
  std::vector<float> gamma(128), keys(MC * 128);
  for (float& g : gamma) g = rnd(0.5f, 1.5f);
  for (float& k : keys) k = rnd(-1, 1);
  Indexer ix(MC, 8, 1e-6f, 10000.f, gamma);
  for (nat pos = 0; pos < MC; ++pos) ix.append(pos, &keys[pos * 128]);
  for (nat b = 0; b < MC / 4; ++b) {
    float ref[128];
    nat count = b == 0 ? 4 : 4;
    poolBlock(&keys[b * 4 * 128], count, gamma.data(), 1e-6f, 10000.f, 8 + 4 * b, ref);
    if (b == 0) {  // block 0 was first pooled alone at position 0 and rewritten when it closed
      for (nat d = 0; d < 128; ++d) CHECK(std::fabs(ix.pooled[d] - ref[d]) < 1e-5f);
    } else {
      for (nat d = 0; d < 128; ++d) CHECK(std::fabs(ix.pooled[b * 128 + d] - ref[d]) < 1e-5f);
    }
    float mean[128], ss = 0, nout = 0, nin = 0;
    for (nat d = 0; d < 128; ++d) { mean[d] = 0; for (nat t = 0; t < 4; ++t) mean[d] += keys[(b * 4 + t) * 128 + d] / 4.f; ss += mean[d] * mean[d]; }
    float inv = 1.f / std::sqrt(ss / 128.f + 1e-6f);
    for (nat d = 64; d < 128; ++d) CHECK(std::fabs(ref[d] - mean[d] * inv * gamma[d]) < 1e-5f);  // unrotated tail dims
    for (nat p = 0; p < 32; ++p) {
      float a = mean[p] * inv * gamma[p], c = mean[p + 32] * inv * gamma[p + 32];
      nin += a * a + c * c; nout += ref[p] * ref[p] + ref[p + 32] * ref[p + 32];
    }
    CHECK(std::fabs(nin - nout) < 1e-3f * (1.f + nin));
  }
  // position 0 alone pools a single key (mean over one element)
  { float one[128], gm[128], o[128], ssq = 0;
    for (nat d = 0; d < 128; ++d) { one[d] = rnd(-1, 1); gm[d] = 1; ssq += one[d] * one[d]; }
    poolBlock(one, 1, gm, 1e-6f, 10000.f, 0, o);  // angle 0: pure RMS norm
    float inv = 1.f / std::sqrt(ssq / 128.f + 1e-6f);
    for (nat d = 0; d < 128; ++d) CHECK(std::fabs(o[d] - one[d] * inv) < 1e-5f); }
}

static void test_qsa_block_scoring() {
  using namespace baloon::qsa_score;
  for (std::int64_t n = -2; n < 300; ++n) {
    bool v = validStep(n, 300, n - 1, n / 4, n < 2051 ? n : 2051);
    CHECK(v == (n >= 1));
    if (v) { CHECK((n / 4) * 4 <= n); CHECK(n - 1 >= 0); }
    CHECK(!validStep(n, 300, n, n / 4, n));
  }
  CHECK(validStep(5000, 6000, 4999, 1250, 2051) && !validStep(5000, 6000, 4999, 1250, 5000) && !validStep(5000, 4000, 4999, 1250, 2051));
  for (nat mb = 0; mb < 500; ++mb) CHECK(gridBlocks(mb) * 32 >= mb);
  std::set<nat> r;
  for (nat b = 0; b < 4; ++b) for (nat l = 0; l < 32; ++l) r.insert(rowOf(b, l));
  CHECK(r.size() == 128);
  for (nat n = 1; n < 300; ++n) for (nat mc = n; mc < n + 5; ++mc) CHECK(n / 4 < mc / 4 + 1);
  for (nat n = 1; n < 100; ++n) for (nat row = 0; row <= n / 4; ++row) CHECK(row < maxBlocksFor(n));
  // K split across 2 warps: injective, covers all 128 columns exactly once
  std::set<nat> cols;
  for (nat w = 0; w < 2; ++w) for (nat l = 0; l < 32; ++l) for (nat it = 0; it < 2; ++it) { CHECK(kCol(w, l, it) < 128); cols.insert(kCol(w, l, it)); }
  CHECK(cols.size() == 128);
  // shared layout
  CHECK(sharedTileFloats() == 1152 && HEADS * COMBINE <= sharedTileFloats());
  for (nat tid = 0; tid < 32; ++tid) { CHECK(loadAIndex(tid) + 3 < 16 * 36); CHECK(loadBIndex(tid) + 3 < 8 * 36); }
  for (nat i = 0; i < 16; ++i) for (nat l = 0; l < 32; ++l) CHECK(i * STRIDE + l < 16 * STRIDE);
  std::set<std::pair<nat, nat>> comb;
  for (nat w = 0; w < 2; ++w) for (nat ia = 0; ia < 2; ++ia) for (nat l = 0; l < 4; ++l) for (nat lane = 0; lane < 32; ++lane) {
    nat i = combineRow(w, ia, l, lane), h = combineHead(l, lane);
    CHECK(i < 64 && h < 8 && combineAddr(h, i) < HEADS * COMBINE);
    comb.insert({h, i});
  }
  CHECK(comb.size() == 8 * 64);  // every (head, row) hit exactly once
  for (nat h = 0; h < 8; ++h) for (nat lane = 0; lane < 32; ++lane) CHECK(combineAddr(h, lane) < HEADS * COMBINE && combineAddr(h, 32 + lane) < HEADS * COMBINE);
  // cell export: each cell below n exported by exactly one scored row; partial row only when n % 4 != 0
  for (nat n = 1; n < 120; ++n) {
    std::vector<int> hit(n, 0);
    for (nat row = 0; row <= n / 4 + 1; ++row) {
      if (!rowScored(row, n)) { CHECK(row > n / 4); continue; }
      for (nat i = cellLo(row); i < cellHi(row, n); ++i) { CHECK(i < n); ++hit[i]; }
    }
    for (nat i = 0; i < n; ++i) { CHECK(hit[i] == 1); CHECK(i / 4 <= n / 4); CHECK(4 * (i / 4) + i % 4 == i); }
    CHECK(rowPartial(n / 4, n) == (n % 4 != 0));
    if (n % 4 != 0) CHECK(n / 4 * 4 < n);
    CHECK(capacitiesOk(n, n / 4 + 1) && !capacitiesOk(n, n / 4));
  }
  CHECK(!capacitiesOk(0, 1));
  // reference scoring: ReLU per head, sum, bias, partial penalty, replicated to cells
  std::vector<float> pooledRow(128), q(4 * 128);
  for (float& v : pooledRow) v = rnd(-1, 1);
  for (float& v : q) v = rnd(-1, 1);
  float expect = 0.25f;
  for (nat h = 0; h < 4; ++h) { float a = 0; for (nat k = 0; k < 128; ++k) a += q[h * 128 + k] * pooledRow[k]; expect += a > 0 ? a : 0; }
  CHECK(std::fabs(scoreRow(pooledRow.data(), q.data(), 4, 0.25f, false) - expect) < 1e-4f);
  CHECK(scoreRow(pooledRow.data(), q.data(), 4, 0.25f, true) < -9e8f);
  CHECK(scoreRow(pooledRow.data(), q.data(), 4, 0.25f, false) >= 0.25f);  // sum of ReLUs is >= 0
  std::vector<float> scores = {1, 2, 3}, cells(10, -1.f);
  exportCells(scores, 10, cells);  // n = 10: rows 0..2, last row partial (cells 8, 9)
  for (nat i = 0; i < 10; ++i) CHECK(cells[i] == scores[i / 4]);
}

static void test_router() {
  using namespace baloon::native_router;
  for (nat lane = 0; lane < 32; ++lane) for (nat i = 0; i < 16; ++i) {
    CHECK(laneOf(expertOf(lane, i)) == lane && slotOf(expertOf(lane, i)) == i && expertOf(lane, i) < 512);
  }
  std::set<nat> es;
  for (nat lane = 0; lane < 32; ++lane) for (nat i = 0; i < 16; ++i) es.insert(expertOf(lane, i));
  CHECK(es.size() == 512);
  for (nat a = 0; a < 15; ++a) for (nat an = 0; an < 6; ++an) for (nat b = 0; b < 15; ++b) for (nat bn = 0; bn < 6; ++bn)
    CHECK(overlap(a, an, b, bn) == overlap(b, bn, a, an));
  CHECK(!overlap(0, 4, 4, 4) && overlap(0, 5, 4, 4));
  for (float v = 0; v < 3; v += 1) for (nat i = 0; i < 4; ++i) for (float v2 = 0; v2 < 3; v2 += 1) for (nat i2 = 0; i2 < 4; ++i2)
    if (better(v, i, v2, i2)) CHECK(!better(v2, i2, v, i));  // asymmetric
  CHECK(better(1.f, 5, 1.f, 6) && !better(1.f, 6, 1.f, 5) && !better(1.f, 5, 1.f, 5));
  for (nat r = 0; r < 10; ++r) CHECK(r < 32);
  CHECK(clampSum(0.f) == SUM_FLOOR && clampSum(1.f) == 1.f && SUM_FLOOR == std::ldexp(1.f, -14));
  // routing: matches the sort-based reference, ties break by lowest index, weights sum to 1
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<float> lg(512);
    for (float& v : lg) v = (trial % 3 == 0) ? float(int(rnd(0, 6))) : rnd(-4, 4);  // trial%3==0 forces many exact ties
    nat ids[10]; float w[10];
    routerTop10(lg.data(), ids, w);
    std::vector<nat> order(512);
    for (nat e = 0; e < 512; ++e) order[e] = e;
    std::stable_sort(order.begin(), order.end(), [&](nat a, nat b) { return lg[a] > lg[b]; });  // stable => lowest index first among ties
    float sel = 0, mx = *std::max_element(lg.begin(), lg.end()), den = 0;
    for (float v : lg) den += std::exp(v - mx);
    for (nat r = 0; r < 10; ++r) { CHECK(ids[r] == order[r]); sel += std::exp(lg[order[r]] - mx) / den; }
    float ws = 0;
    for (nat r = 0; r < 10; ++r) {
      CHECK(std::fabs(w[r] - std::exp(lg[order[r]] - mx) / den / sel) < 1e-5f);
      CHECK(r == 0 || w[r] <= w[r - 1] + 1e-7f);
      ws += w[r];
    }
    CHECK(std::fabs(ws - 1.f) < 1e-4f);
    std::set<nat> u(ids, ids + 10);
    CHECK(u.size() == 10);
  }
  // degenerate: all-equal logits pick experts 0..9 with equal weights 1/10
  { std::vector<float> lg(512, 1.5f); nat ids[10]; float w[10];
    routerTop10(lg.data(), ids, w);
    for (nat r = 0; r < 10; ++r) { CHECK(ids[r] == r); CHECK(std::fabs(w[r] - 0.1f) < 1e-5f); } }
}

static void test_ple() {
  using namespace baloon::ple;
  for (nat n = 0; n < 500; ++n) { CHECK(n <= round16(n) && round16(n) % 16 == 0 && round16(n) < n + 16); }
  CHECK(convRow(9, 4, 3, 0) == 0 && convRow(9, 4, 3, 1) == 3 && convRow(9, 4, 3, 2) == 6 && convRow(9, 4, 3, 3) == 9);
  for (nat nh = 1; nh < 12; ++nh) for (nat c = 0; c < 7; ++c) for (nat row = 0; row < nh; ++row) {
    CHECK(idx(nh, row, c) / nh == c && idx(nh, row, c) % nh == row);
  }
  nat sizes[5] = {10, 33, 16, 1, 64};
  for (nat k = 0; k < 4; ++k) CHECK(regionOffset(sizes, k) + round16(sizes[k]) == regionOffset(sizes, k + 1) && regionOffset(sizes, k) % 16 == 0);
  CHECK(regionOffset(sizes, 0) == 0 && regionOffset(sizes, 5) == 16 + 48 + 16 + 16 + 64);
  for (nat a = 0; a < 20; ++a) for (nat an = 1; an < 8; ++an) for (nat b = 0; b < 20; ++b) for (nat bn = 1; bn < 8; ++bn)
    CHECK(ovl(a, an, b, bn) == (a < b + bn && b < a + an));
  for (double s = -9; s <= 9; s += 0.5) {
    CHECK(sgn(-s) == -sgn(s));
    CHECK(std::fabs(gate(float(-s)) + gate(float(s)) - 1.f) < 1e-6f);  // symmetric about 0.5
  }
  CHECK(sgn(0) == 0 && gate(0.f) == 0.5f);
  // conv: taps read rows 0,3,6,9 of [history | new]; history shifts left and appends the new row
  const nat NH = 9, KERN = 4, DIL = 3, C = 3;
  std::vector<float> hist(NH * C), w(C * KERN);
  for (float& v : hist) v = rnd(-1, 1);
  for (float& v : w) v = rnd(-1, 1);
  for (int step = 0; step < 6; ++step) {
    std::vector<float> before = hist;
    for (nat c = 0; c < C; ++c) {
      float x = rnd(-1, 1);
      float expect = 0;
      for (nat k = 0; k < KERN; ++k) {
        nat row = nat(convRow(NH, KERN, DIL, k));
        expect += w[c * KERN + k] * (row == NH ? x : before[idx(NH, row, c)]);
      }
      float got = convStep(hist, w, NH, KERN, DIL, c, x);
      CHECK(std::fabs(got - silu(expect)) < 1e-5f);
      for (nat r = 0; r + 1 < NH; ++r) CHECK(hist[idx(NH, r, c)] == before[idx(NH, r + 1, c)]);
      CHECK(hist[idx(NH, NH - 1, c)] == x);
      CHECK(hist.size() == before.size());
    }
  }
}

static void test_qsa_attention() {
  using namespace baloon::qsa;
  for (nat nKv = 0; nKv < 100; ++nKv) for (nat topK : {1u, 8u, 64u}) {
    CHECK(selWidth(nKv, topK, 4) <= nKv);
    if (nKv <= topK + 3) CHECK(selWidth(nKv, topK, 4) == nKv);
  }
  for (nat pos = 0; pos < 200; ++pos) {
    Step s = stepOf(pos, 16, 4);
    CHECK(s.nKv == pos + 1 && s.nBid == (pos + 1) / 4 && s.width == std::min<nat>(pos + 1, 16 + 3));
    CHECK(s.nBid * 4 <= pos + 1);
    CHECK((pos + 1) / 4 * 4 + (pos + 1) % 4 == pos + 1);
    if (poolCompletes(pos, 4)) CHECK((pos + 1) % 4 == 0 && (pos + 1) / 4 == pos / 4 + 1 && poolCompletes(pos, 4) == (pos % 4 == 3));
  }
  for (nat h = 0; h < 24; ++h) CHECK(kvHead(h, 24, 2) < 2 && kvHead(h, 24, 2) == h / 12);
  for (nat ps = 1; ps < 20; ps += 3) for (nat pos = 0; pos < 200; ++pos) { CHECK(slotOf(pos, ps) < ps); CHECK(pageOf(pos, ps) * ps + slotOf(pos, ps) == pos); }
  CHECK(gatherAligned(256) && gatherAligned(64) && !gatherAligned(6));
  for (nat hd = 0; hd < 300; ++hd) if (gatherAligned(hd)) CHECK((hd / 4) * 4 == hd);
  for (nat gt = 0; gt <= 6; ++gt) for (nat eq = 0; eq <= 6; ++eq) for (nat width = gt; width <= gt + eq; ++width)
    CHECK(gt + std::min(eq, width - gt) == width);
  // threshold selection equals exact top-k (descending score, ties to the lowest index)
  for (int trial = 0; trial < 300; ++trial) {
    nat n = 1 + nat(rnd(0, 120));
    std::vector<float> s(n);
    for (float& v : s) v = (trial % 2) ? float(int(rnd(0, 5))) : rnd(-1, 1);  // odd trials: heavy ties
    nat width = 1 + nat(rnd(0, float(n)));
    if (width > n) width = n;
    std::vector<nat> got = thresholdSelect(s, width);
    CHECK(got.size() == width);
    std::vector<nat> order(n);
    for (nat i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](nat a, nat b) { return s[a] > s[b]; });
    std::vector<nat> want(order.begin(), order.begin() + width);
    std::sort(want.begin(), want.end());
    CHECK(got == want);
    CHECK(std::is_sorted(got.begin(), got.end()));
  }
  CHECK(thresholdSelect({}, 3).empty() && thresholdSelect({1.f, 2.f}, 0).empty());
}

int main() {
  test_kv_cache_q8();
  test_flash_attention();
  test_gdn_recurrence();
  test_gdn_preprocessing();
  test_quant_mat_vec();
  test_qsa_indexer();
  test_qsa_block_scoring();
  test_router();
  test_ple();
  test_qsa_attention();
  std::printf("u3: %ld checks, %ld failed\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
