// Unit 1 invariant tests: each CHECK mirrors a LAYER 1 statement of the unit theory.
// Exit code is non-zero iff any check fails.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>
#include "bl_cpu_expert_pool.h"
#include "bl_expert_blob_source.h"
#include "bl_expert_dispatch_split.h"
#include "bl_expert_vram_cache.h"
#include "bl_f16_bits.h"
#include "bl_gdn_qsa_layer_layout.h"
#include "bl_mtp_draft_rows.h"
#include "bl_session_loop_indexing.h"
#include "bl_verify_window_groups.h"
#include "bl_weights_loader_segments.h"

using baloon::nat;
static long g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail <= 40) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static void test_expert_vram_cache() {
  using namespace baloon::expert_vram_cache;
  for (nat slots = 0; slots <= 70; ++slots)
    for (nat nl = 1; nl <= 20; ++nl) {
      CHECK(layerLo(slots, nl, 0) == 0);
      CHECK(nl * (slots / nl) <= slots);
      CHECK(layerHi(slots, nl, nl - 1) == slots);
      nat covered = 0;
      for (nat l = 0; l < nl; ++l) {
        CHECK(layerLo(slots, nl, l) <= layerHi(slots, nl, l));
        covered += layerHi(slots, nl, l) - layerLo(slots, nl, l);
        if (l + 1 < nl) CHECK(layerHi(slots, nl, l) == layerLo(slots, nl, l + 1));
        for (nat l2 = l + 1; l2 < nl; ++l2) CHECK(layerHi(slots, nl, l) <= layerLo(slots, nl, l2));
      }
      CHECK(covered == slots);  // ranges tile 0..slots exactly
    }
  for (nat next = 0; next < 12; ++next)
    for (nat hi = 0; hi < 12; ++hi) {
      auto s = admitSlot(next, hi);
      if (s) CHECK(*s < hi && *s == next);
      if (next >= hi) CHECK(!s && admitNext(next, hi) == next);
      CHECK(admitNext(next, hi) >= next);
      if (next <= hi) CHECK(admitNext(next, hi) <= hi);
    }
  for (nat n = 0; n < 2000; ++n) {
    CHECK(align256(n) >= n && align256(n) % 256 == 0 && align256(n) < n + 256);
    CHECK(fillBytes(n, 777) <= 777);
    CHECK(fillBytes(0, 777) == 777);
    if (n > 0 && n <= 777) CHECK(fillBytes(n, 777) == n);
  }
  std::vector<nat> blobs = {1, 255, 256, 257, 1000, 4096, 5};
  nat sum = 0;
  for (nat b : blobs) sum += b;
  CHECK(offsetsFrom(0, blobs) >= sum && offsetsFrom(0, blobs) % 256 == 0);
  CHECK(offsetsFrom(512, blobs) >= 512 + sum && offsetsFrom(512, blobs) % 256 == 0);
  for (nat o : slotOffsets(256, blobs)) CHECK(o % 256 == 0);
  for (nat slot = 0; slot < 9; ++slot) CHECK(slot * 100 + 100 <= 9 * 100);

  // Concrete cache: no eviction, range containment, distinct slots.
  for (int perLayer = 0; perLayer < 2; ++perLayer) {
    ExpertVramCache c(23, 4, 8, perLayer);
    std::set<std::int64_t> used;
    std::vector<std::int64_t> first(4 * 8, ExpertVramCache::kNone);
    for (int round = 0; round < 3; ++round)
      for (nat l = 0; l < 4; ++l)
        for (nat e = 0; e < 8; ++e) {
          std::int64_t before = c.lookup(l, e);
          std::int64_t s = c.admit(l, e);
          if (before != ExpertVramCache::kNone) CHECK(s == before);  // resident keeps its slot
          if (s != ExpertVramCache::kNone) {
            CHECK(s >= 0 && nat(s) < 23);
            if (perLayer) CHECK(nat(s) >= layerLo(23, 4, l) && nat(s) < layerHi(23, 4, l));
            if (first[l * 8 + e] == ExpertVramCache::kNone) {
              first[l * 8 + e] = s;
              CHECK(used.insert(s).second);  // fresh slot is unique
            } else CHECK(first[l * 8 + e] == s);
          }
        }
    for (nat k = 0; k < 32; ++k) if (first[k] != ExpertVramCache::kNone) CHECK(c.lookup(k / 8, k % 8) == first[k]);
    CHECK(c.admit(4, 0) == ExpertVramCache::kNone && c.admit(0, 8) == ExpertVramCache::kNone);
    if (!perLayer) CHECK(used.size() == 23);  // global counter fills the arena completely
  }
}

static void test_expert_blob_source() {
  using namespace baloon::expert_blob_source;
  for (nat ne = 1; ne <= 9; ++ne)
    for (nat nl = 1; nl <= 9; ++nl)
      for (nat l = 0; l < nl; ++l)
        for (nat e = 0; e < ne; ++e) {
          CHECK(flat(ne, l, e) < nl * ne);
          CHECK(checked(nl, ne, l, e).value() == flat(ne, l, e));
          for (nat l2 = 0; l2 < nl; ++l2)
            for (nat e2 = 0; e2 < ne; ++e2)
              if (flat(ne, l, e) == flat(ne, l2, e2)) CHECK(l == l2 && e == e2);  // injective
          for (nat l2 = l + 1; l2 < nl; ++l2) CHECK(flat(ne, l, e) < flat(ne, l2, 0));
          CHECK(sliceOff(64, e) + 64 <= ne * 64);
        }
  for (nat ne = 1; ne < 9; ++ne) {
    CHECK(flat(ne, 0, ne) == flat(ne, 1, 0));            // why the separate check exists
    CHECK(!checked(5, ne, 0, ne).has_value());           // ... and the checked form refuses it
  }
  for (nat off = 0; off < 40; ++off)
    for (nat bb = 1; bb < 8; ++bb)
      for (nat mb = bb; mb < 10; ++mb) {
        nat total = off + bb;
        CHECK(off + mb <= arenaBytes(total, mb));
      }
  for (nat per = 1; per <= 4; ++per)
    for (nat ne = 0; ne <= 8; ++ne)
      for (nat a = 0; a <= ne; ++a)
        for (nat b = 0; b <= 9; ++b)
          if (ggufChunkFits(per, a, b, ne))
            for (nat k = 0; k < b; ++k) CHECK(a + k < ne && gguf_expert_in_range(per, a, b, ne, k));
}

static void test_expert_dispatch_split() {
  using namespace baloon::expert_dispatch_split;
  for (unsigned mask = 0; mask < 512; ++mask) {
    std::vector<nat> ids;
    for (nat i = 0; i < 9; ++i) ids.push_back(i + 100);
    auto hit = [&](nat e) { return (mask >> (e - 100)) & 1u; };
    Split s = splitRouted(ids, hit);
    CHECK(s.hitRows.size() + s.missRows.size() == ids.size());
    CHECK(s.missCount() <= ids.size());
    std::set<nat> seen;
    for (nat r : s.hitRows) CHECK(seen.insert(r).second && hit(ids[r]));
    for (nat r : s.missRows) CHECK(seen.insert(r).second && !hit(ids[r]));
    CHECK(seen.size() == ids.size());
    if (mask == 0) for (nat j = 0; j < s.missRows.size(); ++j) CHECK(s.missRows[j] == j);  // job index == row
  }
  for (nat nmiss = 0; nmiss <= 64; ++nmiss) {
    CHECK(pcieShare(nmiss, 0) == 0);
    for (nat p = 0; p <= 256; ++p) {
      nat m = pcieShare(nmiss, p);
      CHECK(m <= nmiss);
      CHECK(nmiss - pcieRankFirst(nmiss, m) == m);
    }
    CHECK(pcieShare(nmiss, 256) == nmiss);
  }
}

static void test_cpu_expert_pool() {
  using namespace baloon::cpu_expert_pool;
  for (nat rows = 0; rows <= 200; ++rows)
    for (nat n = 1; n <= 40; ++n) {
      CHECK(rowLo(rows, n, 0) == 0);
      CHECK(rowHi(rows, n, n - 1) == rows);
      nat covered = 0;
      for (nat i = 0; i < n; ++i) {
        CHECK(rowLo(rows, n, i) <= rowHi(rows, n, i));
        CHECK(rowHi(rows, n, i) <= rows);
        CHECK(rowHi(rows, n, i) == rowLo(rows, n, i + 1));
        covered += rowHi(rows, n, i) - rowLo(rows, n, i);
      }
      CHECK(covered == rows);
    }
  for (nat cores = 1; cores <= 128; ++cores) {
    CHECK(poolThreads(cores, true) == cores);
    CHECK(poolThreads(cores, false) + 1 == cores);
  }
}

// ---- fp16 reference: decode via ldexp, encode by nearest-neighbour search over all codes ----
static double ref_decode(unsigned h) {
  int sign = (h >> 15) & 1, ex = (h >> 10) & 31, man = h & 1023;
  double v;
  if (ex == 0) v = std::ldexp(double(man), -24);
  else if (ex == 31) v = man ? NAN : INFINITY;
  else v = std::ldexp(1.0 + man / 1024.0, ex - 15);
  return sign ? -v : v;
}
static float bits_to_f(std::uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static std::uint32_t f_to_bits(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }

static void test_f16_bits() {
  using namespace baloon::f16_bits;
  for (nat h = 0; h < 65536; ++h) {
    CHECK(h == signF(h) * 32768 + exF(h) * 1024 + manF(h));
    CHECK(exF(h) < 32 && manF(h) < 1024 && signF(h) <= 1);
  }
  // Exhaustive decode vs reference; NaNs stay NaN; encode(decode(h)) == h for every non-NaN code.
  for (unsigned h = 0; h < 65536; ++h) {
    std::uint32_t b = baloon::f16_bits::f32BitsFromF16(std::uint16_t(h));
    float f = bits_to_f(b);
    double r = ref_decode(h);
    if (std::isnan(r)) { CHECK(std::isnan(f)); CHECK(((b >> 23) & 255) == 255); continue; }
    CHECK(double(f) == r && std::signbit(f) == std::signbit(r));
    CHECK(f16FromF32Bits(b) == h);
    unsigned ex = (h >> 10) & 31;
    if (ex >= 1 && ex <= 30) CHECK(((b >> 23) & 255) == ex + 112);
    if (ex == 31) CHECK(((b >> 23) & 255) == 255);
  }
  // Encode: nearest-even against neighbours, on dense structured + pseudo-random inputs.
  auto check_encode = [&](float x) {
    if (std::isnan(x)) { CHECK((f16FromF32Bits(f_to_bits(x)) & 0x7C00) == 0x7C00 && (f16FromF32Bits(f_to_bits(x)) & 0x3FF)); return; }
    unsigned h = f16FromF32Bits(f_to_bits(x));
    double ax = std::fabs(double(x));
    unsigned mag = h & 0x7FFF;
    CHECK((h >> 15) == (std::signbit(x) ? 1u : 0u));
    if (ax >= 65520.0) { CHECK(mag == 0x7C00); return; }  // overflow threshold (largest finite + half ulp)
    CHECK(mag < 0x7C00);
    double d = std::fabs(ax - ref_decode(mag));
    if (mag > 0) { double dl = std::fabs(ax - ref_decode(mag - 1)); CHECK(d <= dl); if (d == dl) CHECK((mag & 1) == 0); }
    if (mag < 0x7BFF) { double dh = std::fabs(ax - ref_decode(mag + 1)); CHECK(d <= dh); if (d == dh) CHECK((mag & 1) == 0); }
  };
  std::uint32_t st = 12345u;
  for (int i = 0; i < 400000; ++i) { st = st * 1664525u + 1013904223u; check_encode(bits_to_f(st)); }
  // Exact ties and near-ties around every adjacent pair of codes (incl. subnormals, max finite, carry into inf).
  for (unsigned h = 0; h < 0x7C00; ++h) {
    double lo = ref_decode(h), hi = ref_decode(h + 1);
    if (h + 1 == 0x7C00) hi = 65536.0;  // next "virtual" value beyond max finite
    double mid = (lo + hi) / 2;
    float fm = float(mid);
    if (double(fm) == mid) {
      check_encode(fm); check_encode(-fm);
      check_encode(std::nextafterf(fm, 1e30f)); check_encode(std::nextafterf(fm, -1e30f));
    }
  }
  CHECK(f16FromF32Bits(f_to_bits(65504.0f)) == 0x7BFF);
  CHECK(f16FromF32Bits(f_to_bits(65519.9f)) == 0x7BFF);
  CHECK(f16FromF32Bits(f_to_bits(65520.0f)) == 0x7C00);
  CHECK(f16FromF32Bits(f_to_bits(1e30f)) == 0x7C00 && f16FromF32Bits(f_to_bits(-1e30f)) == 0xFC00);
  CHECK(f16FromF32Bits(f_to_bits(std::ldexp(1.0f, -24))) == 1);
  CHECK(f16FromF32Bits(f_to_bits(std::ldexp(1.0f, -25))) == 0);   // tie to even (0)
  CHECK(f16FromF32Bits(f_to_bits(std::ldexp(1.5f, -25))) == 1);   // 1.5 * 2^-25 is a tie -> even... below: 3*2^-26 > half
  CHECK(f16FromF32Bits(f_to_bits(std::ldexp(1.0f, -26))) == 0);
  CHECK(f16FromF32Bits(f_to_bits(1.0f)) == 0x3C00 && f16FromF32Bits(f_to_bits(-2.0f)) == 0xC000);
  CHECK((0x7BFFu + 1u) == 0x7C00u);
}

static void test_gdn_qsa_layer_layout() {
  using namespace baloon::gdn_qsa_layer_layout;
  for (nat n = 0; n <= 4096; n += 256) CHECK(n / 256 * 256 == n && q8kBytes(n) == n / 256 * 292);
  for (nat v = 0; v < 600; ++v) CHECK(align16(v) >= v && align16(v) % 16 == 0 && align16(v) < v + 16);
  std::vector<nat> parts = {1, 17, 16, 3, 400, 0, 15};
  nat sum = 0;
  for (nat p : parts) sum += p;
  CHECK(carveTotal(parts) >= sum);
  Cursor cur;
  nat prevEnd = 0;
  for (nat p : parts) {
    nat off = cur.take(1, p);
    CHECK(off % 16 == 0 && off >= prevEnd && cur.used >= off + p && cur.used % 16 == 0);
    prevEnd = cur.used;
  }
  CHECK(cur.used == carveTotal(parts));
  for (nat used = 0; used < 40; used += 16)
    for (nat c = 0; c < 5; ++c)
      for (nat sz = 0; sz < 20; ++sz)
        CHECK(used + c * sz <= takeUsed(used, c, sz) && takeUsed(used, c, sz) % 16 == 0);
  for (nat a = 0; a < 300; ++a)
    for (nat b = 1; b <= 70; ++b) { CHECK(a <= ceilDivMul(a, b) && ceilDivMul(a, b) % b == 0 && ceilDivMul(a, b) < a + b); }
  for (nat mc = 1; mc <= 12; ++mc)
    for (nat row = 0; row < mc; ++row)
      for (nat h = 1; h <= 6; ++h)
        for (nat j = 0; j < h; ++j) CHECK(row * h + j < mc * h);
  CHECK(ropeRowOk(3, 4, 8) && !ropeRowOk(4, 4, 8));
  for (nat i = 0; i < 5; ++i) {
    CHECK(runStage(0, 0, i));
    CHECK(runStage(5, 0, i));
    for (nat k = 1; k <= 7; ++k) {
      CHECK(runStage(k, 0, i) == (k >= i + 1));
      for (nat j = i; j < 5; ++j) if (runStage(k, 0, j)) CHECK(runStage(k, 0, i));  // prefix closed
    }
    if (i < 3) CHECK(!runStage(0, 2, i) && runStage(0, 1, i));
    if (i >= 3) CHECK(!runStage(0, 1, i) && runStage(0, 2, i));
  }
  for (nat nl = 1; nl < 50; ++nl)
    for (nat l = 0; l < nl; ++l) CHECK(dumpSlotEnd(l, 24) <= dumpBufferBytes(nl, 24));
}

static void test_mtp_draft_rows() {
  using namespace baloon::mtp_draft_rows;
  for (nat used = 0; used <= 1024; used += 256)
    for (nat n = 0; n < 30; ++n)
      for (nat sz = 0; sz < 40; ++sz) {
        CHECK(used + n * sz <= bumpUsed(used, n, sz));
        CHECK(bumpUsed(used, n, sz) % 256 == 0);
      }
  for (nat w = 0; w < 1000; ++w) CHECK(cap64(w) >= w && cap64(w) % 64 == 0 && cap64(w) < w + 64);
  for (nat maxT = 1; maxT <= 40; ++maxT) {
    CHECK(fullRow(maxT) < stagedRows(maxT));
    std::set<nat> rows;
    for (nat t = 0; t < maxT; ++t) rows.insert(t);                 // catch-up rows [0, maxT)
    CHECK(rows.insert(fullRow(maxT)).second);                      // full row disjoint from catch-up
    for (nat j = 1; j + 2 <= maxT; ++j) {
      CHECK(chainRow(maxT, j) < stagedRows(maxT) && chainRow(maxT, j) >= maxT);
      CHECK(chainRow(maxT, j) != fullRow(maxT));
      CHECK(rows.insert(chainRow(maxT, j)).second);                // distinct chain rows
    }
    for (nat T = 0; T <= maxT; ++T) for (nat t = 0; t < T; ++t) CHECK(t < fullRow(maxT));
  }
  for (nat maxT = 1; maxT <= 9; ++maxT)
    for (nat n = 0; n <= 40; ++n) {
      for (nat c = 0; c < n; ++c) CHECK(chunkT(maxT, n, c) >= 1);
      for (nat c = 0; c <= n; ++c) CHECK(chunkT(maxT, n, c) <= maxT && c + chunkT(maxT, n, c) <= baloon::nmax(c, n));
      nat k = prefillChunks(maxT, n);
      CHECK(k == (n + maxT - 1) / maxT);  // terminates, covers the prompt in ceil(n/maxT) chunks
    }
  for (nat size = 0; size < 64; size += 4) CHECK(size / 4 * 4 == size);
}

static void test_session_loop_indexing() {
  using namespace baloon::session_loop_indexing;
  for (nat l = 0; l < 200; ++l) {
    CHECK(gdnIdx(l) + qsaIdx(l) == l);
    CHECK(qsaIdx(l) == l / 4 && qsaIdx(l) == qsaIdxClosed(l));
    CHECK(gdnIdx(l) == l - l / 4 && gdnIdx(l) == gdnIdxClosed(l));
    if (l < 48) {
      if (!isQsa(l)) CHECK(gdnIdx(l) < 36);
      CHECK(qsaIdx(l) < 12);
    }
    for (nat l2 = l + 1; l2 < 200; ++l2)
      if (!isQsa(l) && !isQsa(l2)) CHECK(gdnIdx(l) != gdnIdx(l2));
    CHECK(want(l) == l + 1);
  }
  for (nat a = 1; a <= 70; ++a)
    for (nat n = 0; n < 300; ++n) CHECK(alignUp(n, a) >= n && alignUp(n, a) % a == 0 && alignUp(n, a) < n + a);
  std::int64_t cum[5] = {7, 19, 20, 55, 81};
  std::int64_t s = 0;
  for (int i = 0; i < 5; ++i) s += stageTime(cum, i);
  CHECK(s == cum[4]);
}

static void test_verify_window_groups() {
  using namespace baloon::verify_window_groups;
  for (nat T = 0; T <= 200; ++T) {
    CHECK(tbF(T, 0) == 0 && teF(1, T, 0) == T);                       // one group covers all
    CHECK(tbF(T, 0) == 0 && teF(2, T, 0) == tbF(T, 1) && teF(2, T, 1) == T);  // two groups tile
    if (T >= 2) {
      CHECK(teF(2, T, 0) - tbF(T, 0) >= 1 && teF(2, T, 1) - tbF(T, 1) >= 1);
      CHECK((teF(2, T, 0) - tbF(T, 0)) + (teF(2, T, 1) - tbF(T, 1)) == T);
    }
    CHECK(teF(2, T, 1) - tbF(T, 1) <= teF(2, T, 0) - tbF(T, 0));       // A >= B
  }
  for (nat G = 1; G <= 2; ++G) {
    std::set<nat> rings;
    for (nat l = 0; l < 48; ++l)
      for (nat grp = 0; grp < G; ++grp) {
        nat k = ringK(G, l, grp);
        CHECK(k >= 1 && k <= 48 * G);
        CHECK(ringLayer(G, k) == l && ringGroup(G, k) == grp);
        CHECK((k - 1) / G == l && (k - 1) % G == grp);
        CHECK(rings.insert(k).second);                               // distinct per (layer, group)
      }
    CHECK(rings.size() == 48 * G);
    for (nat K = 0; K < 100; ++K)
      for (nat grp = 0; grp < G; ++grp) CHECK(grp * perGroup(G, K) + perGroup(G, K) <= K);
  }
  for (nat cap = 0; cap < 200; ++cap) {
    CHECK(planI32Count(cap) <= planPtrOff(cap) && planPtrOff(cap) % 2 == 0);
    CHECK(planPtrOff(cap) <= planI32Count(cap) + 1);
  }
  for (nat c = 0; c < 20; ++c)
    for (nat a = 0; a < 20; ++a) {
      CHECK(raiseFlag(c, a) >= c && raiseFlag(c, a) >= a && raiseFlag(c, a) == baloon::nmax(c, a));
      for (nat b = 0; b < 20; ++b) CHECK(raiseFlag(raiseFlag(c, a), b) == raiseFlag(raiseFlag(c, b), a));
    }
  for (nat n = 0; n < 100; ++n) for (nat sz = 1; sz < 9; ++sz) CHECK(flagsetIndex(n, sz) < sz);
}

static void test_weights_loader_segments() {
  using namespace baloon::weights_loader_segments;
  for (nat codes = 0; codes < 20; codes += 3)
    for (nat scales = 0; scales < 40; ++scales)
      for (nat offs = 0; offs < 10; offs += 4) {
        if (scales % 2 == 0) CHECK(dstTotal(codes, scales, offs) == srcTotal(codes, scales, offs, true) + scales / 2);
        CHECK(dstTotal(codes, scales, offs) == srcTotal(codes, scales, offs, false));
        CHECK((codes + srcScale(scales, true)) + offs == srcTotal(codes, scales, offs, true));
        CHECK((codes + srcScale(scales, false)) + offs == srcTotal(codes, scales, offs, false));
      }
  CHECK(CHUNK == 8u * 1024 * 1024 && CHUNK % 2 == 0 && STAGE_OUT == 2 * CHUNK);
  const nat rems[] = {0, 1, 2, 3, 100, 101, CHUNK - 1, CHUNK, CHUNK + 1, CHUNK + 2, 3 * CHUNK + 7, 1ull << 40};
  for (nat rem : rems) {
    CHECK(widenN(rem) % 2 == 0 && widenN(rem) <= rem && widenN(rem) <= CHUNK);
    if (rem > 0 && rem % 2 == 0) CHECK(widenN(rem) > 0);
    CHECK(widenOutBytes(widenN(rem)) <= STAGE_OUT);
  }
  // Full widening loop covers the plane exactly with contiguous output offsets.
  for (nat total : {nat(0), nat(2), nat(10), CHUNK, CHUNK + 2, 3 * CHUNK + 8}) {
    nat done = 0, outOff = 0;
    while (done < total) {
      nat n = widenN(total - done);
      CHECK(n > 0);
      CHECK(done * 2 + widenOutBytes(n) == (done + n) * 2);
      CHECK(outOff == done * 2);
      outOff += widenOutBytes(n);
      done += n;
    }
    CHECK(done == total && outOff == total * 2);
  }
  for (nat done = 0; done < 64; done += 4)
    for (nat n = 0; n < 64; n += 4) CHECK(done / 2 + bf16OutBytes(n) == (done + n) / 2);
  for (nat e = 0; e < 100; ++e) CHECK(2 * e * 2 == 4 * e);
  for (nat a = 1; a <= 64; a *= 2)
    for (nat at = 0; at < 4 * a; at += a)
      for (nat dst = 0; dst < 300; ++dst) {
        CHECK(at + dst <= placeAt(at, dst, a));
        CHECK(placeAt(at, dst, a) % a == 0);
      }
  CHECK(engineKind(4) == 1 && engineKind(5) == 3);
  for (nat k : {0, 1, 2, 3, 6, 7, 8}) CHECK(engineKind(k) == k);
}

int main() {
  test_expert_vram_cache();
  test_expert_blob_source();
  test_expert_dispatch_split();
  test_cpu_expert_pool();
  test_f16_bits();
  test_gdn_qsa_layer_layout();
  test_mtp_draft_rows();
  test_session_loop_indexing();
  test_verify_window_groups();
  test_weights_loader_segments();
  std::printf("u1: %ld checks, %ld failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
