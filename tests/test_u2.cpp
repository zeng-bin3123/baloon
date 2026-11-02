// Unit 2 invariant tests: each CHECK mirrors a LAYER 1 statement of the unit theory.
// Exit code is non-zero iff any check fails.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>
#include "bl_expert_pool_protocol.h"
#include "bl_fused_gated_residual_read.h"
#include "bl_fused_gdn_decode_step.h"
#include "bl_gated_residual_read_write.h"
#include "bl_gdn_reference_kernels.h"
#include "bl_iq_quant_mat_vec.h"
#include "bl_iquant_avx512_rows.h"
#include "bl_q2_avx2_fallback.h"
#include "bl_q2_expert_vnni_kernel.h"

using baloon::nat;
static long g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail <= 40) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static float half_to_float(std::uint16_t h) {
  std::uint32_t b = baloon::f16_bits::f32BitsFromF16(h); float f; std::memcpy(&f, &b, 4); return f;
}
static std::uint16_t float_to_half(float f) {
  std::uint32_t b; std::memcpy(&b, &f, 4); return baloon::f16_bits::f16FromF32Bits(b);
}

static void test_q2_kernel() {
  using namespace baloon::q2_expert;
  // geometry / unpack
  for (nat w = 0; w < 65536; w += 37)
    for (nat j = 0; j < 8; ++j) CHECK(code2(w, j) <= 3 && (((w >> (2 * j)) & 0xFF) & 3) == code2(w, j));
  for (nat l = 0; l < 4; ++l)
    for (nat i = 0; i < 8; ++i) { CHECK((8 * l + i) / 4 == 2 * l + i / 4); CHECK((8 * l + i) % 4 == i % 4); }
  // tail masks / batches
  for (nat nb = 0; nb < 20; ++nb) CHECK(tailMask(nb) <= 0xFFFF);
  for (nat nb = 8; nb < 20; ++nb) CHECK(tailMask(nb) == 0xFFFF);
  for (nat nb = 1; nb < 8; ++nb) CHECK(tailMask(nb) == (1u << (2 * nb)) - 1);
  for (nat nblocks = 1; nblocks < 40; ++nblocks) {
    nat covered = 0;
    for (nat b0 = 0; b0 < nblocks; b0 += 8) {
      nat bs = batchSize(nblocks, b0);
      CHECK(bs >= 1 && bs <= 8 && b0 + bs <= nblocks);
      covered += bs;
    }
    CHECK(covered == nblocks);
  }
  // quantiser rule
  for (std::int64_t u = -600; u <= 600; ++u) {
    CHECK(roundAway(-u) == -roundAway(u));
    std::int64_t q = clamp127(roundAway(u));
    CHECK(q >= -127 && q <= 127 && clamp127(q) == q);
  }
  CHECK(round_half_away(0.5f) == 1 && round_half_away(-0.5f) == -1 && round_half_away(2.5f) == 3 &&
        round_half_away(-2.5f) == -3 && round_half_away(1.4f) == 1);
  // tile dispatch + row layout
  for (nat n = 0; n < 30; ++n) CHECK(ntSel(n) <= 8);
  for (nat n = 1; n <= 8; ++n) CHECK(ntSel(n) == n);
  for (nat n = 0; n < 20; ++n) {
    CHECK(!useMulti(true, n, 16));
    CHECK(useMulti(false, n, 16) == (n >= 1 && n <= 16));
  }
  for (nat r = 0; r < 50; ++r) { CHECK(gateRow(r) != upRow(r)); CHECK(upRow(r) < 2 * 50); for (nat r2 = 0; r2 < 50; ++r2) CHECK(gateRow(r) != upRow(r2)); }

  // numerics: row dot vs direct dequantised dot (offset identity), several lengths
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> U(-2.f, 2.f);
  for (nat nblocks : {1u, 2u, 3u, 8u, 9u, 17u}) {
    nat n = nblocks * QK;
    std::vector<std::uint8_t> row(nblocks * BLOCK_STRIDE);
    std::vector<float> dw(nblocks);
    for (nat b = 0; b < nblocks; ++b) {
      float s = 0.01f + 0.5f * std::fabs(U(rng));
      std::uint16_t h = float_to_half(s);
      dw[b] = half_to_float(h);
      row[b * BLOCK_STRIDE] = h & 0xFF; row[b * BLOCK_STRIDE + 1] = h >> 8;
      for (int i = 0; i < 16; ++i) row[b * BLOCK_STRIDE + 2 + i] = std::uint8_t(rng());
    }
    std::vector<float> x(n);
    for (auto& v : x) v = U(rng);
    QAct a = act_quant_q8_1(x.data(), n);
    for (nat i = 0; i < n; ++i) CHECK(a.q[i] >= -127 && a.q[i] <= 127);
    for (nat c = 0; c < n / QKA; ++c) { std::int32_t s = 0; for (nat j = 0; j < QKA; ++j) s += a.q[c * QKA + j]; CHECK(s == a.s[c]); CHECK(std::abs(s) <= 127 * 32); }
    double ref = 0;
    for (nat b = 0; b < nblocks; ++b)
      for (nat j = 0; j < QK; ++j) {
        int c = int(codeAt(&row[b * BLOCK_STRIDE + 2], j));
        ref += double(dw[b]) * double(c - 1) * double(a.q[b * QK + j]) * double(a.d[(b * QK + j) / QKA]);
      }
    float got = row_dot_z(row.data(), nblocks, a);
    CHECK(std::fabs(double(got) - ref) <= 1e-4 * (1.0 + std::fabs(ref)));
    // quantisation error of the activations is within half a step
    for (nat i = 0; i < n; ++i) CHECK(std::fabs(a.q[i] * a.d[i / QKA] - x[i]) <= 0.5f * a.d[i / QKA] * 1.001f + 1e-6f);

    // AVX2 fallback agrees with the VNNI path (same arithmetic), multi-token with tiles of 4
    for (nat nt : {1u, 3u, 4u, 5u, 9u}) {
      std::vector<baloon::q2_avx2::QAct> acts;
      for (nat t = 0; t < nt; ++t) {
        std::vector<float> xt(n);
        for (auto& v : xt) v = U(rng);
        acts.push_back(baloon::q2_avx2::act_quant_q8_1_avx2(xt.data(), n));
      }
      std::vector<float> out(nt);
      baloon::q2_avx2::q2_0_gguf_rows_multi_avx2(row.data(), 1, nblocks, acts, out.data());
      for (nat t = 0; t < nt; ++t) CHECK(std::fabs(out[t] - row_dot_z(row.data(), nblocks, acts[t])) <= 1e-5f * (1.f + std::fabs(out[t])));
    }
  }
  // all-zero activation gives exactly zero scale and zero codes
  { std::vector<float> z(64, 0.f); QAct a = act_quant_q8_1(z.data(), 64); for (auto q : a.q) CHECK(q == 0); CHECK(a.d[0] == 0.f); }

  // full expert: with identity-like structure, output is finite and the single/oracle path equals the tiled one
  {
    nat H = 128, FF = 128;
    std::vector<std::uint8_t> gu(2 * FF * (H / QK) * BLOCK_STRIDE), dn(H * (FF / QK) * BLOCK_STRIDE);
    auto fill = [&](std::vector<std::uint8_t>& v) {
      for (nat b = 0; b < v.size() / BLOCK_STRIDE; ++b) {
        std::uint16_t h = float_to_half(0.05f);
        v[b * BLOCK_STRIDE] = h & 0xFF; v[b * BLOCK_STRIDE + 1] = h >> 8;
        for (int i = 0; i < 16; ++i) v[b * BLOCK_STRIDE + 2 + i] = std::uint8_t(rng());
      }
    };
    fill(gu); fill(dn);
    std::vector<float> x(H), y(H);
    for (auto& v : x) v = U(rng);
    expert_forward(gu.data(), dn.data(), H, FF, x.data(), y.data());
    bool fin = true; for (float v : y) fin = fin && std::isfinite(v);
    CHECK(fin);
    // independent float reference of the same expert (dequantised weights, un-requantised h): close but not equal
    double num = 0, den = 0;
    for (nat o = 0; o < H; ++o) {
      double acc = 0;
      for (nat r = 0; r < FF; ++r) {
        double g = 0, u = 0;
        for (nat j = 0; j < H; ++j) {
          auto w = [&](nat row, nat jj) {
            const std::uint8_t* blk = &gu[(row * (H / QK) + jj / QK) * BLOCK_STRIDE];
            return double(half_to_float(std::uint16_t(blk[0] | (blk[1] << 8)))) * (double(codeAt(blk + 2, jj % QK)) - 1.0);
          };
          g += w(2 * r, j) * x[j]; u += w(2 * r + 1, j) * x[j];
        }
        double hh = g / (1.0 + std::exp(-g)) * u;
        const std::uint8_t* blk = &dn[(o * (FF / QK) + r / QK) * BLOCK_STRIDE];
        acc += double(half_to_float(std::uint16_t(blk[0] | (blk[1] << 8)))) * (double(codeAt(blk + 2, r % QK)) - 1.0) * hh;
      }
      num += (acc - y[o]) * (acc - y[o]); den += acc * acc;
    }
    CHECK(std::sqrt(num / (den + 1e-30)) < 0.1);  // int8 activation quantisation twice: few-percent error
  }
}

static void test_avx2_unpack() {
  using namespace baloon::q2_avx2;
  std::mt19937 rng(7);
  for (int it = 0; it < 2000; ++it) {
    std::uint8_t b[16], out[64];
    for (auto& v : b) v = std::uint8_t(rng());
    unpack64(b, out);
    for (nat p = 0; p < 64; ++p) {
      CHECK(out[p] == ((b[p / 4] >> (2 * (p % 4))) & 3));
      CHECK(out[p] <= 3);
      CHECK(out[p] == baloon::q2_expert::codeAt(b, p));
    }
  }
  for (nat p = 0; p < 32; ++p) CHECK(p / 4 < 8);
  for (nat p = 32; p < 64; ++p) CHECK(p / 4 >= 8 && p / 4 < 16);
  // int8 store lossless on the clamped range; chunk sums fit in int32
  for (long v = -127; v <= 127; ++v) CHECK(std::int8_t(v) == v);
  CHECK(127L * 32 < 2147483647L);
  for (nat nt = 1; nt < 30; ++nt) {
    nat t0 = 0, steps = 0;
    while (t0 < nt) { nat s = tileSize(nt, t0); CHECK(s >= 1 && s <= 4 && t0 + s <= nt); t0 += s; ++steps; }
    CHECK(t0 == nt);
  }
}

static void test_iquant() {
  using namespace baloon::iquant_rows;
  for (nat ty : {16u, 17u, 18u, 21u, 22u}) CHECK(supported(ty) && blockBytes(ty) > 0);
  CHECK(!supported(29) && !supported(20) && !supported(0));
  CHECK(2 + 64 == blockBytes(16)); CHECK(2 + 64 + 8 == blockBytes(17)); CHECK(2 + 64 + 32 == blockBytes(18));
  CHECK(2 + 64 + 8 + 32 + 4 == blockBytes(21)); CHECK(2 + 64 + 8 + 8 == blockBytes(22));
  for (nat j = 0; j < 4; ++j) {
    CHECK(xxsQs(j).hi <= blockBytes(16));
    CHECK(xsScale(j).hi <= blockBytes(17));
    CHECK(iq2sSign(j).hi <= 66 && iq2sQs(j).hi <= 34 && iq2sScale(j).hi <= blockBytes(22));
    CHECK(iq3xxsWords(j).hi <= blockBytes(18));
    CHECK(iq3sQh(j).hi <= 74 && iq3sSign(j).hi <= 106 && iq3sScale(j).hi <= blockBytes(21));
  }
  std::mt19937 rng(99);
  for (int it = 0; it < 20000; ++it) {
    nat a = rng(); nat k = rng() % 4;
    CHECK(signIndex(a, k) < 128);
    CHECK(topNibble(a) < 16);
    nat v = rng() & 0xFFFF;
    CHECK(iq2xsGridIndex(v) < 512 && iq2xsSignIndex(v) < 128);
    nat q = rng() & 0xFF, h = rng() & 0xFFFF;
    CHECK(iq2sIndex(q, h) < 1024);
    CHECK(iq3sIndex(q, rng(), rng() % 8) < 512);
    nat s = rng() % 16;
    CHECK(scaleFromNibble(s) % 2 == 1 && scaleFromNibble(s) <= 31);
    long y = long(rng() % 2001) - 1000;
    for (bool m : {false, true}) { CHECK(applySign(m, applySign(m, y)) == y); CHECK(std::labs(applySign(m, y)) == std::labs(y)); }
  }
  for (nat nt = 1; nt < 40; ++nt) {
    CHECK(guTile(nt) <= 8);
    nat t0 = 0;
    while (t0 < nt) { nat s = tileSize(nt, t0); CHECK(s >= 1 && s <= 4 && t0 + s <= nt); t0 += s; }
    CHECK(t0 == nt);
  }
  for (nat n = 256; n <= 4096; n += 256) CHECK(blocksPerRow(n) * 256 == n);
}

static void test_pool() {
  using namespace baloon::expert_pool;
  CHECK(skipFirst(true, {0, 1, 2}).size() == 2 && skipFirst(true, {0, 1, 2})[0] == 1);
  CHECK(skipFirst(false, {0, 1, 2}).size() == 3);
  CHECK(skipFirst(true, {}).empty());
  for (nat rows = 0; rows <= 80; ++rows)
    for (nat parts = 1; parts <= 24; ++parts) {
      CHECK(lo(rows, parts, 0) == 0 && lo(rows, parts, parts) == rows);
      for (nat i = 0; i < parts; ++i) CHECK(lo(rows, parts, i) <= lo(rows, parts, i + 1) && lo(rows, parts, i + 1) <= rows);
    }
  // per-task walk across experts: progress, stay in expert, stay within task, covers the task exactly
  for (nat per : {1u, 3u, 7u, 16u})
    for (nat n = 1; n <= 5; ++n)
      for (nat parts = 1; parts <= 9; ++parts) {
        nat total = n * per; std::vector<int> seen(total, 0);
        for (nat t = 0; t < parts; ++t) {
          nat g0 = lo(total, parts, t), g1 = lo(total, parts, t + 1);
          nat r = g0;
          while (r < g1) {
            nat r0 = r % per;
            nat run = walkRun(per, r0, g1, r);
            CHECK(run >= 1 && r0 + run <= per && r + run <= g1);
            CHECK((r / per) * per + r0 == r);
            for (nat k = 0; k < run; ++k) seen[r + k]++;
            r += run;
          }
        }
        for (int s : seen) CHECK(s == 1);
      }
  for (nat th = 0; th < 40; ++th) for (nat n = 1; n < 20; ++n) CHECK(partsA(th, n) >= 1);
  CHECK(perOf(11, 7, 3) == 11 && perOf(11, 7, 5) == 11 && perOf(11, 7, 4) == 7 && perOf(11, 7, 6) == 7);
  for (nat kMax = 1; kMax < 6; ++kMax) for (nat n = 0; n < 20; ++n) {
    CHECK(useRun(n, kMax, 1, false) && useRun(n, kMax, 4, true) && useRun(kMax + 1, kMax, 4, false));
    nat b0 = 0;
    while (b0 < n) { nat bl = batchLen(kMax, n, b0); CHECK(bl >= 1 && bl <= kMax && b0 + bl <= n); b0 += bl; }
    CHECK(b0 == n || n == 0);
  }
  PState s{5, 0};
  CHECK(arrive(leave(s)).parked == 5 && arrive(leave(s)).draining == 0);
  CHECK(leave(s).parked + leave(s).draining == 5);

  // live pool: every job executes exactly once, repeatedly; accounting invariant holds; constructed all parked
  for (nat nw : {0u, 1u, 3u}) {
    ExpertPool pool(nw);
    CHECK(pool.parkedNow() == nw);
    for (int round = 0; round < 40; ++round) {
      nat njobs = 1 + round * 7 % 53;
      std::vector<std::atomic<int>> hits(njobs);
      for (auto& h : hits) h = 0;
      pool.run(njobs, [&](nat i) { hits[i]++; });
      for (auto& h : hits) CHECK(h.load() == 1);
      CHECK(pool.parkedNow() == nw);
    }
    pool.run(0, [](nat) {});
    CHECK(pool.accountingHeld());
  }
}

static void test_geometry() {
  using namespace baloon;
  {
    using namespace fused_gdn_decode;
    std::set<nat> rows;
    for (nat rg = 0; rg < 4; ++rg) for (nat r = 0; r < 32; ++r) { CHECK(threadRow(rg, r) < 128); rows.insert(threadRow(rg, r)); }
    CHECK(rows.size() == 128);
    for (nat rg = 0; rg < 4; ++rg) for (nat c = 0; c < 128; ++c) { CHECK(tid(rg, c) < 512); CHECK((tid(rg, c) < 128) == (rg == 0)); CHECK(warpSlot(tid(rg, c)) < 16); }
    for (nat c = 0; c < 128; ++c) CHECK(warpSlot(c) < 4);
    for (nat h_v : {1u, 16u, 32u}) for (nat row = 0; row < 128; row += 5) for (nat head = 0; head < h_v; ++head) for (nat col = 0; col < 128; col += 9) {
      CHECK(stateIndex(row, head, col, h_v) < 128 * h_v * 128);
      CHECK(stateIndex(row + 1, head, col, h_v) == stateIndex(row, head, col, h_v) + h_v * 128);
      CHECK(outIndex(head, col) < h_v * 128);
    }
    for (nat h_k = 1; h_k < 9; ++h_k) for (nat head = 0; head < 40; ++head) CHECK(qkHead(head, h_k) < h_k);
    for (nat ch : {128u, 384u, 1024u}) for (nat c = 0; c < ch; ++c) {
      CHECK(convHistIndex(c, 2) < 3 * ch && convWeightIndex(c, 3) < 4 * ch);
      for (nat qk : {0u, 1u, 2u}) CHECK(l2Applies(c, qk) == (c < qk * 128));
    }
    for (nat h_v = 1; h_v < 70; ++h_v) {
      CHECK(abGridBlocks(h_v) * 8 >= 2 * h_v);
      for (nat row = 0; row < 2 * h_v; ++row) CHECK(abIsAlpha(row, h_v) ? row < h_v : (row >= h_v && abLocalRow(row, h_v) < h_v));
    }
    CHECK(stepNormArgsValid(2, 4) && !stepNormArgsValid(3, 4) && !stepNormArgsValid(0, 4) && !stepNormArgsValid(2, 0));
    CHECK(convArgsValid(256, 2) && !convArgsValid(256, 3) && !convArgsValid(200, 1));
    for (nat ch : {128u, 256u, 640u}) for (nat qk = 0; qk <= ch / 128; ++qk) CHECK(qk * 128 <= ch);
  }
  {
    using namespace fused_gr;
    CHECK(D == 10240 && WARPS == 8 && DOWN_BLOCKS == 40 && UP_BLOCKS == 80 && UPM_BLOCKS == 160 && TQ == 10);
    for (nat i = 0; i < D; ++i) { CHECK(streamOf(i) < 4 && colOf(i) < N && streamOf(i) * N + colOf(i) == i); if (i % 4 == 0) CHECK(streamOf(i + 3) == streamOf(i)); }
    for (nat b = 0; b < 40; ++b) for (nat w = 0; w < 8; ++w) CHECK(downRow(b, w) < 320);
    for (nat l = 0; l < 32; ++l) { for (nat j = 0; j < 40; ++j) CHECK(downChunk(l, j) < 1280); if (l < 8) CHECK(secondChunk(l) < 40); }
    for (nat base : {0u, 2560u, 5120u, 7680u}) for (nat l = 0; l < 32; ++l) for (nat q = 0; q < 10; ++q) CHECK(tiledChunk(base, l, q) < 1280);
    for (nat T : {1u, 5u, 32u}) for (nat i = 0; i < T * 640; ++i) CHECK(tileCopyToken(i) < T && tileCopyOffset(i) < 640 && tileCopyToken(i) * 640 + tileCopyOffset(i) == i);
    std::set<nat> up, upm;
    for (nat blk = 0; blk < 80; ++blk) for (nat r = 0; r < 128; ++r) { CHECK(upIndex(blk, r) < D); up.insert(upIndex(blk, r)); }
    for (nat blk = 0; blk < 160; ++blk) for (nat r = 0; r < 64; ++r) { CHECK(upmIndex(blk, r) < D); upm.insert(upmIndex(blk, r)); }
    CHECK(up.size() == D);
    CHECK(upm.size() == 160 * 64);
    for (nat blk = 0; blk < 80; ++blk) for (nat t = 0; t < 32; ++t) CHECK(mixedIndex(blk, t) < N);
    CHECK(grSupported(2560, 4, 320) && !grSupported(2048, 4, 320) && !grSupported(2560, 2, 320) && !grSupported(2560, 4, 256));
    CHECK(tokenLaneOwned(8, 16) && tokenLaneOwned(32, 32) && !tokenLaneOwned(8, 33) && !tokenLaneOwned(17, 16));
  }
  {
    using namespace gated_residual;
    for (nat n = 0; n < 500; ++n) { CHECK(alignUp16(n) >= n && alignUp16(n) < n + 16 && alignUp16(n) % 16 == 0); }
    for (nat n_embd : {16u, 64u, 100u, 2560u}) for (nat hc : {1u, 2u, 4u}) for (nat lr : {8u, 33u, 320u}) {
      Workspace w = workspace(n_embd, hc, lr);
      for (int i = 0; i < 4; ++i) CHECK(w.off[i] + w.size[i] <= w.off[i + 1]);  // regions disjoint
      CHECK(w.off[4] + w.size[4] == w.total);
      for (int i = 0; i < 5; ++i) CHECK(w.off[i] % 16 == 0);
      CHECK(w.size[1] >= n_embd * hc * 2);  // xq holds the full stream vector (historical mis-size)
    }
    Workspace r = workspace(2560, 4, 320);
    CHECK(r.size[0] == 40960 && r.size[1] == 20480 && r.size[2] == 640 && r.size[3] == 40960 && r.size[4] == 1280 && r.total == 104320);
    for (nat i = 0; i < 4096; ++i) {
      nat k = downK(i), w = downWarp(i), l = downLane(i);
      CHECK(w < 8 && l < 32 && downElement(k, w, l) == i);
    }
    for (nat k = 0; k < 3; ++k) for (nat w = 0; w < 8; ++w) for (nat l = 0; l < 32; ++l) { nat i = downElement(k, w, l); CHECK(downK(i) == k && downWarp(i) == w && downLane(i) == l); }
    for (nat hd : {1u, 7u, 8u, 9u, 10240u}) CHECK(gateGridBlocks(hd) * 8 >= hd);
    for (nat n : {1u, 255u, 256u, 257u, 2560u}) CHECK(meanGridBlocks(n) * 256 >= n);
    CHECK(injectBlockOk(32) && !injectBlockOk(33));
    CHECK(nativeEven(2560, 4) && !nativeEven(7, 4));
    {
      nat ne = 12, hc = 3;
      std::vector<float> R(ne * hc), bo(ne), w(hc), out(ne * hc);
      for (nat i = 0; i < R.size(); ++i) R[i] = float(i) * 0.25f;
      for (nat i = 0; i < ne; ++i) bo[i] = float(i) - 3.f;
      for (nat c = 0; c < hc; ++c) w[c] = 0.5f + float(c);
      gr_write_ref(R.data(), bo.data(), w.data(), ne, hc, out.data());
      for (nat c = 0; c < hc; ++c) for (nat d = 0; d < ne; ++d) CHECK(out[c * ne + d] == R[c * ne + d] + bo[d] * w[c]);
    }
  }
  {
    using namespace gdn_reference;
    for (nat S : {32u, 64u, 128u}) for (nat j = 0; j < S; ++j) CHECK(j / 32 < gridX(S) && (j / 32) * 32 + j % 32 == j);
    for (nat h_v = 1; h_v < 40; ++h_v) { CHECK(gridY(h_v) == h_v); for (nat by = 0; by < h_v; ++by) CHECK(headsInBlock(by, h_v) == 1); }
    for (nat S : {64u, 128u}) for (nat h_v : {1u, 4u, 32u}) for (nat i = 0; i < S; i += 7) for (nat h = 0; h < h_v; ++h) for (nat j = 0; j < S; j += 5) {
      CHECK(stateIndex(i, h, j, S, h_v) == stateViaColumn(i, h, j, S, h_v));
      CHECK(stateIndex(i, h, j, S, h_v) < S * h_v * S);
    }
    CHECK(stepSizeOk(128) && !stepSizeOk(129));
    for (nat dc : {2u, 4u}) for (nat c = 0; c < 50; ++c) for (nat i = 0; i < dc; ++i) {
      if (i < dc - 1) CHECK(convStateIndex(c, i, dc) < 50 * (dc - 1));
      CHECK(convWeightIndex(c, i, dc) < 50 * dc);
    }
    for (nat C : {1u, 255u, 256u, 257u}) CHECK(convGridBlocks(C) * 256 >= C);
    for (nat n : {1u, 127u, 128u, 129u}) CHECK(betaGridBlocks(n) * 128 >= n);
    CHECK(l2ColsOk(1024) && !l2ColsOk(1025));
    std::vector<long> st = {1, 2, 3};
    auto s3 = slideList(slideList(slideList(st, 10), 11), 12);
    CHECK(s3.size() == 3 && s3[0] == 10 && s3[1] == 11 && s3[2] == 12);
    CHECK(slideList(st, 9).size() == st.size());
    CHECK(slideList({1, 2, 3}, 4) == (std::vector<long>{2, 3, 4}));
  }
  {
    using namespace iq_quant;
    for (nat t : kIqTypes) CHECK(blockBytes(t) > 0 && blockValues(t) > 0);
    for (nat t : kMmvqTypes) { bool in = false; for (nat u : kIqTypes) in = in || u == t; CHECK(in); }
    for (nat t : kDownTypes) { bool in = false; for (nat u : kMmvqTypes) in = in || u == t; CHECK(in); }
    for (nat i = 0; i < 10; ++i) for (nat j = i + 1; j < 10; ++j) CHECK(kIqTypes[i] != kIqTypes[j]);
    CHECK(blockBytes(16) == 66 && blockBytes(29) == 56 && blockBytes(23) == 136 && blockBytes(99) == 0);
    CHECK(rowBytes(16, 512) == 132);
    for (nat ipb : {2u, 8u}) for (nat nb : {1u, 5u}) for (nat k = 0; k < nb * ipb; ++k) { CHECK(callBlock(k, ipb) < nb && callPart(k, ipb) < ipb && callBlock(k, ipb) * ipb + callPart(k, ipb) == k); }
    for (nat part = 0; part < 8; ++part) CHECK(2 * part <= 14 && part <= 7);
    for (nat part = 0; part < 2; ++part) CHECK(2 * part <= 2);
    for (nat m : {1u, 2u, 8u}) for (nat kbx = 0; kbx < 6; ++kbx) CHECK((kbx + 1) * m <= 6 * m);
    Layout L{64, 24, 100, 40};
    CHECK(L.upOff() + L.nFf * L.guRow == L.downOff());
    for (nat r = 0; r < L.nFf; ++r) { CHECK(r * L.guRow + L.guRow <= L.upOff()); CHECK(L.upOff() + r * L.guRow + L.guRow <= L.downOff()); }
    for (nat r = 0; r < L.nEmbd; ++r) CHECK(L.downOff() + r * L.dRow + L.dRow <= L.bytes());
    CHECK(L.bytes() == 2 * 24 * 100 + 64 * 40);
    for (nat n : {0u, 1u, 255u, 256u, 257u, 1000u}) { CHECK(alignUp256(n) >= n && alignUp256(n) % 256 == 0); CHECK(alignUp256(n) + alignUp256(n) + alignUp256(n) == 3 * alignUp256(n)); }
    for (nat n : {1u, 255u, 256u, 257u}) { CHECK(mmvqGridBlocks(n) * 4 >= n); CHECK(guGridBlocks(n) * 8 >= 2 * n); CHECK(downGridBlocks(n) * 8 >= n); CHECK(q8Grid(n) * 256 >= n); }
    for (nat c = 0; c < 3; ++c) for (nat r = 0; r < 7; ++r) CHECK(mmvqOut(c, r, 7) < 3 * 7);
    for (nat e = 0; e < 4; ++e) for (nat r = 0; r < 9; ++r) CHECK(entryOut(e, r, 9) < 4 * 9);
    std::set<nat> o8; for (nat t = 0; t < 32; ++t) for (nat j = 0; j < 8; ++j) { nat o = dq8Offset(t, j); CHECK(o < 256); o8.insert(o); }
    CHECK(o8.size() == 256);
    std::set<nat> o4; for (nat ib = 0; ib < 8; ++ib) for (nat il = 0; il < 4; ++il) for (nat j = 0; j < 4; ++j) for (nat h = 0; h < 2; ++h) { nat o = dq4Offset(ib, il, j, h); CHECK(o < 256); o4.insert(o); }
    CHECK(o4.size() == 256);
    std::set<nat> oq; for (nat b = 0; b < 4; ++b) for (nat p = 0; p < 8; ++p) for (nat j = 0; j < 8; ++j) { nat o = dqQ2Offset(b, p, j); CHECK(o < 256); oq.insert(o); }
    CHECK(oq.size() == 256);
    for (nat t = 0; t < 3; ++t) for (nat b = 0; b < 2; ++b) for (nat o = 0; o < 256; o += 17) CHECK(embedOut(t, b, o, 512) < 3 * 512);
    for (nat r = 0; r < 5; ++r) for (nat p = 0; p < 2; ++p) for (nat c = 0; c < 6; ++c) CHECK(guInterleave(r, p, c, 6) < 2 * 5 * 6);
    // q8_1 quantiser
    float z[32] = {0};
    Q81 bz = quantize_q8_1(z); for (int i = 0; i < 32; ++i) CHECK(bz.q[i] == 0);
    float x[32]; std::mt19937 rng(5); std::uniform_real_distribution<float> U(-3.f, 3.f);
    for (int it = 0; it < 200; ++it) {
      float amax = 0; for (auto& v : x) { v = U(rng); amax = std::fmax(amax, std::fabs(v)); }
      Q81 b = quantize_q8_1(x);
      CHECK(std::fabs(b.d - amax / 127.f) <= 1e-7f);
      bool hit127 = false;
      for (int i = 0; i < 32; ++i) { CHECK(b.q[i] >= -127 && b.q[i] <= 127); CHECK(std::fabs(b.q[i] * b.d - x[i]) <= 0.5f * b.d * 1.001f + 1e-7f); hit127 = hit127 || std::abs(b.q[i]) == 127; }
      CHECK(hit127);
    }
  }
}

int main() {
  test_q2_kernel();
  test_avx2_unpack();
  test_iquant();
  test_pool();
  test_geometry();
  std::printf("unit 2: %ld checks, %ld failures\n", g_checks, g_fail);
  if (g_checks < 1000) { std::printf("too few checks executed\n"); return 2; }
  return g_fail ? 1 : 0;
}
