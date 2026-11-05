// Unit 20 invariants: expert blob layout, format table, activation packing, Q2_0 / i-quant row dots through the
// format-generic entry points (checked against independent scalar models, not against the code under test).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "baloon/cpu_expert_layout.hpp"
#include "baloon/cpu_native_expert.hpp"
using namespace baloon;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static uint32_t rng = 987654321u;
static uint32_t rnd32() { rng = rng * 1664525u + 1013904223u; return rng; }
static float rndf() { return (float)((rnd32() >> 8) & 0xFFFF) / 32768.f - 1.f; }
static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol * (1.f + std::fabs(b)); }
static float h2f(uint16_t h) {   // independent fp16 decode (normal numbers only are used here)
    const int e = (h >> 10) & 31, m = h & 1023; const float s = (h >> 15) ? -1.f : 1.f;
    if (e == 0) return s * std::ldexp((float)m, -24);
    return s * std::ldexp((float)(m + 1024), e - 25);
}

// independent model of one Q2_0 row against one quantized token (chunk c of 32 values uses scale xd[c])
static float q2_row_model(const uint8_t* row, uint32_t nb, const int8_t* q, const float* xd) {
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = row + 18 * b;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        float s = 0.f;
        for (uint32_t i = 0; i < 64; ++i) {
            const int code = (blk[2 + i / 4] >> (2 * (i % 4))) & 3;
            s += (float)((code - 1) * q[b * 64 + i]) * xd[2 * b + i / 32];
        }
        acc += h2f(dh) * s;
    }
    return acc;
}
static void fill_q2(std::vector<uint8_t>& w, uint64_t blocks) {
    for (auto& v : w) v = (uint8_t)rnd32();
    for (uint64_t b = 0; b < blocks; ++b) { uint16_t h = (uint16_t)(0x2800 + (rnd32() & 0x3FF)); std::memcpy(&w[b * 18], &h, 2); }
}

int main() {
    using namespace expert_layout_ns;
    // ---- block table (type -> values / bytes)
    struct { int t; int64_t v, b; } tab[] = {{42, 64, 18}, {16, 256, 66}, {17, 256, 74}, {18, 256, 98}, {21, 256, 110}, {22, 256, 82}};
    for (auto& e : tab) { CHECK(block_values(e.t) == e.v && block_bytes(e.t) == e.b); CHECK(native_fmt(e.t) != NativeFmt::None); CHECK(native_type_id(native_fmt(e.t)) == e.t); }
    for (int t : {0, 1, 2, 7, 8, 19, 20, 23, 41, 43, 100}) { CHECK(block_values(t) == 0 && block_bytes(t) == 0 && native_fmt(t) == NativeFmt::None); }
    CHECK(native_fmt(42) == NativeFmt::Q2_0 && native_fmt(16) == NativeFmt::IQ2_XXS && native_fmt(17) == NativeFmt::IQ2_XS);
    CHECK(native_fmt(18) == NativeFmt::IQ3_XXS && native_fmt(21) == NativeFmt::IQ3_S && native_fmt(22) == NativeFmt::IQ2_S);
    CHECK(block_bytes(16) == 2 + 64 && block_bytes(17) == 2 + 64 + 8 && block_bytes(18) == 2 + 96 && block_bytes(22) == 2 + 64 + 8 + 8 && block_bytes(21) == 2 + 64 + 8 + 32 + 4);
    CHECK(row_bytes(42, 2560) == 720 && row_bytes(42, 640) == 180 && row_bytes(42, 100) == 0 && row_bytes(42, 0) == 0 && row_bytes(42, -64) == 0);
    CHECK(row_bytes(16, 2560) == 660 && row_bytes(16, 640) == 0 && row_bytes(16, 512) == 132);

    // ---- layout arithmetic
    { ExpertLayout l = expert_layout(42, 2560, 640);
      CHECK(l.ok && l.gu_row_bytes == 720 && l.down_row_bytes == 180 && l.up_off == 640 * 720 && l.down_off == 2 * 640 * 720 && l.bytes == 1382400);
      CHECK(l.down_off == l.up_off * 2 && l.bytes == l.down_off + 2560 * l.down_row_bytes); }
    { ExpertLayout l = expert_layout(16, 2560, 512);
      CHECK(l.ok && l.gu_row_bytes == 660 && l.down_row_bytes == 132 && l.up_off == 512 * 660 && l.down_off == 1024 * 660 && l.bytes == 2 * 512 * 660 + 2560 * 132); }
    CHECK(!expert_layout(16, 2560, 640).ok && !expert_layout(7, 2560, 640).ok && !expert_layout(42, 0, 64).ok && !expert_layout(42, 64, 0).ok && !expert_layout(42, 100, 64).ok);
    { ExpertLayout bad = expert_layout(16, 2560, 640); CHECK(bad.bytes == 0 && bad.up_off == 0 && bad.gu_row_bytes == 0); }
    { ExpertLayout o; CHECK(expert_layout_load(o, 42, 2560, 640, 1382400) && o.ok && o.bytes == 1382400);
      CHECK(!expert_layout_load(o, 42, 2560, 640, 1382401) && !o.ok && o.bytes == 0);
      o = expert_layout(42, 2560, 640);
      CHECK(!expert_layout_load(o, 42, 2560, 640, 1382399) && !o.ok);
      CHECK(!expert_layout_load(o, 42, 2560, 640, 0) && !expert_layout_load(o, 9, 2560, 640, 1382400)); }

    // ---- activation byte sizes and capacities
    CHECK(native_act_bytes(NativeFmt::Q2_0, 2560, 1) == 3200 && kNativeActBytes == 3200);
    CHECK(native_act_bytes(NativeFmt::IQ2_XXS, 2560, 1) == 10 * 292 && native_act_bytes(NativeFmt::IQ2_XXS, 2560, 1) <= kNativeActBytes);
    CHECK(native_act_bytes(NativeFmt::Q2_0, 640, 1) == 800 && native_act_bytes(NativeFmt::Q2_0, 640, 1) <= kNativeHBytes);
    CHECK(native_act_bytes(NativeFmt::Q2_0, 640, 3) == 3 * 800 && native_act_bytes(NativeFmt::IQ3_S, 512, 4) == 4 * 2 * 292);
    CHECK(native_act_bytes(NativeFmt::Q2_0, 40, 1) == 0 && native_act_bytes(NativeFmt::IQ3_S, 640, 1) == 0 && native_act_bytes(NativeFmt::None, 256, 1) == 0 && native_act_bytes(NativeFmt::Q2_0, 64, 0) == 0);
    { std::vector<float> x(64, 1.f); alignas(4) uint8_t buf[512];
      CHECK(!native_quant_act(NativeFmt::Q2_0, x.data(), 40, 1, buf) && !native_quant_act(NativeFmt::None, x.data(), 64, 1, buf) && !native_quant_act(NativeFmt::IQ2_S, x.data(), 64, 1, buf)); }

    // ---- activation quantizer: ties to even, clamp, scale = amax/127, chunk sum
    { float x[32]; for (int i = 0; i < 32; ++i) x[i] = 0.f;
      x[0] = 127.f; x[1] = 0.5f; x[2] = 1.5f; x[3] = 2.5f; x[4] = -0.5f; x[5] = -2.5f; x[6] = -127.f; x[7] = 3.5f; x[8] = 0.49f; x[9] = 0.51f;
      int8_t q[32]; float d; int32_t hx; act_quant_any(x, 32, q, &d, &hx);
      CHECK(d == 1.f && q[0] == 127 && q[1] == 0 && q[2] == 2 && q[3] == 2 && q[4] == 0 && q[5] == -2 && q[6] == -127 && q[7] == 4 && q[8] == 0 && q[9] == 1);
      int s = 0; for (int i = 0; i < 32; ++i) s += q[i]; CHECK(hx == s);
      int8_t q2[32]; float d2; int32_t h2; kern::act_quant_q8_1(x, 32, q2, &d2, &h2); CHECK(std::memcmp(q, q2, 32) == 0 && d2 == d && h2 == hx); }
    for (int rep = 0; rep < 50; ++rep) {
        const uint32_t n = 32 * (1 + rnd32() % 6); std::vector<float> x(n); const float sc = 0.01f + (rnd32() % 1000) * 0.05f;
        for (auto& v : x) v = rndf() * sc;
        std::vector<int8_t> q(n); std::vector<float> d(n / 32); std::vector<int32_t> hx(n / 32);
        act_quant_any(x.data(), n, q.data(), d.data(), hx.data());
        for (uint32_t c = 0; c < n / 32; ++c) {
            float amax = 0.f; int s = 0;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[c * 32 + i]));
            CHECK(near(d[c], amax / 127.f, 1e-6f));
            for (int i = 0; i < 32; ++i) { const int v = q[c * 32 + i]; s += v; CHECK(v >= -127 && v <= 127); CHECK(std::fabs(v * d[c] - x[c * 32 + i]) <= 0.5001f * d[c] + 1e-6f); }
            CHECK(hx[c] == s);
        }
    }

    const bool avx512 = cpu_avx512_ok();
    const bool any = avx512 || kern::cpu_features().avx2;
    std::printf("cpu: avx512=%d any=%d\n", (int)avx512, (int)any);

    // ---- Q2_0 rows against the independent model (all token counts 1..9, sub-ranges, nonzero row0)
    if (any) {
        const uint32_t nb = 8, n = nb * 64, rows = 7, nc = n / 32;
        std::vector<uint8_t> w((size_t)rows * nb * 18); fill_q2(w, (uint64_t)rows * nb);
        for (uint32_t nt = 1; nt <= 9; ++nt) {
            std::vector<float> x((size_t)nt * n); for (auto& v : x) v = rndf() * 3.f;
            std::vector<int8_t> q((size_t)nt * n); std::vector<float> d((size_t)nt * nc); std::vector<int32_t> hx((size_t)nt * nc);
            for (uint32_t t = 0; t < nt; ++t) act_quant_any(&x[(size_t)t * n], n, &q[(size_t)t * n], &d[(size_t)t * nc], &hx[(size_t)t * nc]);
            std::vector<float> out((size_t)nt * rows, -777.f);
            CHECK(q2_rows_any(w.data(), 2, 6, nb, nt, q.data(), n, d.data(), hx.data(), nc, out.data(), rows));
            for (uint32_t t = 0; t < nt; ++t)
                for (uint32_t r = 0; r < rows; ++r) {
                    const float got = out[(size_t)t * rows + r];
                    if (r < 2 || r >= 6) { CHECK(got == -777.f); continue; }
                    CHECK(near(got, q2_row_model(&w[(size_t)r * nb * 18], nb, &q[(size_t)t * n], &d[(size_t)t * nc]), 2e-3f));
                }
        }
    }

    // ---- Q2_0 expert pipeline through the format-generic entry points
    if (any) {
        const uint32_t E = 256, F = 128, nt = 3;
        const ExpertLayout l = expert_layout(42, E, F);
        CHECK(l.ok && native_experts_available(42, E, F));
        CHECK(!native_experts_available(42, E + 32, F) && !native_experts_available(42, E, 0) && !native_experts_available(9, E, F));
        std::vector<uint8_t> blob((size_t)l.bytes); fill_q2(blob, (uint64_t)l.bytes / 18);
        std::vector<float> x((size_t)nt * E); for (auto& v : x) v = rndf() * 2.f;
        std::vector<uint8_t> xq(native_act_bytes(NativeFmt::Q2_0, E, nt) + 4);
        uint8_t* xqa = (uint8_t*)(((uintptr_t)xq.data() + 3) / 4 * 4);
        CHECK(native_quant_act(NativeFmt::Q2_0, x.data(), E, nt, xqa));
        // packing: token t's chunk data equals act_quant_any of token t
        const int8_t* pq = (const int8_t*)xqa; const float* pd = (const float*)(xqa + (nt * E + 3) / 4 * 4); const int32_t* ph = (const int32_t*)(pd + nt * (E / 32));
        for (uint32_t t = 0; t < nt; ++t) {
            std::vector<int8_t> q(E); std::vector<float> d(E / 32); std::vector<int32_t> hx(E / 32);
            act_quant_any(&x[(size_t)t * E], E, q.data(), d.data(), hx.data());
            CHECK(std::memcmp(q.data(), pq + (size_t)t * E, E) == 0);
            for (uint32_t c = 0; c < E / 32; ++c) CHECK(d[c] == pd[t * (E / 32) + c] && hx[c] == ph[t * (E / 32) + c]);
        }
        std::vector<float> act((size_t)nt * F, -5.f);
        CHECK(native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F / 2, xqa, nt, act.data(), F));
        CHECK(native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, F / 2, F, xqa, nt, act.data(), F));
        for (uint32_t t = 0; t < nt; ++t)
            for (uint32_t r = 0; r < F; ++r) {
                const float g = q2_row_model(&blob[(size_t)r * l.gu_row_bytes], E / 64, pq + (size_t)t * E, pd + t * (E / 32));
                const float u = q2_row_model(&blob[(size_t)l.up_off + (size_t)r * l.gu_row_bytes], E / 64, pq + (size_t)t * E, pd + t * (E / 32));
                const float ref = g / (1.f + std::exp(-g)) * u;
                CHECK(near(act[(size_t)t * F + r], ref, 3e-3f));
            }
        std::vector<uint8_t> hq(native_act_bytes(NativeFmt::Q2_0, F, nt) + 4);
        uint8_t* hqa = (uint8_t*)(((uintptr_t)hq.data() + 3) / 4 * 4);
        CHECK(native_quant_h(NativeFmt::Q2_0, act.data(), F, nt, hqa));
        const int8_t* hq8 = (const int8_t*)hqa; const float* hd = (const float*)(hqa + (nt * F + 3) / 4 * 4);
        std::vector<float> out((size_t)nt * E, -5.f);
        CHECK(native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, 100, hqa, nt, out.data(), E));
        CHECK(native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 100, E, hqa, nt, out.data(), E));
        for (uint32_t t = 0; t < nt; ++t)
            for (uint32_t r = 0; r < E; ++r)
                CHECK(near(out[(size_t)t * E + r], q2_row_model(&blob[(size_t)l.down_off + (size_t)r * l.down_row_bytes], F / 64, hq8 + (size_t)t * F, hd + t * (F / 32)), 3e-3f));
        // argument validation
        CHECK(!native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F + 1, xqa, nt, act.data(), F + 1));
        CHECK(!native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 5, 4, xqa, nt, act.data(), F));
        CHECK(!native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F, xqa, 0, act.data(), F));
        CHECK(!native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, E + 1, hqa, nt, out.data(), E + 1));
        CHECK(!native_down_rows(NativeFmt::None, blob.data(), E, F, 0, 1, hqa, nt, out.data(), E));
        CHECK(!native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F, xqa, nt, act.data(), F - 1));   // stride shorter than the row range
        CHECK(!native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, E, hqa, nt, out.data(), E - 1));
    }

    // ---- i-quants: availability needs the codebooks; rows are consistent across token groups and with gate/up pairing
    {
        const uint32_t E = 512, F = 256;
        CHECK(!native_experts_available(16, E, F));   // no tables registered yet
        std::vector<uint64_t> g2xxs(256), g2xs(512), g2s(1024); std::vector<uint32_t> g3xxs(256), g3s(512);
        for (auto& v : g2xxs) { v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)(8 + 17 * (rnd32() % 3)) << (8 * i); }
        for (auto& v : g2xs) { v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)(8 + 17 * (rnd32() % 3)) << (8 * i); }
        for (auto& v : g2s) { v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)(8 + 17 * (rnd32() % 3)) << (8 * i); }
        for (auto& v : g3xxs) { v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)(4 + 8 * (rnd32() % 4)) << (8 * i); }
        for (auto& v : g3s) { v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)(4 + 8 * (rnd32() % 4)) << (8 * i); }
        kern::IqTables tb; tb.iq2xxs_grid = g2xxs.data(); tb.iq2xs_grid = g2xs.data(); tb.iq2s_grid = g2s.data(); tb.iq3xxs_grid = g3xxs.data(); tb.iq3s_grid = g3s.data();
        kern::iq512_set_tables(tb);
        CHECK(native_experts_available(16, E, F) == avx512 && native_experts_available(22, E, F) == avx512);
        CHECK(!native_experts_available(16, E, 160) && !native_experts_available(16, 100, F));
        // quantizer: Q8_K blocks
        { std::vector<float> x(512); for (auto& v : x) v = rndf() * 4.f; 
          std::vector<kern::Q8KBlock> bb(2);
          CHECK(native_quant_act(NativeFmt::IQ3_S, x.data(), 512, 1, bb.data()));
          for (int k = 0; k < 2; ++k) {
              float amax = 0.f; for (int i = 0; i < 256; ++i) amax = std::fmax(amax, std::fabs(x[k * 256 + i]));
              CHECK(near(bb[k].d, amax / 127.f, 1e-6f));
              for (int g = 0; g < 16; ++g) { int s = 0; for (int i = 0; i < 16; ++i) s += bb[k].qs[g * 16 + i]; CHECK(bb[k].bsums[g] == s); }
              for (int i = 0; i < 256; ++i) CHECK(bb[k].qs[i] >= -127 && bb[k].qs[i] <= 127 && std::fabs(bb[k].qs[i] * bb[k].d - x[k * 256 + i]) <= 0.5001f * bb[k].d + 1e-6f);
          } }
        if (avx512) {
            for (int type : {16, 17, 18, 21, 22}) {
                const NativeFmt f = native_fmt(type); const ExpertLayout l = expert_layout(type, E, F); CHECK(l.ok);
                std::vector<uint8_t> blob((size_t)l.bytes); for (auto& v : blob) v = (uint8_t)rnd32();
                for (int64_t o = 0; o < l.bytes; ) { // fp16 scale of every block must be finite
                    const int64_t bb = block_bytes(type); uint16_t h = (uint16_t)(0x2C00 + (rnd32() & 0xFF)); std::memcpy(&blob[o], &h, 2); o += bb; }
                const uint32_t nt = 5;
                std::vector<float> x((size_t)nt * E); for (auto& v : x) v = rndf();
                std::vector<kern::Q8KBlock> xq((size_t)nt * (E / 256));
                CHECK(native_quant_act(f, x.data(), E, nt, xq.data()));
                std::vector<float> act((size_t)nt * F, 0.f), g((size_t)nt * F), u((size_t)nt * F);
                CHECK(native_gu_rows(f, blob.data(), E, F, 0, 100, xq.data(), nt, act.data(), F));
                CHECK(native_gu_rows(f, blob.data(), E, F, 100, F, xq.data(), nt, act.data(), F));
                kern::iq512_rows(type, blob.data(), l.gu_row_bytes, 0, F, E / 256, xq.data(), nt, g.data(), F);
                kern::iq512_rows(type, blob.data() + l.up_off, l.gu_row_bytes, 0, F, E / 256, xq.data(), nt, u.data(), F);
                for (size_t i = 0; i < act.size(); ++i) CHECK(near(act[i], g[i] / (1.f + std::exp(-g[i])) * u[i], 1e-4f));
                // token independence: each token alone gives the same row dots as inside the group of 5
                for (uint32_t t = 0; t < nt; ++t) {
                    std::vector<float> one(F);
                    kern::iq512_rows(type, blob.data(), l.gu_row_bytes, 0, F, E / 256, xq.data() + (size_t)t * (E / 256), 1, one.data(), F);
                    for (uint32_t r = 0; r < F; ++r) CHECK(near(one[r], g[(size_t)t * F + r], 1e-4f));
                }
                // down rows
                std::vector<float> h((size_t)nt * F); for (auto& v : h) v = rndf();
                std::vector<kern::Q8KBlock> hq((size_t)nt * (F / 256));
                CHECK(native_quant_h(f, h.data(), F, nt, hq.data()));
                std::vector<float> out((size_t)nt * E, 0.f), ref((size_t)nt * E, 1.f);
                CHECK(native_down_rows(f, blob.data(), E, F, 0, 300, hq.data(), nt, out.data(), E));
                CHECK(native_down_rows(f, blob.data(), E, F, 300, E, hq.data(), nt, out.data(), E));
                kern::iq512_rows(type, blob.data() + l.down_off, l.down_row_bytes, 0, E, F / 256, hq.data(), nt, ref.data(), E);
                for (size_t i = 0; i < out.size(); ++i) CHECK(near(out[i], ref[i], 1e-5f));
                CHECK(!native_gu_rows(f, blob.data(), E, F, 0, F + 1, xq.data(), nt, act.data(), F + 1));
                CHECK(!native_gu_rows(f, blob.data(), E, F, 0, F, xq.data(), nt, act.data(), F - 1));
            }
        } else std::printf("SKIP: i-quant kernels need AVX-512\n");
    }

    if (fails) { std::printf("UNIT 20 TESTS FAILED (%d)\n", fails); return 1; }
    std::printf("UNIT 20 TESTS PASSED\n");
    return 0;
}
