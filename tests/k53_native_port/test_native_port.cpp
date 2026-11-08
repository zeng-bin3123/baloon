// Unit p29442642b2: native CPU expert surface (NativeFmt, act/h capacities, native_{fmt,quant_act,quant_h,gu_rows,down_rows,
// experts_available}) and the i-quant row entry points (iq512_supported, iq512_rows, iq512_gu_rows).
// (1) Port check: every row of the port table names a file in this repo that defines the listed identifier (12 rows).
// (2) Invariants: format<->type bijection, activation-buffer sizing, capacity bounds, Q8_K quantizer bounds, argument
//     validation (nothing written on a rejected call), availability gating, and an i-quant row kernel equal to the scalar model.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baloon/cpu_native_expert.hpp"
using namespace baloon;
static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static bool is_id(char c) { return std::isalnum((unsigned char)c) || c == '_'; }
static bool defines(const std::string& t, const std::string& name, const std::string& kind) {
    for (size_t p = t.find(name); p != std::string::npos; p = t.find(name, p + 1)) {
        if (p > 0 && is_id(t[p - 1])) continue;
        size_t e = p + name.size();
        if (e < t.size() && is_id(t[e])) continue;
        size_t ls = t.rfind('\n', p); ls = ls == std::string::npos ? 0 : ls + 1;
        std::string pre = t.substr(ls, p - ls);
        if (pre.find("//") != std::string::npos) continue;
        size_t q = e; while (q < t.size() && t[q] == ' ') ++q;
        char nx = q < t.size() ? t[q] : 0;
        if (kind == "const") { if (pre.find("constexpr") != std::string::npos && nx == '=') return true; continue; }
        if (kind == "type") { if (pre.find("enum ") != std::string::npos && (nx == '{' || nx == ':')) return true; continue; }
        if (nx == '(' && !pre.empty() && pre.find("return") == std::string::npos && pre.find('=') == std::string::npos &&
            pre.find('.') == std::string::npos && pre.find("->") == std::string::npos && pre.find('(') == std::string::npos) return true;
    }
    return false;
}
static void port_check(const char* root) {
    std::ifstream in(std::string(root) + "/.einsteinOS/port/p29442642b2.tsv");
    CHECK(in.good());
    std::vector<std::pair<std::string, std::string>> kinds;
    { std::ifstream kn(std::string(root) + "/tests/k53_native_port/ref_names.txt"); std::string a, b; while (kn >> a >> b) kinds.push_back({a, b}); }
    std::string line; int n = 0; std::set<std::string> seen;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f; std::stringstream ss(line); std::string t;
        while (std::getline(ss, t, '\t')) f.push_back(t);
        CHECK(f.size() == 4); if (f.size() != 4) continue;
        ++n;
        CHECK(f[0] == "include/ref/kernels/cpu/iq_avx512.hpp" || f[0] == "include/ref/kernels/cpu/native_expert.hpp");
        std::ifstream src(std::string(root) + "/" + f[2]); CHECK(src.good());
        std::stringstream buf; buf << src.rdbuf();
        std::string kind = "func"; for (auto& k : kinds) if (k.first == f[1]) kind = k.second;
        const bool ok = defines(buf.str(), f[3], kind);
        if (!ok) std::printf("port miss: %s -> %s %s\n", f[1].c_str(), f[2].c_str(), f[3].c_str());
        CHECK(ok);
        CHECK(seen.insert(f[0] + "|" + f[1]).second);
    }
    CHECK(n == 12);
}
static uint32_t rng = 20260101u;
static uint32_t rnd32() { rng = rng * 1664525u + 1013904223u; return rng; }
static float rndf() { return (float)((rnd32() >> 8) & 0xFFFF) / 32768.f - 1.f; }

int main(int argc, char** argv) {
    port_check(argc > 1 ? argv[1] : ".");

    // ---- NativeFmt <-> ggml type is a bijection on the six supported types; everything else is None ----
    const int types[6] = {42, 16, 17, 18, 21, 22};
    std::set<int> fm;
    for (int t : types) { NativeFmt f = native_fmt(t); CHECK(f != NativeFmt::None); CHECK(native_type_id(f) == t); fm.insert((int)f); }
    CHECK(fm.size() == 6);
    for (int t = -3; t < 120; ++t) { bool sup = false; for (int s : types) sup |= (s == t); if (!sup) CHECK(native_fmt(t) == NativeFmt::None); }
    CHECK(native_type_id(NativeFmt::None) == 0);

    // ---- iq512_supported: exactly the five i-quant types ----
    for (uint32_t t = 0; t < 64; ++t) CHECK(kern::iq512_supported(t) == (t == 16 || t == 17 || t == 18 || t == 21 || t == 22));

    // ---- capacities: one token of the model widths fits, for every format ----
    CHECK(kNativeActBytes == 3200 && kNativeHBytes == 876);
    for (int t : types) {
        NativeFmt f = native_fmt(t);
        const uint64_t a = native_act_bytes(f, 2560, 1), h = native_act_bytes(f, 640, 1);
        if (f == NativeFmt::Q2_0) { CHECK(a == 3200 && h == 800); }
        else { CHECK(a == 10 * 292 && h == 0); }
        CHECK(a <= kNativeActBytes && h <= kNativeHBytes);
    }
    CHECK(native_act_bytes(NativeFmt::IQ2_XXS, 768, 1) == 3 * 292 && native_act_bytes(NativeFmt::IQ2_XXS, 768, 1) <= kNativeHBytes);
    CHECK(native_act_bytes(NativeFmt::Q2_0, 2560, 3) == 3 * 3200 && native_act_bytes(NativeFmt::IQ3_S, 512, 4) == 4 * 2 * 292);
    CHECK(native_act_bytes(NativeFmt::None, 256, 1) == 0 && native_act_bytes(NativeFmt::Q2_0, 33, 1) == 0 && native_act_bytes(NativeFmt::IQ2_S, 255, 1) == 0);

    // ---- quantizers: per-block scale = amax/127, |q| <= 127, reconstruction error <= d/2, chunk sums exact ----
    {
        const uint32_t n = 512, nt = 3; std::vector<float> x((size_t)n * nt);
        for (auto& v : x) v = rndf() * 7.f;
        std::vector<kern::Q8KBlock> b((size_t)nt * (n / 256));
        CHECK(native_quant_act(NativeFmt::IQ3_S, x.data(), n, nt, b.data()));
        for (uint32_t t = 0; t < nt; ++t) for (uint32_t k = 0; k < n / 256; ++k) {
            const kern::Q8KBlock& q = b[(size_t)t * (n / 256) + k]; const float* xs = &x[(size_t)t * n + k * 256];
            float amax = 0; for (int i = 0; i < 256; ++i) amax = std::fmax(amax, std::fabs(xs[i]));
            CHECK(std::fabs(q.d - amax / 127.f) <= 1e-6f * (1.f + amax));
            for (int i = 0; i < 256; ++i) { CHECK(q.qs[i] >= -127 && q.qs[i] <= 127); CHECK(std::fabs(q.qs[i] * q.d - xs[i]) <= 0.5001f * q.d + 1e-6f); }
            for (int g = 0; g < 16; ++g) { int s = 0; for (int i = 0; i < 16; ++i) s += q.qs[g * 16 + i]; CHECK(q.bsums[g] == s); }
        }
        std::vector<kern::Q8KBlock> bh(b.size());   // the h-quantizer is the same map on the FFN-width vector
        CHECK(native_quant_h(NativeFmt::IQ3_S, x.data(), n, nt, bh.data()));
        CHECK(std::memcmp(b.data(), bh.data(), b.size() * sizeof(kern::Q8KBlock)) == 0);
        std::vector<uint8_t> z(3200);
        std::vector<float> zero(2560, 0.f);
        CHECK(native_quant_act(NativeFmt::IQ2_XS, zero.data(), 2560, 1, z.data()));   // all-zero input: scale 0, all q 0
        const kern::Q8KBlock* zb = (const kern::Q8KBlock*)z.data();
        for (int k = 0; k < 10; ++k) { CHECK(zb[k].d == 0.f); for (int i = 0; i < 256; ++i) CHECK(zb[k].qs[i] == 0); }
        // rejected calls
        CHECK(!native_quant_act(NativeFmt::None, x.data(), 256, 1, b.data()) && !native_quant_act(NativeFmt::Q2_0, x.data(), 40, 1, b.data()));
        CHECK(!native_quant_act(NativeFmt::IQ2_S, x.data(), 100, 1, b.data()) && !native_quant_act(NativeFmt::IQ2_S, nullptr, 256, 1, b.data()));
        CHECK(!native_quant_h(NativeFmt::IQ2_S, x.data(), 256, 0, b.data()) && !native_quant_h(NativeFmt::Q2_0, x.data(), 64, 1, nullptr));
    }

    // ---- availability gating ----
    CHECK(!native_experts_available(9, 2560, 640) && !native_experts_available(0, 2560, 640));
    CHECK(!native_experts_available(42, 2560 + 32, 0) && !native_experts_available(42, 0, 64) && !native_experts_available(42, 100, 64));
    CHECK(!native_experts_available(16, 2560, 640));   // 640 is not a whole number of 256-blocks: layout invalid
    { const bool q2 = kern::cpu_features().usable() || kern::cpu_features().avx2; CHECK(native_experts_available(42, 2560, 640) == q2); }

    // ---- i-quant row entry points: tables set -> rows equal the scalar model; rejected rows write nothing ----
    {
        // synthetic codebooks (data of the format): arbitrary but fixed magnitudes in [1, 25] for grid tables
        static uint64_t g2xxs[256], g2xs[512], g2s[1024]; static uint32_t g3xxs[256], g3s[512];
        auto mag = []() { return (uint8_t)(1 + rnd32() % 25); };
        for (auto& g : g2xxs) { uint8_t m[8]; for (auto& c : m) c = mag(); std::memcpy(&g, m, 8); }
        for (auto& g : g2xs) { uint8_t m[8]; for (auto& c : m) c = mag(); std::memcpy(&g, m, 8); }
        for (auto& g : g2s) { uint8_t m[8]; for (auto& c : m) c = mag(); std::memcpy(&g, m, 8); }
        for (auto& g : g3xxs) { uint8_t m[4]; for (auto& c : m) c = mag(); std::memcpy(&g, m, 4); }
        for (auto& g : g3s) { uint8_t m[4]; for (auto& c : m) c = mag(); std::memcpy(&g, m, 4); }
        kern::IqTables tb; tb.iq2xxs_grid = g2xxs; tb.iq2xs_grid = g2xs; tb.iq2s_grid = g2s; tb.iq3xxs_grid = g3xxs; tb.iq3s_grid = g3s;
        kern::iq512_set_tables(tb);
        const bool avx512 = cpu_avx512_ok();
        for (int t : types) if (t != 42) CHECK(kern::iq512_tables_ready((uint32_t)t));
        for (int t : types) if (t != 42) CHECK(native_experts_available(t, 2560, 768) == avx512 || !expert_layout(t, 2560, 768).ok);
        if (avx512) {
            // consistency of the three entry points on random blocks: rows of a gate|up matrix, then SwiGLU, then down rows
            const uint32_t E = 512, F = 256, nt = 5;
            for (int t : {16, 17, 18, 21, 22}) {
                const NativeFmt f = native_fmt(t); const ExpertLayout l = expert_layout(t, E, F); CHECK(l.ok);
                std::vector<uint8_t> blob((size_t)l.bytes);
                for (auto& c : blob) c = (uint8_t)rnd32();
                for (size_t off = 0; off < blob.size(); off += (size_t)(t == 16 ? 66 : t == 17 ? 74 : t == 18 ? 98 : t == 21 ? 110 : 82)) {   // fp16 scale 0.01..0.05: finite, small
                    const uint16_t h = (uint16_t)(0x2400 + (rnd32() % 0x300)); std::memcpy(&blob[off], &h, 2);
                }
                std::vector<float> x((size_t)E * nt); for (auto& v : x) v = rndf();
                std::vector<uint8_t> xq((size_t)native_act_bytes(f, E, nt)); CHECK(native_quant_act(f, x.data(), E, nt, xq.data()));
                std::vector<float> act((size_t)nt * F, -777.f);
                CHECK(native_gu_rows(f, blob.data(), E, F, 3, 40, xq.data(), nt, act.data(), F));
                std::vector<float> g((size_t)nt * F, 0.f), u((size_t)nt * F, 0.f);
                kern::iq512_rows((uint32_t)t, blob.data(), (size_t)l.gu_row_bytes, 3, 40, E / 256, (const kern::Q8KBlock*)xq.data(), nt, g.data(), F);
                kern::iq512_rows((uint32_t)t, blob.data() + l.up_off, (size_t)l.gu_row_bytes, 3, 40, E / 256, (const kern::Q8KBlock*)xq.data(), nt, u.data(), F);
                for (uint32_t tk = 0; tk < nt; ++tk) for (uint32_t r = 0; r < F; ++r) {
                    const float got = act[(size_t)tk * F + r];
                    if (r < 3 || r >= 40) { CHECK(got == -777.f); continue; }
                    const float gv = g[(size_t)tk * F + r], uv = u[(size_t)tk * F + r], ref = gv / (1.f + std::exp(-gv)) * uv;
                    CHECK(std::fabs(got - ref) <= 2e-3f * (1.f + std::fabs(ref)));
                }
                // token independence: one token alone gives the same row values as inside the batch
                std::vector<float> one(F, 0.f);
                kern::iq512_rows((uint32_t)t, blob.data(), (size_t)l.gu_row_bytes, 3, 40, E / 256, (const kern::Q8KBlock*)xq.data() + 3 * (E / 256), 1, one.data(), F);
                for (uint32_t r = 3; r < 40; ++r) CHECK(std::fabs(one[r] - g[(size_t)3 * F + r]) <= 1e-4f * (1.f + std::fabs(one[r])));
                // down rows
                std::vector<float> hq_in((size_t)F * nt); for (auto& v : hq_in) v = rndf();
                std::vector<uint8_t> hq((size_t)native_act_bytes(f, F, nt)); CHECK(native_quant_h(f, hq_in.data(), F, nt, hq.data()));
                std::vector<float> out((size_t)nt * E, -777.f), ref((size_t)nt * E, 0.f);
                CHECK(native_down_rows(f, blob.data(), E, F, 10, 70, hq.data(), nt, out.data(), E));
                kern::iq512_rows((uint32_t)t, blob.data() + l.down_off, (size_t)l.down_row_bytes, 10, 70, F / 256, (const kern::Q8KBlock*)hq.data(), nt, ref.data(), E);
                for (uint32_t tk = 0; tk < nt; ++tk) for (uint32_t r = 0; r < E; ++r) {
                    if (r < 10 || r >= 70) CHECK(out[(size_t)tk * E + r] == -777.f);
                    else CHECK(out[(size_t)tk * E + r] == ref[(size_t)tk * E + r]);
                }
                // rejected calls write nothing
                std::vector<float> keep(act);
                CHECK(!native_gu_rows(f, blob.data(), E, F, 0, F + 1, xq.data(), nt, act.data(), F + 1));
                CHECK(!native_gu_rows(f, blob.data(), E, F, 5, 4, xq.data(), nt, act.data(), F));
                CHECK(!native_gu_rows(f, blob.data(), E, F, 0, F, xq.data(), 0, act.data(), F));
                CHECK(!native_gu_rows(f, blob.data(), E, F, 0, F, xq.data(), nt, act.data(), F - 1));
                CHECK(!native_down_rows(f, blob.data(), E, F, 0, E + 1, hq.data(), nt, out.data(), E + 1));
                CHECK(!native_down_rows(f, nullptr, E, F, 0, 1, hq.data(), nt, out.data(), E));
                CHECK(act == keep);
            }
        }
    }
    std::printf("checks: %d, failures: %d\n", checks, fails);
    if (fails) { std::printf("FAIL\n"); return 1; }
    std::printf("PASS\n");
    return 0;
}
