// Unit pb03f48e330: the native CPU expert identifier surface (9 reference definitions) and its invariants.
// (1) Port check: each required row has a port-table entry naming a file in this repo that really defines the identifier.
// (2) Invariants, against independent scalar models (not the code under test): format table, activation capacities and
//     packing, activation quantizer (round-half-away-from-zero, scale = amax/127, chunk sums), gate/up SwiGLU rows, down rows,
//     argument validation, availability, i-quant Q8_K blocks and row paths.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baloon/cpu_expert_layout.hpp"
#include "baloon/cpu_native_expert.hpp"
using namespace baloon;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

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
        if (kind == "const") { if ((pre.find("constexpr") != std::string::npos || pre.find("const ") != std::string::npos) && nx == '=') return true; continue; }
        bool head = pre.find("struct ") != std::string::npos || pre.find("class ") != std::string::npos || pre.find("enum ") != std::string::npos || pre.find("using ") != std::string::npos;
        if (kind == "type") { if (head && (nx == '{' || nx == '=' || nx == ':')) return true; continue; }
        if (!head && nx == '(' && !pre.empty() && pre.find("return") == std::string::npos && pre.find('=') == std::string::npos &&
            pre.find('.') == std::string::npos && pre.find("->") == std::string::npos && pre.find('(') == std::string::npos) return true;
    }
    return false;
}
static std::vector<std::vector<std::string>> read_tsv(const std::string& p) {
    std::vector<std::vector<std::string>> rows; std::ifstream in(p); CHECK(in.good());
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f; std::stringstream ss(line); std::string t;
        while (std::getline(ss, t, '\t')) f.push_back(t);
        rows.push_back(f);
    }
    return rows;
}
static void port_check(const std::string& root) {
    auto req = read_tsv(root + "/tests/k72_port_pb03/required_rows.tsv");
    auto tab = read_tsv(root + "/.einsteinOS/port/pb03f48e330.tsv");
    CHECK(req.size() == 9);
    std::map<std::string, std::string> kind;
    for (auto& r : req) { CHECK(r.size() == 3); if (r.size() == 3) kind[r[0] + "|" + r[1]] = r[2]; }
    std::set<std::string> covered;
    for (auto& f : tab) {
        CHECK(f.size() == 4); if (f.size() != 4) continue;
        const std::string key = f[0] + "|" + f[1];
        CHECK(kind.count(key) == 1);
        std::ifstream src(root + "/" + f[2]); CHECK(src.good());
        std::stringstream buf; buf << src.rdbuf();
        const bool ok = defines(buf.str(), f[3], kind.count(key) ? kind[key] : "func");
        if (!ok) std::printf("port miss: %s -> %s %s\n", f[1].c_str(), f[2].c_str(), f[3].c_str());
        CHECK(ok);
        if (ok) covered.insert(key);
    }
    for (auto& r : req) { const bool c = covered.count(r[0] + "|" + r[1]) == 1; if (!c) std::printf("uncovered: %s\n", r[1].c_str()); CHECK(c); }
}

static uint32_t rng = 20260110u;
static uint32_t rnd32() { rng = rng * 1664525u + 1013904223u; return rng; }
static float rndf() { return (float)((rnd32() >> 8) & 0xFFFF) / 32768.f - 1.f; }
static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol * (1.f + std::fabs(b)); }
static float h2f(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023; const float s = (h >> 15) ? -1.f : 1.f;
    if (e == 0) return s * std::ldexp((float)m, -24);
    return s * std::ldexp((float)(m + 1024), e - 25);
}
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
static int ref_round(float v) { return (int)std::round(v); }   // half away from zero (odd-symmetric)

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    port_check(root);

    // ---- NativeFmt / native_fmt: exactly six ggml types map, everything else is None; type id round-trips
    CHECK(native_fmt(42) == NativeFmt::Q2_0 && native_fmt(16) == NativeFmt::IQ2_XXS && native_fmt(17) == NativeFmt::IQ2_XS);
    CHECK(native_fmt(18) == NativeFmt::IQ3_XXS && native_fmt(21) == NativeFmt::IQ3_S && native_fmt(22) == NativeFmt::IQ2_S);
    CHECK(NativeFmt::None == static_cast<NativeFmt>(0));
    { std::set<int> good = {42, 16, 17, 18, 21, 22}; std::set<int> seen;
      for (int t = -3; t < 200; ++t) {
          const NativeFmt f = native_fmt(t);
          if (good.count(t)) { CHECK(f != NativeFmt::None && native_type_id(f) == t); CHECK(seen.insert((int)f).second); }
          else CHECK(f == NativeFmt::None && native_type_id(f) == 0);
      } CHECK(seen.size() == 6); }

    // ---- capacities: kNativeActBytes bounds a token's quantized hidden vector (2560) over all formats, kNativeHBytes the SwiGLU
    // vector (640 rounded to whole 256-blocks for i-quants); both are tight maxima.
    { uint64_t ma = 0, mh = 0;
      for (int t : {42, 16, 17, 18, 21, 22}) {
          const NativeFmt f = native_fmt(t);
          const uint64_t a = native_act_bytes(f, 2560, 1); CHECK(a > 0); if (a > ma) ma = a;
          const uint64_t h = f == NativeFmt::Q2_0 ? native_act_bytes(f, 640, 1) : 3 * 292;   // 640 not a block multiple for i-quants
          if (h > mh) mh = h;
      }
      CHECK(ma == 3200 && kNativeActBytes == ma);
      CHECK(mh == 876 && kNativeHBytes == mh);
      CHECK(kNativeActBytes >= 10 * 292 && kNativeHBytes >= 640 + 20 * 8); }

    // ---- activation sizes
    CHECK(native_act_bytes(NativeFmt::Q2_0, 2560, 1) == 2560 + 80 * 8);
    CHECK(native_act_bytes(NativeFmt::Q2_0, 640, 3) == 3 * (640 + 20 * 8));
    CHECK(native_act_bytes(NativeFmt::IQ3_S, 512, 4) == 4 * 2 * 292 && native_act_bytes(NativeFmt::IQ2_XS, 2560, 2) == 2 * 10 * 292);
    CHECK(native_act_bytes(NativeFmt::Q2_0, 48, 1) == 0 && native_act_bytes(NativeFmt::Q2_0, 40, 1) == 0 && native_act_bytes(NativeFmt::Q2_0, 0, 1) == 0);
    CHECK(native_act_bytes(NativeFmt::IQ3_S, 640, 1) == 0 && native_act_bytes(NativeFmt::IQ2_S, 128, 1) == 0);
    CHECK(native_act_bytes(NativeFmt::None, 256, 1) == 0 && native_act_bytes(NativeFmt::Q2_0, 64, 0) == 0);

    // ---- availability
    CHECK(!native_experts_available(9, 256, 128) && !native_experts_available(42, 256, 0) && !native_experts_available(42, 0, 128));
    CHECK(!native_experts_available(42, 288, 128) && !native_experts_available(42, 256, 100) && !native_experts_available(-1, 256, 128));
    CHECK(!native_experts_available(16, 512, 256));   // no codebooks registered yet
    const bool avx512 = cpu_avx512_ok();
    const bool any = avx512 || kern::cpu_features().avx2;
    std::printf("cpu: avx512=%d any=%d\n", (int)avx512, (int)any);
    CHECK(native_experts_available(42, 256, 128) == any);

    // ---- quantizer argument validation (works on every CPU)
    { std::vector<float> x(64, 1.f); alignas(4) uint8_t buf[512];
      CHECK(!native_quant_act(NativeFmt::Q2_0, x.data(), 40, 1, buf) && !native_quant_act(NativeFmt::Q2_0, x.data(), 48, 1, buf));
      CHECK(!native_quant_act(NativeFmt::None, x.data(), 64, 1, buf) && !native_quant_act(NativeFmt::IQ2_S, x.data(), 64, 1, buf));
      CHECK(!native_quant_act(NativeFmt::Q2_0, nullptr, 64, 1, buf) && !native_quant_act(NativeFmt::Q2_0, x.data(), 64, 1, nullptr));
      CHECK(!native_quant_h(NativeFmt::Q2_0, x.data(), 64, 0, buf) && !native_quant_h(NativeFmt::IQ3_S, x.data(), 64, 1, buf));
      CHECK(native_quant_h(NativeFmt::Q2_0, x.data(), 64, 1, buf)); }

    // ---- quantizer semantics against an independent model: half away from zero, scale amax/127, chunk sums
    { float x[32] = {}; x[0] = 127.f; x[1] = 0.5f; x[2] = 1.5f; x[3] = 2.5f; x[4] = -0.5f; x[5] = -2.5f; x[6] = -127.f; x[7] = 3.5f; x[8] = 0.49f; x[9] = 0.51f;
      alignas(4) uint8_t buf[32 + 8];
      CHECK(native_quant_act(NativeFmt::Q2_0, x, 32, 1, buf));
      const int8_t* q = (const int8_t*)buf; const float d = *(const float*)(buf + 32); const int32_t hx = *(const int32_t*)(buf + 36);
      CHECK(d == 1.f && q[0] == 127 && q[1] == 1 && q[2] == 2 && q[3] == 3 && q[4] == -1 && q[5] == -3 && q[6] == -127 && q[7] == 4 && q[8] == 0 && q[9] == 1);
      int s = 0; for (int i = 0; i < 32; ++i) s += q[i]; CHECK(hx == s); }
    for (int rep = 0; rep < 40; ++rep) {
        const uint32_t n = 32 * (1 + rnd32() % 6), nt = 1 + rnd32() % 4, nc = n / 32; const float sc = 0.01f + (rnd32() % 1000) * 0.05f;
        std::vector<float> x((size_t)nt * n); for (auto& v : x) v = rndf() * sc;
        std::vector<uint8_t> raw(native_act_bytes(NativeFmt::Q2_0, n, nt) + 4, 0xAB);
        uint8_t* buf = (uint8_t*)(((uintptr_t)raw.data() + 3) / 4 * 4);
        CHECK(native_quant_act(NativeFmt::Q2_0, x.data(), n, nt, buf));
        const int8_t* q = (const int8_t*)buf; const float* d = (const float*)(buf + ((uint64_t)nt * n + 3) / 4 * 4); const int32_t* hx = (const int32_t*)(d + (size_t)nt * nc);
        for (uint32_t t = 0; t < nt; ++t) for (uint32_t c = 0; c < nc; ++c) {
            float amax = 0.f; int s = 0;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[(size_t)t * n + c * 32 + i]));
            const float dd = d[(size_t)t * nc + c]; CHECK(near(dd, amax / 127.f, 1e-6f));
            for (int i = 0; i < 32; ++i) { const int v = q[(size_t)t * n + c * 32 + i]; s += v; CHECK(v >= -127 && v <= 127); CHECK(std::fabs(v * dd - x[(size_t)t * n + c * 32 + i]) <= 0.5001f * dd + 1e-6f);
                if (dd > 0) CHECK(std::abs(v - ref_round(x[(size_t)t * n + c * 32 + i] * (1.f / dd))) == 0); }
            CHECK(hx[(size_t)t * nc + c] == s);
        }
        // native_quant_h is the same quantizer
        std::vector<uint8_t> raw2(raw.size()); uint8_t* b2 = (uint8_t*)(((uintptr_t)raw2.data() + 3) / 4 * 4);
        CHECK(native_quant_h(NativeFmt::Q2_0, x.data(), n, nt, b2));
        CHECK(std::memcmp(buf, b2, native_act_bytes(NativeFmt::Q2_0, n, nt)) == 0);
    }

    // scalar and AVX2 forms: half away from zero and bit-identical, including exact ties and near-ties
    { float x[32] = {}; x[0] = 127.f; x[1] = 0.5f; x[2] = 1.5f; x[3] = 2.5f; x[4] = -0.5f; x[5] = -2.5f; x[6] = -1.5f; x[7] = 0.49999997f; x[8] = -0.49999997f;
      int8_t qs[32], qa[32]; float ds, da; int32_t hs, ha;
      kern::act_quant_q8_1(x, 32, qs, &ds, &hs); kern::act_quant_q8_1_avx2(x, 32, qa, &da, &ha);
      CHECK(ds == 1.f && qs[1] == 1 && qs[2] == 2 && qs[3] == 3 && qs[4] == -1 && qs[5] == -3 && qs[6] == -2 && qs[7] == 0 && qs[8] == 0);
      CHECK(std::memcmp(qs, qa, 32) == 0 && ds == da && hs == ha); }

    // ---- Q2_0 expert stages through the entry points
    if (any) {
        const uint32_t E = 256, F = 128, nt = 3;
        const ExpertLayout l = expert_layout(42, E, F);
        CHECK(l.ok);
        std::vector<uint8_t> blob((size_t)l.bytes); fill_q2(blob, (uint64_t)l.bytes / 18);
        std::vector<float> x((size_t)nt * E); for (auto& v : x) v = rndf() * 2.f;
        std::vector<uint8_t> xq(native_act_bytes(NativeFmt::Q2_0, E, nt) + 4); uint8_t* xqa = (uint8_t*)(((uintptr_t)xq.data() + 3) / 4 * 4);
        CHECK(native_quant_act(NativeFmt::Q2_0, x.data(), E, nt, xqa));
        const int8_t* pq = (const int8_t*)xqa; const float* pd = (const float*)(xqa + (nt * E + 3) / 4 * 4);
        std::vector<float> act((size_t)nt * (F + 3), -5.f); const size_t st = F + 3;
        CHECK(native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, 50, xqa, nt, act.data(), st));
        CHECK(native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 50, F, xqa, nt, act.data(), st));
        for (uint32_t t = 0; t < nt; ++t) {
            for (uint32_t r = 0; r < F; ++r) {
                const float g = q2_row_model(&blob[(size_t)r * l.gu_row_bytes], E / 64, pq + (size_t)t * E, pd + t * (E / 32));
                const float u = q2_row_model(&blob[(size_t)l.up_off + (size_t)r * l.gu_row_bytes], E / 64, pq + (size_t)t * E, pd + t * (E / 32));
                CHECK(near(act[t * st + r], g / (1.f + std::exp(-g)) * u, 3e-3f));
            }
            for (uint32_t r = F; r < st; ++r) CHECK(act[t * st + r] == -5.f);   // padding between tokens untouched
        }
        // a partial range writes only its rows
        { std::vector<float> a2((size_t)nt * F, -9.f);
          CHECK(native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 10, 20, xqa, nt, a2.data(), F));
          for (uint32_t t = 0; t < nt; ++t) for (uint32_t r = 0; r < F; ++r) { const bool in = r >= 10 && r < 20; CHECK(in ? a2[t * F + r] != -9.f : a2[t * F + r] == -9.f); }
          CHECK(native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 7, 7, xqa, nt, a2.data(), F)); }   // empty range is a no-op success
        // token independence
        { std::vector<float> a1(F, 0.f);
          for (uint32_t t = 0; t < nt; ++t) {
              std::vector<uint8_t> one(native_act_bytes(NativeFmt::Q2_0, E, 1) + 4); uint8_t* o = (uint8_t*)(((uintptr_t)one.data() + 3) / 4 * 4);
              CHECK(native_quant_act(NativeFmt::Q2_0, &x[(size_t)t * E], E, 1, o));
              CHECK(native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F, o, 1, a1.data(), F));
              for (uint32_t r = 0; r < F; ++r) CHECK(near(a1[r], act[t * st + r], 3e-3f));
          } }
        // down
        std::vector<float> h((size_t)nt * F); for (uint32_t t = 0; t < nt; ++t) for (uint32_t r = 0; r < F; ++r) h[(size_t)t * F + r] = act[t * st + r];
        std::vector<uint8_t> hq(native_act_bytes(NativeFmt::Q2_0, F, nt) + 4); uint8_t* hqa = (uint8_t*)(((uintptr_t)hq.data() + 3) / 4 * 4);
        CHECK(native_quant_h(NativeFmt::Q2_0, h.data(), F, nt, hqa));
        const int8_t* hq8 = (const int8_t*)hqa; const float* hd = (const float*)(hqa + (nt * F + 3) / 4 * 4);
        std::vector<float> out((size_t)nt * (E + 2), -5.f); const size_t os = E + 2;
        CHECK(native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, 100, hqa, nt, out.data(), os));
        CHECK(native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 100, E, hqa, nt, out.data(), os));
        for (uint32_t t = 0; t < nt; ++t) {
            for (uint32_t r = 0; r < E; ++r)
                CHECK(near(out[t * os + r], q2_row_model(&blob[(size_t)l.down_off + (size_t)r * l.down_row_bytes], F / 64, hq8 + (size_t)t * F, hd + t * (F / 32)), 3e-3f));
            for (uint32_t r = E; r < os; ++r) CHECK(out[t * os + r] == -5.f);
        }
        // validation: bad ranges, strides, null pointers, zero tokens, bad format
        CHECK(!native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F + 1, xqa, nt, act.data(), F + 1));
        CHECK(!native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 5, 4, xqa, nt, act.data(), st));
        CHECK(!native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F, xqa, 0, act.data(), st));
        CHECK(!native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F, xqa, nt, act.data(), F - 1));
        CHECK(!native_gu_rows(NativeFmt::Q2_0, nullptr, E, F, 0, F, xqa, nt, act.data(), st) && !native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F, nullptr, nt, act.data(), st));
        CHECK(!native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F, xqa, nt, nullptr, st) && !native_gu_rows(NativeFmt::None, blob.data(), E, F, 0, F, xqa, nt, act.data(), st));
        CHECK(!native_gu_rows(NativeFmt::IQ2_XXS, blob.data(), E, 100, 0, 1, xqa, nt, act.data(), st));   // layout invalid (n_ff not a block multiple)
        CHECK(!native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, E + 1, hqa, nt, out.data(), os));
        CHECK(!native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 3, 2, hqa, nt, out.data(), os));
        CHECK(!native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, E, hqa, 0, out.data(), os));
        CHECK(!native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, E, hqa, nt, out.data(), E - 1));
        CHECK(!native_down_rows(NativeFmt::None, blob.data(), E, F, 0, 1, hqa, nt, out.data(), os));
        CHECK(!native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, E, nullptr, nt, out.data(), os) && !native_down_rows(NativeFmt::Q2_0, nullptr, E, F, 0, E, hqa, nt, out.data(), os));
    } else std::printf("SKIP: no usable SIMD for Q2_0 rows\n");

    // ---- i-quants
    {
        const uint32_t E = 512, F = 256;
        std::vector<uint64_t> g2xxs(256), g2xs(512), g2s(1024); std::vector<uint32_t> g3xxs(256), g3s(512);
        for (auto* v : {&g2xxs, &g2xs, &g2s}) for (auto& e : *v) { e = 0; for (int i = 0; i < 8; ++i) e |= (uint64_t)(8 + 17 * (rnd32() % 3)) << (8 * i); }
        for (auto* v : {&g3xxs, &g3s}) for (auto& e : *v) { e = 0; for (int i = 0; i < 4; ++i) e |= (uint32_t)(4 + 8 * (rnd32() % 4)) << (8 * i); }
        kern::IqTables tb; tb.iq2xxs_grid = g2xxs.data(); tb.iq2xs_grid = g2xs.data(); tb.iq2s_grid = g2s.data(); tb.iq3xxs_grid = g3xxs.data(); tb.iq3s_grid = g3s.data();
        kern::iq512_set_tables(tb);
        for (int t : {16, 17, 18, 21, 22}) CHECK(native_experts_available(t, E, F) == avx512);
        CHECK(!native_experts_available(16, E, 160) && !native_experts_available(16, 100, F) && !native_experts_available(42, E + 32, F));
        { std::vector<float> x(512); for (auto& v : x) v = rndf() * 4.f; std::vector<kern::Q8KBlock> bb(2);
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
                for (int64_t o = 0; o < l.bytes; o += expert_layout_ns::block_bytes(type)) { uint16_t h = (uint16_t)(0x2C00 + (rnd32() & 0xFF)); std::memcpy(&blob[o], &h, 2); }
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
                std::vector<float> hh((size_t)nt * F); for (auto& v : hh) v = rndf();
                std::vector<kern::Q8KBlock> hq((size_t)nt * (F / 256));
                CHECK(native_quant_h(f, hh.data(), F, nt, hq.data()));
                std::vector<float> out((size_t)nt * E, 0.f), ref((size_t)nt * E, 1.f);
                CHECK(native_down_rows(f, blob.data(), E, F, 0, 300, hq.data(), nt, out.data(), E));
                CHECK(native_down_rows(f, blob.data(), E, F, 300, E, hq.data(), nt, out.data(), E));
                kern::iq512_rows(type, blob.data() + l.down_off, l.down_row_bytes, 0, E, F / 256, hq.data(), nt, ref.data(), E);
                for (size_t i = 0; i < out.size(); ++i) CHECK(near(out[i], ref[i], 1e-5f));
                CHECK(!native_gu_rows(f, blob.data(), E, F, 0, F + 1, xq.data(), nt, act.data(), F + 1));
                CHECK(!native_gu_rows(f, blob.data(), E, F, 0, F, xq.data(), nt, act.data(), F - 1));
                CHECK(!native_down_rows(f, blob.data(), E, F, 0, E + 1, hq.data(), nt, out.data(), E + 1));
                CHECK(!native_quant_h(f, hh.data(), 160, nt, hq.data()));   // 160 is not a whole Q8_K block
            }
        } else std::printf("SKIP: i-quant kernels need AVX-512\n");
    }

    if (fails) { std::printf("FAIL: %d checks failed\n", fails); return 1; }
    std::printf("PASS unit pb03f48e330\n");
    return 0;
}
