// Unit padfc8bbf73: port table for the CPU expert kernels (35 rows) + invariants of the kernels themselves:
// feature gate, oracle switch, Q8_1 activation quantization (range, chunk sums, scalar==AVX2), GGUF Q2_0 row dots against an
// independent model, and the blob-layout expert (VNNI vs scalar, multi-token == per-token, row ranges tile the whole).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baloon/kern_cpu_expert.hpp"
using namespace baloon::kern;
static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static bool is_id(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static bool defines(const std::string& t, const std::string& name, const std::string& kind) {
    std::string base = name; size_t cc = name.rfind("::"); if (cc != std::string::npos) base = name.substr(cc + 2);
    for (size_t p = t.find(base); p != std::string::npos; p = t.find(base, p + 1)) {
        if (p > 0 && is_id(t[p - 1]) && !(p >= 2 && t[p - 1] == ':')) continue;
        size_t e = p + base.size(); if (e < t.size() && is_id(t[e])) continue;
        size_t ls = t.rfind('\n', p); ls = ls == std::string::npos ? 0 : ls + 1;
        std::string pre = t.substr(ls, p - ls); if (pre.find("//") != std::string::npos) continue;
        size_t q = e; while (q < t.size() && t[q] == ' ') ++q; char nx = q < t.size() ? t[q] : 0;
        if (kind == "const") { if (pre.find("constexpr") != std::string::npos && (nx == '=' || nx == '{')) return true; continue; }
        size_t nz = pre.find_first_not_of(" \t");
        if (nx == '(' && nz != std::string::npos && pre.find("return") == std::string::npos && pre.find('=') == std::string::npos && pre.find('(') == std::string::npos) {
            if (cc != std::string::npos && pre.find(name.substr(0, cc) + "::") == std::string::npos) continue;  // method: needs Class:: qualifier
            return true;
        }
    }
    return false;
}
static void port_check(const std::string& root) {
    std::ifstream kn(root + "/tests/k82_port_padfc8bbf73/ref_kinds.txt"); std::vector<std::pair<std::string, std::string>> kinds; std::string a, b;
    while (kn >> a >> b) kinds.push_back({a, b});
    std::ifstream in(root + "/.einsteinOS/port/padfc8bbf73.tsv"); CHECK(in.good());
    std::string line; int n = 0; std::set<std::string> seen;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f; std::stringstream ss(line); std::string t; while (std::getline(ss, t, '\t')) f.push_back(t);
        CHECK(f.size() == 4); if (f.size() != 4) continue; ++n;
        CHECK(f[0] == "src/kernels/cpu/expert.cpp");
        std::ifstream src(root + "/" + f[2]); CHECK(src.good()); std::stringstream buf; buf << src.rdbuf();
        std::string kind = "func"; for (auto& k : kinds) if (k.first == f[1]) kind = k.second;
        const bool ok = defines(buf.str(), f[3], kind); if (!ok) std::printf("port miss: %s -> %s %s\n", f[1].c_str(), f[2].c_str(), f[3].c_str());
        CHECK(ok); CHECK(seen.insert(f[1]).second);
    }
    CHECK(n == 35);
}
static uint32_t rng = 20261110u;
static uint32_t rnd32() { rng = rng * 1664525u + 1013904223u; return rng; }
static float rndf() { return (float)((rnd32() >> 8) & 0xFFFF) / 32768.f - 1.f; }
static float h2f(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023; const float s = (h >> 15) ? -1.f : 1.f;
    if (e == 0) return s * std::ldexp((float)m, -24);
    return s * std::ldexp((float)(m + 1024), e - 25);
}
static bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol * (1.f + std::fabs(b)); }

int main(int argc, char** argv) {
    port_check(argc > 1 ? argv[1] : ".");
    // ---- feature gate
    CpuFeatures f = cpu_features();
    CHECK(f.usable() == (f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi));
    CHECK((std::strcmp(f.reason(), "ok") == 0) == f.usable());
    CpuFeatures none; CHECK(!none.usable() && std::strcmp(none.reason(), "ok") != 0);
    CpuFeatures all; all.avx2 = all.avx512f = all.avx512bw = all.avx512vl = all.avx512_vnni = all.avx512_vbmi = true;
    CHECK(all.usable() && std::strcmp(all.reason(), "ok") == 0);
    for (int i = 0; i < 5; ++i) { CpuFeatures g = all; (i == 0 ? g.avx512f : i == 1 ? g.avx512bw : i == 2 ? g.avx512vl : i == 3 ? g.avx512_vnni : g.avx512_vbmi) = false; CHECK(!g.usable() && std::strcmp(g.reason(), "ok") != 0); }
    if (!f.usable()) { std::printf("SKIP kernels: %s\n", f.reason()); std::printf("%d checks, %d failures\n", checks, fails); return fails ? 1 : 0; }
    cpu_require_expert_support();
    // ---- oracle switch
    CHECK(!expert_oracle_q8_0_enabled()); expert_set_oracle_q8_0(true); CHECK(expert_oracle_q8_0_enabled()); expert_set_oracle_q8_0(false); CHECK(!expert_oracle_q8_0_enabled());
    // ---- activation quantization (Q8_1 form)
    {
        const uint32_t n = 256; std::vector<float> x(n); std::vector<int8_t> q(n), q2(n); std::vector<float> d(n / 32), d2(n / 32); std::vector<int32_t> hx(n / 32), hx2(n / 32);
        for (int rep = 0; rep < 50; ++rep) {
            for (auto& v : x) v = rndf() * (1.f + (rnd32() & 7));
            act_quant_q8_1(x.data(), n, q.data(), d.data(), hx.data()); act_quant_q8_1_avx2(x.data(), n, q2.data(), d2.data(), hx2.data());
            CHECK(std::memcmp(q.data(), q2.data(), n) == 0 && std::memcmp(d.data(), d2.data(), sizeof(float) * d.size()) == 0 && hx == hx2);
            for (uint32_t c = 0; c < n / 32; ++c) {
                int32_t s = 0; for (uint32_t i = 0; i < 32; ++i) { const int v = q[c * 32 + i]; CHECK(v >= -127 && v <= 127); s += v; CHECK(std::fabs(v * d[c] - x[c * 32 + i]) <= d[c] * 0.5001f + 1e-6f); }
                CHECK(hx[c] == s);
            }
        }
        std::vector<float> z(32, 0.f); int8_t qz[32]; float dz; int32_t hz; act_quant_q8_1(z.data(), 32, qz, &dz, &hz);
        bool allz = true; for (int i = 0; i < 32; ++i) allz &= qz[i] == 0; CHECK(allz && hz == 0);
        std::vector<float> o(32, 0.f); o[5] = 3.f; o[6] = -3.f; act_quant_q8_1(o.data(), 32, qz, &dz, &hz);   // odd symmetry
        CHECK(qz[5] == -qz[6] && std::abs((int)qz[5]) == 127 && hz == 0);
    }
    // ---- GGUF Q2_0 rows vs independent model (block: fp16 scale + 16 code bytes, 2 chunks of 32 activations per block)
    {
        const uint32_t nb = 40, rows = 7, nt = 3, K = nb * 64;
        std::vector<uint8_t> w((size_t)rows * nb * BB); for (auto& v : w) v = (uint8_t)rnd32();
        for (size_t b = 0; b < (size_t)rows * nb; ++b) { uint16_t h = (uint16_t)(0x2800 + (rnd32() & 0x3FF)); std::memcpy(&w[b * BB], &h, 2); }
        std::vector<float> x((size_t)nt * K); for (auto& v : x) v = rndf();
        std::vector<int8_t> q((size_t)nt * K); std::vector<float> xd((size_t)nt * (K / 32)); std::vector<int32_t> hx((size_t)nt * (K / 32));
        for (uint32_t t = 0; t < nt; ++t) act_quant_q8_1(&x[(size_t)t * K], K, &q[(size_t)t * K], &xd[(size_t)t * (K / 32)], &hx[(size_t)t * (K / 32)]);
        std::vector<float> out(nt * rows), out2(nt * rows);
        q2_0_gguf_rows_multi(w.data(), 0, rows, nb, nt, q.data(), K, xd.data(), hx.data(), K / 32, out.data(), rows);
        q2_0_gguf_rows_multi_avx2(w.data(), 0, rows, nb, nt, q.data(), K, xd.data(), hx.data(), K / 32, out2.data(), rows);
        for (uint32_t t = 0; t < nt; ++t) for (uint32_t r = 0; r < rows; ++r) {
            float acc = 0.f;
            for (uint32_t b = 0; b < nb; ++b) {
                const uint8_t* blk = &w[((size_t)r * nb + b) * BB]; uint16_t dh; std::memcpy(&dh, blk, 2); float s = 0.f;
                for (uint32_t i = 0; i < 64; ++i) { const int code = (blk[2 + i / 4] >> (2 * (i % 4))) & 3; s += (float)((code - 1) * q[(size_t)t * K + b * 64 + i]) * xd[(size_t)t * (K / 32) + 2 * b + i / 32]; }
                acc += h2f(dh) * s;
            }
            CHECK(near(out[t * rows + r], acc, 2e-3f)); CHECK(near(out2[t * rows + r], acc, 2e-3f));
        }
        // row ranges: [1,4) equals the same rows of the full computation
        std::vector<float> part(nt * rows, -7.f);
        q2_0_gguf_rows_multi(w.data(), 1, 4, nb, nt, q.data(), K, xd.data(), hx.data(), K / 32, part.data(), rows);
        for (uint32_t t = 0; t < nt; ++t) for (uint32_t r = 0; r < rows; ++r) { if (r >= 1 && r < 4) CHECK(part[t * rows + r] == out[t * rows + r]); else CHECK(part[t * rows + r] == -7.f); }
    }
    // ---- blob-layout expert
    {
        std::vector<uint8_t> blob(BLOB); for (auto& v : blob) v = (uint8_t)rnd32();
        for (uint64_t o = O_GU_SCALES; o + 2 <= BLOB; o += 2) { uint16_t h = (uint16_t)(0x1C00 + (rnd32() & 0x3FF)); std::memcpy(&blob[o], &h, 2); }
        std::vector<float> x((size_t)MAXT * H); for (auto& v : x) v = rndf();
        std::vector<float> ref(H), got(H); auto* s = new ExpertScratch; auto* sm = new ExpertScratchMulti;
        s2_expert_scalar(blob.data(), x.data(), ref.data()); s2_expert_vnni(blob.data(), x.data(), got.data(), *s);
        double num = 0, den = 0; for (uint32_t i = 0; i < H; ++i) { num += (double)(got[i] - ref[i]) * (got[i] - ref[i]); den += (double)ref[i] * ref[i]; }
        CHECK(den > 0 && std::sqrt(num / den) < 1e-3);
        // quantized-input entry point equals the float entry point
        std::vector<float> got_q(H); act_quant_q8_1(x.data(), H, s->x.q, s->x.d, s->x.hx); s2_expert_vnni_q(blob.data(), s->x, got_q.data(), *s);
        CHECK(std::memcmp(got_q.data(), got.data(), H * sizeof(float)) == 0);
        // multi == per token, for every width 1..MAXT
        for (uint32_t nt = 1; nt <= MAXT; ++nt) {
            std::vector<float> m((size_t)nt * H), one(H);
            s2_expert_vnni_multi(blob.data(), x.data(), nt, m.data(), *sm);
            for (uint32_t t = 0; t < nt; ++t) { s2_expert_vnni(blob.data(), x.data() + (size_t)t * H, one.data(), *s); double dn = 0, dd = 0; for (uint32_t i = 0; i < H; ++i) { const double e = m[(size_t)t * H + i] - one[i]; dn += e * e; dd += (double)one[i] * one[i]; } CHECK(std::sqrt(dn / (dd + 1e-30)) < 1e-4); }
        }
        // row ranges tile the whole: gate/up in two pieces, down in two pieces
        act_quant_q8_1(x.data(), H, s->x.q, s->x.d, s->x.hx);
        std::vector<float> a_full(FF), a_split(FF);
        s2_expert_gu_rows(blob.data(), s->x, 0, FF, a_full.data());
        s2_expert_gu_rows(blob.data(), s->x, 0, 100, a_split.data()); s2_expert_gu_rows(blob.data(), s->x, 100, FF, a_split.data());
        CHECK(std::memcmp(a_full.data(), a_split.data(), FF * sizeof(float)) == 0);
        std::memcpy(s->act, a_full.data(), FF * sizeof(float)); act_quant_q8_1(s->act, FF, s->a.q, s->a.d, s->a.hx);
        std::vector<float> d_full(H), d_split(H);
        s2_expert_down_rows(blob.data(), s->a, 0, H, d_full.data());
        s2_expert_down_rows(blob.data(), s->a, 0, 1000, d_split.data()); s2_expert_down_rows(blob.data(), s->a, 1000, H, d_split.data());
        CHECK(std::memcmp(d_full.data(), d_split.data(), H * sizeof(float)) == 0);
        CHECK(std::memcmp(d_full.data(), got.data(), H * sizeof(float)) == 0);   // pieces compose into the full expert
        // multi row-range pieces equal the single-token pieces
        for (uint32_t t = 0; t < 2; ++t) { act_quant_q8_1(x.data() + (size_t)t * H, H, sm->x[t].q, sm->x[t].d, sm->x[t].hx); }
        s2_expert_gu_rows_multi(blob.data(), sm->x, 2, 0, FF, sm->act);
        act_quant_q8_1(x.data() + H, H, s->x.q, s->x.d, s->x.hx); std::vector<float> a1(FF); s2_expert_gu_rows(blob.data(), s->x, 0, FF, a1.data());
        CHECK(near(sm->act[1][17], a1[17], 1e-4f) && near(sm->act[1][FF - 1], a1[FF - 1], 1e-4f));
        // oracle mode changes nothing about shape/finite-ness and is restored
        expert_set_oracle_q8_0(true); std::vector<float> orc(H); s2_expert_vnni(blob.data(), x.data(), orc.data(), *s); expert_set_oracle_q8_0(false);
        double on = 0, od = 0; bool fin = true; for (uint32_t i = 0; i < H; ++i) { fin &= std::isfinite(orc[i]); on += (double)(orc[i] - ref[i]) * (orc[i] - ref[i]); od += (double)ref[i] * ref[i]; }
        CHECK(fin && std::sqrt(on / od) < 0.05);
        delete s; delete sm;
    }
    std::printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
