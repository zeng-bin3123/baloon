// Unit pb61a53ee92: the decode-kernel identifier surface (88 reference definitions) and the scalar invariants under it.
// (1) Port check: every required (reference file, name, kind) row has exactly one-or-more entries in the port table, and each
//     entry names a file in this repo that really defines the identifier (comments and uses do not count).
// (2) LAYER-1 style invariants on the real host code: bf16 round-to-nearest-even, NeoX RoPE rotation, M-RoPE sectioning,
//     native-path switches, KV-Q8 sizing.
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
#include "baloon/port_scalar.hpp"
using namespace baloon::refk;
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
        if (kind == "macro") { if (pre.find("#define") != std::string::npos || pre.find("# define") != std::string::npos) return true; continue; }
        if (kind == "const") { if ((pre.find("constexpr") != std::string::npos || pre.find("const ") != std::string::npos) && (nx == '=' || nx == '{' || nx == '(')) return true; continue; }
        bool head = pre.find("struct ") != std::string::npos || pre.find("class ") != std::string::npos || pre.find("enum ") != std::string::npos || pre.find("using ") != std::string::npos;
        if (head && (nx == '{' || nx == '=' || nx == ':' || nx == ';')) return true;
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
    auto req = read_tsv(root + "/tests/k71_port_pb61/required_rows.tsv");
    auto tab = read_tsv(root + "/.einsteinOS/port/pb61a53ee92.tsv");
    CHECK(req.size() == 88);
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

static uint32_t rng_s = 12345u;
static uint32_t rnd32() { rng_s = rng_s * 1664525u + 1013904223u; return rng_s; }
static float rndf() { return ((rnd32() >> 8) & 0xFFFF) / 32768.f - 1.f; }

static void bf16_checks() {
    // exhaustive over every bf16 value: widening then narrowing is the identity (NaN words included: bf16_roundtrip)
    for (uint32_t b = 0; b < 65536; ++b) {
        const float f = f32_from_bf16((uint16_t)b);
        CHECK(bf16_from_f32(f) == b);
    }
    // rounding oracle: result is the nearest bf16, ties to even
    for (int i = 0; i < 400000; ++i) {
        uint32_t u = rnd32(); float f; std::memcpy(&f, &u, 4);
        if (!std::isfinite(f)) continue;
        const uint16_t r = bf16_from_f32(f);
        const uint32_t lo = u >> 16, rem = u & 0xFFFFu;
        uint16_t want;
        if (rem > 0x8000u) want = (uint16_t)(lo + 1); else if (rem < 0x8000u) want = (uint16_t)lo; else want = (uint16_t)((lo & 1u) ? lo + 1 : lo);
        CHECK(r == want);
    }
    // exact ties
    CHECK(bf16_from_f32(f32_from_bf16(0x3F80) + 0.0f) == 0x3F80);
    { uint32_t u = 0x3F808000u; float f; std::memcpy(&f, &u, 4); CHECK(bf16_from_f32(f) == 0x3F80); }   // tie, even stays
    { uint32_t u = 0x3F818000u; float f; std::memcpy(&f, &u, 4); CHECK(bf16_from_f32(f) == 0x3F82); }   // tie, odd bumps
    CHECK(bf16_from_f32(INFINITY) == 0x7F80);
    CHECK(bf16_from_f32(-INFINITY) == 0xFF80);
    { uint32_t u = 0x7F7FFFFFu; float f; std::memcpy(&f, &u, 4); CHECK(bf16_from_f32(f) == 0x7F80); }  // overflow rounds to inf
    { uint32_t u = 0xFFFFFFFFu; float f; std::memcpy(&f, &u, 4); CHECK(bf16_from_f32(f) == 0x0000); }   // no NaN branch: wraps
    for (uint32_t u : {0x7F800001u, 0x7F80FFFFu, 0xFF800001u, 0x7FBFFFFFu, 0x7FFFFFFFu}) {   // NaNs: no special case, the
        float f; std::memcpy(&f, &u, 4);                                                     // rounding add carries (mod 2^32)
        CHECK(bf16_from_f32(f) == (uint16_t)((uint32_t)(u + ((u >> 16) & 1u) + 0x7FFFu) >> 16));
    }
}

static void rope_checks() {
    const int n_rot = 64, hd = 80; const float theta = 10000.f;
    float cs[32], sn[32];
    for (int pos : {0, 1, 7, 513}) {
        build_rope_table(cs, sn, n_rot, pos, theta);
        for (int i = 0; i < 32; ++i) CHECK(std::fabs(cs[i] * cs[i] + sn[i] * sn[i] - 1.f) < 1e-5f);
        std::vector<float> x(hd), y;
        for (auto& v : x) v = rndf();
        y = x; rope_neox_apply(y.data(), n_rot, cs, sn);
        for (int i = 0; i < 32; ++i) {   // pair norm preserved
            const float a = x[i] * x[i] + x[i + 32] * x[i + 32], b = y[i] * y[i] + y[i + 32] * y[i + 32];
            CHECK(std::fabs(a - b) < 1e-4f);
        }
        for (int i = n_rot; i < hd; ++i) CHECK(y[i] == x[i]);   // tail passes through
        // inverse rotation restores the input
        float ns[32]; for (int i = 0; i < 32; ++i) ns[i] = -sn[i];
        rope_neox_apply(y.data(), n_rot, cs, ns);
        for (int i = 0; i < hd; ++i) CHECK(std::fabs(y[i] - x[i]) < 1e-4f);
    }
    build_rope_table(cs, sn, n_rot, 0, theta);   // position 0 is the identity
    for (int i = 0; i < 32; ++i) CHECK(cs[i] == 1.f && sn[i] == 0.f);
    float o0, o1; rope_neox_pair(1.f, 0.f, 0.f, 1.f, &o0, &o1); CHECK(o0 == 0.f && o1 == 1.f);
}

static void mrope_checks() {
    const int sec[3] = {11, 11, 10}, p3[3] = {5, 9, 13};
    for (int i = 0; i < 40; ++i) { int want = i < 11 ? 5 : (i < 22 ? 9 : 13); CHECK(mrope_pos(p3, sec, i) == want); }
    const int bad[3] = {4, -1, 3}; CHECK(!mrope_table_set(bad));
    const int bad0[3] = {-1, 4, 3}; CHECK(!mrope_table_set(bad0));
    const int bad2[3] = {4, 4, -3}; CHECK(!mrope_table_set(bad2));
    const int ok[3] = {11, 11, 10}; CHECK(mrope_table_set(ok));
    float cs[32], sn[32], rc[32], rs[32];
    const int same[3] = {17, 17, 17};
    mrope_table(cs, sn, 64, same, 10000.f);
    build_rope_table(rc, rs, 64, 17, 10000.f);   // equal axes collapse to plain RoPE
    for (int i = 0; i < 32; ++i) CHECK(cs[i] == rc[i] && sn[i] == rs[i]);
    const int sec2[3] = {32, 0, 0}; CHECK(mrope_table_set(sec2));
    mrope_table(cs, sn, 64, p3, 10000.f);
    build_rope_table(rc, rs, 64, 5, 10000.f);
    for (int i = 0; i < 32; ++i) CHECK(cs[i] == rc[i]);
    CHECK(mrope_table_set(ok));
}

static void switch_checks() {
    struct S { bool (*get)(); void (*set)(bool); } s[] = {
        {native_moe_combine_enabled, native_moe_combine_set_enabled}, {native_qsa_enabled, native_qsa_set_enabled},
        {native_qsa_indexer_enabled, native_qsa_indexer_set_enabled}, {native_rope_enabled, native_rope_set_enabled}};
    for (auto& x : s) {
        CHECK(x.get());                  // default on
        x.set(false); CHECK(!x.get());
        x.set(true); CHECK(x.get());
    }
    CHECK(kv_q8_bytes_per_cell(8, 128) == 8 * 128 + 8 * 2 * 2);
    CHECK(kv_q8_bytes_per_cell(8, 100) == 0 && kv_q8_bytes_per_cell(0, 128) == 0);
    CHECK((int)NativeFlashAttnStatus::Ok == 0);
}

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    port_check(root); bf16_checks(); rope_checks(); mrope_checks(); switch_checks();
    std::printf("%d checks, %d failures\n", checks, fails);
    if (fails == 0) std::printf("PASS\n");
    return fails == 0 ? 0 : 1;
}
