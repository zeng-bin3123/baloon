// Unit p81030368e8: port check (33 rows of .einsteinOS/port/p81030368e8.tsv) + invariants of KV-Q8 sizing, fused gated-residual
// limits, IQ4 row/expert geometry and the native-GDN switch, on the real host implementations.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baloon/port_kernels_ref.hpp"
#include "baloon/port_scalar.hpp"
using namespace baloon::refk;
static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static bool is_id(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static bool defines(const std::string& t, const std::string& name, const std::string& kind) {
    for (size_t p = t.find(name); p != std::string::npos; p = t.find(name, p + 1)) {
        if (p > 0 && is_id(t[p - 1])) continue;
        size_t e = p + name.size(); if (e < t.size() && is_id(t[e])) continue;
        size_t ls = t.rfind('\n', p); ls = ls == std::string::npos ? 0 : ls + 1;
        std::string pre = t.substr(ls, p - ls); if (pre.find("//") != std::string::npos) continue;
        size_t q = e; while (q < t.size() && t[q] == ' ') ++q; char nx = q < t.size() ? t[q] : 0;
        if (kind == "const") { if (pre.find("constexpr") != std::string::npos && (nx == '=' || nx == '{')) return true; continue; }
        if (kind == "type") { if (pre.find("struct ") != std::string::npos && (nx == '{' || nx == ';')) return true; continue; }
        if (nx == '(' && !pre.empty() && pre.find("return") == std::string::npos && pre.find('=') == std::string::npos && pre.find('(') == std::string::npos) return true;
    }
    return false;
}
static void port_check(const std::string& root) {
    std::ifstream kn(root + "/tests/k81_port_p810/ref_kinds.txt"); std::vector<std::pair<std::string, std::string>> kinds; std::string a, b;
    while (kn >> a >> b) kinds.push_back({a, b});
    std::ifstream in(root + "/.einsteinOS/port/p81030368e8.tsv"); CHECK(in.good());
    std::string line; int n = 0; std::set<std::string> seen;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f; std::stringstream ss(line); std::string t; while (std::getline(ss, t, '\t')) f.push_back(t);
        CHECK(f.size() == 4); if (f.size() != 4) continue; ++n;
        CHECK(f[0].rfind("include/ref/kernels/", 0) == 0);
        std::ifstream src(root + "/" + f[2]); CHECK(src.good()); std::stringstream buf; buf << src.rdbuf();
        std::string kind = "func"; for (auto& k : kinds) if (k.first == f[1]) kind = k.second;
        const bool ok = defines(buf.str(), f[3], kind); if (!ok) std::printf("port miss: %s -> %s %s\n", f[1].c_str(), f[2].c_str(), f[3].c_str());
        CHECK(ok); CHECK(seen.insert(f[0] + "|" + f[1]).second);
    }
    CHECK(n == 33);
}
int main(int argc, char** argv) {
    port_check(argc > 1 ? argv[1] : ".");
    // KV-Q8: one int8 per value + one fp16 scale per KV_Q8_GROUP values; invalid shapes size to 0
    CHECK(KV_Q8_GROUP == 64);
    for (int h = 1; h <= 8; ++h) for (int d = 64; d <= 512; d += 64)
        CHECK(kv_q8_bytes_per_cell(h, d) == (int64_t)h * d + (int64_t)h * (d / KV_Q8_GROUP) * 2);
    CHECK(kv_q8_bytes_per_cell(2, 100) == 0 && kv_q8_bytes_per_cell(0, 64) == 0 && kv_q8_bytes_per_cell(2, 0) == 0 && kv_q8_bytes_per_cell(-1, 64) == 0);
    // fused gated residual: supported iff 1 <= T <= kFusedGrMaxT and n, hc > 0
    CHECK(kFusedGrMaxT == 8);
    for (int T = -1; T <= kFusedGrMaxT + 3; ++T) CHECK(fused_gr_supported(T, 4, 2) == (T >= 1 && T <= kFusedGrMaxT));
    CHECK(!fused_gr_supported(1, 0, 2) && !fused_gr_supported(1, 4, 0));
    // IQ4 geometry
    CHECK(iq_supported(kTypeIq4Nl) && iq_supported(kTypeIq4Xs) && !iq_supported(0) && !iq_supported(21));
    CHECK(iq_row_bytes(kTypeIq4Nl, 64) == 36 && iq_row_bytes(kTypeIq4Xs, 512) == 272);
    CHECK(iq_row_bytes(kTypeIq4Nl, 33) == 0 && iq_row_bytes(kTypeIq4Xs, 255) == 0 && iq_row_bytes(kTypeIq4Nl, 0) == 0 && iq_row_bytes(7, 64) == 0);
    for (int type : {kTypeIq4Nl, kTypeIq4Xs}) {
        const int E = 512, F = 256; NativeExpertLayout l = native_expert_layout(type, E, F);
        CHECK(l.ok && l.gu_row_bytes == iq_row_bytes(type, E) && l.down_row_bytes == iq_row_bytes(type, F));
        CHECK(l.up_off == (int64_t)F * l.gu_row_bytes && l.down_off == 2 * l.up_off && l.bytes == l.down_off + (int64_t)E * l.down_row_bytes);
        CHECK(!native_expert_layout(type, E, 100).ok && !native_expert_layout(type, 100, F).ok);
    }
    CHECK(!native_expert_layout(3, 512, 256).ok);
    CHECK(native_expert_scratch_bytes(512, 256) == 4ull * (512 + 2 * 256 + 512));
    // native GDN switch: default on, set/get round-trips
    CHECK(native_gdn_enabled()); native_gdn_set_enabled(false); CHECK(!native_gdn_enabled()); native_gdn_set_enabled(true); CHECK(native_gdn_enabled());
    std::printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
