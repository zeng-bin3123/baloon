// Unit pc4b51f94e7: device / hit hook / layout / native bindings surface. Port check + invariants on the real implementation.
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "baloon/core_device.hpp"
#include "baloon/core_hit_hook.hpp"
#include "baloon/core_layout.hpp"
#include "baloon/core_mtp.hpp"
#include "baloon/core_native.hpp"
#include "baloon/core_weights.hpp"
using namespace baloon::core;
static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static bool is_id(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static bool defines(const std::string& t, const std::string& n) {
    for (size_t p = t.find(n); p != std::string::npos; p = t.find(n, p + 1)) {
        if (p > 0 && is_id(t[p - 1])) continue;
        size_t e = p + n.size(); if (e < t.size() && is_id(t[e])) continue;
        size_t ls = t.rfind('\n', p); ls = ls == std::string::npos ? 0 : ls + 1;
        std::string pre = t.substr(ls, p - ls);
        if (pre.find("//") != std::string::npos) continue;
        size_t q = e; while (q < t.size() && t[q] == ' ') ++q;
        char nx = q < t.size() ? t[q] : 0;
        bool head = pre.find("struct ") != std::string::npos || pre.find("class ") != std::string::npos ||
                    pre.find("enum ") != std::string::npos || pre.find("using ") != std::string::npos;
        if (head && (nx == '{' || nx == '=' || nx == ':' || nx == ';')) return true;
        if (!head && nx == '(' && !pre.empty() && pre.find("return") == std::string::npos && pre.find('=') == std::string::npos &&
            pre.find('.') == std::string::npos && pre.find("->") == std::string::npos && pre.find('(') == std::string::npos) return true;
    }
    return false;
}
static void port_check(const std::string& root) {
    std::ifstream in(root + "/.einsteinOS/port/pc4b51f94e7.tsv"); CHECK(in.good());
    std::string line; int n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f; std::stringstream ss(line); std::string t;
        while (std::getline(ss, t, '\t')) f.push_back(t);
        CHECK(f.size() == 4); if (f.size() != 4) continue;
        ++n;
        std::ifstream src(root + "/" + f[2]); CHECK(src.good());
        std::stringstream b; b << src.rdbuf();
        if (!defines(b.str(), f[3])) { std::printf("port: %s not defined in %s\n", f[3].c_str(), f[2].c_str()); ++fails; }
    }
    CHECK(n == 17);
}
int main(int argc, char** argv) {
    port_check(argc > 1 ? argv[1] : ".");
    // device: ordinal range, errors, Iluvatar identity
    { CudaError e; DeviceInfo d = device_info(0, 4, &e);
      CHECK(d.valid && d.wave == 64 && d.arch == "ivcore10" && d.total_mem == (32768ull << 20) && e.ok());
      for (int o = 0; o < 4; ++o) CHECK(device_info(o, 4).ordinal == o);
      CHECK(!device_info(-1, 4, &e).valid && e.code == 101);
      CHECK(!device_info(4, 4, &e).valid && e.code == 101);
      CHECK(!device_info(0, 0, &e).valid && e.code == 100);
      CHECK(!CudaError{}.operator bool() && CudaError{2}.operator bool()); }
    // arena: aligned, disjoint, bounded, reset
    { DeviceArena a; a.init(2048); uint64_t end = 0;
      for (uint64_t sz : {1ull, 256ull, 300ull, 7ull}) {
          uint64_t o = a.alloc(sz);
          CHECK(o != UINT64_MAX && o % 256 == 0 && o >= end && o + sz <= 2048); end = o + sz; }
      CHECK(a.alloc(2048) == UINT64_MAX && a.last_error().code == 2);
      DeviceArena b; b.init(512); CHECK(b.alloc(512) == 0 && b.alloc(1) == UINT64_MAX);   // exact fit, then full
      CHECK(b.ptr(511) != nullptr && b.ptr(512) == nullptr);
      b.reset(); CHECK(b.used() == 0 && b.alloc(1) == 0); }
    // hit hook: phase filtering and per-row windows
    { TokenHits th; HitFn f = th.sink(HitPhase::Verify);
      f(HitPhase::Decode, 1, 1); f(HitPhase::Prefill, 1, 2); f(HitPhase::Verify, 2, 3);
      CHECK(th.hits.size() == 1 && th.hits[0].layer == 2 && th.hits[0].expert == 3);
      VerifyHits v; v.reset(2); CHECK(v.record(1, 0, 5) && !v.record(2, 0, 6) && v.total() == 1);
      CHECK((int)HitPhase::Decode == 0 && (int)HitPhase::Prefill == 1 && (int)HitPhase::Verify == 2); }
    // layout: QSA at l%4==3, 36 GDN + 12 QSA, dense per-kind points
    { ModelGeometry g; CHECK(g.n_qsa() == 12 && g.n_gdn() == 36 && check_all(g));
      for (int l = 0; l < 48; ++l) {
          CHECK(is_qsa_layer(l) == (l % 4 == 3));
          LayerView v = layer_view(g, l);
          CHECK(v.layer == l && (v.type == LayerType::Qsa) == is_qsa_layer(l));
          CHECK(v.point == (is_qsa_layer(l) ? l / 4 : l - l / 4));
          CHECK((v.type == LayerType::Gdn) == (v.state_floats > 0)); }
      std::string why; CHECK(!check_layer(g, -1, &why) && !why.empty() && !check_layer(g, 48));
      ModelGeometry z; z.n_layers = 0; CHECK(!check_all(z));
      ModelGeometry bad; bad.n_head = 15; bad.n_head_kv = 4; CHECK(!check_all(bad) || qsa_shapes(15, 256, 4).ok);
      ModelGeometry small; small.n_layers = 5; CHECK(small.n_qsa() == 1 && small.n_gdn() == 4 && check_all(small)); }
    // native embed / dense / head
    { float tab[8] = {1, 2, 3, 4, 5, 6, 7, 8}; NativeEmbed ne{tab, 4, 2}; float o[2];
      CHECK(ne.row(3, o) && o[0] == 7 && o[1] == 8 && !ne.row(4, o) && !ne.row(-1, o) && !NativeEmbed{}.valid());
      CHECK(native_embed() == nullptr); set_native_embed(&ne); CHECK(native_embed() == &ne);
      set_native_embed(nullptr); CHECK(native_embed() == nullptr);
      // dense bind: BF16 row of exactly rows*cols*2 bytes is accepted; wrong size / kind / unplaced rejected
      WeightTable wt; CHECK(wt.place("w 4 0 32 32 0 0 0 0\nq 2 32 32 32 0 0 0 0\n") && wt.find("w") && wt.find("w")->placed);
      std::vector<uint8_t> arena(wt.pool_bytes() + 256, 0); NativeDense nd;
      CHECK(nd.bind(wt, "w", arena.data(), 4, 4) && nd.valid());
      CHECK(!nd.bind(wt, "w", arena.data(), 4, 5) && !nd.valid());
      CHECK(!nd.bind(wt, "q", arena.data(), 4, 4) && !nd.bind(wt, "none", arena.data(), 4, 4) && !nd.bind(wt, "w", arena.data(), 0, 4));
      // identity-ish projection: BF16 1.0 = 0x3F80 on the diagonal
      uint16_t* w = reinterpret_cast<uint16_t*>(arena.data() + wt.find("w")->arena_off);
      for (int i = 0; i < 4; ++i) w[i * 4 + i] = 0x3F80;
      nd.bind(wt, "w", arena.data(), 4, 4); float x[4] = {1, 2, 3, 4}, y[4] = {};
      CHECK(nd.apply(x, y) && y[0] == 1 && y[3] == 4 && !NativeDense{}.apply(x, y));
      MtpTensor t; t.bytes = 3 * 4 * 2; NativeHead h{&t, 3, 4}; CHECK(h.present());
      t.bytes += 1; CHECK(!h.present()); NativeHead h0; CHECK(!h0.present()); }
    std::printf("%s: %d checks, %d failures\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
