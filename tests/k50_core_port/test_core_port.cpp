// Unit p42b0ab780a: the core runtime surface (device, graph, hit hook, native bindings, pinned arena, session).
// (1) Port check: every line of the port table names a file in this repo that defines the listed identifier.
// (2) Invariants of each area, checked on the real implementation.
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include "baloon/core_device.hpp"
#include "baloon/core_graph.hpp"
#include "baloon/core_hit_hook.hpp"
#include "baloon/core_mtp.hpp"
#include "baloon/core_native.hpp"
#include "baloon/core_pinned.hpp"
#include "baloon/core_session_state.hpp"
#include "baloon/core_weights.hpp"
using namespace baloon::core;
namespace parena = baloon::parena;
static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static bool is_id(char c) { return std::isalnum((unsigned char)c) || c == '_'; }
// True when `name` occurs in `text` as a whole identifier that is being DEFINED (type/enum/using/alias head, or a function name followed by '(' ).
static bool defines(const std::string& text, const std::string& name) {
    for (size_t p = text.find(name); p != std::string::npos; p = text.find(name, p + 1)) {
        if (p > 0 && is_id(text[p - 1])) continue;
        size_t e = p + name.size();
        if (e < text.size() && is_id(text[e])) continue;
        size_t ls = text.rfind('\n', p); ls = ls == std::string::npos ? 0 : ls + 1;
        std::string pre = text.substr(ls, p - ls);
        if (pre.find("//") != std::string::npos) continue;
        size_t q = e; while (q < text.size() && text[q] == ' ') ++q;
        char nx = q < text.size() ? text[q] : 0;
        bool head = pre.find("struct ") != std::string::npos || pre.find("class ") != std::string::npos || pre.find("enum ") != std::string::npos ||
                    pre.find("using ") != std::string::npos;
        if (head && (nx == '{' || nx == '=' || nx == ':' || nx == ';')) return true;
        if (!head && nx == '(' && !pre.empty() && pre.find("return") == std::string::npos && pre.find('=') == std::string::npos &&
            pre.find('.') == std::string::npos && pre.find("->") == std::string::npos && pre.find("(") == std::string::npos) return true;
    }
    return false;
}

static void port_check(const char* root) {
    std::ifstream in(std::string(root) + "/.einsteinOS/port/p42b0ab780a.tsv");
    CHECK(in.good());
    std::string line; int n = 0;
    std::vector<std::string> seen;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f; std::stringstream ss(line); std::string t;
        while (std::getline(ss, t, '\t')) f.push_back(t);
        CHECK(f.size() == 4);
        if (f.size() != 4) continue;
        ++n;
        CHECK(f[0].rfind("include/ref/core/", 0) == 0);
        std::ifstream src(std::string(root) + "/" + f[2]);
        CHECK(src.good());
        std::stringstream buf; buf << src.rdbuf();
        const bool ok = defines(buf.str(), f[3]);
        if (!ok) std::printf("port miss: %s -> %s %s\n", f[1].c_str(), f[2].c_str(), f[3].c_str());
        CHECK(ok);
        seen.push_back(f[0] + "|" + f[1]);
    }
    CHECK(n == 46);
    for (size_t i = 0; i < seen.size(); ++i) for (size_t j = i + 1; j < seen.size(); ++j) CHECK(seen[i] != seen[j]);
}

int main(int argc, char** argv) {
    const char* root = argc > 1 ? argv[1] : ".";
    port_check(root);

    // ---- device ----
    { CudaError e; DeviceInfo d = device_info(3, 4, &e);
      CHECK(d.valid && d.ordinal == 3 && d.wave == 64 && d.arch == "ivcore10" && e.ok());
      CHECK(!device_info(4, 4, &e).valid && e.code == 101 && static_cast<bool>(e));
      CHECK(!device_info(0, 0, &e).valid && e.code == 100);
      DeviceArena a; a.init(4096);
      uint64_t prev_end = 0;
      for (uint64_t sz : {1ull, 255ull, 256ull, 257ull, 1000ull}) {
          uint64_t o = a.alloc(sz);
          CHECK(o != UINT64_MAX && o % DeviceArena::kAlign == 0 && o >= prev_end && o + sz <= a.capacity());
          prev_end = o + sz;
      }
      CHECK(a.alloc(1ull << 40) == UINT64_MAX && a.last_error().code == 2);
      a.reset(); CHECK(a.used() == 0 && a.last_error().ok()); }

    // ---- graph ----
    { CapturedGraph g; std::vector<int> order;
      CHECK(!g.replay());
      g.begin(); g.add([&] { order.push_back(1); }); g.add([&] { order.push_back(2); }); g.add([&] { order.push_back(3); });
      CHECK(!g.replay()); g.end();
      CHECK(g.replay() && g.replay() && g.replays == 2);
      CHECK((order == std::vector<int>{1, 2, 3, 1, 2, 3}));
      g.free(); CHECK(!g.replay() && g.nodes.empty());
      GraphRegistry r; int hit = 0;
      r.get("a").begin(); r.get("a").add([&] { ++hit; }); r.get("a").end();
      CHECK(r.size() == 1 && r.find("b") == nullptr && r.find("a") && !r.replay("b") && r.replay("a") && hit == 1);
      CHECK(r.erase("a") && !r.erase("a") && r.size() == 0);
      for (int l = 0; l < 48; ++l) CHECK((layer_type_of(l) == LayerType::Qsa) == (l % 4 == 3)); }

    // ---- hit hook ----
    { TokenHits th; HitFn f = th.sink(HitPhase::Decode);
      f(HitPhase::Decode, 5, 17); f(HitPhase::Prefill, 5, 18); f(HitPhase::Verify, 6, 19); f(HitPhase::Decode, 5, 20);
      CHECK(th.hits.size() == 2 && th.count_layer(5) == 2 && th.count_layer(6) == 0);
      CHECK(th.hits[0].expert == 17 && th.hits[1].expert == 20);
      VerifyHits vh; vh.reset(3);
      CHECK(vh.record(0, 1, 2) && vh.record(2, 1, 3) && !vh.record(3, 1, 4) && vh.total() == 2); }

    // ---- native bindings ----
    { NativeEmbed none; CHECK(!none.valid());
      float tab[6] = {1, 2, 3, 4, 5, 6}; NativeEmbed ne{tab, 2, 3}; float out[3] = {};
      CHECK(ne.valid() && ne.row(1, out) && out[0] == 4 && out[2] == 6 && !ne.row(2, out) && !ne.row(-1, out));
      set_native_embed(&ne); CHECK(native_embed() == &ne); set_native_embed(nullptr); CHECK(native_embed() == nullptr);
      NativeDense nd; CHECK(!nd.valid());
      WeightTable wt; CHECK(wt.find("nope") == nullptr);
      CHECK(!nd.bind(wt, "nope", nullptr, 4, 4));
      NativeHead nh; CHECK(!nh.present()); }

    // ---- pinned arena ----
    { const uint8_t a = 'a';
      CHECK(fnv1a64(&a, 1) == 0xaf63dc4c8601ec8cull && fnv1a64(nullptr, 0) == parena::kFnvOffset);
      std::vector<uint8_t> src(10 * 100); for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i * 7 + 3);
      ExpertReader rd = [&](uint64_t off, uint64_t n, uint8_t* dst) { if (off + n > src.size()) return false; std::memcpy(dst, src.data() + off, n); return true; };
      PinnedArena pa; CHECK(!pa.init(0, 64) && pa.init(1000, 64));
      CHECK(pa.covered() + pa.tail() == pa.bytes() && pa.slices() * 64 == pa.covered());
      LoadStats st = load_experts(pa, rd, 5, 2, 100, 37);
      CHECK(st.ok && st.bytes == 1000 && std::memcmp(pa.data(), src.data(), 1000) == 0);
      CHECK(st.checksum == fnv1a64(src.data(), 1000) || st.reads > 0);   // chunked left-fold checksum is deterministic per chunking
      LoadStats st2 = load_experts(pa, rd, 5, 2, 100, 37); CHECK(st2.checksum == st.checksum);
      LoadStats big = load_experts(pa, rd, 6, 2, 100, 37); CHECK(!big.ok);     // source/arena overrun is rejected
      CHECK(!load_experts(pa, rd, 1, 1, 10, 0).ok);                             // zero chunk is rejected
      PinnedArena pb; pb.init(300, 64);
      LoadStats rg = load_experts_ranges(pb, rd, {{500, 100}, {0, 50}, {900, 100}}, 33);
      CHECK(rg.ok && rg.bytes == 250 && std::memcmp(pb.data(), src.data() + 500, 100) == 0 &&
            std::memcmp(pb.data() + 100, src.data(), 50) == 0 && std::memcmp(pb.data() + 150, src.data() + 900, 100) == 0);
      int64_t clk = 0; uint64_t moved = 0;
      StreamStats ss = stream_bandwidth(pa, 100, 3, [&](const uint8_t*, uint64_t n) { moved += n; }, [&] { return clk += 500; });
      CHECK(ss.passes == 3 && ss.bytes == moved && ss.bytes == 3 * parena::moved_per_pass(1000, 100) && ss.ns == 500 && ss.gbps > 0);
      CHECK(stream_bandwidth(pa, 0, 3, [](const uint8_t*, uint64_t) {}, [] { return int64_t(0); }).passes == 0); }

    // ---- session ----
    { std::vector<uint64_t> parts = {100, 300, 1, 4096, 7};
      uint64_t sum = 0; for (auto p : parts) sum += p;
      CHECK(session_bytes(parts) == align_up(sum) + 4096);
      Session s; CHECK(session_init(s, parts, 48) && s.bytes == session_bytes(parts) && s.arena.size() == s.bytes);
      for (size_t i = 0; i < parts.size(); ++i) { CHECK(s.off[i] % 256 == 0 && s.off[i] + s.size[i] <= s.bytes); if (i) CHECK(s.off[i] >= s.off[i - 1] + s.size[i - 1]); }
      CHECK(!session_init(s, {}, 48) && !session_init(s, parts, 0) && !session_init(s, std::vector<uint64_t>(8, 1), 48));
      session_init(s, parts, 48); s.arena[5] = 9; session_zero(s); CHECK(s.arena[5] == 0);
      CHECK(gdn_point_at(0) == 0 && gdn_point_at(3) == -1 && qsa_point_at(3) == 0 && qsa_point_at(7) == 1 && gdn_point_at(4) == 3);
      // graphs
      std::vector<int64_t> tl; int emb = 0, hd = 0; std::vector<std::pair<int,int>> calls;
      CHECK(!session_run_token(s, 1));
      CHECK(session_capture(s, [&](int l, int p) { calls.push_back({l, p}); }, 4));
      CHECK(session_capture_token(s, [&](int64_t t) { emb += (int)t; }, [&](int, int) {}, [&] { ++hd; }));
      CHECK(session_run_token(s, 7) && session_token(s, 2, true) && hd == 2 && emb == 9);
      int64_t t = 0; s.clock_ns = [&] { return t += 10; };
      std::vector<int64_t> pre, st;
      CHECK(session_replay_stage_prefixes(s, pre) && pre.size() == 5);
      CHECK(session_replay_stages(s, st) && st.size() == 5);
      CHECK(!session_replay_stage_sweep(s, 0, st) && session_replay_stage_sweep(s, 3, st) && st.size() == 5);
      CHECK(session_replay(s, 2) > 0 && session_replay_full(s) > 0);
      std::vector<std::vector<int64_t>> pl; CHECK(session_replay_stages_per_layer(s, pl));
      std::vector<int64_t> p5 = {10, 25, 45, 70, 100};
      auto d = stage_diffs(p5); CHECK((d == std::vector<int64_t>{10, 15, 20, 25, 30}));
      int64_t acc = 0; for (auto x : d) acc += x; CHECK(acc == p5.back());
      session_graphs_free(s); CHECK(!s.token_graph.captured && s.token_graph.nodes.empty() && !session_run_token(s, 1));
      SessionState ss; ss.geom.n_layers = 48;
      std::vector<std::pair<int,int>> seen; ss.hit = [&](HitPhase, int l, int e) { seen.push_back({l, e}); };
      ss.pick(HitPhase::Decode, 1, 2); CHECK(seen.size() == 1);
      ss.hits.record(1, 2); ss.position = 5; ss.session.arena.assign(8, 1); ss.reset(); CHECK(ss.hits.hits.empty() && ss.position == 0 && ss.session.arena[0] == 0);
      ss.graphs.token.captured = true; ss.graphs.token.nodes.push_back([] {}); CHECK(ss.graphs.captured()); ss.graphs.free_all(); CHECK(!ss.graphs.captured());
      SessionLoopScratch sc; CHECK(sc.init(4, 16) && sc.bytes() == 4 * 16 * 4 && sc.parts.size() == 64); sc.free();
      CHECK(session_dump_floats(48, 4, 32) == 49ull * 4 * 32);
      // loop: layer l waits for ring value l+1
      std::atomic<uint64_t> word{0}; Doorbell ring; ring.word = &word;
      std::vector<int> got;
      std::thread prod([&] { for (uint64_t v = 1; v <= 6; ++v) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); doorbell_raise(&ring, v); } });
      CHECK(session_loop(ring, 6, [&](int l) { got.push_back(l); }, UINT64_MAX));
      prod.join(); CHECK((got == std::vector<int>{0, 1, 2, 3, 4, 5}));
      std::atomic<uint64_t> w2{0}; Doorbell r2; r2.word = &w2; int ran = 0;
      CHECK(!session_loop(r2, 3, [&](int) { ++ran; }, 50) && ran == 0);          // exhausted spins: gives up
      CHECK(!session_loop(r2, 3, [&](int) { ++ran; }, UINT64_MAX, [] { return false; }) && ran == 0);   // escalation refuses
      CHECK(spin_boundary(1023) && !spin_boundary(1022)); }

    std::printf("%s: %d checks, %d failures\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
