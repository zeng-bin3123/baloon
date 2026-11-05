// Unit 18 invariants: device arena, graph registry, layer layout, native bindings, pinned loads, session state.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "baloon/core_device.hpp"
#include "baloon/core_graph.hpp"
#include "baloon/core_hit_hook.hpp"
#include "baloon/core_layout.hpp"
#include "baloon/core_native.hpp"
#include "baloon/core_pinned.hpp"
#include "baloon/core_session_state.hpp"
using namespace baloon::core;
namespace parena = baloon::parena;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // device
    CudaError e; DeviceInfo d = device_info(2, 4, &e);
    CHECK(d.valid && d.ordinal == 2 && d.wave == 64 && d.total_mem == 32768ull * 1048576 && e.ok());
    CHECK(!device_info(4, 4, &e).valid && e.code == 101 && !e.ok());
    CHECK(!device_info(-1, 4).valid);
    CHECK(!device_info(0, 0, &e).valid && e.code == 100);
    CHECK(std::strcmp(CudaError{0}.name(), "success") == 0 && std::strcmp(CudaError{2}.name(), "out of memory") == 0);
    DeviceArena a; a.init(1000);
    uint64_t o1 = a.alloc(10), o2 = a.alloc(300), o3 = a.alloc(10);
    CHECK(o1 == 0 && o2 == 256 && o3 == 768 && o1 % 256 == 0 && o2 % 256 == 0 && o3 % 256 == 0);
    CHECK(o2 >= o1 + 10 && o3 >= o2 + 300);
    CHECK(a.alloc(1000) == UINT64_MAX && a.last_error().code == 2);
    CHECK(a.alloc(1) == UINT64_MAX);  // the next aligned offset (1024) lies past the capacity
    CHECK(a.used() <= a.capacity());
    { DeviceArena b; b.init(1000); CHECK(b.alloc(10) == 0); CHECK(b.alloc(900) == UINT64_MAX && b.used() == 10); }  // 256 + 900 > 1000
    a.reset(); CHECK(a.used() == 0 && a.alloc(1000) == 0 && a.alloc(1) == UINT64_MAX);
    CHECK(a.ptr(0) != nullptr && a.ptr(1000) == nullptr);

    // graph
    CHECK(layer_type_of(3) == LayerType::Qsa && layer_type_of(47) == LayerType::Qsa && layer_type_of(0) == LayerType::Gdn && layer_type_of(4) == LayerType::Gdn);
    CHECK(std::strcmp(to_string(LayerType::Qsa), "QSA") == 0 && std::strcmp(to_string(LayerType::Gdn), "GDN") == 0);
    CapturedGraph g; int order = 0, a1 = -1, a2 = -1;
    CHECK(!g.replay());
    g.begin(); g.add([&] { a1 = order++; }); g.add([&] { a2 = order++; });
    CHECK(!g.replay()); g.end();
    CHECK(g.replay() && a1 == 0 && a2 == 1 && g.replays == 1);
    g.free(); CHECK(!g.captured && !g.replay());
    GraphRegistry reg; int hits = 0;
    reg.get("x").begin(); reg.get("x").add([&] { ++hits; }); reg.get("x").end();
    CHECK(reg.size() == 1 && reg.replay("x") && hits == 1 && !reg.replay("y") && reg.find("y") == nullptr);
    CHECK(reg.erase("x") && !reg.replay("x") && !reg.erase("x"));

    // layout: 36 GDN + 12 QSA, dense unique points
    ModelGeometry geom;
    CHECK(geom.n_qsa() == 12 && geom.n_gdn() == 36);
    std::string why; CHECK(check_all(geom, &why));
    for (int l = 0; l < 48; ++l) {
        LayerView v = layer_view(geom, l);
        CHECK(check_layer(geom, l));
        CHECK((v.type == LayerType::Qsa) == is_qsa_layer(l));
        CHECK(v.point == (is_qsa_layer(l) ? l / 4 : l - l / 4));
        CHECK((v.state_floats == gdn_state_floats()) == (v.type == LayerType::Gdn));
    }
    CHECK(layer_view(geom, 3).point == 0 && layer_view(geom, 7).point == 1 && layer_view(geom, 4).point == 3);
    CHECK(!check_layer(geom, -1) && !check_layer(geom, 48));
    ModelGeometry bad = geom; bad.n_head = 15;           // not a multiple of n_head_kv
    CHECK(!check_all(bad) && !check_layer(bad, 3) && check_layer(bad, 0));
    ModelGeometry none; none.n_layers = 0; CHECK(!check_all(none));
    ModelGeometry small; small.n_layers = 8; CHECK(check_all(small) && small.n_qsa() == 2 && small.n_gdn() == 6);

    // native dense / embed
    std::vector<uint8_t> arena(64, 0);
    uint16_t* w = reinterpret_cast<uint16_t*>(arena.data());
    for (int i = 0; i < 6; ++i) w[i] = 0x3F80;           // bf16 1.0
    WeightTable t;
    CHECK(t.place("wq 4 0 12 12 0 0 0 0\nwk 3 12 12 12 0 0 0 0\n"));
    NativeDense nd;
    CHECK(nd.bind(t, "wq", arena.data(), 2, 3) && nd.valid());
    float x[3] = {1, 2, 3}, out[2] = {0, 0};
    CHECK(nd.apply(x, out) && out[0] == 6.f && out[1] == 6.f);
    CHECK(!nd.bind(t, "wq", arena.data(), 2, 4) && !nd.valid());      // size mismatch
    CHECK(!nd.bind(t, "missing", arena.data(), 2, 3));
    CHECK(!nd.bind(t, "wk", arena.data(), 2, 3));                      // kind 3 is F16, not BF16
    CHECK(!nd.apply(x, out));
    float tab[6] = {0, 1, 2, 10, 11, 12}; NativeEmbed ne{tab, 2, 3}; float r[3];
    CHECK(ne.row(1, r) && r[0] == 10 && r[2] == 12 && !ne.row(2, r) && !ne.row(-1, r));
    CHECK(native_embed() == nullptr); set_native_embed(&ne); CHECK(native_embed() == &ne); set_native_embed(nullptr); CHECK(native_embed() == nullptr);

    // pinned
    const uint64_t chunk = 7, blob = 5, bpl = 3, layers = 4, total = blob * bpl * layers;
    std::vector<uint8_t> src(total); for (uint64_t i = 0; i < total; ++i) src[i] = (uint8_t)(i * 31 + 7);
    ExpertReader rd = [&](uint64_t off, uint64_t n, uint8_t* dst) { if (off + n > src.size()) return false; std::memcpy(dst, src.data() + off, n); return true; };
    PinnedArena pa; CHECK(pa.init(total, 16) && !pa.init(0, 16));
    CHECK(pa.init(total, 16) && pa.slices() == 3 && pa.covered() == 48 && pa.tail() == 12 && pa.covered() + pa.tail() == pa.bytes());
    LoadStats ls = load_experts(pa, rd, layers, bpl, blob, chunk);
    CHECK(ls.ok && ls.bytes == total && std::memcmp(pa.data(), src.data(), total) == 0);
    CHECK(ls.reads == 4 * ((15 + chunk - 1) / chunk));
    CHECK(ls.checksum == parena::fnv1a64_chunked(src.data(), total, 1000) && ls.checksum == fnv1a64(src.data(), total));
    LoadStats l2 = load_experts(pa, rd, layers + 1, bpl, blob, chunk);      // reads past the source
    CHECK(!l2.ok);
    LoadStats lr = load_experts_ranges(pa, rd, {{10, 8}, {40, 5}}, 3);
    CHECK(lr.ok && lr.bytes == 13 && std::memcmp(pa.data(), src.data() + 10, 8) == 0 && std::memcmp(pa.data() + 8, src.data() + 40, 5) == 0);
    CHECK(!load_experts_ranges(pa, rd, {{0, total + 1}}, 3).ok && !load_experts_ranges(pa, rd, {{0, 1}}, 0).ok);
    { PinnedArena sm; sm.init(8, 4); ExpertReader any = [](uint64_t, uint64_t n, uint8_t* d) { std::memset(d, 1, n); return true; };
      LoadStats lo = load_experts_ranges(sm, any, {{0, 16}}, 4); CHECK(!lo.ok && lo.bytes == 0);
      LoadStats lo2 = load_experts_ranges(sm, any, {{0, 8}}, 4); CHECK(lo2.ok && lo2.bytes == 8); }
    int64_t now = 0; uint64_t copied = 0;
    StreamStats ss = stream_bandwidth(pa, 8, 3, [&](const uint8_t*, uint64_t n) { copied += n; now += 100; }, [&] { return now; });
    CHECK(ss.passes == 3 && ss.bytes == 3 * (total / 8 * 8) && copied == ss.bytes && ss.ns == (int64_t)(ss.bytes / 8 * 100));
    CHECK(ss.gbps > 0.07 && ss.gbps < 0.09);
    CHECK(stream_bandwidth(pa, 0, 3, [](const uint8_t*, uint64_t) {}, [] { return int64_t(0); }).passes == 0);

    // hits
    TokenHits th; HitFn f = th.sink();
    f(HitPhase::Decode, 3, 9); f(HitPhase::Prefill, 4, 1); f(HitPhase::Decode, 3, 2);
    CHECK(th.hits.size() == 2 && th.count_layer(3) == 2 && th.count_layer(4) == 0);
    VerifyHits vh; vh.reset(3);
    CHECK(vh.record(0, 1, 1) && vh.record(2, 1, 2) && !vh.record(3, 1, 1) && vh.total() == 2);

    // session state: arena carving, graphs, replay order
    SessionState ss2;
    CHECK(ss2.init({100, 300, 256}) && ss2.session.bytes == session_bytes({100, 300, 256}));
    CHECK(ss2.session.off[1] == 256 && ss2.session.off[2] == 768);
    ss2.geom.n_head = 15; SessionState bad2; bad2.geom.n_head = 15; CHECK(!bad2.init({16}));
    std::vector<int> trace;
    CHECK(session_capture_token(ss2.session, [&](int64_t) { trace.push_back(-1); }, [&](int l, int) { trace.push_back(l); }, [&] { trace.push_back(99); }));
    CHECK(session_run_token(ss2.session, 5) && trace.size() == 50 && trace.front() == -1 && trace.back() == 99 && trace[1] == 0 && trace[48] == 47);
    CHECK(ss2.session.token == 5);
    int picks = 0; ss2.hit = [&](HitPhase, int, int) { ++picks; };
    ss2.pick(HitPhase::Verify, 0, 0); CHECK(picks == 1);
    SessionGraphs sg; sg.token = ss2.session.token_graph; CHECK(sg.captured()); sg.free_all(); CHECK(!sg.captured());
    sg.prefix[2].captured = true; sg.prefix[2].nodes.push_back([] {}); sg.free_all();
    for (auto& pg : sg.prefix) CHECK(!pg.captured && pg.nodes.empty());
    ss2.hits.record(1, 1); ss2.position = 4; ss2.reset(); CHECK(ss2.hits.hits.empty() && ss2.position == 0);
    PoolFn pf = [](int, const int32_t*, int k, float* o) { o[0] = (float)k; return true; };
    PoolMultiFn pm = [](int, const int32_t*, int rows, int k, float* o) { o[0] = (float)(rows * k); return true; };
    float po = 0; int32_t ex[2] = {1, 2}; CHECK(pf(0, ex, 2, &po) && po == 2 && pm(0, ex, 3, 2, &po) && po == 6);

    if (fails) { std::printf("%d CHECK(S) FAILED\n", fails); return 1; }
    std::printf("CORE UNIT 18 TESTS PASSED\n");
    return 0;
}
