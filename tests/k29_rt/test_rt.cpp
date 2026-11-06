// Unit pe1ab8a4555 tests: device arena / poison, pinned arena + checksummed loads, replay graphs, layout checks, dense
// bank / embedding / head, bf16 gemv kernel, human(). Needs a device. Built as CUDA (links the rt objects).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "baloon/core_weights.hpp"
#include "baloon/rt_device.hpp"
#include "baloon/rt_graph.hpp"
#include "baloon/rt_kernels.hpp"
#include "baloon/rt_layout.hpp"
#include "baloon/rt_native.hpp"
#include "baloon/rt_pinned.hpp"
#include "baloon/rt_testdata.hpp"

std::string human(uint64_t bytes);   // device_main.cpp (built with RT_DEVICE_MAIN_DISABLED)

using namespace baloon;
using namespace baloon::rt;

static long g_checks = 0, g_fail = 0;
#define T29(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

// ------------------------------------------------------------------------------------------------ pure helpers
static void test_fnv_bounds_human() {
    T29(fnv1a64(nullptr, 0) == kFnvBasis);
    const uint8_t a[] = {'a'}; T29(fnv1a64(a, 1) == 0xaf63dc4c8601ec8cull);
    const char* fb = "foobar"; T29(fnv1a64((const uint8_t*)fb, 6) == 0x85944171f73967e8ull);
    XorRng rng(5);
    for (int it = 0; it < 200; ++it) {   // left fold property
        std::vector<uint8_t> v(1 + rng.next() % 300); for (auto& b : v) b = (uint8_t)rng.next();
        const size_t cut = rng.next() % (v.size() + 1);
        T29(fnv1a64(v.data() + cut, v.size() - cut, fnv1a64(v.data(), cut)) == fnv1a64(v.data(), v.size()));
    }
    T29(uniform_bounds(10, 0).empty());
    for (int it = 0; it < 500; ++it) {
        const uint64_t total = rng.next() % 5000, parts = 1 + rng.next() % 40;
        auto b = uniform_bounds(total, parts);
        T29(b.size() == parts + 1 && b[0] == 0 && b[parts] == total);
        for (uint64_t i = 0; i < parts; ++i) { const uint64_t d = b[i + 1] - b[i]; T29(d == total / parts || d == (total + parts - 1) / parts); }
    }
    T29(uniform_bounds(7, 3) == (std::vector<uint64_t>{0, 3, 5, 7}));
    T29(human(0) == "0 B" && human(1023) == "1023 B" && human(1024) == "1.0 KiB" && human(1536) == "1.5 KiB");
    T29(human(1ull << 20) == "1.0 MiB" && human(32768ull << 20) == "32.0 GiB" && human(1ull << 40) == "1.0 TiB");
}

// ------------------------------------------------------------------------------------------------ layout
static LayoutIndex good_index(const LayoutGeometry& g) {
    LayoutIndex idx;
    for (int l = 0; l < g.n_layers; ++l) {
        LayerView v; v.layer = l;
        idx[v.name("attn_norm")] = {g.n_embd, 0}; idx[v.name("ffn_norm")] = {g.n_embd, 0};
        if (g.is_qsa(l)) {
            idx[v.name("attn_q")] = {g.n_head * g.head_dim, g.n_embd};
            idx[v.name("attn_k")] = {g.n_head_kv * g.head_dim, g.n_embd};
            idx[v.name("attn_v")] = {g.n_head_kv * g.head_dim, g.n_embd};
            idx[v.name("attn_output")] = {g.n_embd, g.n_head * g.head_dim};
        } else { idx[v.name("ssm_in")] = {3 * g.n_embd, g.n_embd}; idx[v.name("ssm_out")] = {g.n_embd, 3 * g.n_embd}; }
    }
    return idx;
}

static void test_layout() {
    LayoutGeometry g;                        // default 48-layer model
    int q = 0; for (int l = 0; l < g.n_layers; ++l) q += g.is_qsa(l);
    T29(q == 12 && g.is_qsa(3) && g.is_qsa(47) && !g.is_qsa(0) && !g.is_qsa(4) && !g.is_qsa(-1));
    LayerView v; v.layer = 7; T29(v.name("attn_q") == "blk.7.attn_q");
    LayoutIndex idx = good_index(g); std::string why;
    T29(check_all(idx, g, &why));
    T29(!check_layer(idx, g, -1, &why) && why == "layer out of range");
    why.clear(); T29(!check_layer(idx, g, g.n_layers, &why) && why == "layer out of range");
    { LayoutGeometry z = g; z.n_layers = 0; T29(!check_all(idx, z, &why) && why == "no layers"); }
    { LayoutGeometry z = g; z.head_dim = 0; T29(!check_layer(idx, z, 0, &why) && why == "bad geometry"); }
    auto expect_bad = [&](LayoutIndex m, const char* needle) {
        std::string w; const bool r = check_all(m, g, &w);
        T29(!r); T29(w.find(needle) != std::string::npos);
    };
    { auto m = idx; m.erase("blk.5.ffn_norm"); expect_bad(m, "missing blk.5.ffn_norm"); }
    { auto m = idx; m["blk.9.attn_norm"].rows += 1; expect_bad(m, "wrong length"); }
    { auto m = idx; m["blk.9.attn_norm"].cols = 2; expect_bad(m, "must be a vector"); }
    { auto m = idx; m["blk.3.attn_q"].rows -= 1; expect_bad(m, "wrong row count"); }
    { auto m = idx; m["blk.3.attn_k"].cols += 1; expect_bad(m, "wrong column count"); }
    { auto m = idx; m["blk.3.attn_v"].cols = 0; expect_bad(m, "must be a matrix"); }
    { auto m = idx; m["blk.3.attn_output"].rows = 5; expect_bad(m, "attn_output"); }
    { auto m = idx; m["blk.0.ssm_in"].cols += 1; expect_bad(m, "blk.0.ssm_in"); }
    { auto m = idx; m["blk.0.ssm_out"].rows += 1; expect_bad(m, "blk.0.ssm_out"); }
    { auto m = idx; m["blk.0.ssm_in"].rows = 0; expect_bad(m, "blk.0.ssm_in"); }
    { auto m = idx; m["blk.3.ssm_in"] = {1, 1}; expect_bad(m, "QSA layer carries"); }
    { auto m = idx; m["blk.2.attn_q"] = {1, 1}; expect_bad(m, "GDN layer carries"); }
    { auto m = idx; m.erase("blk.47.attn_output"); expect_bad(m, "missing blk.47.attn_output"); }
    // check_one directly: rows/cols of 0 in Want2 accept any positive extent, never a vector
    LayoutIndex one; one["blk.1.x"] = {6, 4}; one["blk.1.y"] = {6, 0};
    LayerView v1; v1.layer = 1;
    T29(check_one(one, v1, Want2{"x", 0, 0}) && check_one(one, v1, Want2{"x", 6, 4}) && !check_one(one, v1, Want2{"x", 4, 6}));
    T29(!check_one(one, v1, Want2{"y", 0, 0}) && check_one(one, v1, Want1{"y", 6}) && !check_one(one, v1, Want1{"x", 6}));
    T29(!check_one(one, v1, Want1{"nope", 1}));
    T29(!layout_fail(nullptr, "x"));
    std::string w; T29(!layout_fail(&w, "boom") && w == "boom");
}

// ------------------------------------------------------------------------------------------------ device arena
static void test_arena() {
    T29(device_count() >= 1);
    DeviceDesc d; std::string err;
    T29(query_device(0, &d, &err) && d.ordinal == 0 && !d.name.empty() && d.total_mem > 0 && d.warp > 0);
    err.clear(); T29(!query_device(-1, &d, &err) && err == "invalid device ordinal");
    err.clear(); T29(!query_device(device_count(), &d, &err) && err == "invalid device ordinal");
    T29(RT_CUDA(cudaSuccess) && !cuda_ok(cudaErrorInvalidValue, "t", "f", 1));
    cudaGetLastError();

    DevArena a(0, 1 << 20);
    T29(a.ok() && a.capacity() == (1u << 20) && a.used() == 0);
    std::vector<std::pair<uint64_t, uint64_t>> got;
    XorRng rng(9);
    for (int i = 0; i < 40; ++i) {
        const uint64_t n = 1 + rng.next() % 5000;
        void* p = a.alloc(n);
        if (!p) break;
        const uint64_t off = (uint8_t*)p - a.base();
        T29(off % kArenaAlign == 0);
        for (auto& g : got) T29(off >= g.first + g.second || off + n <= g.first);   // no overlap
        T29(off + n <= a.capacity());
        got.push_back({off, n});
    }
    T29(got.size() >= 20);
    DevArena b(0, 4096);
    T29(b.alloc(4097) == nullptr);                       // bigger than capacity
    T29(b.alloc(UINT64_MAX) == nullptr && b.alloc(UINT64_MAX - 100) == nullptr);   // overflow-safe
    T29(b.alloc(1) != nullptr);                          // used = 1
    T29(b.alloc(UINT64_MAX - 100) == nullptr);           // used_ != 0: at + bytes would wrap
    T29(b.alloc(4096 - 256) != nullptr);                 // exactly fills [256, 4096)
    T29(b.alloc(1) == nullptr);                          // nothing left
    T29(b.used() == 4096);
    b.reset(); T29(b.used() == 0 && b.alloc(4096) == b.base());
    // reset(poison) overwrites the whole block
    DevArena c(0, 8192); void* p = c.alloc(8192);
    std::vector<uint8_t> pat(8192, 0x37); cudaMemcpy(p, pat.data(), 8192, cudaMemcpyHostToDevice);
    c.reset(true);
    std::vector<uint8_t> back(8192); cudaMemcpy(back.data(), c.base(), 8192, cudaMemcpyDeviceToHost);
    bool all = true; for (uint8_t x : back) all = all && x == DevArena::kPoisonByte; T29(all);
    // poison_fill honours n exactly and handles odd sizes
    uint8_t* q = (uint8_t*)c.base();
    cudaMemset(q, 0, 8192);
    T29(poison_fill(q + 3, 1001, 0xA5)); cudaDeviceSynchronize();
    cudaMemcpy(back.data(), q, 8192, cudaMemcpyDeviceToHost);
    bool pat_ok = true; for (int i = 0; i < 8192; ++i) pat_ok = pat_ok && back[i] == ((i >= 3 && i < 1004) ? 0xA5 : 0x00); T29(pat_ok);
    T29(poison_fill(nullptr, 0, 1) && !poison_fill(nullptr, 5, 1));
    DevArena bad(0, 0); T29(!bad.ok() && bad.alloc(1) == nullptr);
    DevArena bad2(device_count() + 3, 4096); T29(!bad2.ok());
    cudaGetLastError();
}

// ------------------------------------------------------------------------------------------------ pinned arena
static void test_pinned() {
    PinArena a;
    T29(!a.reserve(0) && a.backing() == Backing::None && a.data() == nullptr);
    T29(a.reserve(1000003, 7));
    T29(a.bytes() == 1000003 && a.slices() == 7 && a.registered_bytes() <= a.bytes() && a.backing() != Backing::None);
    bool zero = true; for (uint64_t i = 0; i < a.bytes(); i += 4099) zero = zero && a.data()[i] == 0; T29(zero);
    std::memset(a.data(), 0x5A, a.bytes());
    a.release(); T29(a.data() == nullptr && a.bytes() == 0 && a.slices() == 0 && a.backing() == Backing::None);
    T29(a.reserve(10, 100) && a.slices() == 1);           // never more slices than pages
    T29(a.reserve(4ull << 20, 0) && a.slices() == 1);     // 0 slices behaves as 1
    a.release();

    // checksummed loads: source is a deterministic byte function
    auto src = [](uint64_t i) { return (uint8_t)(i * 2654435761ull >> 13); };
    long calls = 0;
    Reader rd = [&](uint64_t off, uint64_t n, uint8_t* dst) { ++calls; for (uint64_t i = 0; i < n; ++i) dst[i] = src(off + i); return true; };
    PinArena b; T29(b.reserve(5 * 3 * 100, 3));
    LoadReport r = load_experts(b, rd, 5, 3, 100, 64);
    T29(r.ok && r.bytes == 1500);
    T29(r.reads == 5 * ((300 + 63) / 64) && calls == (long)r.reads);
    std::vector<uint8_t> all(1500); for (uint64_t i = 0; i < 1500; ++i) all[i] = src(i);
    T29(std::memcmp(all.data(), b.data(), 1500) == 0);
    T29(r.checksum == fnv1a64(all.data(), 1500));
    T29(!load_experts(b, rd, 6, 3, 100, 64).ok);                     // would run past the arena
    T29(!load_experts(b, rd, 5, 3, 100, 0).ok);                      // chunk 0
    T29(!load_experts(b, rd, 2, UINT64_MAX / 2, 4, 64).ok);          // product overflows
    T29(!load_experts(b, rd, UINT64_MAX / 2, 3, 100, 64).ok);        // layer offset overflows
    { long n = 0; Reader bad = [&](uint64_t, uint64_t m, uint8_t* d) { std::memset(d, 1, m); return ++n < 3; };
      LoadReport f = load_experts(b, bad, 5, 3, 100, 64); T29(!f.ok && f.reads == 2); }   // stops at the first failing read
    T29(load_experts(b, rd, 0, 3, 100, 64).ok && load_experts(b, rd, 0, 3, 100, 64).bytes == 0);
    std::vector<std::pair<uint64_t, uint64_t>> ranges = {{1000, 10}, {5, 300}, {70, 0}, {9000, 77}};
    LoadReport rr = load_experts_ranges(b, rd, ranges, 50);
    T29(rr.ok && rr.bytes == 387);
    std::vector<uint8_t> want; for (auto& g : ranges) for (uint64_t i = 0; i < g.second; ++i) want.push_back(src(g.first + i));
    T29(std::memcmp(want.data(), b.data(), want.size()) == 0 && rr.checksum == fnv1a64(want.data(), want.size()));
    T29(!load_experts_ranges(b, rd, {{0, 1000}, {0, 1000}}, 64).ok);  // second range overflows the arena
    T29(!load_experts_ranges(b, rd, {{0, 10}}, 0).ok);

    // streaming to the device moves exactly the whole chunks and lands byte for byte
    for (uint64_t i = 0; i < b.bytes(); ++i) b.data()[i] = src(i + 17);
    DevArena dev(0, 4096);
    uint8_t* d = (uint8_t*)dev.alloc(1500); T29(d != nullptr);
    cudaMemset(d, 0xEE, 1500);
    BandwidthReport bw = stream_bandwidth(b, d, 400, 3);
    T29(bw.ok && bw.passes == 3 && bw.bytes == 3 * 1200 && bw.ms >= 0);
    std::vector<uint8_t> back(1500); cudaMemcpy(back.data(), d, 1500, cudaMemcpyDeviceToHost);
    bool land = true; for (int i = 0; i < 1500; ++i) land = land && back[i] == (i < 1200 ? src(i + 17) : 0xEE); T29(land);
    T29(!stream_bandwidth(b, d, 0, 1).ok && !stream_bandwidth(b, nullptr, 8, 1).ok);
    T29(clear_error() == 0);
}

// ------------------------------------------------------------------------------------------------ replay graphs
static void test_graph() {
    uint8_t* d = nullptr; cudaMalloc((void**)&d, 64);
    ReplayGraph g;
    T29(!g.ready() && !g.launch() && g.wait_ms() < 0 && !g.push([](cudaStream_t) { return cudaSuccess; }) && !g.end());
    cudaStream_t s; cudaStreamCreate(&s);
    g.begin();
    T29(!g.push(nullptr));
    g.push([&](cudaStream_t st) { return cudaMemsetAsync(d, 0x11, 64, st); });
    g.push([&](cudaStream_t st) { return cudaMemsetAsync(d, 0x22, 32, st); });
    T29(!g.ready() && !g.launch(s));                        // not ready before end()
    T29(g.end() && g.ready() && g.size() == 2 && !g.end());
    T29(g.launch(s) && g.wait_ms() >= 0 && g.launches() == 1);
    uint8_t h[64]; cudaMemcpy(h, d, 64, cudaMemcpyDeviceToHost);
    bool order = true; for (int i = 0; i < 64; ++i) order = order && h[i] == (i < 32 ? 0x22 : 0x11); T29(order);   // recorded order
    cudaMemset(d, 0, 64);
    T29(g.launch(s) && g.launch(s) && g.launches() == 3 && g.wait_ms() >= 0);   // replayable
    cudaMemcpy(h, d, 64, cudaMemcpyDeviceToHost); T29(h[0] == 0x22 && h[63] == 0x11);
    g.begin(); T29(!g.ready() && g.wait_ms() < 0);          // a fresh recording has no finished launch to time
    g.push([&](cudaStream_t st) { return cudaMemsetAsync(d, 0x11, 64, st); });
    g.end();
    // a failing op stops the replay
    int ran = 0; g.begin();
    g.push([&](cudaStream_t) { ++ran; return cudaSuccess; });
    g.push([&](cudaStream_t) { ++ran; return cudaErrorInvalidValue; });
    g.push([&](cudaStream_t) { ++ran; return cudaSuccess; });
    g.end();
    T29(!g.launch(s) && ran == 2 && g.wait_ms() < 0);
    cudaGetLastError();
    g.begin(); T29(!g.ready() && g.size() == 0);            // begin drops the old recording
    g.end(); g.reset(); T29(!g.ready() && g.size() == 0 && !g.launch(s));

    GraphBook book; int count = 0;
    T29(book.size() == 0 && book.find("a") == nullptr && !book.launch("a"));
    T29(book.record("a", [&](ReplayGraph& r) { r.push([&](cudaStream_t) { ++count; return cudaSuccess; }); }));
    T29(book.size() == 1 && book.find("a") && book.find("a")->ready() && book.find("b") == nullptr);
    T29(book.launch("a", s) && book.launch("a", s) && count == 2);
    T29(book.record("a", [&](ReplayGraph& r) { r.push([&](cudaStream_t) { count += 10; return cudaSuccess; }); }));
    T29(book.size() == 1 && book.launch("a", s) && count == 12);   // replaced, not duplicated
    T29(book.record("b", [](ReplayGraph&) {}) && book.size() == 2 && book.launch("b", s));
    cudaStreamSynchronize(s); cudaStreamDestroy(s); cudaFree(d);
    T29(!graph_fail("expected test message"));
}

// ------------------------------------------------------------------------------------------------ kernels + native
static void test_kernels() {
    XorRng rng(21);
    const int64_t shapes[][2] = {{1, 1}, {3, 255}, {5, 256}, {7, 257}, {16, 1000}, {33, 4096}, {2, 3}};
    for (auto& sh : shapes) {
        const int64_t rows = sh[0], cols = sh[1];
        auto w = make_bf16(rng, rows * cols); auto x = make_f32(rng, cols);
        uint16_t* dw; float *dx, *dy;
        cudaMalloc((void**)&dw, w.size() * 2); cudaMalloc((void**)&dx, x.size() * 4); cudaMalloc((void**)&dy, rows * 4);
        cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        T29(bf16_gemv(dw, dx, rows, cols, dy));
        std::vector<float> y(rows); cudaMemcpy(y.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
        for (int64_t r = 0; r < rows; ++r) {
            double sabs = 0; for (int64_t c = 0; c < cols; ++c) sabs += std::fabs((double)bf16_to_f32(w[r * cols + c]) * x[c]);
            T29(std::fabs((double)y[r] - gemv_row_ref(&w[r * cols], x.data(), cols)) <= 1e-6 + 1e-5 * sabs);
        }
        T29(bf16_gemv(dw, dx, rows, cols, dy) && true);
        std::vector<float> y2(rows); cudaMemcpy(y2.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
        T29(std::memcmp(y.data(), y2.data(), rows * 4) == 0);            // deterministic
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }
    float f; T29(!bf16_gemv(nullptr, &f, 1, 1, &f) && !bf16_gemv((const uint16_t*)&f, &f, 0, 1, &f) && !bf16_gemv((const uint16_t*)&f, &f, 1, 0, &f));
    T29(bf16_to_f32(0x3F80) == 1.0f && bf16_to_f32(0xC000) == -2.0f && f32_to_bf16_trunc(1.0f) == 0x3F80 && f32_to_bf16_trunc(-2.0f) == 0xC000);
}

static void test_native() {
    // eligible()
    core::WIndexRow r; r.placed = true; r.engine_kind = 1; r.dst_bytes = 8;
    T29(eligible(r));
    { auto m = r; m.placed = false; T29(!eligible(m)); }
    { auto m = r; m.engine_kind = 3; T29(!eligible(m)); }
    { auto m = r; m.engine_kind = 0; T29(!eligible(m)); }
    { auto m = r; m.dst_bytes = 0; T29(!eligible(m)); }
    { auto m = r; m.dst_bytes = 7; T29(!eligible(m)); }

    XorRng rng(31);
    const int64_t R1 = 6, C1 = 40, R2 = 3, C2 = 17;
    auto w1 = make_bf16(rng, R1 * C1), w2 = make_bf16(rng, R2 * C2);
    std::vector<uint8_t> src; src.insert(src.end(), (uint8_t*)w1.data(), (uint8_t*)(w1.data() + w1.size()));
    const uint64_t off2 = src.size(); src.insert(src.end(), (uint8_t*)w2.data(), (uint8_t*)(w2.data() + w2.size()));
    std::vector<float> junk = make_f32(rng, 8); const uint64_t off3 = src.size();
    src.insert(src.end(), (uint8_t*)junk.data(), (uint8_t*)(junk.data() + junk.size()));
    std::string index = "zeta.w 4 0 " + std::to_string(w1.size() * 2) + " " + std::to_string(w1.size() * 2) + " 0 0 0 0\n" +
                        "alpha.w 4 " + std::to_string(off2) + " " + std::to_string(w2.size() * 2) + " " + std::to_string(w2.size() * 2) + " 0 0 0 0\n" +
                        "other.f32 0 " + std::to_string(off3) + " 32 32 0 0 0 0\n";
    core::WReader rd = [&](uint64_t off, uint64_t n, uint8_t* buf) { if (off + n > src.size()) return false; std::memcpy(buf, src.data() + off, n); return true; };
    core::WeightTable tab; std::string err;
    core::WeightTable placer; T29(placer.place(index, {}, &err));
    std::vector<uint8_t> arena(placer.pool_bytes() ? placer.pool_bytes() : 1);
    T29(tab.load_from(index, rd, arena.data(), arena.size(), {}, &err));
    ShapeMap shapes; shapes["zeta.w"] = {R1, C1}; shapes["alpha.w"] = {R2, C2}; shapes["other.f32"] = {4, 2}; shapes["ghost"] = {1, 1};
    DenseBank bank;
    T29(bank.load(tab, arena.data(), shapes, 0, &err));
    T29((bank.served_names() == std::vector<std::string>{"alpha.w", "zeta.w"}));           // sorted; f32 + unknown rows skipped
    T29(bank.resident_bytes() == (w1.size() + w2.size()) * 2 && bank.ordinal() == 0);
    DenseView v; T29(bank.find("zeta.w", &v) && v.rows == (uint64_t)R1 && v.cols == (uint64_t)C1 && !bank.find("other.f32", &v) && !bank.find("ghost", &v));
    auto x = make_f32(rng, C1);
    float *dx, *dy; cudaMalloc((void**)&dx, C1 * 4); cudaMalloc((void**)&dy, R1 * 4);
    cudaMemcpy(dx, x.data(), C1 * 4, cudaMemcpyHostToDevice);
    T29(bank.apply("zeta.w", dx, dy));
    std::vector<float> y(R1); cudaMemcpy(y.data(), dy, R1 * 4, cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < R1; ++i) T29(std::fabs((double)y[i] - gemv_row_ref(&w1[i * C1], x.data(), C1)) < 1e-4);
    T29(!bank.apply("nope", dx, dy));
    cudaFree(dx); cudaFree(dy);
    ShapeMap wrong = shapes; wrong["zeta.w"] = {R1, C1 + 1};
    T29(!bank.load(tab, arena.data(), wrong, 0, &err) && err.find("zeta.w") != std::string::npos);
    T29(bank.served_names().empty() && bank.resident_bytes() == 0);                         // a failed load leaves nothing resident
    ShapeMap zero = shapes; zero["zeta.w"] = {0, C1}; T29(!bank.load(tab, arena.data(), zero, 0, &err));
    T29(!bank.load(tab, nullptr, shapes, 0, &err));
    T29(!bank.load(tab, arena.data(), shapes, device_count() + 2, &err) && bank.served_names().empty());
    cudaGetLastError();

    // embedding table
    const int64_t V = 50, N = 24; auto tbl = make_f32(rng, V * N);
    EmbedTable emb; T29(!emb.valid() && !emb.gather_one(0, nullptr));
    T29(emb.load(tbl.data(), V, N, 0, &err) && emb.valid() && emb.vocab() == V && emb.width() == N);
    std::vector<float> row(N);
    T29(emb.gather_one(0, row.data()) && std::memcmp(row.data(), tbl.data(), N * 4) == 0);
    T29(emb.gather_one(V - 1, row.data()) && std::memcmp(row.data(), tbl.data() + (V - 1) * N, N * 4) == 0);
    T29(!emb.gather_one(V, row.data()) && !emb.gather_one(-1, row.data()));
    int32_t toks[] = {3, 49, -1, 50, 0, 7}; const int cnt = 6;
    int32_t* dt; float* dout; cudaMalloc((void**)&dt, sizeof toks); cudaMalloc((void**)&dout, cnt * N * 4);
    cudaMemcpy(dt, toks, sizeof toks, cudaMemcpyHostToDevice);
    T29(emb.gather_dev(dt, cnt, dout));
    std::vector<float> got(cnt * N); cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < cnt; ++i) {
        const bool valid = toks[i] >= 0 && toks[i] < V;
        for (int c = 0; c < N; ++c) T29(got[i * N + c] == (valid ? tbl[(int64_t)toks[i] * N + c] : 0.f));
    }
    T29(emb.gather_dev(dt, 0, dout) && !emb.gather_dev(nullptr, 2, dout));
    cudaFree(dt); cudaFree(dout);
    T29(!emb.load(nullptr, V, N, 0, &err) && !emb.valid());
    T29(!emb.load(tbl.data(), 0, N, 0, &err) && !emb.load(tbl.data(), INT64_MAX, 4, 0, &err));
    // process-wide binding is cleared when the bound table dies
    T29(native_embed() == nullptr);
    {
        EmbedTable tmp; T29(tmp.load(tbl.data(), V, N, 0, &err));
        set_native_embed(&tmp); T29(native_embed() == &tmp);
    }
    T29(native_embed() == nullptr);
    set_native_embed(&emb); T29(native_embed() == &emb); set_native_embed(nullptr); T29(native_embed() == nullptr);
    cudaGetLastError();

    // head projection
    const int64_t HV = 37, HN = 90; auto hw = make_bf16(rng, HV * HN); auto hx = make_f32(rng, HN);
    HeadProj head; std::vector<float> logits(HV);
    T29(!head.valid() && !head.run(hx.data(), logits.data()));
    T29(head.load(hw.data(), HV, HN, 0, &err) && head.valid() && head.vocab() == HV && head.width() == HN);
    T29(head.run(hx.data(), logits.data()));
    for (int64_t i = 0; i < HV; ++i) T29(std::fabs((double)logits[i] - gemv_row_ref(&hw[i * HN], hx.data(), HN)) < 1e-4);
    T29(!head.run(nullptr, logits.data()) && !head.run(hx.data(), nullptr));
    T29(!head.load(nullptr, HV, HN, 0, &err) && !head.valid());
    T29(!head.load(hw.data(), HV, 0, 0, &err));
    cudaGetLastError();
}

int main() {
    test_fnv_bounds_human();
    test_layout();
    test_arena();
    test_pinned();
    test_graph();
    test_kernels();
    test_native();
    std::printf("checks: %ld, failures: %ld\n", g_checks, g_fail);
    if (g_fail) return 1;
    std::printf("UNIT 29 TESTS PASSED\n");
    return 0;
}
