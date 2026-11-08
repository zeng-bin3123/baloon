// Unit p8ce7113562: device runtime (arena, poison fill, replay graphs, dense/embed/head residency).
// Invariants checked on every usable GPU: 256-byte aligned non-overlapping arena ranges, overflow-safe refusal, reset rewinds,
// poison fill writes exactly [dst, dst+n), graphs replay in recording order and refuse misuse, the embedding gather returns
// table rows (zeros for invalid ids), the head projection equals the host reference, eligibility of weight rows.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "baloon/rt_device.hpp"
#include "baloon/rt_graph.hpp"
#include "baloon/rt_kernels.hpp"
#include "baloon/rt_native.hpp"
#include "baloon/rt_testdata.hpp"
using namespace baloon::rt;
static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

// Device-side verifier: counts bytes in [p, p+n) that differ from v (grid-stride, 32-bit atomic count; n < 2^32 here).
__global__ void count_mismatch_kernel(const uint8_t* p, uint64_t n, uint8_t v, unsigned int* bad) {
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned int local = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) if (p[i] != v) ++local;
    if (local) atomicAdd(bad, local);
}
// Returns the number of mismatching bytes (UINT64_MAX when the check itself failed; the reason is printed).
static uint64_t device_mismatches(const void* d, uint64_t n, uint8_t v) {
    if (n >= (1ull << 32)) return UINT64_MAX;
    unsigned int* dbad = nullptr; unsigned int h = 0;
    cudaError_t e = cudaMalloc((void**)&dbad, sizeof h);
    if (e != cudaSuccess) { std::printf("verifier: malloc: %s\n", cudaGetErrorString(e)); cudaGetLastError(); return UINT64_MAX; }
    uint64_t r = UINT64_MAX;
    e = cudaMemset(dbad, 0, sizeof h);
    if (e == cudaSuccess) {
        count_mismatch_kernel<<<64, 256>>>((const uint8_t*)d, n, v, dbad);
        e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e == cudaSuccess) e = cudaMemcpy(&h, dbad, sizeof h, cudaMemcpyDeviceToHost);
        if (e == cudaSuccess) r = h;
    }
    if (e != cudaSuccess) { std::printf("verifier: %s\n", cudaGetErrorString(e)); cudaGetLastError(); }
    cudaFree(dbad);
    return r;
}

static std::vector<uint8_t> dl(const void* d, uint64_t n) {
    std::vector<uint8_t> v(n);
    if (n) CHECK(cudaMemcpy(v.data(), d, n, cudaMemcpyDeviceToHost) == cudaSuccess);
    return v;
}

static void test_arena(int ord) {
    DevArena a(ord, 1 << 16);
    CHECK(a.ok() && a.capacity() == (1u << 16) && a.used() == 0);
    void* p1 = a.alloc(1); void* p2 = a.alloc(300); void* p3 = a.alloc(256);
    CHECK(p1 && p2 && p3);
    CHECK(((uintptr_t)p1 % kArenaAlign) == 0 && ((uintptr_t)p2 % kArenaAlign) == 0 && ((uintptr_t)p3 % kArenaAlign) == 0);
    CHECK((uint8_t*)p2 >= (uint8_t*)p1 + 1 && (uint8_t*)p3 >= (uint8_t*)p2 + 300);
    CHECK((uint8_t*)p2 - (uint8_t*)p1 == 256 && (uint8_t*)p3 - (uint8_t*)p2 == 512);
    CHECK(a.alloc(UINT64_MAX) == nullptr && a.alloc(UINT64_MAX - 100) == nullptr);   // overflow-safe refusal
    CHECK(a.alloc((1 << 16)) == nullptr);                                            // does not fit after earlier allocs
    const uint64_t u = a.used();
    CHECK(a.alloc(0) != nullptr && a.used() >= u);                                   // zero-size is legal
    a.reset();
    CHECK(a.used() == 0 && a.alloc(1 << 16) == a.base());                            // full block fits after a rewind
    CHECK(a.alloc(1) == nullptr);                                                    // and then it is exactly full
    DevArena bad(ord, 0);
    CHECK(!bad.ok() && bad.alloc(1) == nullptr);
    // poison on reset covers the whole block
    a.reset(true);
    auto bytes = dl(a.base(), a.capacity());
    bool all = true; for (uint8_t b : bytes) if (b != 0xDB) { all = false; break; }
    CHECK(all);
    CHECK(DevArena::kPoisonByte == 0xDB);   // the documented poison pattern
    CHECK(device_mismatches(a.base(), a.capacity(), 0xDB) == 0);          // verified on the device too
    CHECK(device_mismatches(a.base(), a.capacity(), 0xDA) == a.capacity());   // the verifier really detects differences
}

static void test_poison(int ord) {
    DevArena a(ord, 4096);
    CHECK(a.ok());
    CHECK(poison_fill(a.base(), 4096, 0x00));
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(poison_fill(a.base() + 100, 1000, 0xA5));
    CHECK(poison_fill(a.base(), 0, 0xFF));          // n == 0 is a no-op success even for null-free calls
    CHECK(poison_fill(nullptr, 0, 1));
    CHECK(!poison_fill(nullptr, 16, 1));
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    auto v = dl(a.base(), 4096);
    bool ok = true;
    for (int i = 0; i < 4096; ++i) { const uint8_t want = (i >= 100 && i < 1100) ? 0xA5 : 0x00; if (v[i] != want) ok = false; }
    CHECK(ok);
    // a range larger than one grid pass (grid-stride loop) is still filled completely
    const uint64_t big = 4096ull * 256 * 3 + 17;
    DevArena b(ord, big);
    CHECK(b.ok() && poison_fill(b.base(), big, 0x3C) && cudaDeviceSynchronize() == cudaSuccess);
    auto w = dl(b.base(), big); bool ok2 = true; for (uint8_t x : w) if (x != 0x3C) { ok2 = false; break; }
    CHECK(ok2);
    CHECK(device_mismatches(b.base(), big, 0x3C) == 0);
}

static void test_graph(int ord) {
    (void)ord;
    cudaStream_t s; CHECK(cudaStreamCreate(&s) == cudaSuccess);
    ReplayGraph g;
    CHECK(!g.ready() && !g.launch(s) && !g.push([](cudaStream_t) { return cudaSuccess; }));   // not recording / not ready
    CHECK(!g.end());
    g.begin();
    std::vector<int> order;
    CHECK(g.push([&](cudaStream_t) { order.push_back(1); return cudaSuccess; }));
    CHECK(g.push([&](cudaStream_t) { order.push_back(2); return cudaSuccess; }));
    CHECK(g.push([&](cudaStream_t) { order.push_back(3); return cudaSuccess; }));
    CHECK(!g.push(nullptr));
    CHECK(!g.ready() && !g.launch(s));                // recording is not finished
    CHECK(g.end() && g.ready() && g.size() == 3 && !g.end());
    CHECK(g.wait_ms() < 0);                           // never ran
    CHECK(g.launch(s) && g.launches() == 1);
    CHECK(g.wait_ms() >= 0.0);
    CHECK(order == (std::vector<int>{1, 2, 3}));
    CHECK(g.launch(s) && g.launch(s) && g.launches() == 3 && order.size() == 9);
    // a failing op stops the replay and poisons the timing
    ReplayGraph f; f.begin(); int ran = 0;
    f.push([&](cudaStream_t) { ++ran; return cudaSuccess; });
    f.push([&](cudaStream_t) { ++ran; return cudaErrorInvalidValue; });
    f.push([&](cudaStream_t) { ++ran; return cudaSuccess; });
    CHECK(f.end() && !f.launch(s) && ran == 2 && f.wait_ms() < 0);
    cudaGetLastError();
    f.reset(); CHECK(!f.ready() && f.size() == 0);
    // begin() drops the old recording
    g.begin(); CHECK(!g.ready() && g.size() == 0); CHECK(g.end() && g.launch(s) && order.size() == 9);
    // GPU work inside a graph really runs
    DevArena a(ord, 1024);
    ReplayGraph k; k.begin();
    k.push([&](cudaStream_t st) { return poison_fill(a.base(), 1024, 0x77, st) ? cudaSuccess : cudaErrorInvalidValue; });
    k.end(); CHECK(k.launch(s) && k.wait_ms() >= 0);
    CHECK(cudaStreamSynchronize(s) == cudaSuccess);
    auto v = dl(a.base(), 1024); bool ok = true; for (uint8_t x : v) if (x != 0x77) ok = false; CHECK(ok);
    GraphBook gb; int c = 0;
    CHECK(gb.find("x") == nullptr && !gb.launch("x", s));
    CHECK(gb.record("x", [&](ReplayGraph& r) { r.push([&](cudaStream_t) { ++c; return cudaSuccess; }); }));
    CHECK(gb.size() == 1 && gb.find("x") && gb.find("x")->ready());
    CHECK(gb.launch("x", s) && c == 1);
    CHECK(gb.record("x", [&](ReplayGraph& r) { r.push([&](cudaStream_t) { c += 10; return cudaSuccess; }); }));   // replaces
    CHECK(gb.size() == 1 && gb.launch("x", s) && c == 11);
    cudaStreamDestroy(s);
}

static void test_native(int ord) {
    XorRng rng(5);
    const int64_t V = 37, N = 70;
    std::vector<float> tab = make_f32(rng, V * N);
    {
        EmbedTable e;
        CHECK(!e.valid() && e.vocab() == 0 && native_embed() == nullptr);
        CHECK(!e.load(nullptr, V, N, ord) && !e.load(tab.data(), 0, N, ord) && !e.load(tab.data(), V, -1, ord));
        CHECK(!e.load(tab.data(), INT64_MAX, 8, ord));                       // size overflow refused
        CHECK(e.load(tab.data(), V, N, ord) && e.valid() && e.vocab() == V && e.width() == N);
        std::vector<float> row(N, -1.f);
        CHECK(e.gather_one(5, row.data()) && std::memcmp(row.data(), &tab[5 * N], N * 4) == 0);
        std::vector<float> keep(N, 9.f);
        CHECK(!e.gather_one(V, keep.data()) && !e.gather_one(-1, keep.data()) && keep[0] == 9.f);
        const int32_t toks[5] = {0, (int32_t)V - 1, -3, (int32_t)V, 12};
        int32_t* dt; float* dout;
        CHECK(cudaMalloc((void**)&dt, sizeof toks) == cudaSuccess && cudaMalloc((void**)&dout, 5 * N * 4) == cudaSuccess);
        CHECK(cudaMemcpy(dt, toks, sizeof toks, cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(e.gather_dev(dt, 5, dout));
        std::vector<float> o(5 * N); CHECK(cudaMemcpy(o.data(), dout, 5 * N * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
        for (int i = 0; i < 5; ++i) {
            const bool valid = toks[i] >= 0 && toks[i] < V; bool ok = true;
            for (int j = 0; j < N; ++j) if (o[i * N + j] != (valid ? tab[(int64_t)toks[i] * N + j] : 0.f)) ok = false;
            CHECK(ok);
        }
        cudaFree(dt); cudaFree(dout);
        EmbedTable idle; CHECK(!idle.gather_dev(nullptr, 0, nullptr) && !idle.gather_one(0, keep.data()));
        set_native_embed(&e); CHECK(native_embed() == &e);
        set_native_embed(nullptr); CHECK(native_embed() == nullptr);
        set_native_embed(&e);
    }
    CHECK(native_embed() == nullptr);                                        // destructor clears a dangling binding
    {
        const int64_t HV = 129, HN = 193;
        std::vector<uint16_t> w = make_bf16(rng, HV * HN); std::vector<float> x = make_f32(rng, HN);
        HeadProj h; std::vector<float> lg(HV);
        CHECK(!h.valid() && !h.run(x.data(), lg.data()));
        CHECK(!h.load(nullptr, HV, HN, ord) && !h.load(w.data(), HV, 0, ord) && !h.load(w.data(), INT64_MAX, 4, ord));
        CHECK(h.load(w.data(), HV, HN, ord) && h.valid() && h.vocab() == HV && h.width() == HN);
        CHECK(h.run(x.data(), lg.data()) && !h.run(nullptr, lg.data()) && !h.run(x.data(), nullptr));
        bool ok = true;
        for (int64_t r = 0; r < HV; ++r) { const double ref = gemv_row_ref(&w[r * HN], x.data(), HN); if (std::fabs((double)lg[r] - ref) > 1e-3 * (1 + std::fabs(ref))) ok = false; }
        CHECK(ok);
        std::vector<float> lg2(HV); CHECK(h.run(x.data(), lg2.data()) && lg2 == lg);   // deterministic
    }
    baloon::core::WIndexRow r; r.placed = true; r.engine_kind = 1; r.dst_bytes = 8;
    CHECK(eligible(r));
    { auto q = r; q.placed = false; CHECK(!eligible(q)); }
    { auto q = r; q.engine_kind = 2; CHECK(!eligible(q)); }
    { auto q = r; q.dst_bytes = 0; CHECK(!eligible(q)); }
    { auto q = r; q.dst_bytes = 7; CHECK(!eligible(q)); }
}

static void test_info(int ord) {
    DeviceDesc d; std::string err;
    CHECK(query_device(ord, &d, &err) && d.ordinal == ord && d.total_mem > 0 && d.warp > 0 && !d.name.empty());
    CHECK(!query_device(-1, &d, &err) && !err.empty());
    CHECK(!query_device(device_count(), &d, nullptr));
    CHECK(cuda_ok(cudaSuccess, "x") && !cuda_ok(cudaErrorInvalidValue, "x")); cudaGetLastError();
}

int main() {
    const int n = device_count();
    CHECK(n >= 1);
    if (n < 1) { std::printf("no device\n"); return 1; }
    int use = n;
    if (const char* g = std::getenv("EINSTEINOS_GPUS")) { int c = 1; for (const char* p = g; *p; ++p) if (*p == ',') ++c; if (c < use) use = c; }
    for (int ord = 0; ord < use; ++ord) {
        CHECK(cudaSetDevice(ord) == cudaSuccess);
        test_info(ord); test_arena(ord); test_poison(ord); test_graph(ord); test_native(ord);
        std::printf("device %d done\n", ord);
    }
    std::printf("%d checks, %d failures\n", checks, fails);
    if (fails) return 1;
    std::printf("UNIT 54 TESTS PASSED\n");
    return 0;
}
