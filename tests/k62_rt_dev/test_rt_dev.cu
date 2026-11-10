// Unit pe66b5ae2b1 test (CUDA, needs devices): the device arena, poison fill, device query, error check and human() invariants,
// checked on EVERY visible device, then once more with all devices working at the same time.
//  * arena ranges are 256-byte aligned, never overlap, never exceed capacity; a request that does not fit (including sizes that
//    would wrap 64 bits) is refused without consuming anything; reset() rewinds; reset(true) poisons the whole block with 0xDB;
//  * poison_fill writes exactly [dst, dst+n) (neighbours untouched), for any n including > one grid sweep; n == 0 ok, null refused;
//  * query_device refuses ordinals outside [0,count) with a message, reports sane fields, leaves the current device unchanged;
//  * cuda_ok is true exactly for cudaSuccess; human() prints 1 decimal in binary units ("1023 B", "1.5 KiB", "16.0 EiB").
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "baloon/rt_device.hpp"

std::string human(uint64_t bytes);   // device_main.cpp (built with RT_DEVICE_MAIN_DISABLED)

using namespace baloon::rt;

static const uint64_t kAL = 256;            // literal on purpose: the header constant itself is under test
static const uint8_t kPOISON = 0xDB;
static long g_checks = 0, g_fail = 0;
#define T(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 80) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static std::vector<uint8_t> dl(const void* d, uint64_t n) {
    std::vector<uint8_t> v((size_t)n);
    if (n) T(cudaMemcpy(v.data(), d, (size_t)n, cudaMemcpyDeviceToHost) == cudaSuccess);
    return v;
}

static void test_arena(int dev) {
    const uint64_t cap = 1 << 20;
    int prev = 0; T(cudaSetDevice(0) == cudaSuccess); cudaGetDevice(&prev);
    {
        DevArena a(dev, cap);
        T(a.ok()); if (!a.ok()) return;
        T(a.capacity() == cap && a.used() == 0 && a.ordinal() == dev);
        int cur = -1; cudaGetDevice(&cur); T(cur == prev);                     // constructor restores the current device
        struct R { uint64_t off, n; };
        std::vector<R> got;
        const uint64_t sizes[] = {1, 255, 256, 257, 1000, 3, 4096, 511};
        uint64_t last_end = 0;
        for (uint64_t n : sizes) {
            uint8_t* p = (uint8_t*)a.alloc(n);
            T(p != nullptr); if (!p) continue;
            const uint64_t off = (uint64_t)(p - a.base());
            T(off % kAL == 0);
            T(off >= last_end);                                                // bump: never overlaps an earlier range
            T(off + n <= cap);
            T(off - last_end < kAL);                                   // wastes less than one alignment unit
            last_end = off + n;
            T(a.used() == last_end);
            got.push_back({off, n});
        }
        for (size_t i = 0; i < got.size(); ++i)
            for (size_t j = i + 1; j < got.size(); ++j) T(got[i].off + got[i].n <= got[j].off || got[j].off + got[j].n <= got[i].off);
        // refusals consume nothing
        const uint64_t before = a.used();
        T(a.alloc(UINT64_MAX) == nullptr);
        T(a.alloc(UINT64_MAX - 100) == nullptr);                               // would wrap if added to the offset
        T(a.alloc(cap) == nullptr);
        T(a.alloc(cap - before + 1) == nullptr);
        T(a.used() == before);
        // exact fit: the rest of the block after alignment is allocatable, then nothing more
        const uint64_t at = (before + kAL - 1) / kAL * kAL;
        void* rest = a.alloc(cap - at);
        T(rest != nullptr && (uint8_t*)rest == a.base() + at);
        T(a.used() == cap);
        T(a.alloc(1) == nullptr);
        // reset rewinds to the very start
        a.reset();
        T(a.used() == 0);
        T(a.alloc(10) == a.base());
        // poisoned reset: every byte of the block becomes 0xDB, used rewinds
        T(cudaSetDevice(dev) == cudaSuccess);                                  // raw device accesses need the owning device current
        T(cudaMemset(a.base(), 0, cap) == cudaSuccess);
        T(cudaDeviceSynchronize() == cudaSuccess);
        T(cudaSetDevice(prev) == cudaSuccess);
        a.reset(true);                                                         // must work whatever device is current
        T(a.used() == 0);
        T(cudaSetDevice(dev) == cudaSuccess);
        std::vector<uint8_t> b = dl(a.base(), cap);
        T(cudaSetDevice(prev) == cudaSuccess);
        uint64_t bad = 0; for (uint8_t x : b) if (x != kPOISON) ++bad;
        T(bad == 0); T(DevArena::kPoisonByte == kPOISON); T(kArenaAlign == kAL);
    }
    int cur = -1; cudaGetDevice(&cur); T(cur == prev);                         // destructor restores the current device too
    // degenerate constructions are refused cleanly and the arena then refuses every allocation
    { DevArena z(dev, 0); T(!z.ok()); T(z.alloc(1) == nullptr); T(z.capacity() == 0); }
    { DevArena bad(dev, 1ull << 50); T(!bad.ok()); T(bad.alloc(1) == nullptr); cudaGetLastError(); }
    { DevArena bad(-1, 4096); T(!bad.ok()); T(bad.alloc(1) == nullptr); cudaGetLastError(); }
    { DevArena bad(1 << 20, 4096); T(!bad.ok()); cudaGetLastError(); }
    // the device is still usable afterwards
    { DevArena again(dev, 4096); T(again.ok()); T(again.alloc(100) != nullptr); }
}

static void test_poison_fill(int dev) {
    T(cudaSetDevice(dev) == cudaSuccess);
    const uint64_t cap = 4ull << 20;
    DevArena a(dev, cap);
    T(a.ok()); if (!a.ok()) return;
    struct Case { uint64_t off, n; uint8_t v; };
    const Case cases[] = {{300, 1, 0xAB}, {0, 255, 0x11}, {1, 257, 0x22}, {4093, 4099, 0x5A}, {17, 1000003, 0xEE}, {1, (3ull << 20) + 7, 0x7F}, {0, 0, 0x33}};
    for (const Case& c : cases) {
        T(cudaMemset(a.base(), 0, cap) == cudaSuccess);
        T(poison_fill(a.base() + c.off, c.n, c.v));
        T(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<uint8_t> b = dl(a.base(), cap);
        uint64_t in_bad = 0, out_bad = 0;
        for (uint64_t i = 0; i < cap; ++i) {
            const bool inside = i >= c.off && i < c.off + c.n;
            if (inside ? b[i] != c.v : b[i] != 0) { if (inside) ++in_bad; else ++out_bad; }
        }
        T(in_bad == 0); T(out_bad == 0);
    }
    T(!poison_fill(nullptr, 16, 0x01));                                       // null destination refused
    T(poison_fill(nullptr, 0, 0x01));                                         // empty fill is a no-op success
    // on a non-default stream the fill is ordered after earlier work on that stream
    cudaStream_t s; T(cudaStreamCreate(&s) == cudaSuccess);
    T(cudaMemsetAsync(a.base(), 0, 4096, s) == cudaSuccess);
    T(poison_fill(a.base(), 4096, 0x9C, s));
    T(cudaStreamSynchronize(s) == cudaSuccess);
    std::vector<uint8_t> b = dl(a.base(), 4096);
    uint64_t bad = 0; for (uint8_t x : b) if (x != 0x9C) ++bad;
    T(bad == 0);
    cudaStreamDestroy(s);
}

static void test_query(int dev, int count) {
    int prev = 0; T(cudaSetDevice(dev) == cudaSuccess); cudaGetDevice(&prev);
    DeviceDesc d; std::string err;
    T(query_device(dev, &d, &err));
    T(d.ordinal == dev && !d.name.empty());
    T(d.total_mem >= (1ull << 30));
    T(d.free_mem > 0 && d.free_mem <= d.total_mem);
    T(d.warp > 0 && d.sm_count > 0 && d.cc_major >= 0);
    int cur = -1; cudaGetDevice(&cur); T(cur == prev);                         // the current device is left alone
    const int bad[] = {-1, count, count + 1, 1 << 20};
    for (int o : bad) {
        DeviceDesc x; x.name = "untouched"; std::string e;
        T(!query_device(o, &x, &e));
        T(!e.empty());
        T(x.name == "untouched");                                              // a refused query writes nothing
    }
    T(query_device(dev, nullptr, nullptr));                                    // output and error pointers are optional
}

static void test_misc() {
    T(cuda_ok(cudaSuccess, "ok"));
    T(!cuda_ok(cudaErrorInvalidValue, "bad"));
    T(!cuda_ok(cudaErrorMemoryAllocation, "oom", "f.cu", 7));
    T(device_count() >= 1);
    T(human(0) == "0 B");
    T(human(1) == "1 B");
    T(human(1023) == "1023 B");
    T(human(1024) == "1.0 KiB");
    T(human(1536) == "1.5 KiB");
    T(human(1ull << 20) == "1.0 MiB");
    T(human((1ull << 20) * 3 / 2) == "1.5 MiB");
    T(human(32768ull << 20) == "32.0 GiB");
    T(human(1ull << 40) == "1.0 TiB");
    T(human(1ull << 50) == "1.0 PiB");
    T(human(1ull << 60) == "1.0 EiB");
    T(human(UINT64_MAX) == "16.0 EiB");
}

int main() {
    const int n = device_count();
    std::printf("devices: %d\n", n);
    T(n >= 1);
    test_misc();
    for (int d = 0; d < n; ++d) {
        test_arena(d); test_poison_fill(d); test_query(d, n);
        std::printf("device %d done\n", d);
    }
    // every device at once: independent arenas must not disturb one another
    std::vector<long> fails(n, 0);
    std::vector<std::thread> th;
    for (int d = 0; d < n; ++d) th.emplace_back([d, &fails]() {
        if (cudaSetDevice(d) != cudaSuccess) { fails[d] = 1; return; }
        DevArena a(d, 1 << 20);
        if (!a.ok()) { fails[d] = 1; return; }
        const uint8_t pat = (uint8_t)(0x40 + d);
        for (int it = 0; it < 20; ++it) {
            uint8_t* p = (uint8_t*)a.alloc(65536);
            if (!p || !poison_fill(p, 65536, pat)) { fails[d] = 1; return; }
            if (cudaDeviceSynchronize() != cudaSuccess) { fails[d] = 1; return; }
            std::vector<uint8_t> b(65536);
            if (cudaMemcpy(b.data(), p, 65536, cudaMemcpyDeviceToHost) != cudaSuccess) { fails[d] = 1; return; }
            for (uint8_t x : b) if (x != pat) { fails[d] = 1; return; }
            a.reset();
        }
    });
    for (auto& t : th) t.join();
    for (int d = 0; d < n; ++d) { T(fails[d] == 0); }
    std::printf("%ld checks, %ld failed\n", g_checks, g_fail);
    if (g_fail) { std::printf("RT-DEV TESTS FAILED\n"); return 1; }
    std::printf("RT-DEV TESTS PASSED\n");
    return 0;
}
