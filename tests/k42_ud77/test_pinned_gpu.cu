// test_pinned_gpu.cu - PinnedArenaLoader on the real device: sliced registration + resident tail cover the arena exactly,
// parallel per-layer load (each layer by exactly one worker, per-layer FNV == hash of the layer's bytes), range packing,
// and a host->device streaming round trip that must reproduce the arena bytes.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/core_pinned_hw.hpp"
#include <core/pinned.cu>   // single translation unit: the kernel under test is defined and launched in this executable

using namespace baloon::hw;
static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

int main() {
    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0);
    if (ndev <= 0) { std::printf("UD77 GPU TESTS FAILED (no device)\n"); return 1; }
    int dev = 0;
    if (const char* g = std::getenv("EINSTEINOS_GPUS")) dev = std::atoi(g);
    if (dev < 0 || dev >= ndev) dev = 0;
    CHECK(cudaSetDevice(dev) == cudaSuccess);
    void* probe = nullptr; CHECK(cudaMalloc(&probe, 1 << 20) == cudaSuccess); cudaFree(probe);   // really opens the device

    std::mt19937_64 rng(5);
    // ---- FNV: chunking transparent, quirk seed differs from the standard basis, empty keeps the seed
    std::vector<uint8_t> buf(9001); for (auto& b : buf) b = (uint8_t)rng();
    const uint64_t whole = fnv1a64(buf.data(), buf.size());
    for (uint64_t ch : {1ull, 13ull, 4096ull, 9001ull}) {
        uint64_t h = kRefFnvSeed;
        for (uint64_t pos = 0; pos < buf.size(); pos += ch) h = fnv1a64(buf.data() + pos, std::min<uint64_t>(ch, buf.size() - pos), h);
        CHECK(h == whole);
    }
    CHECK(fnv1a64(buf.data(), 0, 42) == 42);
    CHECK(kRefFnvSeed == 1469598103934665603ull && kRefFnvSeed != kStdFnvBasis);
    CHECK(fnv1a64(buf.data(), buf.size(), kStdFnvBasis) != whole);
    { const uint8_t one = 0; CHECK(fnv1a64(&one, 1, 0) == 0);                                    // (0 ^ 0) * prime
      CHECK(fnv1a64(&one, 1, 1) == kFnvPrime); }                                                  // (1 ^ 0) * prime

    // ---- uniform_bounds
    for (int i = 0; i < 500; ++i) {
        const uint64_t total = rng() % 100000, parts = rng() % 40 + 1;
        auto b = uniform_bounds(total, parts);
        CHECK(b.size() == parts + 1 && b.front() == 0 && b.back() == total);
        for (uint64_t k = 0; k < parts; ++k) { const uint64_t w = b[k + 1] - b[k]; CHECK(w == total / parts || w == total / parts + 1); }
    }
    CHECK(uniform_bounds(10, 0).empty());

    // ---- arena: registered prefix + locked tail cover the arena exactly
    for (uint64_t slice : {0ull, 4096ull, 65536ull, 1ull << 20, 3ull << 20}) {
        PinnedArena a; const uint64_t bytes = (8ull << 20) + 12345;
        CHECK(a.reserve(bytes, slice));
        CHECK(a.bytes() == bytes && a.data() != nullptr);
        CHECK(a.registered_bytes() + a.locked_bytes() == bytes && a.locked_begin() == a.registered_bytes());
        CHECK(a.registered_bytes() <= bytes && a.registered_bytes() % a.slice_bytes() == 0);
        if (a.registered_bytes() == bytes / a.slice_bytes() * a.slice_bytes()) CHECK(a.locked_bytes() < a.slice_bytes());
        CHECK(a.registered_bytes() > 0);                                                           // pinning really worked on this device
        bool zero = true; for (uint64_t i = 0; i < bytes; i += 4099) zero &= (a.data()[i] == 0); CHECK(zero);
        a.release(); CHECK(a.data() == nullptr && a.bytes() == 0);
    }
    { PinnedArena a; CHECK(!a.reserve(0)); }

    // ---- parallel per-layer load
    const uint64_t L = 6, bpl = 5, blob = 4109, lb = bpl * blob, total = L * lb;
    std::vector<uint8_t> src(total); for (auto& b : src) b = (uint8_t)rng();
    std::atomic<uint64_t> calls{0};
    Reader rd = [&](uint64_t off, uint64_t n, uint8_t* dst) { ++calls; if (off + n > src.size()) return false; std::memcpy(dst, src.data() + off, n); return true; };
    for (unsigned workers : {1u, 2u, 4u, 9u}) {
        PinnedArena a; CHECK(a.reserve(total, 32768));
        calls = 0;
        LoadResult r = load_experts(a, rd, L, bpl, blob, 1000, workers);
        CHECK(r.ok && r.bytes == total && r.hashes.size() == L);
        CHECK(std::memcmp(a.data(), src.data(), total) == 0);                                      // layers tile the whole source
        CHECK(r.reads == L * ((lb + 999) / 1000) && calls == r.reads);                              // no byte read twice or skipped
        for (uint64_t l = 0; l < L; ++l) {
            CHECK(r.hashes[l] == fnv1a64(src.data() + l * lb, lb));                                 // per-layer hash == hash of the concatenation
            CHECK(r.claimed_by[l] < std::max(1u, workers));                                         // every layer claimed by exactly one worker
        }
    }
    { // a failing read latches and the load reports failure
        PinnedArena a; CHECK(a.reserve(total));
        Reader bad = [&](uint64_t off, uint64_t n, uint8_t* dst) { if (off >= 3 * lb) return false; std::memcpy(dst, src.data() + off, n); return true; };
        LoadResult r = load_experts(a, bad, L, bpl, blob, 1000, 3); CHECK(!r.ok); CHECK(r.bytes < total);
    }
    { PinnedArena a; CHECK(a.reserve(total - 1)); CHECK(!load_experts(a, rd, L, bpl, blob, 1000, 2).ok);   // arena too small is refused
      CHECK(!load_experts(a, rd, L, bpl, blob, 0, 1).ok); }                                                // chunk 0 is refused

    // ---- ranges packed back to back
    { PinnedArena a; CHECK(a.reserve(5000));
      std::vector<std::pair<uint64_t, uint64_t>> rg{{100, 700}, {9000, 1}, {20000, 3000}};
      LoadResult r = load_experts_ranges(a, rd, rg, 256);
      CHECK(r.ok && r.bytes == 3701 && r.hashes.size() == 3);
      CHECK(std::memcmp(a.data(), src.data() + 100, 700) == 0 && a.data()[700] == src[9000] && std::memcmp(a.data() + 701, src.data() + 20000, 3000) == 0);
      CHECK(r.hashes[2] == fnv1a64(src.data() + 20000, 3000));
      rg.push_back({0, 5000}); CHECK(!load_experts_ranges(a, rd, rg, 256).ok); }

    // ---- stream to the device and back
    { PinnedArena a; const uint64_t bytes = (4ull << 20) + 77; CHECK(a.reserve(bytes, 1 << 20));
      for (uint64_t i = 0; i < bytes; ++i) a.data()[i] = (uint8_t)(i * 131 + (i >> 8));
      void* d = nullptr; CHECK(cudaMalloc(&d, bytes) == cudaSuccess && d);
      CHECK(cudaMemset(d, 0, bytes) == cudaSuccess);
      const uint64_t chunk = 1 << 20;
      Bandwidth bw = stream_bandwidth(a, d, chunk, 3);
      CHECK(bw.ok && bw.passes == 3 && bw.bytes == 3 * (bytes / chunk * chunk) && bw.gbps > 0);
      std::vector<uint8_t> back(bytes, 0xEE);
      CHECK(cudaMemcpy(back.data(), d, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
      const uint64_t moved = bytes / chunk * chunk;
      CHECK(std::memcmp(back.data(), a.data(), moved) == 0);                                       // whole chunks arrive intact
      bool tail_untouched = true; for (uint64_t i = moved; i < bytes; ++i) tail_untouched &= (back[i] == 0); CHECK(tail_untouched);   // only whole chunks move
      { // the GPU kernel hashes each streamed chunk; it must equal the host hash of the same chunk
        std::vector<uint64_t> dh;
        CHECK(device_chunk_hashes(d, bytes, chunk, dh) && dh.size() == bytes / chunk);
        for (size_t i = 0; i < dh.size(); ++i) CHECK(dh[i] == fnv1a64(a.data() + i * chunk, chunk));
        {   // direct launch of the kernel from the test itself
            const uint64_t n = bytes / chunk; uint64_t* d_out = nullptr;
            CHECK(cudaMalloc((void**)&d_out, n * sizeof(uint64_t)) == cudaSuccess);
            fnv1a_chunks_kernel<<<(unsigned)((n + 63) / 64), 64>>>((const uint8_t*)d, chunk, n, kRefFnvSeed, d_out);
            CHECK(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess);
            std::vector<uint64_t> h2(n, 0);
            CHECK(cudaMemcpy(h2.data(), d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost) == cudaSuccess);
            for (uint64_t i = 0; i < n; ++i) CHECK(h2[i] == fnv1a64(a.data() + i * chunk, chunk));
            cudaFree(d_out);
        }
        CHECK(device_chunk_hashes(d, bytes, 4099, dh) && dh.size() == bytes / 4099 && dh[3] == fnv1a64(a.data() + 3 * 4099, 4099));
        CHECK(!device_chunk_hashes(d, bytes, 0, dh) && !device_chunk_hashes(nullptr, bytes, chunk, dh));
      }
      CHECK(!stream_bandwidth(a, d, 0, 1).ok);
      cudaFree(d); }

    std::printf("%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail) { std::printf("UD77 GPU TESTS FAILED\n"); return 1; }
    std::printf("UD77 GPU TESTS PASSED\n");
    return 0;
}
