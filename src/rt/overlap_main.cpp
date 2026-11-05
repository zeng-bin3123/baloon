// overlap_main.cpp - streams pinned chunks to the device and runs a gemv on each, serially and with copy/compute
// overlap on two streams (double buffered). Both schedules must give identical results; the overlapped schedule must
// not be slower than the serial one beyond a small tolerance.
// usage: overlap_main [--device D] [--chunks N]   exit 0 ok, 1 failure, 2 usage.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "baloon/rt_device.hpp"
#include "baloon/rt_kernels.hpp"
#include "baloon/rt_pinned.hpp"
#include "baloon/rt_testdata.hpp"

using namespace baloon::rt;

struct Timing {
    double serial_ms = 0, overlap_ms = 0;
    uint64_t serial_sum = 0, overlap_sum = 0;
    bool ok = true;
};

static const int64_t kRows = 1024, kCols = 2048;
static const uint64_t kChunk = (uint64_t)kRows * kCols * 2;

static bool check(const Timing& t) {
    if (!t.ok) { std::fprintf(stderr, "a schedule failed to run\n"); return false; }
    if (t.serial_sum != t.overlap_sum) { std::fprintf(stderr, "results differ: %016llx vs %016llx\n", (unsigned long long)t.serial_sum, (unsigned long long)t.overlap_sum); return false; }
    if (t.overlap_ms > t.serial_ms * 1.10 + 2.0) { std::fprintf(stderr, "overlap slower than serial: %.2f ms vs %.2f ms\n", t.overlap_ms, t.serial_ms); return false; }
    return true;
}

static bool sync_ok(cudaStream_t s) { if (cudaStreamSynchronize(s) != cudaSuccess) { clear_error(); return false; } return true; }

int main(int argc, char** argv) {
    int dev = 0, chunks = 32;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--device") && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--chunks") && i + 1 < argc) chunks = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: %s [--device D] [--chunks N]\n", argv[0]); return 2; }
    }
    if (dev < 0 || dev >= device_count() || chunks < 2) { std::fprintf(stderr, "bad device or chunk count\n"); return 1; }
    if (cudaSetDevice(dev) != cudaSuccess) return 1;

    PinArena pin;
    if (!pin.reserve(kChunk * chunks, 4)) { std::fprintf(stderr, "cannot reserve host arena\n"); return 1; }
    XorRng rng(3);
    { std::vector<uint16_t> w = make_bf16(rng, kChunk * chunks / 2); std::memcpy(pin.data(), w.data(), w.size() * 2); }
    std::vector<float> x = make_f32(rng, kCols);

    DevArena arena(dev, 4 * kChunk + (4ull << 20));
    if (!arena.ok()) return 1;
    uint16_t* buf[2] = {(uint16_t*)arena.alloc(kChunk), (uint16_t*)arena.alloc(kChunk)};
    float* dx = (float*)arena.alloc(kCols * 4);
    float* dy = (float*)arena.alloc((uint64_t)chunks * kRows * 4);
    if (!buf[0] || !buf[1] || !dx || !dy) return 1;
    cudaMemcpy(dx, x.data(), kCols * 4, cudaMemcpyHostToDevice);

    cudaStream_t cs, ks;
    if (cudaStreamCreate(&cs) != cudaSuccess || cudaStreamCreate(&ks) != cudaSuccess) return 1;
    cudaEvent_t copied[2], done[2];
    for (int i = 0; i < 2; ++i) { cudaEventCreate(&copied[i]); cudaEventCreate(&done[i]); }

    auto digest = [&]() {
        std::vector<float> y((size_t)chunks * kRows);
        cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
        return fnv1a64((const uint8_t*)y.data(), y.size() * 4);
    };
    auto serial = [&](bool* ok) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < chunks; ++c) {
            if (cudaMemcpyAsync(buf[0], pin.data() + (uint64_t)c * kChunk, kChunk, cudaMemcpyHostToDevice, cs) != cudaSuccess) *ok = false;
            if (!sync_ok(cs)) *ok = false;
            if (!bf16_gemv(buf[0], dx, kRows, kCols, dy + (int64_t)c * kRows, ks)) *ok = false;
            if (!sync_ok(ks)) *ok = false;
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    auto overlapped = [&](bool* ok) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < chunks; ++c) {
            const int b = c & 1;
            if (c >= 2) cudaStreamWaitEvent(cs, done[b], 0);               // buffer b is free once chunk c-2 was consumed
            if (cudaMemcpyAsync(buf[b], pin.data() + (uint64_t)c * kChunk, kChunk, cudaMemcpyHostToDevice, cs) != cudaSuccess) *ok = false;
            cudaEventRecord(copied[b], cs);
            cudaStreamWaitEvent(ks, copied[b], 0);
            if (!bf16_gemv(buf[b], dx, kRows, kCols, dy + (int64_t)c * kRows, ks)) *ok = false;
            cudaEventRecord(done[b], ks);
        }
        if (!sync_ok(cs) || !sync_ok(ks)) *ok = false;
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    Timing t; t.serial_ms = t.overlap_ms = 1e30;
    serial(&t.ok); overlapped(&t.ok);                                       // warm-up
    for (int rep = 0; rep < 3; ++rep) {
        cudaMemset(dy, 0, (uint64_t)chunks * kRows * 4);
        t.serial_ms = std::min(t.serial_ms, serial(&t.ok)); t.serial_sum = digest();
        cudaMemset(dy, 0, (uint64_t)chunks * kRows * 4);
        t.overlap_ms = std::min(t.overlap_ms, overlapped(&t.ok)); t.overlap_sum = digest();
        if (t.serial_sum != t.overlap_sum) break;
    }
    // spot check one chunk against the CPU reference
    {
        std::vector<float> y(kRows);
        cudaMemcpy(y.data(), dy + (int64_t)(chunks - 1) * kRows, kRows * 4, cudaMemcpyDeviceToHost);
        const uint16_t* w = (const uint16_t*)(pin.data() + (uint64_t)(chunks - 1) * kChunk);
        for (int64_t r = 0; r < kRows; r += 97) {
            const double ref = gemv_row_ref(w + r * kCols, x.data(), kCols);
            if (std::abs((double)y[r] - ref) > 1e-3 + 1e-4 * std::abs(ref)) { std::fprintf(stderr, "row %lld off the CPU reference\n", (long long)r); t.ok = false; }
        }
    }
    const double mb = (double)kChunk * chunks / (1024.0 * 1024.0);
    std::printf("serial %.2f ms, overlapped %.2f ms (%.0f MiB, backing %d)\n", t.serial_ms, t.overlap_ms, mb, (int)pin.backing());
    cudaStreamDestroy(cs); cudaStreamDestroy(ks);
    for (int i = 0; i < 2; ++i) { cudaEventDestroy(copied[i]); cudaEventDestroy(done[i]); }
    if (!check(t)) { std::fprintf(stderr, "OVERLAP FAILED\n"); return 1; }
    std::printf("OVERLAP OK\n");
    return 0;
}
