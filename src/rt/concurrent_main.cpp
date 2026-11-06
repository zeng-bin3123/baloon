// concurrent_main.cpp - one thread per device runs the same seeded bf16 gemv repeatedly; every device must match the
// CPU reference and every device must produce bit-identical output (the kernel's reduction order is fixed).
// usage: concurrent_main [--devices N] [--iters K]   exit 0 ok, 1 failure, 2 usage.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "baloon/rt_device.hpp"
#include "baloon/rt_kernels.hpp"
#include "baloon/rt_pinned.hpp"
#include "baloon/rt_testdata.hpp"

using namespace baloon::rt;

struct Run {
    int dev = 0;
    bool ok = false;
    uint64_t checksum = 0;
    double max_err = 0, ms = 0;
    std::string msg;
};

static const int64_t kRows = 512, kCols = 1024;

static void run_device(int dev, int iters, Run* r) {
    r->dev = dev;
    const auto t0 = std::chrono::steady_clock::now();
    XorRng rng(7);                                   // same seed on every device
    std::vector<uint16_t> w = make_bf16(rng, (uint64_t)kRows * kCols);
    std::vector<float> x = make_f32(rng, kCols);
    DevArena arena(dev, 16ull << 20);
    if (!arena.ok()) { r->msg = "arena"; return; }
    if (cudaSetDevice(dev) != cudaSuccess) { r->msg = "setdevice"; return; }
    auto* dw = (uint16_t*)arena.alloc(w.size() * 2);
    auto* dx = (float*)arena.alloc(x.size() * 4);
    auto* dy = (float*)arena.alloc(kRows * 4);
    if (!dw || !dx || !dy) { r->msg = "alloc"; return; }
    if (cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) { r->msg = "upload"; return; }
    std::vector<float> y(kRows);
    for (int it = 0; it < iters; ++it) {
        if (!bf16_gemv(dw, dx, kRows, kCols, dy)) { r->msg = "launch"; return; }
        if (cudaMemcpy(y.data(), dy, kRows * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { r->msg = "download"; return; }
        const uint64_t c = fnv1a64((const uint8_t*)y.data(), y.size() * 4);
        if (it == 0) r->checksum = c;
        else if (c != r->checksum) { r->msg = "output changed between iterations"; return; }
    }
    for (int64_t i = 0; i < kRows; ++i) {
        double sabs = 0;
        for (int64_t c = 0; c < kCols; ++c) sabs += std::fabs((double)bf16_to_f32(w[i * kCols + c]) * x[c]);
        const double err = std::fabs((double)y[i] - gemv_row_ref(&w[i * kCols], x.data(), kCols));
        const double tol = 1e-6 + 1e-5 * sabs;
        if (err > tol) { r->msg = "mismatch with CPU reference at row " + std::to_string(i); return; }
        if (err > r->max_err) r->max_err = err;
    }
    r->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    r->ok = true;
}

// Every run succeeded and all checksums agree; `why` receives the first problem.
static bool check(const std::vector<Run>& runs, std::string* why) {
    if (runs.empty()) { *why = "no runs"; return false; }
    for (const Run& r : runs) if (!r.ok) { *why = "device " + std::to_string(r.dev) + ": " + r.msg; return false; }
    for (const Run& r : runs) if (r.checksum != runs[0].checksum) { *why = "device " + std::to_string(r.dev) + " differs from device " + std::to_string(runs[0].dev); return false; }
    return true;
}

int main(int argc, char** argv) {
    int ndev = device_count(), iters = 8;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--devices") && i + 1 < argc) ndev = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: %s [--devices N] [--iters K]\n", argv[0]); return 2; }
    }
    if (ndev > device_count()) ndev = device_count();
    if (ndev <= 0 || iters <= 0) { std::fprintf(stderr, "no device to run on\n"); return 1; }
    std::vector<Run> runs(ndev);
    std::vector<std::thread> th;
    for (int d = 0; d < ndev; ++d) th.emplace_back(run_device, d, iters, &runs[d]);
    for (auto& t : th) t.join();
    for (const Run& r : runs) std::printf("device %d: %s checksum %016llx max_err %.3g %.1f ms\n", r.dev, r.ok ? "ok" : r.msg.c_str(), (unsigned long long)r.checksum, r.max_err, r.ms);
    std::string why;
    if (!check(runs, &why)) { std::fprintf(stderr, "CONCURRENT FAILED: %s\n", why.c_str()); return 1; }
    std::printf("CONCURRENT OK devices=%d\n", ndev);
    return 0;
}
