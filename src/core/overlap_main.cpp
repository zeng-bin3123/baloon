// overlap_main.cpp - compute / transfer overlap benchmark.
// Measures compute-alone time (async memset on stream 0), transfer-alone time (DtoH on stream 1) and
// the time when both run concurrently to compute an observed speedup ratio.  The Timing struct holds
// all three measurements; check() aborts on a CUDA error.
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define BALOON_OV_HAVE_CUDA 1
#include <cuda_runtime.h>
#endif
#endif

#ifndef BALOON_OV_HAVE_CUDA
typedef int cudaError_t;
enum { cudaSuccess = 0 };
static cudaError_t cudaGetLastError() { return cudaSuccess; }
static const char* cudaGetErrorString(cudaError_t) { return "no CUDA"; }
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>

// check: abort on any non-success CUDA error.
static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %d: %s\n", (int)err, cudaGetErrorString(err));
        std::abort();
    }
}

// Timing: holds the three wall-clock measurements of one overlap experiment.
struct Timing {
    double compute_ms  = 0.0;   // compute stream alone
    double transfer_ms = 0.0;   // transfer stream alone
    double overlap_ms  = 0.0;   // both streams concurrently
    double speedup     = 1.0;   // (compute + transfer) / overlap
};

#ifdef BALOON_OV_HAVE_CUDA
static Timing measure(int device, size_t bytes) {
    Timing t;
    check(cudaSetDevice(device));
    // Page-locked host buffer for async DtoH transfers.
    void* h = nullptr; void* d = nullptr;
    check(cudaMallocHost(&h, bytes));
    check(cudaMalloc(&d, bytes));
    cudaStream_t sc = nullptr, st = nullptr;
    check(cudaStreamCreate(&sc));
    check(cudaStreamCreate(&st));
    // --- compute alone ---
    auto t0 = std::chrono::steady_clock::now();
    check(cudaMemsetAsync(d, 0xAB, bytes, sc));
    check(cudaStreamSynchronize(sc));
    t.compute_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
    // --- transfer alone ---
    t0 = std::chrono::steady_clock::now();
    check(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, st));
    check(cudaStreamSynchronize(st));
    t.transfer_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
    // --- overlap: compute + transfer concurrently ---
    t0 = std::chrono::steady_clock::now();
    check(cudaMemsetAsync(d, 0xCD, bytes, sc));
    check(cudaMemcpyAsync(h, d, bytes / 2, cudaMemcpyDeviceToHost, st));
    check(cudaStreamSynchronize(sc));
    check(cudaStreamSynchronize(st));
    t.overlap_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
    const double serial = t.compute_ms + t.transfer_ms;
    t.speedup = (t.overlap_ms > 0) ? serial / t.overlap_ms : 1.0;
    cudaStreamDestroy(sc); cudaStreamDestroy(st);
    cudaFreeHost(h); cudaFree(d);
    return t;
}
#endif

int main(int argc, char** argv) {
    int    device = (argc > 1) ? std::atoi(argv[1]) : 0;
    size_t bytes  = (argc > 2) ? (size_t)std::atoll(argv[2]) : (1u << 24);
#ifdef BALOON_OV_HAVE_CUDA
    const Timing t = measure(device, bytes);
    std::printf("overlap: compute=%.2f ms  transfer=%.2f ms  "
                "overlap=%.2f ms  speedup=%.2fx\n",
                t.compute_ms, t.transfer_ms, t.overlap_ms, t.speedup);
#else
    (void)device; (void)bytes;
    std::printf("CUDA not available; built without GPU support\n");
#endif
    return 0;
}

