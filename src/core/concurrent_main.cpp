// concurrent_main.cpp - concurrent multi-stream GPU workload driver.
// Launches N parallel CUDA streams each running the same async memset as a proxy compute kernel,
// then synchronises all streams and reports aggregate throughput.  The Run struct carries the
// parameters and results of one experiment; check() aborts on a CUDA error.
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define BALOON_CM_HAVE_CUDA 1
#include <cuda_runtime.h>
#endif
#endif

#ifndef BALOON_CM_HAVE_CUDA
typedef int cudaError_t;
enum { cudaSuccess = 0 };
static cudaError_t cudaGetLastError() { return cudaSuccess; }
static const char* cudaGetErrorString(cudaError_t) { return "no CUDA"; }
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

// check: abort on any non-success CUDA error code.
static void check(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %d: %s\n", (int)err, cudaGetErrorString(err));
        std::abort();
    }
}

// Run: parameters and results of one concurrent stream experiment.
struct Run {
    int    device           = 0;
    int    streams          = 4;
    int    iterations       = 10;
    size_t bytes_per_stream = 1u << 22;   // 4 MiB
    double elapsed_ms       = 0.0;
    double throughput_gbs   = 0.0;
    bool   ok               = false;
};

#ifdef BALOON_CM_HAVE_CUDA
static Run do_run(int device, int nstreams, int iters, size_t bytes) {
    Run r;
    r.device = device; r.streams = nstreams; r.iterations = iters; r.bytes_per_stream = bytes;
    check(cudaSetDevice(device));
    std::vector<cudaStream_t> sv((size_t)nstreams);
    std::vector<void*>        bv((size_t)nstreams, nullptr);
    for (int i = 0; i < nstreams; ++i) {
        check(cudaStreamCreate(&sv[i]));
        check(cudaMalloc(&bv[i], bytes));
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < nstreams; ++i)
            check(cudaMemsetAsync(bv[i], 0xAB, bytes, sv[i]));
        for (int i = 0; i < nstreams; ++i)
            check(cudaStreamSynchronize(sv[i]));
    }
    const auto t1  = std::chrono::steady_clock::now();
    r.elapsed_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double gb = (double)nstreams * iters * bytes / (1ull << 30);
    r.throughput_gbs = gb / (r.elapsed_ms / 1000.0);
    r.ok = true;
    for (int i = 0; i < nstreams; ++i) { cudaFree(bv[i]); cudaStreamDestroy(sv[i]); }
    return r;
}
#endif

int main(int argc, char** argv) {
    int    device  = (argc > 1) ? std::atoi(argv[1]) : 0;
    int    nstream = (argc > 2) ? std::atoi(argv[2]) : 4;
    int    iters   = (argc > 3) ? std::atoi(argv[3]) : 10;
    size_t bytes   = 1u << 22;
#ifdef BALOON_CM_HAVE_CUDA
    Run r = do_run(device, nstream, iters, bytes);
    if (!r.ok) { std::fprintf(stderr, "concurrent run failed\n"); return 1; }
    std::printf("concurrent: device=%d streams=%d iters=%d "
                "bytes=%zu elapsed=%.1f ms  throughput=%.2f GiB/s\n",
                r.device, r.streams, r.iterations,
                r.bytes_per_stream, r.elapsed_ms, r.throughput_gbs);
#else
    (void)device; (void)nstream; (void)iters; (void)bytes;
    std::printf("CUDA not available; built without GPU support\n");
#endif
    return 0;
}

