// native_head.cpp - device-resident embedding table and output head implementation.
// NativeEmbed::load uploads float32 [vocab, n] to the GPU; gather_dev launches a grid-stride
// gather kernel; gather_one copies one row from the host mirror without touching the device.
// NativeHead::load uploads BF16 [vocab, n]; run() does host→device memcpy, BF16 dot-product kernel,
// device→host memcpy, then synchronises.  All device operations are skipped (return false) when
// CUDA is not available at compile time.
#include "baloon/core_native_head.hpp"

#include <cstring>
#include <cstdio>

// ---- always-available error helper -------------------------------------------------
namespace {
static bool nh_set_err(std::string* e, const char* m) { if (e) *e = m; return false; }
}  // namespace

// ---- CUDA-only helpers and kernels -------------------------------------------------
#ifdef BALOON_NH_HAVE_CUDA

namespace {
struct DevScope {
    int prev = 0; bool ok = false;
    explicit DevScope(int ord) {
        cudaGetDevice(&prev);
        ok = (cudaSetDevice(ord) == cudaSuccess);
        if (!ok) cudaGetLastError();
    }
    ~DevScope() { cudaSetDevice(prev); }
};
}  // namespace

// Token gather: table[token * n .. +n] → out[i * n .. +n] for each i in [0, count).
static __global__ void nh_gather_kernel(const float* __restrict__ table,
                                        int64_t vocab, int64_t n,
                                        const int32_t* __restrict__ tokens,
                                        int64_t count,
                                        float* __restrict__ out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int32_t t = tokens[i];
    float* dst = out + i * n;
    if (t < 0 || (int64_t)t >= vocab) {
        for (int64_t j = 0; j < n; ++j) dst[j] = 0.f;
        return;
    }
    const float* src = table + (int64_t)t * n;
    for (int64_t j = 0; j < n; ++j) dst[j] = src[j];
}

// BF16 output projection: logits[row] = sum_j W[row*n+j] * x[j], W stored as BF16.
static __global__ void nh_head_kernel(const uint16_t* __restrict__ w,
                                      int64_t vocab, int64_t n,
                                      const float* __restrict__ x,
                                      float* __restrict__ logits) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= vocab) return;
    const uint16_t* wr = w + row * n;
    float acc = 0.f;
    for (int64_t j = 0; j < n; ++j) {
        uint32_t bits = (uint32_t)wr[j] << 16;
        float wf;
        __builtin_memcpy(&wf, &bits, sizeof(wf));
        acc += wf * x[j];
    }
    logits[row] = acc;
}

#endif  // BALOON_NH_HAVE_CUDA

namespace baloon {
namespace head {

// ---- NativeEmbed::load -------------------------------------------------------------
bool NativeEmbed::load(const float* host_table, int64_t vocab, int64_t n,
                       int ordinal, std::string* err) {
#ifdef BALOON_NH_HAVE_CUDA
    if (!host_table || vocab <= 0 || n <= 0) return nh_set_err(err, "invalid args");
    DevScope sc(ordinal);
    if (!sc.ok) return nh_set_err(err, "cannot select device");
    const uint64_t bytes = (uint64_t)vocab * (uint64_t)n * 4;
    void* d = nullptr;
    if (cudaMalloc(&d, (size_t)bytes) != cudaSuccess) {
        cudaGetLastError(); return nh_set_err(err, "cudaMalloc failed");
    }
    if (cudaMemcpy(d, host_table, (size_t)bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError(); cudaFree(d); return nh_set_err(err, "cudaMemcpy failed");
    }
    dev_ = d; host_ = host_table; vocab_ = vocab; n_ = n; ord_ = ordinal;
    return true;
#else
    (void)host_table; (void)vocab; (void)n; (void)ordinal;
    return nh_set_err(err, "no CUDA");
#endif
}

// ---- NativeEmbed::gather_dev -------------------------------------------------------
bool NativeEmbed::gather_dev(const int32_t* d_tokens, int64_t count,
                             float* d_out, cudaStream_t s) const {
#ifdef BALOON_NH_HAVE_CUDA
    if (!dev_ || !d_tokens || count <= 0 || !d_out) return false;
    const int threads = 256;
    const int blocks  = (int)((count + threads - 1) / threads);
    nh_gather_kernel<<<blocks, threads, 0, s>>>(
        (const float*)dev_, vocab_, n_, d_tokens, count, d_out);
    return cudaGetLastError() == cudaSuccess;
#else
    (void)d_tokens; (void)count; (void)d_out; (void)s; return false;
#endif
}

// ---- NativeEmbed::gather_one -------------------------------------------------------
bool NativeEmbed::gather_one(int64_t token, float* out) const {
    if (!host_ || token < 0 || token >= vocab_ || !out) return false;
    std::memcpy(out, host_ + token * n_, (size_t)n_ * sizeof(float));
    return true;
}

// ---- NativeEmbed::~NativeEmbed -----------------------------------------------------
NativeEmbed::~NativeEmbed() {
#ifdef BALOON_NH_HAVE_CUDA
    if (dev_) { cudaFree(dev_); dev_ = nullptr; }
#endif
    host_ = nullptr; vocab_ = n_ = 0;
}

// ---- NativeHead::load --------------------------------------------------------------
bool NativeHead::load(const uint16_t* host_w, int64_t vocab, int64_t n,
                      int ordinal, std::string* err) {
#ifdef BALOON_NH_HAVE_CUDA
    if (!host_w || vocab <= 0 || n <= 0) return nh_set_err(err, "invalid args");
    DevScope sc(ordinal);
    if (!sc.ok) return nh_set_err(err, "cannot select device");
    const uint64_t bytes = (uint64_t)vocab * (uint64_t)n * 2;
    void* d = nullptr;
    if (cudaMalloc(&d, (size_t)bytes) != cudaSuccess) {
        cudaGetLastError(); return nh_set_err(err, "cudaMalloc failed");
    }
    if (cudaMemcpy(d, host_w, (size_t)bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError(); cudaFree(d); return nh_set_err(err, "cudaMemcpy failed");
    }
    dev_ = d; vocab_ = vocab; n_ = n; ord_ = ordinal;
    return true;
#else
    (void)host_w; (void)vocab; (void)n; (void)ordinal;
    return nh_set_err(err, "no CUDA");
#endif
}

// ---- NativeHead::run ---------------------------------------------------------------
bool NativeHead::run(const float* x, float* logits) const {
#ifdef BALOON_NH_HAVE_CUDA
    if (!dev_ || !x || !logits) return false;
    void* d_x = nullptr; void* d_logits = nullptr;
    const size_t xb = (size_t)n_     * sizeof(float);
    const size_t lb = (size_t)vocab_ * sizeof(float);
    if (cudaMalloc(&d_x,      xb) != cudaSuccess ||
        cudaMalloc(&d_logits, lb) != cudaSuccess) {
        cudaGetLastError();
        if (d_x) cudaFree(d_x);
        return false;
    }
    bool ok = (cudaMemcpy(d_x, x, xb, cudaMemcpyHostToDevice) == cudaSuccess);
    if (ok) {
        const int threads = 256;
        const int blocks  = (int)((vocab_ + threads - 1) / threads);
        nh_head_kernel<<<blocks, threads>>>(
            (const uint16_t*)dev_, vocab_, n_, (const float*)d_x, (float*)d_logits);
        ok = (cudaGetLastError() == cudaSuccess);
    }
    if (ok) ok = (cudaDeviceSynchronize() == cudaSuccess);
    if (ok) ok = (cudaMemcpy(logits, d_logits, lb, cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d_x); cudaFree(d_logits);
    if (!ok) cudaGetLastError();
    return ok;
#else
    (void)x; (void)logits; return false;
#endif
}

// ---- NativeHead::~NativeHead -------------------------------------------------------
NativeHead::~NativeHead() {
#ifdef BALOON_NH_HAVE_CUDA
    if (dev_) { cudaFree(dev_); dev_ = nullptr; }
#endif
    vocab_ = n_ = 0;
}

// ---- process-wide embedding slot ---------------------------------------------------
// g_embed is the process-wide const pointer to the active embedding table.
const NativeEmbed* g_embed = nullptr;

const NativeEmbed* native_embed()              { return g_embed; }
void set_native_embed(const NativeEmbed* e)    { g_embed = e; }

}  // namespace head
}  // namespace baloon

