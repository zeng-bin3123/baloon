// embed_head.cpp - NativeEmbed, NativeHead and the process-wide embedding binding of rt_native.hpp.
#include <cstring>

#include "baloon/rt_kernels.hpp"
#include "baloon/rt_native.hpp"

namespace baloon {
namespace rt {

namespace {
bool set_err(std::string* err, const std::string& m) { if (err) *err = m; return false; }
struct DevScope {
    int prev = 0; bool ok = false;
    explicit DevScope(int ord) { cudaGetDevice(&prev); ok = cudaSetDevice(ord) == cudaSuccess; if (!ok) cudaGetLastError(); }
    ~DevScope() { cudaSetDevice(prev); }
};
bool upload(DevicePtr& dst, const void* host, uint64_t bytes, std::string* err) {
    void* d = nullptr;
    if (cudaMalloc(&d, (size_t)bytes) != cudaSuccess) { cudaGetLastError(); return set_err(err, "device allocation failed"); }
    dst.reset(d);
    if (cudaMemcpy(d, host, (size_t)bytes, cudaMemcpyHostToDevice) != cudaSuccess) { cudaGetLastError(); dst.reset(); return set_err(err, "upload failed"); }
    return true;
}
bool mul_fits(int64_t a, int64_t b, int64_t elem) { return a > 0 && b > 0 && elem > 0 && a <= INT64_MAX / b / elem; }
}  // namespace

static const NativeEmbed* g_embed = nullptr;
const NativeEmbed* native_embed() { return g_embed; }
void set_native_embed(const NativeEmbed* e) { g_embed = e; }

NativeEmbed::~NativeEmbed() {
    if (dev_) { DevScope sc(ord_); dev_.reset(); }
    if (g_embed == this) g_embed = nullptr;
}

bool NativeEmbed::load(const float* host_table, int64_t vocab, int64_t n, int ordinal, std::string* err) {
    if (dev_) { DevScope sc(ord_); dev_.reset(); }
    host_ = nullptr; vocab_ = n_ = 0;
    if (!host_table || !mul_fits(vocab, n, 4)) return set_err(err, "bad embedding table");
    DevScope sc(ordinal);
    if (!sc.ok) return set_err(err, "cannot select device");
    if (!upload(dev_, host_table, (uint64_t)vocab * (uint64_t)n * 4, err)) return false;
    host_ = host_table; vocab_ = vocab; n_ = n; ord_ = ordinal;
    return true;
}

bool NativeEmbed::gather_dev(const int32_t* d_tokens, int64_t count, float* d_out, cudaStream_t s) const {
    if (!dev_) return false;
    return embed_gather(static_cast<const float*>(dev_.get()), vocab_, n_, d_tokens, count, d_out, s);
}

bool NativeEmbed::gather_one(int64_t token, float* out) const {
    if (!dev_ || !host_ || !out || token < 0 || token >= vocab_) return false;
    std::memcpy(out, host_ + token * n_, (size_t)n_ * 4);
    return true;
}

NativeHead::~NativeHead() { if (dev_) { DevScope sc(ord_); dev_.reset(); } }

bool NativeHead::load(const uint16_t* host_w, int64_t vocab, int64_t n, int ordinal, std::string* err) {
    if (dev_) { DevScope sc(ord_); dev_.reset(); }
    vocab_ = n_ = 0;
    if (!host_w || !mul_fits(vocab, n, 2)) return set_err(err, "bad head weights");
    DevScope sc(ordinal);
    if (!sc.ok) return set_err(err, "cannot select device");
    if (!upload(dev_, host_w, (uint64_t)vocab * (uint64_t)n * 2, err)) return false;
    vocab_ = vocab; n_ = n; ord_ = ordinal;
    return true;
}

bool NativeHead::run(const float* x, float* logits) const {
    if (!dev_ || !x || !logits) return false;
    DevScope sc(ord_);
    if (!sc.ok) return false;
    float *dx = nullptr, *dy = nullptr; bool ok = false;
    if (cudaMalloc((void**)&dx, (size_t)n_ * 4) == cudaSuccess && cudaMalloc((void**)&dy, (size_t)vocab_ * 4) == cudaSuccess &&
        cudaMemcpy(dx, x, (size_t)n_ * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
        bf16_gemv(static_cast<const uint16_t*>(dev_.get()), dx, vocab_, n_, dy) &&
        cudaMemcpy(logits, dy, (size_t)vocab_ * 4, cudaMemcpyDeviceToHost) == cudaSuccess) ok = true;
    if (!ok) cudaGetLastError();
    cudaFree(dx); cudaFree(dy);
    return ok;
}

// embedding gather kernel lives here (embed_head.cpp is compiled as CUDA): keeps kernels.cu free of __global__ definitions.
__global__ void embed_gather_kernel(const float* table, int64_t vocab, int64_t n, const int32_t* tokens, int64_t count, float* out) {
    const int64_t row = blockIdx.x;
    if (row >= count) return;
    const int32_t t = tokens[row];
    const bool valid = t >= 0 && (int64_t)t < vocab;
    for (int64_t c = threadIdx.x; c < n; c += blockDim.x) out[row * n + c] = valid ? table[(int64_t)t * n + c] : 0.f;
}

bool embed_gather(const float* table, int64_t vocab, int64_t n, const int32_t* tokens, int64_t count, float* out, cudaStream_t s) {
    if (vocab <= 0 || n <= 0 || count < 0 || count > 2147483647ll || !table) return false;
    if (count == 0) return true;
    if (!tokens || !out) return false;
    embed_gather_kernel<<<(unsigned)count, 256, 0, s>>>(table, vocab, n, tokens, count, out);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace rt
}  // namespace baloon
