// embed_head.cpp - NativeEmbed, NativeHead and the process-wide embedding binding of rt_native.hpp.
#include <cstring>
#include <vector>

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

NativeHead::~NativeHead() { if (dev_ || norm_dev_ || norm_buf_) { DevScope sc(ord_); dev_.reset(); norm_dev_.reset(); norm_buf_.reset(); } }

bool NativeHead::load(const uint16_t* host_w, int64_t vocab, int64_t n, int ordinal, std::string* err) {
    if (dev_ || norm_dev_ || norm_buf_) { DevScope sc(ord_); dev_.reset(); norm_dev_.reset(); norm_buf_.reset(); }
    vocab_ = n_ = 0;
    if (!host_w || !mul_fits(vocab, n, 2)) return set_err(err, "bad head weights");
    DevScope sc(ordinal);
    if (!sc.ok) return set_err(err, "cannot select device");
    if (!upload(dev_, host_w, (uint64_t)vocab * (uint64_t)n * 2, err)) return false;
    vocab_ = vocab; n_ = n; ord_ = ordinal;
    return true;
}

bool NativeHead::load_norm(const uint16_t* host_norm_w, int64_t n, std::string* err) {
    if (norm_dev_ || norm_buf_) { DevScope sc(ord_); norm_dev_.reset(); norm_buf_.reset(); }
    if (!host_norm_w || n <= 0 || n != n_) return set_err(err, "bad norm weight");
    // Convert BF16 -> f32 on host (n_embd=2048 -> 8 KiB, negligible cost).
    std::vector<float> host_f32((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        uint32_t u = (uint32_t)host_norm_w[i] << 16;
        float f; std::memcpy(&f, &u, 4);
        host_f32[(size_t)i] = f;
    }
    DevScope sc(ord_);
    if (!sc.ok) return set_err(err, "cannot select device");
    if (!upload(norm_dev_, host_f32.data(), (uint64_t)n * 4, err)) return false;
    // Pre-allocate norm scratch so run_dev() contains no cudaMalloc (required for CUDA graph capture).
    void* buf = nullptr;
    if (cudaMalloc(&buf, (size_t)n * 4) != cudaSuccess) { cudaGetLastError(); norm_dev_.reset(); return set_err(err, "norm_buf alloc failed"); }
    norm_buf_.reset(buf);
    return true;
}

// Single-row weighted RMS-norm: y[i] = x[i] * rsqrt(sum(x^2)/n + eps) * w[i].
// 256 threads, shared-memory reduction (matches the dense_bank pattern).
__global__ void rms_norm_one_kernel(const float* x, const float* w, float* y, int64_t n, float eps) {
    __shared__ float part[256];
    float acc = 0.f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(part[0] / (float)n + eps);
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

bool rms_norm_one(const float* x, const float* w, float* y, int64_t n, float eps, cudaStream_t s) {
    if (n <= 0 || !x || !w || !y) return false;
    rms_norm_one_kernel<<<1, 256, 0, s>>>(x, w, y, n, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool NativeHead::run_dev(const float* d_x, float* d_logits, cudaStream_t s) const {
    if (!dev_ || !d_x || !d_logits) return false;
    if (norm_dev_) {
        // applies model.language_model.norm using pre-alloc'd scratch (no cudaMalloc → graph-capturable)
        float* dxn = static_cast<float*>(norm_buf_.get());
        return rms_norm_one(d_x, static_cast<const float*>(norm_dev_.get()), dxn, n_, 1e-6f, s) &&
               bf16_gemv(static_cast<const uint16_t*>(dev_.get()), dxn, vocab_, n_, d_logits, s);
    }
    // projects through lm_head to the full 248 320-token vocabulary logit vector (no norm)
    return bf16_gemv(static_cast<const uint16_t*>(dev_.get()), d_x, vocab_, n_, d_logits, s);
}

bool NativeHead::run(const float* x, float* logits) const {
    if (!dev_ || !x || !logits) return false;
    DevScope sc(ord_);
    if (!sc.ok) return false;
    float *dx = nullptr, *dxn = nullptr, *dy = nullptr; bool ok = false;
    if (cudaMalloc((void**)&dx, (size_t)n_ * 4) != cudaSuccess) goto done;
    if (cudaMalloc((void**)&dy, (size_t)vocab_ * 4) != cudaSuccess) goto done;
    if (cudaMemcpy(dx, x, (size_t)n_ * 4, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    if (norm_dev_) {
        // applies model.language_model.norm: y = x * rsqrt(mean(x^2)+eps) * norm_w
        if (cudaMalloc((void**)&dxn, (size_t)n_ * 4) != cudaSuccess) goto done;
        if (!rms_norm_one(dx, static_cast<const float*>(norm_dev_.get()), dxn, n_)) goto done;
    }
    // projects through lm_head to the full 248 320-token vocabulary logit vector
    ok = bf16_gemv(static_cast<const uint16_t*>(dev_.get()),
                   norm_dev_ ? dxn : dx,
                   vocab_, n_, dy) &&
         cudaMemcpy(logits, dy, (size_t)vocab_ * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
done:
    if (!ok) cudaGetLastError();
    cudaFree(dx); cudaFree(dxn); cudaFree(dy);
    return ok;
}

// NativeLayerNorm: one input_layernorm or post_attention_layernorm weight for one layer.
// apply_inplace calls rms_norm_one with d_x as both input and output — the kernel is safe
// for x==y because each thread exclusively owns its index in the write pass.
NativeLayerNorm::~NativeLayerNorm() { if (dev_) { DevScope sc(ord_); dev_.reset(); } }

bool NativeLayerNorm::load(const uint16_t* host_w, int64_t n, int ordinal, std::string* err) {
    if (dev_) { DevScope sc(ord_); dev_.reset(); }
    n_ = 0;
    if (!host_w || n <= 0) return set_err(err, "bad layer norm weight");
    std::vector<float> host_f32((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        uint32_t u = (uint32_t)host_w[(size_t)i] << 16;
        float f; std::memcpy(&f, &u, 4);
        host_f32[(size_t)i] = f;
    }
    DevScope sc(ordinal);
    if (!sc.ok) return set_err(err, "cannot select device");
    if (!upload(dev_, host_f32.data(), (uint64_t)n * 4, err)) return false;
    n_ = n; ord_ = ordinal;
    return true;
}

bool NativeLayerNorm::apply_inplace(float* d_x, cudaStream_t s) const {
    if (!dev_ || !d_x) return false;
    // In-place: pass d_x as both x and y.  rms_norm_one_kernel first accumulates
    // sum(x^2) over all threads into shared memory, then writes y[i] = x[i]*inv*w[i].
    // Since each thread owns a disjoint index set, x==y is safe (no aliasing hazard).
    return rms_norm_one(d_x, static_cast<const float*>(dev_.get()), d_x, n_, 1e-6f, s);
}

NativeLayerNorm::~NativeLayerNorm() { if (dev_) { DevScope sc(ord_); dev_.reset(); } }

bool NativeLayerNorm::load(const uint16_t* host_w, int64_t n, int ordinal, std::string* err) {
    if (dev_) { DevScope sc(ord_); dev_.reset(); }
    n_ = 0;
    if (!host_w || n <= 0) return set_err(err, "bad layer norm weight");
    std::vector<float> host_f32((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        uint32_t u = (uint32_t)host_w[(size_t)i] << 16;
        float f; std::memcpy(&f, &u, 4);
        host_f32[(size_t)i] = f;
    }
    DevScope sc(ordinal);
    if (!sc.ok) return set_err(err, "cannot select device");
    if (!upload(dev_, host_f32.data(), (uint64_t)n * 4, err)) return false;
    n_ = n; ord_ = ordinal;
    return true;
}

bool NativeLayerNorm::apply_inplace(float* d_x, cudaStream_t s) const {
    if (!dev_ || !d_x) return false;
    // In-place: pass d_x as both x and y.  rms_norm_one_kernel first accumulates
    // sum(x^2) over all threads into shared memory, then writes y[i] = x[i]*inv*w[i].
    // Since each thread owns a disjoint index set, x==y is safe (no aliasing hazard).
    return rms_norm_one(d_x, static_cast<const float*>(dev_.get()), d_x, n_, 1e-6f, s);
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
