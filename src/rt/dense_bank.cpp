// dense_bank.cpp - NativeDense of rt_native.hpp.
#include <algorithm>

#include "baloon/rt_kernels.hpp"
#include "baloon/rt_native.hpp"

namespace baloon {
namespace rt {

namespace {
bool set_err(std::string* err, const std::string& m) { if (err) *err = m; return false; }
struct DevScope {   // restores the current device on exit
    int prev = 0; bool ok = false;
    explicit DevScope(int ord) { cudaGetDevice(&prev); ok = cudaSetDevice(ord) == cudaSuccess; if (!ok) cudaGetLastError(); }
    ~DevScope() { cudaSetDevice(prev); }
};
}  // namespace

bool eligible(const core::WIndexRow& r) {
    return r.placed && r.engine_kind == 1 && r.dst_bytes > 0 && r.dst_bytes % 2 == 0;
}

NativeDense::~NativeDense() { clear(); }

void NativeDense::clear() {
    if (!e_.empty()) { DevScope sc(ord_); e_.clear(); }
    bytes_ = 0;
}

bool NativeDense::load(const core::WeightTable& t, const uint8_t* host_arena, const ShapeMap& shapes, int ordinal, std::string* err) {
    clear();
    if (!host_arena) return set_err(err, "no host arena");
    std::vector<Pending> pend;
    for (const auto& r : t.rows()) {
        if (!eligible(r)) continue;
        auto it = shapes.find(r.name);
        if (it == shapes.end()) continue;
        const uint64_t rows = it->second.first, cols = it->second.second;
        if (rows == 0 || cols == 0 || rows > UINT64_MAX / cols / 2 || rows * cols * 2 != r.dst_bytes)
            return set_err(err, "shape of " + r.name + " does not match its byte count");
        Pending p; p.name = r.name; p.host = reinterpret_cast<const uint16_t*>(host_arena + r.arena_off); p.rows = rows; p.cols = cols;
        pend.push_back(std::move(p));
    }
    DevScope sc(ordinal);
    if (!sc.ok) return set_err(err, "cannot select device");
    ord_ = ordinal;
    for (const auto& p : pend) {
        void* d = nullptr;
        const uint64_t nb = p.rows * p.cols * 2;
        if (cudaMalloc(&d, (size_t)nb) != cudaSuccess) { cudaGetLastError(); clear(); return set_err(err, "device allocation failed for " + p.name); }
        Entry en; en.buf.reset(d); en.rows = p.rows; en.cols = p.cols;
        if (cudaMemcpy(d, p.host, (size_t)nb, cudaMemcpyHostToDevice) != cudaSuccess) { cudaGetLastError(); clear(); return set_err(err, "upload failed for " + p.name); }
        e_[p.name] = std::move(en); bytes_ += nb;
    }
    return true;
}

std::vector<std::string> NativeDense::served_names() const {
    std::vector<std::string> v;
    for (const auto& kv : e_) v.push_back(kv.first);     // std::map iterates in sorted order
    return v;
}

bool NativeDense::find(const std::string& name, DenseView* out) const {
    auto it = e_.find(name);
    if (it == e_.end()) return false;
    if (out) { out->dev = static_cast<const uint16_t*>(it->second.buf.get()); out->rows = it->second.rows; out->cols = it->second.cols; }
    return true;
}

bool NativeDense::apply(const std::string& name, const float* d_x, float* d_out, cudaStream_t s) const {
    DenseView v;
    if (!find(name, &v)) return false;
    return bf16_gemv(v.dev, d_x, (int64_t)v.rows, (int64_t)v.cols, d_out, s);
}

// bf16 gemv kernel lives here (dense_bank.cpp is compiled as CUDA): keeps kernels.cu free of __global__ definitions.
__global__ void bf16_gemv_kernel(const uint16_t* w, const float* x, int64_t cols, float* out) {
    __shared__ float part[256];
    const uint16_t* row = w + (int64_t)blockIdx.x * cols;
    float acc = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        union { uint32_t u; float f; } cv; cv.u = (uint32_t)row[c] << 16;
        acc += cv.f * x[c];
    }
    part[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = part[0];
}

bool bf16_gemv(const uint16_t* w, const float* x, int64_t rows, int64_t cols, float* out, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || rows > 2147483647ll || !w || !x || !out) return false;
    bf16_gemv_kernel<<<(unsigned)rows, 256, 0, s>>>(w, x, cols, out);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace rt
}  // namespace baloon
