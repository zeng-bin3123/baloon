// gemm.cu - device implementation of baloon::pfdev::Gemm (see include/baloon/prefill_gemm_dev.hpp).
// One 64x64 output tile per block of 256 threads (16x16, each thread a 4x4 register tile), reduction in 32-wide slices that
// ascend in k, so every output element is a single ascending-k f32 sum. Out-of-range tile entries load as zero (the zero
// padding of pfgemm's tile geometry) and are never stored. Wave64-safe: no warp-level operations are used.
#include "baloon/prefill_gemm_dev.hpp"

#include <cstdio>
#include <cstring>

#include "baloon/f16_bits.hpp"

namespace baloon {
namespace pfdev {

namespace {
thread_local char g_err[256] = "";
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::snprintf(g_err, sizeof g_err, "%s: %s", what ? what : "cuda", cudaGetErrorString(e));
    return false;
}
const char* ck_last() { return g_err; }
void ck_clear() { g_err[0] = 0; }

namespace {

constexpr int TM = (int)pfgemm::kTileM, TN = (int)pfgemm::kTileN, TK = (int)pfgemm::kTileK;   // 64, 64, 32
constexpr int TX = 16, TY = 16, REG = 4;                                                       // 16 * 4 = 64
constexpr uint64_t kRowsPerLaunch = 1ull << 16;                                                // gridDim.y <= 1024 (safe on every device)

struct LdF32 {
    static constexpr uint64_t bytes = 4;
    __device__ static float get(const void* p, uint64_t i) { return ((const float*)p)[i]; }
};
struct LdBf16 {
    static constexpr uint64_t bytes = 2;
    __device__ static float get(const void* p, uint64_t i) { return f16bits::bf16_bits_to_f32(((const uint16_t*)p)[i]); }
};
struct LdF16 {
    static constexpr uint64_t bytes = 2;
    __device__ static float get(const void* p, uint64_t i) { return f16bits::f32_from_f16(((const uint16_t*)p)[i]); }
};

template <class LA, class LW>
__global__ void gemm_tile_kernel(const void* A, const void* W, float* C, uint64_t m, uint64_t n, uint64_t k) {
    __shared__ float As[TK][TM + 1];
    __shared__ float Ws[TK][TN + 1];
    const int tid = threadIdx.y * TX + threadIdx.x;
    const uint64_t row0 = (uint64_t)blockIdx.y * TM, col0 = (uint64_t)blockIdx.x * TN;
    float acc[REG][REG];
#pragma unroll
    for (int i = 0; i < REG; ++i)
#pragma unroll
        for (int j = 0; j < REG; ++j) acc[i][j] = 0.f;

    for (uint64_t k0 = 0; k0 < k; k0 += TK) {
        for (int e = tid; e < TM * TK; e += TX * TY) {
            const int mi = e / TK, kk = e % TK;
            const uint64_t c = k0 + kk;
            const uint64_t r = row0 + mi, q = col0 + mi;
            As[kk][mi] = (r < m && c < k) ? LA::get(A, r * k + c) : 0.f;
            Ws[kk][mi] = (q < n && c < k) ? LW::get(W, q * k + c) : 0.f;
        }
        __syncthreads();
        for (int kk = 0; kk < TK; ++kk) {
            float a[REG], w[REG];
#pragma unroll
            for (int i = 0; i < REG; ++i) a[i] = As[kk][threadIdx.y + TY * i];
#pragma unroll
            for (int j = 0; j < REG; ++j) w[j] = Ws[kk][threadIdx.x + TX * j];
#pragma unroll
            for (int i = 0; i < REG; ++i)
#pragma unroll
                for (int j = 0; j < REG; ++j) acc[i][j] += a[i] * w[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < REG; ++i) {
        const uint64_t r = row0 + threadIdx.y + TY * i;
#pragma unroll
        for (int j = 0; j < REG; ++j) {
            const uint64_t c = col0 + threadIdx.x + TX * j;
            if (r < m && c < n) C[r * n + c] = acc[i][j];
        }
    }
}

template <bool BF>
__global__ void to_half_kernel(const float* src, uint16_t* dst, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = BF ? f16bits::round_f32_to_bf16_bits(src[i]) : f16bits::f16_from_f32(src[i]);
}

template <class LA, class LW>
bool launch_tiles(const void* a, const void* w, float* out, uint64_t m, uint64_t n, uint64_t k, cudaStream_t s) {
    const uint64_t gx = (n + TN - 1) / TN;
    if (gx > 2147483647ull) return false;
    for (uint64_t r0 = 0; r0 < m; r0 += kRowsPerLaunch) {
        const uint64_t rows = (m - r0 < kRowsPerLaunch) ? m - r0 : kRowsPerLaunch;
        const dim3 grid((unsigned)gx, (unsigned)((rows + TM - 1) / TM));
        const void* a_rows = (const char*)a + r0 * k * LA::bytes;
        gemm_tile_kernel<LA, LW><<<grid, dim3(TX, TY), 0, s>>>(a_rows, w, out + r0 * n, rows, n, k);
        if (!ck(cudaGetLastError(), "gemm_tile_kernel launch")) return false;
    }
    return true;
}

bool convert_rows(const float* src, uint16_t* dst, uint64_t count, bool bf, cudaStream_t s) {
    const uint64_t blocks = (count + 255) / 256;
    if (blocks > 2147483647ull) return false;
    if (bf) to_half_kernel<true><<<(unsigned)blocks, 256, 0, s>>>(src, dst, count);
    else to_half_kernel<false><<<(unsigned)blocks, 256, 0, s>>>(src, dst, count);
    return ck(cudaGetLastError(), "to_half_kernel launch");
}

}  // namespace

Gemm::~Gemm() { release(); }

void Gemm::release() {
    if (owned_) {
        if (ws_) ck(cudaFree(ws_), "cudaFree workspace");
        if (stream_) ck(cudaStreamDestroy(stream_), "cudaStreamDestroy");
    }
    ready_ = false; owned_ = false; ws_ = nullptr; ws_bytes_ = 0; stream_ = 0;
}

bool Gemm::init(int device, uint64_t workspace_bytes) {
    release();
    if (!ck(cudaSetDevice(device), "cudaSetDevice")) return false;
    void* ws = nullptr;
    if (workspace_bytes > 0 && !ck(cudaMalloc(&ws, (size_t)workspace_bytes), "cudaMalloc workspace")) return false;
    cudaStream_t st = 0;
    if (!ck(cudaStreamCreate(&st), "cudaStreamCreate")) { if (ws) cudaFree(ws); return false; }
    device_ = device; ws_ = ws; ws_bytes_ = workspace_bytes; stream_ = st; owned_ = true; ready_ = true;
    return true;
}

bool Gemm::init_external(void* workspace, uint64_t bytes, cudaStream_t stream) {
    release();
    if (bytes > 0 && (workspace == nullptr || ((uintptr_t)workspace & 15u) != 0)) return false;
    if (bytes == 0 && workspace != nullptr) workspace = nullptr;
    int dev = 0;
    if (!ck(cudaGetDevice(&dev), "cudaGetDevice")) return false;
    device_ = dev; ws_ = workspace; ws_bytes_ = bytes; stream_ = stream; owned_ = false; ready_ = true;
    return true;
}

bool Gemm::sync() {
    if (!ready_) return false;
    return ck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

bool Gemm::native(const float* a, const float* w, float* out, uint64_t m, uint64_t n, uint64_t k) {
    if (!ready_ || !a || !w || !out) return false;
    pfgemm::Gemm d; d.m = m; d.n = n; d.k = k; d.a_type = Elem::F32; d.w_type = Elem::F32;
    if (!d.valid()) return false;
    return launch_tiles<LdF32, LdF32>(a, w, out, m, n, k, stream_);
}

bool Gemm::run(const void* a, Elem a_type, Elem w_type, const void* w, float* out, uint64_t m, uint64_t n, uint64_t k) {
    if (!ready_ || !a || !w || !out) return false;
    if (a_type != Elem::F32 && a_type != w_type) return false;
    pfgemm::Gemm d; d.m = m; d.n = n; d.k = k; d.a_type = a_type; d.w_type = w_type;
    if (!d.valid()) return false;
    const bool bf = (w_type == Elem::BF16);
    if (a_type == Elem::F32) {
        if (d.scratch_bytes() > ws_bytes_ || ws_ == nullptr) return false;
        uint16_t* scratch = (uint16_t*)ws_;
        if (!convert_rows((const float*)a, scratch, m * k, bf, stream_)) return false;
        return bf ? launch_tiles<LdBf16, LdBf16>(scratch, w, out, m, n, k, stream_)
                  : launch_tiles<LdF16, LdF16>(scratch, w, out, m, n, k, stream_);
    }
    return bf ? launch_tiles<LdBf16, LdBf16>(a, w, out, m, n, k, stream_)
              : launch_tiles<LdF16, LdF16>(a, w, out, m, n, k, stream_);
}

bool Gemm::bf16(const void* a, Elem a_type, const uint16_t* w, float* out, uint64_t m, uint64_t n, uint64_t k) {
    return run(a, a_type, Elem::BF16, w, out, m, n, k);
}

bool Gemm::f16(const void* a, Elem a_type, const uint16_t* w, float* out, uint64_t m, uint64_t n, uint64_t k) {
    return run(a, a_type, Elem::F16, w, out, m, n, k);
}

}  // namespace pfdev
}  // namespace baloon
