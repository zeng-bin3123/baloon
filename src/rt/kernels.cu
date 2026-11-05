// kernels.cu - bf16 gemv and embedding gather kernels.
#include "baloon/rt_kernels.hpp"

namespace baloon {
namespace rt {

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

__global__ void embed_gather_kernel(const float* table, int64_t vocab, int64_t n, const int32_t* tokens, int64_t count, float* out) {
    const int64_t row = blockIdx.x;
    if (row >= count) return;
    const int32_t t = tokens[row];
    const bool valid = t >= 0 && (int64_t)t < vocab;
    for (int64_t c = threadIdx.x; c < n; c += blockDim.x) out[row * n + c] = valid ? table[(int64_t)t * n + c] : 0.f;
}

bool bf16_gemv(const uint16_t* w, const float* x, int64_t rows, int64_t cols, float* out, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || rows > 2147483647ll || !w || !x || !out) return false;
    bf16_gemv_kernel<<<(unsigned)rows, 256, 0, s>>>(w, x, cols, out);
    return cudaGetLastError() == cudaSuccess;
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
