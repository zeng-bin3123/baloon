// ilu_f16_encode_top.cu - device evaluation of the fp16 encoder's top-level split (inf/NaN vs finite overflow) and of the
// full f16_from_f32 on raw fp32 bit patterns. Used to check that the device encoder classifies exactly like the model.
#include <cuda_runtime.h>
#include <cstdint>
#include "baloon/f16_encode_top.hpp"

namespace baloon {
namespace f16_top_dev {

static __global__ void f16_encode_top_kernel(const uint32_t* bits, int32_t* top, uint16_t* enc, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t x = bits[i];
    top[i] = f16bits::encode_top((x >> 16) & 0x8000u, (x >> 23) & 0xFFu, x & 0x7FFFFFu);
    enc[i] = f16bits::f16_from_f32(f16bits::float_of(x));
}

int f16_encode_top_step(const uint32_t* bits, int32_t* top, uint16_t* enc, int64_t n, void* stream) {
    if (n < 0 || (n > 0 && (!bits || !top || !enc))) return -1;
    if (n == 0) return 0;
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    if (blocks > 0x7FFFFFFF) return -1;
    f16_encode_top_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(bits, top, enc, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    e = cudaStreamSynchronize((cudaStream_t)stream);
    return e == cudaSuccess ? 0 : (int)e;
}

}  // namespace f16_top_dev
}  // namespace baloon
