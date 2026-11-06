// ilu_embed_codes.cu - embedding column decode on the device. One CUDA block per requested row; threads stride over the
// elements, each reading the byte that holds its field and shifting by (i % perbyte) * code_bits.
#include <cuda_runtime.h>
#include <cstdint>
#include "baloon/embed_codes_kernels.hpp"

namespace baloon {
namespace embed_codes {
namespace {
constexpr int THREADS = 128;

__global__ void embed_codes_kernel(const uint8_t* __restrict__ table, int code_bits, int64_t vocab, int64_t n,
                                   const int32_t* __restrict__ ids, uint32_t* __restrict__ out, uint8_t* __restrict__ row_ok) {
    const int k = blockIdx.x;
    const int64_t tok = ids[k];
    const bool ok = tok >= 0 && tok < vocab;
    const int per = 8 / code_bits;
    const uint32_t mask = (1u << code_bits) - 1u;
    const uint8_t* row = table + (ok ? (size_t)tok * (size_t)(n / per) : 0);
    uint32_t* o = out + (size_t)k * (size_t)n;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const unsigned shift = (unsigned)(i % per) * (unsigned)code_bits;
        o[i] = ok ? (((uint32_t)row[(size_t)(i / per)] >> shift) & mask) : 0u;
    }
    if (row_ok && threadIdx.x == 0) row_ok[k] = ok ? 1 : 0;
}
}  // namespace

cudaError_t embed_codes_rows(const uint8_t* table, int code_bits, int64_t vocab, int64_t n, const int32_t* ids, int n_ids,
                             uint32_t* out, uint8_t* row_ok, cudaStream_t s) {
    const bool bits_ok = code_bits == 2 || code_bits == 4 || code_bits == 8;
    if (!table || !ids || !out || !bits_ok || vocab <= 0 || n <= 0 || n_ids < 0) return cudaErrorInvalidValue;
    if (n % (8 / code_bits) != 0) return cudaErrorInvalidValue;
    if (n_ids == 0) return cudaSuccess;
    embed_codes_kernel<<<n_ids, THREADS, 0, s>>>(table, code_bits, vocab, n, ids, out, row_ok);
    return cudaGetLastError();
}
}  // namespace embed_codes
}  // namespace baloon
