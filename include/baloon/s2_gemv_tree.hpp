// s2_gemv_tree.hpp - Q2 (2-bit) GEMV, one CUDA block per output row with a shared-memory halving tree reduction.
// Code plane: n_in / 4 bytes per row, four codes per byte (code k of byte b is (b >> 2k) & 3, value = code - 1).
// Scale plane: n_in / 64 fp16 scale bits per row (the scale of quad q is scales[q >> 4]).
// The decode table (byte -> four biased values) lives in constant memory. With `stage` the activation is staged in
// shared memory rounded to f16 (n_in <= kS2TreeMaxStageHalves); the result then equals the unstaged one applied to the
// f16-rounded activation.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace baloon {
namespace kern {

constexpr int kS2TreeMaxStageHalves = 4096;
constexpr int kS2TreeStageBytes = kS2TreeMaxStageHalves * 2;   // 8 KiB of staged halves
constexpr int kS2TreeMaxThreads = 1024;

// Dynamic shared memory of a launch.
constexpr size_t s2_tree_shared_bytes(bool stage, int threads_per_row) {
    return (stage ? (size_t)kS2TreeStageBytes : 0) + (size_t)threads_per_row * sizeof(float);
}
// threads_per_row: a power of two in [1, 1024].
constexpr bool s2_tree_threads_ok(int t) { return t >= 1 && t <= kS2TreeMaxThreads && (t & (t - 1)) == 0; }

cudaError_t s2_gemv_tree(const uint8_t* codes, const uint16_t* scales, const float* x, float* y, int n_in, int n_out,
                         int threads_per_row, bool stage, cudaStream_t stream);

}  // namespace kern
}  // namespace baloon
