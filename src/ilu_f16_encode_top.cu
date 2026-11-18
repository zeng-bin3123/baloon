// ilu_f16_encode_top.cu - device-side f16 encoder top-level split.
// The __global__ kernel and GPU test driver live in
// tests/gpu_unit_f16bits_top/test_encode_top_gpu.cu (a test file).
// This compilation unit exposes the encode_top header to the CUDA toolchain.
#include "baloon/f16_encode_top.hpp"
