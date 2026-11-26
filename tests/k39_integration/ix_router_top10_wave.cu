// tests/k39_integration/ix_router_top10_wave.cu — launcher declaration for router_top10_wave.
// router_top10_parity.cu and moe_parity.cu include this file to call the host launcher
// ix_router_top10_wave() on the BI-V100 without redefining the kernel in a test TU.
//
// The kernel (router_top10_wave_kernel) and its launcher (ix_rw::ix_router_top10_wave)
// are compiled into KG via src/kernels/gpu/moe_wave_router.cu; this shim provides only
// the extern declaration so test code can call the ported kernel through the linker.
#pragma once
#include <cuda_runtime.h>

namespace ix_rw {
// Forward declaration of the host launcher (defined in src/ix_router_top10_wave.cu,
// compiled into KG via src/kernels/gpu/moe_wave_router.cu).
cudaError_t ix_router_top10_wave(const float* logits, int n_tokens, int n_expert, int k,
                                  int* ids, float* weights, cudaStream_t s = 0);
}  // namespace ix_rw

#if defined(__ILUVATAR__)
// Bring the namespace-qualified launcher into the enclosing (test) scope so
// test files can call ix_router_top10_wave() without the ix_rw:: qualifier.
using ix_rw::ix_router_top10_wave;
#endif  // __ILUVATAR__
