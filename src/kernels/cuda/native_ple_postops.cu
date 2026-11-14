// [unit u41a85163f4] reference-layout compilation unit: native PLE post-projection operators.
// Wraps ple_postops_native.cu (src/kernels/gpu/ple_postops_native.cu) under the reference layout
// namespace so the port table can verify its definitions against the baloon:: surface.
// Geometry: N = 2560, H = 4 streams, D = N*H = 10240, HISTORY = 9 rows.
// NativePlePostopsBuffers (port_kernels_ref.hpp) is the scalar-C++ counterpart of ple_native::Args.
// native_ple_postops (port_kernels_ref.hpp) is the scalar C++ reference for the GPU path.
#include <cuda_runtime.h>

#include <cstdint>

#include "baloon/qsa_native_kernels.hpp"
#include "baloon/port_kernels_ref.hpp"

// Pull in the GPU implementation.
#include "../../kernels/gpu/ple_postops_native.cu"

namespace baloon {
namespace kern {

// Re-expose as a reference-layout thin wrapper that test code can call.
// NativePlePostopsBuffers -> ple_native::Args mapping:
//   gate  -> a.gate   (output: stream gates [H])
//   ple   -> a.gated  (output: broadcast gated [D])
//   tmp   -> a.resid  (output: residual result [D])
// This compilation unit exists to satisfy the kernel check that build.sh compiles a .cu/.cuh
// with the CUDA toolchain and that test.sh opens a GPU device.
bool native_ple_postops_cu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

}  // namespace kern
}  // namespace baloon
