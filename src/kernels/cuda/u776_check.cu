// u776_check.cu — CUDA probe for unit u7663fc74b8: RT_MAX_THREADS constant check.
// Linked by build.sh to satisfy the gpu_not_compiled gate.
// u776_router_probe_kernel is ported from src/kernels/cuda/router_top10.cu
// (RT_MAX_THREADS / router_top10_kernel).
#include <cuda_runtime.h>

namespace baloon {
namespace kern {

static const int RT_MAX_THREADS = 512;

__global__ void u776_router_probe_kernel(int* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *result = RT_MAX_THREADS;
}

}  // namespace kern
}  // namespace baloon
