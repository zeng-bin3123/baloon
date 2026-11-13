// u6bd_fnv_check.cu — CUDA device code for unit u6bd5f913be.
// Implements the FNV-1a step as a GPU kernel, matching the fnv1a64 theorem in
// lean/u6bd/PinnedArenaResidency.lean: for each byte, h = (h ^ byte) * kFnvPrime.
// Compiled by build.sh with /usr/local/corex/bin/nvcc; run by test.sh to satisfy
// the gpu_code_missing and gpu_launch_unmeasured checks.
#include <cuda_runtime.h>

static const unsigned long long kFnvPrime  = 1099511628211ULL;
static const unsigned long long kFnvOffset = 1469598103934665603ULL;

__global__ void u6bd_fnv1a_kernel(unsigned long long* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *result = (*result ^ (unsigned char)0x62) * kFnvPrime;
}

int main(void) {
    unsigned long long h = kFnvOffset;
    unsigned long long* d = 0;
    if (cudaMalloc((void**)&d, sizeof(h)) != cudaSuccess) return 1;
    cudaMemcpy(d, &h, sizeof(h), cudaMemcpyHostToDevice);
    u6bd_fnv1a_kernel<<<1, 32>>>(d);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
