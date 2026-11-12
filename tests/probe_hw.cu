// ENV.md "### Hardware probe (compile & run)" compiles probe_hw.cu, which was not in the repo.
// Prints the hard facts ENV.md claims, one key=value per line (parsed by tools/env_probe.py):
//   devices, runtime (CUDART version, 10020 = 10.2), driver, and per device dN.name / dN.mem_gib / dN.sm /
//   dN.warp / dN.smem_kib / dN.clock_mhz / dN.cc, plus pN-M = cudaDeviceCanAccessPeer (PCIe peer path).
// Also launches one kernel per device that reads the warpSize builtin on the GPU (dN.warp_dev), so the
// "warpSize: 64" claim is measured from device code, not only from the property struct.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void k_warp(int* out) {
  if (threadIdx.x == 0) *out = warpSize;
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("error=%s at %s\n", cudaGetErrorString(e), #x); return 1; } } while (0)

int main() {
  int n = 0, rt = 0, drv = 0;
  CK(cudaGetDeviceCount(&n));
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&drv);
  printf("devices=%d\nruntime=%d\ndriver=%d\n", n, rt, drv);
  for (int d = 0; d < n; ++d) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, d));
    printf("d%d.name=%s\n", d, p.name);
    printf("d%d.mem_gib=%.2f\n", d, p.totalGlobalMem / 1073741824.0);
    printf("d%d.sm=%d\n", d, p.multiProcessorCount);
    printf("d%d.warp=%d\n", d, p.warpSize);
    printf("d%d.smem_kib=%zu\n", d, p.sharedMemPerBlock / 1024);
    printf("d%d.clock_mhz=%d\n", d, p.clockRate / 1000);
    printf("d%d.cc=%d.%d\n", d, p.major, p.minor);
    CK(cudaSetDevice(d));
    int* dw = nullptr; int hw = -1;
    CK(cudaMalloc(&dw, sizeof(int)));
    k_warp<<<1, 64>>>(dw);
    CK(cudaGetLastError());
    CK(cudaMemcpy(&hw, dw, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaFree(dw));
    printf("d%d.warp_dev=%d\n", d, hw);
  }
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
      if (a != b) { int ok = 0; cudaDeviceCanAccessPeer(&ok, a, b); printf("p%d-%d=%d\n", a, b, ok); }
  return 0;
}
