// Phase 0 capability probe for BI-V100 (ivcore10). Build once per -DPROBE_<NAME> (see build_probe_phase0.sh).
// Prints one line: PROBE <name> <PASS|FAIL> <detail>. A build failure is reported by the script as COMPILE_FAIL.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#ifdef PROBE_MMA
#include <mma.h>
#include <cuda_fp16.h>
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("PROBE %s FAIL cuda_error=%s\n", PNAME, cudaGetErrorString(e)); return 1; } } while (0)

#if defined(PROBE_DEVPROPS)
#define PNAME "devprops"
int main() {
  int n = 0; CK(cudaGetDeviceCount(&n));
  printf("PROBE devprops PASS device_count=%d\n", n);
  for (int i = 0; i < n; i++) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, i));
    printf("DEV %d name=\"%s\" cc=%d.%d sm_count=%d warpSize=%d smem_per_block=%zu smem_per_sm=%zu "
           "regs_per_block=%d max_threads_per_block=%d max_threads_per_sm=%d clock_khz=%d "
           "mem_bus_bits=%d mem_clock_khz=%d global_mem_mb=%zu l2_bytes=%d\n",
           i, p.name, p.major, p.minor, p.multiProcessorCount, p.warpSize, p.sharedMemPerBlock,
           p.sharedMemPerMultiprocessor, p.regsPerBlock, p.maxThreadsPerBlock,
           p.maxThreadsPerMultiProcessor, p.clockRate, p.memoryBusWidth, p.memoryClockRate,
           p.totalGlobalMem >> 20, p.l2CacheSize);
  }
  return 0;
}

#elif defined(PROBE_SHFL64)
#define PNAME "shfl64"
__global__ void k(int *out) { int v = threadIdx.x; for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64); out[threadIdx.x] = v; }
int main() {
  int *d, h[64]; CK(cudaMalloc(&d, sizeof h)); k<<<1, 64>>>(d); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  int ok = 1; for (int i = 0; i < 64; i++) if (h[i] != 2016) ok = 0;   // sum 0..63
  printf("PROBE shfl64 %s lane0=%d expect=2016\n", ok ? "PASS" : "FAIL", h[0]);
  return !ok;
}

#elif defined(PROBE_DP4A)
#define PNAME "dp4a"
__global__ void k(int *out) {
  int a = 0x01020304, b = 0x05060708;            // signed bytes: 1*5+2*6+3*7+4*8 = 70
  out[0] = __dp4a(a, b, 100);                    // expect 170
}
int main() {
  int *d, h = 0; CK(cudaMalloc(&d, 4)); k<<<1, 1>>>(d); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  printf("PROBE dp4a %s got=%d expect=170\n", h == 170 ? "PASS" : "FAIL", h);
  return h != 170;
}

#elif defined(PROBE_MATCH_ANY)
#define PNAME "match_any_sync"
__global__ void k(unsigned long long *out) { unsigned long long m = __match_any_sync(0xffffffffffffffffULL, (int)(threadIdx.x & 3)); out[threadIdx.x] = m; }
int main() {
  unsigned long long *d, h[64]; CK(cudaMalloc(&d, sizeof h)); k<<<1, 64>>>(d); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  int ok = 1; for (int i = 0; i < 64; i++) if (__builtin_popcountll(h[i]) != 16) ok = 0;   // 4 groups of 16 lanes
  printf("PROBE match_any_sync %s lane0_mask=0x%llx\n", ok ? "PASS" : "FAIL", h[0]);
  return !ok;
}

#elif defined(PROBE_BALLOT64)
#define PNAME "ballot64"
__global__ void k(unsigned long long *out) { out[0] = __ballot(threadIdx.x >= 32); }
int main() {
  unsigned long long *d, h = 0; CK(cudaMalloc(&d, 8)); k<<<1, 64>>>(d); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost));
  printf("PROBE ballot64 %s mask=0x%llx expect=0xffffffff00000000\n", h == 0xffffffff00000000ULL ? "PASS" : "FAIL", h);
  return h != 0xffffffff00000000ULL;
}

#elif defined(PROBE_MMA)
#define PNAME "wmma_fp16_16x16x16"
using namespace nvcuda;
__global__ void k(float *out) {
  __shared__ half A[256], B[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) { A[i] = __float2half(1.0f); B[i] = __float2half(2.0f); }
  __syncthreads();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f); wmma::load_matrix_sync(a, A, 16); wmma::load_matrix_sync(b, B, 16);
  wmma::mma_sync(c, a, b, c); wmma::store_matrix_sync(out, c, 16, wmma::mem_row_major);
}
int main() {
  float *d, h[256]; CK(cudaMalloc(&d, sizeof h)); k<<<1, 64>>>(d); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  int ok = 1; for (int i = 0; i < 256; i++) if (h[i] != 32.0f) ok = 0;   // 16 * 1 * 2
  printf("PROBE wmma_fp16_16x16x16 %s c00=%g expect=32\n", ok ? "PASS" : "FAIL", h[0]);
  return !ok;
}

#elif defined(PROBE_THREADFENCE_SYSTEM)
#define PNAME "threadfence_system"
__global__ void k(int *o) { o[0] = 1; __threadfence_system(); o[1] = 2; }
int main() { int *d; CK(cudaMalloc(&d, 8)); k<<<1, 1>>>(d); CK(cudaDeviceSynchronize()); printf("PROBE threadfence_system PASS (ran; docs say unsupported - verify no hang)\n"); return 0; }

#elif defined(PROBE_NANOSLEEP)
#define PNAME "nanosleep"
__global__ void k(int *o) { __nanosleep(100); o[0] = 1; }
int main() { int *d; CK(cudaMalloc(&d, 4)); k<<<1, 1>>>(d); CK(cudaDeviceSynchronize()); printf("PROBE nanosleep PASS\n"); return 0; }

#elif defined(PROBE_CUDA_GRAPH)
#define PNAME "cuda_graph"
__global__ void k(int *o) { atomicAdd(o, 1); }
int main() {
  int *d, h = 0; CK(cudaMalloc(&d, 4)); CK(cudaMemset(d, 0, 4));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal)); k<<<1, 1, 0, s>>>(d); k<<<1, 1, 0, s>>>(d);
  CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
  for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s)); CK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  printf("PROBE cuda_graph %s counter=%d expect=6\n", h == 6 ? "PASS" : "FAIL", h);
  return h != 6;
}

#elif defined(PROBE_HBM_BW)
#define PNAME "hbm_bw"
__global__ void cp(const float4 *a, float4 *b, size_t n) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i]; }
int main() {
  size_t bytes = 1ull << 30, n = bytes / 16; float4 *a, *b; CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes));
  CK(cudaMemset(a, 1, bytes)); cp<<<1024, 256>>>(a, b, n); CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  for (int i = 0; i < 10; i++) cp<<<1024, 256>>>(a, b, n);
  cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1);
  printf("PROBE hbm_bw PASS copy_GBps=%.1f (read+write counted, 1GiB x10)\n", 10.0 * 2 * bytes / (ms * 1e-3) / 1e9);
  return 0;
}
#else
#error "define one PROBE_* macro"
#endif
