// Probe for Strata FNX_PF_PEER (deebb73a/8088d056) on BI-V100: can a kernel on card A store into card B's plain
// cudaMalloc memory, and is it visible to B after a flag? Strata uses AMD fine-grained memory + __threadfence_system +
// s_sleep; none of those exist here. Uses __threadfence() only. Run on the machine with /usr/local/corex (build_peer_probe.sh).
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); } } while (0)
__global__ void push(float* peer_box, volatile unsigned* peer_flag, int n, unsigned seq) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) peer_box[i] = 1.0f + i;
  __threadfence();            // device-level only; __threadfence_system is unsupported on ivcore10
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) *peer_flag = seq;   // NOTE: single block, so the flag may race other blocks' stores: this is the thing under test
}
__global__ void wait_check(const float* box, volatile unsigned* flag, int n, unsigned seq, unsigned* bad) {
  if (threadIdx.x == 0) { long long t0 = clock64(); while ((int)(*flag - seq) < 0) if (clock64() - t0 > 2000000000LL) { atomicAdd(bad, 1u << 31); break; } }
  __syncthreads(); __threadfence();
  for (int i = threadIdx.x; i < n; i += blockDim.x) if (((const volatile float*)box)[i] != 1.0f + i) atomicAdd(bad, 1u);
}
int main() {
  int nd = 0; CK(cudaGetDeviceCount(&nd)); printf("devices=%d warpSize check below\n", nd);
  for (int a = 0; a < nd; ++a) { for (int b = 0; b < nd; ++b) { int c = 0; if (a != b) CK(cudaDeviceCanAccessPeer(&c, a, b)); printf("%d ", a == b ? 9 : c); } printf("\n"); }
  const int N = 1 << 16; const int A = 0, B = 1; if (nd < 2) return 0;
  float* box; unsigned *flag, *bad; 
  CK(cudaSetDevice(B)); CK(cudaMalloc(&box, N * 4)); CK(cudaMalloc(&flag, 4)); CK(cudaMalloc(&bad, 4));
  CK(cudaMemset(flag, 0, 4)); CK(cudaMemset(bad, 0, 4)); CK(cudaMemset(box, 0, N * 4));
  CK(cudaSetDevice(A)); cudaError_t pe = cudaDeviceEnablePeerAccess(B, 0); printf("enable peer A->B: %s\n", cudaGetErrorString(pe));
  CK(cudaSetDevice(B)); pe = cudaDeviceEnablePeerAccess(A, 0); printf("enable peer B->A: %s\n", cudaGetErrorString(pe));
  for (unsigned seq = 1; seq <= 200; ++seq) {            // repeat: a race shows up as a nonzero count, not necessarily on round 1
    CK(cudaSetDevice(B)); CK(cudaMemset(box, 0, N * 4)); CK(cudaDeviceSynchronize());
    wait_check<<<1, 256>>>(box, flag, N, seq, bad);       // B waits first
    CK(cudaSetDevice(A)); push<<<64, 256>>>(box, flag, N, seq); CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(B)); CK(cudaDeviceSynchronize());
  }
  unsigned h = 0; CK(cudaSetDevice(B)); CK(cudaMemcpy(&h, bad, 4, cudaMemcpyDeviceToHost));
  printf("peer-store/flag test: bad=0x%x over 200 rounds (%s)\n", h, h ? "NOT SAFE: use cudaMemcpyPeerAsync + event barrier" : "ok");
  return 0;
}
