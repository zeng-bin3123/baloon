// ua96_geometry.cu - launch-geometry coverage kernels (see include/baloon/ua96_geometry.hpp).
// Lane/warp numbers are derived from threadIdx explicitly with a width of 32 (the theory's warp), so the maps are
// independent of the hardware wave width (64 on ivcore10).
#include "baloon/ua96_geometry.hpp"

namespace baloon {
namespace ua96 {
namespace {

__device__ inline void claim(uint32_t* cnt, long idx, long n, uint32_t* oob) {
    if (idx < 0 || idx >= n) { atomicAdd(oob, 1u); return; }
    atomicAdd(&cnt[idx], 1u);
}

// down kernel: 40 blocks x 256 threads = 8 virtual warps; row = blk*8 + warp, one claim per warp (lane 0)
__global__ void down_cover_kernel(uint32_t* cnt, uint32_t* oob) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) claim(cnt, (long)blockIdx.x * 8 + warp, kLr, oob);
}

// up kernel: 80 blocks x 128 threads; thread t = stream*32 + dd owns stacked row stream*N + blk*32 + dd
__global__ void up_cover_kernel(uint32_t* cnt, uint32_t* oob) {
    const int stream = threadIdx.x / kUpCols, dd = threadIdx.x % kUpCols;
    claim(cnt, (long)stream * kN + (long)blockIdx.x * kUpCols + dd, kD, oob);
}

// multi-token up kernel: 160 blocks x 64 threads; 4 streams x 16 columns
__global__ void upm_cover_kernel(uint32_t* cnt, uint32_t* oob) {
    const int stream = threadIdx.x / kUpmCols, dd = threadIdx.x % kUpmCols;
    claim(cnt, (long)stream * kN + (long)blockIdx.x * kUpmCols + dd, kD, oob);
}

// gate kernel: ceil(D/8) blocks of 8 virtual warps, row = blk*8 + warp (rows >= D are skipped by the kernel's guard)
__global__ void gate_cover_kernel(uint32_t* cnt, uint32_t* oob) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const long row = (long)blockIdx.x * 8 + warp;
    if (lane == 0 && row < kD) claim(cnt, row, kD, oob);
}

// GDN step: S*RG = 512 threads per block; thread t -> (row group t/128, column t%128), 32 registers per thread
// rowOf(rg, r) = rg*32 + r; every (row, column) cell must be written by exactly one (thread, register)
__global__ void step_cover_kernel(uint32_t* cnt, uint32_t* oob) {
    const int rg = threadIdx.x / kS, col = threadIdx.x % kS;
    for (int r = 0; r < kStepRpg; ++r) claim(cnt, (long)(rg * kStepRpg + r) * kS + col, (long)kS * kS, oob);
}

// conv: blocks of 128 threads over C channels, guarded by c < C
__global__ void conv_cover_kernel(uint32_t* cnt, uint32_t* oob, int C) {
    const int c = blockIdx.x * 128 + threadIdx.x;
    if (c < C) claim(cnt, c, C, oob);
}

}  // namespace

cudaError_t cover_all(const CoverBuffers& b, cudaStream_t s) {
    if (!b.down || !b.up || !b.upm || !b.gate || !b.step_rows || !b.conv || !b.oob) return cudaErrorInvalidValue;
    down_cover_kernel<<<kDownBlocks, 256, 0, s>>>(b.down, b.oob);
    up_cover_kernel<<<kUpBlocks, kHc * kUpCols, 0, s>>>(b.up, b.oob);
    upm_cover_kernel<<<kUpmBlocks, kHc * kUpmCols, 0, s>>>(b.upm, b.oob);
    gate_cover_kernel<<<(kD + 7) / 8, 256, 0, s>>>(b.gate, b.oob);
    step_cover_kernel<<<1, kS * kStepRg, 0, s>>>(b.step_rows, b.oob);
    conv_cover_kernel<<<(kConvChannels + 127) / 128, 128, 0, s>>>(b.conv, b.oob, kConvChannels);
    return cudaGetLastError();
}

}  // namespace ua96
}  // namespace baloon
