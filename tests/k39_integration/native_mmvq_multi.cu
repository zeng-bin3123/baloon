// native_mmvq_multi.cu — multi-column MMVQ smoke test for ix_mmvq_wave.cu on ivcore10 (BI-V100).
// Tests ix_mmvq::ix_mmvq_q8_0 for ncols 1..8 with constant-valued weights/activations, checks
// numerical correctness: for all-ones quants and unit scales the output equals n_in for every element.
// Accepts --selftest flag (ignored; always runs full test).
// exit 0 + "selftest: ok" on pass; exit 1 on failure.
#include "../../src/ix_mmvq_wave.cu"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){                     \
    std::fprintf(stderr,"CUDA err %s at %s:%d\n",cudaGetErrorString(_e),         \
                 __FILE__,__LINE__); std::exit(2); } } while(0)
#define T(cond) do { if(cond){++g_pass;}else{++g_fail;                           \
    std::fprintf(stderr,"FAIL %s:%d\n",__FILE__,__LINE__);} } while(0)

// Q8_0 block: 2-byte f16 scale + 32 int8 quants = 34 bytes.
// Sets scale = f16(1.0) = 0x3C00, all quants = q.
static void make_q8_0(uint8_t* blk, int8_t q) {
    blk[0] = 0x00; blk[1] = 0x3C;                           // f16(1.0) LE
    for (int j = 0; j < 32; ++j) blk[2 + j] = (uint8_t)q;
}

// Q8_1 activation block: half2(d,s) + 32 int8 quants = 36 bytes.
// Sets d = f16(1.0) (the scale used by Q80Traits::apply), quants = q.
static void make_q8_1(uint8_t* blk, int8_t q) {
    blk[0] = 0x00; blk[1] = 0x3C;   // d = f16(1.0) LE
    blk[2] = 0x00; blk[3] = 0x50;   // s = f16(32.0) LE (not read by Q8_0 path)
    for (int j = 0; j < 32; ++j) blk[4 + j] = (uint8_t)q;
}

// Run one (n_in, n_out, ncols) configuration.  All quants = 1, all scales = 1.
// Expected output: every element = n_in  (n_blocks blocks * 32 quants * 1*1 = n_in).
static void test_ncols(int n_in, int n_out, int ncols) {
    const int q8_0_bytes = 34, q8_1_bytes = 36;
    const int n_blocks = n_in / 32;

    // Build host-side weight matrix: n_out rows, each n_blocks Q8_0 blocks.
    std::vector<uint8_t> wh((size_t)n_out * n_blocks * q8_0_bytes);
    for (int r = 0; r < n_out; ++r)
        for (int b = 0; b < n_blocks; ++b)
            make_q8_0(wh.data() + ((size_t)r * n_blocks + b) * q8_0_bytes, 1);

    // Build host-side activation matrix: ncols columns, each n_blocks Q8_1 blocks.
    std::vector<uint8_t> xh((size_t)ncols * n_blocks * q8_1_bytes);
    for (int c = 0; c < ncols; ++c)
        for (int b = 0; b < n_blocks; ++b)
            make_q8_1(xh.data() + ((size_t)c * n_blocks + b) * q8_1_bytes, 1);

    // Allocate device buffers.
    uint8_t* dw = nullptr; uint8_t* dx = nullptr; float* dy = nullptr;
    CK(cudaMalloc((void**)&dw, wh.size()));
    CK(cudaMalloc((void**)&dx, xh.size()));
    CK(cudaMalloc((void**)&dy, (size_t)n_out * ncols * sizeof(float)));
    CK(cudaMemcpy(dw, wh.data(), wh.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, xh.data(), xh.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(dy, 0, (size_t)n_out * ncols * sizeof(float)));

    // Run the kernel.
    cudaError_t e = ix_mmvq::ix_mmvq_q8_0(dw, dx, dy, n_in, n_out, ncols, 0);
    CK(cudaDeviceSynchronize());

    char nm[64];
    std::snprintf(nm, sizeof(nm), "ix_mmvq_q8_0 n_in=%d n_out=%d ncols=%d launch", n_in, n_out, ncols);
    T(e == cudaSuccess);

    // Check output: every element should equal n_in (1*1 per quant * 32 quants/block * n_blocks blocks).
    std::vector<float> yh((size_t)n_out * ncols);
    CK(cudaMemcpy(yh.data(), dy, yh.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const float expected = (float)n_in;
    for (int c = 0; c < ncols; ++c) {
        for (int r = 0; r < n_out; ++r) {
            float got = yh[(size_t)c * n_out + r];
            if (std::fabsf(got - expected) > 0.5f) {
                ++g_fail;
                std::fprintf(stderr, "FAIL ncols=%d col=%d row=%d: got=%.1f expected=%.1f\n",
                             ncols, c, r, got, expected);
            } else {
                ++g_pass;
            }
        }
    }

    cudaFree(dw); cudaFree(dx); cudaFree(dy);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;   // --selftest accepted, ignored
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::printf("native_mmvq_multi: no GPU device\n"); return 2;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("native_mmvq_multi: cannot open GPU 0\n"); return 2;
    }
    // Test ncols 1..8 with n_in=64 (2 Q8_0 blocks), n_out=4 — basic smoke check.
    for (int nc = 1; nc <= 8; ++nc) test_ncols(64, 4, nc);
    // Larger n_in to cover multi-block accumulation.
    for (int nc = 1; nc <= 4; ++nc) test_ncols(256, 8, nc);
    std::printf("native_mmvq_multi: pass=%d fail=%d\n", g_pass, g_fail);
    if (g_fail > 0) { std::printf("native_mmvq_multi FAILED\n"); return 1; }
    std::printf("native_mmvq_multi selftest: ok\n"); return 0;
}
