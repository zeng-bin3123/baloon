// gr_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
// Sentence 2 (top-k select expert GEMV): also compiles ix_mmvq_wave.cu native_mmvq_wave_kernel for Q8_0 expert GEMV on ivcore10.
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main

// Sentence 2: native_mmvq_wave_kernel for expert GEMV — compiled by COREX via CUFLAGS -D__ILUVATAR__
#include "../../src/ix_mmvq_wave.cu"

#ifdef __ILUVATAR__
// test_mmvq_wave_smoke: verifies ix_mmvq_q8_0 launches and produces a non-NaN result for Q8_0 weights.
// n_in=64 (2 Q8_0 blocks of 32), n_out=4, ncols=1.
static void test_mmvq_wave_smoke() {
    const int n_in = 64, n_out = 4, ncols = 1;
    // Build a minimal Q8_0 weight block: 34 bytes each (2-byte f16 scale + 32 int8 quants)
    const int q8_0_bytes = 34, n_blocks = n_in / 32;
    std::vector<uint8_t> wbuf(n_out * n_blocks * q8_0_bytes, 0);
    // Fill scale = 1.0f (f16 bits 0x3C00), quants = 1
    for (int o = 0; o < n_out; ++o)
        for (int b = 0; b < n_blocks; ++b) {
            uint8_t* blk = wbuf.data() + (o * n_blocks + b) * q8_0_bytes;
            blk[0] = 0x00; blk[1] = 0x3C;  // f16(1.0)
            for (int q = 2; q < q8_0_bytes; ++q) blk[q] = 1;
        }
    // Q8_1 activation: 36 bytes per block (ds: 2 half2, qs: 32 int8), all quants = 1, scale = 1
    const int q8_1_bytes = 36, x_blocks = n_in / 32;
    std::vector<uint8_t> xbuf(ncols * x_blocks * q8_1_bytes, 0);
    for (int col = 0; col < ncols; ++col)
        for (int b = 0; b < x_blocks; ++b) {
            uint8_t* blk = xbuf.data() + (col * x_blocks + b) * q8_1_bytes;
            // d = 1.0f (half bits 0x3C00), s = 32.0 (sum of 32 ones, stored as int16 in bytes 2-3)
            blk[0] = 0x00; blk[1] = 0x3C;  // d = f16(1.0)
            blk[2] = 0x00; blk[3] = 0x3C;  // s = f16(32.0) — approximate
            for (int q = 4; q < q8_1_bytes; ++q) blk[q] = 1;
        }
    void* d_w; void* d_x; float* d_y;
    cudaMalloc(&d_w, wbuf.size()); cudaMalloc(&d_x, xbuf.size());
    cudaMalloc((void**)&d_y, n_out * ncols * sizeof(float));
    cudaMemcpy(d_w, wbuf.data(), wbuf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, xbuf.data(), xbuf.size(), cudaMemcpyHostToDevice);
    cudaError_t e = ix_mmvq::ix_mmvq_q8_0(d_w, d_x, d_y, n_in, n_out, ncols);
    ok(e == cudaSuccess && sync_ok(), "ix_mmvq_q8_0 launches");
    std::vector<float> y(n_out);
    cudaMemcpy(y.data(), d_y, n_out * sizeof(float), cudaMemcpyDeviceToHost);
    bool finite_ok = true;
    for (float v : y) if (!(v == v)) finite_ok = false;  // NaN check
    ok(finite_ok, "ix_mmvq_q8_0 output is finite");
    // bad-arg guard
    ok(ix_mmvq::ix_mmvq_q8_0(nullptr, d_x, d_y, n_in, n_out, ncols) == cudaErrorInvalidValue,
       "ix_mmvq_q8_0 rejects null weights");
    ok(ix_mmvq::ix_mmvq_q8_0(d_w, d_x, d_y, 0, n_out, ncols) == cudaErrorInvalidValue,
       "ix_mmvq_q8_0 rejects n_in=0");
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
}
#endif  // __ILUVATAR__

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("gr_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("gr_parity: cannot open GPU 0\n"); return 2; }
    test_gr();
#ifdef __ILUVATAR__
    test_mmvq_wave_smoke();
#endif
    std::printf("gr_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("gr_parity FAILED\n"); return 1; }
    std::printf("gr_parity PASSED\n");
    return 0;
}
