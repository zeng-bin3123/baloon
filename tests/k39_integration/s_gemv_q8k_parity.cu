// s_gemv_q8k_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main
int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("s_gemv_q8k_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("s_gemv_q8k_parity: cannot open GPU 0\n"); return 2; }
    test_formats();
    std::printf("s_gemv_q8k_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("s_gemv_q8k_parity FAILED\n"); return 1; }
    std::printf("s_gemv_q8k_parity PASSED\n");
    return 0;
}
