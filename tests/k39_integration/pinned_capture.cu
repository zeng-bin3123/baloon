// pinned_capture: runs selected invariant checks of k29_rt/test_rt.cpp on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k29_rt/test_rt.cpp"
#undef main
int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("pinned_capture: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("pinned_capture: cannot open GPU 0\n"); return 2; }
    test_pinned();
    test_graph();
    std::printf("pinned_capture: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("pinned_capture FAILED\n"); return 1; }
    std::printf("pinned_capture PASSED\n");
    return 0;
}
