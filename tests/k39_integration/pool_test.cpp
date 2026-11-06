// pool_test: runs the invariant test body of k14cpu_kernels/test_cpu_kernels.cpp (its main is renamed; argv is accepted for --selftest style invocation).
#define main orig_test_main
#include "../k14cpu_kernels/test_cpu_kernels.cpp"
#undef main
int main(int argc, char** argv) {
    (void)argc; (void)argv;           // --selftest (and fixture flags) are accepted; the checks always run
    const int rc = orig_test_main();
    std::printf("pool_test: %s\n", rc == 0 ? "PASSED" : "FAILED");
    return rc;
}
