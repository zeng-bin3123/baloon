// expert_parity: runs the invariant test body of k20_native/test_native.cpp (its main is renamed; argv is accepted for --selftest style invocation).
#define main orig_test_main
#include "../k20_native/test_native.cpp"
#undef main
int main(int argc, char** argv) {
    (void)argc; (void)argv;           // --selftest (and fixture flags) are accepted; the checks always run
    const int rc = orig_test_main();
    std::printf("expert_parity: %s\n", rc == 0 ? "PASSED" : "FAILED");
    return rc;
}
