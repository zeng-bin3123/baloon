// ple_reader_test: runs the invariant test body of k21_ple_reader/test_ple_reader_theory.cpp (its main is renamed; argv is accepted for --selftest style invocation).
#define main orig_test_main
#include "../k21_ple_reader/test_ple_reader_theory.cpp"
#undef main
int main(int argc, char** argv) {
    (void)argc; (void)argv;           // --selftest (and fixture flags) are accepted; the checks always run
    const int rc = orig_test_main();
    std::printf("ple_reader_test: %s\n", rc == 0 ? "PASSED" : "FAILED");
    return rc;
}
