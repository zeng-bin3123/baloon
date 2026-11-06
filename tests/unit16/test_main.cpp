#include <cstdio>
#include "check.hpp"

void test_parse_list(); void test_parse_args(); void test_validate_fit(); void test_argmax();
void test_teacher_forcing(); void test_prefill_split(); void test_spec_equivalence(); void test_spec_rounds();
void test_pool_multi(); void test_serve(); void test_slots_pcie();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_parse_list); RUN(test_parse_args); RUN(test_validate_fit); RUN(test_argmax);
    RUN(test_teacher_forcing); RUN(test_prefill_split); RUN(test_spec_equivalence); RUN(test_spec_rounds);
    RUN(test_pool_multi); RUN(test_serve); RUN(test_slots_pcie);
    std::printf("checks=%ld failures=%ld\n", testh::failures() == 0 ? testh::checks() : testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT16 TESTS FAILED\n"); return 1; }
    std::printf("UNIT16 TESTS PASSED\n");
    return 0;
}
