#include <cstdio>
#include "check.hpp"

void test_vep_split_last(); void test_vep_split_random(); void test_vep_valid_enc(); void test_vep_grid_pos();
void test_vep_grid_derive(); void test_vep_file_size(); void test_vep_magic_header(); void test_vep_file_roundtrip();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_vep_split_last); RUN(test_vep_split_random); RUN(test_vep_valid_enc); RUN(test_vep_grid_pos);
    RUN(test_vep_grid_derive); RUN(test_vep_file_size); RUN(test_vep_magic_header); RUN(test_vep_file_roundtrip);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT14 TESTS FAILED\n"); return 1; }
    std::printf("UNIT14 TESTS PASSED\n");
    return 0;
}
