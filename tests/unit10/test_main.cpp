#include <cstdio>
#include "check.hpp"

void test_sf_effort(); void test_sf_budget(); void test_sf_hold(); void test_sf_parser(); void test_sf_parser_random();
void test_ss_guard(); void test_ss_detok(); void test_ss_run(); void test_ss_finish(); void test_ss_cache_auth();
void test_vc_configure(); void test_vc_lists();
void test_ve_command(); void test_ve_header(); void test_ve_grid(); void test_ve_args(); void test_ve_roundtrip_random();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_sf_effort); RUN(test_sf_budget); RUN(test_sf_hold); RUN(test_sf_parser); RUN(test_sf_parser_random);
    RUN(test_ss_guard); RUN(test_ss_detok); RUN(test_ss_run); RUN(test_ss_finish); RUN(test_ss_cache_auth);
    RUN(test_vc_configure); RUN(test_vc_lists);
    RUN(test_ve_command); RUN(test_ve_header); RUN(test_ve_grid); RUN(test_ve_args); RUN(test_ve_roundtrip_random);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT10 TESTS FAILED\n"); return 1; }
    std::printf("UNIT10 TESTS PASSED\n");
    return 0;
}
