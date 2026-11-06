#include <cstdio>
#include "check.hpp"

void test_verifier_groups_ring(); void test_verifier_plan_layout(); void test_verifier_staging_flag_accept();
void test_weights_chunks(); void test_weights_segments_place(); void test_arena();
void test_generate_positions(); void test_generate_cache_pcie_spec();
void test_ple_jobs(); void test_direct_file_io(); void test_row_cache_model(); void test_ngram_context();
void test_dequant_geometry(); void test_dequant_simple(); void test_dequant_kquants(); void test_prefill_host_geometry();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_verifier_groups_ring); RUN(test_verifier_plan_layout); RUN(test_verifier_staging_flag_accept);
    RUN(test_weights_chunks); RUN(test_weights_segments_place); RUN(test_arena);
    RUN(test_generate_positions); RUN(test_generate_cache_pcie_spec);
    RUN(test_ple_jobs); RUN(test_direct_file_io); RUN(test_row_cache_model); RUN(test_ngram_context);
    RUN(test_dequant_geometry); RUN(test_dequant_simple); RUN(test_dequant_kquants); RUN(test_prefill_host_geometry);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT12 TESTS FAILED\n"); return 1; }
    std::printf("UNIT12 TESTS PASSED\n");
    return 0;
}
