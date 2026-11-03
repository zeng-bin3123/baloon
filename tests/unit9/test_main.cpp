#include <cstdio>
#include "check.hpp"

void test_mtp_bump(); void test_mtp_window_rows(); void test_mtp_dvocab();
void test_vp_groups_rings(); void test_vp_staging_plan(); void test_vp_flag_commit();
void test_vi_conv_commit(); void test_vi_window_steps(); void test_vi_blob_bcast();
void test_sc_indexing(); void test_sc_expected(); void test_sc_choose_learn();
void test_sd_basic(); void test_sd_random(); void test_sd_table();
void test_sc_theory_tables_and_indexing(); void test_sc_theory_observe_and_commit(); void test_sc_theory_gain_threshold();
void test_sd_theory_table(); void test_sd_theory_match_and_propose(); void test_sd_theory_ways();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_mtp_bump); RUN(test_mtp_window_rows); RUN(test_mtp_dvocab);
    RUN(test_vp_groups_rings); RUN(test_vp_staging_plan); RUN(test_vp_flag_commit);
    RUN(test_vi_conv_commit); RUN(test_vi_window_steps); RUN(test_vi_blob_bcast);
    RUN(test_sc_indexing); RUN(test_sc_expected); RUN(test_sc_choose_learn);
    RUN(test_sd_basic); RUN(test_sd_random); RUN(test_sd_table);
    RUN(test_sc_theory_tables_and_indexing); RUN(test_sc_theory_observe_and_commit); RUN(test_sc_theory_gain_threshold);
    RUN(test_sd_theory_table); RUN(test_sd_theory_match_and_propose); RUN(test_sd_theory_ways);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT9 TESTS FAILED\n"); return 1; }
    std::printf("UNIT9 TESTS PASSED\n");
    return 0;
}
