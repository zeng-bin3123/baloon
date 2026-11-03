#include <cstdio>
#include "check.hpp"

void test_q2e_geometry(); void test_q2e_codes(); void test_q2e_offset_identity(); void test_q2e_quant();
void test_q2e_row_dot(); void test_q2e_multi(); void test_q2e_dispatch_masks(); void test_q2e_cpu(); void test_q2e_forward();
void test_iq_formats(); void test_iq_layout_tiles(); void test_iq_indices_scales(); void test_iq_saturation();
void test_iq_row_dot(); void test_iq_batching();
void test_avx2_layout(); void test_avx2_unpack(); void test_avx2_exact(); void test_avx2_rows();
void test_geom_sizes_traits(); void test_geom_threads_rows(); void test_geom_bytes();
void test_sg_codes(); void test_sg_groups(); void test_sg_lanes(); void test_sg_rows();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_q2e_geometry); RUN(test_q2e_codes); RUN(test_q2e_offset_identity); RUN(test_q2e_quant);
    RUN(test_q2e_row_dot); RUN(test_q2e_multi); RUN(test_q2e_dispatch_masks); RUN(test_q2e_cpu); RUN(test_q2e_forward);
    RUN(test_iq_formats); RUN(test_iq_layout_tiles); RUN(test_iq_indices_scales); RUN(test_iq_saturation);
    RUN(test_iq_row_dot); RUN(test_iq_batching);
    RUN(test_avx2_layout); RUN(test_avx2_unpack); RUN(test_avx2_exact); RUN(test_avx2_rows);
    RUN(test_geom_sizes_traits); RUN(test_geom_threads_rows); RUN(test_geom_bytes);
    RUN(test_sg_codes); RUN(test_sg_groups); RUN(test_sg_lanes); RUN(test_sg_rows);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT4 TESTS FAILED\n"); return 1; }
    std::printf("UNIT4 TESTS PASSED\n");
    return 0;
}
