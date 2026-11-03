#include <cstdio>
#include "check.hpp"

void test_residency_ranges(); void test_residency_admit(); void test_residency_table(); void test_residency_sized_fill_profile();
void test_source_geometry(); void test_source_split(); void test_source_pcie_latch();
void test_block_split(); void test_block_stages_align(); void test_block_qgate();
void test_loop_layers(); void test_loop_capture_dump();
void test_verify_commit(); void test_verify_accept();
void test_graph_registry();
void test_compose_align_planes(); void test_compose_dispatch_stage(); void test_compose_carve();
void test_mtpd_bump_window(); void test_mtpd_rows_chunks();
void test_driver_carve(); void test_driver_indexing();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_residency_ranges); RUN(test_residency_admit); RUN(test_residency_table); RUN(test_residency_sized_fill_profile);
    RUN(test_source_geometry); RUN(test_source_split); RUN(test_source_pcie_latch);
    RUN(test_block_split); RUN(test_block_stages_align); RUN(test_block_qgate);
    RUN(test_loop_layers); RUN(test_loop_capture_dump);
    RUN(test_verify_commit); RUN(test_verify_accept);
    RUN(test_graph_registry);
    RUN(test_compose_align_planes); RUN(test_compose_dispatch_stage); RUN(test_compose_carve);
    RUN(test_mtpd_bump_window); RUN(test_mtpd_rows_chunks);
    RUN(test_driver_carve); RUN(test_driver_indexing);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT11 TESTS FAILED\n"); return 1; }
    std::printf("UNIT11 TESTS PASSED\n");
    return 0;
}
