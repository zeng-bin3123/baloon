#include <cstdio>
#include "check.hpp"

void test_lb_align_carve(); void test_lb_planes_dispatch(); void test_lb_layers_dump();
void test_fg_step_threads(); void test_fg_state_addr(); void test_fg_conv_ab();
void test_gr_slide_window(); void test_gr_grids();
void test_ns_rows_cols(); void test_ns_validate();
void test_np_hist_grid(); void test_np_spans();
void test_fk_geometry(); void test_fk_tiles();
void test_gw_workspace(); void test_gw_write_indexing();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_lb_align_carve); RUN(test_lb_planes_dispatch); RUN(test_lb_layers_dump);
    RUN(test_fg_step_threads); RUN(test_fg_state_addr); RUN(test_fg_conv_ab);
    RUN(test_gr_slide_window); RUN(test_gr_grids);
    RUN(test_ns_rows_cols); RUN(test_ns_validate);
    RUN(test_np_hist_grid); RUN(test_np_spans);
    RUN(test_fk_geometry); RUN(test_fk_tiles);
    RUN(test_gw_workspace); RUN(test_gw_write_indexing);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT6 TESTS FAILED\n"); return 1; }
    std::printf("UNIT6 TESTS PASSED\n");
    return 0;
}
