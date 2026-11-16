#include <cstdio>
#include "check.hpp"

// scope items (ub25e8eba7f)
void test_run_fits();
void test_dump_positions();
void test_next_input();
void test_n_batched();
void test_accept();
void test_window();
void test_pcie_trace();
// additional theory items
void test_trace_valid();
void test_auto_slots();
void test_with_profile();
void test_ceil_div();
void test_prefill_mib();
void test_swap_ok();
void test_decay();
void test_ple_advance();
void test_img_pos();
void test_admitted();
void test_produced_step();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    // scope items
    RUN(test_run_fits);
    RUN(test_dump_positions);
    RUN(test_next_input);
    RUN(test_n_batched);
    RUN(test_accept);
    RUN(test_window);
    RUN(test_pcie_trace);
    // additional theory items
    RUN(test_trace_valid);
    RUN(test_auto_slots);
    RUN(test_with_profile);
    RUN(test_ceil_div);
    RUN(test_prefill_mib);
    RUN(test_swap_ok);
    RUN(test_decay);
    RUN(test_ple_advance);
    RUN(test_img_pos);
    RUN(test_admitted);
    RUN(test_produced_step);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT_GENDRV TESTS FAILED\n"); return 1; }
    std::printf("UNIT_GENDRV TESTS PASSED\n");
    return 0;
}
