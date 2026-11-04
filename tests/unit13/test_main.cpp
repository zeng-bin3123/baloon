#include <cstdio>
#include "check.hpp"
void test_pm_clamp_and_buckets(); void test_pm_expected(); void test_pm_best_gate(); void test_pm_observe_and_clip(); void test_pm_controller();
void test_sd_ways_and_keys(); void test_sd_probe_mask(); void test_sd_match_loop(); void test_sd_clamps(); void test_sd_continuation(); void test_sd_end_to_end();
void test_port_expected(); void test_port_choose_argmax(); void test_port_mix(); void test_port_find_slot_and_reset();
void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}
int main() {
    RUN(test_pm_clamp_and_buckets); RUN(test_pm_expected); RUN(test_pm_best_gate); RUN(test_pm_observe_and_clip); RUN(test_pm_controller);
    RUN(test_sd_ways_and_keys); RUN(test_sd_probe_mask); RUN(test_sd_match_loop); RUN(test_sd_clamps); RUN(test_sd_continuation); RUN(test_sd_end_to_end);
    RUN(test_port_expected); RUN(test_port_choose_argmax); RUN(test_port_mix); RUN(test_port_find_slot_and_reset);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT13 TESTS FAILED\n"); return 1; }
    std::printf("UNIT13 TESTS PASSED\n");
    return 0;
}
