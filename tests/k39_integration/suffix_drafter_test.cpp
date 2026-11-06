// suffix_drafter_test: suffix-drafter invariants (unit 9 + unit 13 test bodies).
#include "run_harness.hpp"
void test_sd_basic(); void test_sd_random(); void test_sd_table();
void test_sd_theory_table(); void test_sd_theory_match_and_propose(); void test_sd_theory_ways();
void test_sd_ways_and_keys(); void test_sd_probe_mask(); void test_sd_match_loop(); void test_sd_clamps(); void test_sd_continuation(); void test_sd_end_to_end();
void test_port_mix(); void test_port_find_slot_and_reset();
int main() {
    RUN(test_sd_basic); RUN(test_sd_random); RUN(test_sd_table);
    RUN(test_sd_theory_table); RUN(test_sd_theory_match_and_propose); RUN(test_sd_theory_ways);
    RUN(test_sd_ways_and_keys); RUN(test_sd_probe_mask); RUN(test_sd_match_loop); RUN(test_sd_clamps); RUN(test_sd_continuation); RUN(test_sd_end_to_end);
    RUN(test_port_mix); RUN(test_port_find_slot_and_reset);
    return finish("suffix_drafter_test");
}
