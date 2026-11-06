// controller_test: speculative-decoding controller invariants (unit 9 + unit 13 test bodies).
#include "run_harness.hpp"
void test_sc_indexing(); void test_sc_expected(); void test_sc_choose_learn();
void test_sc_theory_tables_and_indexing(); void test_sc_theory_observe_and_commit(); void test_sc_theory_gain_threshold();
void test_pm_clamp_and_buckets(); void test_pm_expected(); void test_pm_best_gate(); void test_pm_observe_and_clip(); void test_pm_controller();
void test_port_expected(); void test_port_choose_argmax();
int main() {
    RUN(test_sc_indexing); RUN(test_sc_expected); RUN(test_sc_choose_learn);
    RUN(test_sc_theory_tables_and_indexing); RUN(test_sc_theory_observe_and_commit); RUN(test_sc_theory_gain_threshold);
    RUN(test_pm_clamp_and_buckets); RUN(test_pm_expected); RUN(test_pm_best_gate); RUN(test_pm_observe_and_clip); RUN(test_pm_controller);
    RUN(test_port_expected); RUN(test_port_choose_argmax);
    return finish("controller_test");
}
