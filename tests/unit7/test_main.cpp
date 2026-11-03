#include <cstdio>
#include "check.hpp"

void test_qc_step_width(); void test_qc_paging(); void test_qc_order_key(); void test_qc_topk(); void test_qc_misc();
void test_qd_chunks(); void test_qd_partitions(); void test_qd_scratch(); void test_qd_merge();
void test_qb_weights(); void test_qb_digits(); void test_qb_radix(); void test_qb_grids();
void test_qs_columns(); void test_qs_accumulator(); void test_qs_guards();
void test_qa_slots(); void test_qa_rope(); void test_qa_validation(); void test_qa_simulation();
void test_fb_heads_loop(); void test_fb_memory(); void test_fb_valid();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_qc_step_width); RUN(test_qc_paging); RUN(test_qc_order_key); RUN(test_qc_topk); RUN(test_qc_misc);
    RUN(test_qd_chunks); RUN(test_qd_partitions); RUN(test_qd_scratch); RUN(test_qd_merge);
    RUN(test_qb_weights); RUN(test_qb_digits); RUN(test_qb_radix); RUN(test_qb_grids);
    RUN(test_qs_columns); RUN(test_qs_accumulator); RUN(test_qs_guards);
    RUN(test_qa_slots); RUN(test_qa_rope); RUN(test_qa_validation); RUN(test_qa_simulation);
    RUN(test_fb_heads_loop); RUN(test_fb_memory); RUN(test_fb_valid);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT7 TESTS FAILED\n"); return 1; }
    std::printf("UNIT7 TESTS PASSED\n");
    return 0;
}
