#include <cstdio>
#include "check.hpp"

void test_pk_constants(); void test_pk_geometry(); void test_pk_index_maps(); void test_pk_blob(); void test_pk_numeric();
void test_pd_alloc(); void test_pd_chunks(); void test_pd_grouping(); void test_pd_rings_sizes();
void test_ga_positions(); void test_ga_prefill_cache(); void test_ga_pcie_trace_serve();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_pk_constants); RUN(test_pk_geometry); RUN(test_pk_index_maps); RUN(test_pk_blob); RUN(test_pk_numeric);
    RUN(test_pd_alloc); RUN(test_pd_chunks); RUN(test_pd_grouping); RUN(test_pd_rings_sizes);
    RUN(test_ga_positions); RUN(test_ga_prefill_cache); RUN(test_ga_pcie_trace_serve);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT8 TESTS FAILED\n"); return 1; }
    std::printf("UNIT8 TESTS PASSED\n");
    return 0;
}
