#include <cstdio>
#include "check.hpp"

void test_cache_align_and_offsets(); void test_cache_layer_ranges(); void test_cache_admission_and_addressing();
void test_source_blob_index(); void test_source_pinned_and_dispatch(); void test_dispatch_contract();
void test_iq_calls_layout(); void test_s2_blob();
void test_router_order_and_topk(); void test_router_clamp_and_launch();
void test_pool_split_and_segments(); void test_pool_routing_batches_cores();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_cache_align_and_offsets); RUN(test_cache_layer_ranges); RUN(test_cache_admission_and_addressing);
    RUN(test_source_blob_index); RUN(test_source_pinned_and_dispatch); RUN(test_dispatch_contract);
    RUN(test_iq_calls_layout); RUN(test_s2_blob);
    RUN(test_router_order_and_topk); RUN(test_router_clamp_and_launch);
    RUN(test_pool_split_and_segments); RUN(test_pool_routing_batches_cores);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT3 TESTS FAILED\n"); return 1; }
    std::printf("UNIT3 TESTS PASSED\n");
    return 0;
}
