#include <cstdio>
#include "check.hpp"

void test_cache_ranges(); void test_cache_admit_fill(); void test_profile(); void test_source_blobs(); void test_dispatch();
void test_gguf_load(); void test_layer_align(); void test_layer_planes(); void test_layer_stages(); void test_layer_flags();
void test_layer_moe_misc(); void test_mtp(); void test_session(); void test_verify(); void test_weights(); void test_prefill();
void test_serve();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_cache_ranges); RUN(test_cache_admit_fill); RUN(test_profile); RUN(test_source_blobs); RUN(test_dispatch);
    RUN(test_gguf_load); RUN(test_layer_align); RUN(test_layer_planes); RUN(test_layer_stages); RUN(test_layer_flags);
    RUN(test_layer_moe_misc); RUN(test_mtp); RUN(test_session); RUN(test_verify); RUN(test_weights); RUN(test_prefill);
    RUN(test_serve);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT18 TESTS FAILED\n"); return 1; }
    std::printf("UNIT18 TESTS PASSED\n");
    return 0;
}
