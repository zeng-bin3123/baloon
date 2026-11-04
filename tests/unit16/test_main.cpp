#include <cstdio>
#include "check.hpp"

void test_reader_page_math(); void test_reader_cache_and_tickets(); void test_reader_files(); void test_reader_concurrent();
void test_prefill_index_theory(); void test_prefill_blob_theory(); void test_prefill_api_constants();
void test_gen_theory(); void test_gen_loops(); void test_gen_options_pool();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_reader_page_math); RUN(test_reader_cache_and_tickets); RUN(test_reader_files); RUN(test_reader_concurrent);
    RUN(test_prefill_index_theory); RUN(test_prefill_blob_theory); RUN(test_prefill_api_constants);
    RUN(test_gen_theory); RUN(test_gen_loops); RUN(test_gen_options_pool);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT16 TESTS FAILED\n"); return 1; }
    std::printf("UNIT16 TESTS PASSED\n");
    return 0;
}
