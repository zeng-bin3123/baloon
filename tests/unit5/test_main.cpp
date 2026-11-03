#include <cstdio>
#include "check.hpp"

void test_row_job_geometry(); void test_row_cache(); void test_row_slots(); void test_row_latency_tickets();
void test_hash_constants(); void test_hash_mixing(); void test_hash_context(); void test_hash_iq4();
void test_io_open_validation(); void test_io_protocol(); void test_io_bytes();
void test_block_history(); void test_block_conv_taps(); void test_block_streams_gate(); void test_block_scratch();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_row_job_geometry); RUN(test_row_cache); RUN(test_row_slots); RUN(test_row_latency_tickets);
    RUN(test_hash_constants); RUN(test_hash_mixing); RUN(test_hash_context); RUN(test_hash_iq4);
    RUN(test_io_open_validation); RUN(test_io_protocol); RUN(test_io_bytes);
    RUN(test_block_history); RUN(test_block_conv_taps); RUN(test_block_streams_gate); RUN(test_block_scratch);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT5 TESTS FAILED\n"); return 1; }
    std::printf("UNIT5 TESTS PASSED\n");
    return 0;
}
