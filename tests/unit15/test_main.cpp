#include <cstdio>
#include "check.hpp"

void test_blob_valid_iff(); void test_flat_alias_insufficient(); void test_pack_size(); void test_job_compaction();
void test_pcie_share(); void test_blob_split();
void test_registry_theory(); void test_end_ok(); void test_wait_fuel();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_blob_valid_iff); RUN(test_flat_alias_insufficient); RUN(test_pack_size); RUN(test_job_compaction);
    RUN(test_pcie_share); RUN(test_blob_split);
    RUN(test_registry_theory); RUN(test_end_ok); RUN(test_wait_fuel);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT15 TESTS FAILED\n"); return 1; }
    std::printf("UNIT15 TESTS PASSED\n");
    return 0;
}
