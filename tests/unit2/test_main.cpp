#include <cstdio>
#include "check.hpp"

void test_weights_scalar_theorems(); void test_weights_widen_chunked(); void test_weights_bf16_promote(); void test_weights_segments_and_arena();
void test_gguf_geometry(); void test_gguf_shapes_widen_range(); void test_gguf_shards(); void test_gguf_lazy_image();
void test_canon_registry(); void test_canon_integer_facts(); void test_canon_bit_exact_roundtrip(); void test_canon_q2_raw_roundtrip(); void test_canon_q8_and_scale_min();
void test_writer_code_algebra(); void test_writer_packing(); void test_writer_quantizer(); void test_writer_padding(); void test_writer_file_layout();
void test_pack_codes(); void test_pack_alignment(); void test_pack_blob_arithmetic(); void test_pack_expert_blob_roles(); void test_pack_quantised_role_end_to_end(); void test_pack_scale_dtype_and_partition();
void test_mtpq_shapes_and_sizes(); void test_mtpq_scale_search(); void test_mtpq_q4(); void test_mtpq_q8(); void test_mtpq_tensor_and_checkpoint();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_weights_scalar_theorems); RUN(test_weights_widen_chunked); RUN(test_weights_bf16_promote); RUN(test_weights_segments_and_arena);
    RUN(test_gguf_geometry); RUN(test_gguf_shapes_widen_range); RUN(test_gguf_shards); RUN(test_gguf_lazy_image);
    RUN(test_canon_registry); RUN(test_canon_integer_facts); RUN(test_canon_bit_exact_roundtrip); RUN(test_canon_q2_raw_roundtrip); RUN(test_canon_q8_and_scale_min);
    RUN(test_writer_code_algebra); RUN(test_writer_packing); RUN(test_writer_quantizer); RUN(test_writer_padding); RUN(test_writer_file_layout);
    RUN(test_pack_codes); RUN(test_pack_alignment); RUN(test_pack_blob_arithmetic); RUN(test_pack_expert_blob_roles); RUN(test_pack_quantised_role_end_to_end); RUN(test_pack_scale_dtype_and_partition);
    RUN(test_mtpq_shapes_and_sizes); RUN(test_mtpq_scale_search); RUN(test_mtpq_q4); RUN(test_mtpq_q8); RUN(test_mtpq_tensor_and_checkpoint);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT2 TESTS FAILED\n"); return 1; }
    std::printf("UNIT2 TESTS PASSED\n");
    return 0;
}
