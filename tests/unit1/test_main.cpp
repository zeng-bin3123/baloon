#include <chrono>
#include <cstdio>
#include "check.hpp"

void test_f16_decode_exhaustive(); void test_f16_roundtrip_all_halves(); void test_f16_rne_against_oracle();
void test_f16_round_normal_contract(); void test_f16_subnormal_contract(); void test_f16_norm_contract(); void test_bf16();
void test_dequant_geometry(); void test_scale_min_k4(); void test_dequant_q2_0(); void test_dequant_q4_0(); void test_dequant_q5_0();
void test_dequant_q6_k(); void test_dequant_q4_k_q5_k(); void test_dequant_q3_k(); void test_dequant_iq4_xs();
void test_dequant_random_bytes_bounded();
void test_kvq8_clamp(); void test_kvq8_quantize_group(); void test_kvq8_threads_and_groups(); void test_kvq8_append_addressing();
void test_kvq8_append_token_end_to_end(); void test_kvq8_gather_decomposition();
void test_session_alignment(); void test_session_gdn_index(); void test_session_gdn_slices(); void test_session_qsa_arena();
void test_session_ring_and_stages();
void test_planner_geometry_constants(); void test_planner_budget_derivations(); void test_planner_make_plan_invariants();
void test_planner_theory_closure(); void test_planner_edges(); void test_planner_monotone_in_context(); void test_planner_costs_form();
void test_bpe_byte_map(); void test_bpe_lossless_bytes(); void test_bpe_merge_step(); void test_bpe_special_and_bookkeeping();
void test_bpe_encode_decode();
void test_sampler_window(); void test_sampler_penalties(); void test_sampler_combine_and_argmax(); void test_sampler_launch_and_rng();

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
    std::fflush(stdout);
}

int main() {
    RUN(test_f16_decode_exhaustive); RUN(test_f16_roundtrip_all_halves); RUN(test_f16_rne_against_oracle);
    RUN(test_f16_round_normal_contract); RUN(test_f16_subnormal_contract); RUN(test_f16_norm_contract); RUN(test_bf16);
    RUN(test_dequant_geometry); RUN(test_scale_min_k4); RUN(test_dequant_q2_0); RUN(test_dequant_q4_0); RUN(test_dequant_q5_0);
    RUN(test_dequant_q6_k); RUN(test_dequant_q4_k_q5_k); RUN(test_dequant_q3_k); RUN(test_dequant_iq4_xs);
    RUN(test_dequant_random_bytes_bounded);
    RUN(test_kvq8_clamp); RUN(test_kvq8_quantize_group); RUN(test_kvq8_threads_and_groups); RUN(test_kvq8_append_addressing);
    RUN(test_kvq8_append_token_end_to_end); RUN(test_kvq8_gather_decomposition);
    RUN(test_session_alignment); RUN(test_session_gdn_index); RUN(test_session_gdn_slices); RUN(test_session_qsa_arena);
    RUN(test_session_ring_and_stages);
    RUN(test_planner_geometry_constants); RUN(test_planner_budget_derivations); RUN(test_planner_make_plan_invariants);
    RUN(test_planner_theory_closure); RUN(test_planner_edges); RUN(test_planner_monotone_in_context); RUN(test_planner_costs_form);
    RUN(test_bpe_byte_map); RUN(test_bpe_lossless_bytes); RUN(test_bpe_merge_step); RUN(test_bpe_special_and_bookkeeping);
    RUN(test_bpe_encode_decode);
    RUN(test_sampler_window); RUN(test_sampler_penalties); RUN(test_sampler_combine_and_argmax); RUN(test_sampler_launch_and_rng);
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("UNIT1 TESTS FAILED\n"); return 1; }
    std::printf("UNIT1 TESTS PASSED\n");
    return 0;
}
