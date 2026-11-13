// tests/u7f1/test_plan_segments.cpp - DenseWeightLoader: out_bytes, weight_align256, conv_for.
// All functions here are constexpr in core_weights.hpp (no .cpp link needed).
#include "baloon/core_weights.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
int main() {
    std::puts("=== test_plan_segments ===");
    using baloon::core::WConv;
    using baloon::core::out_bytes;
    using baloon::core::kWeightChunk;
    using baloon::core::kStageOutBytes;
    using baloon::core::weight_align256;
    using baloon::core::map_kind;
    using baloon::core::conv_for;
    // kWeightChunk = 8 MiB, chunk_even
    CHK(kWeightChunk == 8u * 1024u * 1024u);
    CHK(kWeightChunk % 2 == 0);
    CHK(kStageOutBytes == 2u * kWeightChunk);
    // Copy is identity; promoted_sizes; widen_exact_size
    for (uint64_t e : {1u, 16u, 1024u}) {
        CHK(out_bytes(WConv::Copy,     e)       == e);
        CHK(out_bytes(WConv::Bf16High, 4u * e)  == 2u * e);
        CHK(out_bytes(WConv::ToF16,    4u * e)  == 2u * e);
        CHK(out_bytes(WConv::WidenF16, 2u * e)  == 4u * e);
    }
    // outBytes_le_double: no conversion grows past 2n
    for (uint64_t n : {0u, 4u, 8u, 1024u}) {
        CHK(out_bytes(WConv::Copy,     n) <= 2u * n);
        CHK(out_bytes(WConv::Bf16High, n) <= 2u * n);
        CHK(out_bytes(WConv::ToF16,    n) <= 2u * n);
        CHK(out_bytes(WConv::WidenF16, n) <= 2u * n);
    }
    // stage_out_suffices: CHUNK fits in kStageOutBytes
    CHK(out_bytes(WConv::Copy,     kWeightChunk) <= kStageOutBytes);
    CHK(out_bytes(WConv::Bf16High, kWeightChunk) <= kStageOutBytes);
    CHK(out_bytes(WConv::ToF16,    kWeightChunk) <= kStageOutBytes);
    CHK(out_bytes(WConv::WidenF16, kWeightChunk) <= kStageOutBytes);
    // map_kind: remap_raw
    CHK(map_kind(4) == 1); CHK(map_kind(5) == 3);
    CHK(map_kind(0) == 0); CHK(map_kind(2) == 2);
    // conv_for
    CHK(conv_for(true,  0) == WConv::Copy);
    CHK(conv_for(false, 1) == WConv::Bf16High);
    CHK(conv_for(false, 3) == WConv::ToF16);
    // weight_align256: align256_ge, align256_mod
    for (uint64_t x : {0u, 1u, 255u, 256u, 257u, 1382400u}) {
        CHK(weight_align256(x) >= x);
        CHK(weight_align256(x) % 256 == 0);
    }
    return u7f1::finish("TEST_PLAN_SEGMENTS");
}
