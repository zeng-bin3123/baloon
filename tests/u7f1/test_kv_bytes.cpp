// tests/u7f1/test_kv_bytes.cpp — VramMemoryPlan theorems: indexer_default, indexer_errors, kvBytesPerToken_default.
// Theory:
//   indexer_default     : indexerBytes {} = 32
//   indexer_errors      : 1 * 256 / 4 = 2 * 32  (2x too big)
//                         4 * 128 / 4 = 4 * 32  (4x too big)
//   kvBytesPerToken_default : kvBytesPerToken {} = 13056
#include "baloon/plan.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
int main() {
    std::puts("=== test_kv_bytes ===");
    using namespace baloon::plan;
    Geometry g;

    // indexer_default: idx_key_heads * idx_key_dim / qsa_block = 1 * 128 / 4 = 32
    const uint64_t idx = g.idx_key_heads * g.idx_key_dim / g.qsa_block;
    CHK(idx == 32u);

    // indexer_errors: previously used formulas were 2x and 4x too large
    CHK(1u * 256u / 4u == 2u * 32u);  // wrong: 64 instead of 32
    CHK(4u * 128u / 4u == 4u * 32u);  // wrong: 128 instead of 32

    // kv elements: 2 KV heads * head_dim
    const uint64_t kv_elems = 2u * g.n_kv_heads * g.head_dim;  // 2*2*256 = 1024
    CHK(kv_elems == 1024u);

    // kv scales: one fp16 scale per 64-element group → kv_elems / kv_group * 2 B
    const uint64_t kv_scales = kv_elems / g.kv_group * 2u;     // 1024/64*2 = 32
    CHK(kv_scales == 32u);

    // per-layer: kv_elems + kv_scales + idx = 1024 + 32 + 32 = 1088
    const uint64_t per_layer = kv_elems + kv_scales + idx;
    CHK(per_layer == 1088u);

    // total: n_qsa * per_layer = 12 * 1088 = 13056
    const uint64_t total = g.n_qsa * per_layer;
    CHK(total == 13056u);
    CHK(kv_bytes_per_token(g) == 13056u);

    return u7f1::finish("TEST_KV_BYTES");
}

