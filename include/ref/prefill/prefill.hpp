// ref/prefill/prefill.hpp — reference specification: batched prompt processing types.
// The prompt is cut into chunks of T tokens; per chunk every layer runs:
//   hc_read → mixer (GDN or QSA) → MoE half → write.
// Experts are grouped by id (counting sort), staged through an 8-slot ring (lookahead 7),
// and dequantized through a 2-slot ring. Buffers come from a bump allocator.
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

namespace baloon {
namespace core {

struct PrefillConfig {
    uint64_t T = 512;
    int n_layers = 48;
    uint64_t n_embd = 0, hc = 1, attn_dim = 0, kv_dim = 0;
    uint32_t n_experts = 512, top_k = 10;
    uint64_t blob_bytes = 0, dq_bytes = 0;
    uint64_t gemm_scratch = 0, gemm_ws = 0;
};

struct PrefillHooks {
    std::function<void(int layer, uint64_t c0, uint64_t T)> hc_read, mixer, moe, write;
};

// PrefillStats: counters updated by Prefill::run() on each call.
struct PrefillStats {
    uint64_t chunks = 0;        // total number of T-token chunks processed
    uint64_t tokens = 0;        // total tokens across all chunks
    uint64_t layer_calls = 0;   // n_layers * chunks (one call per layer per chunk)
    uint64_t gdn_calls = 0;     // GDN mixer invocations
    uint64_t qsa_calls = 0;     // QSA mixer invocations
    std::vector<uint64_t> chunk_lens;  // length of each chunk (tokens), in call order
};

// Prefill: batched prompt processor.  Owns a bump-allocated device buffer region.
// bytes_needed() gives the minimum region size; init(region, n_bytes) installs an external region.
// run() processes n_tokens, calling the hooks in layer order, and returns false only when the
// region is too small (detected by replaying the allocation sequence without touching the GPU).
class Prefill {
public:
    explicit Prefill(const PrefillConfig& c);
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;
    ~Prefill();

    uint64_t counted_bytes() const;   // exact total from the allocation replay
    uint64_t bytes_needed() const;    // counted_bytes() + 8 MiB alignment slack
    bool init(void* region, uint64_t n_bytes);  // borrow an external region (nullptr = own)
    bool run(uint64_t n_tokens, const PrefillHooks& h, PrefillStats* st = nullptr);
};

}  // namespace core
}  // namespace baloon
