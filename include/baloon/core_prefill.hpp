// core_prefill.hpp - batched prompt processing (host model): the prompt is cut into chunks of T tokens; per chunk every
// layer runs a hyper-connection read, a GDN / QSA mixer, an MoE half and a write. MoE experts are grouped by id on the
// host (counting sort), staged through an 8-slot host->device ring (lookahead 7) and dequantized through a 2-slot ring.
// Buffers come from a bump allocator: each request is rounded to 256 bytes plus 256 bytes of slack and fails exactly
// when the region would overflow; bytes_needed replays the allocation sequence of init.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace baloon {
namespace core {

constexpr uint64_t kPrefillStageRing = 8;       // STAGE
constexpr uint64_t kPrefillDqRing = 2;          // DQ
constexpr uint64_t kPrefillLookahead = 7;
constexpr uint64_t kPrefillSlackBytes = 8ull << 20;   // alignment slack added by bytes_needed
constexpr uint64_t gdn_floats_per_layer() { return 128ull * 48 * 128 + 10240ull * 3; }   // 817152

constexpr uint64_t round_bytes(uint64_t b) { return (b + 256 + 255) / 256 * 256; }

struct PrefillConfig {
    uint64_t T = 512;                           // chunk size (tokens)
    int n_layers = 48;
    uint64_t n_embd = 0, hc = 1, attn_dim = 0, kv_dim = 0;
    uint32_t n_experts = 512, top_k = 10;
    uint64_t blob_bytes = 0, dq_bytes = 0;      // staged expert blob / dequantized expert
    uint64_t gemm_scratch = 0, gemm_ws = 0;
};

// Counting-sort grouping of (token, k) pairs by expert id.
struct ExpertGrouping {
    std::vector<uint64_t> off;                  // n_experts + 1 block starts: off[e+1] = off[e] + cnt[e]
    std::vector<uint64_t> row;                  // pair i -> its grouped row (strictly inside its expert's block)
    std::vector<uint32_t> order;                // non-empty experts ascending
    std::vector<uint32_t> stage_slot, dq_slot;  // per entry of `order`: j % 8 and j % 2
};
bool group_experts(const std::vector<uint32_t>& experts, uint32_t n_experts, ExpertGrouping& g);

uint64_t MAXBLOB(const std::vector<uint64_t>& blob_sizes);   // staging slot size: the largest blob

struct PrefillHooks {
    std::function<void(int layer, uint64_t c0, uint64_t T)> hc_read, mixer, moe, write;
};
struct PrefillStats {
    uint64_t chunks = 0, tokens = 0, layer_calls = 0, gdn_calls = 0, qsa_calls = 0;
    std::vector<uint64_t> chunk_lens;
};

class Prefill {
public:
    explicit Prefill(const PrefillConfig& c);
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;
    ~Prefill();

    uint64_t counted_bytes() const;                         // counting-mode replay of the allocation sequence (exact total)
    uint64_t bytes_needed() const;                          // counted_bytes() + 8 MiB: the size to reserve for a borrowed region
    // Bump mode over a borrowed region; region == nullptr: own a buffer sized by bytes_needed. false when it does not fit.
    bool init(void* region, uint64_t cap);
    bool initialized() const { return inited_; }
    uint64_t used() const { return used_; }
    bool run(uint64_t n_tokens, const PrefillHooks& h, PrefillStats* st = nullptr);

    const std::vector<uint64_t>& alloc_sequence() const { return seq_; }
    const std::vector<uint64_t>& offsets() const { return offs_; }

private:
    PrefillConfig c_;
    std::vector<uint64_t> seq_, offs_;
    std::vector<uint8_t> owned_;
    bool inited_ = false;
    uint64_t used_ = 0;
};

// Reference projections used by the prefill GEMMs: out[t*rows + r] = sum_j w[r*cols + j] * x[t*cols + j].
void bf16_proj(const uint16_t* w, const float* x, uint64_t T, uint64_t rows, uint64_t cols, float* out);
void native_proj(const float* w, const float* x, uint64_t T, uint64_t rows, uint64_t cols, float* out);

}  // namespace core
}  // namespace baloon
