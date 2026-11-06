// core_mtp.hpp - the multi-token-prediction draft layer (host model): window/capacity sizing, step-row staging, the
// probability-gated draft chain, chunked prefill and tensor lookup by name AND kind.
// window_ = window only when 0 < window < max_cells else 0; cap = ceil(len/64)*64 with len = window_ or max_cells;
// step/pos staging has 2*max_t rows (T catch-up rows, chain step j in row max_t + j - 1, full-layer row 2*max_t - 1);
// the chain continues while j < max_t - 1 and the last draft's probability >= min_p, remaining slots are zero-filled.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace baloon {
namespace core {

struct MtpTensor {
    std::string name, kind;       // kind: "f32", "bf16" or "q8"
    uint64_t off = 0, bytes = 0;
};

struct DraftStep { int32_t token = 0; double prob = 0; };
using DraftFn = std::function<DraftStep(uint64_t j, int32_t prev_token)>;   // j = chain step (0 = the catch-up draft)

// The draft head's native (bf16) output projection: rows = vocab, cols = n_embd; present only when the tensor is bf16
// and its byte size is a whole number of rows.
struct NativeHead {
    const MtpTensor* w = nullptr;
    uint64_t vocab = 0, n_embd = 0;
    bool present() const { return w != nullptr && vocab > 0 && n_embd > 0 && w->bytes == vocab * n_embd * 2; }
};

class MtpDrafter {
public:
    struct Config { uint64_t max_cells = 0, window = 0, max_t = 0; double min_p = 0; };

    MtpDrafter() = default;
    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;
    ~MtpDrafter();

    // Tensor table: one "name kind offset bytes" line per tensor.
    bool load(const std::string& index_path);
    bool load_text(const std::string& text);
    bool load_tensors(std::vector<MtpTensor> t);
    bool bind(const Config& cfg);
    bool bound() const { return bound_; }

    const MtpTensor* f32(const std::string& name) const;
    const MtpTensor* bf16(const std::string& name) const;
    const MtpTensor* q8(const std::string& name) const;

    NativeHead native_head(const std::string& name, uint64_t n_embd) const;   // bf16 lookup + shape check
    uint64_t window() const { return window_; }
    uint64_t cap() const { return cap_; }
    uint64_t round_T() const { return round_T_; }
    uint64_t step_rows() const { return 2 * cfg_.max_t; }
    const Config& config() const { return cfg_; }

    // Chunked prefill of n prompt cells starting at cell0: chunks of max_t, chunks entirely below first_needed skipped.
    // chunk_fn(c, T) runs one chunk; returns how many chunks ran.
    uint64_t prefill(uint64_t n, int64_t cell0, int64_t first_needed, const std::function<void(uint64_t c, uint64_t T)>& chunk_fn);
    uint64_t capture_prefill();                         // K/V-only graph table: entries 1..max_t
    bool capture_round(uint64_t T);                     // stage the 2*max_t step rows for a window of T
    bool capture_step(uint64_t j);                      // chain row of step j (1 <= j <= max_t - 2)
    bool record_forward(uint64_t row);                  // one recorded forward pass on a staged row
    bool draft_first(uint64_t T, uint64_t a, const DraftFn& fn, DraftStep& out);
    // Returns the number of drafts produced (>= 1), or -1 on bad arguments; out has max_t - 1 slots, tail zero-filled.
    int64_t draft(uint64_t T, uint64_t a, const DraftFn& fn, std::vector<int32_t>& out);

    const std::vector<uint64_t>& prefill_chunks() const { return chunks_run_; }
    const std::vector<uint64_t>& recorded_rows() const { return rows_; }
    uint64_t prefill_graphs() const { return prefill_graphs_; }
    const std::vector<uint8_t>& staged_kind() const { return staged_; }   // per row: 0 empty, 1 catch-up, 2 chain, 3 full

private:
    const MtpTensor* find(const std::string& name, const char* kind) const;
    Config cfg_;
    bool bound_ = false;
    uint64_t window_ = 0, cap_ = 0, prefill_graphs_ = 0;
    std::vector<MtpTensor> tensors_;
    std::vector<uint64_t> chunks_run_, rows_;
    std::vector<uint8_t> staged_;
    uint64_t round_T_ = 0;
};

}  // namespace core
}  // namespace baloon
