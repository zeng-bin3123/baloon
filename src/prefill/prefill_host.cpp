// prefill_host.cpp - host model of the batched prefill orchestration (see core_prefill.hpp).
#include "baloon/core_prefill.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace core {
namespace prefill_impl {

using Clock = std::chrono::steady_clock;
[[maybe_unused]] constexpr uint64_t C = 512;                       // default chunk of prompt tokens
[[maybe_unused]] constexpr uint64_t N = 48;                        // layers of the artifact
[[maybe_unused]] constexpr uint64_t DQ = kPrefillDqRing;
[[maybe_unused]] constexpr uint64_t STAGE = kPrefillStageRing;
[[maybe_unused]] constexpr float EPS = 1e-6f;
constexpr uint64_t GEMM_SCRATCH = 32ull << 20;    // default GEMM scratch
constexpr uint64_t GEMM_WS = 16ull << 20;         // default GEMM workspace

inline double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
inline void need(bool cond, const char* what) { if (!cond) throw std::runtime_error(std::string("prefill: ") + what); }

// Allocator with the three modes of the driver; every request is charged round_bytes().
struct Alloc {
    enum Mode { Count, Bump, Owned } mode = Count;
    uint64_t cap = 0, used = 0;
    bool failed = false;
    uint64_t take(uint64_t bytes) {
        const uint64_t r = round_bytes(bytes);
        if (mode == Count) { const uint64_t at = used; used += r; return at; }
        const uint64_t at = used;
        if (used + r > cap) { failed = true; return at; }
        used += r;
        return at;
    }
};

}  // namespace prefill_impl

uint64_t MAXBLOB(const std::vector<uint64_t>& blob_sizes) {
    uint64_t m = 0;
    for (uint64_t b : blob_sizes) m = std::max(m, b);
    return m;
}

bool group_experts(const std::vector<uint32_t>& experts, uint32_t n_experts, ExpertGrouping& g) {
    g = ExpertGrouping{};
    if (n_experts == 0) return false;
    std::vector<uint64_t> cnt(n_experts, 0);
    for (uint32_t e : experts) {
        if (e >= n_experts) return false;
        ++cnt[e];
    }
    g.off.assign((size_t)n_experts + 1, 0);
    for (uint32_t e = 0; e < n_experts; ++e) g.off[e + 1] = g.off[e] + cnt[e];     // off[e+1] = off[e] + cnt[e]
    std::vector<uint64_t> cur(g.off.begin(), g.off.end() - 1);
    g.row.resize(experts.size());
    for (size_t i = 0; i < experts.size(); ++i) g.row[i] = cur[experts[i]]++;       // stays inside the expert's block
    uint32_t j = 0;
    for (uint32_t e = 0; e < n_experts; ++e) {
        if (!cnt[e]) continue;
        g.order.push_back(e);
        g.stage_slot.push_back((uint32_t)(j % kPrefillStageRing));
        g.dq_slot.push_back((uint32_t)(j % kPrefillDqRing));
        ++j;
    }
    return true;
}

Prefill::Prefill(const PrefillConfig& c) : c_(c) {
    const uint64_t T = c.T;
    if (c_.gemm_scratch == 0) c_.gemm_scratch = prefill_impl::GEMM_SCRATCH;
    if (c_.gemm_ws == 0) c_.gemm_ws = prefill_impl::GEMM_WS;
    seq_ = {
        T * c.n_embd * 4,                        // normalised chunk input
        T * c.hc * c.n_embd * 4,                 // residual streams
        T * c.attn_dim * 4,                      // mixer output
        T * 2 * c.kv_dim * 4,                    // K / V rows
        T * (uint64_t)c.n_experts * 4,           // router logits
        T * (uint64_t)c.top_k * 4,               // top-k ids
        T * (uint64_t)c.top_k * 4,               // grouped rows
        kPrefillStageRing * c.blob_bytes,        // host->device staging ring
        kPrefillDqRing * c.dq_bytes,             // dequantized expert ring
        c_.gemm_scratch,
        c_.gemm_ws,
    };
}

Prefill::~Prefill() = default;

uint64_t Prefill::bytes_needed() const {
    prefill_impl::Alloc a;                       // counting mode
    for (uint64_t b : seq_) a.take(b);
    return a.used;
}

bool Prefill::init(void* region, uint64_t cap) {
    inited_ = false; offs_.clear(); used_ = 0; owned_.clear();
    prefill_impl::Alloc a;
    if (region) {
        a.mode = prefill_impl::Alloc::Bump; a.cap = cap;
    } else {
        owned_.assign((size_t)bytes_needed(), 0);       // owned mode sized by the preceding count
        a.mode = prefill_impl::Alloc::Owned; a.cap = owned_.size();
    }
    for (uint64_t b : seq_) offs_.push_back(a.take(b));
    if (a.failed) { offs_.clear(); owned_.clear(); return false; }
    used_ = a.used;
    inited_ = true;
    return true;
}

bool Prefill::run(uint64_t n_tokens, const PrefillHooks& h, PrefillStats* st) {
    if (c_.T == 0 || c_.n_layers <= 0) return false;
    PrefillStats s;
    for (uint64_t c0 = 0; c0 < n_tokens;) {
        const uint64_t T = std::min<uint64_t>(c_.T, n_tokens - c0);
        for (int l = 0; l < c_.n_layers; ++l) {
            if (h.hc_read) h.hc_read(l, c0, T);
            if (h.mixer) h.mixer(l, c0, T);
            if (h.moe) h.moe(l, c0, T);
            if (h.write) h.write(l, c0, T);
            ++s.layer_calls;
            (l % 4 == 3 ? s.qsa_calls : s.gdn_calls) += 1;
        }
        s.chunk_lens.push_back(T);
        ++s.chunks;
        s.tokens += T;
        c0 += T;
    }
    if (st) *st = s;
    return true;
}

void bf16_proj(const uint16_t* w, const float* x, uint64_t T, uint64_t rows, uint64_t cols, float* out) {
    for (uint64_t t = 0; t < T; ++t)
        for (uint64_t r = 0; r < rows; ++r) {
            double a = 0;
            for (uint64_t j = 0; j < cols; ++j) a += (double)f16::bf16_to_float(w[r * cols + j]) * (double)x[t * cols + j];
            out[t * rows + r] = (float)a;
        }
}
void native_proj(const float* w, const float* x, uint64_t T, uint64_t rows, uint64_t cols, float* out) {
    for (uint64_t t = 0; t < T; ++t)
        for (uint64_t r = 0; r < rows; ++r) {
            double a = 0;
            for (uint64_t j = 0; j < cols; ++j) a += (double)w[r * cols + j] * (double)x[t * cols + j];
            out[t * rows + r] = (float)a;
        }
}

}  // namespace core
}  // namespace baloon
