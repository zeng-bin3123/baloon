// mtp.cpp - host model of the MTP drafter (see core_mtp.hpp).
#include "baloon/core_mtp.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "baloon/mtp_drafter.hpp"

namespace baloon {
namespace core {
namespace mtp_impl {

using Clock = std::chrono::steady_clock;
[[maybe_unused]] constexpr double EPS = 1e-6;
[[maybe_unused]] constexpr int GGML_Q8_0 = 8;       // ggml type id of Q8_0 (the "q8" kind)

// Bump allocator over the draft layer's persistent scratch: 256-byte aligned regions.
struct Bump {
    uint64_t used = 0, cap = UINT64_MAX;
    bool failed = false;
    uint64_t take(uint64_t bytes) {
        const uint64_t at = used, next = (used + bytes + 255) / 256 * 256;
        if (next > cap) { failed = true; return at; }
        used = next;
        return at;
    }
};

inline double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// A tensor table row is addressable when it lies inside a pack of `pack_bytes`.
const MtpTensor* mapped(const std::vector<MtpTensor>& t, const std::string& name, uint64_t pack_bytes) {
    for (auto& x : t) if (x.name == name && x.off + x.bytes <= pack_bytes) return &x;
    return nullptr;
}

struct Shape { uint64_t rows = 0, cols = 0; };
Shape shapes_of(const MtpTensor& t, uint64_t elem_bytes, uint64_t cols) {
    Shape s;
    if (elem_bytes == 0 || cols == 0 || t.bytes % (elem_bytes * cols) != 0) return s;
    s.cols = cols; s.rows = t.bytes / (elem_bytes * cols);
    return s;
}

bool finish_capture(std::vector<uint8_t>& staged, uint64_t rows) { return staged.size() == rows; }

}  // namespace mtp_impl

MtpDrafter::~MtpDrafter() = default;

bool MtpDrafter::load_tensors(std::vector<MtpTensor> t) {
    for (auto& x : t)
        if (x.name.empty() || (x.kind != "f32" && x.kind != "bf16" && x.kind != "q8")) return false;
    tensors_ = std::move(t);
    return true;
}

bool MtpDrafter::load_text(const std::string& text) {
    std::vector<MtpTensor> t;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        MtpTensor x;
        unsigned long long off = 0, bytes = 0;
        if (!(ls >> x.name >> x.kind >> off >> bytes)) return false;
        x.off = off; x.bytes = bytes;
        t.push_back(std::move(x));
    }
    return load_tensors(std::move(t));
}

bool MtpDrafter::load(const std::string& index_path) {
    std::string text;
    if (!mtp_impl::read_file(index_path, text)) return false;
    return load_text(text);
}

bool MtpDrafter::bind(const Config& cfg) {
    bound_ = false;
    if (cfg.max_cells == 0 || cfg.max_t < 2 || cfg.max_t > 64) return false;
    cfg_ = cfg;
    window_ = (cfg.window > 0 && cfg.window < cfg.max_cells) ? cfg.window : 0;
    const uint64_t len = window_ > 0 ? window_ : cfg.max_cells;
    cap_ = (len + 63) / 64 * 64;
    staged_.assign(2 * cfg.max_t, 0);
    rows_.clear(); chunks_run_.clear();
    prefill_graphs_ = 0; round_T_ = 0;
    bound_ = true;
    return true;
}

const MtpTensor* MtpDrafter::find(const std::string& name, const char* kind) const {
    for (auto& t : tensors_) if (t.name == name && t.kind == kind) return &t;     // name AND kind
    return nullptr;
}
const MtpTensor* MtpDrafter::f32(const std::string& name) const { return find(name, "f32"); }
const MtpTensor* MtpDrafter::bf16(const std::string& name) const { return find(name, "bf16"); }
const MtpTensor* MtpDrafter::q8(const std::string& name) const { return find(name, "q8"); }

NativeHead MtpDrafter::native_head(const std::string& name, uint64_t n_embd) const {
    NativeHead h;
    const MtpTensor* t = bf16(name);
    if (!t || n_embd == 0 || t->bytes % (n_embd * 2) != 0) return h;
    h.w = t; h.n_embd = n_embd; h.vocab = t->bytes / (n_embd * 2);
    return h;
}

uint64_t MtpDrafter::prefill(uint64_t n, int64_t cell0, int64_t first_needed,
                             const std::function<void(uint64_t, uint64_t)>& chunk_fn) {
    chunks_run_.clear();
    if (!bound_) return 0;
    uint64_t ran = 0;
    for (uint64_t c = 0; c < n;) {
        const uint64_t T = std::min<uint64_t>(cfg_.max_t, n - c);
        const bool below = cell0 + (int64_t)c + (int64_t)T <= first_needed;     // chunk wholly below the first needed cell
        if (!below) {
            if (chunk_fn) chunk_fn(c, T);
            chunks_run_.push_back(c);
            ++ran;
        }
        c += T;
    }
    return ran;
}

uint64_t MtpDrafter::capture_prefill() {
    if (!bound_) return 0;
    prefill_graphs_ = cfg_.max_t;       // one K/V-only graph per chunk length 1..max_t
    return prefill_graphs_;
}

bool MtpDrafter::capture_round(uint64_t T) {
    if (!bound_ || T < 1 || T > cfg_.max_t) return false;
    std::fill(staged_.begin(), staged_.end(), 0);
    for (uint64_t r = 0; r < T; ++r) staged_[r] = 1;                              // T catch-up rows
    for (uint64_t j = 1; j + 2 <= cfg_.max_t; ++j) staged_[mtp_drafter::chain_row(cfg_.max_t, j)] = 2;   // j = 1..max_t-2
    staged_[mtp_drafter::full_layer_row(cfg_.max_t)] = 3;                         // full-layer row 2*max_t - 1
    round_T_ = T;
    return mtp_impl::finish_capture(staged_, 2 * cfg_.max_t);
}

bool MtpDrafter::capture_step(uint64_t j) {
    if (!bound_ || j < 1 || j + 2 > cfg_.max_t) return false;
    return record_forward(mtp_drafter::chain_row(cfg_.max_t, j));
}

bool MtpDrafter::record_forward(uint64_t row) {
    if (!bound_ || row >= 2 * cfg_.max_t) return false;
    rows_.push_back(row);
    return true;
}

bool MtpDrafter::draft_first(uint64_t T, uint64_t a, const DraftFn& fn, DraftStep& out) {
    if (!bound_ || !mtp_drafter::draft_arg_row_ok(T, cfg_.max_t, a) || !fn) return false;
    out = fn(0, 0);
    return record_forward(a);          // the caught-up row a is a real window row
}

int64_t MtpDrafter::draft(uint64_t T, uint64_t a, const DraftFn& fn, std::vector<int32_t>& out) {
    out.clear();
    if (!bound_ || !fn || !mtp_drafter::draft_arg_row_ok(T, cfg_.max_t, a)) return -1;
    out.assign(cfg_.max_t - 1, 0);                                  // unused slots stay zero-filled
    DraftStep d;
    if (!draft_first(T, a, fn, d)) return -1;
    out[0] = d.token;
    uint64_t j = 1;
    double last_p = d.prob;
    while (j < cfg_.max_t - 1 && last_p >= cfg_.min_p) {
        if (!record_forward(mtp_drafter::chain_row(cfg_.max_t, j))) break;
        const DraftStep s = fn(j, out[j - 1]);
        out[j] = s.token;
        last_p = s.prob;
        ++j;
    }
    return (int64_t)j;
}

}  // namespace core
}  // namespace baloon
