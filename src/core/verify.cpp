// verify.cpp - host model of the window verifier (see core_verify.hpp).
#include "baloon/core_verify.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "baloon/expert_dispatch_contract.hpp"
#include "baloon/verify_window_protocol.hpp"
#include "baloon/window_verifier.hpp"

namespace baloon {
namespace core {
namespace verify_impl {

using Clock = std::chrono::steady_clock;
[[maybe_unused]] constexpr double EPS = 1e-6;
static const bool g_dbg = std::getenv("BALOON_VDBG") != nullptr;
#define VDBG(...) do { if (verify_impl::g_dbg) std::fprintf(stderr, __VA_ARGS__); } while (0)

// Bump allocator for the verifier's device scratch (256-byte aligned regions).
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

// Precondition check used by init: throws with the failed requirement.
inline void need(bool cond, const char* what) { if (!cond) throw std::runtime_error(std::string("verify: ") + what); }

// A (offset, bytes) span lies inside a mapped pack.
inline bool mapped(uint64_t off, uint64_t bytes, uint64_t pack_bytes) { return off + bytes <= pack_bytes; }

// Native-format tensor kinds: raw 16-bit (BF16 = 4, F16 = 5) are consumed natively.
inline bool native_of(int kind) { return kind == 4 || kind == 5; }

struct Shape { uint64_t rows = 0, cols = 0; };
inline Shape shapes_of(uint64_t bytes, uint64_t elem_bytes, uint64_t cols) {
    Shape s;
    if (elem_bytes == 0 || cols == 0 || bytes % (elem_bytes * cols) != 0) return s;
    s.cols = cols; s.rows = bytes / (elem_bytes * cols);
    return s;
}

}  // namespace verify_impl

Verifier::~Verifier() = default;

bool Verifier::init(const Config& c) {
    inited_ = false;
    try {
        verify_impl::need(c.max_t >= 1 && c.max_t <= 8, "max_t in 1..8");
        verify_impl::need(c.groups == 1 || c.groups == 2, "groups is 1 or 2");
        verify_impl::need(c.n_layers >= 1 && c.plan_cap >= 1, "layers and plan capacity are positive");
    } catch (const std::exception& e) {
        VDBG("%s\n", e.what());
        return false;
    }
    c_ = c;
    const uint64_t words = winv::plan_layout(c.plan_cap).total;
    plan_[0].assign((size_t)words, 0);
    plan_[1].assign((size_t)words, 0);
    staging_.clear();
    R_.assign((size_t)(c.max_t * c.row_floats), 0.0f);
    schedule_.clear(); window_.clear();
    T_ = slot_ = fetches_ = staged_ = 0;
    flag_.store(0);
    commit_armed_ = false;
    inited_ = true;
    return true;
}

bool Verifier::record_window(const int32_t* tokens, uint64_t T) {
    if (!inited_ || !tokens || T < 1 || T > c_.max_t) return false;
    window_.assign(tokens, tokens + T);
    T_ = T;
    fetches_ = 0; staged_ = 0;
    flag_.store(0);
    return true;
}

bool Verifier::capture() {
    if (!inited_) return false;
    schedule_.clear();
    const uint64_t K = c_.n_layers * c_.groups;
    for (uint64_t k = 0; k < K; ++k) {
        Step s;
        s.layer = winv::layer_of(k, c_.groups);
        s.group = winv::group_of(k, c_.groups);
        s.ring = winv::ring_of_k(k);
        s.row_begin = 0; s.row_end = 0;                       // filled per window in run()
        schedule_.push_back(s);
    }
    return true;
}

bool Verifier::capture_commit() {
    if (!inited_) return false;
    commit_armed_ = true;
    return true;
}

uint64_t Verifier::run(const std::function<void(const Step&)>& body) {
    if (!inited_ || schedule_.empty() || T_ == 0) return 0;
    uint64_t n = 0;
    for (auto& s : schedule_) {
        Step cur = s;
        cur.row_begin = winv::group_begin(T_, c_.groups, s.group);
        cur.row_end = winv::group_end(T_, c_.groups, s.group);
        if (body) body(cur);
        raise_flag((uint32_t)cur.ring);
        ++n;
    }
    return n;
}

void Verifier::set_plan_slot(uint64_t group) { slot_ = group & 1; }

bool Verifier::publish_plan(uint64_t layer, uint64_t group, const std::vector<int32_t>& words) {
    if (!inited_ || layer >= c_.n_layers || group >= c_.groups) return false;
    auto& slot = plan_[(group == 0 && c_.groups == 1) ? slot_ : (group & 1)];
    if (words.size() > slot.size()) return false;
    std::copy(words.begin(), words.end(), slot.begin());
    raise_flag((uint32_t)winv::ring_of(layer, group, c_.groups));
    return true;
}

void Verifier::raise_flag(uint32_t v) {
    uint32_t cur = flag_.load(std::memory_order_relaxed);
    while (cur < v && !flag_.compare_exchange_weak(cur, v, std::memory_order_acq_rel)) {}
}

bool Verifier::fetch_dma(const uint8_t* src, uint64_t bytes) {
    if (!inited_ || !src || !expert_dispatch::may_fetch(fetches_, c_.staging_cap)) return false;
    const size_t at = staging_.size();
    staging_.resize(at + (size_t)bytes);
    std::memcpy(staging_.data() + at, src, (size_t)bytes);
    ++fetches_;
    staged_ += bytes;
    return true;
}

uint64_t Verifier::accept_len(const std::vector<int32_t>& greedy) const {
    if (T_ == 0) return 0;
    return winv::accept_from(T_, [&](uint64_t a) { return a < greedy.size() && window_[a + 1] == greedy[a]; });
}

bool Verifier::commit(uint64_t a, int64_t pos0, std::vector<int32_t>& entries) {
    entries.clear();
    if (!inited_ || !commit_armed_ || !winv::commit_in_range(T_, a)) return false;
    entries = verify::commit_vector(pos0, (int64_t)a + 1, (int64_t)c_.max_t);   // keep the a+1 emitted positions
    return true;
}

float* Verifier::final_R(uint64_t a) {
    if (!inited_ || a >= T_ || c_.row_floats == 0) return nullptr;
    return R_.data() + a * c_.row_floats;
}

}  // namespace core
}  // namespace baloon
