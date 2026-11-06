// core_hit_hook.hpp - expert "hit" observation hooks (host model). A hit hook is called once per routed expert pick with
// the phase it happened in; TokenHits / VerifyHits record them for one token / one verify window.
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

namespace baloon {
namespace core {

enum class HitPhase : int { Decode = 0, Prefill = 1, Verify = 2 };
using HitFn = std::function<void(HitPhase phase, int layer, int expert)>;

struct HitRec { int layer = 0, expert = 0; };

struct TokenHits {
    std::vector<HitRec> hits;
    void record(int layer, int expert) { hits.push_back({layer, expert}); }
    void clear() { hits.clear(); }
    HitFn sink(HitPhase want = HitPhase::Decode) {
        return [this, want](HitPhase p, int l, int e) { if (p == want) record(l, e); };
    }
    uint64_t count_layer(int layer) const { uint64_t n = 0; for (auto& h : hits) n += h.layer == layer; return n; }
};

// One TokenHits per window position (row); rows outside [0, T) are rejected.
struct VerifyHits {
    std::vector<TokenHits> rows;
    void reset(uint64_t T) { rows.assign((size_t)T, TokenHits{}); }
    bool record(uint64_t row, int layer, int expert) { if (row >= rows.size()) return false; rows[(size_t)row].record(layer, expert); return true; }
    uint64_t total() const { uint64_t n = 0; for (auto& r : rows) n += r.hits.size(); return n; }
};

}  // namespace core
}  // namespace baloon
