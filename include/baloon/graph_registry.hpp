// graph_registry.hpp - captured-graph registry keyed by (layer type, n_tokens): record-once, find exact key,
// zero-node captures refused.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>

namespace baloon {
namespace graph_registry {

using Key = std::pair<uint64_t, uint64_t>;

constexpr bool end_ok(uint64_t nodes) { return nodes != 0; }

// Bounded poll: n attempts at times t, t+1, ...; true iff some poll observes completion. Zero bound never succeeds.
template <class Pred>
constexpr bool wait_fuel(uint64_t n, Pred done, uint64_t t = 0) {
    for (uint64_t i = 0; i < n; ++i)
        if (done(t + i)) return true;
    return false;
}

class Registry {
public:
    // First match wins, as in a map. Returns false when absent.
    bool find(Key k, uint64_t& v) const {
        for (const auto& e : m_)
            if (e.first == k) { v = e.second; return true; }
        return false;
    }
    bool has(Key k) const { uint64_t v; return find(k, v); }
    // Returns true if a new graph was added; an existing key is a no-op ("already recorded").
    bool record(Key k, uint64_t v) {
        if (has(k)) return false;
        m_.insert(m_.begin(), {k, v});
        ++captures_;
        return true;
    }
    uint64_t size() const { return m_.size(); }
    uint64_t captures() const { return captures_; }
private:
    std::vector<std::pair<Key, uint64_t>> m_;
    uint64_t captures_ = 0;
};

}  // namespace graph_registry
}  // namespace baloon
