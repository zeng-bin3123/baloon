// prefill_session.hpp - chunked prefill bookkeeping: visits every prompt position exactly once in pieces of at most
// `chunk`, in order, and reports what it did.
#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>

namespace baloon {
namespace prefill_session {

struct PrefillStats {
    uint64_t chunks = 0;     // callbacks issued
    uint64_t tokens = 0;     // positions covered (== n on completion)
    uint64_t max_chunk = 0;  // largest piece issued
};

class Prefill {
public:
    using Fn = std::function<void(uint64_t start, uint64_t len)>;
    explicit Prefill(uint64_t chunk) : chunk_(chunk ? chunk : 1) {}
    uint64_t chunk() const { return chunk_; }
    // Covers positions [0, n): pieces [c0, c0 + min(chunk, n - c0)).
    PrefillStats run(uint64_t n, const Fn& fn) const {
        PrefillStats s;
        for (uint64_t c0 = 0; c0 < n;) {
            const uint64_t t = std::min(chunk_, n - c0);
            if (fn) fn(c0, t);
            ++s.chunks; s.tokens += t; s.max_chunk = std::max(s.max_chunk, t);
            c0 += t;
        }
        return s;
    }
private:
    uint64_t chunk_;
};

}  // namespace prefill_session
}  // namespace baloon
