// verify_window_commit.hpp - speculative verify window: accepted draft prefix, n_keep = accept+1, commit replays a prefix
// through the state update.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace verify_commit {

constexpr uint64_t kMaxT = 8;

// State after committing the first n tokens of the window.
template <class S, class Step> S commit_state(Step step, S s, const std::vector<uint32_t>& toks, uint64_t n) {
    for (uint64_t i = 0; i < n && i < toks.size(); ++i) s = step(s, toks[i]);
    return s;
}

// Longest prefix of drafts matching greedy outputs.
inline uint64_t accept_len(const std::vector<uint32_t>& drafts, const std::vector<uint32_t>& greedy) {
    uint64_t n = 0;
    while (n < drafts.size() && n < greedy.size() && drafts[n] == greedy[n]) ++n;
    return n;
}
inline uint64_t n_keep(const std::vector<uint32_t>& drafts, const std::vector<uint32_t>& greedy) {
    return accept_len(drafts, greedy) + 1;
}
constexpr uint64_t max_drafts(uint64_t T) { return T - 1; }

}  // namespace verify_commit
}  // namespace baloon
