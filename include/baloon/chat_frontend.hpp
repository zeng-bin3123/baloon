// chat_frontend.hpp - symbol-level contracts of the text frontend: signed thinking-budget ranks, partial-tag hold-back,
// parameter newline stripping, image-pad expansion, context admission and detokenizer delta lengths.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace baloon {
namespace chat_frontend {

// 0 = low, 1 = medium, 2 = xhigh. Monotone in n; negative budgets are low.
inline uint32_t budget_rank(int64_t n) { return n < 2048 ? 0u : (n < 8192 ? 1u : 2u); }

// Longest n in 1..tag.size()-1 such that the last n symbols of text equal the first n of tag (0 if none).
inline size_t hold_one(const std::vector<uint64_t>& text, const std::vector<uint64_t>& tag) {
    size_t best = 0;
    for (size_t n = 1; n < tag.size(); ++n) {
        if (n > text.size()) break;
        if (std::equal(text.end() - (std::ptrdiff_t)n, text.end(), tag.begin())) best = n;
    }
    return best;
}
// Emitted prefix length for a buffer of length len when `keep` symbols are held back.
inline size_t emitted_len(size_t len, size_t keep) { return len - std::min(len, keep); }

constexpr uint64_t kNewline = 10;
inline std::vector<uint64_t> drop_lead(std::vector<uint64_t> l) {
    if (!l.empty() && l.front() == kNewline) l.erase(l.begin());
    return l;
}
inline std::vector<uint64_t> drop_trail(std::vector<uint64_t> l) {
    if (!l.empty() && l.back() == kNewline) l.pop_back();
    return l;
}
// At most one newline removed at each end.
inline std::vector<uint64_t> strip_nl(const std::vector<uint64_t>& l) { return drop_trail(drop_lead(l)); }

// Each pad token becomes counts[i] pads (in order); pads beyond the counts stay single; no counts -> unchanged.
inline std::vector<uint64_t> expand(uint64_t pad, const std::vector<uint64_t>& ids, const std::vector<uint64_t>& counts) {
    std::vector<uint64_t> out;
    size_t ci = 0;
    for (uint64_t t : ids) {
        if (t == pad && ci < counts.size()) out.insert(out.end(), counts[ci++], pad);
        else out.push_back(t);
    }
    return out;
}

// Admission: rejected (never truncated) when prompt + max_new exceeds the context.
inline bool fits(uint64_t prompt, uint64_t max_new, uint64_t max_ctx) { return prompt + max_new <= max_ctx; }
inline uint64_t delta_len(uint64_t text_len, uint64_t sent) { return text_len > sent ? text_len - sent : 0; }

}  // namespace chat_frontend
}  // namespace baloon
