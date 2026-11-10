// frontend_contracts.hpp - chat frontend and serve-protocol decisions in their contract form: reasoning-effort spellings, Anthropic
// budget ranks (1 low, 2 medium, 3 xhigh), partial-tag hold-back, newline stripping, context admission, image-pad expansion,
// incremental detokenisation, finish / stop reasons, content-block opening, the 64-entry cache bound and API-key checks.
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace baloon {
namespace frontend_contract {

// 0 none, 1 low, 2 medium, 3 xhigh; nullopt = unknown spelling (client error 400).
inline std::optional<int> effort_level(const std::string& s) {
    if (s == "none" || s == "off" || s == "minimal" || s == "disabled") return 0;
    if (s == "low") return 1;
    if (s == "medium") return 2;
    if (s == "high" || s == "xhigh" || s == "max" || s == "maximum") return 3;
    return std::nullopt;
}
constexpr int budget_rank(uint64_t n) { return n < 2048 ? 1 : (n < 8192 ? 2 : 3); }

// Longest suffix of text that is a PROPER non-empty prefix of tag (always shorter than the tag).
inline size_t hold_len(const std::vector<uint32_t>& text, const std::vector<uint32_t>& tag) {
    size_t best = 0;
    for (size_t n = 1; n < tag.size() && n <= text.size(); ++n) {
        bool eq = true;
        for (size_t i = 0; i < n; ++i) if (text[text.size() - n + i] != tag[i]) { eq = false; break; }
        if (eq) best = n;
    }
    return best;
}
// One leading and one trailing newline dropped.
inline std::vector<uint32_t> strip_nl(std::vector<uint32_t> l) {
    if (!l.empty() && l.front() == 10) l.erase(l.begin());
    if (!l.empty() && l.back() == 10) l.pop_back();
    return l;
}

constexpr bool accepts(uint64_t prompt, uint64_t max_new, uint64_t ctx) { return prompt + max_new <= ctx; }
inline std::vector<uint32_t> expand(uint32_t pad, const std::vector<uint32_t>& ids, const std::vector<uint64_t>& counts) {
    std::vector<uint32_t> out; size_t ci = 0;
    for (uint32_t t : ids) {
        if (t == pad && ci < counts.size()) out.insert(out.end(), counts[ci++], pad);
        else out.push_back(t);
    }
    return out;
}
inline std::vector<uint32_t> delta_of(const std::vector<uint32_t>& old_, const std::vector<uint32_t>& new_) {
    return new_.size() > old_.size() ? std::vector<uint32_t>(new_.begin() + (std::ptrdiff_t)old_.size(), new_.end()) : std::vector<uint32_t>();
}

enum class Finish { Stop, Length };
constexpr int openai_finish(uint64_t calls, Finish f) { return (calls > 0 && f == Finish::Stop) ? 2 : (f == Finish::Stop ? 0 : 1); }
constexpr int anthropic_stop(bool used_tool, Finish f) { return (used_tool && f == Finish::Stop) ? 2 : (f == Finish::Stop ? 0 : 1); }

enum class Kind { Reasoning, Content, Tool };
// Tool calls always open a block; reasoning/text continue while the kind is unchanged. cur < 0 means no block open.
constexpr bool new_block(int cur, Kind k) { return k == Kind::Tool ? true : cur != (int)k; }

constexpr size_t kImageCacheMax = 64;
// Drop the oldest entry when the cache exceeds the bound.
template <class T> inline std::vector<T> evict(std::vector<T> l) { if (l.size() > kImageCacheMax) l.erase(l.begin()); return l; }
inline bool authorized(const std::string& configured, const std::string& presented) { return configured.empty() || presented == configured; }

}  // namespace frontend_contract
}  // namespace baloon
