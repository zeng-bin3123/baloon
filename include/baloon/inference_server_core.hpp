// Pure logic of the OpenAI / Anthropic HTTP front end over a token-id engine.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace baloon::server {

// ---- context admission -------------------------------------------------------

// Admitted iff prompt + max_new <= ctx; never truncation (overflow-safe).
inline bool fits_context(uint64_t prompt_len, uint64_t max_new, uint64_t ctx) {
    return prompt_len <= ctx && max_new <= ctx - prompt_len;
}

// ---- finish reasons ----------------------------------------------------------

enum class Finish { Stop, Length };
enum class OpenFinish { Stop, Length, ToolCalls };
enum class AnthStop { EndTurn, MaxTokens, ToolUse };

inline OpenFinish openai_finish(uint64_t calls, Finish f) {
    if (f == Finish::Length) return OpenFinish::Length;
    return calls > 0 ? OpenFinish::ToolCalls : OpenFinish::Stop;
}

inline AnthStop anthropic_stop(bool used_tool, Finish f) {
    if (f == Finish::Length) return AnthStop::MaxTokens;
    return used_tool ? AnthStop::ToolUse : AnthStop::EndTurn;
}

// ---- authorisation -----------------------------------------------------------

// No configured key: everything is authorised; otherwise the presented key must equal it.
inline bool authorized(const std::string& key, const std::string& presented) {
    return key.empty() || presented == key;
}

// ---- image pad expansion -----------------------------------------------------

struct PadExpansion {
    std::vector<int32_t> ids;       // expanded token ids
    std::vector<uint32_t> unused;   // image token counts not consumed
};

// Each `pad` token is replaced by that many pads (the next image's token count).
// A pad with no remaining count is kept as is (the caller then reports a mismatch).
inline PadExpansion expand_pads(int32_t pad, const std::vector<int32_t>& ids,
                                const std::vector<uint32_t>& counts) {
    PadExpansion r;
    size_t ci = 0;
    for (int32_t t : ids) {
        if (t == pad) {
            if (ci < counts.size()) {
                r.ids.insert(r.ids.end(), counts[ci], pad);
                ++ci;
            } else {
                r.ids.push_back(t);
            }
        } else {
            r.ids.push_back(t);
        }
    }
    r.unused.assign(counts.begin() + static_cast<std::ptrdiff_t>(ci), counts.end());
    return r;
}

// ---- bounded cache -----------------------------------------------------------

inline constexpr size_t kVisionCacheCap = 64;

// Pure form: append, evict the oldest entry when the size would exceed 64.
template <class T>
inline std::vector<T> cache_insert(std::vector<T> c, T x) {
    c.push_back(std::move(x));
    if (c.size() > kVisionCacheCap) c.erase(c.begin());
    return c;
}

// Stateful form of the same rule.
template <class T>
class BoundedCache {
public:
    void insert(T x) {
        q_.push_back(std::move(x));
        if (q_.size() > kVisionCacheCap) q_.pop_front();
    }
    size_t size() const { return q_.size(); }
    const std::deque<T>& entries() const { return q_; }

private:
    std::deque<T> q_;
};

// ---- incremental detokenisation ---------------------------------------------

// True when the decoded text ends in the Unicode replacement character (U+FFFD).
inline bool ends_with_replacement(const std::string& s) {
    static const char kRepl[] = "\xEF\xBF\xBD";
    return s.size() >= 3 && s.compare(s.size() - 3, 3, kRepl, 3) == 0;
}

struct DetokStep {
    std::string emitted;
    size_t sent;
};

// One push: with a pending partial character emit nothing and keep the cursor;
// otherwise emit exactly the not-yet-sent suffix and advance the cursor to the end.
inline DetokStep detok_push(const std::string& text, size_t sent, bool partial) {
    if (partial) return {std::string(), sent};
    return {sent < text.size() ? text.substr(sent) : std::string(), text.size()};
}

class Detokenizer {
public:
    // Feed the full decoded text so far; returns the delta to send.
    std::string push(const std::string& decoded) {
        DetokStep s = detok_push(decoded, sent_, ends_with_replacement(decoded));
        sent_ = s.sent;
        return s.emitted;
    }
    size_t sent() const { return sent_; }

private:
    size_t sent_ = 0;
};

// ---- Anthropic content blocks -----------------------------------------------

enum class BlockKind { Thinking, Text, ToolUse };

// A new block starts when the kind changes, and always for a tool_use.
inline bool block_starts(std::optional<BlockKind> open_kind, BlockKind want) {
    return !(open_kind && *open_kind == want) || want == BlockKind::ToolUse;
}

}  // namespace baloon::server

