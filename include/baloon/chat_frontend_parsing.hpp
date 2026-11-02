// Text-only request/response layer helpers: reasoning-effort normalisation, thinking-budget
// mapping, the partial-tag hold-back rule of the incremental parser, tool-call value trimming.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace baloon::frontend {

// ---- reasoning effort -------------------------------------------------------

enum class Level { Low, Medium, XHigh };

inline int level_rank(Level l) {
    switch (l) {
        case Level::Low: return 0;
        case Level::Medium: return 1;
        case Level::XHigh: return 2;
    }
    return 0;
}

enum class EffortKey {
    None, Off, Minimal, Disabled, False,
    Low, Medium, High, XHigh, Max, Maximum,
    Unknown
};

// Template kwargs: nothing, thinking disabled, or a level.
struct Kw {
    enum class Kind { Empty, NoThink, Level } kind = Kind::Empty;
    Level level = Level::Low;  // meaningful only when kind == Level
    bool operator==(const Kw& o) const {
        return kind == o.kind && (kind != Kind::Level || level == o.level);
    }
};

// Client spelling (exact, lower-case) -> key; anything else is Unknown.
inline EffortKey effort_key_from_string(const std::string& s) {
    if (s == "none") return EffortKey::None;
    if (s == "off") return EffortKey::Off;
    if (s == "minimal") return EffortKey::Minimal;
    if (s == "disabled") return EffortKey::Disabled;
    if (s == "false") return EffortKey::False;
    if (s == "low") return EffortKey::Low;
    if (s == "medium") return EffortKey::Medium;
    if (s == "high") return EffortKey::High;
    if (s == "xhigh") return EffortKey::XHigh;
    if (s == "max") return EffortKey::Max;
    if (s == "maximum") return EffortKey::Maximum;
    return EffortKey::Unknown;
}

// Effort -> kwargs, or nullopt (HTTP 400) for an unknown spelling; never throws.
inline std::optional<Kw> effort_kwargs(EffortKey k) {
    auto lvl = [](Level l) { return Kw{Kw::Kind::Level, l}; };
    switch (k) {
        case EffortKey::None:
        case EffortKey::Off:
        case EffortKey::Minimal:
        case EffortKey::Disabled:
        case EffortKey::False: return Kw{Kw::Kind::NoThink, Level::Low};
        case EffortKey::Low: return lvl(Level::Low);
        case EffortKey::Medium: return lvl(Level::Medium);
        case EffortKey::High:
        case EffortKey::XHigh:
        case EffortKey::Max:
        case EffortKey::Maximum: return lvl(Level::XHigh);
        case EffortKey::Unknown: return std::nullopt;
    }
    return std::nullopt;
}

inline std::optional<Kw> effort_kwargs(const std::string& spelling) {
    return effort_kwargs(effort_key_from_string(spelling));
}

// Anthropic budget_tokens -> level: < 2048 low, < 8192 medium, else xhigh (monotone).
inline Level budget_effort(uint64_t n) {
    if (n < 2048) return Level::Low;
    if (n < 8192) return Level::Medium;
    return Level::XHigh;
}

// ---- partial tag hold-back --------------------------------------------------

// Length of the longest suffix of `text` that is a proper prefix of `tag`
// (strictly shorter than the tag; 0 for an empty tag).
template <class Seq>
inline size_t hold(const Seq& text, const Seq& tag) {
    if (tag.empty()) return 0;
    size_t n = tag.size() - 1;
    if (n > text.size()) n = text.size();
    for (size_t k = n; k > 0; --k) {
        bool ok = true;
        for (size_t i = 0; i < k; ++i) {
            if (!(text[text.size() - k + i] == tag[i])) { ok = false; break; }
        }
        if (ok) return k;
    }
    return 0;
}

// Incremental scanner for one tag: never emits a partial tag.
class TagScanner {
public:
    explicit TagScanner(std::string tag) : tag_(std::move(tag)) {}

    // Feed a chunk. Appends the safely emittable text to `out`. Returns true once the tag was
    // found; text before the tag is emitted and the remainder after the tag stays in buffer().
    bool feed(const std::string& chunk, std::string& out) {
        buf_ += chunk;
        if (!tag_.empty()) {
            size_t p = buf_.find(tag_);
            if (p != std::string::npos) {
                out += buf_.substr(0, p);
                buf_.erase(0, p + tag_.size());
                return true;
            }
        }
        size_t h = hold(buf_, tag_);
        out += buf_.substr(0, buf_.size() - h);
        buf_.erase(0, buf_.size() - h);
        return false;
    }

    // End of stream without the tag: release whatever was held back.
    void flush(std::string& out) {
        out += buf_;
        buf_.clear();
    }

    const std::string& buffer() const { return buf_; }

private:
    std::string tag_;
    std::string buf_;
};

// ---- tool-call value trimming ----------------------------------------------

// Loses at most one leading and one trailing newline (applied in that order).
inline std::string trim_value(const std::string& v) {
    size_t b = 0, e = v.size();
    if (b < e && v[b] == '\n') ++b;
    if (b < e && v[e - 1] == '\n') --e;
    return v.substr(b, e - b);
}

// ---- end of generation ------------------------------------------------------

enum class PState { Reasoning, Content, Call };
enum class EvKind { Reasoning, Content, ToolCall };

// An unterminated tool call is flushed as content, never as a tool_call event.
inline EvKind finish_kind(PState s) {
    switch (s) {
        case PState::Reasoning: return EvKind::Reasoning;
        case PState::Content: return EvKind::Content;
        case PState::Call: return EvKind::Content;
    }
    return EvKind::Content;
}

}  // namespace baloon::frontend

