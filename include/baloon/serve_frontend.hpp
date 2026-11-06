// serve_frontend.hpp - request normalisation and the incremental output parser of the serving frontend.
// Effort names map onto {none, low, medium, xhigh} ("high" means xhigh); Anthropic thinking budgets bucket into
// <2048 low, <8192 medium, else xhigh (monotone). A streamed delta never exposes a partial tag: the held suffix is a
// proper prefix of the tag. The parser only moves reasoning -> content -> call -> content, never back to reasoning.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace baloon {
namespace serve_frontend {

enum class Level { Low, Medium, XHigh };

struct Effort {
    bool known = false;     // false: unrecognised spelling (the caller answers 400)
    bool disabled = false;  // true: thinking switched off
    Level level = Level::Low;
};

inline Effort effort_of(const std::string& s) {
    Effort e;
    if (s == "none" || s == "off" || s == "minimal" || s == "disabled" || s == "false") { e.known = true; e.disabled = true; }
    else if (s == "low") { e.known = true; e.level = Level::Low; }
    else if (s == "medium") { e.known = true; e.level = Level::Medium; }
    else if (s == "high" || s == "xhigh" || s == "max" || s == "maximum") { e.known = true; e.level = Level::XHigh; }
    return e;
}

inline uint32_t budget_rank(uint64_t n) { return n < 2048 ? 0u : (n < 8192 ? 1u : 2u); }
inline Level budget_level(uint64_t n) {
    const uint32_t r = budget_rank(n);
    return r == 0 ? Level::Low : (r == 1 ? Level::Medium : Level::XHigh);
}

// Longest suffix of text (searching candidates from n downward) equal to a prefix of tag.
template <class T>
inline size_t hold_from(const std::vector<T>& text, const std::vector<T>& tag, size_t n) {
    if (n > text.size()) n = text.size();
    if (n > tag.size()) n = tag.size();
    for (; n > 0; --n) {
        bool eq = true;
        for (size_t i = 0; i < n; ++i)
            if (!(text[text.size() - n + i] == tag[i])) { eq = false; break; }
        if (eq) return n;
    }
    return 0;
}
// Candidates start at len(tag)-1: only proper prefixes are ever held.
template <class T>
inline size_t hold(const std::vector<T>& text, const std::vector<T>& tag) {
    return tag.empty() ? 0 : hold_from(text, tag, tag.size() - 1);
}
inline size_t hold(const std::string& text, const std::string& tag) {
    return tag.empty() ? 0 : hold_from(std::vector<char>(text.begin(), text.end()), std::vector<char>(tag.begin(), tag.end()), tag.size() - 1);
}

enum class PState { Reasoning, Content, Call };
enum class Ev { ThinkEnd, CallStart, CallEnd, Other };
inline PState next(PState s, Ev e) {
    if (s == PState::Reasoning && e == Ev::ThinkEnd) return PState::Content;
    if (s == PState::Content && e == Ev::CallStart) return PState::Call;
    if (s == PState::Call && e == Ev::CallEnd) return PState::Content;
    return s;
}

// Removes one leading newline (at most one character).
inline std::string strip_one(const std::string& v) { return (!v.empty() && v[0] == '\n') ? v.substr(1) : v; }

struct Delta {
    std::string reasoning, content, call;  // call: raw text of completed/in-flight tool call bodies
    uint32_t calls_completed = 0;
};

// Incremental parser. feed() consumes text and returns the newly releasable deltas; finish() flushes held text.
class OutputParser {
public:
    OutputParser(bool start_in_reasoning = true, std::string think_end = "</think>",
                 std::string call_start = "<tool_call>", std::string call_end = "</tool_call>")
        : st_(start_in_reasoning ? PState::Reasoning : PState::Content),
          te_(std::move(think_end)), cs_(std::move(call_start)), ce_(std::move(call_end)) {}
    PState state() const { return st_; }
    uint32_t calls() const { return calls_; }

    Delta feed(const std::string& chunk) { buf_ += chunk; return drain(false); }
    Delta finish() { return drain(true); }

private:
    PState st_;
    std::string te_, cs_, ce_, buf_;
    uint32_t calls_ = 0;

    void emit(Delta& d, const std::string& s) const {
        if (s.empty()) return;
        (st_ == PState::Reasoning ? d.reasoning : st_ == PState::Content ? d.content : d.call) += s;
    }
    Delta drain(bool final_) {
        Delta d;
        for (;;) {
            const std::string& tag = st_ == PState::Reasoning ? te_ : st_ == PState::Content ? cs_ : ce_;
            const size_t p = buf_.find(tag);
            if (p != std::string::npos) {
                emit(d, buf_.substr(0, p));
                buf_.erase(0, p + tag.size());
                const PState before = st_;
                st_ = next(st_, st_ == PState::Reasoning ? Ev::ThinkEnd : st_ == PState::Content ? Ev::CallStart : Ev::CallEnd);
                if (before == PState::Call && st_ == PState::Content) { ++calls_; ++d.calls_completed; }
                continue;
            }
            const size_t h = final_ ? 0 : hold(buf_, tag);
            emit(d, buf_.substr(0, buf_.size() - h));
            buf_.erase(0, buf_.size() - h);
            return d;
        }
    }
};

}  // namespace serve_frontend
}  // namespace baloon
