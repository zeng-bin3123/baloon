// serve_engine.hpp - the single resident engine behind the HTTP front: admission against the context, the
// GEN / T / DONE / ERR line protocol, the 64-entry image-embedding cache, API-key authorization and finish reasons.
// A request is admitted only if prompt + max tokens fit the context (rejected, never truncated); the engine streams at
// most max_new `T` lines and finishes with exactly one `DONE`; the cache never exceeds 64 entries (oldest evicted first);
// with no API key every request is authorized; `tool_calls` / `tool_use` iff a call was made and generation stopped
// naturally.
#pragma once
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace baloon {
namespace serve_engine {

constexpr size_t kVisionCacheMax = 64;

constexpr bool admit(uint64_t prompt_len, uint64_t max_new, uint64_t max_ctx) { return prompt_len + max_new <= max_ctx; }

struct Req { uint64_t streamed = 0, max_new = 0; bool done = false; };
inline Req on_token(Req r) {
    if (!r.done && r.streamed < r.max_new) ++r.streamed;      // never more than max_new tokens, none after DONE
    return r;
}
inline Req on_done(Req r) { r.done = true; return r; }       // idempotent: a request finishes exactly once

// Oldest-first bounded cache of image embeddings (keys only; the engine keeps the payload beside the key).
template <class K>
inline std::vector<K> insert_cache(std::vector<K> cache, K x) {
    cache.push_back(std::move(x));
    if (cache.size() > kVisionCacheMax) cache.erase(cache.begin());
    return cache;
}

inline bool authorized(bool configured, const std::string& key, const std::string& given) {
    return !configured || given == key;
}

enum class Finish { Stop, Length, ToolCalls };
inline Finish openai_finish(uint64_t calls, bool stopped) {
    return (stopped && calls > 0) ? Finish::ToolCalls : (stopped ? Finish::Stop : Finish::Length);
}
enum class Stop { EndTurn, MaxTokens, ToolUse };
inline Stop anthropic_stop(bool used_tool, bool stopped) {
    return (used_tool && stopped) ? Stop::ToolUse : (stopped ? Stop::EndTurn : Stop::MaxTokens);
}
inline const char* finish_name(Finish f) { return f == Finish::ToolCalls ? "tool_calls" : (f == Finish::Stop ? "stop" : "length"); }
inline const char* stop_name(Stop s) { return s == Stop::ToolUse ? "tool_use" : (s == Stop::EndTurn ? "end_turn" : "max_tokens"); }
constexpr uint64_t total_tokens(uint64_t prompt, uint64_t completion) { return prompt + completion; }

// ---- line protocol ------------------------------------------------------------------------------------------------------
struct GenLine { bool ok = false; uint64_t max_new = 0; std::vector<int64_t> tokens; };
// "GEN <max_new> <tok> <tok> ..." (at least one prompt token, max_new >= 1).
inline GenLine parse_gen(const std::string& line) {
    GenLine g;
    if (line.compare(0, 4, "GEN ") != 0) return g;
    size_t i = 4;
    auto next = [&](long long& v) {
        while (i < line.size() && line[i] == ' ') ++i;
        if (i >= line.size()) return false;
        size_t j = i;
        bool neg = false;
        if (line[j] == '-') { neg = true; ++j; }
        if (j >= line.size() || line[j] < '0' || line[j] > '9') return false;
        long long x = 0;
        for (; j < line.size() && line[j] >= '0' && line[j] <= '9'; ++j) x = x * 10 + (line[j] - '0');
        if (j < line.size() && line[j] != ' ') return false;
        v = neg ? -x : x; i = j;
        return true;
    };
    long long m;
    if (!next(m) || m < 1) return g;
    g.max_new = (uint64_t)m;
    long long t;
    while (next(t)) g.tokens.push_back(t);
    while (i < line.size() && line[i] == ' ') ++i;
    if (i != line.size() || g.tokens.empty()) { g.tokens.clear(); g.max_new = 0; return g; }
    g.ok = true;
    return g;
}
inline std::string fmt_token(int64_t t) { return "T " + std::to_string(t); }
inline std::string fmt_done() { return "DONE"; }
inline std::string fmt_err(const std::string& m) { return "ERR " + m; }

// Engine-side request: feeds `T` lines until max_new, then exactly one DONE.
class Stream {
public:
    explicit Stream(uint64_t max_new) { r_.max_new = max_new; }
    std::optional<std::string> token(int64_t t) {
        const Req n = on_token(r_);
        if (n.streamed == r_.streamed) return std::nullopt;
        r_ = n;
        return fmt_token(t);
    }
    std::optional<std::string> done() {
        if (r_.done) return std::nullopt;
        r_ = on_done(r_);
        return fmt_done();
    }
    const Req& req() const { return r_; }

private:
    Req r_;
};

}  // namespace serve_engine
}  // namespace baloon
