// serve_server.hpp - contracts of the HTTP serving layer: never-truncate context guard, incremental detokenisation
// length model, token accounting up to the stop token, finish-reason mapping for both API dialects, the 64-entry
// image-embedding cache bound, max-token defaulting and API-key authorisation.
#pragma once
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace baloon {
namespace serve_server {

// Accept iff prompt + max_new <= context; the prompt is never altered.
inline bool accepts(uint64_t prompt_len, uint64_t max_new, uint64_t ctx) { return prompt_len + max_new <= ctx; }

// Detokenizer length model: sent advances to len unless a partial character is held; never backwards.
inline uint64_t detok_push(uint64_t sent, uint64_t len, bool held) { return held ? sent : len; }
inline uint64_t delta_len(uint64_t sent, uint64_t len, bool held) { return held ? 0 : len - sent; }

// (completion tokens counted, stopped by stop token); the stop token itself is counted.
template <class IsStop>
inline std::pair<uint64_t, bool> run_n(IsStop is_stop, const std::vector<uint32_t>& toks) {
    uint64_t n = 0;
    for (uint32_t t : toks) { ++n; if (is_stop(t)) return {n, true}; }
    return {n, false};
}

enum class Finish { Stop, Length };
// 0 = stop, 1 = length, 2 = tool_calls
inline int openai_finish(uint64_t calls, Finish f) { return (calls > 0 && f == Finish::Stop) ? 2 : (f == Finish::Stop ? 0 : 1); }
// 0 = end_turn, 1 = max_tokens, 2 = tool_use
inline int anthropic_stop(bool used_tool, Finish f) { return (used_tool && f == Finish::Stop) ? 2 : (f == Finish::Stop ? 0 : 1); }

constexpr size_t IMAGE_CACHE_MAX = 64;
inline size_t cache_add(size_t size) { return size + 1 > IMAGE_CACHE_MAX ? size : size + 1; }

// FIFO image cache: insert, then evict the oldest while over the bound.
template <class V>
class ImageCache {
public:
    size_t size() const { return q_.size(); }
    bool has(const std::string& k) const { for (auto& e : q_) if (e.first == k) return true; return false; }
    const V* get(const std::string& k) const { for (auto& e : q_) if (e.first == k) return &e.second; return nullptr; }
    void put(const std::string& k, V v) {
        for (auto& e : q_) if (e.first == k) { e.second = std::move(v); return; }
        q_.emplace_back(k, std::move(v));
        while (q_.size() > IMAGE_CACHE_MAX) q_.pop_front();
    }
private:
    std::deque<std::pair<std::string, V>> q_;
};

// max_completion_tokens or max_tokens or 1024
inline uint64_t pick_max(uint64_t a, uint64_t b) { return a != 0 ? a : (b != 0 ? b : 1024); }

// No key configured, or the presented key matches.
inline bool authorized(const std::string& configured, const std::string& presented) {
    return configured.empty() || presented == configured;
}

}  // namespace serve_server
}  // namespace baloon
