// generate_options.hpp - option validation, teacher forcing and speculative acceptance of the generate driver.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace gen_opts {

struct Opts { uint64_t n_tokens, max_new, max_context; };

// max_new > 0, max_context > 0, max_new <= max_context, prompt fits in max_context - max_new, and positions (+8 margin) fit int32.
constexpr bool valid_opts(const Opts& o) {
    return 0 < o.n_tokens && 0 < o.max_new && 0 < o.max_context && o.max_new <= o.max_context &&
           o.n_tokens <= o.max_context - o.max_new && o.max_context <= 2147483647ull - 8;
}

// Next input: the prompt's own next token while the prompt lasts, otherwise the sampled token.
inline int64_t next_input(const std::vector<int64_t>& prompt, uint64_t pos, int64_t sampled) {
    return pos + 1 < prompt.size() ? prompt[(size_t)(pos + 1)] : sampled;
}

// Number of accepted drafts: the longest prefix of drafts equal to the window's verified outputs. A round emits accepted + 1 tokens.
inline uint64_t accepted(const std::vector<int64_t>& drafts, const std::vector<int64_t>& outs) {
    uint64_t a = 0;
    while (a < drafts.size() && a < outs.size() && drafts[(size_t)a] == outs[(size_t)a]) ++a;
    return a;
}
constexpr uint64_t round_emits(uint64_t accepted_drafts) { return accepted_drafts + 1; }

// The routing trace is written after d.layers advanced: it records d.layers - 1, valid iff 1 <= d.layers <= 48.
constexpr bool trace_layers_valid(uint64_t layers_after) { return 1 <= layers_after && layers_after <= 48; }
constexpr uint64_t trace_layer(uint64_t layers_after) { return layers_after - 1; }

// Slots lent to the prompt path: k = ceil(need / blob); k * blob >= need.
constexpr uint64_t ceil_div(uint64_t need, uint64_t blob) { return (need + blob - 1) / blob; }

}  // namespace gen_opts
}  // namespace baloon
