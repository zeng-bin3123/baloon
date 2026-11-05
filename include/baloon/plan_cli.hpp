// plan_cli.hpp - command line front end of the VRAM planner: strict number parsing, option parsing and the run loop.
//
//   plan_cli <context_tokens> <pool_bytes> [key=value ...]
//   plan_cli <context_tokens> --device <bytes> --reserve <bytes> [key=value ...]
//
// Numbers are decimal u64 with an optional binary suffix k/m/g (x2^10/2^20/2^30); an empty string, a sign, trailing
// junk or a value that does not fit in u64 is a usage error (never wrapped). Keys: the Geometry fields (n_layers, n_qsa,
// n_kv_heads, head_dim, kv_group, qsa_block, idx_key_heads, idx_key_dim, ssm_state, ssm_v_heads, ssm_conv, ssm_channels)
// and the Costs fields (expert_blob, dense, embd, mtp, workspace, state).
// Exit codes: 0 the plan closes (printed on stdout), 1 it does not close (reason on stderr), 2 usage error.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>
#include "baloon/plan.hpp"

namespace baloon {
namespace plan {
namespace cli {

// Strict u64 parse with optional k/m/g suffix. Returns false on any malformed or overflowing input.
inline bool parse_u64(const std::string& s, uint64_t& out) {
    if (s.empty()) return false;
    size_t i = 0;
    unsigned __int128 v = 0;
    for (; i < s.size() && s[i] >= '0' && s[i] <= '9'; ++i) {
        v = v * 10 + (unsigned)(s[i] - '0');
        if (v > (unsigned __int128)UINT64_MAX) return false;
    }
    if (i == 0) return false;
    unsigned shift = 0;
    if (i < s.size()) {
        if (i + 1 != s.size()) return false;
        const char c = s[i];
        shift = (c == 'k' || c == 'K') ? 10 : (c == 'm' || c == 'M') ? 20 : (c == 'g' || c == 'G') ? 30 : 0;
        if (shift == 0) return false;
    }
    v <<= shift;
    if (v > (unsigned __int128)UINT64_MAX) return false;
    out = (uint64_t)v;
    return true;
}

struct Config {
    uint64_t ctx = 0, pool = 0, device = 0, reserve = 0;
    bool have_pool = false, have_device = false, have_reserve = false;
    Geometry g;
    Costs c;
};

// Assigns one named field; false if the key is unknown.
inline bool set_field(Config& cfg, const std::string& key, uint64_t v) {
    struct F { const char* k; uint64_t* p; };
    const F fields[] = {
        {"n_layers", &cfg.g.n_layers}, {"n_qsa", &cfg.g.n_qsa}, {"n_kv_heads", &cfg.g.n_kv_heads},
        {"head_dim", &cfg.g.head_dim}, {"kv_group", &cfg.g.kv_group}, {"qsa_block", &cfg.g.qsa_block},
        {"idx_key_heads", &cfg.g.idx_key_heads}, {"idx_key_dim", &cfg.g.idx_key_dim}, {"ssm_state", &cfg.g.ssm_state},
        {"ssm_v_heads", &cfg.g.ssm_v_heads}, {"ssm_conv", &cfg.g.ssm_conv}, {"ssm_channels", &cfg.g.ssm_channels},
        {"expert_blob", &cfg.c.expert_blob}, {"dense", &cfg.c.dense}, {"embd", &cfg.c.embd}, {"mtp", &cfg.c.mtp},
        {"workspace", &cfg.c.workspace}, {"state", &cfg.c.state},
    };
    for (const F& f : fields) if (key == f.k) { *f.p = v; return true; }
    return false;
}

// Parses argv[1..]; on failure returns false and fills err.
inline bool parse_args(const std::vector<std::string>& a, Config& cfg, std::string& err) {
    std::vector<std::string> pos;
    for (size_t i = 0; i < a.size(); ++i) {
        const std::string& s = a[i];
        if (s == "--device" || s == "--reserve") {
            uint64_t v;
            if (i + 1 >= a.size() || !parse_u64(a[i + 1], v)) { err = "bad value for " + s; return false; }
            if (s == "--device") { cfg.device = v; cfg.have_device = true; } else { cfg.reserve = v; cfg.have_reserve = true; }
            ++i;
        } else if (s.find('=') != std::string::npos) {
            const size_t eq = s.find('=');
            uint64_t v;
            if (!parse_u64(s.substr(eq + 1), v)) { err = "bad number in " + s; return false; }
            if (!set_field(cfg, s.substr(0, eq), v)) { err = "unknown key " + s.substr(0, eq); return false; }
        } else if (s.size() > 1 && s[0] == '-' && s[1] == '-') {
            err = "unknown option " + s; return false;
        } else pos.push_back(s);
    }
    if (pos.empty() || pos.size() > 2) { err = "expected <context_tokens> and a pool (<pool_bytes> or --device)"; return false; }
    if (!parse_u64(pos[0], cfg.ctx)) { err = "bad context_tokens"; return false; }
    if (pos.size() == 2) {
        if (cfg.have_device || cfg.have_reserve) { err = "pool_bytes conflicts with --device/--reserve"; return false; }
        if (!parse_u64(pos[1], cfg.pool)) { err = "bad pool_bytes"; return false; }
        cfg.have_pool = true;
    } else {
        if (!cfg.have_device) { err = "no pool: give <pool_bytes> or --device"; return false; }
        cfg.pool = vram_pool_bytes(cfg.device, cfg.reserve);
        cfg.have_pool = true;
    }
    return true;
}

inline const char* usage() {
    return "usage: plan_cli <context_tokens> <pool_bytes> [key=value ...]\n"
           "       plan_cli <context_tokens> --device <bytes> [--reserve <bytes>] [key=value ...]\n";
}

// Whole program minus process plumbing: returns the exit code, writes to the given streams.
inline int run(const std::vector<std::string>& args, std::FILE* out, std::FILE* err) {
    Config cfg;
    std::string e;
    if (!parse_args(args, cfg, e)) { std::fprintf(err, "plan_cli: %s\n%s", e.c_str(), usage()); return 2; }
    const Result r = make_plan(cfg.ctx, cfg.g, cfg.c, cfg.pool);
    if (std::holds_alternative<Plan>(r)) { std::fprintf(out, "%s\n", to_string(r).c_str()); return 0; }
    std::fprintf(err, "%s\n", to_string(r).c_str());
    return 1;
}

}  // namespace cli
}  // namespace plan
}  // namespace baloon
