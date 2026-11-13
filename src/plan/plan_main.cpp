// plan_main.cpp - VRAM plan command-line tool.
// Usage: plan_main [context_tokens [pool_bytes]]
// Prints the plan or refusal for the given context size and VRAM pool.
// Port: src/plan/plan_main.cpp → main (plan entry-point for unit u7f1ccea730).
#include "baloon/plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <variant>

static void usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s [context_tokens [pool_bytes]]\n", argv0);
    std::fprintf(stderr, "  context_tokens  number of tokens in context (default: 32768)\n");
    std::fprintf(stderr, "  pool_bytes      VRAM pool bytes (default: derived from 32 GiB device)\n");
}

int main(int argc, char** argv) {
    uint64_t ctx  = 32768;
    // Default: 32 GiB device minus 512 MiB runtime reserve.
    uint64_t pool = baloon::plan::vram_pool_bytes(32ull * 1024 * 1024 * 1024,
                                                   512ull * 1024 * 1024);

    if (argc >= 2) {
        if (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        ctx = static_cast<uint64_t>(std::stoull(argv[1]));
    }
    if (argc >= 3) {
        pool = static_cast<uint64_t>(std::stoull(argv[2]));
    }

    baloon::plan::Geometry g;
    baloon::plan::Costs    c;
    // Account for the fixed GDN recurrent state.
    c.state = baloon::plan::state_bytes(g);

    const uint64_t kv = baloon::plan::kv_bytes_per_token(g);
    std::fprintf(stdout, "geometry: %llu layers (%llu QSA, %llu GDN)\n",
                 (unsigned long long)g.n_layers,
                 (unsigned long long)g.n_qsa,
                 (unsigned long long)g.n_gdn());
    std::fprintf(stdout, "kv_bytes_per_token: %llu  state_bytes: %llu\n",
                 (unsigned long long)kv, (unsigned long long)c.state);
    std::fprintf(stdout, "pool: %llu  context: %llu\n",
                 (unsigned long long)pool, (unsigned long long)ctx);

    baloon::plan::Result r = baloon::plan::make_plan(ctx, g, c, pool);
    std::puts(baloon::plan::to_string(r).c_str());

    if (std::holds_alternative<baloon::plan::Plan>(r)) {
        const auto& p = std::get<baloon::plan::Plan>(r);
        std::fprintf(stdout, "dram_experts: %llu\n",
                     (unsigned long long)(c.dense + c.embd));
        return 0;
    }
    // DoesNotClose: exit non-zero so callers can detect refusal.
    const auto& d = std::get<baloon::plan::DoesNotClose>(r);
    std::fprintf(stderr, "refusal reason: %s\n",
                 baloon::plan::to_string(d.reason));
    return 1;
}

