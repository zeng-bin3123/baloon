// generate_cli.cpp - entry point of the `generate` driver. Runs the decode loop on the deterministic synthetic backend
// (the real model backend plugs in through gencli::Drive). Exit codes: 0 ok, 1 run failed, 2 usage error.
#include <cstdio>
#include <iostream>

#include "baloon/generate_cli.hpp"

using namespace baloon::gencli;

int main(int argc, char** argv) {
    Options o;
    std::string err;
    if (!parse_args(argc, argv, o, err)) {
        std::fprintf(stderr, "error: %s\n%s", err.c_str(), usage(argc > 0 ? argv[0] : "generate").c_str());
        return 2;
    }
    if (o.help) { std::fputs(usage(argc > 0 ? argv[0] : "generate").c_str(), stdout); return 0; }
    err = validate(o);
    if (!err.empty()) {
        std::fprintf(stderr, "error: %s\n%s", err.c_str(), usage(argc > 0 ? argv[0] : "generate").c_str());
        return 2;
    }
    const int64_t vocab = o.vocab;
    auto make = [vocab](int) { return synthetic_drive(vocab); };
    const int64_t slots = resolve_slots(o);
    if (o.expert_cache_auto || o.expert_cache > 0) std::printf("expert_slots: %lld\n", (long long)slots);
    std::printf("pcie_num: %lld/256\n", (long long)baloon::generate::pcie_num(o.pcie));
    if (o.serve) {
        const int64_t n = serve_loop(std::cin, std::cout, o, make);
        std::fprintf(stderr, "served %lld requests\n", (long long)n);
        return 0;
    }
    const std::vector<std::vector<int64_t>> prompts{o.prompt};
    const auto res = drive_pool_multi(make, prompts, o, o.workers);
    const DriveResult& r = res[0];
    if (!r.ok) { std::fprintf(stderr, "error: %s\n", r.error.c_str()); return 1; }
    std::printf("tokens: %s\n", join_i64(r.tokens).c_str());
    std::printf("rows: %zu\n", r.row_argmax.size());
    std::printf("positions: %lld\n", (long long)r.positions);
    if (o.spec >= 2) std::printf("spec_rounds: %lld\n", (long long)r.rounds);
    if (o.trace) std::printf("trace_layer(48): %lld\n", (long long)baloon::generate::trace_layer(48));
    return 0;
}
