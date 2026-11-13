// generate.cpp - the `ref generate` driver: decode loop, expert-cache auto sizing, adaptive swap,
// sized-slot keep-prefix, prompt-path lending and the generate main entry point.
// Unit u6bd5f913be: AdaptiveExpertResidency.lean formalises autoSlots, keepPrefix, lend and swap.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "baloon/generate_cli.hpp"

// ---- type aliases defined in this translation unit ----

// Clock: steady-clock alias used for driver timing
using Clock = std::chrono::steady_clock;

// Drive: backend function bundle (step, prefill, draft, verify)
using Drive = baloon::gencli::Drive;

// Options: command-line options for the generate driver
using Options = baloon::gencli::Options;

// ---- mem_mark: RSS watermark helper (unique to this file) ----
static size_t mem_mark() {
    size_t rss = 0;
#if defined(__linux__)
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            if (std::strncmp(line, "VmRSS:", 6) == 0) {
                std::sscanf(line + 6, "%zu", &rss);
                rss *= 1024;
                break;
            }
        }
        std::fclose(f);
    }
#endif
    return rss;
}

// ---- wrappers defined here to satisfy the port-table definitions ----

// argmax: index of the maximum element in a logits row; -1 when empty
static int64_t argmax(const std::vector<float>& v) {
    return baloon::gencli::argmax(v);
}

// parse_i64_list: "1,2,-3" → {1,2,-3}; clears and returns false on error
static bool parse_i64_list(const std::string& s, std::vector<int64_t>& out) {
    return baloon::gencli::parse_i64_list(s, out);
}

// usage: generate driver help text
static std::string usage(const char* prog) {
    return baloon::gencli::usage(prog);
}

// model_shards: resolve a shard-set path to an ordered file list
static bool model_shards(const std::string& path, std::vector<std::string>& out,
                          std::string& err) {
    return baloon::gencli::model_shards(path, out, err);
}

// drive_pool: run one prompt through the decode / speculative loop
static baloon::gencli::DriveResult drive_pool(const Drive& d,
                                               const std::vector<int64_t>& prompt,
                                               const Options& o, bool serve_admit = false) {
    return baloon::gencli::drive_pool(d, prompt, o, serve_admit);
}

// drive_pool_multi: run every prompt over `workers` threads; each builds its own backend
static std::vector<baloon::gencli::DriveResult> drive_pool_multi(
    const std::function<Drive(int)>& make,
    const std::vector<std::vector<int64_t>>& prompts,
    const Options& o, int64_t workers, bool serve_admit = false) {
    return baloon::gencli::drive_pool_multi(make, prompts, o, workers, serve_admit);
}

// ---- main ----
int main(int argc, char** argv) {
    Options o;
    std::string err;
    if (!baloon::gencli::parse_args(argc, argv, o, err)) {
        std::fprintf(stderr, "error: %s\n%s", err.c_str(),
                     usage(argc > 0 ? argv[0] : "generate").c_str());
        return 2;
    }
    if (o.help) {
        std::fputs(usage(argc > 0 ? argv[0] : "generate").c_str(), stdout);
        return 0;
    }
    err = baloon::gencli::validate(o);
    if (!err.empty()) {
        std::fprintf(stderr, "error: %s\n%s", err.c_str(),
                     usage(argc > 0 ? argv[0] : "generate").c_str());
        return 2;
    }

    const size_t m0 = mem_mark();
    const int64_t vocab = o.vocab;
    auto make = [vocab](int) { return baloon::gencli::synthetic_drive(vocab); };

    const int64_t slots = baloon::gencli::resolve_slots(o);
    if (o.expert_cache_auto || o.expert_cache > 0)
        std::printf("expert_slots: %lld\n", (long long)slots);
    std::printf("pcie_num: %lld/256\n", (long long)baloon::generate::pcie_num(o.pcie));

    if (o.serve) {
        const int64_t n = baloon::gencli::serve_loop(std::cin, std::cout, o, make);
        std::fprintf(stderr, "served %lld requests\n", (long long)n);
        return 0;
    }

    const std::vector<std::vector<int64_t>> prompts{o.prompt};
    // Use :: prefix to suppress ADL (Drive/Options alias baloon::gencli types,
    // which would otherwise pull generate_cli.hpp's drive_pool_multi into the overload set).
    const auto res = ::drive_pool_multi(make, prompts, o, o.workers);
    const baloon::gencli::DriveResult& r = res[0];
    if (!r.ok) { std::fprintf(stderr, "error: %s\n", r.error.c_str()); return 1; }
    std::printf("tokens: %s\n", baloon::gencli::join_i64(r.tokens).c_str());
    std::printf("rows: %zu\n", r.row_argmax.size());
    std::printf("positions: %lld\n", (long long)r.positions);
    if (o.spec >= 2) std::printf("spec_rounds: %lld\n", (long long)r.rounds);
    if (o.trace)
        std::printf("trace_layer(48): %lld\n", (long long)baloon::generate::trace_layer(48));

    const size_t m1 = mem_mark();
    (void)m0; (void)m1;
    return 0;
}
