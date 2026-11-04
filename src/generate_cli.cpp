// generate_cli.cpp - the generate driver: option parsing and validation, the teacher-forced decode loop
// (embed -> layers -> head -> sample is behind the Engine interface), the speculative verify loop with the suffix
// drafter, a multi-request pool, and a command-line entry point backed by a deterministic toy engine.
// Define BALOON_GENERATE_NO_MAIN to include this file from a test without the entry point.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "baloon/generate_driver.hpp"
#include "baloon/suffix_drafter.hpp"

namespace baloon {
namespace gen {

using Clock = std::chrono::steady_clock;

// A run fits iff max_new > 0, max_context > 0, max_new <= max_context and the prompt fits in max_context - max_new.
inline bool run_fits(int64_t n_prompt, int64_t max_new, int64_t max_ctx) {
    return max_new > 0 && max_ctx > 0 && max_new <= max_ctx && n_prompt >= 1 && n_prompt <= max_ctx - max_new;
}
// Prompt positions that go through the batched path (--prefill-until); the last prompt position stays on the token path.
inline int64_t n_batched(int64_t pu, int64_t n) { return (pu > 0 && pu < n - 1) ? pu : n - 1; }

// Index of the largest logit (lowest index on ties); NaN never wins.
int64_t argmax(const float* x, int64_t n) {
    int64_t best = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (x[i] != x[i]) continue;
        if (best < 0 || x[i] > x[best]) best = i;
    }
    return best < 0 ? 0 : best;
}

// Comma separated signed integers, e.g. "1, 2,3". Rejects empty items, junk and overflow.
bool parse_i64_list(const std::string& s, std::vector<int64_t>* out) {
    out->clear();
    size_t i = 0;
    while (true) {
        while (i < s.size() && s[i] == ' ') ++i;
        const size_t st = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        const size_t ds = i;
        __int128 v = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            v = v * 10 + (s[i] - '0');
            if (v > ((__int128)1 << 63)) return false;
            ++i;
        }
        if (i == ds) return false;
        if (s[st] == '-') v = -v;
        if (v > INT64_MAX || v < INT64_MIN) return false;
        out->push_back((int64_t)v);
        while (i < s.size() && s[i] == ' ') ++i;
        if (i == s.size()) return true;
        if (s[i] != ',') return false;
        ++i;
    }
}

// What the driver needs from a model: logits of one position, and the argmax of every position of a verify window.
struct Engine {
    virtual ~Engine() = default;
    virtual int64_t vocab() const = 0;
    virtual void step(int64_t tok, int64_t pos, std::vector<float>& logits) = 0;
    virtual void verify(const std::vector<int64_t>& window, int64_t pos0, std::vector<int64_t>& outs) {
        std::vector<float> l;
        outs.clear();
        for (size_t j = 0; j < window.size(); ++j) {
            step(window[j], pos0 + (int64_t)j, l);
            outs.push_back(argmax(l.data(), (int64_t)l.size()));
        }
    }
};
// Stateless toy model: the successor of token t is (3t + 1) mod V.
struct ToyEngine : Engine {
    int64_t v;
    explicit ToyEngine(int64_t vocab) : v(vocab) {}
    int64_t vocab() const override { return v; }
    void step(int64_t tok, int64_t, std::vector<float>& l) override {
        l.assign((size_t)v, 0.f);
        l[(size_t)(((tok % v + v) % v * 3 + 1) % v)] = 1.f;
    }
};

struct Options {
    std::vector<int64_t> prompt{1};
    int64_t max_new = 16, max_context = 4096, prefill_until = 0, spec = 1, vocab = 1000, threads = 1, requests = 1;
    bool dump = false, help = false;
    // Returns false and fills *err on an invalid command line.
    bool parse(int argc, char** argv, std::string* err) {
        auto bad = [&](const std::string& m) { *err = m; return false; };
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto need = [&](int64_t* dst) -> bool {
                if (i + 1 >= argc) return false;
                std::vector<int64_t> v;
                if (!parse_i64_list(argv[++i], &v) || v.size() != 1) return false;
                *dst = v[0];
                return true;
            };
            if (a == "--help" || a == "-h") help = true;
            else if (a == "--dump") dump = true;
            else if (a == "--prompt") { if (i + 1 >= argc || !parse_i64_list(argv[++i], &prompt)) return bad("bad --prompt"); }
            else if (a == "--max-new") { if (!need(&max_new)) return bad("bad --max-new"); }
            else if (a == "--max-context") { if (!need(&max_context)) return bad("bad --max-context"); }
            else if (a == "--prefill-until") { if (!need(&prefill_until)) return bad("bad --prefill-until"); }
            else if (a == "--spec") { if (!need(&spec)) return bad("bad --spec"); }
            else if (a == "--vocab") { if (!need(&vocab)) return bad("bad --vocab"); }
            else if (a == "--threads") { if (!need(&threads)) return bad("bad --threads"); }
            else if (a == "--requests") { if (!need(&requests)) return bad("bad --requests"); }
            else return bad("unknown option " + a);
        }
        if (help) return true;
        if (vocab < 2) return bad("--vocab must be >= 2");
        if (spec < 1 || spec > 16) return bad("--spec must be in 1..16");
        if (threads < 1 || requests < 1) return bad("--threads/--requests must be >= 1");
        for (int64_t t : prompt) if (t < 0 || t >= vocab) return bad("prompt token outside the vocabulary");
        if (!run_fits((int64_t)prompt.size(), max_new, max_context)) return bad("prompt + max-new does not fit in max-context");
        return true;
    }
};

// One generation request and its result counters.
struct Drive {
    Engine* eng = nullptr;
    std::vector<int64_t> prompt;
    int64_t max_new = 0, max_ctx = 0, prefill_until = 0, spec = 1;
    std::vector<int64_t> out;
    int64_t rows = 0;                       // logits rows produced: n_prompt - 1 + max_new on the token path
    int64_t rounds = 0, drafted = 0, accepted = 0;
    double ms = 0;
    std::string error;
};

// Token path: teacher-forced through the prompt, then the sampled token is fed back.
static bool run_plain(Drive& d) {
    const int64_t np = (int64_t)d.prompt.size();
    std::vector<float> l;
    int64_t pos = 0;
    const int64_t nb = n_batched(d.prefill_until, np);
    for (; pos < nb; ++pos) { d.eng->step(d.prompt[(size_t)pos], pos, l); ++d.rows; }   // batched prefill: nothing sampled
    int64_t in = d.prompt[(size_t)pos];
    while ((int64_t)d.out.size() < d.max_new) {
        d.eng->step(in, pos, l);
        ++d.rows;
        int64_t next = 0;
        if (pos >= np - 1) { next = argmax(l.data(), (int64_t)l.size()); d.out.push_back(next); }
        in = pos + 1 < np ? d.prompt[(size_t)(pos + 1)] : next;   // teacher forcing inside the prompt
        ++pos;
    }
    return true;
}

// Speculative path: window = last emitted token + up to spec-1 suffix drafts; commit accepted + 1 tokens per round.
static bool run_spec(Drive& d) {
    const int64_t np = (int64_t)d.prompt.size();
    std::vector<float> l;
    const int64_t nb = n_batched(d.prefill_until, np);
    int64_t pos = 0;
    for (; pos < nb; ++pos) { d.eng->step(d.prompt[(size_t)pos], pos, l); ++d.rows; }
    for (; pos < np; ++pos) {
        d.eng->step(d.prompt[(size_t)pos], pos, l); ++d.rows;
        if (pos == np - 1) d.out.push_back(argmax(l.data(), (int64_t)l.size()));
    }
    suffix::SuffixDrafter dr;
    for (int64_t t : d.prompt) dr.append((uint32_t)t);
    if ((int64_t)d.out.size() < d.max_new) dr.append((uint32_t)d.out.back()); else return true;
    std::vector<int64_t> outs;
    while ((int64_t)d.out.size() < d.max_new) {
        const int64_t p0 = np + (int64_t)d.out.size() - 1;               // position of the last emitted token
        const int64_t room = std::min<int64_t>(d.spec, d.max_ctx - p0);  // never past the context
        std::vector<int64_t> window{d.out.back()};
        if (room > 1) for (uint32_t t : dr.propose((uint64_t)(room - 1))) window.push_back((int64_t)t);
        d.eng->verify(window, p0, outs);
        int64_t a = 0;
        const int64_t k = (int64_t)window.size() - 1;
        while (a < k && window[(size_t)(a + 1)] == outs[(size_t)a]) ++a;   // longest accepted prefix of the drafts
        ++d.rounds; d.drafted += k; d.accepted += a;
        for (int64_t j = 0; j <= a && (int64_t)d.out.size() < d.max_new; ++j) {
            d.out.push_back(outs[(size_t)j]);
            dr.append((uint32_t)outs[(size_t)j]);
        }
    }
    return true;
}

// Run one request (token path when spec == 1, speculative otherwise).
bool drive_pool(Drive& d) {
    const auto t0 = Clock::now();
    d.out.clear(); d.rows = d.rounds = d.drafted = d.accepted = 0;
    if (!d.eng || !run_fits((int64_t)d.prompt.size(), d.max_new, d.max_ctx)) { d.error = "request does not fit"; return false; }
    const bool ok = d.spec > 1 ? run_spec(d) : run_plain(d);
    d.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return ok;
}

// Run many requests on up to `threads` workers; every request is run exactly once. Returns the number that succeeded.
int64_t drive_pool_multi(std::vector<Drive>& ds, int64_t threads) {
    std::atomic<size_t> next{0};
    std::atomic<int64_t> okc{0};
    auto work = [&] {
        for (size_t i; (i = next.fetch_add(1)) < ds.size();) if (drive_pool(ds[i])) ++okc;
    };
    std::vector<std::thread> th;
    for (int64_t t = 1; t < std::max<int64_t>(1, std::min<int64_t>(threads, (int64_t)ds.size())); ++t) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
    return okc.load();
}

void usage(const char* prog) {
    std::printf("usage: %s [--prompt t0,t1,..] [--max-new N] [--max-context N] [--prefill-until N] [--spec S]\n"
                "          [--vocab V] [--threads K] [--requests R] [--dump]\n", prog);
}

}  // namespace gen
}  // namespace baloon

#ifndef BALOON_GENERATE_NO_MAIN
int main(int argc, char** argv) {
    using namespace baloon::gen;
    Options o;
    std::string err;
    if (!o.parse(argc, argv, &err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); usage(argv[0]); return 2; }
    if (o.help) { usage(argv[0]); return 0; }
    ToyEngine eng(o.vocab);
    std::vector<Drive> ds((size_t)o.requests);
    for (auto& d : ds) { d.eng = &eng; d.prompt = o.prompt; d.max_new = o.max_new; d.max_ctx = o.max_context; d.prefill_until = o.prefill_until; d.spec = o.spec; }
    const int64_t ok = drive_pool_multi(ds, o.threads);
    if (ok != o.requests) { std::fprintf(stderr, "error: %lld request(s) failed\n", (long long)(o.requests - ok)); return 1; }
    for (const auto& d : ds) {
        std::printf("tokens:");
        for (int64_t t : d.out) std::printf(" %lld", (long long)t);
        std::printf("\n");
        if (o.dump) std::printf("rows=%lld rounds=%lld drafted=%lld accepted=%lld ms=%.3f\n", (long long)d.rows, (long long)d.rounds, (long long)d.drafted, (long long)d.accepted, d.ms);
    }
    return 0;
}
#endif
