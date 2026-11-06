#include <limits>
#include <sstream>
#include "check.hpp"
#include "baloon/generate_cli.hpp"
using namespace baloon::gencli;
namespace gen = baloon::generate;

static std::vector<int64_t> rand_prompt(testh::Rng& rng, int64_t n, int64_t vocab) {
    std::vector<int64_t> p;
    for (int64_t i = 0; i < n; ++i) p.push_back(rng.below((uint32_t)vocab));
    return p;
}
// independent oracle of the first-order synthetic model: tokens[k] = f^(k+1)(last prompt token)
static std::vector<int64_t> oracle(const std::vector<int64_t>& prompt, int64_t m, int64_t V) {
    std::vector<int64_t> o;
    int64_t t = prompt.back();
    for (int64_t k = 0; k < m; ++k) { t = (t % V * 31 + 7) % V; o.push_back(t); }
    return o;
}
static Options opts(int64_t m, int64_t c, int64_t pu = 0, int64_t S = 0) {
    Options o; o.max_new = m; o.max_context = c; o.prefill_until = pu; o.spec = S; return o;
}

void test_parse_list() {
    testh::Rng rng(1);
    std::vector<int64_t> v;
    CHECK(parse_i64_list("", v) && v.empty());
    CHECK(parse_i64_list("7", v) && v.size() == 1 && v[0] == 7);
    CHECK(parse_i64_list("1,-2,30", v) && v.size() == 3 && v[0] == 1 && v[1] == -2 && v[2] == 30);
    for (int it = 0; it < 200; ++it) {
        std::vector<int64_t> src;
        const int n = 1 + (int)rng.below(20);
        for (int i = 0; i < n; ++i) src.push_back((int64_t)rng.u32() - (int64_t)(1u << 30));
        CHECK(parse_i64_list(join_i64(src), v) && v == src);  // round trip
    }
    for (const char* bad : {",", "1,", ",1", "1,,2", "a", "1,a", " 1", "1 ", "+1", "1.5", "--1", "-", "0x10"}) {
        v = {9};
        CHECK_MSG(!parse_i64_list(bad, v) && v.empty(), "%s", bad);
    }
    CHECK(parse_i64_list("9223372036854775807", v) && v[0] == std::numeric_limits<int64_t>::max());
    CHECK(parse_i64_list("-9223372036854775808", v) && v[0] == std::numeric_limits<int64_t>::min());
    CHECK(!parse_i64_list("9223372036854775808", v));
    CHECK(!parse_i64_list("1,99999999999999999999", v));
}

void test_parse_args() {
    auto run = [](std::vector<const char*> a, Options& o, std::string& e) { a.insert(a.begin(), "prog"); return parse_args((int)a.size(), a.data(), o, e); };
    Options o; std::string e;
    CHECK(run({"--prompt-ids", "1,2,3", "--max-new", "5", "--max-context", "100", "--prefill-until", "1", "--spec", "4",
               "--expert-cache", "auto", "--pcie", "0.25", "--serve", "--trace", "--workers", "3", "--gpus", "0,1"}, o, e));
    CHECK(o.prompt == (std::vector<int64_t>{1, 2, 3}) && o.max_new == 5 && o.max_context == 100 && o.prefill_until == 1);
    CHECK(o.spec == 4 && o.expert_cache_auto && o.pcie == 0.25 && o.serve && o.trace && o.workers == 3 && o.gpus.size() == 2);
    o = Options();
    CHECK(run({"--expert-cache", "12"}, o, e) && !o.expert_cache_auto && o.expert_cache == 12);
    o = Options(); CHECK(run({"-h"}, o, e) && o.help);
    o = Options(); CHECK(!run({"--bogus"}, o, e) && !e.empty());
    o = Options(); CHECK(!run({"--max-new"}, o, e) && !e.empty());
    o = Options(); CHECK(!run({"--max-new", "x"}, o, e));
    o = Options(); CHECK(!run({"--prompt-ids", "1,,2"}, o, e));
    o = Options(); CHECK(!run({"--expert-cache", "lots"}, o, e));
    o = Options(); CHECK(!run({"--pcie", "0.5x"}, o, e));
    CHECK(usage("prog").find("--max-new") != std::string::npos && usage("prog").find("prog") != std::string::npos);
}

// validate() must agree with the fit predicate: 0<m, 0<c, m<=c, n <= c-m  (=> n+m <= c)
void test_validate_fit() {
    for (int64_t n = 1; n <= 12; ++n) for (int64_t m = -2; m <= 12; ++m) for (int64_t c = -2; c <= 14; ++c) {
        Options o = opts(m, c);
        o.prompt.assign((size_t)n, 1);
        const bool fits = 0 < m && 0 < c && m <= c && n <= c - m;
        CHECK_MSG((validate(o) == "") == fits, "n=%lld m=%lld c=%lld", (long long)n, (long long)m, (long long)c);
        if (fits) CHECK(n + m <= c);  // runFits_total
        Drive d = synthetic_drive(11);
        const DriveResult r = drive_pool(d, o.prompt, o);
        CHECK(r.ok == fits);
        // serve admission: n + m + 8 <= c
        const DriveResult s = drive_pool(d, o.prompt, o, true);
        CHECK(s.ok == (fits && n + m + 8 <= c));
    }
    Options o = opts(4, 100); CHECK(validate(o) != "");  // empty prompt, not serve
    o.serve = true; CHECK(validate(o) == "");
    o = opts(4, 100); o.prompt = {1}; o.spec = 1; CHECK(validate(o) != "");
    o.spec = 2; CHECK(validate(o) == ""); o.spec = -1; CHECK(validate(o) != "");
    o.spec = 0; o.prefill_until = -1; CHECK(validate(o) != "");
    o.prefill_until = 0; o.workers = 0; CHECK(validate(o) != "");
}

void test_argmax() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(argmax(std::vector<float>{}) == -1);
    CHECK(argmax(nullptr, 3) == -1);
    CHECK(argmax(std::vector<float>{5}) == 0);
    CHECK(argmax(std::vector<float>{1, 3, 2}) == 1);
    CHECK(argmax(std::vector<float>{3, 3, 3}) == 0);       // ties: lowest index
    CHECK(argmax(std::vector<float>{1, 3, 3}) == 1);
    CHECK(argmax(std::vector<float>{nan, 1, 0}) == 1);     // NaN never wins
    CHECK(argmax(std::vector<float>{0, nan, 1, nan}) == 2);
    CHECK(argmax(std::vector<float>{nan, nan}) == 0);
    CHECK(argmax(std::vector<float>{-5, -1, -3}) == 1);
    testh::Rng rng(2);
    for (int it = 0; it < 200; ++it) {
        std::vector<float> x((size_t)(1 + rng.below(50)));
        for (auto& v : x) v = (float)(int)rng.below(10);
        int64_t ref = 0;
        for (size_t i = 1; i < x.size(); ++i) if (x[i] > x[(size_t)ref]) ref = (int64_t)i;
        CHECK(argmax(x) == ref);
    }
}

// A position-dependent model, so teacher forcing and position order are observable.
struct Rec { int64_t tok, pos; };
static Drive recording(int64_t V, std::vector<Rec>* calls) {
    Drive d;
    d.step = [V, calls](int64_t tok, int64_t pos, std::vector<float>& lg) {
        calls->push_back({tok, pos});
        lg.assign((size_t)V, 0.f);
        lg[(size_t)((tok * 3 + pos * 5 + 1) % V)] = 1.f;
    };
    return d;
}

void test_teacher_forcing() {
    const int64_t V = 53;
    testh::Rng rng(3);
    for (int64_t np = 1; np <= 9; ++np) for (int64_t m = 1; m <= 7; ++m) {
        auto prompt = rand_prompt(rng, np, V);
        std::vector<Rec> calls;
        Options o = opts(m, 64);
        const DriveResult r = drive_pool(recording(V, &calls), prompt, o);
        CHECK(r.ok);
        // one row per position 0 .. np-2+m, each processed once, in order
        CHECK((int64_t)calls.size() == np - 1 + m);
        CHECK((int64_t)r.row_argmax.size() == np - 1 + m);  // dumpPositions
        CHECK(r.positions == np - 1 + m);
        CHECK((int64_t)r.tokens.size() == m);               // produced_at_end
        // independent simulation
        int64_t cur = prompt[0];
        std::vector<int64_t> exp_tokens;
        bool ok = true;
        for (int64_t pos = 0; pos < np - 1 + m; ++pos) {
            if ((int64_t)calls.size() <= pos || calls[(size_t)pos].pos != pos || calls[(size_t)pos].tok != cur) { ok = false; break; }
            const int64_t nx = (cur * 3 + pos * 5 + 1) % V;
            if (pos >= np - 1) exp_tokens.push_back(nx);                // nothing produced before the last prompt position
            cur = pos + 1 < np ? prompt[(size_t)pos + 1] : nx;          // nextInput: prompt token inside, sampled outside
        }
        CHECK_MSG(ok, "np=%lld m=%lld", (long long)np, (long long)m);
        CHECK(r.tokens == exp_tokens);
        // prompt positions are fed the prompt token itself
        for (int64_t pos = 0; pos < np && pos < (int64_t)calls.size(); ++pos) CHECK(calls[(size_t)pos].tok == prompt[(size_t)pos]);
    }
}

void test_prefill_split() {
    const int64_t V = 61;
    testh::Rng rng(4);
    for (int64_t np = 1; np <= 10; ++np) for (int64_t pu = 0; pu <= 12; ++pu) {
        auto prompt = rand_prompt(rng, np, V);
        const int64_t m = 4;
        // with prefill: positions below nb never reach step(); rows and tokens are unchanged
        Drive with = synthetic_drive(V);
        std::vector<int64_t> stepped;
        auto base_step = with.step;
        with.step = [&](int64_t t, int64_t p, std::vector<float>& l) { stepped.push_back(p); base_step(t, p, l); };
        Options o = opts(m, 64, pu);
        const DriveResult a = drive_pool(with, prompt, o);
        Drive without = synthetic_drive(V); without.prefill = nullptr;
        const DriveResult b = drive_pool(without, prompt, o);
        const int64_t nb = (0 < pu && pu < np - 1) ? pu : np - 1;
        CHECK(a.n_batched == nb && nb <= np - 1 && nb < np);   // nBatched_le, posStart_lt
        CHECK(a.ok && b.ok && a.tokens == b.tokens && a.row_argmax == b.row_argmax);
        CHECK(a.tokens == oracle(prompt, m, V));
        CHECK((int64_t)a.row_argmax.size() == np - 1 + m);
        CHECK((int64_t)stepped.size() == np - 1 + m - nb);
        for (size_t i = 0; i < stepped.size(); ++i) CHECK(stepped[i] == nb + (int64_t)i);  // token path starts at nb
        if (!stepped.empty()) CHECK(stepped.front() <= np - 1);                              // last prompt position stays on the token path
    }
    // a wrong row count from prefill is an error, not a silent mismatch
    Drive bad = synthetic_drive(V);
    bad.prefill = [](const std::vector<int64_t>&, int64_t n, std::vector<std::vector<float>>& rows) { rows.assign((size_t)n + 1, std::vector<float>(3, 0.f)); };
    Options o = opts(2, 32);
    CHECK(!drive_pool(bad, {1, 2, 3, 4}, o).ok);
}

// With a first-order model, the speculative loop must emit exactly the teacher-forced tokens, whatever the draft quality.
void test_spec_equivalence() {
    testh::Rng rng(5);
    for (int64_t V : {7, 64, 257}) for (int64_t np = 1; np <= 8; ++np) for (int64_t m = 1; m <= 25; m += 2) for (int64_t S = 0; S <= 9; ++S) {
        if (S == 1) continue;
        for (int64_t we : {0, 1, 2, 5}) {
            auto prompt = rand_prompt(rng, np, V);
            Drive d = synthetic_drive(V, we);
            Options o = opts(m, 128, (int64_t)rng.below(4), S);
            const DriveResult r = drive_pool(d, prompt, o);
            CHECK_MSG(r.ok, "V=%lld np=%lld m=%lld S=%lld we=%lld", (long long)V, (long long)np, (long long)m, (long long)S, (long long)we);
            CHECK_MSG(r.tokens == oracle(prompt, m, V), "V=%lld np=%lld m=%lld S=%lld we=%lld", (long long)V, (long long)np, (long long)m, (long long)S, (long long)we);
            CHECK((int64_t)r.tokens.size() == m);  // never overshoots max_new
            if (S >= 2) CHECK(r.rounds >= 1 && (int64_t)r.accepted_hist.size() == r.rounds);
            else CHECK(r.rounds == 0);
        }
    }
}

void test_spec_rounds() {
    // accept_prefix: longest equal prefix
    CHECK(accept_prefix({}, {}) == 0);
    CHECK(accept_prefix({1, 2, 3}, {1, 2, 3}) == 3);   // accept_self
    CHECK(accept_prefix({1, 2, 3}, {1, 9, 3}) == 1);   // stops at the first mismatch
    CHECK(accept_prefix({9, 2, 3}, {1, 2, 3}) == 0);
    CHECK(accept_prefix({1, 2, 3}, {1, 2}) == 2);      // bounded by both lengths
    CHECK(accept_prefix({1}, {1, 2, 3}) == 1);
    testh::Rng rng(6);
    for (int it = 0; it < 200; ++it) {
        std::vector<int64_t> x; for (int i = 0, n = (int)rng.below(12); i < n; ++i) x.push_back(rng.below(3));
        CHECK(accept_prefix(x, x) == (int64_t)x.size());
        std::vector<int64_t> y; for (int i = 0, n = (int)rng.below(12); i < n; ++i) y.push_back(rng.below(3));
        size_t ref = 0; while (ref < x.size() && ref < y.size() && x[ref] == y[ref]) ++ref;
        CHECK(accept_prefix(x, y) == (int64_t)ref);
        CHECK(accept_prefix(x, y) <= (int64_t)x.size() && accept_prefix(x, y) <= (int64_t)y.size());  // accept_le_left/right
    }
    // round accounting: every round commits accepted+1 tokens within the window; positions advance by the commits
    const int64_t V = 97, np = 5, m = 40, S = 6;
    Drive d = synthetic_drive(V, 7);
    std::vector<int64_t> window_sizes, starts;
    auto vb = d.verify;
    d.verify = [&](const std::vector<int64_t>& w, int64_t pos) { window_sizes.push_back((int64_t)w.size()); starts.push_back(pos); return vb(w, pos); };
    auto prompt = rand_prompt(rng, np, V);
    const DriveResult r = drive_pool(d, prompt, opts(m, 256, 0, S));
    CHECK(r.ok && r.tokens == oracle(prompt, m, V));
    CHECK(window_sizes.size() == r.accepted_hist.size());
    int64_t committed = 0, pos = np - 1;
    bool partial = false, full = false, zero = false;
    for (size_t i = 0; i < r.accepted_hist.size(); ++i) {
        const int64_t a = r.accepted_hist[i], T = window_sizes[i];
        CHECK(T >= 1 && T <= S);                       // windowLen_le / windowLen_pos
        CHECK(a >= 0 && a <= T - 1 && a + 1 <= T);     // commit_le_window
        CHECK(starts[i] == pos);                       // next window starts after the committed cells
        pos += a + 1; committed += a + 1;
        zero |= a == 0; full |= (T > 1 && a == T - 1); partial |= (a > 0 && a < T - 1);
        CHECK(committed <= m);
    }
    CHECK(committed == m && r.positions == np - 1 + m);
    CHECK(zero && full && partial);  // the draft cadence exercises all acceptance regimes
    // producedStep: min(m, produced + a + 1) <= m and strictly increases below m
    for (int64_t M = 1; M < 10; ++M) for (int64_t p = 0; p < 12; ++p) for (int64_t a = 0; a < 6; ++a) {
        const int64_t q = std::min(M, p + a + 1);
        CHECK(q <= M);
        if (p < M) CHECK(p < q);
    }
    // a verifier that returns the wrong number of outputs is an error
    Drive bad = synthetic_drive(V);
    bad.verify = [](const std::vector<int64_t>& w, int64_t) { return std::vector<int64_t>(w.size() + 1, 0); };
    CHECK(!drive_pool(bad, {1, 2}, opts(6, 64, 0, 4)).ok);
    // spec requested but no draft/verify provided: falls back to the token path with identical output
    Drive plain = synthetic_drive(V); plain.draft = nullptr;
    const DriveResult f = drive_pool(plain, {3, 4, 5}, opts(7, 64, 0, 4));
    CHECK(f.ok && f.rounds == 0 && f.tokens == oracle({3, 4, 5}, 7, V));
}

void test_pool_multi() {
    const int64_t V = 101;
    testh::Rng rng(7);
    std::vector<std::vector<int64_t>> prompts;
    for (int i = 0; i < 40; ++i) prompts.push_back(rand_prompt(rng, 1 + (int64_t)rng.below(10), V));
    Options o = opts(9, 64, 0, 3);
    prompts.push_back(std::vector<int64_t>(70, 1));  // does not fit
    prompts.push_back({});                           // empty
    auto make = [](int) { return synthetic_drive(V); };
    const auto ref = drive_pool_multi(make, prompts, o, 1);
    for (int64_t w : {1, 2, 3, 8, 64}) {
        const auto got = drive_pool_multi(make, prompts, o, w);
        CHECK(got.size() == prompts.size());
        for (size_t i = 0; i < prompts.size(); ++i) {
            CHECK(got[i].ok == ref[i].ok);
            CHECK(got[i].tokens == ref[i].tokens);   // request order preserved, no cross-talk
            if (i < 40) CHECK(got[i].ok && got[i].tokens == oracle(prompts[i], 9, V));
        }
        CHECK(!got[40].ok && !got[41].ok);
    }
    CHECK(drive_pool_multi(make, {}, o, 4).empty());
    // a backend that cannot be built fails its requests cleanly
    auto thrower = [](int) -> Drive { throw std::runtime_error("no device"); };
    const auto bad = drive_pool_multi(thrower, {{1, 2}, {3}}, o, 2);
    CHECK(bad.size() == 2 && !bad[0].ok && !bad[1].ok && bad[0].error == "no device");
    // a backend that throws mid-run fails only that request
    auto flaky = [](int) { Drive d = synthetic_drive(V); auto s = d.step; d.step = [s](int64_t t, int64_t p, std::vector<float>& l) { if (t == 13) throw std::runtime_error("boom"); s(t, p, l); }; d.prefill = nullptr; return d; };
    const auto fl = drive_pool_multi(flaky, {{13, 2}, {1, 2}}, opts(3, 64), 2);
    CHECK(!fl[0].ok && fl[0].error == "boom" && fl[1].ok);
}

void test_serve() {
    const int64_t V = 257;
    Options o = opts(4, 64); o.serve = true;
    auto make = [](int) { return synthetic_drive(V); };
    std::istringstream in("GEN 3 1,2,3\n\nGEN 5 10\nGEN 0 1\nGEN 3 \nGEN 3 1,x\nGEN 60 1,2,3\nGEN 56 1\nFOO 1 2\nGEN 2 -4\nQUIT\nGEN 1 1\n");
    std::ostringstream out;
    const int64_t n = serve_loop(in, out, o, make);
    CHECK(n == 9);  // blank skipped, QUIT stops
    std::istringstream res(out.str());
    std::vector<std::string> lines; std::string l; while (std::getline(res, l)) lines.push_back(l);
    CHECK(lines.size() == 9);
    if (lines.size() == 9) {
        auto toks = [&](const std::vector<int64_t>& p, int64_t m) { return "OK " + std::to_string(m) + " " + join_i64(oracle(p, m, V)); };
        CHECK(lines[0] == toks({1, 2, 3}, 3));
        CHECK(lines[1] == toks({10}, 5));
        CHECK(lines[2].rfind("ERR", 0) == 0);   // max_new 0
        CHECK(lines[3].rfind("ERR", 0) == 0);   // empty prompt
        CHECK(lines[4].rfind("ERR", 0) == 0);   // bad id
        CHECK(lines[5].rfind("ERR", 0) == 0);   // 3 + 60 + 8 > 64
        // boundary: n + m + 8 <= ctx  ->  1 + 56 + 8 = 65 > 64 rejected
        CHECK(lines[6].rfind("ERR", 0) == 0);
        CHECK(lines[7].rfind("ERR", 0) == 0);   // unknown command
        CHECK(lines[8].rfind("ERR", 0) == 0);   // negative id
    }
    std::istringstream in2("GEN 55 1\n");  // 1 + 55 + 8 = 64 <= 64 admitted
    std::ostringstream out2;
    serve_loop(in2, out2, o, make);
    CHECK(out2.str().rfind("OK 55 ", 0) == 0);
}

void test_slots_pcie() {
    testh::Rng rng(8);
    for (int it = 0; it < 500; ++it) {
        Options o; o.expert_cache_auto = true;
        o.vram_free = (int64_t)rng.below(1u << 30); o.vram_reserve = (int64_t)rng.below(1u << 30); o.blob = 1 + (int64_t)rng.below(1u << 20);
        const int64_t s = resolve_slots(o);
        const int64_t avail = o.vram_free > o.vram_reserve ? o.vram_free - o.vram_reserve : 0;
        CHECK(s == avail / o.blob);                       // autoSlots
        CHECK(s * o.blob <= avail);                       // autoSlots_fit: never exceeds free VRAM
        CHECK(s * o.blob <= o.vram_free);
        CHECK((s + 1) * o.blob > avail);                  // and is maximal
        o.profile_len = (int64_t)rng.below(2000);
        const int64_t t = resolve_slots(o);
        CHECK(t == std::min(s, o.profile_len) && t <= s);  // withProfile_le
    }
    Options o; o.expert_cache_auto = true;
    CHECK(resolve_slots(o) == -1);            // missing inputs
    o.vram_free = 100; o.blob = 0; CHECK(resolve_slots(o) == -1);
    o.blob = 10; o.vram_reserve = 500; CHECK(resolve_slots(o) == 0);  // reserve larger than free
    o = Options(); o.expert_cache = 7; CHECK(resolve_slots(o) == 7);
    // PCIe numerator is clamped into 0..256
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK(gen::pcie_num(nan) == 0 && gen::pcie_num(-1.0) == 0 && gen::pcie_num(0.0) == 0);
    CHECK(gen::pcie_num(1.0) == 256 && gen::pcie_num(7.5) == 256 && gen::pcie_num(0.5) == 128);
    CHECK(gen::pcie_num(std::numeric_limits<double>::infinity()) == 256 && gen::pcie_num(-std::numeric_limits<double>::infinity()) == 0);
    // routing trace index: counter l+1 after dispatch maps back to l in 0..47
    for (uint64_t l = 0; l < 48; ++l) { const int64_t t = gen::trace_layer(l + 1); CHECK(t == (int64_t)l && t >= 0 && t < 48); }
    for (int64_t x = -3; x < 300; ++x) { const int64_t c = gen::clamp_pcie(x); CHECK(c >= 0 && c <= 256 && (x < 0 || x > 256 || c == x)); }
}
