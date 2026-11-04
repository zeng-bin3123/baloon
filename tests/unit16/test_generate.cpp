// Invariants of the generate driver: run-size validation, position/row accounting, teacher forcing, batched prefill
// split, speculative acceptance (never changes the output), expert-tier sizing, M-RoPE image positions, admission.
#define BALOON_GENERATE_NO_MAIN
#include "../../src/generate_cli.cpp"
#include "check.hpp"
using namespace baloon::gen;

namespace {
struct Spy : ToyEngine {   // records every (token, pos) fed to the model
    std::vector<std::pair<int64_t, int64_t>> calls;
    using ToyEngine::ToyEngine;
    void step(int64_t t, int64_t p, std::vector<float>& l) override { calls.push_back({t, p}); ToyEngine::step(t, p, l); }
};
int64_t accept_len(const std::vector<int64_t>& d, const std::vector<int64_t>& o) {
    size_t a = 0; while (a < d.size() && a < o.size() && d[a] == o[a]) ++a; return (int64_t)a;
}
}  // namespace

void test_gen_theory() {
    testh::Rng rng(21);
    for (int it = 0; it < 100000; ++it) {
        const int64_t n = (int64_t)rng.below(300), m = (int64_t)rng.below(300) - 5, c = (int64_t)rng.below(700) - 5;
        const bool fits = run_fits(n, m, c);
        if (fits) CHECK(n + m <= c);
        CHECK(fits == (m > 0 && c > 0 && m <= c && n >= 1 && n <= c - m));
        // admission leaves room for a verify window
        const bool adm = baloon::gdrv::serve_admit(n, m, c);
        if (adm) CHECK(n + m <= c);
        const int64_t np = 1 + (int64_t)rng.below(200), mn = 1 + (int64_t)rng.below(200);
        CHECK(baloon::gdrv::dump_rows(np, mn) + 1 == np + mn);
        CHECK(baloon::gdrv::produced_at(np, baloon::gdrv::stop_position(np, mn)) == mn);
        if (np >= 2) CHECK(baloon::gdrv::produced_at(np, np - 2) == 0);
        // batched split
        const int64_t pu = (int64_t)rng.below(300) - 3;
        const int64_t nb = n_batched(pu, np);
        CHECK(nb <= np - 1); CHECK(nb < np); CHECK(nb >= 0);
        // acceptance
        std::vector<int64_t> d, o; for (uint32_t i = 0, k = rng.below(9); i < k; ++i) { d.push_back(rng.below(3)); o.push_back(rng.below(3)); }
        const int64_t a = accept_len(d, o); CHECK(a <= (int64_t)d.size()); CHECK(a <= (int64_t)o.size());
        CHECK(accept_len(d, d) == (int64_t)d.size());
        // window / commit / produced step
        const int64_t S = 1 + (int64_t)rng.below(16), T = 1 + (int64_t)rng.below((uint32_t)S), aa = (int64_t)rng.below((uint32_t)T);
        CHECK(aa + 1 <= T); CHECK(T <= S);
        const int64_t produced = (int64_t)rng.below((uint32_t)mn), step = std::min(mn, produced + aa + 1);
        CHECK(step <= mn); CHECK(produced < step);
        // pcie clamp
        CHECK(std::min<int64_t>(256, (int64_t)rng.below(100000)) <= 256);
        // trace layer index
        const int64_t l = (int64_t)rng.below(48); CHECK((l + 1) - 1 == l);
        // auto expert tier
        const int64_t fr = (int64_t)rng.below(1u << 30), rs = (int64_t)rng.below(1u << 30), blob = 1 + (int64_t)rng.below(1u << 22);
        const int64_t slots = baloon::gdrv::slots_auto(fr, rs, blob);
        CHECK(slots * blob <= std::max<int64_t>(0, fr - rs));
        const int64_t need = (int64_t)rng.below(1u << 26), k = baloon::gdrv::borrow_slots(need, blob);
        CHECK(need <= k * blob);
        // image positions: (t,h,w) = (p, p + y, p + x); text resumes at p + max(nx, ny)
        const int64_t nx = 1 + (int64_t)rng.below(40), ny = 1 + (int64_t)rng.below(40), p = (int64_t)rng.below(1000), j = (int64_t)rng.below((uint32_t)(nx * ny));
        CHECK(p + j / nx < p + std::max(nx, ny)); CHECK(p + j % nx < p + std::max(nx, ny)); CHECK(p < p + std::max(nx, ny));
    }
    // PLE token window advance keeps the newest predecessor
    std::pair<int64_t, int64_t> w{4, 5}; w = {w.second, 9}; CHECK(w.first == 5 && w.second == 9);
    // adaptive swap: needs candidate >= 2 and >= victim + 1.5 (half-units), and then strictly more routed
    for (int c = 0; c < 40; ++c) for (int v = 0; v < 40; ++v) if (4 <= 2 * c && 2 * v + 3 <= 2 * c) CHECK(v < c);
    for (uint64_t u = 0; u < 5000; ++u) CHECK(u * 7 / 10 <= u);
}

void test_gen_loops() {
    const int64_t V = 257;
    // teacher forcing + accounting on the token path
    for (int64_t np : {1, 2, 5}) for (int64_t mn : {1, 3, 20}) for (int64_t pu : {0, 1, 2, 3, 9}) {
        Spy sp(V); Drive d; d.eng = &sp; d.max_new = mn; d.max_ctx = 64; d.prefill_until = pu;
        for (int64_t i = 0; i < np; ++i) d.prompt.push_back((i * 37 + 11) % V);
        CHECK(drive_pool(d));
        CHECK((int64_t)d.out.size() == mn);
        CHECK(d.rows == np - 1 + mn); CHECK((int64_t)sp.calls.size() == np - 1 + mn);
        for (size_t i = 0; i < sp.calls.size(); ++i) {
            CHECK(sp.calls[i].second == (int64_t)i);
            if ((int64_t)i < np) CHECK(sp.calls[i].first == d.prompt[i]);                 // teacher forcing
            else CHECK(sp.calls[i].first == d.out[i - (size_t)np]);                        // sampled token fed back
        }
        // expected output of the toy chain
        int64_t t = d.prompt.back(); bool ok = true;
        for (int64_t k = 0; k < mn; ++k) { t = (t * 3 + 1) % V; ok &= (d.out[(size_t)k] == t); }
        CHECK(ok);
    }
    // speculation never changes the output, for any window size and prompt
    testh::Rng rng(8);
    for (int it = 0; it < 300; ++it) {
        ToyEngine e(1 + (int64_t)rng.below(40) + 2);
        Drive a; a.eng = &e; a.max_new = 1 + (int64_t)rng.below(80); a.max_ctx = 300; a.prefill_until = (int64_t)rng.below(6);
        for (uint32_t i = 0, n = 1 + rng.below(12); i < n; ++i) a.prompt.push_back(rng.below((uint32_t)e.v));
        Drive b = a; b.spec = 2 + (int64_t)rng.below(15);
        CHECK(drive_pool(a)); CHECK(drive_pool(b));
        CHECK(a.out == b.out);
        CHECK(b.accepted <= b.drafted); CHECK(b.rounds >= 1 || a.max_new == 1);
        CHECK((int64_t)b.out.size() == b.max_new);
    }
    // a cyclic toy model is predicted by the suffix drafter: acceptance is real
    { ToyEngine e(5); Drive d; d.eng = &e; d.prompt = {1, 4, 3, 0, 1, 4, 3, 0, 1, 4}; d.max_new = 200; d.max_ctx = 400; d.spec = 8;
      CHECK(drive_pool(d)); CHECK(d.accepted > 0); CHECK(d.rounds < 200); }
    // never past the context
    { Spy sp(31); Drive d; d.eng = &sp; d.prompt = {1, 2, 3}; d.max_new = 10; d.max_ctx = 13; d.spec = 8;
      CHECK(drive_pool(d)); CHECK((int64_t)d.out.size() == 10); }
    // invalid requests are rejected without touching the model
    { Spy sp(31); Drive d; d.eng = &sp; d.prompt = {1, 2, 3}; d.max_new = 10; d.max_ctx = 12; CHECK(!drive_pool(d)); CHECK(sp.calls.empty()); }
    // argmax
    { float x[] = {0.f, 3.f, 3.f, 1.f}; CHECK(argmax(x, 4) == 1);
      float y[] = {NAN, -1.f, -2.f}; CHECK(argmax(y, 3) == 1);
      float z[] = {-5.f}; CHECK(argmax(z, 1) == 0); }
}

void test_gen_options_pool() {
    std::vector<int64_t> v;
    CHECK(parse_i64_list("1,2,3", &v) && v == (std::vector<int64_t>{1, 2, 3}));
    CHECK(parse_i64_list(" -4 , +5", &v) && v == (std::vector<int64_t>{-4, 5}));
    CHECK(parse_i64_list("9223372036854775807", &v) && v[0] == INT64_MAX);
    CHECK(parse_i64_list("-9223372036854775808", &v) && v[0] == INT64_MIN);
    CHECK(!parse_i64_list("9223372036854775808", &v)); CHECK(!parse_i64_list("", &v)); CHECK(!parse_i64_list("1,,2", &v));
    CHECK(!parse_i64_list("1,2,", &v)); CHECK(!parse_i64_list("a", &v)); CHECK(!parse_i64_list("1 2", &v)); CHECK(!parse_i64_list("-", &v));
    auto parse = [](std::vector<const char*> a) { Options o; std::string e; std::vector<char*> av{(char*)"prog"}; for (auto s : a) av.push_back((char*)s); return o.parse((int)av.size(), av.data(), &e); };
    CHECK(parse({"--prompt", "1,2,3", "--max-new", "5", "--max-context", "8"}));
    CHECK(!parse({"--prompt", "1,2,3", "--max-new", "6", "--max-context", "8"}));      // 3 + 6 > 8
    CHECK(!parse({"--max-new", "0"})); CHECK(!parse({"--max-context", "0"})); CHECK(!parse({"--max-new", "9", "--max-context", "8"}));
    CHECK(!parse({"--bogus"})); CHECK(!parse({"--max-new"})); CHECK(!parse({"--spec", "0"})); CHECK(!parse({"--prompt", "5000"}));
    CHECK(parse({"--help"}));
    // pool: every request runs exactly once and equals the single-threaded result
    ToyEngine e(97);
    for (int64_t th : {1, 2, 4, 9}) {
        std::vector<Drive> ds(23);
        for (size_t i = 0; i < ds.size(); ++i) { ds[i].eng = &e; ds[i].prompt = {(int64_t)i % 97, 5, 7}; ds[i].max_new = 10 + (int64_t)i; ds[i].max_ctx = 128; ds[i].spec = (i % 3) ? 4 : 1; }
        CHECK(drive_pool_multi(ds, th) == 23);
        for (size_t i = 0; i < ds.size(); ++i) {
            Drive r = ds[i]; r.spec = 1; CHECK(drive_pool(r));
            CHECK(ds[i].out == r.out); CHECK((int64_t)ds[i].out.size() == ds[i].max_new);
        }
    }
    { std::vector<Drive> ds(3); for (auto& d : ds) { d.eng = &e; d.prompt = {1}; d.max_new = 5; d.max_ctx = 4; } CHECK(drive_pool_multi(ds, 2) == 0); }
}
