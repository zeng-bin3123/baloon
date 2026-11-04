// SessionGraphs + MtpDrafter + Verifier invariants.
#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include "baloon/core_mtp.hpp"
#include "baloon/core_session.hpp"
#include "baloon/core_verify.hpp"
#include "check.hpp"

using namespace baloon::core;

void test_session() {
    // alignment / sizing: carve fits inside the reported size
    testh::Rng rng(9);
    for (int it = 0; it < 3000; ++it) {
        std::vector<uint64_t> parts(1 + rng.below(7));
        for (auto& p : parts) p = rng.below(100000);
        const uint64_t bytes = session_bytes(parts);
        CHECK(bytes % 256 == 0);
        uint64_t sum = 0, used = 0;
        for (uint64_t p : parts) { sum += p; used = align_up(used + p); }
        CHECK(used <= bytes && bytes == align_up(sum) + 4096);
        Session s;
        CHECK(session_init(s, parts, 48) && s.bytes == bytes && s.arena.size() == bytes);
        for (size_t i = 0; i < parts.size(); ++i) { CHECK(s.off[i] % 256 == 0 && s.off[i] + s.size[i] <= bytes); if (i) CHECK(s.off[i - 1] + s.size[i - 1] <= s.off[i]); }
    }
    for (uint64_t n = 0; n < 3000; ++n) { const uint64_t a = align_up(n); CHECK(a >= n && a <= n + 255 && a % 256 == 0); }
    Session bad;
    CHECK(!session_init(bad, {}, 48) && !session_init(bad, std::vector<uint64_t>(8, 1), 48) && !session_init(bad, {1}, 0));
    // state indices: 36 GDN / 12 QSA, consecutive, injective
    int gmax = -1, qmax = -1;
    for (int l = 0; l < 48; ++l) {
        const int g = gdn_point_at(l), q = qsa_point_at(l);
        if (l % 4 == 3) { CHECK(g == -1 && q >= 0 && q < 12); qmax = std::max(qmax, q); }
        else { CHECK(q == -1 && g >= 0 && g < 36); gmax = std::max(gmax, g); }
        for (int m = l + 1; m < 48; ++m) { if (l % 4 != 3 && m % 4 != 3) CHECK(gdn_point_at(l) != gdn_point_at(m)); if (l % 4 == 3 && m % 4 == 3) CHECK(qsa_point_at(l) != qsa_point_at(m)); }
        if (l < 47 && l % 4 != 3 && (l + 1) % 4 != 3) CHECK(gdn_point_at(l + 1) == gdn_point_at(l) + 1);
    }
    CHECK(gmax == 35 && qmax == 11 && gdn_point_at(-1) == -1 && qsa_point_at(-1) == -1);
    CHECK(gdn_state_floats() == 817152);
    CHECK(session_dump_floats(48, 4, 100) == 49 * 4 * 100);
    for (uint64_t l = 0; l < 48; ++l) CHECK((l + 1) * 7 + 7 <= session_dump_floats(48, 1, 7));
    CHECK(ple_hist_bytes(10, 7) % 16 == 0 && ple_hist_bytes(10, 7) >= 280);
    // stage diffs telescope
    for (int it = 0; it < 500; ++it) {
        std::vector<int64_t> p; int64_t t = 0;
        for (int i = 0; i < 5; ++i) { t += (int64_t)rng.below(1000); p.push_back(t); }
        auto d = stage_diffs(p);
        int64_t sum = 0; for (auto x : d) { sum += x; CHECK(x >= 0); }
        CHECK(d.size() == 5 && sum == p.back() && d[0] == p[0] && d[1] == p[1] - p[0]);
    }
    // graphs: eager == graph order; prefix graphs run layers with stage_prefix k; clock is injectable
    Session s;
    s.n_layers = 6;
    std::vector<std::string> log;
    static int64_t clk = 0;
    s.clock_ns = []() { return clk; };
    LayerFn layer = [&](int l, int k) { log.push_back("L" + std::to_string(l) + "/" + std::to_string(k)); clk += (k == 0 ? 100 : 10 * k); };
    CHECK(!session_run_token(s, 1) && !stage_token(s, 1, 1));
    CHECK(session_capture(s, layer, 6));
    CHECK(session_capture_token(s, [&](int64_t t) { log.push_back("E" + std::to_string(t)); }, layer, [&] { log.push_back("H"); }));
    log.clear();
    CHECK(session_token(s, 42, false));
    auto eager = log; log.clear();
    CHECK(session_token(s, 42, true));
    CHECK(log == eager && eager.size() == 8 && eager.front() == "E42" && eager.back() == "H" && s.token == 42);
    log.clear();
    CHECK(stage_token(s, 7, 3) && log.size() == 6 && log[0] == "L0/3" && log[5] == "L5/3");
    CHECK(!stage_token(s, 7, 0) && !stage_token(s, 7, 6));
    std::vector<int64_t> pre, st;
    CHECK(session_replay_stage_prefixes(s, pre) && pre.size() == 5);
    for (int k = 0; k < 5; ++k) CHECK(pre[(size_t)k] == 6 * 10 * (k + 1));       // fake clock: layer cost 10*k
    CHECK(session_replay_stages(s, st) && st.size() == 5);
    int64_t sum = 0; for (auto x : st) sum += x;
    CHECK(sum == pre.back() && st[0] == 60 && st[1] == 60);
    CHECK(session_replay_stage_sweep(s, 3, st) && st.size() == 5 && st[4] == 60 && !session_replay_stage_sweep(s, 0, st));
    std::vector<std::vector<int64_t>> pl;
    CHECK(session_replay_stages_per_layer(s, pl) && pl.size() == 6 && pl[0].size() == 5);
    CHECK(session_replay(s, 3) == 3 * 600 && session_replay(s, 0) == -1);
    CHECK(session_replay_full(s) == 6 * 50);
    session_zero(s);
    session_graphs_free(s);
    CHECK(!s.token_graph.captured && !s.prefix_graph[0].captured && !session_run_token(s, 1) && session_replay_full(s) == -1);
    CHECK(!session_token(s, 1, false));
    // scratch for the parts staging
    SessionLoopScratch sc;
    CHECK(!sc.init(0, 8) && !sc.init(2, 0));
    CHECK(sc.init(3, 8) && sc.bytes() == 96 && sc.bytes() > 0 && sc.parts.size() == 24);
    sc.free(); CHECK(sc.bytes() == 0);
    // spin boundary fires every 1024th spin
    for (uint64_t x = 0; x < 5000; ++x) { CHECK(spin_boundary(x + 1024) == spin_boundary(x)); CHECK(spin_boundary(x) == (x % 1024 == 1023)); }
    // session_loop: layer l consumes after ring value l+1; a producer thread raises the ring
    Doorbell d; doorbell_init(&d);
    std::vector<int> seen; std::atomic<uint64_t> published{0};
    std::thread prod([&] { for (uint64_t v = 1; v <= 48; ++v) { std::this_thread::yield(); published = v; doorbell_raise(&d, v); } });
    bool ok = session_loop(d, 48, [&](int l) { CHECK(doorbell_value(&d) >= (uint64_t)l + 1); seen.push_back(l); });
    prod.join();
    CHECK(ok && seen.size() == 48);
    for (int i = 0; i < 48; ++i) CHECK(seen[(size_t)i] == i);
    Doorbell d2; doorbell_init(&d2); doorbell_raise(&d2, 2);
    int cnt = 0;
    CHECK(!session_loop(d2, 5, [&](int) { ++cnt; }, 5000) && cnt == 2);        // gives up when the ring never arrives
    uint64_t esc = 0;
    CHECK(!session_loop(d2, 5, [&](int) {}, UINT64_MAX, [&] { ++esc; return esc < 3; }) && esc == 3);
    Doorbell none;
    CHECK(!session_loop(none, 3, nullptr));
    doorbell_free(&d); doorbell_free(&d2);
}

void test_mtp() {
    testh::Rng rng(13);
    MtpDrafter m;
    CHECK(!m.bind({0, 0, 4, 0.5}) && !m.bind({100, 0, 1, 0.5}));
    for (int it = 0; it < 2000; ++it) {
        const uint64_t cells = 1 + rng.below(5000), win = rng.below(6000), mt = 2 + rng.below(8);
        CHECK(m.bind({cells, win, mt, 0.5}));
        const uint64_t ew = (win > 0 && win < cells) ? win : 0, len = ew ? ew : cells;
        CHECK(m.window() == ew && m.window() <= cells);
        CHECK(m.cap() % 64 == 0 && m.cap() >= len && m.cap() < len + 64);
        CHECK(m.step_rows() == 2 * mt);
    }
    // step-row staging: T catch-up rows, chain rows above them, full row last, all distinct inside 2*max_t
    for (uint64_t mt = 3; mt <= 9; ++mt) for (uint64_t T = 1; T <= mt; ++T) {
        CHECK(m.bind({1000, 0, mt, 0.5}) && m.capture_round(T) && m.round_T() == T);
        const auto& k = m.staged_kind();
        CHECK(k.size() == 2 * mt);
        for (uint64_t r = 0; r < T; ++r) CHECK(k[r] == 1 || k[r] == 2);
        for (uint64_t j = 1; j + 2 <= mt; ++j) { const uint64_t row = mt + j - 1; CHECK(row >= mt && row < 2 * mt - 1 && k[row] == 2); }
        CHECK(k[2 * mt - 1] == 3);
    }
    CHECK(!m.capture_round(0) && !m.capture_round(10));
    // draft chain: matches the fuel-loop model exactly; never beyond max_t-1; zero filled; stops on a low probability
    for (int it = 0; it < 5000; ++it) {
        const uint64_t mt = 2 + rng.below(8), T = 1 + rng.below(mt), a = rng.below(T);
        const double minp = (double)rng.below(10) / 10.0;
        CHECK(m.bind({300, 0, mt, minp}));
        std::vector<double> probs(mt + 2);
        for (auto& p : probs) p = (double)rng.below(11) / 10.0;
        uint64_t j = 1;                                               // model: j drafts so far (the first is always produced)
        while (j < mt - 1 && probs[j - 1] >= minp) ++j;
        std::vector<int32_t> out;
        const int64_t n = m.draft(T, a, [&](uint64_t jj, int32_t prev) { return DraftStep{(int32_t)(100 + jj + (prev ? 0 : 0)), probs[jj]}; }, out);
        CHECK(n == (int64_t)j && n >= 1 && (uint64_t)n <= mt - 1);
        CHECK(out.size() == mt - 1);
        for (uint64_t i = 0; i < out.size(); ++i) CHECK(out[i] == (i < (uint64_t)n ? (int32_t)(100 + i) : 0));
        if (probs[0] < minp) CHECK(n == 1);
    }
    std::vector<int32_t> o;
    CHECK(m.bind({300, 0, 4, 0.5}));
    auto fn = [](uint64_t, int32_t) { return DraftStep{1, 1.0}; };
    CHECK(m.draft(0, 0, fn, o) == -1 && m.draft(5, 0, fn, o) == -1 && m.draft(2, 2, fn, o) == -1 && m.draft(3, 2, fn, o) >= 1);
    CHECK(m.draft(2, 0, nullptr, o) == -1);
    // chunked prefill: chunks partition the prompt, skipped chunks lie wholly below the first needed cell
    for (int it = 0; it < 4000; ++it) {
        const uint64_t mt = 2 + rng.below(8), n = rng.below(60);
        const int64_t cell0 = (int64_t)rng.below(30), first = (int64_t)rng.below(120) - 10;
        CHECK(m.bind({300, 0, mt, 0.5}));
        std::vector<std::pair<uint64_t, uint64_t>> ran;
        const uint64_t cnt = m.prefill(n, cell0, first, [&](uint64_t c, uint64_t T) { ran.emplace_back(c, T); });
        CHECK(cnt == ran.size());
        uint64_t pos = 0, k = 0;
        for (uint64_t c = 0; c < n; ) {
            const uint64_t T = std::min<uint64_t>(mt, n - c);
            CHECK(T >= 1 && T <= mt && c + T <= n);
            const bool skip = cell0 + (int64_t)c + (int64_t)T <= first;
            if (skip) { for (uint64_t t = 0; t < T; ++t) CHECK(cell0 + (int64_t)c + (int64_t)t < first); }
            else { CHECK(k < ran.size() && ran[k].first == c && ran[k].second == T); ++k; }
            c += T; pos = c;
        }
        CHECK(pos == n && k == ran.size());
    }
    CHECK(m.bind({300, 0, 4, 0.5}));
    CHECK(m.capture_prefill() == 4 && m.prefill_graphs() == 4);
    CHECK(m.capture_step(1) && m.capture_step(2) && !m.capture_step(3) && !m.capture_step(0));
    CHECK(!m.record_forward(8) && m.record_forward(7));
    // tensor lookup by name AND kind
    CHECK(m.load_text("# mtp\nw.a f32 0 16\nw.a bf16 16 8\nw.b q8 24 34\nw.c bf16 64 2\n"));
    CHECK(m.f32("w.a") && m.f32("w.a")->off == 0 && m.bf16("w.a") && m.bf16("w.a")->off == 16);
    CHECK(m.q8("w.b") && m.q8("w.b")->bytes == 34);
    CHECK(m.f32("w.b") == nullptr && m.q8("w.a") == nullptr && m.bf16("w.b") == nullptr && m.f32("zzz") == nullptr);   // kind mismatch -> null
    for (const char* nm : {"w.a", "w.b", "w.c"}) {
        if (auto* t = m.f32(nm)) CHECK(t->name == nm && t->kind == "f32");
        if (auto* t = m.bf16(nm)) CHECK(t->name == nm && t->kind == "bf16");
        if (auto* t = m.q8(nm)) CHECK(t->name == nm && t->kind == "q8");
    }
    {
        NativeHead h = m.native_head("w.a", 4);                       // bf16 w.a is 8 bytes = 1 row of 4
        CHECK(h.present() && h.vocab == 1 && h.n_embd == 4 && h.w == m.bf16("w.a"));
        CHECK(!m.native_head("w.a", 3).present() && !m.native_head("w.b", 4).present() && !m.native_head("w.a", 0).present());
    }
    CHECK(!m.load_text("w.a f64 0 4\n") && !m.load_text("w.a f32 0\n"));
    CHECK(m.load_text("") && m.f32("w.a") == nullptr);
    std::string path = "/tmp/baloon_u18_mtp_" + std::to_string(getpid()) + ".txt";
    FILE* f = std::fopen(path.c_str(), "wb"); if (f) { std::fputs("x f32 1 2\n", f); std::fclose(f); }
    CHECK(m.load(path) && m.f32("x") && !m.load(path + ".none"));
    std::remove(path.c_str());
}

void test_verify() {
    Verifier v;
    CHECK(!v.init({0, 4, 1, 4, 4, 8}) && !v.init({9, 4, 1, 4, 4, 8}) && !v.init({4, 4, 3, 4, 4, 8}) && !v.init({4, 0, 1, 4, 4, 8}));
    for (uint64_t G = 1; G <= 2; ++G) {
        CHECK(v.init({8, 5, G, 16, 3, 4}));
        int32_t win[8] = {10, 11, 12, 13, 14, 15, 16, 17};
        CHECK(!v.record_window(win, 0) && !v.record_window(win, 9));
        for (uint64_t T = 1; T <= 8; ++T) {
            CHECK(v.record_window(win, T) && v.capture());
            CHECK(v.schedule().size() == 5 * G);
            std::vector<uint64_t> rings; std::vector<std::vector<int>> cover(5, std::vector<int>(T, 0));
            uint64_t last_flag = 0;
            const uint64_t n = v.run([&](const Verifier::Step& s) {
                rings.push_back(s.ring);
                CHECK(s.ring == s.layer * G + s.group + 1);
                CHECK(s.row_begin <= s.row_end && s.row_end <= T);
                for (uint64_t r = s.row_begin; r < s.row_end; ++r) ++cover[s.layer][r];
                CHECK(v.flag() >= last_flag); last_flag = v.flag();
            });
            CHECK(n == 5 * G && v.flag() == 5 * G);
            for (size_t i = 0; i < rings.size(); ++i) CHECK(rings[i] == i + 1);          // strictly increasing ring values
            for (auto& row : cover) for (int c : row) CHECK(c == 1);                      // groups tile the T rows exactly
        }
    }
    CHECK(v.init({8, 3, 2, 4, 2, 4}));
    int32_t win[8] = {5, 6, 7, 8, 9, 10, 11, 12};
    CHECK(v.record_window(win, 5));
    // accepted prefix: window[a+1] == greedy[a]
    CHECK(v.accept_len({6, 7, 8, 9}) == 4 && v.accept_len({6, 7, 0, 9}) == 2 && v.accept_len({0, 7}) == 0 && v.accept_len({}) == 0);
    for (int it = 0; it < 200; ++it) {
        std::vector<int32_t> g = {6, 7, 8, 9};
        const size_t cut = (size_t)it % 5;
        if (cut < 4) g[cut] = -1;
        CHECK(v.accept_len(g) == cut);
    }
    std::vector<int32_t> ent;
    CHECK(!v.commit(0, 100, ent));                                 // commit graph not armed yet
    CHECK(v.capture_commit());
    for (uint64_t a = 0; a < 5; ++a) {
        CHECK(v.commit(a, 100, ent) && ent.size() == 2 + 8 && ent[0] == (int32_t)(a + 1) && ent[1] == (int32_t)a);
        for (uint64_t t = 0; t < 8; ++t) CHECK(ent[2 + t] == (t < a + 1 ? (int32_t)(100 + t) : -1));
    }
    CHECK(!v.commit(5, 100, ent));                                 // a + 1 must not exceed T
    // monotone progress flag under contention
    v.raise_flag(0);
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t) th.emplace_back([&, t] { for (uint32_t i = 1; i <= 2000; ++i) v.raise_flag(i * 4 + (uint32_t)t); });
    for (auto& x : th) x.join();
    CHECK(v.flag() == 2000 * 4 + 3);
    v.raise_flag(7); CHECK(v.flag() == 2000 * 4 + 3);
    // plan slots: two slots, publish raises the ring flag; oversized plans are rejected
    CHECK(v.record_window(win, 4));
    std::vector<int32_t> plan(10, 7);
    CHECK(v.publish_plan(2, 1, plan) && v.plan_slot_words(1)[0] == 7 && v.plan_slot_words(0)[0] == 0 && v.flag() == 6);
    CHECK(!v.publish_plan(3, 0, plan) && !v.publish_plan(0, 2, plan));
    CHECK(!v.publish_plan(0, 0, std::vector<int32_t>(100000, 1)));
    v.set_plan_slot(1);
    // fetch budget: below both the staging capacity (2) and 64
    uint8_t blob[16] = {1};
    CHECK(v.fetch_dma(blob, 16) && v.fetch_dma(blob, 16) && !v.fetch_dma(blob, 16) && v.fetches() == 2 && v.staged_bytes() == 32);
    CHECK(!v.fetch_dma(nullptr, 1));
    // residual rows: row a of the T window
    CHECK(v.final_R(0) != nullptr && v.final_R(3) != nullptr && v.final_R(4) == nullptr && v.final_R(3) == v.final_R(0) + 3 * 4);
}
