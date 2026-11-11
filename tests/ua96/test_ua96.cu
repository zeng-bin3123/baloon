// test_ua96.cu - invariant tests for unit ua9684a588f (SessionGraphReplay, WeightPackLoader, MemoryPlanner,
// PleDirectReader, PleNgram, SpecDraftController, FusedGatedResidualRead, FusedGdnStepNorm, GatedResidualReference,
// GdnReferenceKernels). Every check calls the repository's real implementation and compares against an independent
// statement of the LAYER 1 law (host double-precision references, literal numbers from the theory, exhaustive scans).
// Any failure makes the binary exit non-zero.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "baloon/core_session.hpp"
#include "baloon/core_weights.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/kg_kernels.hpp"
#include "baloon/plan.hpp"
#include "baloon/ple_ngram_kernels.hpp"
#include "baloon/ple_reader.hpp"
#include "baloon/spec_controller.hpp"
#include "baloon/ua96_geometry.hpp"

using namespace baloon;

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)
static std::mt19937_64 rng(20260709);
static uint64_t rn(uint64_t n) { return rng() % n; }
static float rf(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

// ================================================================================== SessionGraphReplay
static void test_session() {
    // layer numbering: l % 4 == 3 is QSA; GDN index = l - l/4 (closed form), QSA index = l/4
    int n_gdn = 0, n_qsa = 0, prev_gdn = -1;
    for (int l = 0; l < 48; ++l) {
        const int g = core::gdn_point_at(l), q = core::qsa_point_at(l);
        if (l % 4 == 3) {
            CHECK(g == -1 && q == l / 4); ++n_qsa;
        } else {
            CHECK(q == -1 && g == l - l / 4 && g == prev_gdn + 1);   // strictly increasing, distinct slices
            prev_gdn = g; ++n_gdn;
        }
    }
    CHECK(n_gdn == 36 && n_qsa == 12);
    CHECK(core::gdn_point_at(-1) == -1 && core::qsa_point_at(-1) == -1);

    // slices [i*F, i*F+F) of distinct GDN layers never overlap and stay inside 36*F
    const uint64_t F = core::gdn_state_floats();
    CHECK(F == 817152ull);
    for (int a = 0; a < 48; ++a) for (int b = a + 1; b < 48; ++b) {
        const int ga = core::gdn_point_at(a), gb = core::gdn_point_at(b);
        if (ga < 0 || gb < 0) continue;
        CHECK((uint64_t)ga * F + F <= (uint64_t)gb * F);
    }
    CHECK((uint64_t)core::gdn_point_at(46) * F + F <= 36 * F);

    // arena sizing: align256(sum) + 4096; covers the raw sum; carve fits and stays 256-aligned
    for (int it = 0; it < 300; ++it) {
        std::vector<uint64_t> parts(1 + rn(core::kSessionRegions));
        uint64_t sum = 0;
        for (auto& p : parts) { p = rn(100000); sum += p; }
        const uint64_t bytes = core::session_bytes(parts);
        CHECK(bytes >= sum + 4096 && (bytes - 4096) % 256 == 0 && bytes - 4096 - sum < 256);
        core::Session s;
        CHECK(core::session_init(s, parts, 48));
        CHECK(s.bytes == bytes && s.arena.size() == bytes);
        for (size_t i = 0; i < parts.size(); ++i) {
            CHECK(s.off[i] % 256 == 0 && s.size[i] == parts[i] && s.off[i] + s.size[i] <= bytes);
            if (i + 1 < parts.size()) CHECK(s.off[i] + s.size[i] <= s.off[i + 1]);   // regions disjoint
        }
    }
    { core::Session s; CHECK(!core::session_init(s, {}, 48)); CHECK(!core::session_init(s, {1, 2}, 0)); }

    // stage prefixes: prefix k runs layer(l, k) for every layer in order; invalid prefixes are rejected
    core::Session s;
    CHECK(core::session_init(s, {1024, 2048}, 48));
    std::vector<std::pair<int, int>> calls;
    CHECK(core::session_capture(s, [&](int l, int k) { calls.push_back({l, k}); }, 48));
    CHECK(!core::stage_token(s, 7, 0) && !core::stage_token(s, 7, 6));
    for (int k = 1; k <= 5; ++k) {
        calls.clear();
        CHECK(core::stage_token(s, 11, k));
        CHECK(calls.size() == 48);
        bool in_order = true;
        for (int l = 0; l < (int)calls.size(); ++l) in_order = in_order && calls[l].first == l && calls[l].second == k;
        CHECK(in_order);
        CHECK(s.token == 11);
    }

    // timing: stage times are the differences of cumulative prefix times and telescope to the full time
    int64_t clock = 0;
    s.clock_ns = [&]() { return clock; };
    const int64_t cum[5] = {3, 7, 12, 20, 31};   // per-layer cumulative cost of prefix 1..5
    core::Session t;
    CHECK(core::session_init(t, {64}, 48));
    t.clock_ns = [&]() { return clock; };
    CHECK(core::session_capture(t, [&](int, int k) { clock += cum[k - 1]; }, 48));
    std::vector<int64_t> pre, st;
    CHECK(core::session_replay_stage_prefixes(t, pre) && pre.size() == 5);
    for (int k = 0; k < 5; ++k) CHECK(pre[k] == 48 * cum[k]);
    CHECK(core::session_replay_stages(t, st) && st.size() == 5);
    int64_t acc = 0;
    for (int k = 0; k < 5; ++k) { CHECK(st[k] == 48 * (cum[k] - (k ? cum[k - 1] : 0))); acc += st[k]; }
    CHECK(acc == pre[4]);
    std::vector<std::vector<int64_t>> per;
    CHECK(core::session_replay_stages_per_layer(t, per) && per.size() == 48);
    for (auto& v : per) { CHECK(v.size() == 5); int64_t a = 0; for (auto x : v) a += x; CHECK(a == cum[4]); }
    CHECK(core::stage_diffs({5, 9, 9, 30, 31}) == (std::vector<int64_t>{5, 4, 0, 21, 1}));

    // doorbell ring protocol: layer l waits for ring value l+1 (ring_strict)
    CHECK(core::session_dump_floats(48, 4, 2560) == 49ull * 4 * 2560);
    CHECK(core::spin_boundary(1023) && !core::spin_boundary(1022) && core::spin_boundary(2047));
}

// ================================================================================== WeightPackLoader
static std::string idx_line(const std::string& n, int kind, uint64_t so, uint64_t sb, uint64_t db, uint64_t c, uint64_t s, uint64_t o, int fp16) {
    return n + " " + std::to_string(kind) + " " + std::to_string(so) + " " + std::to_string(sb) + " " + std::to_string(db) + " " +
           std::to_string(c) + " " + std::to_string(s) + " " + std::to_string(o) + " " + std::to_string(fp16) + "\n";
}
static uint64_t align256(uint64_t n) { return (n + 255) / 256 * 256; }

static void test_weights() {
    using namespace core;
    CHECK(kWeightChunk == 8388608ull);
    // chunk_len_of = min(CHUNK, remaining): progress, termination, evenness
    for (uint64_t rem : {1ull, 2ull, 4ull, 8388607ull, 8388608ull, 8388609ull, 25000000ull}) {
        uint64_t left = rem; int rounds = 0;
        while (left) { const uint64_t c = chunk_len_of(left); CHECK(c > 0 && c <= kWeightChunk && c <= left); left -= c; ++rounds; }
        CHECK(rounds == (int)((rem + kWeightChunk - 1) / kWeightChunk));
    }
    CHECK(weight_align256(0) == 0 && weight_align256(1) == 256 && weight_align256(256) == 256 && weight_align256(257) == 512);

    // segment accounting: dst = src + scales/2 with fp16 scales, src = dst without
    for (int it = 0; it < 200; ++it) {
        const uint64_t c = rn(500), sc = 2 * rn(300), o = 4 * rn(40);
        for (int fp16 = 0; fp16 < 2; ++fp16) {
            WIndexRow r; r.name = "w"; r.kind = 2; r.src_off = 0; r.codes = c; r.scales = sc; r.offset = o; r.scales_fp16 = fp16;
            r.src_bytes = c + (fp16 ? sc / 2 : sc) + o; r.dst_bytes = c + sc + o;
            std::vector<WSeg> segs;
            CHECK(plan_segments(r, segs));
            uint64_t ss = 0, ds = 0; for (auto& g : segs) { ss += g.src_bytes; ds += g.dst_bytes; }
            CHECK(ss == r.src_bytes && ds == r.dst_bytes);
            if (fp16) CHECK(r.dst_bytes == r.src_bytes + sc / 2); else CHECK(r.dst_bytes == r.src_bytes);
            WIndexRow bad = r; bad.src_bytes += 1;
            CHECK(!plan_segments(bad, segs));       // totals disagree -> refused
        }
    }

    // exact fp16 -> f32 widening, chunk independence, contiguous destination offsets
    {
        const int n = 301;
        std::vector<uint16_t> half(n); for (auto& h : half) h = (uint16_t)rn(0x7c00);   // finite, any sign-less pattern
        for (int i = 0; i < n; i += 7) half[i] |= 0x8000;
        std::vector<uint8_t> src((size_t)n * 2); std::memcpy(src.data(), half.data(), src.size());
        WReader rd = [&](uint64_t off, uint64_t len, uint8_t* buf) { if (off + len > src.size()) return false; std::memcpy(buf, src.data() + off, (size_t)len); return true; };
        WSeg seg; seg.src_off = 0; seg.dst_off = 16; seg.src_bytes = src.size(); seg.dst_bytes = src.size() * 2; seg.conv = WConv::WidenF16;
        std::vector<uint8_t> ref;
        for (uint64_t chunk : {2ull, 6ull, 7ull, 64ull, 1000ull, (unsigned long long)kWeightChunk}) {
            std::vector<uint8_t> dst(16 + seg.dst_bytes + 16, 0xAB);
            const uint64_t w = convert_segment(seg, rd, dst.data(), dst.size(), chunk);
            CHECK(w == seg.dst_bytes);
            bool exact = true;
            for (int i = 0; i < n; ++i) { float f; std::memcpy(&f, dst.data() + 16 + 4 * i, 4); if (f != f16bits::f32_from_f16(half[i])) exact = false; }
            CHECK(exact);
            CHECK(dst[0] == 0xAB && dst[15] == 0xAB && dst[16 + seg.dst_bytes] == 0xAB);   // never writes outside the segment
            if (ref.empty()) ref = dst; else CHECK(dst == ref);
        }
        std::vector<uint8_t> small(10);
        CHECK(convert_segment(seg, rd, small.data(), small.size()) == UINT64_MAX);          // destination too small
        seg.src_bytes = 5; seg.dst_bytes = 10;
        CHECK(convert_segment(seg, rd, ref.data(), ref.size()) == UINT64_MAX);               // odd source cannot widen
    }
    // promoted rows: bf16 keeps the high half, f16 re-rounds
    {
        std::vector<float> f = {1.0f, -3.5f, 0.1f, 65504.0f, 1e-8f, 123456.789f};
        std::vector<uint8_t> src(f.size() * 4); std::memcpy(src.data(), f.data(), src.size());
        WReader rd = [&](uint64_t off, uint64_t len, uint8_t* buf) { std::memcpy(buf, src.data() + off, (size_t)len); return off + len <= src.size(); };
        for (int mode = 0; mode < 2; ++mode) {
            WSeg seg; seg.src_bytes = src.size(); seg.dst_bytes = src.size() / 2; seg.conv = mode ? WConv::ToF16 : WConv::Bf16High;
            for (uint64_t chunk : {4ull, 8ull, 12ull, 1000ull}) {
                std::vector<uint8_t> dst(seg.dst_bytes, 0);
                CHECK(convert_segment(seg, rd, dst.data(), dst.size(), chunk) == seg.dst_bytes);
                for (size_t i = 0; i < f.size(); ++i) {
                    uint16_t h; std::memcpy(&h, dst.data() + 2 * i, 2);
                    uint32_t bits; std::memcpy(&bits, &f[i], 4);
                    if (mode == 0) CHECK(h == (uint16_t)(bits >> 16));
                    else CHECK(h == f16bits::f16_from_f32(f[i]));
                }
            }
        }
    }
    // arena compaction under a skip set: kept rows re-placed in index order, 256-aligned, inside the pool
    for (int it = 0; it < 60; ++it) {
        const int n = 1 + (int)rn(12);
        std::string text; std::vector<uint64_t> sz; std::set<std::string> skip;
        for (int i = 0; i < n; ++i) {
            const uint64_t db = 4 * (1 + rn(400));
            sz.push_back(db);
            text += idx_line("t" + std::to_string(i), 0, 0, db, db, 0, 0, 0, 0);
            if (rn(3) == 0) skip.insert("t" + std::to_string(i));
        }
        WeightTable t;
        CHECK(t.place(text, skip));
        uint64_t at = 0;
        for (int i = 0; i < n; ++i) {
            const std::string nm = "t" + std::to_string(i);
            const WIndexRow* r = t.find(nm);
            if (skip.count(nm)) { CHECK(r == nullptr); continue; }
            CHECK(r != nullptr && r->placed);
            if (!r) continue;
            CHECK(r->arena_off == at && r->arena_off % 256 == 0);
            CHECK(r->arena_off + sz[i] <= t.pool_bytes());
            at += align256(sz[i]);
        }
        CHECK(t.pool_bytes() >= at);
    }
    // loading end-to-end, with the arena guard
    {
        std::vector<uint8_t> pack; std::string text;
        std::vector<float> fv = {1.0f, 2.0f, 3.0f, 4.0f};
        pack.resize(16); std::memcpy(pack.data(), fv.data(), 16);
        text += idx_line("a", 1, 0, 16, 8, 0, 0, 0, 0);
        WReader rd = [&](uint64_t off, uint64_t len, uint8_t* buf) { if (off + len > pack.size()) return false; std::memcpy(buf, pack.data() + off, (size_t)len); return true; };
        WeightTable t; std::vector<uint8_t> arena(1024, 0xCC); std::string err;
        CHECK(t.load_from(text, rd, arena.data(), arena.size(), {}, &err));
        const WIndexRow* a = t.find("a");
        CHECK(a && a->engine_kind == map_kind(1));
        if (a) for (int i = 0; i < 4; ++i) { uint16_t h; std::memcpy(&h, arena.data() + a->arena_off + 2 * i, 2); CHECK(h == (uint16_t)([&] { uint32_t b; std::memcpy(&b, &fv[i], 4); return b >> 16; }())); }
        std::vector<uint8_t> tiny(4, 0xCC);
        CHECK(!t.load_from(text, rd, tiny.data(), tiny.size(), {}, &err) && tiny[0] == 0xCC);
        CHECK(!t.load_from("garbage\n", rd, arena.data(), arena.size(), {}, &err));
        CHECK(t.find("missing") == nullptr);
    }
    CHECK(map_kind(4) == 1 && map_kind(5) == 3 && map_kind(2) == 2 && is_raw16(4) && is_raw16(5) && !is_raw16(1));
    CHECK(conv_for(true, 1) == WConv::Copy && conv_for(false, 1) == WConv::Bf16High && conv_for(false, 3) == WConv::ToF16 && conv_for(false, 0) == WConv::Copy);
}

// ================================================================================== MemoryPlanner
static void test_planner() {
    plan::Geometry g;   // default model geometry from the theory
    CHECK(g.n_gdn() == 36);
    CHECK(plan::state_bytes(g) == 117669888ull);
    CHECK(plan::state_bytes(g) / 1382400ull == 85);
    CHECK(plan::kv_bytes_per_token(g) == 13056ull);
    // kv cost is monotone in the number of QSA layers
    plan::Geometry g2 = g; g2.n_qsa = 6;
    CHECK(plan::kv_bytes_per_token(g2) <= plan::kv_bytes_per_token(g));

    CHECK(plan::vram_pool_bytes(10, 3) == 7 && plan::vram_pool_bytes(3, 10) == 0 && plan::vram_pool_bytes(5, 5) == 0 && plan::vram_pool_bytes(UINT64_MAX, 0) == UINT64_MAX);

    for (int it = 0; it < 4000; ++it) {
        plan::Costs c;
        c.expert_blob = rn(8) == 0 ? 0 : 1 + rn(5000000);
        c.dense = rn(2000000000ull); c.embd = rn(500000000ull); c.mtp = rn(300000000ull); c.workspace = rn(100000000ull); c.state = rn(200000000ull);
        const uint64_t ctx = rn(200000), pool = rn(8000000000ull);
        const auto res = plan::make_plan(ctx, g, c, pool);
        const uint64_t kvt = plan::kv_bytes_per_token(g);
        const unsigned __int128 kv = (unsigned __int128)kvt * ctx;
        if (const auto* p = std::get_if<plan::Plan>(&res)) {
            CHECK(p->cache_slots >= 1);                                   // makePlan_slots_pos
            CHECK(p->vram_used <= p->vram_budget && p->vram_budget <= pool);   // makePlan_budget / used_le
            CHECK(p->cache_bytes >= p->cache_slots * c.expert_blob);          // makePlan_slots_fit
            CHECK((unsigned __int128)p->kv_bytes == kv && p->max_context == ctx);
            CHECK((unsigned __int128)p->kv_bytes + (unsigned __int128)p->cache_slots * c.expert_blob <= p->vram_used);
            CHECK(c.expert_blob > 0);
        } else {
            const auto& d = std::get<plan::DoesNotClose>(res);
            // completeness: a refusal happens only if blob == 0, kv + fixed > pool, or no whole slot is left
            CHECK(d.blob == c.expert_blob && d.pool == pool);
            if (d.reason == plan::Reason::ZeroExpertBlob) CHECK(c.expert_blob == 0);
            else if (d.reason == plan::Reason::FixedExceedsPool) CHECK((unsigned __int128)d.kv_bytes + d.fixed > (unsigned __int128)pool || d.kv_bytes == UINT64_MAX || d.fixed == UINT64_MAX);
            else CHECK((unsigned __int128)d.kv_bytes + d.fixed <= (unsigned __int128)pool && pool - d.kv_bytes - d.fixed < c.expert_blob);
            CHECK(std::strlen(plan::to_string(d.reason)) > 10);
        }
        // overflow persists (kv_mono / overflow_persists): a larger context cannot close if a smaller one did not for lack of kv
        if (std::get_if<plan::DoesNotClose>(&res) && std::get<plan::DoesNotClose>(res).reason == plan::Reason::FixedExceedsPool) {
            const auto bigger = plan::make_plan(ctx + 1 + rn(1000), g, c, pool);
            CHECK(std::get_if<plan::DoesNotClose>(&bigger) != nullptr);
        }
    }
    // an unrepresentable cost refuses instead of wrapping
    plan::Costs c; c.expert_blob = 1000;
    const auto huge = plan::make_plan(UINT64_MAX, g, c, UINT64_MAX);
    CHECK(std::get_if<plan::DoesNotClose>(&huge) != nullptr);
}

// ================================================================================== PleNgram
static void test_ple_ngram() {
    CHECK(HEADS_PER_NGRAM == 8 && NGRAM_SIZE == 3 && PLE_N_HEADS == 16 && PLE_CONV_KERNEL == 4 && NG_HIST == 9);
    CHECK(NG_HC == 4 && NG_N_EMBD == 2560 && NG_HC_DIM == 10240 && PLE_HEAD_DIM == 160 && PLE_ROW_BYTES == 90);
    CHECK(PLE_TABLE_ROWS == 320001536ull && PLE_EOS_TOKEN_ID == 248044 && TOKEN_NULL == -1 && NG_RMS_EPS == 1e-6f);
    CHECK(PLE_N_HEADS * PLE_HEAD_DIM == NG_N_EMBD && NG_HC * NG_N_EMBD == NG_HC_DIM);

    const PleConsts c = ple_artifact_consts();
    uint64_t run = 0, sum = 0;
    for (int h = 0; h < PLE_N_HEADS; ++h) {
        CHECK(c.head_offsets[h] == run);                // offsets are prefix sums of the vocab sizes
        run += c.head_vocab_sizes[h]; sum += c.head_vocab_sizes[h];
    }
    CHECK(sum <= PLE_TABLE_ROWS && c.table_rows == PLE_TABLE_ROWS);
    CHECK(c.head_vocab_sizes[15] == 20000171ull && c.head_offsets[15] + c.head_vocab_sizes[15] == 320001446ull);
    CHECK(c.mult[0] == 23703573157769ull && c.mult[1] == 20109073645365ull && c.mult[2] == 8052911324071ull);

    const int64_t eos = PLE_EOS_TOKEN_ID;
    for (int it = 0; it < 20000; ++it) {
        int64_t w[3]; for (auto& x : w) x = (rn(6) == 0) ? TOKEN_NULL : (rn(8) == 0 ? eos : (int64_t)rn(248000));
        uint64_t rows[PLE_N_HEADS];
        ngram_rows(w, 3, eos, rows);
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            CHECK(rows[h] >= c.head_offsets[h] && rows[h] < c.head_offsets[h] + c.head_vocab_sizes[h]);   // hashed_row_in_table
            CHECK(rows[h] < PLE_TABLE_ROWS);
        }
        // window is OLDEST FIRST: w[2] is the current token, w[1] its predecessor, w[0] the oldest.
        // eos cut propagates: a missing/EOS predecessor makes everything older irrelevant
        if (w[1] < 0 || w[1] == eos) {
            int64_t w2[3] = {(int64_t)rn(248000), w[1], w[2]};
            uint64_t r2[PLE_N_HEADS]; ngram_rows(w2, 3, eos, r2);
            for (int h = 0; h < PLE_N_HEADS; ++h) CHECK(r2[h] == rows[h]);
        }
        // bigram heads (first 8) never look at the oldest token
        int64_t wo[3] = {(int64_t)rn(248000), w[1], w[2]};
        uint64_t ro[PLE_N_HEADS]; ngram_rows(wo, 3, eos, ro);
        for (int h = 0; h < HEADS_PER_NGRAM; ++h) CHECK(ro[h] == rows[h]);
        // an EOS or missing oldest token is the same as a cut context at that position
        int64_t wn[3] = {TOKEN_NULL, w[1], w[2]}, we[3] = {eos, w[1], w[2]};
        uint64_t rnn[PLE_N_HEADS], ree[PLE_N_HEADS]; ngram_rows(wn, 3, eos, rnn); ngram_rows(we, 3, eos, ree);
        for (int h = 0; h < PLE_N_HEADS; ++h) CHECK(rnn[h] == ree[h]);
    }
    // window of length 1 equals a window with missing predecessors
    { int64_t one[1] = {42}; int64_t three[3] = {TOKEN_NULL, TOKEN_NULL, 42}; uint64_t a[16], b[16]; ngram_rows(one, 1, eos, a); ngram_rows(three, 3, eos, b); for (int h = 0; h < 16; ++h) CHECK(a[h] == b[h]); }
    // mixed hash: xor of products, equal terms cancel, symmetric under swapping (value, multiplier) pairs
    { int64_t ctx[3] = {1234567, 1234567, 99}; CHECK(ngram_mixed(ctx, 2) == ((uint64_t)1234567 * c.mult[0] ^ (uint64_t)1234567 * c.mult[1]));
      CHECK(ngram_mixed(ctx, 1) == (uint64_t)1234567 * c.mult[0]);
      CHECK(ngram_mixed(ctx, 3) == (ngram_mixed(ctx, 2) ^ ((uint64_t)99 * c.mult[2]))); }

    // IQ4_NL: 16 sorted int8 codes; dequantization of a row: byte j of a block carries elements j (low) and j+16 (high)
    int prev = -1000;
    for (int i = 0; i < 16; ++i) { const int v = iq4nl_code(i); CHECK(v > prev && v >= -128 && v <= 127); prev = v; }
    CHECK(iq4nl_code(0) == -127 && iq4nl_code(8) == 1 && iq4nl_code(15) == 113 && iq4nl_code(16) == iq4nl_code(0));
    for (int it = 0; it < 200; ++it) {
        uint8_t row[PLE_ROW_BYTES]; for (auto& b : row) b = (uint8_t)rn(256);
        float scales[5];
        for (int blk = 0; blk < 5; ++blk) { scales[blk] = rf(-2, 2); const uint16_t h = f16bits::f16_from_f32(scales[blk]); scales[blk] = f16bits::f32_from_f16(h); row[blk * 18] = h & 0xff; row[blk * 18 + 1] = h >> 8; }
        float out[PLE_HEAD_DIM];
        iq4nl_dequant_row(row, out);
        bool ok = true;
        for (int blk = 0; blk < 5; ++blk) for (int e = 0; e < 32; ++e) {
            const uint8_t byte = row[blk * 18 + 2 + (e % 16)];
            const int nib = e < 16 ? (byte & 15) : (byte >> 4);
            const float want = scales[blk] * (float)iq4nl_code(nib);
            if (std::fabs(out[blk * 32 + e] - want) > 1e-6f * (1.f + std::fabs(want))) ok = false;
        }
        CHECK(ok);
    }
    ple_prefetch_enable(true); CHECK(ple_prefetch_enabled()); ple_prefetch_enable(false); CHECK(!ple_prefetch_enabled());
}

// ================================================================================== PleDirectReader
static void test_ple_reader() {
    using namespace pler;
    CHECK(PAGE == 4096 && ROW_BYTES == 90 && SLOT_BYTES == 8192 && WAYS == 8 && LATENCY_RING == 65536 && EMPTY == ~0ull);
    for (uint64_t off : {0ull, 1ull, 89ull, 4005ull, 4006ull, 4095ull, 4096ull, 8190ull, 123456789ull, 4096ull * 77 + 4006}) {
        const uint64_t first = first_of(off), len = length_of(off), in = off - first;
        CHECK(first <= off && first % PAGE == 0 && len % PAGE == 0 && (len == PAGE || len == 2 * PAGE) && len <= SLOT_BYTES);
        CHECK(off + ROW_BYTES <= first + len && in < PAGE && in + ROW_BYTES <= len);
    }
    for (int it = 0; it < 20000; ++it) {
        const uint64_t off = rn(1ull << 36);
        const uint64_t first = first_of(off), len = length_of(off);
        CHECK(off + ROW_BYTES <= first + len && len <= SLOT_BYTES && off - first + ROW_BYTES <= len);
        const uint64_t off2 = first + rn(PAGE);
        CHECK(first_of(off2) == first);
        CHECK(std::max(length_of(off), length_of(off2)) <= SLOT_BYTES);
        const bool straddles = (off % PAGE) + ROW_BYTES > PAGE;
        CHECK((len == 2 * PAGE) == straddles);
    }
    // ticket counter skips 0 and wraps
    CHECK(next_ticket(0) == 1 && next_ticket(0xFFFFFFFEu) == 0xFFFFFFFFu && next_ticket(0xFFFFFFFFu) == 1);
    for (int it = 0; it < 1000; ++it) CHECK(next_ticket((uint32_t)rng()) != 0);
    CHECK(ring_index(65535) == 65535 && ring_index(65536) == 0 && ring_index(1ull << 40) < LATENCY_RING);
    CHECK(pct_index(10, 3) == 3 && pct_index(10, 99) == 9 && pct_index(1, 5) == 0);
    CHECK(cache_sets(1000) * WAYS <= 1000 && cache_sets(7) == 0 && cache_sets(8) == 1);

    // set-associative cache: capacity, lookup/insert, round-robin replacement inside a set, index in range
    RowCache cache(8 * 64);
    CHECK(cache.sets() == 64 && cache.capacity() == 512 && cache.resident() == 0);
    uint8_t buf[ROW_BYTES], out[ROW_BYTES];
    for (uint64_t r = 0; r < 5000; ++r) CHECK(cache.set_index(r * 7919) < cache.sets());
    std::memset(buf, 0x5A, sizeof buf);
    CHECK(!cache.lookup(12345, out));
    cache.insert(12345, buf);
    CHECK(cache.lookup(12345, out) && std::memcmp(out, buf, ROW_BYTES) == 0 && cache.resident() == 1);
    std::memset(buf, 0x11, sizeof buf); cache.insert(12345, buf);                      // overwrite, not a new way
    CHECK(cache.resident() == 1 && cache.lookup(12345, out) && out[0] == 0x11);
    {   // fill one set beyond WAYS: exactly WAYS rows stay, the oldest are evicted first (round robin)
        std::vector<uint64_t> same; const uint64_t target = cache.set_index(1);
        for (uint64_t r = 2; same.size() < WAYS + 3 && r < 10000000; ++r) if (cache.set_index(r) == target) same.push_back(r);
        RowCache c2(8 * 64);
        for (size_t i = 0; i < same.size(); ++i) { std::memset(buf, (int)i + 1, sizeof buf); c2.insert(same[i], buf); }
        CHECK(c2.resident() == WAYS);
        for (size_t i = 0; i < same.size(); ++i) CHECK(c2.lookup(same[i], out) == (i >= same.size() - WAYS));
        c2.clear(); CHECK(c2.resident() == 0);
    }
    CHECK(RowCache(0).capacity() == 0 && !RowCache(0).lookup(1, out));

    // the real reader over a real file: byte-exact rows, cache hits, ticket protocol, stats
    const char* path = "ua96_ple_table.bin";
    const uint64_t table_off = 4096 + 17, n_rows = 3000;
    std::vector<uint8_t> file(table_off + n_rows * ROW_BYTES + 8192);
    for (auto& b : file) b = (uint8_t)rn(256);
    { FILE* f = std::fopen(path, "wb"); CHECK(f != nullptr); if (f) { std::fwrite(file.data(), 1, file.size(), f); std::fclose(f); } }
    {
        PleReader r;
        CHECK(!r.is_open());
        const bool opened = r.open(path, table_off, n_rows, 512, 4);
        CHECK(opened);
        if (opened) {
            CHECK(r.is_open() && r.cache_capacity() <= 512 && r.cache_capacity() % WAYS == 0);
            std::vector<uint64_t> rows;
            for (int i = 0; i < 64; ++i) rows.push_back(rn(n_rows));
            rows.push_back(0); rows.push_back(n_rows - 1);
            std::vector<uint8_t> dst(rows.size() * ROW_BYTES, 0);
            const uint32_t t1 = r.issue(rows.data(), rows.size(), dst.data());
            CHECK(t1 != 0 && r.collect(t1));
            bool exact = true;
            for (size_t i = 0; i < rows.size(); ++i) if (std::memcmp(dst.data() + i * ROW_BYTES, file.data() + table_off + rows[i] * ROW_BYTES, ROW_BYTES) != 0) exact = false;
            CHECK(exact);
            const ReaderStats s1 = r.stats();
            CHECK(s1.rows_requested == rows.size() && s1.errors == 0 && s1.jobs_done == s1.jobs_issued && s1.inflight == 0);
            CHECK(s1.bytes_read % PAGE == 0 && s1.bytes_read > 0);
            CHECK(r.cache_size() > 0 && r.cache_size() <= r.cache_capacity());
            // second pass over the same rows: all cache hits, no new reads, same bytes
            std::vector<uint8_t> dst2(rows.size() * ROW_BYTES, 0);
            const uint32_t t2 = r.issue(rows.data(), rows.size(), dst2.data());
            CHECK(t2 != 0 && t2 != t1 && r.collect(t2));
            const ReaderStats s2 = r.stats();
            CHECK(s2.bytes_read == s1.bytes_read && s2.cache_hits > s1.cache_hits && dst2 == dst);
            CHECK(s2.percentile(1.0) >= s2.percentile(0.0));
            r.reset_stats();
            CHECK(r.stats().rows_requested == 0 && r.stats().cache_hits == 0);
            // rows >= n_rows read as zeros: a valid ticket, no file read, buffer zero-filled
            {
                uint64_t bad[2] = {n_rows + 5, n_rows}; uint8_t two[2 * ROW_BYTES]; std::memset(two, 0xAB, sizeof two);
                const uint64_t b0 = r.stats().bytes_read, j0 = r.stats().jobs_issued;
                const uint32_t tb = r.issue(bad, 2, two);
                CHECK(tb != 0 && r.collect(tb));
                bool allz = true; for (uint8_t c : two) allz = allz && c == 0;
                CHECK(allz && r.stats().bytes_read == b0 && r.stats().jobs_issued == j0);
                // mixed in-range + out-of-range: in-range bytes match the table, out-of-range are zero
                uint64_t mix[2] = {0, n_rows + 1}; uint8_t mo[2 * ROW_BYTES], ref0[ROW_BYTES];
                const uint32_t tm = r.issue(mix, 2, mo); CHECK(tm != 0 && r.collect(tm));
                const uint32_t tr = r.issue(mix, 1, ref0); CHECK(tr != 0 && r.collect(tr));
                bool z2 = true; for (size_t k = ROW_BYTES; k < 2 * ROW_BYTES; ++k) z2 = z2 && mo[k] == 0;
                CHECK(z2 && std::memcmp(mo, ref0, ROW_BYTES) == 0);
            }
            // injected latency is observed in the sample ring
            r.set_injected_delay_us(2000);
            uint64_t cold = 0; for (uint64_t cand = 0; cand < n_rows; ++cand) { uint8_t tmp[ROW_BYTES]; const uint32_t t = r.issue(&cand, 1, tmp); if (t) { CHECK(r.collect(t)); cold = cand; break; } }
            (void)cold;
            r.set_injected_delay_us(0);
            r.close();
            CHECK(!r.is_open());
        }
        PleReader bad_open;
        CHECK(!bad_open.open("ua96_no_such_file.bin", 0, 10, 8, 2));
        PleReader too_big;
        CHECK(!too_big.open(path, table_off, n_rows * 100, 8, 2));   // size check: table must lie inside the file
    }
    std::remove(path);
}

// ================================================================================== SpecDraftController
static void test_spec() {
    using namespace spec;
    CHECK(K_MAX == 8);
    for (uint64_t m = 0; m < 200; ++m) { CHECK(bucket(m) < 4); if (m) CHECK(bucket(m) >= bucket(m - 1)); }
    CHECK(bucket(4) == 0 && bucket(5) == 1 && bucket(7) == 1 && bucket(8) == 2 && bucket(15) == 2 && bucket(16) == 3);
    for (uint64_t k = 0; k < 40; ++k) { CHECK(step_idx(k) <= K_MAX); CHECK(step_idx(k) == std::min<uint64_t>(k, K_MAX)); }
    for (int i = 1; i <= (int)K_MAX; ++i) { CHECK(kDenseRatio[i] >= kDenseRatio[i - 1]); CHECK(kDistinctRatio[i] > kDistinctRatio[i - 1]); }
    CHECK(kDenseRatio[0] == 100 && kDistinctRatio[0] == 100);

    // expected committed tokens: in [1, 1+k]; exactly 1+k at p=1, exactly 1 at p=0; monotone in k
    for (int it = 0; it < 2000; ++it) {
        double p[K_MAX]; for (auto& x : p) x = std::uniform_real_distribution<double>(0, 1)(rng);
        double prev = 1.0;
        for (uint64_t k = 0; k <= K_MAX; ++k) { const double e = expected(p, k); CHECK(e >= 1.0 && e <= 1.0 + (double)k + 1e-12 && e >= prev - 1e-12); prev = e; }
    }
    { double one[K_MAX], zero[K_MAX]; for (auto& x : one) x = 1.0; for (auto& x : zero) x = 0.0;
      for (uint64_t k = 0; k <= K_MAX; ++k) { CHECK(std::fabs(expected(one, k) - (1.0 + (double)k)) < 1e-12 && expected(zero, k) == 1.0); } }
    // integer recursion: expect_nat >= 1, <= 1 + k * 100^k-scale when p <= 1 (here p in 0/1)
    { auto p1 = [](uint64_t) { return 1ull; }; auto p0 = [](uint64_t) { return 0ull; };
      for (uint64_t k = 0; k < 9; ++k) { CHECK(expect_nat(p1, k) == 1 + k && expect_nat(p0, k) == 1); } }
    // per-mille form: runstep_le, expected_ge_one, expected <= 1000 + 1000*k for p <= 1000
    for (int it = 0; it < 2000; ++it) {
        std::vector<uint64_t> ps(K_MAX); for (auto& x : ps) x = rn(1001);
        auto P = [&](uint64_t i) { return ps[i]; };
        for (uint64_t k = 0; k <= K_MAX; ++k) { const uint64_t e = expected_pm(P, k); CHECK(e >= 1000 && e <= 1000 + 1000 * k); }
    }
    // pick_best never goes below the baseline; gate keeps exact margin
    for (int it = 0; it < 1000; ++it) {
        std::vector<uint64_t> c(rn(6)); for (auto& x : c) x = rn(1000);
        const uint64_t base = rn(1000), b = pick_best(base, c);
        CHECK(b >= base); for (auto x : c) CHECK(b >= x);
    }
    CHECK(keep_draft(105, 100) && !keep_draft(104, 100) && keep_draft(1050000000000ull, 1000000000000ull) && !keep_draft(1049999999999ull, 1000000000000ull));
    CHECK(clears_gate_pm(1050, 1000, 50) && !clears_gate_pm(1049, 1000, 50));
    CHECK(lookup_clip(3) == 3 && lookup_clip(100) == K_MAX);
    // learning: seen <= k, accepted <= seen, rejected prefix + 1; ema stays in range
    for (uint64_t k = 0; k <= 12; ++k) for (uint64_t a = 0; a <= 14; ++a) {
        const uint64_t s = seen(k, a);
        CHECK(s <= k && std::min(a, k) <= s);
        if (a < k) CHECK(s == a + 1);
        CHECK(committed(a, k) == 1 + std::min(a, k));
    }
    for (int it = 0; it < 20000; ++it) {
        const uint64_t S = 1 + rn(1000), e = rn(S + 1), x = rn(S + 1), h = rn(S + 1);
        CHECK(ema_step(S, e, x, h) <= S);
    }

    // the controller itself
    Controller c;
    CHECK(c.expected(Source::None, 5) == 1.0 && c.expected(Source::Mtp, 0) == 1.0);
    for (uint64_t k = 1; k <= K_MAX; ++k) {
        const double e = c.expected(Source::Mtp, k);
        CHECK(e > c.expected(Source::Mtp, k - 1) && e <= 1.0 + (double)k);
    }
    CHECK(c.expected(Source::Mtp, 100) == c.expected(Source::Mtp, K_MAX));                 // k clipped to K_MAX
    for (uint64_t m = 1; m < 40; ++m) CHECK(c.expected(Source::Lookup, 4, m) >= c.expected(Source::Lookup, 4, m - 1) - 1e-12);   // longer match, no worse
    {   // cost flat: speculation wins and takes the longest chain
        const Choice ch = c.choose(K_MAX, 0, 0);
        CHECK(ch.drafter == Source::Mtp && ch.k == K_MAX);
        CHECK(ch.rate >= 1.0 / c.cost.step_ms(0) * (1.0 + c.threshold));
    }
    {   // verify passes that cost far more than the gain: plain decoding, k = 0
        Controller costly; for (uint64_t i = 1; i <= K_MAX; ++i) costly.cost.table[i] = 50.0 * (double)(i + 1);
        const Choice ch = costly.choose(K_MAX, 8, 20);
        CHECK(ch.drafter == Source::None && ch.k == 0 && std::fabs(ch.rate - 1.0) < 1e-12);
    }
    {   // nothing available: baseline; lookup alone is chosen when it is the only candidate that wins
        Controller l; CHECK(l.choose(0, 0, 0).drafter == Source::None);
        const Choice ch = l.choose(0, 5, 20);
        CHECK(ch.drafter == Source::Lookup && ch.k >= 1 && ch.k <= 5);
    }
    // choose never returns less than plain decoding, whatever the cost table
    for (int it = 0; it < 500; ++it) {
        Controller r;
        for (uint64_t i = 0; i <= K_MAX; ++i) r.cost.table[i] = 0.5 + std::uniform_real_distribution<double>(0, 3)(rng);
        r.cost.mtp_ms_per_tok = std::uniform_real_distribution<double>(0, 0.5)(rng);
        r.cost.lookup_ms = std::uniform_real_distribution<double>(0, 0.5)(rng);
        const Choice ch = r.choose(rn(12), rn(12), rn(30));
        CHECK(ch.rate >= 1.0 / r.cost.step_ms(0) - 1e-12 && ch.k <= K_MAX);
        if (ch.drafter == Source::None) CHECK(ch.k == 0);
    }
    // observe: probabilities stay in [0,1]; positions beyond `seen` are untouched; a hit moves up, a miss moves down
    for (int it = 0; it < 3000; ++it) {
        Controller r; Controller before = r;
        const uint64_t k = 1 + rn(K_MAX), acc = rn(k + 3), lm = rn(30);
        const Drafter d = rn(2) ? Drafter::Mtp : Drafter::Lookup;
        r.observe(d, k, acc, lm);
        const uint64_t n = seen(k, acc);
        if (d == Drafter::Mtp) {
            for (uint64_t i = 0; i < K_MAX; ++i) {
                CHECK(r.mtp_p[i] >= 0.0 && r.mtp_p[i] <= 1.0);
                if (i >= n) CHECK(r.mtp_p[i] == before.mtp_p[i]);
                else if (i < acc) CHECK(r.mtp_p[i] > before.mtp_p[i]);
                else CHECK(r.mtp_p[i] < before.mtp_p[i]);
            }
        } else {
            for (int b = 0; b < 4; ++b) CHECK(r.lookup_q[b] >= 0.0 && r.lookup_q[b] <= 1.0);
            for (int b = 0; b < 4; ++b) if ((uint64_t)b != bucket(lm)) CHECK(r.lookup_q[b] == before.lookup_q[b]);
        }
    }
    { Controller r; for (int i = 0; i < 2000; ++i) r.observe(Drafter::Mtp, K_MAX, rn(K_MAX + 1), 0);
      for (uint64_t i = 0; i < K_MAX; ++i) CHECK(r.mtp_p[i] >= 0.0 && r.mtp_p[i] <= 1.0);
      Controller z; z.observe(Drafter::None, 4, 4); z.observe(Drafter::Mtp, 0, 0);
      for (uint64_t i = 0; i < K_MAX; ++i) CHECK(z.mtp_p[i] == 0.86); }
    {   // learning converges: constant full acceptance drives the table toward 1, constant rejection toward 0
        Controller up, down;
        for (int i = 0; i < 400; ++i) { up.observe(Drafter::Mtp, 4, 4); down.observe(Drafter::Mtp, 4, 0); }
        CHECK(up.mtp_p[0] > 0.999 && up.mtp_p[3] > 0.999 && up.mtp_p[4] == 0.86);
        CHECK(down.mtp_p[0] < 0.001 && down.mtp_p[1] == 0.86);
    }
}

// ================================================================================== GPU kernels (GDN + gated residual)
template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t n_) : n(n_) { cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T)); cudaMemset(p, 0, (n ? n : 1) * sizeof(T)); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { if (n) cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice); }
    Dev(const Dev&) = delete; Dev& operator=(const Dev&) = delete;
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> v(n); if (n) cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }
    operator T*() const { return p; }
};
static bool sync_ok() { return cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess; }
static std::vector<float> vf(size_t n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = rf(lo, hi); return v; }
static double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static bool near_all(const std::vector<float>& a, const std::vector<double>& r, double rtol, double atol) {
    if (a.size() != r.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::isnan(a[i]) || std::fabs((double)a[i] - r[i]) > atol + rtol * std::fabs(r[i])) return false;
    return true;
}

static void test_gdn_gpu() {
    const int S = 128, h_k = 2, h_v = 4;
    const size_t SS = (size_t)S * h_v * S;
    // --- delta rule exactness (delta_write_exact / delta_exact): gate = 0 (decay 1), beta = 1, one-hot key at row r
    //     => after the step the state row r equals v for every head, other rows unchanged. Fused and reference agree.
    {
        const int r0 = 37;
        auto st0 = vf(SS, -0.5f, 0.5f);
        std::vector<float> q(h_k * S, 0.f), k(h_k * S, 0.f), v = vf(h_v * S, -1, 1), gate(h_v, 0.f), beta(h_v, 1.f), z = vf(h_v * S, -1, 1), nw(S, 1.f);
        for (int h = 0; h < h_k; ++h) { k[h * S + r0] = 1.f; q[h * S + r0] = 1.f; }
        for (int variant = 0; variant < 2; ++variant) {
            Dev<float> st(st0), dq(q), dk(k), dv(v), dg(gate), db(beta), dz(z), dnw(nw), o((size_t)h_v * S), y((size_t)h_v * S);
            if (variant == 0) CHECK(kg::gdn_step(st, dq, dk, dv, dg, db, o, S, h_k, h_v) == cudaSuccess && sync_ok());
            else CHECK(kg::fused_gdn_step_norm(st, dq, dk, dv, dg, db, dz, dnw, y, h_k, h_v, 1e-6f) == cudaSuccess && sync_ok());
            auto s = st.get(); bool row_is_v = true, rest_same = true;
            for (int row = 0; row < S; ++row) for (int h = 0; h < h_v; ++h) for (int j = 0; j < S; ++j) {
                const float got = s[((size_t)row * h_v + h) * S + j];
                if (row == r0) { if (std::fabs(got - v[h * S + j]) > 1e-5f) row_is_v = false; }
                else if (std::fabs(got - st0[((size_t)row * h_v + h) * S + j]) > 1e-6f) rest_same = false;
            }
            CHECK(row_is_v && rest_same);
            if (variant == 0) { auto ov = o.get(); bool eq = true; for (int h = 0; h < h_v; ++h) for (int j = 0; j < S; ++j) if (std::fabs(ov[h * S + j] - v[h * S + j]) > 1e-5f) eq = false; CHECK(eq); }
        }
    }
    // --- zero write strength only decays: state' = exp(gate) * state, output = q . state'
    {
        auto st0 = vf(SS, -0.5f, 0.5f), q = vf(h_k * S, -1, 1), k = vf(h_k * S, -1, 1), v = vf(h_v * S, -1, 1);
        auto gate = vf(h_v, -1.5f, -0.1f); std::vector<float> beta(h_v, 0.f);
        Dev<float> st(st0), dq(q), dk(k), dv(v), dg(gate), db(beta), o((size_t)h_v * S);
        CHECK(kg::gdn_step(st, dq, dk, dv, dg, db, o, S, h_k, h_v) == cudaSuccess && sync_ok());
        auto s = st.get(); std::vector<double> want(SS);
        for (int row = 0; row < S; ++row) for (int h = 0; h < h_v; ++h) for (int j = 0; j < S; ++j) { const size_t i = ((size_t)row * h_v + h) * S + j; want[i] = (double)st0[i] * std::exp((double)gate[h]); }
        CHECK(near_all(s, want, 1e-5, 1e-6));
        std::vector<double> wo((size_t)h_v * S);
        for (int h = 0; h < h_v; ++h) for (int j = 0; j < S; ++j) { double a = 0; for (int row = 0; row < S; ++row) a += want[((size_t)row * h_v + h) * S + j] * q[(h % h_k) * S + row]; wo[h * S + j] = a; }
        CHECK(near_all(o.get(), wo, 1e-4, 1e-4));
    }
    // --- fused step+norm == reference step then out_norm, for several (h_k, h_v) head pairings (head % h_k)
    for (int hk : {1, 2, 4}) for (int hv : {4, 8}) {
        if (hv % hk) continue;
        const size_t ss = (size_t)S * hv * S;
        auto st0 = vf(ss, -0.5f, 0.5f), q = vf(hk * S, -1, 1), k = vf(hk * S, -1, 1), v = vf(hv * S, -1, 1), z = vf(hv * S, -2, 2), nw = vf(S, 0.5f, 1.5f);
        auto gate = vf(hv, -1.5f, -0.1f), beta = vf(hv, 0.1f, 0.9f);
        Dev<float> sA(st0), sB(st0), dq(q), dk(k), dv(v), dz(z), dnw(nw), dg(gate), db(beta), o((size_t)hv * S), yA((size_t)hv * S), yB((size_t)hv * S);
        CHECK(kg::gdn_step(sA, dq, dk, dv, dg, db, o, S, hk, hv) == cudaSuccess && kg::gdn_out_norm(o, dz, dnw, yA, hv, S, 1e-6f) == cudaSuccess && sync_ok());
        CHECK(kg::fused_gdn_step_norm(sB, dq, dk, dv, dg, db, dz, dnw, yB, hk, hv, 1e-6f) == cudaSuccess && sync_ok());
        CHECK(sA.get() == sB.get());
        std::vector<double> ry; { auto ya = yA.get(); ry.assign(ya.begin(), ya.end()); }
        CHECK(near_all(yB.get(), ry, 1e-4, 1e-5));
        // out_norm law: y = rms_norm(o) * w * sigmoid(z)
        auto ov = o.get(); std::vector<double> want((size_t)hv * S);
        for (int h = 0; h < hv; ++h) { double s2 = 0; for (int i = 0; i < S; ++i) s2 += (double)ov[h * S + i] * ov[h * S + i]; const double inv = 1.0 / std::sqrt(s2 / S + 1e-6);
            for (int i = 0; i < S; ++i) { const double zz = z[h * S + i]; want[h * S + i] = ov[h * S + i] * inv * nw[i] * sigm(zz); } }
        CHECK(near_all(yA.get(), want, 1e-4, 1e-5));
    }
    // --- conv: four taps, oldest history meets tap 0, history slides by one, newest slot holds the input
    {
        const int C = (2 * h_k + h_v) * S, dc = 4;
        auto cs0 = vf((size_t)C * (dc - 1), -1, 1), xin = vf(C, -1, 1), kW = vf((size_t)C * dc, -1, 1);
        Dev<float> c1(cs0), c2(cs0), dx(xin), dW(kW), o1(C), o2(C);
        CHECK(kg::gdn_conv_step(c1, dx, dW, o1, C, dc) == cudaSuccess && sync_ok());
        std::vector<double> want(C);
        for (int c = 0; c < C; ++c) want[c] = (double)cs0[c * 3] * kW[c * 4] + (double)cs0[c * 3 + 1] * kW[c * 4 + 1] + (double)cs0[c * 3 + 2] * kW[c * 4 + 2] + (double)xin[c] * kW[c * 4 + 3];
        CHECK(near_all(o1.get(), want, 1e-5, 1e-6));
        auto hist = c1.get(); bool slide = true;
        for (int c = 0; c < C; ++c) if (hist[c * 3] != cs0[c * 3 + 1] || hist[c * 3 + 1] != cs0[c * 3 + 2] || hist[c * 3 + 2] != xin[c]) slide = false;
        CHECK(slide);
        // fused conv + l2: same history, v channels unchanged by the norm, q|k heads unit length
        CHECK(kg::fused_gdn_conv_l2(c2, dx, dW, o2, h_k, h_v, dc, 1e-6f) == cudaSuccess && sync_ok());
        CHECK(c2.get() == hist);
        auto fo = o2.get(); bool v_same = true, unit = true;
        for (int c = 2 * h_k * S; c < C; ++c) if (std::fabs(fo[c] - want[c]) > 1e-5 + 1e-5 * std::fabs(want[c])) v_same = false;
        for (int hd = 0; hd < 2 * h_k; ++hd) { double s2 = 0; for (int i = 0; i < S; ++i) s2 += (double)fo[hd * S + i] * fo[hd * S + i]; if (std::fabs(s2 - 1.0) > 1e-3) unit = false; }
        CHECK(v_same && unit);
        // with an all-zero history only the last tap contributes
        std::vector<float> zero_hist((size_t)C * 3, 0.f); Dev<float> c3(zero_hist), o3(C);
        CHECK(kg::gdn_conv_step(c3, dx, dW, o3, C, dc) == cudaSuccess && sync_ok());
        std::vector<double> w3(C); for (int c = 0; c < C; ++c) w3[c] = (double)xin[c] * kW[c * 4 + 3];
        CHECK(near_all(o3.get(), w3, 1e-6, 1e-7));
        // two consecutive steps: the input of step one is the middle tap at step two
        std::vector<float> x2 = vf(C, -1, 1); Dev<float> dx2(x2), o4(C);
        CHECK(kg::gdn_conv_step(c1, dx2, dW, o4, C, dc) == cudaSuccess && sync_ok());
        auto h2 = c1.get(); bool twice = true; for (int c = 0; c < C; ++c) if (h2[c * 3 + 1] != xin[c] || h2[c * 3 + 2] != x2[c]) twice = false;
        CHECK(twice);
        CHECK(kg::gdn_l2_norm(nullptr, 1, 128, 1e-6f) != cudaSuccess);
    }
    // --- beta/gate: beta = sigmoid(b), gate = -exp(a_log) * softplus(a + dt_bias); stacked alpha/beta projection rows
    {
        const int hv = 48, n_embd = 2560;
        auto wa = vf((size_t)hv * n_embd, -0.1f, 0.1f), wb = vf((size_t)hv * n_embd, -0.1f, 0.1f), x = vf(n_embd, -1, 1), alog = vf(hv, -1, 1), dtb = vf(hv, -1, 1);
        std::vector<uint16_t> ba(wa.size()), bb(wb.size());
        std::vector<float> wa_r(wa.size()), wb_r(wb.size());
        for (size_t i = 0; i < wa.size(); ++i) { ba[i] = f16bits::round_f32_to_bf16_bits(wa[i]); wa_r[i] = f16bits::bf16_bits_to_f32(ba[i]); bb[i] = f16bits::round_f32_to_bf16_bits(wb[i]); wb_r[i] = f16bits::bf16_bits_to_f32(bb[i]); }
        Dev<uint16_t> dwa(ba), dwb(bb); Dev<float> dx(x), dl(alog), dd(dtb), beta(hv), g(hv);
        CHECK(kg::fused_gdn_ab(dwa, dwb, dx, dl, dd, beta, g, hv, n_embd) == cudaSuccess && sync_ok());
        std::vector<double> rb(hv), rg(hv);
        for (int h = 0; h < hv; ++h) { double a = 0, b = 0; for (int i = 0; i < n_embd; ++i) { a += (double)wa_r[(size_t)h * n_embd + i] * x[i]; b += (double)wb_r[(size_t)h * n_embd + i] * x[i]; }
            rb[h] = sigm(b); const double zz = a + dtb[h]; rg[h] = -std::exp((double)alog[h]) * (zz > 20 ? zz : std::log1p(std::exp(zz))); }
        CHECK(near_all(beta.get(), rb, 2e-3, 1e-4) && near_all(g.get(), rg, 2e-3, 1e-4));
        // beta in (0,1), gate <= 0 (decay factor in (0,1])
        bool range = true; for (auto b : beta.get()) range = range && b > 0.f && b < 1.f; for (auto gg : g.get()) range = range && gg <= 0.f; CHECK(range);
        // plain beta gate is an in-place sigmoid; covers every head (grid >= n)
        for (int n : {1, 3, 127, 128, 129, 300}) {
            auto b0 = vf(n, -4, 4); Dev<float> db(b0);
            CHECK(kg::gdn_beta_gate(db, n) == cudaSuccess && sync_ok());
            std::vector<double> w(n); for (int i = 0; i < n; ++i) w[i] = sigm(b0[i]);
            CHECK(near_all(db.get(), w, 1e-5, 1e-6));
        }
    }
}

static void test_gr_gpu() {
    // workspace: five 16-aligned regions, 104,320 bytes at the real geometry; xq holds 2 bytes per element
    kg::GrDims d{2560, 4, 320};
    CHECK(kg::gr_workspace_bytes(d) == 104320);
    {
        Dev<uint8_t> buf(104320); kg::GrWs ws;
        CHECK(kg::gr_workspace_init(buf, 104320, d, &ws));
        CHECK(!kg::gr_workspace_init(buf, 104319, d, &ws));
        CHECK(kg::gr_workspace_init(buf, 104320, d, &ws));
        const uint8_t* b = buf;
        CHECK((const uint8_t*)ws.xn == b && (const uint8_t*)ws.xq - b == 40960 && (const uint8_t*)ws.lq - b == 61440 &&
              (const uint8_t*)ws.gated - b == 62080 && (const uint8_t*)ws.lo - b == 103040 && ws.bytes == 104320);
    }
    for (int it = 0; it < 100; ++it) {      // alignment laws over random geometries
        kg::GrDims o{1 + (int)rn(5000), 1 + (int)rn(6), 1 + (int)rn(700)};
        const size_t bytes = kg::gr_workspace_bytes(o);
        auto a16 = [](size_t n) { return (n + 15) / 16 * 16; };
        const size_t hd = (size_t)o.n * o.hc;
        const size_t want = a16(hd * 4) + a16(hd * 2) + a16((size_t)o.lr * 2) + a16(hd * 4) + a16((size_t)o.lr * 4);
        CHECK(bytes % 16 == 0 && bytes == want);
        Dev<uint8_t> buf(bytes + 16); kg::GrWs ws;
        CHECK(kg::gr_workspace_init(buf, bytes, o, &ws) && !kg::gr_workspace_init(buf, bytes - 1, o, &ws));
    }
    // fused geometry gate and the write law out[i] = R[i] + block_out[i % n] * w[i / n]
    CHECK(kg::fused_gr_supported(2560, 4, 320) && !kg::fused_gr_supported(2560, 4, 321) && !kg::fused_gr_supported(64, 4, 16) && !kg::fused_gr_supported(2560, 3, 320));
    {
        for (int n : {64, 2560}) {
            const int hc = 4, hd = n * hc;
            auto R = vf(hd, -2, 2), bo = vf(n, -1, 1), w = vf(hc, -1, 1);
            Dev<float> dR(R), dbo(bo), dw(w), out(hd);
            CHECK(kg::gr_write(dR, dbo, dw, out, n, hc) == cudaSuccess && sync_ok());
            std::vector<double> want(hd); for (int i = 0; i < hd; ++i) want[i] = (double)R[i] + (double)bo[i % n] * w[i / n];
            CHECK(near_all(out.get(), want, 1e-6, 1e-6));
        }
    }
    // fused read == reference read, and both equal the host statement of the law (rms-norm, low-rank gate, mean, injection)
    {
        const int n = 2560, hc = 4, lr = 320, hd = n * hc;
        auto R = vf(hd, -2, 2);
        std::vector<uint16_t> down((size_t)lr * hd), up((size_t)hd * lr), inj((size_t)hc * n);
        for (auto& x : down) x = f16bits::round_f32_to_bf16_bits(rf(-0.05f, 0.05f));
        for (auto& x : up) x = f16bits::round_f32_to_bf16_bits(rf(-0.3f, 0.3f));
        for (auto& x : inj) x = f16bits::round_f32_to_bf16_bits(rf(-0.2f, 0.2f));
        Dev<float> dR(R), dx(n), dw(hc), fx(n), fw(hc), lo(lr);
        Dev<uint16_t> dd(down), du(up), di(inj);
        Dev<uint8_t> buf(kg::gr_workspace_bytes(d)); kg::GrWs ws; kg::gr_workspace_init(buf, buf.n, d, &ws);
        CHECK(kg::gr_read(dR, dd, du, di, ws, d, 1e-6f, dx, dw) == cudaSuccess && kg::fused_gr_read(dR, dd, du, di, lo, fx, fw, 1e-6f) == cudaSuccess && sync_ok());
        // host reference with the same bf16 activation rounding the kernels use
        auto bf = [](uint16_t b) { return f16bits::bf16_bits_to_f32(b); };
        auto rbf = [](float x) { return f16bits::bf16_bits_to_f32(f16bits::round_f32_to_bf16_bits(x)); };
        std::vector<float> xn(hd), xq(hd), lq(lr), lov(lr), gated(hd);
        for (int c = 0; c < hc; ++c) { float s = 0; for (int i = 0; i < n; ++i) s += R[c * n + i] * R[c * n + i]; const float inv = 1.0f / std::sqrt(s / (float)n + 1e-6f); for (int i = 0; i < n; ++i) { xn[c * n + i] = R[c * n + i] * inv; xq[c * n + i] = rbf(xn[c * n + i]); } }
        for (int k = 0; k < lr; ++k) { double a = 0; for (int i = 0; i < hd; ++i) a += (double)bf(down[(size_t)k * hd + i]) * xq[i]; lov[k] = (float)(a * sigm(a)); lq[k] = rbf(lov[k]); }
        for (int i = 0; i < hd; ++i) { double a = 0; for (int k = 0; k < lr; ++k) a += (double)bf(up[(size_t)i * lr + k]) * lq[k]; gated[i] = xn[i] * (float)sigm(a); }
        std::vector<double> rx(n), rw(hc);
        for (int i = 0; i < n; ++i) { float s = 0; for (int c = 0; c < hc; ++c) s += gated[c * n + i]; rx[i] = s / (float)hc; }
        for (int c = 0; c < hc; ++c) { double a = 0; for (int i = 0; i < n; ++i) a += (double)bf(inj[(size_t)c * n + i]) * rx[i]; rw[c] = a; }
        CHECK(near_all(dx.get(), rx, 3e-3, 2e-4) && near_all(dw.get(), rw, 3e-3, 2e-3));
        CHECK(near_all(fx.get(), rx, 3e-3, 2e-4) && near_all(fw.get(), rw, 3e-3, 2e-3));
        std::vector<double> rlo(lov.begin(), lov.end());
        CHECK(near_all(lo.get(), rlo, 3e-3, 2e-4));
        // multi-token variant equals per-token launches for T = 1..32 (lane k serves token k), and T > 32 is refused
        for (int T : {1, 2, 5, 32}) {
            auto RT = vf((size_t)T * hd, -2, 2);
            Dev<float> dRT(RT), mlo((size_t)T * lr), mx((size_t)T * n), mw((size_t)T * hc);
            kg::GrMulti m; m.R = dRT; m.down = dd; m.up = du; m.inject = di; m.lo = mlo; m.x_out = mx; m.w_out = mw; m.eps = 1e-6f; m.T = T;
            CHECK(kg::fused_gr_read_multi(m) == cudaSuccess && sync_ok());
            auto MX = mx.get(), MW = mw.get(); bool same = true;
            for (int t = 0; t < T; ++t) {
                std::vector<float> Rt(RT.begin() + (size_t)t * hd, RT.begin() + (size_t)(t + 1) * hd);
                Dev<float> dRt(Rt), x1(n), w1(hc), l1(lr);
                kg::fused_gr_read(dRt, dd, du, di, l1, x1, w1, 1e-6f); sync_ok();
                auto X = x1.get(), W = w1.get();
                for (int i = 0; i < n; ++i) if (std::fabs(X[i] - MX[(size_t)t * n + i]) > 2e-3 * (1 + std::fabs(X[i]))) same = false;
                for (int c = 0; c < hc; ++c) if (std::fabs(W[c] - MW[t * hc + c]) > 2e-3 * (1 + std::fabs(W[c]))) same = false;
            }
            CHECK(same);
        }
        kg::GrMulti bad; bad.T = 33; bad.R = dR; bad.down = dd; bad.up = du; bad.inject = di; bad.lo = lo; bad.x_out = fx; bad.w_out = fw;
        CHECK(kg::fused_gr_read_multi(bad) != cudaSuccess);
        bad.T = 0; CHECK(kg::fused_gr_read_multi(bad) != cudaSuccess);
        // streams are normalised independently: scaling one stream of R must not change the other streams' xn contribution
        // -> the injection weights of scaled R equal those of R when every stream is scaled by the same factor (rms-norm is scale free)
        std::vector<float> R2(R); for (auto& x : R2) x *= 3.0f;
        Dev<float> dR2(R2), x2(n), w2(hc), l2(lr);
        CHECK(kg::fused_gr_read(dR2, dd, du, di, l2, x2, w2, 1e-6f) == cudaSuccess && sync_ok());
        std::vector<double> rx0; { auto a = fx.get(); rx0.assign(a.begin(), a.end()); }
        CHECK(near_all(x2.get(), rx0, 5e-3, 5e-4));
    }
}

// device-side launch geometry: every claimed element is claimed exactly once, nothing out of range, and the bounds are
// the theory's literal numbers (40 down blocks x 8 warps = 320, 80 up blocks x 32 = 2560, 160 x 16 = 2560, 4 x 32 = 128)
static void test_geometry_gpu() {
    using namespace ua96;
    CHECK(kDownBlocks * 8 == kLr && kUpBlocks * kUpCols == kN && kUpmBlocks * kUpmCols == kN && kN * kHc == kD);
    CHECK(kStepRg * kStepRpg == kS && kS * kStepRg == 512);
    Dev<uint32_t> down(kLr), up(kD), upm(kD), gate(kD), step((size_t)kS * kS), conv(kConvChannels), oob(1);
    CoverBuffers b{down, up, upm, gate, step, conv, oob};
    CHECK(cover_all(b) == cudaSuccess && sync_ok());
    auto all_one = [](const std::vector<uint32_t>& v) { for (auto x : v) if (x != 1u) return false; return !v.empty(); };
    CHECK(all_one(down.get()));
    CHECK(all_one(up.get()));
    CHECK(all_one(upm.get()));
    CHECK(all_one(gate.get()));
    CHECK(all_one(step.get()));
    CHECK(all_one(conv.get()));
    CHECK(oob.get()[0] == 0u);
    CoverBuffers bad = b; bad.up = nullptr;
    CHECK(cover_all(bad) != cudaSuccess);
}

int main() {
#ifndef UA96_HOST_ONLY   // the kernel build never defines this; it only exists for syntax/host checks on machines without a GPU
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::fprintf(stderr, "no CUDA device\n"); return 2; }
    cudaSetDevice(0);
#endif
    test_session();
    test_weights();
    test_planner();
    test_ple_ngram();
    test_ple_reader();
    test_spec();
#ifndef UA96_HOST_ONLY
    test_gdn_gpu();
    test_gr_gpu();
    test_geometry_gpu();
#endif
    std::printf("checks=%d failures=%d\n", g_checks, g_fail);
    if (g_fail) { std::printf("UA96 UNIT TESTS FAILED\n"); return 1; }
    std::printf("UA96 UNIT TESTS PASSED\n");
    return 0;
}

