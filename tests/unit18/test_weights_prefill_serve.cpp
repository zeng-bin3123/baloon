// WeightLoader + PrefillOrchestration + ServeEngine invariants.
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>
#include "baloon/core_prefill.hpp"
#include "baloon/core_weights.hpp"
#include "baloon/f16_conversion.hpp"
#include "baloon/serve_engine.hpp"
#include "check.hpp"

using namespace baloon::core;
namespace f16 = baloon::f16;

static std::string idx_line(const std::string& n, int kind, uint64_t so, uint64_t sb, uint64_t db, uint64_t c, uint64_t s, uint64_t o, int fp16) {
    return n + " " + std::to_string(kind) + " " + std::to_string(so) + " " + std::to_string(sb) + " " + std::to_string(db) + " " +
           std::to_string(c) + " " + std::to_string(s) + " " + std::to_string(o) + " " + std::to_string(fp16) + "\n";
}

void test_weights() {
    CHECK(kWeightChunk == 8388608ull);
    // chunk loop: progress, bounded, element-preserving
    for (uint64_t rem : {1ull, 7ull, 8388607ull, 8388608ull, 8388609ull, 20000000ull}) {
        const uint64_t n = chunk_len_of(rem);
        CHECK(n > 0 && n <= kWeightChunk && rem - n < rem);
        if (rem >= kWeightChunk) CHECK(n % 2 == 0);
        if (rem % 4 == 0) CHECK(n % 4 == 0);
    }
    // placement: 256-byte aligned rows, never overlapping; compaction by the skip set
    testh::Rng rng(17);
    for (int it = 0; it < 500; ++it) {
        std::string text; std::vector<uint64_t> sz; std::set<std::string> skip;
        const int n = 1 + (int)rng.below(8);
        for (int i = 0; i < n; ++i) {
            const uint64_t db = 1 + rng.below(1000);
            sz.push_back(db);
            text += idx_line("t" + std::to_string(i), 0, 0, db, db, 0, 0, 0, 0);
            if (rng.below(3) == 0) skip.insert("t" + std::to_string(i));
        }
        WeightTable t;
        CHECK(t.place(text, skip));
        uint64_t at = 0;
        for (int i = 0; i < n; ++i) {
            const WIndexRow* r = t.find("t" + std::to_string(i));
            if (skip.count("t" + std::to_string(i))) { CHECK(r == nullptr); continue; }
            CHECK(r && r->arena_off == at && r->arena_off % 256 == 0);
            CHECK(r->arena_off + sz[(size_t)i] <= r->arena_off + weight_align256(sz[(size_t)i]));   // rows never overlap
            at += weight_align256(sz[(size_t)i]);
        }
        CHECK(t.pool_bytes() == at && at % 256 == 0);
    }
    // kind mapping and conversion selection
    CHECK(map_kind(4) == 1 && map_kind(5) == 3 && map_kind(0) == 0 && map_kind(1) == 1 && map_kind(2) == 2 && map_kind(7) == 7);
    CHECK(conv_for(true, 1) == WConv::Copy && conv_for(true, 3) == WConv::Copy && conv_for(false, 1) == WConv::Bf16High &&
          conv_for(false, 3) == WConv::ToF16 && conv_for(false, 0) == WConv::Copy);
    // segments: totals checked before any byte moves
    WIndexRow q; q.codes = 64; q.scales = 8; q.offset = 4; q.scales_fp16 = true; q.src_bytes = 64 + 4 + 4; q.dst_bytes = 76;
    std::vector<WSeg> sg;
    CHECK(plan_segments(q, sg) && sg.size() == 3 && sg[1].conv == WConv::WidenF16 && sg[1].src_bytes == 4 && sg[1].dst_bytes == 8);
    CHECK(sg[0].src_off + sg[0].src_bytes == sg[1].src_off && sg[1].src_off + sg[1].src_bytes == sg[2].src_off);
    CHECK(sg[0].dst_off + sg[0].dst_bytes == sg[1].dst_off && sg[1].dst_off + sg[1].dst_bytes == sg[2].dst_off);
    q.src_bytes += 1; CHECK(!plan_segments(q, sg));
    q.src_bytes -= 1; q.dst_bytes += 1; CHECK(!plan_segments(q, sg));
    q.dst_bytes -= 1; q.scales = 7; CHECK(!plan_segments(q, sg));                    // odd fp16 plane
    WIndexRow pr; pr.kind = 1; pr.src_bytes = 40; pr.dst_bytes = 20;                  // promoted: 4 B in, 2 B out
    CHECK(plan_segments(pr, sg) && sg.size() == 1 && sg[0].conv == WConv::Bf16High);
    pr.dst_bytes = 40; CHECK(!plan_segments(pr, sg));
    WIndexRow raw; raw.kind = 4; raw.src_bytes = 40; raw.dst_bytes = 40;
    CHECK(plan_segments(raw, sg) && sg[0].conv == WConv::Copy);
    // end to end through a reader: widen exact, bf16 high half, f16 RNE, raw copy, quantized planes
    std::vector<uint8_t> pack;
    auto put = [&](const void* p, size_t n) { const uint8_t* b = (const uint8_t*)p; pack.insert(pack.end(), b, b + n); };
    std::string text;
    uint64_t off = 0;
    // row A: quantized, codes 8 bytes, 2 fp16 scales, 4 offset bytes
    uint8_t codes[8]; for (int i = 0; i < 8; ++i) codes[i] = (uint8_t)(i + 1);
    uint16_t sc[2] = {f16::float_to_half(1.5f), f16::float_to_half(-0.25f)};
    uint8_t offs[4] = {9, 8, 7, 6};
    put(codes, 8); put(sc, 4); put(offs, 4);
    text += idx_line("A", 2, off, 16, 8 + 8 + 4, 8, 8, 4, 1); off += 16;
    // row B: promoted bf16 from f32
    float fb[3] = {1.0f, -3.5f, 0.1f};
    put(fb, 12);
    text += idx_line("B", 1, off, 12, 6, 0, 0, 0, 0); off += 12;
    // row C: promoted f16 from f32
    float fc[3] = {1.0f, 65504.0f, 1e-8f};
    put(fc, 12);
    text += idx_line("C", 3, off, 12, 6, 0, 0, 0, 0); off += 12;
    // row D: raw bf16 (kind 4) copied
    uint16_t rd[3] = {0x3F80, 0x4000, 0x4040};
    put(rd, 6);
    text += idx_line("D", 4, off, 6, 6, 0, 0, 0, 0); off += 6;
    WReader reader = [&](uint64_t o, uint64_t n, uint8_t* buf) { if (o + n > pack.size()) return false; std::memcpy(buf, pack.data() + o, (size_t)n); return true; };
    WeightTable t;
    std::vector<uint8_t> arena(4096, 0xCC);
    std::string err;
    CHECK(t.load_from(text, reader, arena.data(), arena.size(), {}, &err));
    const WIndexRow *a = t.find("A"), *b = t.find("B"), *c = t.find("C"), *d = t.find("D");
    CHECK(a && b && c && d);
    if (a && b && c && d) {
        CHECK(std::memcmp(&arena[a->arena_off], codes, 8) == 0);
        float w0, w1; std::memcpy(&w0, &arena[a->arena_off + 8], 4); std::memcpy(&w1, &arena[a->arena_off + 12], 4);
        CHECK(w0 == 1.5f && w1 == -0.25f);                                              // fp16 -> f32 is exact
        CHECK(std::memcmp(&arena[a->arena_off + 16], offs, 4) == 0);
        for (int i = 0; i < 3; ++i) { uint16_t h; std::memcpy(&h, &arena[b->arena_off + 2 * i], 2); CHECK(h == f16::float_high_half(fb[i])); }
        for (int i = 0; i < 3; ++i) { uint16_t h; std::memcpy(&h, &arena[c->arena_off + 2 * i], 2); CHECK(h == f16::float_to_half(fc[i])); }
        CHECK(std::memcmp(&arena[d->arena_off], rd, 6) == 0);
        CHECK(a->engine_kind == 2 && b->engine_kind == 1 && c->engine_kind == 3 && d->engine_kind == 1);
    }
    CHECK(t.pool_bytes() == 256 * 4);
    // arena too small -> rejected before any byte moves; bad totals -> rejected
    std::vector<uint8_t> small(t.pool_bytes() - 1, 0xCC);
    CHECK(!t.load_from(text, reader, small.data(), small.size(), {}, &err) && small[0] == 0xCC);
    CHECK(!t.load_from(idx_line("X", 2, 0, 15, 20, 8, 8, 4, 1), reader, arena.data(), arena.size(), {}, &err));
    CHECK(!t.load_from("bad line\n", reader, arena.data(), arena.size(), {}, &err));
    // skip set compacts the arena: skipped row's bytes are never read
    CHECK(t.load_from(text, reader, arena.data(), arena.size(), {"A", "C"}, &err) && t.pool_bytes() == 512 && t.find("B")->arena_off == 0 && t.find("D")->arena_off == 256);
    // chunked conversion never splits an element for any chunk size (including tiny ones)
    for (uint64_t ch : {1ull, 2ull, 3ull, 4ull, 5ull, 7ull, 8ull}) {
        std::vector<float> src(37); for (size_t i = 0; i < src.size(); ++i) src[i] = (float)i * 0.37f - 5.0f;
        std::vector<uint8_t> o(37 * 2, 0);
        WSeg s{0, 0, 37 * 4, 37 * 2, WConv::ToF16};
        WReader rr = [&](uint64_t of, uint64_t n, uint8_t* buf) { std::memcpy(buf, (const uint8_t*)src.data() + of, (size_t)n); return true; };
        CHECK(convert_segment(s, rr, o.data(), o.size(), ch) == 37 * 2);
        for (size_t i = 0; i < src.size(); ++i) { uint16_t h; std::memcpy(&h, &o[2 * i], 2); CHECK(h == f16::float_to_half(src[i])); }
        std::vector<uint16_t> hs(21); for (size_t i = 0; i < hs.size(); ++i) hs[i] = f16::float_to_half((float)i - 3);
        std::vector<uint8_t> w(21 * 4);
        WSeg s2{0, 0, 21 * 2, 21 * 4, WConv::WidenF16};
        WReader r2 = [&](uint64_t of, uint64_t n, uint8_t* buf) { std::memcpy(buf, (const uint8_t*)hs.data() + of, (size_t)n); return true; };
        CHECK(convert_segment(s2, r2, w.data(), w.size(), ch) == 21 * 4);
        for (size_t i = 0; i < hs.size(); ++i) { float f; std::memcpy(&f, &w[4 * i], 4); CHECK(f == (float)i - 3); }
    }
    // file based load
    std::string dir = "/tmp/baloon_u18_w_" + std::to_string(getpid());
    CHECK(system(("mkdir -p " + dir).c_str()) == 0);
    FILE* fi = std::fopen((dir + "/index.txt").c_str(), "wb"); FILE* fw = std::fopen((dir + "/weights.bin").c_str(), "wb");
    CHECK(fi && fw);
    if (fi && fw) { std::fwrite(text.data(), 1, text.size(), fi); std::fwrite(pack.data(), 1, pack.size(), fw); }
    if (fi) std::fclose(fi);
    if (fw) std::fclose(fw);
    std::vector<uint8_t> ar2(4096);
    WeightTable t2;
    CHECK(t2.load(dir, ar2.data(), ar2.size(), {}, &err) && t2.find("D") && std::memcmp(&ar2[t2.find("D")->arena_off], rd, 6) == 0);
    CHECK(!t2.load(dir + "_missing", ar2.data(), ar2.size(), {}, &err));
    CHECK(system(("rm -rf " + dir).c_str()) == 0);
}

void test_prefill() {
    testh::Rng rng(23);
    // chunks partition n exactly, each chunk in 1..T
    for (int it = 0; it < 3000; ++it) {
        PrefillConfig c; c.T = 1 + rng.below(9); c.n_layers = 1 + (int)rng.below(9);
        Prefill p(c);
        const uint64_t n = rng.below(80);
        std::vector<std::vector<int>> per_layer((size_t)c.n_layers);
        uint64_t cover = 0, expect_c0 = 0;
        PrefillHooks h;
        h.hc_read = [&](int l, uint64_t c0, uint64_t T) { per_layer[(size_t)l].push_back(0); CHECK(c0 % c.T == 0); CHECK(T >= 1 && T <= c.T && c0 + T <= n); };
        h.mixer = [&](int l, uint64_t, uint64_t) { CHECK(per_layer[(size_t)l].back() == 0); per_layer[(size_t)l].back() = 1; };
        h.moe = [&](int l, uint64_t, uint64_t) { CHECK(per_layer[(size_t)l].back() == 1); per_layer[(size_t)l].back() = 2; };
        h.write = [&](int l, uint64_t c0, uint64_t T) { CHECK(per_layer[(size_t)l].back() == 2); per_layer[(size_t)l].back() = 3; if (l == c.n_layers - 1) { CHECK(c0 == expect_c0); expect_c0 += T; cover += T; } };
        PrefillStats st;
        CHECK(p.run(n, h, &st));
        uint64_t sum = 0;
        for (uint64_t x : st.chunk_lens) { CHECK(x >= 1 && x <= c.T); sum += x; }
        CHECK(sum == n && cover == n && st.tokens == n && st.chunks == st.chunk_lens.size());
        CHECK(st.layer_calls == st.chunks * (uint64_t)c.n_layers && st.gdn_calls + st.qsa_calls == st.layer_calls);
        for (auto& v : per_layer) for (int s : v) CHECK(s == 3);
    }
    { PrefillConfig c; c.T = 0; Prefill p(c); CHECK(!p.run(5, PrefillHooks{})); }
    // bump allocator: 256-aligned, always 256 slack, fails exactly on overflow
    for (uint64_t b = 0; b < 2000; ++b) { const uint64_t r = round_bytes(b); CHECK(r % 256 == 0 && r >= b + 256 && r < b + 512); }
    for (int it = 0; it < 500; ++it) {
        PrefillConfig c; c.T = 1 + rng.below(8); c.n_embd = 1 + rng.below(100); c.hc = 1 + rng.below(3); c.attn_dim = rng.below(100);
        c.kv_dim = rng.below(30); c.n_experts = 1 + (uint32_t)rng.below(20); c.top_k = 1 + (uint32_t)rng.below(4);
        c.blob_bytes = rng.below(500); c.dq_bytes = rng.below(900); c.gemm_scratch = 1 + rng.below(3000); c.gemm_ws = 1 + rng.below(3000);
        Prefill p(c);
        uint64_t sum = 0;
        for (uint64_t b : p.alloc_sequence()) sum += round_bytes(b);
        const uint64_t need = p.bytes_needed();
        CHECK(need == sum);                                                    // counting mode replays the sequence exactly
        std::vector<uint8_t> region((size_t)need + 8);
        CHECK(p.init(region.data(), need) && p.used() == need && p.initialized());
        for (size_t i = 0; i < p.offsets().size(); ++i) { CHECK(p.offsets()[i] % 256 == 0); CHECK(p.offsets()[i] + p.alloc_sequence()[i] + 256 <= need); }
        CHECK(!p.init(region.data(), need - 1) && !p.initialized());          // one byte short: fails exactly
        CHECK(p.init(nullptr, 0) && p.used() == need);                         // owned mode
    }
    CHECK(MAXBLOB({3, 9, 2}) == 9 && MAXBLOB({}) == 0);
    // staging rings: slots in one lookahead window never alias
    for (uint64_t i = 0; i < 200; ++i) for (uint64_t k = i + 1; k < i + 8; ++k) CHECK(i % kPrefillStageRing != k % kPrefillStageRing);
    for (uint64_t j = 0; j < 200; ++j) CHECK(j % kPrefillDqRing != (j + 1) % kPrefillDqRing);
    CHECK(kPrefillLookahead < kPrefillStageRing && kPrefillStageRing == 8 && kPrefillDqRing == 2);
    // expert grouping: counting sort places every pair strictly inside its block, bijectively
    for (int it = 0; it < 3000; ++it) {
        const uint32_t E = 1 + (uint32_t)rng.below(40);
        std::vector<uint32_t> ex(rng.below(120));
        for (auto& e : ex) e = (uint32_t)rng.below(E);
        ExpertGrouping g;
        CHECK(group_experts(ex, E, g));
        CHECK(g.off.size() == E + 1 && g.off[0] == 0 && g.off[E] == ex.size() && g.row.size() == ex.size());
        std::vector<int> used(ex.size(), 0);
        for (size_t i = 0; i < ex.size(); ++i) {
            CHECK(g.row[i] >= g.off[ex[i]] && g.row[i] < g.off[ex[i] + 1]);
            CHECK(g.row[i] < ex.size() && ++used[g.row[i]] == 1);
        }
        for (uint32_t e = 0; e < E; ++e) CHECK(g.off[e] <= g.off[e + 1]);
        CHECK(g.order.size() == g.stage_slot.size() && g.order.size() == g.dq_slot.size());
        for (size_t j = 0; j < g.order.size(); ++j) { CHECK(g.stage_slot[j] == j % 8 && g.dq_slot[j] == j % 2); if (j) CHECK(g.order[j] > g.order[j - 1]); }
        // stable: pairs of one expert keep their order
        for (size_t i = 1; i < ex.size(); ++i) for (size_t j = 0; j < i; ++j) if (ex[i] == ex[j]) CHECK(g.row[j] < g.row[i]);
    }
    ExpertGrouping g;
    CHECK(!group_experts({5}, 5, g) && !group_experts({}, 0, g) && group_experts({}, 3, g) && g.off.size() == 4);
    // GDN state layout
    CHECK(gdn_floats_per_layer() == 817152 && 128ull * 48 * 128 + 10240ull * 3 == gdn_floats_per_layer());
    for (uint64_t idx = 0; idx < 36; ++idx) CHECK(idx * gdn_floats_per_layer() + 128ull * 48 * 128 < (idx + 1) * gdn_floats_per_layer());
    // reference projections
    uint16_t w[6] = {0x3F80, 0x4000, 0x4040, 0x4080, 0, 0xBF80};   // [[1,2,3],[4,0,-1]]
    float x[6] = {1, 1, 1, 2, 0, 1}, o[4], wf[6] = {1, 2, 3, 4, 0, -1};
    bf16_proj(w, x, 2, 2, 3, o);
    CHECK(o[0] == 6 && o[1] == 3 && o[2] == 5 && o[3] == 7);
    float o2[4];
    native_proj(wf, x, 2, 2, 3, o2);
    for (int i = 0; i < 4; ++i) CHECK(o2[i] == o[i]);
}

void test_serve() {
    using namespace baloon::serve_engine;
    for (uint64_t p = 0; p < 60; ++p) for (uint64_t m = 0; m < 60; ++m) for (uint64_t c : {0ull, 10ull, 59ull, 60ull, 100ull, 119ull}) {
        CHECK(admit(p, m, c) == (p + m <= c));
        if (admit(p, m, c)) CHECK(p + m <= c && p <= c);
        else CHECK(c < p + m);                                              // rejected, never truncated
    }
    // line protocol: at most max_new T lines, exactly one DONE, nothing after it
    testh::Rng rng(29);
    for (int it = 0; it < 2000; ++it) {
        const uint64_t mn = 1 + rng.below(8);
        Stream s(mn);
        uint64_t tok = 0, dones = 0;
        const int events = 1 + (int)rng.below(30);
        bool done_seen = false;
        for (int e = 0; e < events; ++e) {
            if (rng.below(6) == 0) { if (s.done()) { ++dones; done_seen = true; } }
            else if (auto l = s.token(e)) { ++tok; CHECK(!done_seen); CHECK(l->rfind("T ", 0) == 0); }
            CHECK(s.req().streamed <= mn);
        }
        CHECK(tok <= mn && dones <= 1);
    }
    Req r{0, 3, false};
    CHECK(on_token(on_done(r)).streamed == 0 && on_done(on_done(r)).done && on_done(on_done(r)).streamed == on_done(r).streamed);
    // image cache stays within 64, evicts the oldest first
    std::vector<int> cache;
    for (int i = 0; i < 300; ++i) {
        cache = insert_cache(cache, i);
        CHECK(cache.size() <= kVisionCacheMax && cache.back() == i);
        if (i >= 64) CHECK(cache.front() == i - 63);
    }
    CHECK(cache.size() == 64);
    // authorization
    CHECK(authorized(false, "k", "") && authorized(false, "k", "zzz"));
    CHECK(authorized(true, "k", "k") && !authorized(true, "k", "x") && !authorized(true, "k", ""));
    // finish reasons: tool_calls iff a call was made and generation stopped naturally
    for (uint64_t calls = 0; calls < 3; ++calls) for (int st = 0; st < 2; ++st) {
        Finish f = openai_finish(calls, st);
        CHECK((f == Finish::ToolCalls) == (st && calls > 0));
        if (!st) CHECK(f == Finish::Length);
        if (st && calls == 0) CHECK(f == Finish::Stop);
    }
    for (int u = 0; u < 2; ++u) for (int st = 0; st < 2; ++st) {
        Stop s = anthropic_stop(u, st);
        CHECK((s == Stop::ToolUse) == (u && st));
        if (!st) CHECK(s == Stop::MaxTokens);
        if (st && !u) CHECK(s == Stop::EndTurn);
    }
    CHECK(std::string(finish_name(Finish::ToolCalls)) == "tool_calls" && std::string(stop_name(Stop::ToolUse)) == "tool_use");
    CHECK(total_tokens(7, 5) == 12);
    // GEN line parsing
    GenLine g = parse_gen("GEN 4 1 2 3");
    CHECK(g.ok && g.max_new == 4 && g.tokens == std::vector<int64_t>({1, 2, 3}));
    CHECK(!parse_gen("GEN 0 1").ok && !parse_gen("GEN 4").ok && !parse_gen("GEN 4 x").ok && !parse_gen("HELLO 4 1").ok && !parse_gen("GEN 4 1 2x").ok);
    CHECK(fmt_token(5) == "T 5" && fmt_done() == "DONE" && fmt_err("x") == "ERR x");
}
