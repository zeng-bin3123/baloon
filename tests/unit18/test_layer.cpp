// LayerComposition invariants.
#include <cmath>
#include <cstring>
#include <vector>
#include "baloon/core_layer.hpp"
#include "check.hpp"

using namespace baloon::core;

void test_layer_align() {
    for (uint64_t n = 0; n < 5000; ++n) {
        const uint64_t a = align_up16(n);
        CHECK(a >= n && a % 16 == 0 && a < n + 16 && align_up16(a) == a);
    }
    for (uint64_t k = 0; k < 200; ++k) CHECK(q8k_bytes(256 * k) == 292 * k && q8k_bytes(256 * (k + 1)) == q8k_bytes(256 * k) + 292);
    CHECK(Q8K_BYTES_PER_BLOCK == 292 && Q8K_ELEMS_PER_BLOCK == 256);
    // cursor: covers the region, aligned, monotone
    testh::Rng rng(3);
    Cursor c;
    uint64_t prev = 0;
    for (int i = 0; i < 2000; ++i) {
        const uint64_t cnt = rng.below(50), sz = rng.below(40);
        const uint64_t at = c.take(cnt, sz);
        CHECK(at == prev && at % 16 == 0);
        CHECK(at + cnt * sz <= c.used && c.used % 16 == 0 && c.used >= prev && c.used < at + cnt * sz + 16);
        prev = c.used;
    }
    Cursor lim; lim.cap = 64;
    lim.take(4, 16); CHECK(!lim.failed);
    lim.take(1, 1); CHECK(lim.failed);
    // every *_buffers_bytes total equals the aligned sum and init carves aligned, disjoint regions inside it
    for (int it = 0; it < 300; ++it) {
        BlockDims d{1 + rng.below(4), 1 + rng.below(300), 1 + rng.below(4), rng.below(9), rng.below(65), rng.below(5)};
        const uint64_t bytes = block_buffers_bytes(d);
        CHECK(bytes % 16 == 0);
        std::vector<uint64_t> parts = {d.T * d.n_embd * 4, d.T * d.hc * d.n_embd * 4, d.T * d.n_head * d.head_dim * 4,
                                       d.T * 2 * d.n_head_kv * d.head_dim * 4, d.T * d.n_embd * 4, d.T * q8k_bytes(d.n_embd)};
        uint64_t want = 0, plain = 0;
        for (uint64_t p : parts) { want += align_up16(p); plain += p; }
        CHECK(bytes == want && bytes >= plain);
        std::vector<uint8_t> raw(bytes + 32);
        uint8_t* base = raw.data() + (16 - reinterpret_cast<uintptr_t>(raw.data()) % 16) % 16;
        BlockBuffers b;
        CHECK(block_buffers_init(b, base, bytes, d) && b.bytes == bytes);
        for (int i = 0; i < kBlockRegions; ++i) {
            CHECK(b.off[i] % 16 == 0 && b.off[i] + b.size[i] <= bytes);
            if (i + 1 < kBlockRegions) CHECK(b.off[i] + b.size[i] <= b.off[i + 1]);
        }
        BlockBuffers small;
        if (bytes > 0) CHECK(!block_buffers_init(small, base, bytes - 1, d));
        CHECK(!block_buffers_init(small, nullptr, bytes, d));
    }
    GdnDims gd{2, 100, 50, 4, 64};
    MoeDims md{2, 64, 16, 4};
    QsaDims qd{2, 4, 8, 2, 128};
    QsaStateDims sd{256, 2, 8, 64};
    alignas(16) static uint8_t arena[1 << 20];
    GdnBuffers g; MoEBuffers m; QsaBuffers q; QsaState s;
    CHECK(gdn_buffers_init(g, arena, sizeof arena, gd) && g.bytes == gdn_buffers_bytes(gd) && g.bytes % 16 == 0);
    CHECK(moe_buffers_init(m, arena, sizeof arena, md) && m.bytes == moe_buffers_bytes(md) && m.bytes % 16 == 0);
    CHECK(qsa_buffers_init(q, arena, sizeof arena, qd) && q.bytes == qsa_buffers_bytes(qd) && q.bytes % 16 == 0);
    qsa_set_kv_int8(false);
    const uint64_t f32_bytes = qsa_state_bytes(sd);
    qsa_set_kv_int8(true);
    CHECK(qsa_kv_int8());
    const uint64_t i8_bytes = qsa_state_bytes(sd);
    CHECK(i8_bytes < f32_bytes);                                   // int8 KV is smaller than fp32 KV
    CHECK(qsa_state_init(s, arena, sizeof arena, sd) && s.bytes == i8_bytes);
    std::memset(arena, 0x5A, 4096);
    qsa_state_zero(s);
    for (uint64_t i = 0; i < s.bytes; ++i) CHECK(arena[i] == 0);
    qsa_set_kv_int8(false);
    float st[16]; for (float& v : st) v = 3.0f;
    gdn_buffers_zero_state(st, 16);
    for (float v : st) CHECK(v == 0.0f);
}

void test_layer_planes() {
    {
        QsaShapes q = qsa_shapes(16, 128, 2);
        CHECK(q.ok && q.q_dim == 2048 && q.kv_dim == 256 && q.q_full == 4096 && q.group == 8);
        CHECK(!qsa_shapes(16, 128, 3).ok && !qsa_shapes(0, 128, 1).ok && !qsa_shapes(4, 0, 1).ok && !qsa_shapes(4, 8, 0).ok);
    }
    testh::Rng rng(11);
    int accepted = 0;
    for (int it = 0; it < 20000; ++it) {
        WeightRef w;
        const int bits[] = {0, 2, 3, 4, 8};
        w.code_bits = bits[rng.below(5)];
        w.codes = rng.below(3) ? 1 + rng.below(100) : 0;
        w.scales = rng.below(3) ? 1 + rng.below(50) : 0;
        w.offset = rng.below(2) ? rng.below(20) : 0;
        w.bytes = rng.below(4) ? w.codes + w.scales + w.offset : rng.below(300);
        w.has_offset = rng.below(2);
        Planes p = plane_ptrs(w);
        const bool want = (w.code_bits == 2 || w.code_bits == 4 || w.code_bits == 8) && w.codes != 0 && w.scales != 0 &&
                          w.codes + w.scales + w.offset == w.bytes && (w.has_offset == (w.offset != 0));
        CHECK(p.ok == want);
        if (p.ok) {
            ++accepted;
            CHECK(p.scales_start + w.scales + w.offset == w.bytes);          // planes partition the tensor
            CHECK(p.offset_start + w.offset == w.bytes);
            if (!w.has_offset) CHECK(w.codes + w.scales == w.bytes);
        }
    }
    CHECK(accepted > 100);
    // dispatch rule
    for (int n = 0; n < 2; ++n) for (int b : {2, 4, 8}) for (int q = 0; q < 2; ++q) {
        WeightRef w; w.native = n; w.code_bits = b; w.wants_q8k = q;
        Gemv g = sform_of(w);
        if (n) CHECK(g == Gemv::Native);
        else if (b == 2) CHECK(g == Gemv::S2Q80);
        else if (q) CHECK(g == Gemv::Q8k);
        else CHECK(g == Gemv::LegacyQ80);
        if (g == Gemv::Q8k) CHECK(b != 2);
    }
    WeightRef w; w.code_bits = 4; w.codes = 8; w.scales = 2; w.offset = 0; w.bytes = 10; w.wants_q8k = true;
    int hits[4] = {0, 0, 0, 0};
    GemvExec ex;
    ex.native = [&](const Planes&) { ++hits[0]; };
    ex.s2q80 = [&](const Planes&) { ++hits[1]; };
    ex.legacy_q80 = [&](const Planes&) { ++hits[2]; };
    ex.q8k = [&](const Planes& p) { ++hits[3]; CHECK(p.scales_start == 8); };
    Gemv chosen = Gemv::Native;
    CHECK(gemv_quantized(w, ex, &chosen) && chosen == Gemv::Q8k && hits[3] == 1 && hits[0] + hits[1] + hits[2] == 0);
    w.bytes = 11;
    CHECK(!gemv_quantized(w, ex) && hits[3] == 1);                          // bad planes: nothing runs
    w.bytes = 10; ex.q8k = nullptr;
    CHECK(!gemv_quantized(w, ex));                                          // missing executor
}

void test_layer_stages() {
    // runStage: default runs all 5; half 1 = stages 0..2; half 2 = 3..4; positive prefix k runs 0..k-1 regardless of half
    for (int i = 0; i < 5; ++i) {
        CHECK(stage_runs(0, 0, i));
        CHECK(stage_runs(0, 1, i) == (i < 3));
        CHECK(stage_runs(0, 2, i) == (i >= 3));
        for (int k = 1; k <= 6; ++k) for (int half = 0; half < 3; ++half) CHECK(stage_runs(k, half, i) == (i < k));
        for (int half = 0; half < 3; ++half) CHECK(stage_runs(5, half, i));
    }
    CHECK(!stage_runs(0, 0, 5) && !stage_runs(0, 0, -1));
    // block_layer_pre executes exactly the selected stages, in order
    for (int pfx = 0; pfx <= 5; ++pfx) for (int half = 0; half < 3; ++half) {
        std::vector<int> order;
        BlockHooks h;
        for (int i = 0; i < 5; ++i) h.stages.stage[i] = [&order, i](int) { order.push_back(i); };
        BlockCall c; c.layer = 7; c.stage_prefix = pfx; c.half = half;
        BlockResult r = block_layer_pre(c, h);
        uint32_t mask = 0;
        for (int i = 0; i < 5; ++i) if (stage_runs(pfx, half, i)) mask |= 1u << i;
        CHECK(r.stage_mask == mask);
        for (size_t j = 1; j < order.size(); ++j) CHECK(order[j - 1] < order[j]);
        CHECK(order.size() == (size_t)__builtin_popcount(mask));
    }
    // fused hyper-connection bookkeeping and post write-back
    layer_set_fused_gr(true);
    CHECK(layer_fused_gr(0, 0, true) && !layer_fused_gr(1, 0, true) && !layer_fused_gr(0, 1, true) && !layer_fused_gr(0, 0, false));
    for (int layer = 0; layer < 48; ++layer) {
        int writes = 0;
        BlockHooks h; h.gr_write = [&] { ++writes; };
        BlockCall c; c.layer = layer; c.n_layers = 48;
        BlockResult r = block_layer(c, h);
        CHECK(r.fused);
        CHECK(r.wrote_r == (layer == 47) && writes == (layer == 47 ? 1 : 0));   // inner layers defer, the last materialises
    }
    layer_set_fused_gr(false);
    for (int layer = 0; layer < 48; ++layer) {
        int writes = 0;
        BlockHooks h; h.gr_write = [&] { ++writes; };
        BlockCall c; c.layer = layer;
        block_layer(c, h);
        CHECK(writes == 1);                                                     // unfused: gr_write always runs
    }
    {   // half = 1 (mixer only) has no post write
        int writes = 0; BlockHooks h; h.gr_write = [&] { ++writes; };
        BlockCall c; c.layer = 3; c.half = 1;
        block_layer(c, h); CHECK(writes == 0);
    }
    // PLE: only layer 1, only the mixer-side halves
    for (int layer = 0; layer < 6; ++layer) for (int half = 0; half < 3; ++half) {
        int ples = 0;
        BlockHooks h; h.ple = [&] { ++ples; };
        BlockCall c; c.layer = layer; c.half = half; c.ple_ready = true;
        BlockResult r = block_layer_pre(c, h);
        const bool want = layer == 1 && (half == 0 || half == 1);
        CHECK(ple_applies(true, layer, half) == want && r.ple_ran == want && ples == (want ? 1 : 0));
        CHECK(!ple_applies(false, layer, half));
    }
    // 48 layers split 12 QSA / 36 GDN; gdn_layer / qsa_layer enforce the kind
    int nq = 0;
    for (int l = 0; l < 48; ++l) {
        nq += is_qsa_layer(l);
        BlockCall c; c.layer = l; BlockHooks h;
        CHECK(gdn_layer(c, h) == !is_qsa_layer(l));
        CHECK(qsa_layer(c, h) == is_qsa_layer(l));
    }
    CHECK(nq == 12);
}

void test_layer_flags() {
    // verify-window compatibility == conjunction of the flags with the short flash adapter OFF
    for (int m = 0; m < 128; ++m) {
        const bool nb = m & 1, gr = m & 2, gf = m & 4, fa = m & 8, fl = m & 16, fs = m & 32, qi = m & 64;
        layer_set_native_bf16(nb); layer_set_fused_gr(gr); layer_set_fused_gdn(gf); layer_set_fast_attn(fa);
        layer_set_native_flash_attn_short(fl); layer_set_fast_select(fs);
        for (int ng = 0; ng < 2; ++ng) {
            const bool want = nb && gr && gf && ng && fa && !fl && fs && qi;
            CHECK(layer_verify_compatible(ng, qi) == want);
        }
    }
    layer_set_native_bf16(false); layer_set_fused_gr(false); layer_set_fused_gdn(false); layer_set_fast_attn(false);
    layer_set_native_flash_attn_short(false); layer_set_fast_select(false);
    layer_set_publish_kernel(true); CHECK(layer_publish_kernel()); layer_set_publish_kernel(false); CHECK(!layer_publish_kernel());
    CHECK(!layer_shared_early());
    layer_set_shared_early(true); CHECK(layer_shared_early()); layer_set_shared_early(false); CHECK(!layer_shared_early());
    // dump layout
    testh::Rng rng(5);
    for (int it = 0; it < 500; ++it) {
        const uint64_t n = 1 + rng.below(300), hc = 1 + rng.below(5), a = rng.below(400), nvk = rng.below(80);
        const uint64_t stride = dump_stride_floats(n, hc, a, nvk);
        CHECK(stride == 2 * n + 2 * hc + a + 5 * nvk + 8);
        uint64_t end = 0;
        for (int i = 0; i < kDumpSlots; ++i) {
            DumpSlot s = dump_slot(i, n, hc, a, nvk);
            CHECK(s.off == end);                                           // slots are contiguous
            end = s.off + s.len;
            CHECK(end <= stride);
        }
        CHECK(end == stride);                                              // the last KV dump ends exactly at the stride
        DumpSlot last = dump_slot(kDumpSlots - 1, n, hc, a, nvk);
        CHECK(last.off + last.len == stride);
        CHECK(dump_slot(kDumpSlots, n, hc, a, nvk).len == 0);
    }
}

void test_layer_moe_misc() {
    testh::Rng rng(21);
    for (int it = 0; it < 2000; ++it) {
        const int E = 2 + (int)rng.below(30), k = 1 + (int)rng.below((uint32_t)E);
        std::vector<float> lg((size_t)E);
        for (auto& v : lg) v = (float)(int)rng.below(9) * 0.5f;           // many ties
        std::vector<int32_t> idx((size_t)k); std::vector<float> w((size_t)k);
        CHECK(moe_route(lg.data(), E, k, idx.data(), w.data()));
        double sum = 0;
        for (int i = 0; i < k; ++i) { sum += w[(size_t)i]; CHECK(w[(size_t)i] > 0); if (i) CHECK(w[(size_t)i] <= w[(size_t)i - 1] + 1e-6f); }
        CHECK(std::fabs(sum - 1.0) < 1e-4);
        std::vector<int> seen((size_t)E, 0);
        for (int i = 0; i < k; ++i) { CHECK(idx[(size_t)i] >= 0 && idx[(size_t)i] < E); ++seen[(size_t)idx[(size_t)i]]; }
        for (int c : seen) CHECK(c <= 1);
        for (int e = 0; e < E; ++e) if (!seen[(size_t)e]) CHECK(lg[(size_t)e] <= lg[(size_t)idx[(size_t)k - 1]]);   // nothing better was left out
        for (int i = 1; i < k; ++i) CHECK(lg[(size_t)idx[(size_t)i]] <= lg[(size_t)idx[(size_t)i - 1]]);
        for (int i = 1; i < k; ++i) if (lg[(size_t)idx[(size_t)i]] == lg[(size_t)idx[(size_t)i - 1]]) CHECK(idx[(size_t)i] > idx[(size_t)i - 1]);   // lower index first on ties
    }
    float lg[3] = {0, 0, 0}; int32_t id[3]; float wt[3];
    CHECK(!moe_route(lg, 3, 4, id, wt) && !moe_route(lg, 3, 0, id, wt) && !moe_route(nullptr, 3, 1, id, wt));
    // combine / shared / finish / layer
    const int n = 6;
    float shared[6], parts[3 * 6], out[6], resid[6];
    for (int j = 0; j < n; ++j) { shared[j] = (float)j + 1; resid[j] = 0.25f * (float)j; for (int p = 0; p < 3; ++p) parts[p * n + j] = (float)(p + 1) * (float)(j + 1); }
    float cw[3] = {0.5f, 0.25f, 0.25f};
    moe_combine_parts(parts, cw, 3, n, out);
    for (int j = 0; j < n; ++j) CHECK(std::fabs(out[j] - (float)(j + 1) * (0.5f + 0.5f + 0.75f)) < 1e-5f);
    moe_shared(shared, 0.0f, n, out);
    for (int j = 0; j < n; ++j) CHECK(std::fabs(out[j] - 0.5f * shared[j]) < 1e-6f);
    float moe[6]; for (int j = 0; j < n; ++j) moe[j] = 1.0f;
    moe_finish(resid, moe, shared, n, out);
    for (int j = 0; j < n; ++j) CHECK(out[j] == resid[j] + 1.0f + shared[j]);
    float logits[4] = {3.0f, 1.0f, 2.0f, 0.0f}, lay[6];
    int called = 0;
    CHECK(moe_layer(logits, 4, 2, [&](int e, float* o) { ++called; for (int j = 0; j < n; ++j) o[j] = (float)e; }, shared, 0.0f, resid, n, lay));
    CHECK(called == 2);
    CHECK(!moe_layer(logits, 4, 9, [&](int, float*) {}, shared, 0.0f, resid, n, lay));
    // embedding / head
    std::vector<float> table(5 * 4); for (size_t i = 0; i < table.size(); ++i) table[i] = (float)i;
    float row[4];
    CHECK(embed_row(table.data(), 5, 4, 3, row) && row[0] == 12 && row[3] == 15);
    CHECK(!embed_row(table.data(), 5, 4, 5, row) && !embed_row(table.data(), 5, 4, -1, row));
    float x[4] = {1, 0, 0, 2}, lgt[5];
    lm_head(table.data(), x, 5, 4, lgt);
    for (int v = 0; v < 5; ++v) CHECK(lgt[v] == table[(size_t)v * 4] + 2 * table[(size_t)v * 4 + 3]);
    float rs[8] = {1, 0, 0, 2, 0, 0, 0, 0}, mixw[2] = {1.0f, 5.0f}, lgm[5];
    lm_head_mix(table.data(), rs, mixw, 2, 5, 4, lgm);
    for (int v = 0; v < 5; ++v) CHECK(lgm[v] == lgt[v]);
    uint16_t bw[8] = {0x3F80, 0x4000, 0, 0xBF80, 0x4040, 0, 0, 0x3F80};     // 1, 2, 0, -1 | 3, 0, 0, 1
    float bx[4] = {1, 1, 1, 1}, bo[2];
    project_bf16(bw, bx, 2, 4, bo);
    CHECK(bo[0] == 2.0f && bo[1] == 4.0f);
    // PLE state machine issue -> stage -> finish
    PleRun r;
    float res[4] = {1, 1, 1, 1};
    CHECK(!ple_stage_token(r, table.data(), 5, 4) && !ple_finish_token(r, res, 1.0f));
    CHECK(ple_issue_token(r, 2) && !ple_issue_token(r, 3));
    CHECK(ple_stage_token(r, table.data(), 5, 4) && r.state == PleRun::Staged);
    CHECK(ple_finish_token(r, res, 0.5f) && res[0] == 1 + 0.5f * 8 && r.state == PleRun::Done);
    CHECK(ple_issue_token(r, 0));
    CHECK(!ple_stage_token(r, table.data(), 5, 0));
    CHECK(ple_run_scratch_bytes(4, 3) % 16 == 0 && ple_run_scratch_bytes(4, 3) >= 3 * 16 + 3 * 8);
    // doorbell: monotone ring
    Doorbell d;
    CHECK(doorbell_value(&d) == 0);
    doorbell_init(&d);
    doorbell_raise(&d, 5); doorbell_raise(&d, 3);
    CHECK(doorbell_value(&d) == 5);
    doorbell_reset(&d); CHECK(doorbell_value(&d) == 0);
    doorbell_free(&d); CHECK(d.word == nullptr);
    doorbell_free(&d);
    // stage timers with a fake clock
    static double now = 0;
    stage_timing_set_clock([]() { return now; });
    stage_timing_reset();
    stage_mark_begin(1); now = 5; stage_mark_end(1);
    CHECK(stage_timing_state().calls[1] == 0);                             // disabled: nothing recorded
    stage_timing_enable(true);
    now = 10; stage_mark_begin(2); now = 14; stage_mark_end(2);
    now = 20; stage_mark_begin(2); now = 21; stage_mark_end(2);
    stage_mark_end(3);                                                    // end without begin is ignored
    CHECK(stage_timing_state().calls[2] == 2 && std::fabs(stage_timing_state().total_ms[2] - 5.0) < 1e-12);
    CHECK(stage_timing_state().calls[3] == 0);
    stage_mark_begin(99); stage_mark_end(99);
    CHECK(std::string(stage_timing_name(2)) == "mixer_b" && std::string(stage_timing_name(99)) == "?");
    CHECK(stage_timing_report().find("calls=2") != std::string::npos);
    stage_timing_enable(false); stage_timing_set_clock(nullptr); stage_timing_reset();
    uint16_t hb[3]; float hf[3] = {1.0f, -2.0f, 0.5f};
    dump_half(hf, 3, hb);
    CHECK(hb[0] == 0x3C00 && hb[1] == 0xC000 && hb[2] == 0x3800);
}
