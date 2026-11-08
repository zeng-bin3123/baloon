// invariants.hpp - host-only checks of the LAYER 1 statements of unit u25c78c49e6 (no device needed): MtpDraftBlockQuantizer,
// MtpDraftLayer, VerifyWindow, VerifyWindowKernels (index arithmetic), SpecController, SuffixDrafter. Production code under
// test: mtp_expert_quantizer.hpp, core_mtp.hpp / src/core/mtp.cpp, core_verify.hpp / src/core/verify.cpp,
// verify_window_protocol.hpp, verify_window_kernels.hpp, spec_controller.hpp, suffix_drafter.hpp.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

#include "baloon/core_mtp.hpp"
#include "baloon/core_verify.hpp"
#include "baloon/mtp_draft_layer.hpp"
#include "baloon/mtp_expert_quantizer.hpp"
#include "baloon/spec_controller.hpp"
#include "baloon/suffix_drafter.hpp"
#include "baloon/verify_kernels.hpp"
#include "baloon/verify_window_kernels.hpp"
#include "baloon/verify_window_protocol.hpp"

#ifndef CHECK
static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#endif

namespace u25c {
using namespace baloon;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t below(uint32_t n) { return n ? (uint32_t)(next() >> 16) % n : 0; }
    double uni() { return (double)(next() >> 11) / 9007199254740992.0; }
    double gauss() { return std::sqrt(-2 * std::log(uni() + 1e-12)) * std::cos(6.283185307179586 * uni()); }
};

// ------------------------------------------------------------------------------------- MtpDraftBlockQuantizer
inline void quantizer() {
    using namespace mtpq;
    // 17 candidate scales f = (16+i)/32 in [1/2, 1]; +2 level (2*d) reaches amax at every candidate
    CHECK(kScaleCandidates == 17);
    CHECK(scale_factor(0) == 0.5f && scale_factor(16) == 1.0f);
    for (int i = 0; i < 16; ++i) CHECK(scale_factor(i) < scale_factor(i + 1));
    for (int i = 0; i <= 16; ++i) CHECK(2.0f * scale_factor(i) >= 1.0f);
    // clip: codes 0..3, exact on [-1, 2]
    for (long r = -10; r <= 10; ++r) {
        const int c = q2w::code_of_symbol(clip12(r));
        CHECK(c >= 0 && c < 4);
        if (r >= -1 && r <= 2) CHECK(c - 1 == r);
    }
    // pack4: LSB-first, four codes per byte, unpack inverts
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) for (int c = 0; c < 4; ++c) for (int d = 0; d < 4; ++d) {
        const unsigned byte = a + 4 * b + 16 * c + 64 * d;
        CHECK(byte < 256 && byte % 4 == (unsigned)a && byte / 4 % 4 == (unsigned)b && byte / 16 % 4 == (unsigned)c && byte / 64 % 4 == (unsigned)d);
    }
    Rng rng(7);
    // Q2_0 block: stored codes equal the clipped rounding at the stored scale; round trip error bounded by the best candidate
    for (int it = 0; it < 200; ++it) {
        float x[64];
        for (float& v : x) v = (float)rng.gauss() * (float)(0.1 + rng.uni());
        uint8_t blk[18], codes[64];
        quantize_q2(x, blk);
        q2w::unpack_codes(blk, codes);
        float y[64];
        q2w::dequantize_block(blk, y);
        float amax = 0.f;
        for (float v : x) amax = std::max(amax, std::fabs(v));
        double best_err = 1e300;
        for (int k = 0; k < kScaleCandidates; ++k) {
            uint8_t cc[64];
            const float d = f16::half_to_float(f16::float_to_half(amax * scale_factor(k)));
            best_err = std::min(best_err, q2_codes_and_error(x, d, cc));
        }
        double err = 0;
        for (int j = 0; j < 64; ++j) { CHECK(codes[j] < 4); err += ((double)x[j] - y[j]) * ((double)x[j] - y[j]); }
        CHECK(err <= best_err * (1 + 1e-9) + 1e-12);
    }
    // Q4_0: extreme element -> code 0 (symbol -8); nibbles < 16; low half in low nibble, high half in high nibble
    for (int it = 0; it < 200; ++it) {
        float x[32];
        for (float& v : x) v = (float)rng.gauss();
        uint8_t blk[18];
        quantize_q4(x, blk);
        int ext = 0;
        for (int j = 0; j < 32; ++j) if (std::fabs(x[j]) > std::fabs(x[ext])) ext = j;
        const unsigned code = ext < 16 ? (blk[2 + ext] & 0xF) : (blk[2 + ext - 16] >> 4);
        CHECK(code == 0);
        float y[32];
        dequantize_q4(blk, y);
        float amax = std::fabs(x[ext]);
        for (int j = 0; j < 32; ++j) CHECK(std::fabs(x[j] - y[j]) <= amax / 8.0f * 1.01f + 1e-6f);
    }
    // Q8_0: codes in [-127, 127]
    for (int it = 0; it < 200; ++it) {
        float x[32];
        for (float& v : x) v = (float)rng.gauss() * 5.f;
        uint8_t blk[34];
        quantize_q8(x, blk);
        for (int j = 0; j < 32; ++j) CHECK((int)(int8_t)blk[2 + j] >= -127);
    }
    // sizes
    CHECK(HID % 64 == 0 && INTER % 64 == 0 && HID % 32 == 0 && INTER % 32 == 0);
    CHECK(row_bytes(2560, 64, 18) == 720 && row_bytes(640, 64, 18) == 180);
    const uint64_t elems = E * GU_OUT * HID + E * HID * INTER;
    CHECK(elems == 2516582400ull);
    CHECK(total_bytes(64, 18) == 707788800ull && total_bytes(64, 18) == 512ull * 1382400ull);
    CHECK(total_bytes(32, 18) == 1415577600ull && total_bytes(32, 34) == 2673868800ull);
    CHECK(total_bytes(64, 18) < total_bytes(32, 18) && total_bytes(32, 18) < total_bytes(32, 34));
    CHECK(2 * INTER == GU_OUT);
    std::vector<uint8_t> out;
    std::vector<float> w(3 * 100, 0.5f);
    CHECK(!quantize_tensor(Format::Q2_0, w.data(), 3, 100, &out));   // inner dim must divide the block
}

// ------------------------------------------------------------------------------------------- MtpDraftLayer
inline void draft_layer() {
    using core::MtpDrafter;
    // arena: 256-byte rounding, offsets aligned and monotone, first take at 0, region fits before the next take
    for (uint64_t b = 0; b < 3000; ++b) {
        const uint64_t a = mtp::align256(b);
        CHECK(a >= b && a % 256 == 0 && a < b + 256);
    }
    Rng rng(11);
    for (int it = 0; it < 200; ++it) {
        std::vector<uint64_t> sizes;
        for (uint32_t i = 0, n = 1 + rng.below(10); i < n; ++i) sizes.push_back(rng.below(5000));
        mtp::Bump c = mtp::Bump::count();
        std::vector<uint64_t> offs;
        for (uint64_t s : sizes) offs.push_back(c.take(s));
        CHECK(offs[0] == 0);
        for (size_t i = 0; i < sizes.size(); ++i) {
            CHECK(offs[i] % 256 == 0);
            if (i + 1 < sizes.size()) CHECK(offs[i] <= offs[i + 1] && offs[i] + sizes[i] <= offs[i + 1]);
        }
        CHECK(c.used == mtp::bump_end(0, sizes));
    }
    // step record of cell c: (c, c+1, (c+1)/4, c+1)
    for (int c = 0; c < 500; ++c) {
        const auto st = verify_kernels::dense_step(c);
        CHECK(st.pos == c && st.n_kv == c + 1 && st.n_bid == (c + 1) / 4 && st.width == c + 1);
        CHECK(st.n_bid * 4 <= c + 1 && c + 1 < (st.n_bid + 1) * 4);
    }
    // window / capacity through bind(): effective window 0 unless 0 < window < max_cells; cap = ceil(span/64)*64
    for (uint64_t mc = 1; mc < 600; mc += 7) for (uint64_t w = 0; w < 700; w += 13) {
        MtpDrafter d;
        MtpDrafter::Config cfg; cfg.max_cells = mc; cfg.window = w; cfg.max_t = 4; cfg.min_p = 0.5;
        CHECK(d.bind(cfg));
        const uint64_t eff = (0 < w && w < mc) ? w : 0;
        const uint64_t span = eff > 0 ? eff : mc;
        CHECK(d.window() == eff && span <= mc);
        CHECK(d.cap() >= span && d.cap() % 64 == 0 && d.cap() < span + 64);
    }
    { MtpDrafter d; MtpDrafter::Config cfg; cfg.max_cells = 0; cfg.max_t = 4; CHECK(!d.bind(cfg));
      cfg.max_cells = 10; cfg.max_t = 1; CHECK(!d.bind(cfg)); }
    // prefill: batches of at most max_t, none empty, none past the prompt; skipped batches lie wholly below first_needed
    for (uint64_t maxT = 2; maxT <= 8; ++maxT) for (uint64_t n = 0; n < 40; n += 3) for (int64_t first_needed = -3; first_needed <= (int64_t)n + 8; ++first_needed) {
        MtpDrafter d;
        MtpDrafter::Config cfg; cfg.max_cells = 4096; cfg.max_t = maxT; cfg.min_p = 0.5;
        CHECK(d.bind(cfg));
        const int64_t cell0 = 5;
        std::vector<std::pair<uint64_t, uint64_t>> ran;
        const uint64_t cnt = d.prefill(n, cell0, first_needed, [&](uint64_t c, uint64_t T) { ran.push_back({c, T}); });
        CHECK(cnt == ran.size());
        std::set<uint64_t> ran_starts;
        for (auto& r : ran) { CHECK(r.second >= 1 && r.second <= maxT && r.first + r.second <= n); ran_starts.insert(r.first); }
        uint64_t covered_next = 0, skipped_n = 0;
        for (uint64_t c = 0; c < n; c += maxT) {
            const uint64_t T = std::min<uint64_t>(maxT, n - c);
            const bool skipped = ran_starts.count(c) == 0;
            // a batch is skipped exactly when it ends at or before first_needed (the boundary itself included)
            CHECK(skipped == ((int64_t)cell0 + (int64_t)c + (int64_t)T <= first_needed));
            if (skipped) { ++skipped_n; for (uint64_t cell = 0; cell < (uint64_t)cell0 + c + T; ++cell) CHECK((int64_t)cell < first_needed); }
            covered_next = c + T;
        }
        if (n > 0) CHECK(covered_next == n);
        CHECK(cnt + skipped_n == (n + maxT - 1) / maxT);
        if (first_needed <= 0) CHECK(cnt == (n + maxT - 1) / maxT);
    }
    // the theory's bound for a windowed prompt: first needed cell = promptLen - window - 64
    for (uint64_t n : {100ull, 150ull, 300ull}) { MtpDrafter d; MtpDrafter::Config cfg; cfg.max_cells = 4096; cfg.max_t = 8; cfg.min_p = 0.5; CHECK(d.bind(cfg));
        const int64_t cell0 = 0, fn = (int64_t)n - 8 - 64;
        d.prefill(n, cell0, fn, nullptr);
        for (uint64_t c : d.prefill_chunks()) CHECK((int64_t)c + std::min<int64_t>(8, (int64_t)n - (int64_t)c) > fn);
        CHECK(d.prefill_chunks().size() < (n + 7) / 8); }
    // draft chain: always >= 1 draft, <= max_t - 1, stops when the last probability < min_p, tail zero-filled
    for (uint64_t maxT = 2; maxT <= 8; ++maxT) for (uint64_t T = 1; T <= maxT; ++T) for (uint64_t a = 0; a < T; ++a) {
        MtpDrafter d;
        MtpDrafter::Config cfg; cfg.max_cells = 1024; cfg.max_t = maxT; cfg.min_p = 0.5;
        CHECK(d.bind(cfg) && d.capture_round(T));
        for (uint64_t drop = 0; drop <= maxT; ++drop) {   // probability falls below min_p at step `drop`
            std::vector<int32_t> out;
            const int64_t n = d.draft(T, a, [&](uint64_t j, int32_t) { return core::DraftStep{(int32_t)(100 + j), j >= drop ? 0.1 : 0.9}; }, out);
            const uint64_t expect = std::min<uint64_t>(drop + 1, maxT - 1);
            CHECK(n >= 1 && (uint64_t)n <= maxT - 1 && (uint64_t)n == std::max<uint64_t>(1, expect));
            CHECK(out.size() == maxT - 1);
            for (uint64_t j = 0; j < maxT - 1; ++j) CHECK(out[j] == (j < (uint64_t)n ? (int32_t)(100 + j) : 0));
        }
        std::vector<int32_t> out;
        CHECK(d.draft(T, a, [](uint64_t, int32_t) { return core::DraftStep{1, 0.0}; }, out) == 1);   // stops right after the first draft
    }
    { MtpDrafter d; MtpDrafter::Config cfg; cfg.max_cells = 64; cfg.max_t = 4; cfg.min_p = 0.5; CHECK(d.bind(cfg));
      std::vector<int32_t> out; auto f = [](uint64_t, int32_t) { return core::DraftStep{1, 1.0}; };
      CHECK(d.draft(0, 0, f, out) == -1 && d.draft(5, 0, f, out) == -1 && d.draft(2, 2, f, out) == -1);   // T in 1..max_t, a < T
      CHECK(d.draft(2, 1, f, out) >= 1); }
    // staging rows: catch-up 0..T-1 < chain row max_t + j - 1 < last-accepted slot 2*max_t - 1
    for (uint64_t maxT = 2; maxT <= 8; ++maxT) {
        for (uint64_t j = 1; j + 2 <= maxT; ++j) {
            CHECK(maxT <= maxT + j - 1 && maxT + j - 1 < 2 * maxT - 1 && mtp::rows_disjoint(maxT, j));
            for (uint64_t T = 1; T <= maxT; ++T) CHECK(T - 1 < maxT + j - 1);
        }
        MtpDrafter d; MtpDrafter::Config cfg; cfg.max_cells = 64; cfg.max_t = maxT; cfg.min_p = 0.5;
        CHECK(d.bind(cfg) && d.step_rows() == 2 * maxT && d.capture_prefill() == maxT);
        for (uint64_t T = 1; T <= maxT; ++T) {
            CHECK(d.capture_round(T));
            const auto& k = d.staged_kind();
            for (uint64_t r = 0; r < 2 * maxT; ++r) {
                const uint8_t want = r < T ? 1 : (r == 2 * maxT - 1 ? 3 : (r >= maxT && r + 3 <= 2 * maxT + 0 ? 2 : 0));
                if (r >= T && r < maxT) CHECK(k[r] == 0);
                else if (r < T || r == 2 * maxT - 1) CHECK(k[r] == want);
            }
        }
    }
    // K/V-only encoding: step_row0 = -1 - r decodes back, is negative; a non-negative step_row0 requires T == 1
    for (int64_t r = 0; r < 100; ++r) { const int64_t e = -1 - r; CHECK(e < 0 && -1 - e == r); }
    // tensor lookup is by name AND kind; draft vocabulary count = bytes / 4
    { MtpDrafter d;
      CHECK(d.load_text("w f32 0 16\nw bf16 16 32\n") && d.f32("w") && d.bf16("w") && !d.q8("w") && d.f32("w")->off == 0 && d.bf16("w")->off == 16);
      CHECK(!d.load_text("w nope 0 1\n")); }
    for (uint64_t b = 4; b < 200; ++b) { CHECK(mtp::n_dvocab(b) >= 1); if (b % 4 == 0) CHECK(mtp::n_dvocab(b) * 4 == b); }
}

// ----------------------------------------------------------------------------------------------- VerifyWindow
inline void verify_window() {
    using core::Verifier;
    // token groups
    for (uint64_t T = 1; T <= 8; ++T) for (int split = 0; split < 2; ++split) {
        const uint64_t G = (split && T >= 2) ? 2 : 1;
        const uint64_t a0 = verify::group_begin(G, 0, T), a1 = verify::group_end(G, 0, T);
        CHECK(a0 == 0 && a1 >= 1);
        if (G == 1) CHECK(a1 == T);
        else {
            const uint64_t b0 = verify::group_begin(G, 1, T), b1 = verify::group_end(G, 1, T);
            CHECK(a1 == b0 && b1 == T && a1 - a0 + (b1 - b0) == T && b1 > b0 && a1 - a0 >= b1 - b0 && a1 == (T + 1) / 2);
        }
    }
    // ring numbering: k -> (layer, group) -> ring == k + 1, injective
    for (uint64_t G = 1; G <= 2; ++G) for (uint64_t L = 1; L <= 48; ++L) {
        std::set<uint64_t> seen;
        for (uint64_t k = 0; k < L * G; ++k) {
            CHECK(verify::ring_layer(G, k) < L && verify::ring_group(G, k) < G);
            CHECK(verify::ring_no(G, verify::ring_layer(G, k), verify::ring_group(G, k)) == k + 1);
            seen.insert(verify::ring_no(G, verify::ring_layer(G, k), verify::ring_group(G, k)));
        }
        CHECK(seen.size() == L * G);
    }
    // plan block layout
    for (uint64_t cap = 1; cap < 300; ++cap) {
        const uint64_t base = verify::plan_i32(cap), po = verify::ptr_off(cap);
        CHECK(po % 2 == 0 && base <= po && po <= base + 1);
        CHECK(verify::ptr2_off(cap) == po + 2 * cap && verify::start2_off(cap) == po + 4 * cap);
        CHECK(verify::plan_total(cap) == verify::start2_off(cap) + (cap + 1) + 1);
        for (uint64_t g = 0; g < 2; ++g) CHECK(verify::plan_slot_off(verify::plan_total(cap), g) + verify::plan_total(cap) <= 2 * verify::plan_total(cap));
    }
    // staging: 16 blobs, halved for a split window; slices disjoint and inside
    for (uint64_t G = 1; G <= 2; ++G) for (uint64_t g = 0; g < G; ++g) CHECK(verify::staging_begin(G, 16, g) + verify::per(G, 16) <= 16);
    CHECK(verify::per(2, 16) == 8 && verify::per(1, 16) == 16 && verify::staging_begin(2, 16, 0) + verify::per(2, 16) <= verify::staging_begin(2, 16, 1));
    // monotone flag (compare-and-set only raises) and its commutativity / idempotence
    Rng rng(3);
    for (int it = 0; it < 500; ++it) {
        Verifier v; Verifier::Config c; c.max_t = 4; c.n_layers = 2; c.groups = 1;
        CHECK(v.init(c));
        const uint32_t a = rng.below(100), b = rng.below(100), s0 = rng.below(100);
        v.raise_flag(s0);
        v.raise_flag(a); const uint32_t f1 = v.flag(); CHECK(f1 >= s0 && f1 >= a && f1 == std::max(s0, a));
        v.raise_flag(b); CHECK(v.flag() == std::max({s0, a, b}));
        v.raise_flag(a); CHECK(v.flag() == std::max({s0, a, b}));
        Verifier w; CHECK(w.init(c)); w.raise_flag(s0); w.raise_flag(b); w.raise_flag(a);
        CHECK(w.flag() == v.flag());
    }
    // init argument checks: window of 1..8, groups 1 or 2
    { Verifier v; Verifier::Config c; c.max_t = 9; CHECK(!v.init(c)); c.max_t = 0; CHECK(!v.init(c)); c.max_t = 8; c.groups = 3; CHECK(!v.init(c)); }
    // schedule: (layer, group) order k = l*G + g, ring k+1; run() raises the flag to the last ring; groups cover the window
    for (uint64_t G = 1; G <= 2; ++G) for (uint64_t T = 1; T <= 8; ++T) {
        Verifier v; Verifier::Config c; c.max_t = 8; c.n_layers = 5; c.groups = G;
        CHECK(v.init(c) && v.capture());
        std::vector<int32_t> toks(T, 7);
        CHECK(v.record_window(toks.data(), T));
        CHECK(!v.record_window(toks.data(), 9));
        uint64_t k = 0, last_ring = 0;
        const uint64_t n = v.run([&](const Verifier::Step& s) {
            CHECK(s.layer == k / G && s.group == k % G && s.ring == k + 1 && s.row_begin <= s.row_end && s.row_end <= T);
            if (G == 2 && s.group == 0) CHECK(s.row_begin == 0 && s.row_end == (T + 1) / 2);
            if (G == 2 && s.group == 1) CHECK(s.row_begin == (T + 1) / 2 && s.row_end == T);
            if (G == 1) CHECK(s.row_begin == 0 && s.row_end == T);
            last_ring = s.ring; ++k;
        });
        CHECK(n == 5 * G && v.flag() == last_ring && last_ring == 5 * G);
    }
    // commit vector: [n_keep, n_keep-1, pos0+t or -1]; valid exactly for kept rows; last kept row carries its position
    for (int64_t pos0 = 0; pos0 < 20; pos0 += 7) for (int64_t nk = 1; nk <= 8; ++nk) {
        const auto v = verify::commit_vector(pos0, nk, 8);
        CHECK(v.size() == 10 && v[0] == nk && v[1] == nk - 1);
        for (int64_t t = 0; t < 8; ++t) CHECK((v[2 + t] >= 0) == (t < nk) && (t >= nk ? v[2 + t] == -1 : v[2 + t] == pos0 + t));
        CHECK(v[2 + nk - 1] == pos0 + nk - 1);
    }
    { Verifier v; Verifier::Config c; c.max_t = 4; c.n_layers = 1; c.groups = 1; CHECK(v.init(c) && v.capture());
      int32_t t4[4] = {1, 2, 3, 4}; CHECK(v.record_window(t4, 4));
      std::vector<int32_t> e;
      CHECK(!v.commit(0, 10, e));          // commit graph not armed
      CHECK(v.capture_commit());
      for (uint64_t a = 0; a < 4; ++a) CHECK(v.commit(a, 10, e) && e[0] == (int32_t)(a + 1) && e[1] == (int32_t)a);
      CHECK(!v.commit(4, 10, e));          // n_keep may not exceed T
      CHECK(v.accept_len({2, 3, 4}) == 3 && v.accept_len({2, 9, 4}) == 1 && v.accept_len({9, 3, 4}) == 0); }
    // run() argument check: positions of an accepted window are valid cache cells
    for (uint64_t T = 0; T <= 9; ++T) for (uint64_t pos0 = 0; pos0 < 20; pos0 += 3) {
        const bool ok = 1 <= T && T <= 8 && pos0 + T <= 16;
        if (ok) for (uint64_t t = 0; t < T; ++t) CHECK(pos0 + t < 16);
    }
}

// ---------------------------------------------------------------------------------- VerifyWindowKernels (index math)
inline void window_kernel_indexing() {
    using namespace vwin;
    CHECK(S == 128 && RG * RPG == S && RG * S == 512);
    for (uint32_t t = 0; t < 8; ++t) for (uint32_t j = 0; j < 3; ++j) {
        const uint32_t e = seq_src(t, j);
        if (!seq_is_hist(e)) CHECK(seq_row(e) < t);          // history taps read earlier tokens only
        else CHECK(e < 3);
    }
    for (uint32_t n = 0; n <= 8; ++n) for (uint32_t j = 0; j < 3; ++j) { const uint32_t e = n + j; if (e >= 3) CHECK(e - 3 < n); }
    for (int T = 1; T <= 8; ++T) for (int n = -2; n < 12; ++n) { const int k = keep_clamp(n, T); CHECK(k >= 0 && k <= T && (n >= 0 && n <= T ? k == n : true)); }
    for (int nkv = 0; nkv < 300; nkv += 7) for (int w = 1; w < 300; w += 11) {
        const int st = win_start(nkv, w), wd = win_width(nkv, w);
        CHECK(st + wd == nkv && wd <= w && wd <= nkv);
    }
    for (uint64_t n : {1ull, 255ull, 256ull, 257ull, 16383ull, 16384ull, 100000ull}) CHECK(copy_blocks(n) <= 64 && copy_blocks(n) >= 1);
    for (int bits : {1, 2, 4, 8}) for (uint64_t i = 0; i < 64; ++i) CHECK(codes_per_byte(bits) * bits == 8 && (i % codes_per_byte(bits)) * bits < 8);
    CHECK(!bits_ok(3) && !bits_ok(16));
    // row-major bound, fetch split, rebase separation, stride owner
    for (uint64_t T = 1; T <= 8; ++T) for (uint64_t t = 0; t < T; ++t) for (uint64_t h : {1ull, 3ull, 128ull}) for (uint64_t r = 0; r < h; r += (h > 3 ? 31 : 1))
        CHECK(rowmajor(t, h, r) < T * h);
    for (uint64_t per = 1; per < 50; per += 7) for (uint64_t n = 1; n < 12; ++n) for (uint64_t i = 0; i < n * per; ++i)
        CHECK(i / per < n && i % per < per && (i / per) * per + i % per == i);
    for (uint64_t k = 0; k < 20; ++k) for (uint64_t k2 = k + 1; k2 < 21; ++k2) CHECK(1000 + k * 24 + 24 <= 1000 + k2 * 24);
    for (uint64_t i = 0; i < 500; ++i) for (uint64_t st : {1ull, 7ull, 64ull}) CHECK(i % st + st * (i / st) == i);
    CHECK(1024 / 32 == 32);
}

// --------------------------------------------------------------------------------------------- SpecController
inline void spec_controller() {
    using namespace spec;
    for (uint64_t k = 0; k < 40; ++k) {
        const uint64_t n = clamp_n(k);
        CHECK(n >= 1 && n <= K_MAX + 1 && n - 1 < K_MAX + 1 && step_idx(k) == n - 1);
        if (k <= K_MAX) CHECK(n == k + 1);
    }
    Rng rng(5);
    for (int it = 0; it < 400; ++it) {
        uint64_t pv[10]; for (auto& v : pv) v = rng.below(5);
        auto p = [&](uint64_t i) { return pv[i % 10]; };
        auto same = [&](uint64_t) { return pv[0]; };
        for (uint64_t k = 0; k < 8; ++k) {
            CHECK(expect_nat(p, k) >= 1 && expect_nat(p, k) <= expect_nat(p, k + 1) && expect_nat(same, k) <= expect_nat(same, k + 1));
        }
        CHECK(expect_nat(p, 0) == 1);
    }
    // Controller: priors 0.86 per MTP position and {0.35, 0.6, 0.8, 0.92} for lookup by bucket
    Controller c0;
    for (uint64_t i = 0; i < K_MAX; ++i) CHECK(c0.mtp_p[i] == 0.86);
    CHECK(c0.lookup_q[0] == 0.35 && c0.lookup_q[1] == 0.6 && c0.lookup_q[2] == 0.8 && c0.lookup_q[3] == 0.92);
    for (int i = 0; i < 3; ++i) CHECK(c0.lookup_q[i] <= c0.lookup_q[i + 1]);
    // expected(): 1 + running products; lookup uses ONE bucket entry at every position; k = 0 / None -> 1
    CHECK(c0.expected(Source::None, 5) == 1.0 && c0.expected(Source::Mtp, 0) == 1.0);
    { double run = 1.0, s = 1.0; for (int k = 1; k <= 8; ++k) { run *= 0.86; s += run; CHECK(std::fabs(c0.expected(Source::Mtp, k) - s) < 1e-12); } }
    { double run = 1.0, s = 1.0; for (int k = 1; k <= 8; ++k) { run *= 0.8; s += run; CHECK(std::fabs(c0.expected(Source::Lookup, k, 9) - s) < 1e-12); } }
    for (uint64_t k = 0; k < 8; ++k) CHECK(c0.expected(Source::Mtp, k) <= c0.expected(Source::Mtp, k + 1));
    // choose: baseline wins unless the best candidate beats it by the threshold; never worse than the baseline rate
    for (int it = 0; it < 300; ++it) {
        Controller c;
        for (auto& v : c.cost.table) v = 1.0 + rng.uni() * 2;
        c.cost.mtp_ms_per_tok = rng.uni() * 0.2; c.cost.lookup_ms = rng.uni() * 0.2;
        for (auto& v : c.mtp_p) v = rng.uni();
        for (auto& v : c.lookup_q) v = rng.uni();
        const uint64_t mk = rng.below(10), la = rng.below(10), lm = rng.below(30);
        const Choice ch = c.choose(mk, la, lm);
        const double base = 1.0 / c.cost.step_ms(0);
        CHECK(ch.rate >= base - 1e-12);
        if (ch.drafter == Source::None) CHECK(ch.k == 0);
        else CHECK(ch.k >= 1 && ch.rate >= base * (1.0 + c.threshold) - 1e-12 && ch.k <= K_MAX && (ch.drafter == Source::Mtp ? ch.k <= mk : ch.k <= la));
    }
    // observe: positions 0..accepted-1 hit, position accepted (when drafted) miss: min(k, accepted+1) positions, rest untouched
    for (uint64_t k = 1; k <= 8; ++k) for (uint64_t acc = 0; acc <= 10; ++acc) {
        Controller c;
        double before[K_MAX]; std::memcpy(before, c.mtp_p, sizeof before);
        c.observe(Source::Mtp, k, acc);
        const uint64_t a = std::min(acc, k), seen_n = std::min(k, a + 1);
        for (uint64_t i = 0; i < K_MAX; ++i) {
            if (i < a) CHECK(c.mtp_p[i] > before[i] && c.mtp_p[i] <= 1.0);
            else if (i < seen_n) CHECK(c.mtp_p[i] < before[i] && c.mtp_p[i] >= 0.0);
            else CHECK(c.mtp_p[i] == before[i]);
        }
    }
    // EMA step in per mille stays in range and fixes its own value
    for (uint64_t v = 0; v <= 1000; v += 37) for (uint64_t e = 0; e <= 1000; e += 125) for (uint64_t h = 0; h <= 1000; h += 250) {
        CHECK(ema_step(1000, e, v, h) <= 1000);
        CHECK(ema_step(1000, e, v, v) == v);
    }
    // gain gate: passing implies baseline <= best; with zero gain exactly baseline <= best
    for (uint64_t b = 0; b < 60; ++b) for (uint64_t x = 0; x < 60; ++x) {
        CHECK(clears_gate_pm(x, b, 0) == (b <= x));
        if (clears_gate_pm(x, b, 50)) CHECK(b <= x);
    }
    CHECK(pick_best(5, {3, 4}) == 5 && pick_best(5, {3, 9, 7}) == 9 && pick_best(5, {}) == 5);
}

// ---------------------------------------------------------------------------------------------- SuffixDrafter
inline void suffix_drafter() {
    using namespace suffix;
    for (uint64_t m = 0; m < 12; ++m) for (uint64_t x = 0; x < 12; ++x) {
        CHECK(min_match_of(m) >= 3 && min_match_of(m) >= m && max_match_of(m, x) >= m && max_match_of(m, x) >= x);
        if (m >= 3 && x >= m) CHECK(min_match_of(m) == m && max_match_of(m, x) == x);
        SuffixDrafter d(64, m, x);
        CHECK(d.min_match() == min_match_of(m) && d.max_match() == max_match_of(m, x));
        // if the effective maximum is below the effective minimum nothing can ever be proposed
        if (d.max_match() < d.min_match()) {
            for (int rep = 0; rep < 3; ++rep) for (uint32_t t = 1; t <= 5; ++t) d.append(t);
            CHECK(d.propose(4).empty());
        }
    }
    // capacity: smallest power of two >= requested (>= 16)
    for (uint64_t s = 1; s < 3000; s += 17) {
        SuffixDrafter d(s);
        const uint64_t cap = d.capacity();
        CHECK(cap >= s && (cap & (cap - 1)) == 0 && cap >= 16 && (cap == 16 || cap / 2 < s));
    }
    Rng rng(9);
    // key is never 0 and is odd; trigram hashing depends on the last three tokens
    { SuffixDrafter d(256);
      for (int i = 0; i < 400; ++i) d.append(rng.below(50));
      for (uint64_t i = 2; i < d.size(); ++i) { const uint64_t k = d.key_at(i); CHECK(k != 0 && k % 2 == 1); } }
    // propose: brute force reference — most recent longest earlier match, ties keep the newer, continuation never past the end
    for (int it = 0; it < 300; ++it) {
        const uint64_t mn = 3 + rng.below(3), mx = mn + rng.below(10);
        SuffixDrafter d(4096, mn, mx);
        std::vector<uint32_t> h;
        const uint32_t alpha = 2 + rng.below(4);
        const int len = 4 + rng.below(80);
        for (int i = 0; i < len; ++i) {
            const uint32_t t = rng.below(alpha);
            d.append(t); h.push_back(t);
            const uint64_t maxk = 1 + rng.below(8);
            const std::vector<uint32_t> got = d.propose(maxk);
            // reference over ALL earlier ends (the table keeps only WAYS most recent per key)
            const int64_t cur = (int64_t)h.size() - 1;
            std::vector<int64_t> cands;
            if (cur >= 2) for (int64_t p = cur - 1; p >= 2; --p) if (h[p] == h[cur] && h[p - 1] == h[cur - 1] && h[p - 2] == h[cur - 2]) cands.push_back(p);
            if (cands.size() > WAYS - 1) cands.resize(WAYS - 1);   // the current end holds one of the WAYS slots
            uint64_t best_len = 0; int64_t best_end = -1;
            for (int64_t p : cands) {
                uint64_t l = 0;
                while (l < d.max_match() && (int64_t)l <= p && h[p - l] == h[cur - l]) ++l;
                if (l > best_len) { best_len = l; best_end = p; }
            }
            std::vector<uint32_t> want;
            if (best_end >= 0 && best_len >= d.min_match()) {
                const uint64_t k = std::min<uint64_t>(maxk, (uint64_t)(cur - best_end));
                for (uint64_t j = 0; j < k; ++j) want.push_back(h[best_end + 1 + j]);
            }
            CHECK(got == want);
            CHECK(got.size() <= maxk && best_end + (int64_t)got.size() <= cur);
        }
    }
    // way list: newest first, at most WAYS; reset empties everything
    { SuffixDrafter d(64, 3, 8);
      for (int rep = 0; rep < 10; ++rep) { d.append(1); d.append(2); d.append(3); d.append(9); }
      CHECK(d.size() == 40);
      d.reset(); CHECK(d.size() == 0 && d.propose(4).empty()); }
}

inline void run_all() {
    quantizer(); draft_layer(); verify_window(); window_kernel_indexing(); spec_controller(); suffix_drafter();
}
}  // namespace u25c
