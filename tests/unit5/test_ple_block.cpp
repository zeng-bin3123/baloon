#include <cmath>
#include <vector>
#include "baloon/ple_block_indexing.hpp"
#include "check.hpp"

using namespace baloon::ple_block;

void test_block_history() {
    float col[9];
    for (int i = 0; i < 9; ++i) col[i] = (float)i;
    advance_column(col, 9, 99.f);
    for (int i = 0; i < 8; ++i) CHECK(col[i] == (float)(i + 1));   // shifted up by one
    CHECK(col[8] == 99.f);                                         // new value last
    // multi-channel layout is row-fastest and channels do not bleed into each other
    const int C = 7;
    std::vector<float> h(kHistRows * C), x(C), model(kHistRows * C);
    for (int c = 0; c < C; ++c) for (int r = 0; r < kHistRows; ++r) h[hist_index(r, kHistRows, c)] = (float)(100 * c + r);
    model = h;
    for (int step = 0; step < 20; ++step) {
        for (int c = 0; c < C; ++c) x[c] = (float)(1000 * step + c);
        history_advance(h.data(), kHistRows, C, x.data());
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r + 1 < kHistRows; ++r) model[hist_index(r, kHistRows, c)] = model[hist_index(r + 1, kHistRows, c)];
            model[hist_index(kHistRows - 1, kHistRows, c)] = x[c];
        }
        CHECK(h == model);
    }
    for (int c = 0; c < C; ++c) CHECK(h[hist_index(kHistRows - 1, kHistRows, c)] == (float)(1000 * 19 + c));
    // flat index bound
    for (int c = 0; c < C; ++c) for (int r = 0; r < kHistRows; ++r) CHECK(hist_index(r, kHistRows, c) < (int64_t)kHistRows * C);
}

void test_block_conv_taps() {
    const int rows[4] = {tap_row(9, 4, 3, 0), tap_row(9, 4, 3, 1), tap_row(9, 4, 3, 2), tap_row(9, 4, 3, 3)};
    CHECK(rows[0] == 0 && rows[1] == 3 && rows[2] == 6 && rows[3] == 9);
    for (int k = 0; k < 4; ++k) { CHECK(rows[k] <= 9); CHECK((rows[k] == 9) == (k == 3)); }
    // conv: only tap 3 sees the new value, taps 0..2 read history rows 0,3,6
    float col[9]; for (int i = 0; i < 9; ++i) col[i] = (float)(i + 1);
    const float w[4] = {1.f, 10.f, 100.f, 1000.f};
    CHECK(conv_channel(col, 9, 0.f, w, 4, 3) == 1.f * 1 + 10.f * 4 + 100.f * 7);
    CHECK(conv_channel(col, 9, 2.f, w, 4, 3) == 1.f * 1 + 10.f * 4 + 100.f * 7 + 2000.f);
    // advancing the history then convolving with a placeholder equals convolving the old column with the new value
    // for the last tap: new-row semantics are consistent with the shifted layout (row 9 becomes stored row 8)
    testh::Rng r(10);
    for (int t = 0; t < 1000; ++t) {
        float c0[9]; for (auto& v : c0) v = (float)r.gauss();
        float xn = (float)r.gauss(); float ww[4]; for (auto& v : ww) v = (float)r.gauss();
        const float a = conv_channel(c0, 9, xn, ww, 4, 3);
        const float b = ww[0] * c0[0] + ww[1] * c0[3] + ww[2] * c0[6] + ww[3] * xn;
        CHECK(a == b);
        // other geometries stay inside [0, nhist]
        for (int k = 0; k < 3; ++k) CHECK(tap_row(6, 3, 3, k) <= 6);
    }
}

void test_block_streams_gate() {
    const int n = kEmbeddingDim, hc = kStreamCount;
    CHECK(kStreamDim == 10240 && kStreamDim / n == hc);
    for (int c = 0; c < hc; ++c) for (int d : {0, 1, n / 2, n - 1}) {
        const int64_t i = stream_index(c, d, n);
        CHECK(i < (int64_t)hc * n && gate_stream(i, n) == c && value_index(i, n) == d);
    }
    for (int64_t i = 0; i < kStreamDim; ++i) { CHECK(gate_stream(i, n) < hc); CHECK(value_index(i, n) < n); }
    // streams tile the vector exactly once
    for (int c = 0; c + 1 < hc; ++c) CHECK(stream_index(c, n - 1, n) + 1 == stream_index(c + 1, 0, n));

    CHECK(sgn(3.f) == 1.f && sgn(-3.f) == -1.f && sgn(0.f) == 0.f);
    testh::Rng r(7);
    for (int i = 0; i < 100000; ++i) {
        const float s = (float)(r.gauss() * 10.0);
        CHECK(sgn(-s) == -sgn(s));
        CHECK(sgn(s) * s >= 0.f);
        const float g = gate(s), gm = gate(-s);
        CHECK(g > 0.f && g < 1.f);
        CHECK(std::fabs((g + gm) - 1.f) < 1e-6f);               // symmetric about 1/2
        const float ref = 1.f / (1.f + std::exp(-(sgn(s) * std::sqrt(std::max(std::fabs(s), 1e-6f)))));
        CHECK(g == ref);
        CHECK((s > 0) == (g > 0.5f) || s == 0.f);
    }
    CHECK(gate(0.f) == 0.5f);
    {   // sub-floor magnitude: sgn(s) = +-1 but |s| is floored to 1e-6, so z = +-sqrt(1e-6) = +-1e-3
        const float z = std::sqrt(kGateFloor);
        const float want = 1.f / (1.f + std::exp(-z));
        CHECK(std::fabs(z - 1e-3f) < 1e-7f);
        CHECK(gate(1e-9f) == want && gate(1e-9f) > 0.5f);
        CHECK(gate(-1e-9f) == 1.f / (1.f + std::exp(z)) && gate(-1e-9f) < 0.5f);
        CHECK(std::fabs(gate(1e-9f) - 0.5f) < 1e-3f);
        CHECK(gate(1e-9f) == gate(kGateFloor));                  // everything at or below the floor collapses
    }
}

void test_block_scratch() {
    for (uint64_t x = 0; x < 5000; ++x) {
        CHECK(align16(x) >= x && align16(x) % 16 == 0 && align16(x) - x < 16);
    }
    testh::Rng r(13);
    for (int i = 0; i < 100000; ++i) {
        const uint64_t f = r.below(1u << 24), q = r.below(1u << 24), e = r.below(1u << 24);
        const Scratch s = scratch_layout(f, q, e);
        CHECK(s.float_off == 0 && s.q8_off % 16 == 0 && s.bf16_off % 16 == 0);
        CHECK(s.q8_off >= f && s.bf16_off >= s.q8_off + q);          // regions do not overlap
        CHECK(s.total >= f + q + e + 256 - 0 && s.total >= s.bf16_off + e + 256);
        CHECK(align16(f) + align16(q) + e + 256 >= f + q + e);
    }
    const Scratch s = scratch_layout(10, 20, 30);
    CHECK(s.q8_off == 16 && s.bf16_off == 48 && s.total == 48 + 30 + 256);
}
