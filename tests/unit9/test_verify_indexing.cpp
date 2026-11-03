#include <set>
#include <vector>
#include "baloon/verify_window_indexing.hpp"
#include "check.hpp"
using namespace baloon::verify_idx;

void test_vi_conv_commit() {
    for (uint64_t t = 0; t < 40; ++t) for (uint64_t j = 0; j < 3; ++j) {
        if (!conv_from_hist(t, j)) CHECK(conv_x_index(t, j) < t + 1 && conv_x_index(t, j) + 3 == t + j && t + j - 3 < t + 1);
        else CHECK(conv_src(t, j) < 3);
        if (j < 3 && 3 <= t + j) CHECK(t + j - 3 < t);               // refers to an already-existing token < t (j<3)
    }
    // commit against the concatenated-sequence model
    testh::Rng rng(3);
    for (uint64_t n = 0; n < 30; ++n) {
        float hist[3], out[3]; std::vector<float> x(n);
        for (auto& v : hist) v = (float)rng.below(1000);
        for (auto& v : x) v = (float)rng.below(1000) + 1000.f;
        std::vector<float> cat(hist, hist + 3); cat.insert(cat.end(), x.begin(), x.end());
        commit_history(hist, x.data(), n, out);
        for (uint64_t j = 0; j < 3; ++j) { CHECK(out[j] == cat[n + j]); if (n > 0 && 3 <= n + j) CHECK(n + j - 3 < n); }
        if (n == 0) for (int j = 0; j < 3; ++j) CHECK(out[j] == hist[j]);          // nothing moves
        if (n >= 3) { CHECK(out[0] == x[n - 3] && out[2] == x[n - 1]); }           // last three tokens
    }
    for (uint64_t bx = 0; bx < 20; ++bx) for (uint64_t tx = 0; tx < 128; ++tx) {
        const uint64_t i = bx * 128 + tx;
        CHECK(chan_block(i) == bx && chan_lane(i) == tx);
    }
    for (uint64_t rg = 0; rg < 4; ++rg) for (uint64_t r = 0; r < 32; ++r) CHECK(gdn_row(rg, r) < 128);
}

void test_vi_window_steps() {
    for (uint64_t n = 0; n < 200; ++n) for (uint64_t w = 1; w < 80; ++w) {
        const uint64_t s = win_start(n, w), wd = win_width(n, w);
        CHECK(s == (n > w ? n - w : 0));
        CHECK(wd <= w && s + wd == n);
        const std::vector<uint32_t> ids = window_ids(n, w);
        CHECK(ids.size() == wd);
        for (size_t j = 0; j < ids.size(); ++j) CHECK(ids[j] == s + j && ids[j] < n);
    }
    std::vector<int32_t> c;
    for (int i = 0; i < 50; ++i) c.push_back(i * 7);
    const std::vector<int32_t> d = dense_steps(c);
    CHECK(d.size() == 200);
    for (size_t i = 0; i < c.size(); ++i) {
        CHECK(d[4 * i] == c[i] && d[4 * i + 1] == c[i] + 1 && d[4 * i + 2] == (c[i] + 1) / 4 && d[4 * i + 3] == c[i] + 1);
        CHECK((c[i] + 1) / 4 <= c[i] + 1);
    }
    std::set<uint64_t> slots;
    for (uint64_t i = 0; i < 30; ++i) for (uint64_t k = 0; k < 4; ++k) CHECK(slots.insert(i * 4 + k).second);   // slots distinct
}

void test_vi_blob_bcast() {
    for (uint64_t per = 1; per < 40; ++per) for (uint64_t n = 1; n < 12; ++n) for (uint64_t i = 0; i < per * n; ++i) {
        CHECK(per * fetch_blob(i, per) + fetch_off(i, per) == i);
        CHECK(fetch_blob(i, per) < n && fetch_off(i, per) < per);
    }
    for (uint64_t base = 0; base < 1000; base += 97) for (uint64_t bytes = 1; bytes < 300; bytes += 37) for (uint64_t k = 0; k < 20; ++k)
        CHECK(rebase_ptr(base, k + 1, bytes) == rebase_ptr(base, k, bytes) + bytes);
    for (uint64_t n = 1; n < 30; ++n) for (uint64_t hc = 1; hc < 5; ++hc) for (uint64_t i = 0; i < n * hc; ++i) {
        CHECK(bcast_src(i, n) < n && bcast_stream(i, n) < hc);
        CHECK(bcast_stream(i, n) * n + bcast_src(i, n) == i);
        CHECK(i / 256 < bcast_grid(n, hc));
    }
    for (uint64_t row16 = 1; row16 < 20; ++row16) for (uint64_t n = 1; n < 10; ++n) for (uint64_t i = 0; i < row16 * n; ++i)
        CHECK(gather_row(i, row16) < n);
}
