#include <vector>
#include "baloon/fused_gr_kernels.hpp"
#include "baloon/gr_gated_residual.hpp"
#include "check.hpp"
namespace fk = baloon::fused_gr;
namespace gr = baloon::gr;

void test_fk_geometry() {
    CHECK(fk::kD == 10240 && fk::kD == fk::kN * fk::kHc);
    for (int i = 0; i < fk::kD; ++i) { CHECK(fk::stream_of(i) < 4 && fk::col_of(i) < fk::kN && fk::flat(fk::stream_of(i), fk::col_of(i)) == i); }
    for (int t = 0; 4 * t + 3 < fk::kD; ++t) CHECK(fk::vec4_same_stream(t));
    CHECK(fk::kDownBlocks == 40 && fk::kDownBlocks * fk::kWarps == fk::kLr);
    std::vector<int> rows(fk::kLr, 0);
    for (int b = 0; b < fk::kDownBlocks; ++b) for (int w = 0; w < fk::kWarps; ++w) { CHECK(fk::down_row(b, w) < fk::kLr); rows[fk::down_row(b, w)]++; }
    for (int c : rows) CHECK(c == 1);
    CHECK(fk::kRowChunks == 1280 && fk::kRowChunks / 32 == 40 && fk::kLrChunks == 40);
    CHECK(fk::kUpBlocks * fk::kUpCols == fk::kN && fk::kUpMultiBlocks * fk::kUpMultiCols == fk::kN);
    for (int lane = 0; lane < 32; ++lane) CHECK(fk::second_pass_lane(lane) == (lane < 8) && (!fk::second_pass_lane(lane) || 32 + lane < fk::kLrChunks));
}

void test_fk_tiles() {
    CHECK(fk::kTileChunks == 320 && fk::kChunksPerLane == 10 && fk::kD / fk::kTile == 4);
    std::vector<int> ch(fk::kTileChunks, 0);
    for (int l = 0; l < 32; ++l) for (int q = 0; q < fk::kChunksPerLane; ++q) { CHECK(fk::tile_chunk(l, q) < fk::kTileChunks); ch[fk::tile_chunk(l, q)]++; }
    for (int c : ch) CHECK(c == 1);
    // up: (stream, block, dd) -> distinct weight rows covering [0, D)
    std::vector<int> w(fk::kD, 0);
    for (int c = 0; c < 4; ++c) for (int b = 0; b < fk::kUpBlocks; ++b) for (int dd = 0; dd < fk::kUpCols; ++dd) w[fk::up_weight_row(c, b, dd)]++;
    for (int x : w) CHECK(x == 1);
    for (int r = 0; r < 128; ++r) { CHECK(fk::up_row_stream(r) < 4 && fk::up_row_col(r) < 32 && fk::up_row_stream(r) * 32 + fk::up_row_col(r) == r); }
    std::vector<int> rr(128, 0);
    for (int w8 = 0; w8 < 8; ++w8) for (int k = 0; k < 16; ++k) rr[w8 + 8 * k]++;
    for (int x : rr) CHECK(x == 1);
    for (int r = 0; r < 128; ++r) CHECK(fk::warp_of_row(r) < 8 && fk::warp_step_of_row(r) < 16 && fk::warp_of_row(r) + 8 * fk::warp_step_of_row(r) == r);
    for (int T = 1; T <= 32; ++T) { for (int k = 0; k < T; ++k) CHECK(fk::token_lane_ok(k, T)); CHECK(fk::tile_bytes(T) == (int64_t)T * 10240); }
    CHECK(!fk::token_lane_ok(0, 33) && !fk::token_lane_ok(5, 5));
}

void test_gw_workspace() {
    for (uint64_t x = 0; x < 3000; ++x) { CHECK(gr::al(x) >= x && gr::al(x) % 16 == 0 && gr::al(x) < x + 16); }
    testh::Rng r(31);
    for (int it = 0; it < 50000; ++it) {
        uint64_t s[5]; uint64_t raw = 0, al_sum = 0;
        for (auto& v : s) { v = r.below(1u << 22); raw += v; al_sum += gr::al(v); }
        const gr::Workspace w = gr::workspace_layout(s);
        CHECK(w.total == al_sum && w.total >= raw);
        CHECK(w.off[0] == 0);
        for (int i = 0; i < 5; ++i) {
            CHECK(w.off[i] % 16 == 0);
            const uint64_t end = (i + 1 < 5) ? w.off[i + 1] : w.total;
            CHECK(w.off[i] + w.size[i] <= end);           // no region overlaps its successor
        }
    }
    const uint64_t s[5] = {1, 16, 17, 0, 100};
    const gr::Workspace w = gr::workspace_layout(s);
    CHECK(w.off[1] == 16 && w.off[2] == 32 && w.off[3] == 64 && w.off[4] == 64 && w.total == 64 + 112);
}

void test_gw_write_indexing() {
    const int n = 37, hc = 4;
    testh::Rng r(3);
    std::vector<float> R(n * hc), bo(n), w(hc), out(n * hc);
    for (auto& v : R) v = (float)r.gauss();
    for (auto& v : bo) v = (float)r.gauss();
    for (auto& v : w) v = (float)r.gauss();
    gr::write(R.data(), bo.data(), w.data(), out.data(), n, hc);
    for (int c = 0; c < hc; ++c) for (int d = 0; d < n; ++d) CHECK(out[c * n + d] == R[c * n + d] + bo[d] * w[c]);
    for (int64_t i = 0; i < n * hc; ++i) { CHECK(gr::stream_of(i, n) < hc && gr::col_of(i, n) < n); CHECK(n * gr::stream_of(i, n) + gr::col_of(i, n) == i); }
    // mean of equal streams equals the stream (hc = 4, power of two so exact)
    std::vector<float> x(n * hc), m(n);
    for (int d = 0; d < n; ++d) { const float v = (float)r.gauss(); for (int c = 0; c < hc; ++c) x[c * n + d] = v; }
    gr::mean_streams(x.data(), m.data(), n, hc);
    for (int d = 0; d < n; ++d) CHECK(m[d] == x[d]);
    for (int hcdim : {1, 7, 8, 9, 320, 10240}) { std::vector<int> hit(hcdim, 0);
        for (int b = 0; b < gr::gate_blocks(hcdim); ++b) for (int w8 = 0; w8 < 8; ++w8) { const int row = gr::gate_row(b, w8); if (row < hcdim) hit[row]++; }
        for (int c : hit) CHECK(c == 1); }
    for (int nn : {1, 255, 256, 257, 2560}) for (int d = 0; d < nn; ++d) CHECK(d / 256 < gr::mean_blocks(nn));
    // down: strided index set partitioned by (warp, lane, step)
    std::vector<int> hit(256 * 5, 0);
    for (int w8 = 0; w8 < 8; ++w8) for (int l = 0; l < 32; ++l) for (int k = 0; k < 5; ++k) hit[gr::down_index(w8, l, k)]++;
    for (int c : hit) CHECK(c == 1);
    CHECK(gr::kScratchFloats * 4 <= gr::kScratchDoubles * 8);
    for (int nn : {1, 2, 2560}) CHECK(gr::even_hc_dim(nn));
    CHECK(gr::inject_threads(4) / 32 == 4);
}
