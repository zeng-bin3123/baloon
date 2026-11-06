#include <vector>
#include "baloon/gdn_qsa_layer_buffers.hpp"
#include "check.hpp"
using namespace baloon::layer_buffers;

void test_lb_align_carve() {
    for (uint64_t n = 0; n < 5000; ++n) { CHECK(align16(n) >= n && align16(n) % 16 == 0 && align16(n) - n < 16); }
    for (uint64_t n = 0; n <= 256 * 64; n += 256) CHECK(q8k_bytes(n) * 256 == n * 292);
    CHECK(q8k_bytes(255) == 0 && q8k_bytes(256) == 292 && q8k_bytes(2560) == 2920);
    testh::Rng r(21);
    for (int it = 0; it < 20000; ++it) {
        std::vector<uint64_t> l(r.below(12));
        for (auto& x : l) x = r.below(1u << 20);
        uint64_t sum = 0;
        for (auto x : l) sum += align16(x);
        const uint64_t acc = r.below(1u << 16) * 16;
        CHECK(carve_end(acc, l) == acc + sum);
        CHECK(off16(l, l.size()) == sum && off16(l, l.size() + 3) == sum);
        for (size_t i = 0; i <= l.size(); ++i) {
            CHECK(off16(l, i) % 16 == 0 && off16(l, i) <= sum);
            if (i < l.size()) CHECK(off16(l, i) + l[i] <= off16(l, i + 1));   // regions never overlap
        }
        CHECK(off16(l, 0) == 0);
    }
    CHECK(carve_end(0, {}) == 0 && carve_end(7, {}) == 7);
    CHECK(carve_end(0, {1, 16, 17}) == 16 + 16 + 32);
}

void test_lb_planes_dispatch() {
    testh::Rng r(5);
    for (int i = 0; i < 20000; ++i) {
        Planes p{r.below(1u << 20), r.below(1u << 16), r.below(1u << 16)};
        const uint64_t bytes = p.codes + p.scales + p.offs;
        CHECK(planes_tile(p, bytes));
        CHECK(!planes_tile(p, bytes + 1));
        if (bytes) CHECK(!planes_tile(p, bytes - 1));
        CHECK(scales_off(p) + p.scales <= bytes && offs_off(p) + p.offs == bytes);
    }
    for (int bits : {2, 3, 4, 5, 6, 8}) for (bool w : {false, true}) {
        CHECK(kernel_of(true, bits, w) == Kind::native);
        if (bits == 2) CHECK(kernel_of(false, bits, w) == Kind::s2);
        else CHECK(kernel_of(false, bits, w) == (w ? Kind::q8k : Kind::q8_0));
    }
    CHECK(kernel_of(false, 2, true) == Kind::s2);   // 2-bit beats wants_q8k
}

void test_lb_layers_dump() {
    CHECK(count_qsa(kLayers) == 12 && count_gdn(kLayers) == 36);
    for (int l = 0; l < kLayers; ++l) CHECK(is_qsa_layer(l) == (l % 4 == 3));
    CHECK(!is_qsa_layer(0) && is_qsa_layer(3) && is_qsa_layer(47));
    for (uint64_t hd : {64u, 128u, 256u}) for (uint64_t nh : {1u, 8u, 24u}) for (uint64_t h = 0; h < nh; ++h) {
        CHECK(qgate_gate_off(h, hd) + hd <= qgate_total(nh, hd));
        CHECK(qgate_row_off(h, hd) + 2 * hd == (h + 1) * 2 * hd);
    }
    CHECK(qgate_total(24, 256) == 24 * 512);
    CHECK(rope_pos_ok(10, 5, 16) && !rope_pos_ok(10, 6, 16) && !rope_pos_ok(10, 100, 16));
    testh::Rng r(9);
    for (int i = 0; i < 20000; ++i) {
        const uint64_t n = r.below(4096), hc = r.below(8), nhhd = r.below(1u << 16), nvk = r.below(1u << 12) * 2;
        const uint64_t s = dump_stride(n, hc, nhhd, nvk);
        CHECK(dump_kcur_off(n, hc, nhhd, nvk) + nvk == s);
        CHECK(dump_gather_end(n, hc, nhhd, nvk) <= s);
        CHECK(dump_layer_off(3, s) + s == dump_layer_off(4, s));
    }
    for (int m = 0; m < 64; ++m) {
        const bool f[6] = {(m & 1) != 0, (m & 2) != 0, (m & 4) != 0, (m & 8) != 0, (m & 16) != 0, (m & 32) != 0};
        const bool ok = verify_compatible(f[0], f[1], f[2], f[3], f[4], f[5]);
        CHECK(ok == (m == 63));
        if (ok) CHECK(f[1]);
    }
}
