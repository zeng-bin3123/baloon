#include <cstdint>
#include <random>
#include <vector>
#include "baloon/session_state_layout.hpp"
#include "check.hpp"

using namespace baloon::session;

void test_session_alignment() {
    std::mt19937_64 rng(3);
    std::vector<size_t> ns = {0, 1, 255, 256, 257, 511, 512, 513, 1 << 20, (1u << 20) + 1};
    for (int i = 0; i < 100000; ++i) ns.push_back((size_t)(rng() >> 20));
    for (size_t n : ns) {
        const size_t a = align_up_256(n);
        CHECK(a >= n && a % 256 == 0 && a - n < 256);
        CHECK(align_up_256(a) == a);                       // idempotent
        if (n % 256 == 0) CHECK(a == n);                   // exact multiples are untouched
    }
}

void test_session_gdn_index() {
    int count = 0;
    for (int l = 0; l < kLayers; ++l) {
        // definition: number of non-QSA layers before l
        int def = 0;
        for (int i = 0; i < l; ++i) if (!is_qsa_layer(i)) ++def;
        CHECK(gdn_index(l) == def);
        CHECK(gdn_index(l) == l - l / 4);
        if (!is_qsa_layer(l)) {
            CHECK(gdn_index(l) < kNumGdn);
            if (l + 1 < kLayers && !is_qsa_layer(l + 1)) CHECK(gdn_index(l + 1) == gdn_index(l) + 1);
            ++count;
        } else {
            CHECK(qsa_index(l) == (l - 3) / 4 && qsa_index(l) < kNumQsa);
        }
    }
    CHECK(count == kNumGdn);
    CHECK(gdn_index(48) == 36);
    // distinct GDN layers get distinct indices covering 0..35
    std::vector<int> seen(kNumGdn, 0);
    for (int l = 0; l < kLayers; ++l) if (!is_qsa_layer(l)) ++seen[gdn_index(l)];
    for (int c : seen) CHECK(c == 1);
}

void test_session_gdn_slices() {
    for (size_t S1 : {1u, 7u, 3145728u / 4}) for (size_t S2 : {1u, 5u, 30720u / 4}) {
        const size_t F = gdn_state_floats(S1, S2);
        for (size_t idx = 0; idx < kNumGdn; ++idx) {
            CHECK(gdn_conv_offset(idx, S1, S2) == gdn_slice_offset(idx, F) + S1);        // conv starts where state ends
            CHECK(gdn_conv_offset(idx, S1, S2) + S2 == gdn_slice_offset(idx + 1, F));    // and ends at the slice end
            CHECK(gdn_slice_offset(idx, F) + F <= gdn_arena_floats(kNumGdn, F));         // inside the allocation
        }
    }
}

void test_session_qsa_arena() {
    for (size_t first : {1u, 10u, 4096u}) for (size_t rest : {1u, 3u, 1024u}) for (size_t n : {1u, 2u, 12u}) {
        const size_t arena = qsa_arena(first, rest, n);
        CHECK(arena == first + (n - 1) * rest);
        for (size_t i = 0; i < n; ++i) {
            CHECK(qsa_off(first, rest, i) + qsa_size(first, rest, i) <= arena);
            if (i + 1 < n) CHECK(qsa_off(first, rest, i) + qsa_size(first, rest, i) == qsa_off(first, rest, i + 1));  // tiles, no overlap
        }
        CHECK(qsa_off(first, rest, 0) == 0 && qsa_size(first, rest, 0) == first);
        CHECK(qsa_off(first, rest, n - 1) + qsa_size(first, rest, n - 1) == arena);   // last state ends exactly at the end
    }
}

void test_session_ring_and_stages() {
    for (int l = 0; l < kLayers; ++l) {
        CHECK(ring_wait_value(l) > 0);                                  // never confused with a zeroed doorbell
        CHECK(ring_wait_value(l) < ring_wait_value(l + 1));             // strictly monotone
        CHECK(ring_wait_value(l) == (uint64_t)l + 1);
    }
    std::mt19937_64 rng(11);
    for (int i = 0; i < 100000; ++i) {
        int64_t t[5];
        for (auto& x : t) x = (int64_t)(rng() % 2000001) - 1000000;
        const auto st = stage_times(t[0], t[1], t[2], t[3], t[4]);
        int64_t sum = 0;
        for (auto x : st) sum += x;
        CHECK(sum == t[4]);                                             // telescopes to the full prefix
        CHECK(st[0] == t[0] && st[1] == t[1] - t[0] && st[4] == t[4] - t[3]);
    }
}
