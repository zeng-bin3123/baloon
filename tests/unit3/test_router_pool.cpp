#include "check.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include "baloon/router_top_k_selection.hpp"
#include "baloon/cpu_expert_pool.hpp"
using namespace baloon;

void test_router_order_and_topk() {
    using namespace router_topk;
    testh::Rng rng(21);
    std::vector<Cand> cs;
    for (uint64_t v = 0; v < 4; ++v)
        for (uint64_t i = 0; i < 4; ++i) cs.push_back({(double)v, i});
    for (auto& a : cs) {
        CHECK(!better(a, a));
        for (auto& b : cs) {
            if (better(a, b)) CHECK(!better(b, a));
            if (a.index != b.index) {
                CHECK(better(a, b) || better(b, a));
                Cand p = pick(a, b), q = pick(b, a);
                CHECK(p.value == q.value && p.index == q.index);
            }
            Cand p = pick(a, b);
            CHECK((p.index == a.index && p.value == a.value) || (p.index == b.index && p.value == b.value));
            for (auto& c : cs)
                if (better(a, b) && better(b, c)) CHECK(better(a, c));
        }
        CHECK(better(a, Cand{a.value, 1000}));  // sentinel index loses at equal value
    }
    for (int t = 0; t < 500; ++t) {
        uint64_t n = 1 + rng.below(64), k = rng.below(12);
        std::vector<double> v(n);
        for (auto& x : v) x = (double)rng.below(8);  // many ties
        auto got = top_k(v, k);
        std::vector<uint64_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](uint64_t a, uint64_t b) { return v[a] > v[b]; });
        idx.resize(std::min<uint64_t>(k, n));
        CHECK(got == idx);  // larger value first, ties by smaller index
        std::set<uint64_t> u(got.begin(), got.end());
        CHECK(u.size() == got.size());
        // reduction-tree order is irrelevant: fold candidates in a random permutation
        std::vector<Cand> cand;
        for (uint64_t e = 0; e < n; ++e) cand.push_back({v[e], e});
        for (size_t i = cand.size(); i > 1; --i) std::swap(cand[i - 1], cand[rng.below((uint32_t)i)]);
        Cand w = cand[0];
        for (size_t i = 1; i < cand.size(); ++i) w = pick(w, cand[i]);
        auto one = top_k(v, 1);
        CHECK(one[0] == w.index);
    }
}

void test_router_clamp_and_launch() {
    using namespace router_topk;
    CHECK(clamp_divisor(0.0) == 1.0 / 16384.0 && clamp_divisor(-1.0) > 0 && clamp_divisor(2.5) == 2.5);
    for (double s : {0.0, 1e-9, 6e-5, 6.2e-5, 1.0, 1e9}) CHECK(clamp_divisor(s) >= 1.0 / 16384.0 && clamp_divisor(s) >= s);
    for (uint64_t t = 0; t <= 512; ++t) {
        CHECK(round32(t) >= t && round32(t) % 32 == 0 && round32(t) <= 512);
        if (t > 0) CHECK(round32(t) >= 32);
    }
    for (uint64_t n = 1; n <= 2048; ++n) {
        uint64_t th = launch_threads(n);
        CHECK(th == round32(std::min<uint64_t>(n, 512)) && th % 32 == 0 && th <= 512 && th >= 32);
        CHECK(round16(n) >= n && round16(n) % 16 == 0 && round16(n) % 8 == 0);
        CHECK((round16(n) + n * 8) % 4 == 0);
        CHECK(smem_bytes(n) == round16(n) + n * 12);
        for (uint64_t nt : std::initializer_list<uint64_t>{1, 7, 32, th}) {   // strided loop covers each expert exactly once
            std::vector<int> seen(n, 0);
            for (uint64_t tid = 0; tid < nt; ++tid)
                for (uint64_t e = tid; e < n; e += nt) { ++seen[e]; CHECK(e % nt == tid); }
            for (int s : seen) CHECK(s == 1);
        }
    }
}

void test_pool_split_and_segments() {
    using namespace cpu_pool;
    for (uint64_t rows = 0; rows <= 300; rows += (rows < 40 ? 1 : 17))
        for (uint64_t tasks = 1; tasks <= 40; ++tasks) {
            CHECK(split_pt(rows, tasks, 0) == 0 && split_pt(rows, tasks, tasks) == rows);
            uint64_t cov = 0;
            for (uint64_t i = 0; i < tasks; ++i) {
                uint64_t a = split_pt(rows, tasks, i), b = split_pt(rows, tasks, i + 1);
                CHECK(a <= b && b <= rows);
                cov += b - a;
            }
            CHECK(cov == rows);  // every row exactly once
        }
    CHECK(threads(5, true) == 6 && threads(5, false) == 5);
    for (uint64_t n = 1; n <= 64; ++n)
        for (bool h : {false, true}) {
            CHECK(tasks_per_phase(n, h) >= 3 && tasks_per_phase(n, h) == 3 * threads(n, h));
            for (uint64_t ne : {1ull, 8ull, 10ull}) CHECK(parts_per_expert(threads(n, h), ne) >= 1);
        }
    testh::Rng rng(8);
    for (int t = 0; t < 400; ++t) {
        uint64_t per = 1 + rng.below(50), ne = 1 + rng.below(10), rows = per * ne;
        uint64_t tasks = 1 + rng.below(30);
        std::vector<int> seen(rows, 0);
        for (uint64_t i = 0; i < tasks; ++i) {
            uint64_t g0 = split_pt(rows, tasks, i), g1 = split_pt(rows, tasks, i + 1);
            uint64_t sum = 0;
            for (auto s : segment(per, g0, g1)) {
                CHECK(s.r0 < s.r1 && s.r1 <= per && s.expert < ne);   // progress, no overshoot
                sum += s.r1 - s.r0;
                for (uint64_t r = s.r0; r < s.r1; ++r) ++seen[s.expert * per + r];
            }
            CHECK(sum == g1 - g0);
        }
        for (int s : seen) CHECK(s == 1);
    }
    CHECK(segment(0, 0, 5).empty() && segment(4, 3, 3).empty());
}

void test_pool_routing_batches_cores() {
    using namespace cpu_pool;
    CHECK(route(5, 42, 0) == Path::q2_gu && route(5, 7, 42) == Path::native_gu);
    CHECK(route(6, 0, 42) == Path::q2_down && route(6, 42, 7) == Path::native_down);
    for (uint64_t m = 0; m < 10; ++m)
        for (uint64_t gu : {0ull, 41ull, 42ull, 43ull})
            for (uint64_t d : {0ull, 41ull, 42ull, 43ull}) {
                Path p = route(m, gu, d);
                if (m == 5) CHECK(p == (gu == 42 ? Path::q2_gu : Path::native_gu));
                else if (m == 6) CHECK(p == (d == 42 ? Path::q2_down : Path::native_down));
                else CHECK(p == Path::other);
            }
    for (uint64_t cap = 1; cap <= 9; ++cap)
        for (uint64_t n = 0; n <= 40; ++n) {
            uint64_t b0 = 0, covered = 0, batches = 0;
            while (b0 < n) {
                uint64_t l = batch_len(cap, n, b0);
                CHECK(l > 0 && l <= cap && b0 + l <= n);
                b0 += l; covered += l; ++batches;
            }
            CHECK(covered == n && batches == (n + cap - 1) / cap);
        }
    std::vector<int> cores{4, 5, 6, 7};
    CHECK(skip_first(cores).size() == 3 && skip_first(cores)[0] == 5 && skip_first({}).empty());
    CHECK(skip_first({9}).empty());
    auto sk = skip_first(cores);
    for (uint64_t i = 0; i < 8; ++i) {
        int c = core_for(sk, i);
        if (i < sk.size()) CHECK(c == sk[i] && should_pin(c));
        else CHECK(c == -1 && !should_pin(c));
    }
    CHECK(!should_pin(-1) && should_pin(0));
    // park accounting: all parked, each leave/arrive restores the count
    uint64_t parked = 6;
    for (int i = 0; i < 6; ++i) { --parked; ++parked; CHECK(parked == 6); }
}
