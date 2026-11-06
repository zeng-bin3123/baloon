// Invariants of the CPU expert pool theory (claiming, completion handover, cores, worker count, row-range
// arithmetic, run_split decomposition, expert batching) checked on the arithmetic and on the real thread pool.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>
#include "baloon/cpu_expert_pool.hpp"
#include "check.hpp"
using namespace baloon::cpu_pool;

void test_pool_claim_and_handover_arith() {
    for (uint64_t n = 0; n <= 40; ++n) {
        uint64_t valid = 0;
        for (uint64_t k = 0; k < 60; ++k) {
            CHECK(claim_index(k) == k);
            CHECK(valid_claim(claim_index(k), n) == (k < n));   // valid iff in range(n)
            valid += valid_claim(claim_index(k), n);
        }
        CHECK(valid == std::min<uint64_t>(60, n));              // exactly min(m,n) of the first m claims succeed
        for (uint64_t m = 0; m <= 60; m += 7) {
            uint64_t c = 0;
            for (uint64_t k = 0; k < m; ++k) c += valid_claim(k, n);
            CHECK(c == std::min(m, n));
        }
    }
    for (uint64_t n = 0; n < 20; ++n) {
        CHECK(in_drain(n, n) == 0);                             // all parked: nobody can race a counter reset
        for (uint64_t p = 0; p <= n; ++p) CHECK(in_drain(n, p) == n - p);
    }
    CHECK(in_drain(2, 1) > 0);  // done == njobs alone (3 == 3) leaves a worker in the drain, so parked must be checked too
}

void test_pool_cores_workers_rows() {
    CHECK(skip_first({}).empty());
    for (size_t sz = 1; sz < 20; ++sz) {
        std::vector<int> c(sz); std::iota(c.begin(), c.end(), 10);
        CHECK(skip_first(c).size() + 1 == c.size());
        if (sz > 1) CHECK(skip_first(c).front() == 11);
        CHECK(skip_first(c).size() <= c.size());
    }
    for (uint64_t req = 0; req < 70; ++req)
        for (uint64_t nc = 0; nc < 70; nc += 3) {
            const uint64_t w = n_workers(req, nc);
            CHECK(w >= 1);
            if (req > 0) CHECK(w == req);
            else CHECK(w == std::max<uint64_t>(1, nc));
        }
    CHECK(n_workers(0, 0) == 1 && n_workers(0, 7) == 7 && n_workers(5, 7) == 5);

    for (uint64_t mrows = 0; mrows < 400; mrows += 13)
        for (uint64_t mt = 1; mt < 50; ++mt) {
            CHECK(split_pt(mrows, mt, 0) == 0 && split_pt(mrows, mt, mt) == mrows);
            for (uint64_t i = 0; i < mt; ++i) CHECK(split_pt(mrows, mt, i) <= split_pt(mrows, mt, i + 1));
        }
    // flat row decomposes as (expert, row within expert)
    for (uint64_t per = 1; per < 30; ++per)
        for (uint64_t r = 0; r < 200; ++r) CHECK(r == per * (r / per) + r % per && r % per < per);
    // the clamped walk always makes progress and never crosses the expert edge
    for (uint64_t per = 1; per < 20; ++per)
        for (uint64_t r0 = 0; r0 < per; ++r0)
            for (uint64_t rem = 1; rem < 30; ++rem) {
                const uint64_t end = std::min(per, r0 + rem);
                CHECK(r0 < end && end <= per);
            }
    // expert index of a flat row below n*per is below n
    for (uint64_t per = 1; per < 12; ++per)
        for (uint64_t n = 1; n < 12; ++n)
            for (uint64_t r = 0; r < n * per; ++r) CHECK(r / per < n);
}

void test_pool_split_decomposition_and_chunks() {
    for (uint64_t thr = 1; thr < 70; ++thr)
        for (uint64_t n = 1; n < 100; ++n) {
            const uint64_t parts = parts_per_expert(thr, n);
            CHECK(parts >= 1);
            CHECK(3 * thr <= n * ((3 * thr + n - 1) / n));      // unclamped quotient gives >= 3 tasks per thread
            for (uint64_t i = 0; i < n * parts; i += 1 + (n * parts) / 17) {
                CHECK(task_expert(i, parts) < n && task_part(i, parts) < parts);
                CHECK(task_expert(i, parts) * parts + task_part(i, parts) == i);
            }
        }
    for (uint64_t ff = 0; ff < 700; ff += 29)
        for (uint64_t parts = 1; parts < 40; ++parts) CHECK(split_pt(ff, parts, parts) == ff && ff * parts / parts == ff);
    for (uint64_t k = 1; k <= 100; k += (k < 10 ? 1 : 13))
        for (uint64_t n = 0; n < 300; n += 7) {
            const auto cs = chunks(k, n);
            CHECK(std::accumulate(cs.begin(), cs.end(), uint64_t(0)) == n);   // every expert covered exactly once
            for (uint64_t c : cs) CHECK(c > 0 && c <= k);
            CHECK(cs.size() == (n + k - 1) / k);
        }
    CHECK(chunks(0, 5).empty() && chunks(4, 0).empty());
    for (uint64_t c : chunks(kMaxSplitMulti, 1000)) CHECK(c <= 96);       // at most 96 buffers per native batch
    CHECK(kMaxSplitMulti == 96);
}

void test_pool_cpu_list_parsing() {
    CHECK((parse_cpu_list("0-3,8,10-11\n") == std::vector<int>{0, 1, 2, 3, 8, 10, 11}));
    CHECK((parse_cpu_list("5") == std::vector<int>{5}));
    CHECK((parse_cpu_list("0,64") == std::vector<int>{0, 64}));
    CHECK(parse_cpu_list("3-1").empty() && parse_cpu_list("a").empty() && parse_cpu_list("1,,2").empty() && parse_cpu_list("1-").empty());
    CHECK(is_core_leader(0, {0, 64}) && !is_core_leader(64, {0, 64}) && is_core_leader(5, {}));
    const auto all = physical_cores(false), skip = physical_cores(true);
    CHECK(!all.empty());
    CHECK(skip.size() + 1 == all.size() || (all.size() == 1 && skip.empty()));
    for (size_t i = 0; i < skip.size(); ++i) CHECK(skip[i] == all[i + 1]);
}

// Real pool: every job index runs exactly once, over many batches, with and without the host working; run() returns
// only after every worker has parked again, so nothing is still running when it returns.
void test_pool_real_exactly_once() {
    for (unsigned nw : {0u, 1u, 2u, 5u, 12u})
        for (bool host : {false, true}) {
            if (nw == 0 && !host) continue;  // handled serially below
            baloon::cpu_pool::Pool pool(nw, {}, host);
            CHECK(pool.workers() == nw && pool.thread_count() == threads(nw, host));
            testh::Rng rng(100 + nw);
            for (int batch = 0; batch < 150; ++batch) {
                const uint64_t n = rng.below(200);
                std::vector<std::atomic<int>> hits(n);
                for (auto& h : hits) h.store(0);
                std::atomic<int> inflight{0};
                pool.run(n, [&](uint64_t i) { ++inflight; hits[i].fetch_add(1); if ((i & 7) == 0) std::this_thread::yield(); --inflight; });
                CHECK(inflight.load() == 0);                 // nothing still inside a job at return
                for (uint64_t i = 0; i < n; ++i) CHECK(hits[i].load() == 1);
            }
        }
    baloon::cpu_pool::Pool serial(0, {}, false);
    std::vector<int> hits(17, 0);
    serial.run(17, [&](uint64_t i) { ++hits[i]; });
    for (int h : hits) CHECK(h == 1);
}

// Slow job on a late-waking worker: the completion handover must wait for it (done alone is not enough).
void test_pool_real_handover_waits_for_drain() {
    baloon::cpu_pool::Pool pool(4, {}, false);
    for (int rep = 0; rep < 40; ++rep) {
        std::atomic<int> finished{0};
        pool.run(3, [&](uint64_t i) {
            if (i == 1) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ++finished;
        });
        CHECK(finished.load() == 3);
    }
    // back-to-back batches reuse the counters: no stale claim leaks into the next batch
    std::atomic<uint64_t> sum{0};
    for (int rep = 0; rep < 500; ++rep) pool.run(5, [&](uint64_t i) { sum += i + 1; });
    CHECK(sum.load() == 500u * 15u);
}

void test_pool_real_split_and_multi_cover_rows() {
    for (unsigned nw : {1u, 3u, 7u})
        for (bool host : {false, true}) {
            baloon::cpu_pool::Pool pool(nw, {}, host);
            testh::Rng rng(7 + nw);
            for (int t = 0; t < 60; ++t) {
                const uint64_t n = 1 + rng.below(40), ff = rng.below(300);
                std::vector<std::atomic<int>> seen(n * ff);
                for (auto& s : seen) s.store(0);
                std::atomic<int> bad{0};
                pool.run_split(n, ff, [&](uint64_t e, uint64_t r0, uint64_t r1) {
                    if (!(e < n && r0 < r1 && r1 <= ff)) { bad.fetch_add(1); return; }
                    for (uint64_t r = r0; r < r1; ++r) seen[e * ff + r].fetch_add(1);
                });
                CHECK(bad.load() == 0);
                for (auto& s : seen) CHECK(s.load() == 1);   // every row of every expert exactly once

                const uint64_t per = 1 + rng.below(120);
                std::vector<std::atomic<int>> seen2(n * per);
                for (auto& s : seen2) s.store(0);
                pool.run_multi(n, per, [&](const Seg& sg) {
                    if (!(sg.expert < n && sg.r0 < sg.r1 && sg.r1 <= per)) { bad.fetch_add(1); return; }
                    for (uint64_t r = sg.r0; r < sg.r1; ++r) seen2[sg.expert * per + r].fetch_add(1);
                });
                CHECK(bad.load() == 0);
                for (auto& s : seen2) CHECK(s.load() == 1);
            }
        }
}

void test_pool_real_pinned_workers() {
    const auto cores = physical_cores(true);
    baloon::cpu_pool::Pool pool(std::min<size_t>(3, cores.size() + 1), cores, true);
    std::atomic<int> n{0};
    pool.run(64, [&](uint64_t) { ++n; });
    CHECK(n.load() == 64);
}
