// Invariants of WindowVerifier, WeightLoader, PinnedArena (LAYER 1 statements).
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>
#include "check.hpp"
#include "baloon/window_verifier.hpp"
#include "baloon/weight_loader.hpp"
#include "baloon/pinned_arena.hpp"
using namespace baloon;

void test_verifier_groups_ring() {
    for (uint64_t T = 1; T <= 200; ++T) {
        CHECK(winv::te0(T, 1) == T);
        CHECK(winv::te0(T, 2) == (T + 1) / 2);
        CHECK(winv::group_begin(T, 1, 0) == 0 && winv::group_end(T, 1, 0) == T);
        if (T >= 2) {
            const uint64_t m = winv::group_end(T, 2, 0);
            CHECK(m > 0 && T - m > 0);                      // both groups non-empty
            CHECK(winv::group_begin(T, 2, 1) == m);         // they tile [0,T)
            CHECK(winv::group_end(T, 2, 1) == T);
            CHECK(winv::tb(T, 1) == m);
        }
    }
    for (uint64_t G = 1; G <= 2; ++G)
        for (uint64_t k = 0; k < 500; ++k) {
            const uint64_t l = winv::layer_of(k, G), g = winv::group_of(k, G);
            CHECK(g < G);
            CHECK(winv::ring_of(l, g, G) == winv::ring_of_k(k));
            CHECK(winv::ring_of(l, g, G) == k + 1);
            for (uint64_t n = 1; n <= 8; ++n) if (k < n * G) CHECK(l < n);
        }
}

void test_verifier_plan_layout() {
    for (uint64_t cap = 0; cap <= 4096; ++cap) {
        const uint64_t po = winv::ptr_off(cap);
        CHECK(po % 2 == 0);
        CHECK(4 + (cap + 1) + cap + cap <= po && po <= 4 + (cap + 1) + cap + cap + 1);
        CHECK(winv::ptr_off_device(cap) == po);
        const auto L = winv::plan_layout(cap);
        CHECK(L.ptr == po && L.ptr % 2 == 0);
        CHECK(L.start == 4 && L.dst == 4 + cap + 1 && L.tok == L.dst + cap);
        CHECK(L.tok + cap <= L.ptr);                         // pad only, no overlap with pointer area
        CHECK(L.ptr + 2 * cap <= L.ptr2 && L.ptr2 + 2 * cap <= L.start2);  // u64 arrays do not overlap
        CHECK(L.start2 + cap + 1 <= L.total);                // start2 ends inside the slot
        CHECK(L.total == winv::plan_i32(cap));
    }
    for (uint64_t p = 0; p < 5000; p += 7) { CHECK(winv::host_plan_fits(p)); CHECK(winv::device_plan_fits(p)); }
}

void test_verifier_staging_flag_accept() {
    for (uint64_t K = 0; K < 300; ++K) {
        CHECK(2 * winv::per_group(K, 2) <= K);
        CHECK(winv::group_stage_off(K, 2, 0) + winv::per_group(K, 2) == winv::group_stage_off(K, 2, 1));
        CHECK(winv::per_group(K, 1) == K);
    }
    testh::Rng rng(12);
    for (int it = 0; it < 20000; ++it) {
        const uint32_t cur = rng.u32() % 1000, v = rng.u32() % 1000;
        const uint32_t r = winv::raise(cur, v);
        CHECK(r >= cur && r >= v && r == std::max(cur, v));
        const uint32_t c2 = cur + rng.u32() % 50;
        CHECK(winv::raise(cur, v) <= winv::raise(c2, v));
    }
    // CAS loop under contention: final value is the maximum, never lowered.
    std::atomic<uint32_t> flag{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 8; ++t) th.emplace_back([&, t] { for (uint32_t v = 0; v < 2000; ++v) winv::raise_flag(flag, (uint32_t)(v * 8 + t)); });
    for (auto& x : th) x.join();
    CHECK(flag.load() == 1999u * 8 + 7);
    // acceptance
    for (int it = 0; it < 5000; ++it) {
        const uint64_t T = 1 + rng.below(40);
        std::vector<int> win(T + 1), outv(T + 1);
        for (uint64_t i = 0; i <= T; ++i) { win[i] = (int)rng.below(3); outv[i] = rng.below(5) ? win[i] : (int)rng.below(3); }
        const uint64_t a = winv::accept_from(T, [&](uint64_t i) { return win[i + 1] == outv[i]; });
        CHECK(a + 1 <= T);
        CHECK(winv::commit_in_range(T, a));
        for (uint64_t i = 0; i < a; ++i) CHECK(win[i + 1] == outv[i]);                        // all accepted match
        if (a + 1 < T) CHECK(win[a + 1] != outv[a]);                                          // stopped at first mismatch
    }
}

void test_weights_chunks() {
    using namespace wldr;
    for (uint64_t rem : std::vector<uint64_t>{1, 4, kChunk - 4, kChunk, kChunk + 4, 3 * kChunk + 12}) {
        CHECK(chunk_len(rem) > 0 && chunk_len(rem) <= rem && chunk_len(rem) <= kChunk);
        if (rem % 4 == 0) CHECK(chunk_len(rem) % 4 == 0);
        if (rem % 2 == 0) CHECK(chunk_len(rem) % 2 == 0);
    }
    for (uint64_t total : {8ull, 100ull, 4096ull}) {  // loop conserves done+rem=total
        uint64_t done = 0, rem = total;
        while (rem) { const uint64_t n = chunk_len(rem, 36); done += n; rem -= n; CHECK(done + rem == total); }
        CHECK(done == total);
    }
    CHECK(kStageOut == 2 * kChunk);
    CHECK(kChunk / 4 * 2 <= kStageOut && kChunk / 2 * 4 <= kStageOut);
    // Real transforms: chunk size must not change the result; destination ranges contiguous.
    testh::Rng rng(5);
    std::vector<uint8_t> f32b(4 * 1000), f16b(2 * 1001);
    for (auto& b : f32b) b = (uint8_t)rng.u32();
    for (auto& b : f16b) b = (uint8_t)rng.u32();
    std::vector<uint8_t> ref_bf(2 * 1000), ref_w(4 * 1001);
    CHECK(transform_chunked(Xform::Bf16High, f32b.data(), f32b.size(), ref_bf.data(), 1ull << 30) == ref_bf.size());
    CHECK(transform_chunked(Xform::WidenF16, f16b.data(), f16b.size(), ref_w.data(), 1ull << 30) == ref_w.size());
    for (size_t i = 0; i < 1000; ++i) {                       // bf16 = high half of the f32
        uint16_t h; std::memcpy(&h, ref_bf.data() + 2 * i, 2);
        uint32_t u; std::memcpy(&u, f32b.data() + 4 * i, 4);
        CHECK(h == (uint16_t)(u >> 16));
    }
    for (size_t i = 0; i < 1001; ++i) {                       // widening matches the scalar converter
        uint16_t h; std::memcpy(&h, f16b.data() + 2 * i, 2);
        float f; std::memcpy(&f, ref_w.data() + 4 * i, 4);
        const float e = f16::half_to_float(h);
        CHECK(std::memcmp(&f, &e, 4) == 0);
    }
    for (uint64_t chunk : {4ull, 8ull, 12ull, 36ull, 400ull, 4000ull}) {
        std::vector<uint8_t> a(ref_bf.size()), b(ref_w.size()), c(f32b.size());
        CHECK(transform_chunked(Xform::Bf16High, f32b.data(), f32b.size(), a.data(), chunk) == a.size());
        CHECK(transform_chunked(Xform::WidenF16, f16b.data(), f16b.size(), b.data(), chunk) == b.size());
        CHECK(transform_chunked(Xform::Copy, f32b.data(), f32b.size(), c.data(), chunk) == c.size());
        CHECK(a == ref_bf && b == ref_w && c == f32b);
    }
    std::vector<uint8_t> tmp(16);
    CHECK(transform_chunked(Xform::Bf16High, f32b.data(), 6, tmp.data()) == UINT64_MAX);   // element would be split
    CHECK(transform_chunked(Xform::WidenF16, f16b.data(), 3, tmp.data()) == UINT64_MAX);
    CHECK(transform_chunked(Xform::Copy, f32b.data(), 4, tmp.data(), 3) == UINT64_MAX);    // chunk not multiple of 4
}

void test_weights_segments_place() {
    using namespace wldr;
    std::vector<Seg> segs;
    for (uint64_t cb : {0ull, 64ull, 4096ull})
        for (uint64_t sb : {0ull, 8ull, 256ull})
            for (uint64_t ob : {0ull, 16ull}) {
                CHECK(plan_quantized(cb, sb, ob, false, segs));
                CHECK(seg_dst_total(segs) == cb + sb + ob && seg_src_total(segs) == cb + sb + ob);
                CHECK(plan_quantized(cb, sb, ob, true, segs));
                CHECK(seg_dst_total(segs) == cb + sb + ob);                 // dst totals equal index total
                CHECK(seg_src_total(segs) == cb + sb / 2 + ob);
                CHECK(segs[1].x == Xform::WidenF16 && segs[1].src_bytes * 2 == segs[1].dst_bytes);
                CHECK(segs[0].dst_off + segs[0].dst_bytes == segs[1].dst_off && segs[1].dst_off + segs[1].dst_bytes == segs[2].dst_off);
                CHECK(segs[0].src_off + segs[0].src_bytes == segs[1].src_off && segs[1].src_off + segs[1].src_bytes == segs[2].src_off);
            }
    CHECK(!plan_quantized(10, 7, 3, true, segs) && segs.empty());               // odd fp16 scale plane rejected
    CHECK(src_scale_bytes(7, true) * 2 != 7);
    CHECK(plan_quantized(10, 7, 3, false, segs));                               // odd is fine without fp16
    CHECK(promoted_ok(100, 400, 200)); CHECK(!promoted_ok(100, 400, 400)); CHECK(!promoted_ok(100, 399, 199));
    testh::Rng rng(9);
    for (int it = 0; it < 3000; ++it) {
        const size_t n = 1 + rng.below(20);
        std::vector<uint64_t> sz(n); std::vector<bool> skip(n);
        for (size_t i = 0; i < n; ++i) { sz[i] = rng.below(5000); skip[i] = rng.below(3) == 0; }
        const auto p = place(sz, skip);
        CHECK(p.total % 256 == 0);
        uint64_t expect_end = 0;
        for (size_t i = 0; i < n; ++i) {
            CHECK(align_up256(sz[i]) >= sz[i] && align_up256(sz[i]) % 256 == 0);
            if (skip[i]) { CHECK(p.off[i] == UINT64_MAX); continue; }
            CHECK(p.off[i] % 256 == 0);
            CHECK(p.off[i] == expect_end);                                      // index order, compacted, no overlap
            CHECK(p.off[i] + sz[i] <= p.total);                                 // never overflows pool
            expect_end += align_up256(sz[i]);
        }
        CHECK(expect_end == p.total);
    }
}

void test_arena() {
    using namespace parena;
    testh::Rng rng(21);
    for (int it = 0; it < 5000; ++it) {
        const uint64_t bytes = rng.u32() % 100000, slice = 1 + rng.below(5000);
        CHECK(slice_covered(bytes, slice) <= bytes);
        CHECK(slice_tail(bytes, slice) < slice);
        CHECK(slice_covered(bytes, slice) + slice_tail(bytes, slice) == bytes);
        CHECK(slice_count(bytes, slice) * slice == slice_covered(bytes, slice));
    }
    std::vector<uint8_t> buf(10007);
    for (auto& b : buf) b = (uint8_t)rng.u32();
    const uint64_t whole = fnv1a64(buf.data(), buf.size());
    for (uint64_t c : {1ull, 2ull, 3ull, 7ull, 64ull, 4096ull, 10007ull, 50000ull}) CHECK(fnv1a64_chunked(buf.data(), buf.size(), c) == whole);
    const uint8_t abc[3] = {'a', 'b', 'c'};
    CHECK(fnv1a64(nullptr, 0) == kFnvOffset);                                    // hashing nothing keeps the seed
    const uint8_t a1[1] = {'a'};
    CHECK(fnv1a64(a1, 1) == 0xaf63dc4c8601ec8cull);                              // published FNV-1a 64 test vector
    CHECK(fnv1a64(abc, 3) == fnv1a64(abc + 1, 2, fnv1a64(abc, 1)));              // fold property
    CHECK(fnv1a64(buf.data(), 0, 12345) == 12345);
    for (uint64_t X : {1ull, 640ull, 1382400ull * 3})
        for (uint64_t L = 0; L < 100; ++L) {
            CHECK(layer_off(1, X, L) + X == layer_off(1, X, L + 1));            // adjacent, disjoint
            CHECK(layer_off(3, X, L) == L * layer_bytes(3, X));
        }
    for (int it = 0; it < 2000; ++it) {
        const uint64_t total = rng.below(100000), chunk = 1 + rng.below(3000);
        uint64_t pos = 0, rem = total;
        while (rem) { const uint64_t n = read_step(rem, chunk); CHECK(n > 0 && n <= rem); pos += n; rem -= n; CHECK(pos + rem == total); }
        CHECK(moved_per_pass(total, chunk) <= total);
        CHECK((moved_per_pass(total, chunk) == total) == (total % chunk == 0));
    }
}
