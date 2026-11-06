// test_ud77.cpp - LAYER 1 invariants of the seven unit-3 modules, checked on the production code:
// PinnedArenaLoader (host formulas), ArenaCarving (core_layer gdn/moe arenas), MemoryPlanner (plan.hpp / memory_planner.hpp),
// ExpertBlobPacking (ref_pack_builder.hpp), LazyExpertLoader (gguf_lazy_loader.hpp, generate_cli model_shards),
// ExpertSourceDispatch (core_expert_source, expert_dispatch_contract) and PleReaderPaging (ple_reader.hpp, real file I/O).
#include <unistd.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <variant>
#include <vector>

#include "baloon/core_expert_source.hpp"
#include "baloon/core_layer.hpp"
#include "baloon/core_pinned.hpp"
#include "baloon/expert_dispatch_contract.hpp"
#include "baloon/expert_source_dispatch.hpp"
#include "baloon/generate_cli.hpp"
#include "baloon/gguf_lazy_loader.hpp"
#include "baloon/memory_planner.hpp"
#include "baloon/pinned_arena.hpp"
#include "baloon/plan.hpp"
#include "baloon/ple_reader.hpp"
#include "baloon/ref_pack_builder.hpp"

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

using namespace baloon;
static std::mt19937_64 rng(77);
static std::string tmpdir() { char t[] = "/tmp/ud77_XXXXXX"; return mkdtemp(t); }

// ---------------------------------------------------------------- PinnedArenaLoader (host formulas)
static void test_pinned() {
    // chunking is transparent: chunked hash == whole hash, any chunk size
    std::vector<uint8_t> buf(10007);
    for (auto& b : buf) b = (uint8_t)rng();
    const uint64_t whole = parena::fnv1a64(buf.data(), buf.size(), 1469598103934665603ull);
    for (uint64_t ch : {1ull, 7ull, 64ull, 4096ull, 10007ull, 99999ull})
        CHECK(parena::fnv1a64_chunked(buf.data(), buf.size(), ch, 1469598103934665603ull) == whole);
    CHECK(parena::fnv1a64(buf.data(), 0, 5) == 5);                       // empty input keeps the seed
    CHECK(1469598103934665603ull != 14695981039346656037ull);            // the dropped-digit quirk seed is not the standard one
    // layer tiling
    for (uint64_t L = 0; L < 48; ++L) CHECK(parena::layer_off(512, 1382400, L + 1) == parena::layer_off(512, 1382400, L) + 512 * 1382400);
    CHECK(parena::layer_off(512, 1382400, 48) == 33973862400ull);
    // sliced registration covers the arena exactly
    for (int i = 0; i < 2000; ++i) {
        const uint64_t bytes = rng() % 100000 + 1, slice = rng() % 5000 + 1;
        CHECK(parena::slice_covered(bytes, slice) <= bytes);
        CHECK(parena::slice_tail(bytes, slice) < slice);
        CHECK(parena::slice_covered(bytes, slice) + parena::slice_tail(bytes, slice) == bytes);
    }
    // read-loop accounting through the host arena model: bytes consumed + remaining invariant, loop terminates
    core::PinnedArena a; CHECK(a.init(100003, 4096));
    std::vector<uint8_t> src(100003); for (auto& b : src) b = (uint8_t)rng();
    core::ExpertReader rd = [&](uint64_t off, uint64_t n, uint8_t* dst) { if (off + n > src.size()) return false; std::memcpy(dst, src.data() + off, n); return true; };
    core::LoadStats st; CHECK(core::load_span(a, rd, 0, src.size(), 0, 777, st));
    CHECK(st.bytes == src.size() && st.reads == (src.size() + 776) / 777);
    CHECK(std::memcmp(a.data(), src.data(), src.size()) == 0);
    CHECK(st.checksum == parena::fnv1a64(src.data(), src.size()));
}

// ---------------------------------------------------------------- ArenaCarving
static void test_carving() {
    using namespace core;
    // align16
    for (uint64_t n : {0ull, 1ull, 15ull, 16ull, 17ull, 4095ull}) {
        const uint64_t a = align_up16(n);
        CHECK(a >= n && a % 16 == 0 && a < n + 16 && align_up16(a) == a);
    }
    // real GDN geometry
    uint64_t s[kGdnArenaRegions]; gdn_arena_sizes(kRealGdnGeometry, s);
    CHECK(s[9] == 24576 && s[14] == 3145728 && s[15] == 122880);
    CHECK(gdn_arena_bytes(kRealGdnGeometry) == 3490096);
    alignas(16) static uint8_t arena[3490096 + 64];
    Regions<kGdnArenaRegions> r;
    CHECK(gdn_arena_init(r, arena, sizeof arena, kRealGdnGeometry) == gdn_arena_bytes(kRealGdnGeometry));   // init consumes exactly bytes
    for (int i = 0; i < kGdnArenaRegions; ++i) {
        CHECK(r.off[i] % 16 == 0 && r.size[i] == s[i]);
        if (i + 1 < kGdnArenaRegions) CHECK(r.off[i] + s[i] <= r.off[i + 1]);                           // no overlap
        if (i + 1 < kGdnArenaRegions) CHECK(r.off[i + 1] - r.off[i] == align_up16(s[i]));
    }
    CHECK(r.off[kGdnArenaRegions - 1] + s[kGdnArenaRegions - 1] <= r.bytes);
    Regions<kGdnArenaRegions> r2;
    CHECK(gdn_arena_init(r2, arena, gdn_arena_bytes(kRealGdnGeometry) - 1, kRealGdnGeometry) == UINT64_MAX);   // a short arena is refused
    CHECK(gdn_arena_init(r2, arena + 1, sizeof arena - 1, kRealGdnGeometry) == UINT64_MAX);                    // misaligned base is refused
    // random geometries: bytes == init cursor end, total >= raw sum
    alignas(16) static uint8_t big[1 << 22];
    for (int i = 0; i < 300; ++i) {
        GdnGeometry g{(rng() % 8 + 1) * 256, (rng() % 8 + 1) * 64, (rng() % 8 + 1) * 256, rng() % 16 + 1, (rng() % 4 + 1) * 8, rng() % 4 + 2};
        uint64_t z[kGdnArenaRegions]; gdn_arena_sizes(g, z);
        uint64_t raw = 0; for (uint64_t v : z) raw += v;
        const uint64_t b = gdn_arena_bytes(g);
        CHECK(b >= raw && b < raw + 16 * kGdnArenaRegions);
        if (b <= sizeof big) { Regions<kGdnArenaRegions> q; CHECK(gdn_arena_init(q, big, sizeof big, g) == b); }
        const uint64_t nq = rng() % 5000, ne = rng() % 600 + 1, k = rng() % 12 + 1;
        const uint64_t mb = moe_arena_bytes(2560, ne, k, nq);
        Regions<kMoeArenaRegions> m;
        CHECK(moe_arena_init(m, big, sizeof big, 2560, ne, k, nq) == mb);
        CHECK(kMoeArenaRegions == 9);
    }
    {   // MoE arena: nine regions, independently re-derived sizes (aligned sum) at the real shape with a shared-expert scratch of 1000 B
        const uint64_t nE = 2560, ne = 512, k = 10, sh = 1000;
        const uint64_t want[9] = {nE * 2, nE * 2, ne * 4, k * 4, k * 4, nE * 4, sh, nE / 32 * 34, nE / 256 * 292};
        uint64_t sum = 0; for (uint64_t v : want) sum += (v + 15) / 16 * 16;
        CHECK(moe_arena_bytes(nE, ne, k, sh) == sum);
        uint64_t z[kMoeArenaRegions]; moe_arena_sizes(nE, ne, k, sh, z);
        for (int i = 0; i < 9; ++i) CHECK(z[i] == want[i]);
        Regions<kMoeArenaRegions> m; CHECK(moe_arena_init(m, big, sizeof big, nE, ne, k, sh) == sum);
        CHECK(moe_arena_init(m, big, sum - 1, nE, ne, k, sh) == UINT64_MAX);
    }
    // the shared RoPE table is paid once
    CHECK(qsa_states_total(12, 1000, 64) == 12 * 1000 + 64 && qsa_states_total(12, 1000, 64) + 11 * 64 == 12 * (1000 + 64));
}

// ---------------------------------------------------------------- MemoryPlanner
static void test_planner() {
    planner::Geometry g;
    CHECK(planner::state_bytes(g) == 117669888ull);
    CHECK(planner::state_bytes(g) / 1382400 == 85);
    CHECK(planner::kv_bytes_per_token(g) == 13056ull);
    CHECK(planner::kExpertBlobBytes == 1382400);
    for (int i = 0; i < 20000; ++i) {
        planner::Costs c; c.expert_blob = rng() % 2000 + 1; c.dense = rng() % 5000; c.embd = rng() % 5000; c.mtp = rng() % 500;
        c.workspace = rng() % 500; c.state = rng() % 5000;
        const uint64_t ctx = rng() % 50, pool = rng() % 400000;
        const uint64_t kv = 17 * ctx, fixed = c.dense + c.embd + c.mtp + c.workspace + c.state;
        const auto p = planner::make_plan(ctx, g, c, pool);
        const uint64_t kvt = planner::kv_bytes_per_token(g) * ctx;
        const bool should_close = kvt + fixed <= pool && (pool - kvt - fixed) / c.expert_blob != 0;
        (void)kv;
        CHECK(p.has_value() == should_close);                       // refuses exactly when it cannot close
        if (!p) continue;
        CHECK(p->kv_bytes + fixed + p->cache_bytes == pool);        // plan closes
        CHECK(p->cache_slots >= 1 && p->cache_slots * c.expert_blob <= p->cache_bytes);
        CHECK(p->vram_budget == pool && p->vram_used <= pool);
        CHECK(p->vram_used == p->kv_bytes + fixed + c.expert_blob * p->cache_slots);
    }
    // plan.hpp front-end: variant of Plan / DoesNotClose, same arithmetic
    plan::Geometry pg; plan::Costs pc; pc.expert_blob = 1382400; pc.dense = 1ull << 30;
    const auto ok = plan::make_plan(4096, pg, pc, 24ull << 30);
    CHECK(std::holds_alternative<plan::Plan>(ok));
    const auto bad = plan::make_plan(4096, pg, pc, 1ull << 20);
    CHECK(std::holds_alternative<plan::DoesNotClose>(bad));
    CHECK(!plan::to_string(std::get<plan::DoesNotClose>(bad)).empty());
    CHECK(plan::vram_pool_bytes(10, 20) == 0);                      // truncating subtraction, never wraps
}

// ---------------------------------------------------------------- ExpertBlobPacking
static void test_packing() {
    using namespace pack;
    CHECK(kGuCodes == 819200 && kDCodes == 409600 && kGuScales == 102400 && kDScales == 51200);
    CHECK(kBlobBytes == 1382400 && kRoleBytes == 460800);
    CHECK(kGuCodesOff == 0 && kDCodesOff == 819200 && kGuScalesOff == 1228800 && kDScalesOff == 1331200);
    CHECK(kGuCodesOff < kDCodesOff && kDCodesOff < kGuScalesOff && kGuScalesOff < kDScalesOff && kDScalesOff + kDScales == kBlobBytes);
    // theory order: gate_up codes | down codes | gate_up scales | down scales  => 0, 819200, 1228800, 1331200
    CHECK(kBlobBytes == 819200 + 409600 + 102400 + 51200);
    CHECK(kExpertsPerLayer * kBlobBytes == 707788800ull);
    CHECK(48 * kExpertsPerLayer * kBlobBytes == 33973862400ull);
    for (int i = 0; i < 5000; ++i) {
        const uint64_t l = rng() % 48, e = rng() % 512, l2 = rng() % 48, e2 = rng() % 512;
        CHECK((expert_flat_index(l, e) == expert_flat_index(l2, e2)) == (l == l2 && e == e2));      // injective
        CHECK(expert_blob_offset(l, e) + kBlobBytes <= 33973862400ull);
    }
    {   // build -> rebuild round trip on random roles, and the blob regions hold what the offsets say
        std::vector<uint8_t> g(kRoleBytes), u(kRoleBytes), d(kRoleBytes), blob;
        for (auto& b : g) { b = (uint8_t)rng(); }
        for (auto& b : u) { b = (uint8_t)rng(); }
        for (auto& b : d) { b = (uint8_t)rng(); }
        CHECK(build_expert_blob(g, u, d, &blob) && blob.size() == kBlobBytes && verify_expert_blob(blob, g, u, d));
        CHECK(std::memcmp(blob.data() + kGuCodesOff, g.data() + 2, 16) == 0);                       // gate row 0 -> slot 0
        CHECK(std::memcmp(blob.data() + kGuCodesOff + (kScGu) * 16, u.data() + 2, 16) == 0);        // up row 0 -> slot 1 (row 1 of the interleave)
        CHECK(std::memcmp(blob.data() + kDScalesOff, d.data(), 2) == 0);
        CHECK(std::memcmp(blob.data() + kGuScalesOff, g.data(), 2) == 0);
        std::vector<uint8_t> bad = g; bad.pop_back(); CHECK(!build_expert_blob(bad, u, d, &blob));
    }
    // interleave: gate row r -> 2r, up row r -> 2r+1 covers [0,1280) exactly once
    std::vector<int> seen(1280, 0);
    for (int r = 0; r < 640; ++r) { ++seen[2 * r]; ++seen[2 * r + 1]; }
    for (int c : seen) CHECK(c == 1);
}

// ---------------------------------------------------------------- LazyExpertLoader
static void test_lazy() {
    CHECK(gguf::expert_bytes(2560ull * 640, 64, 18) == 460800);
    CHECK(gguf::row_bytes(160, 32, 18) == 90 && gguf::ple_row_bytes() == 90);
    CHECK(gguf::ple_table_bytes() == 28800138240ull && gguf::kPleVocab == 320001536ull);
    for (int i = 0; i < 3000; ++i) {
        const uint64_t n = rng() % 64 + 1, per = (rng() % 20 + 1) * 64, e = rng() % n, e2 = rng() % n;
        const uint64_t len = gguf::expert_bytes(per, 64, 18);
        CHECK(gguf::expert_start(e, per, 64, 18) + len <= n * len);                                  // in bounds
        CHECK(gguf::expert_start(e, per, 64, 18) == gguf::expert_start_py(e, per, 64, 18));
        if (e < e2) CHECK(gguf::expert_start(e, per, 64, 18) + len <= gguf::expert_start(e2, per, 64, 18));   // disjoint
    }
    CHECK(gguf::expert_start(0, 4096, 64, 18) == 0);
    CHECK(gguf::bf16_widen_bits(0x3f80) == 0x3f800000u);                                               // widening is exact (<<16)
    CHECK(gguf::shards_complete(3, 3) && !gguf::shards_complete(2, 3));
    // shard discovery through the production model_shards: exactly `total` siblings
    const std::string d = tmpdir();
    auto touch = [&](const std::string& n) { FILE* f = std::fopen((d + "/" + n).c_str(), "wb"); if (f) std::fclose(f); };
    touch("m-00001-of-00003.gguf"); touch("m-00002-of-00003.gguf");
    std::vector<std::string> out; std::string err;
    CHECK(!gencli::model_shards(d + "/m-00001-of-00003.gguf", out, err));                          // one sibling missing -> refused
    touch("m-00003-of-00003.gguf");
    CHECK(gencli::model_shards(d + "/m-00002-of-00003.gguf", out, err) && out.size() == 3);
    CHECK(out[0].find("00001-of-00003") != std::string::npos && out[2].find("00003-of-00003") != std::string::npos);
    CHECK(gencli::model_shards(d + "/plain.gguf", out, err) && out.size() == 1);                   // unsharded name is a single file
    CHECK(!gencli::model_shards("", out, err));
}

// ---------------------------------------------------------------- ExpertSourceDispatch
struct Sink : core::GpuPlanSink { int calls = 0; void plan(int64_t, const core::ExpertDispatch&) override { ++calls; } };
static void test_dispatch() {
    using namespace core;
    CHECK(source_dispatch::kPackBytes == 33973862400ull && source_dispatch::kPackBlobs == 24576);
    CHECK(48ull * 10 * 1382400 == 663552000ull && 2560 / 32 * 34 == 2720);
    // file acceptance: exactly nl*ne*blob bytes
    const std::string d = tmpdir();
    const int64_t NL = 3, NE = 5, BL = 24;
    auto mk = [&](const std::string& n, size_t sz) {
        std::vector<uint8_t> v(sz); for (size_t i = 0; i < sz; ++i) v[i] = (uint8_t)(i * 31 + 7);
        FILE* f = std::fopen((d + "/" + n).c_str(), "wb"); std::fwrite(v.data(), 1, sz, f); std::fclose(f); return d + "/" + n;
    };
    const size_t exact = (size_t)(NL * NE * BL);
    { FileExpertSource f; CHECK(f.open(mk("ok.bin", exact), NL, NE, BL)); CHECK(f.file_bytes() == exact);
      for (int64_t l = 0; l < NL; ++l) for (int64_t e = 0; e < NE; ++e) {
          const uint8_t* b = f.blob(l, e); CHECK(b != nullptr);
          if (b) CHECK(b[0] == (uint8_t)(((l * NE + e) * BL) * 31 + 7));                           // blob index = layer*NE + expert, no padding
      }
      CHECK(f.blob(NL, 0) == nullptr && f.blob(0, NE) == nullptr && f.blob(-1, 0) == nullptr); }
    { FileExpertSource f; CHECK(!f.open(mk("short.bin", exact - 1), NL, NE, BL)); }
    { FileExpertSource f; CHECK(!f.open(mk("long.bin", exact + 1), NL, NE, BL)); }
    { ArenaExpertSource a; CHECK(a.open(mk("arena.bin", exact), NL, NE, BL)); CHECK(a.arena_bytes() >= exact);
      const uint8_t* b = a.blob(2, 4); CHECK(b != nullptr); }
    { ArenaExpertSource a; CHECK(!a.open(mk("arena_bad.bin", exact + 3), NL, NE, BL)); }
    // hit/miss split: every router index handled by exactly one of CPU job / GPU hit / PCIe
    FileExpertSource f; f.open(d + "/ok.bin", NL, NE, BL);
    Sink sink;
    for (int it = 0; it < 3000; ++it) {
        const int k = (int)(rng() % 10) + 1; std::vector<bool> hit(k); std::vector<int32_t> ex(k);
        for (int i = 0; i < k; ++i) { hit[i] = rng() & 1; ex[i] = (int32_t)(rng() % NE); }
        const int pn = (int)(rng() % 257);
        ExpertDispatch o;
        CHECK(expert_pool_dispatch(f, &sink, (int64_t)(rng() % NL), ex.data(), hit, pn, o));
        std::vector<int> cover(k, 0);
        for (auto& j : o.jobs) { ++cover[j.out_index]; CHECK(!hit[j.out_index] && j.blob); }
        for (int32_t i : o.gpu_hits) { ++cover[i]; CHECK(hit[i]); }
        for (auto& j : o.gpu_pcie) { ++cover[j.out_index]; CHECK(!hit[j.out_index] && j.blob); }
        for (int c : cover) CHECK(c == 1);
        uint64_t nmiss = 0; for (bool h : hit) nmiss += !h;
        CHECK(o.gpu_pcie.size() == (nmiss * (uint64_t)pn) / 256 && o.gpu_pcie.size() <= nmiss);
    }
    CHECK(sink.calls == 3000);
    { ExpertDispatch o; std::vector<bool> h{false}; int32_t e = 99;
      CHECK(!expert_pool_dispatch(f, nullptr, 0, &e, h, 0, o) && o.failed);                         // out-of-range expert latches failed
      CHECK(!expert_hit_run(h, 257, o)); }
    // hit-path wiring is all-or-nothing and needs the fp32 activation scales
    expert_dispatch::HitCfg c{true, true, true, true, true, true, true, true, true, true};
    CHECK(expert_dispatch::hits_ready(c));
    c.xq_scale = false; CHECK(!expert_dispatch::hits_ready(c));
    c.xq_scale = true; c.d_slot = false; CHECK(!expert_dispatch::hits_ready(c));
    // failure latch: clearing the message never re-arms
    source_dispatch::DispatchState ds; ds.step(true); CHECK(!ds.failed);
    ds.step(false, "boom"); CHECK(ds.failed && ds.msg == "boom"); ds.clear_message(); CHECK(ds.failed); ds.step(true); CHECK(ds.failed);
    CHECK(source_dispatch::pcie_share(1000, 256) == 1000 && source_dispatch::pcie_share(7, 0) == 0);
}

// ---------------------------------------------------------------- PleReaderPaging
static void test_ple() {
    using namespace pler;
    for (uint64_t at = 0; at < 3 * PAGE + 200; ++at) {
        const uint64_t first = first_of(at), len = length_of(at), in = at - first;
        CHECK(first % PAGE == 0 && first <= at && len % PAGE == 0 && (len == PAGE || len == 2 * PAGE));
        CHECK(in < PAGE && in + ROW_BYTES <= len);                                                  // row bytes lie inside the job
        if (in + ROW_BYTES <= PAGE) CHECK(len == PAGE); else CHECK(len == 2 * PAGE);
    }
    for (int i = 0; i < 20000; ++i) {
        const uint64_t at = rng() % (28800138240ull + 5000000000ull);
        CHECK(at - first_of(at) + ROW_BYTES <= length_of(at));
    }
    CHECK(EMPTY > 320001536ull && WAYS == 8 && ROW_BYTES == 90);
    // tickets wrap but never yield 0
    CHECK(next_ticket(0xFFFFFFFFu) == 1 && next_ticket(0) == 1 && next_ticket(5) == 6);
    // cache geometry
    for (uint64_t rows : {0ull, 7ull, 8ull, 100ull, 4096ull}) {
        RowCache c(rows); CHECK(c.capacity() <= rows && c.sets() == rows / 8);
        for (int i = 0; i < 200 && c.sets(); ++i) CHECK(c.set_index(rng()) < c.sets());
    }
    { RowCache c(64); std::vector<uint8_t> row(ROW_BYTES, 0xAB), out(ROW_BYTES);
      std::vector<uint64_t> same;                                                                   // 9 rows landing in one set
      for (uint64_t r = 0; same.size() < 9; ++r) if (c.set_index(r) == c.set_index(1)) same.push_back(r);
      for (size_t i = 0; i < 8; ++i) { row[0] = (uint8_t)i; c.insert(same[i], row.data()); }
      for (size_t i = 0; i < 8; ++i) { CHECK(c.lookup(same[i], out.data()) && out[0] == (uint8_t)i); }
      c.insert(same[8], row.data());                                                                // 9th evicts the round-robin victim (way 0)
      CHECK(!c.lookup(same[0], out.data()) && c.lookup(same[8], out.data()) && c.lookup(same[1], out.data()));
      CHECK(c.resident() == 8); }
    // real reader on a file: dedup by page, straddle jobs, slot conservation, ticket wrap, failure on collect of unknown ticket
    const std::string d = tmpdir(), path = d + "/ple.bin";
    const uint64_t off = 4096 + 50, nrows = 2000, table = nrows * ROW_BYTES;
    std::vector<uint8_t> file(off + table + 100);
    for (size_t i = 0; i < file.size(); ++i) file[i] = (uint8_t)((i * 2654435761u) >> 13);
    { FILE* f = std::fopen(path.c_str(), "wb"); std::fwrite(file.data(), 1, file.size(), f); std::fclose(f); }
    PleReader r;
    CHECK(!r.open(path.c_str(), off, nrows + 100000, 64, 4));                                       // table past end of file is refused
    CHECK(r.open(path.c_str(), off, nrows, 64, 4) && r.is_open() && r.cache_capacity() <= 64);
    std::vector<uint64_t> rows;
    for (int i = 0; i < 300; ++i) rows.push_back(rng() % nrows);
    std::vector<uint8_t> out(rows.size() * ROW_BYTES, 0);
    const uint32_t t = r.issue(rows.data(), rows.size(), out.data());
    CHECK(t != 0 && r.collect(t));
    for (size_t i = 0; i < rows.size(); ++i) CHECK(std::memcmp(&out[i * ROW_BYTES], &file[off + rows[i] * ROW_BYTES], ROW_BYTES) == 0);
    auto st = r.stats();
    CHECK(st.free_slots + st.inflight == st.max_inflight && st.max_inflight == 4);
    CHECK(st.jobs_issued <= rows.size() && st.jobs_done == st.jobs_issued);
    { // two rows in one page share one job
      PleReader q; CHECK(q.open(path.c_str(), off, nrows, 0, 4));
      uint64_t two[2] = {0, 1}; uint8_t o2[2 * ROW_BYTES];
      const uint32_t tt = q.issue(two, 2, o2); CHECK(tt != 0 && q.collect(tt)); CHECK(q.stats().jobs_issued == 1);
      CHECK(!q.collect(0xDEADBEEFu)); }                                                            // never issued -> fails
    { PleReader q; CHECK(q.open(path.c_str(), off, nrows, 0, 2)); q.debug_set_next_ticket(0xFFFFFFFEu);
      uint64_t one[1] = {3}; uint8_t o1[ROW_BYTES];
      for (int i = 0; i < 4; ++i) { const uint32_t tt = q.issue(one, 1, o1); CHECK(tt != 0); CHECK(q.collect(tt)); } }
    CHECK(planner::kExpertBlobBytes == 1382400);
    r.close(); CHECK(!r.is_open());
    CHECK(320001536ull * 90 == 28800138240ull);
}

int main() {
    test_pinned(); test_carving(); test_planner(); test_packing(); test_lazy(); test_dispatch(); test_ple();
    std::printf("%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail) { std::printf("UD77 TESTS FAILED\n"); return 1; }
    std::printf("UD77 TESTS PASSED\n");
    return 0;
}
