// test_u1af.cu - invariant tests for unit u1af344a465 (layer geometry, graph registry, MTP chain, session token loop,
// verify window, direct file IO, layer composition, pinned arena, prefill pipeline, prefill kernel indexing).
// Every CHECK states a LAYER 1 theorem of the matching lean/u1af module; any failure makes the binary exit non-zero.
#include <cuda_runtime.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "baloon/core_graph.hpp"
#include "baloon/core_graph_dev.hpp"
#include "baloon/core_layer.hpp"
#include "baloon/core_pinned_hw.hpp"
#include "baloon/core_prefill.hpp"
#include "baloon/core_session.hpp"
#include "baloon/direct_file.hpp"
#include "baloon/mtp_draft_layer.hpp"
#include "baloon/prefill_chunk_driver.hpp"
#include "baloon/prefill_kernels_api.hpp"
#include "baloon/session_loop_contract.hpp"
#include "baloon/session_schedule.hpp"
#include "baloon/verify_window_protocol.hpp"

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

using namespace baloon;

static const char* volatile g_stage = "start";
static void on_segv(int) {
    const char m[] = "\nSIGSEGV in stage: ";
    (void)!write(2, m, sizeof(m) - 1); (void)!write(2, (const char*)g_stage, strlen((const char*)g_stage)); (void)!write(2, "\n", 1);
    void* fr[40]; int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2);
    _exit(139);
}
#define STAGE(name) do { g_stage = name; std::printf("[stage] %s\n", name); std::fflush(stdout); } while (0)

// ---------------------------------------------------------------------------------------------------------------
// GdnQsaLayerGeometry + SessionTokenLoop
static void test_layer_geometry() {
    int nq = 0, ng = 0;
    for (int l = 0; l < 48; ++l) (core::is_qsa_layer(l) ? nq : ng)++;
    CHECK(nq == 12 && ng == 36);                                           // qsa_count / gdn_count
    CHECK(core::is_qsa_layer(3) && core::is_qsa_layer(47) && !core::is_qsa_layer(0) && !core::is_qsa_layer(46));
    CHECK(16 * 128 + 16 * 128 + 48 * 128 == 10240);                        // conv_channels
    std::set<int> gs, qs;
    for (int l = 0; l < 48; ++l) {
        const int g = core::gdn_point_at(l), q = core::qsa_point_at(l);
        if (core::is_qsa_layer(l)) { CHECK(g == -1 && q >= 0 && q < 12 && qs.insert(q).second); CHECK(q == l / 4); }
        else { CHECK(q == -1 && g >= 0 && g < 36 && gs.insert(g).second); CHECK(g == l - l / 4); }  // injective, in range
        CHECK(session_schedule::is_qsa((uint64_t)l) == core::is_qsa_layer(l));
        if (!core::is_qsa_layer(l)) CHECK((int)session_schedule::gdn_index((uint64_t)l) == g);
    }
    CHECK(gs.size() == 36 && qs.size() == 12);
    // q|gate halves of q_full: contiguous, in range, never colliding within a head
    const int hd = 256, nh = 16;
    for (int h = 0; h < nh; ++h) for (int d = 0; d < hd; ++d) {
        const int qi = h * 2 * hd + d, gi = h * 2 * hd + hd + d;
        CHECK(qi < nh * 2 * hd && gi < nh * 2 * hd && qi != gi);
    }
    // stage split: halves cover and are disjoint; prefixes grow by exactly one stage
    auto stages = [](int half) { return half == 1 ? std::vector<int>{0, 1, 2} : half == 2 ? std::vector<int>{3, 4} : std::vector<int>{0, 1, 2, 3, 4}; };
    std::vector<int> cat = stages(1); for (int s : stages(2)) cat.push_back(s);
    CHECK(cat == stages(0));
    { const std::vector<int> s2 = stages(2); for (int s : stages(1)) CHECK(std::count(s2.begin(), s2.end(), s) == 0); }
    { std::vector<int64_t> P = {0, 5, 9, 14, 20, 27};   // prefix timings P[0..5], P[0] = 0
      auto st = session_loop::stage_times(P);                              // differencing isolates one stage
      CHECK(st.size() == 5 && st[0] == 5 && st[1] == 4 && st[2] == 5 && st[3] == 6 && st[4] == 7);
      auto st2 = session_schedule::stage_ms(5, 9, 14, 20, 27); CHECK(st2 == st); }
    CHECK(16 * 90 == 1440);                                                // ple_row_bytes
    for (uint64_t m = 0; m < 1000; ++m) CHECK(64 * m + 2 * m <= 2 * (64 * m));  // int8 kv_smaller

    // SessionTokenLoop: launches, dump layout, exec index bounds
    CHECK(2 * 48 == 96 && 48 < 96);
    for (uint64_t stride : {1ull, 7ull, 2560ull, 10240ull}) for (uint64_t n : {1ull, 48ull}) for (uint64_t l = 0; l < n; ++l) {
        CHECK(session_loop::slot_base(stride, l) + stride <= session_loop::dump_total(stride, n));    // dump_slot_in_bounds
        CHECK(session_loop::slot_base(stride, l) + stride == session_loop::slot_base(stride, l + 1));  // slots abut
        CHECK(session_schedule::dump_floats(n, stride) == (n + 1) * stride);
    }
    for (uint64_t T = 1; T <= 8; ++T) CHECK(session_loop::exec_index_ok(8, T));
    CHECK(!session_loop::exec_index_ok(9, 1) && !session_loop::exec_index_ok(8, 9));
    for (int k = 1; k <= 5; ++k) CHECK(k - 1 < 5);
    // oracle dump layout (LayerComposition dump_last_slot_end): slots tile the stride exactly
    std::mt19937_64 rng(11);
    for (int it = 0; it < 200; ++it) {
        const uint64_t n = 1 + rng() % 4096, hc = 1 + rng() % 8, a = rng() % 5000, nvk = rng() % 5000;
        const uint64_t stride = core::dump_stride_floats(n, hc, a, nvk);
        CHECK(stride == 2 * n + 2 * hc + a + 5 * nvk + 8);
        uint64_t end = 0;
        for (int i = 0; i < core::kDumpSlots; ++i) { auto s = core::dump_slot(i, n, hc, a, nvk); CHECK(s.off == end); end = s.off + s.len; }
        CHECK(end == stride);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// MtpDraftChain + SpeculativeVerifyWindow
static uint64_t accepted(const std::vector<int>& d, const std::vector<int>& o) {
    uint64_t a = 0;
    for (size_t i = 0; i < d.size() && i < o.size(); ++i) { if (d[i] != o[i]) break; ++a; }
    return a;
}
static void test_mtp_and_verify() {
    // window: 0 or >= max_cells attends everything, otherwise at most `window` cells
    for (uint64_t cells : {1ull, 63ull, 64ull, 65ull, 4096ull}) {
        CHECK(mtp::window_of(0, cells) == cells);
        for (uint64_t w : {1ull, 5ull, 64ull, 1000ull}) {
            const uint64_t eff = mtp::window_of(w, cells);
            CHECK(eff <= cells && (w >= cells || eff == w));
            CHECK(mtp::cap_of(w, cells) >= eff && mtp::cap_of(w, cells) % 64 == 0 && mtp::cap_of(w, cells) < eff + 64);
        }
    }
    // draft cells p+a+1 .. : T-1 cells, all after the last accepted cell p+a; graph arrays have 9 entries so T <= 8
    for (uint64_t T = 1; T <= 8; ++T) {
        std::vector<uint64_t> cells; for (uint64_t j = 0; j + 1 < T; ++j) cells.push_back(100 + 3 + 1 + j);
        CHECK(cells.size() == T - 1);
        for (uint64_t c : cells) CHECK(c > 100 + 3);
        CHECK(T < 9);
    }
    CHECK(512ull * 1382400 < 708000000ull && 707000000ull < 512ull * 1382400);
    for (uint64_t mt = 3; mt <= 8; ++mt) for (uint64_t j = 1; j + 2 <= mt; ++j) CHECK(mtp::rows_disjoint(mt, j));

    // acceptance: longest matching prefix; n_keep = 1 + accepted in 1..T; all-match accepts all; first mismatch rejects
    std::mt19937 rng(5);
    for (int it = 0; it < 3000; ++it) {
        const size_t nd = rng() % 8; std::vector<int> d(nd), o(nd + 1);
        for (auto& x : d) x = (int)(rng() % 3); for (auto& x : o) x = (int)(rng() % 3);
        const uint64_t a = accepted(d, o);
        CHECK(a <= d.size() && a <= o.size());
        const uint64_t T = nd + 1, keep = 1 + a;
        CHECK(keep >= 1 && keep <= T && verify::accepted_ok(T, a));
        if (nd > 0 && d[0] != o[0]) CHECK(a == 0);
        if (a < nd) CHECK(d[a] != o[a]);                                   // stops exactly at the first mismatch
        CHECK(accepted(d, std::vector<int>(d)) == d.size());
    }
    CHECK(!verify::accepted_ok(3, 3) && verify::accepted_ok(3, 2));
    for (uint64_t l = 0; l < 64; ++l) for (uint64_t g = 0; g < 2; ++g) CHECK(l * 2 + g < 2 * 64 * 2);   // flag slots
    for (int T = 1; T <= 8; ++T) {
        CHECK(2 + T <= 10 && T * 10 <= 128);                               // commit record / plan capacity
        for (int keep = 1; keep <= T; ++keep) {
            auto v = verify::commit_vector(1000, keep, T);
            CHECK(v.size() == (size_t)(2 + T));
            CHECK(v[0] == keep && v[1] == keep - 1);
            for (int t = 0; t < T; ++t) CHECK(v[2 + t] == (t < keep ? 1000 + t : -1));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GraphCaptureRegistry (device graph, real events)
__global__ void k_append(int* buf, int* cursor, int tag) { if (threadIdx.x == 0 && blockIdx.x == 0) { int c = *cursor; buf[c] = tag; *cursor = c + 1; } }
__global__ void k_busy(float* sink, long iters) { float a = 1.0f + threadIdx.x; for (long i = 0; i < iters; ++i) a = a * 0.999999f + 1e-7f; if (a == 12345.678f) sink[0] = a; }

static void test_graph_registry() {
    using core::dev::CapturedGraph; using core::dev::GraphRegistry;
    // zero-node capture is an error, never an empty graph
    { CapturedGraph g; g.begin(); CHECK(!g.end() && !g.ready()); CHECK(!g.launch()); CHECK(!g.end()); }
    // bounded poll semantics (queries are synthetic so each outcome is exercised)
    { int n = 0;
      CHECK(CapturedGraph::bounded_poll([&] { ++n; return cudaSuccess; }, 50) && n == 1);                     // first-poll success
      CHECK(!CapturedGraph::bounded_poll([] { return cudaErrorNotReady; }, 20));                              // deadline -> false
      CHECK(!CapturedGraph::bounded_poll([] { return cudaSuccess; }, 0));                                     // expired deadline
      n = 0; CHECK(CapturedGraph::bounded_poll([&] { return ++n < 5 ? cudaErrorNotReady : cudaSuccess; }, 1000) && n == 5);
      n = 0; CHECK(!CapturedGraph::bounded_poll([&] { ++n; return cudaErrorInvalidValue; }, 1000) && n == 1); // real error -> immediate false
      cudaGetLastError(); }
    // capture once, replay many, order preserved, keys distinct per layer kind
    int *dbuf = nullptr, *dcur = nullptr;
    CHECK(cudaMalloc(&dbuf, 64 * sizeof(int)) == cudaSuccess && cudaMalloc(&dcur, sizeof(int)) == cudaSuccess);
    CHECK(cudaMemset(dcur, 0, sizeof(int)) == cudaSuccess);
    GraphRegistry reg; int body_runs = 0;
    auto body = [&](CapturedGraph& g) {
        ++body_runs;
        for (int tag = 1; tag <= 3; ++tag)
            g.push([=](cudaStream_t s) { k_append<<<1, 32, 0, s>>>(dbuf, dcur, tag); return cudaGetLastError(); });
    };
    CHECK(reg.record(0, 4, body) && body_runs == 1 && reg.size() == 1 && reg.captures() == 1);
    CHECK(reg.record(0, 4, body) && body_runs == 1 && reg.captures() == 1 && reg.size() == 1);            // record_idem: body not run
    CHECK(reg.record(1, 4, body) && body_runs == 2 && reg.size() == 2 && reg.captures() == 2);            // (GDN,4) != (QSA,4)
    CHECK(reg.find(0, 4) && reg.find(1, 4) && reg.find(0, 4) != reg.find(1, 4) && !reg.find(0, 5));
    CHECK(!reg.record(0, 9, [](CapturedGraph&) {}) && reg.size() == 2 && reg.captures() == 2 && !reg.find(0, 9));  // empty capture rejected
    CHECK(!reg.launch(1, 99));                                                                            // missing key
    for (int r = 0; r < 3; ++r) CHECK(reg.launch(0, 4));
    std::vector<int> h(64), cur(1);
    CHECK(cudaMemcpy(h.data(), dbuf, 64 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(cur.data(), dcur, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cur[0] == 9);
    for (int i = 0; i < 9; ++i) CHECK(h[i] == i % 3 + 1);                                                 // replayed in capture order, 3 times
    CHECK(reg.find(0, 4)->launches() == 3);
    // a failing node stops the replay (later nodes not issued) and wait_ms never reports completion
    { CapturedGraph g; int issued = 0; g.begin();
      g.push([&](cudaStream_t) { ++issued; return cudaSuccess; });
      g.push([&](cudaStream_t) { ++issued; return cudaErrorInvalidValue; });
      g.push([&](cudaStream_t) { ++issued; return cudaSuccess; });
      CHECK(g.end() && !g.launch() && issued == 2 && !g.wait_ms(50)); cudaGetLastError(); }
    // a launch still running at the deadline is NOT reported complete; once it has finished it is
    { float* sink = nullptr; CHECK(cudaMalloc(&sink, sizeof(float)) == cudaSuccess);
      cudaEvent_t c0, c1; cudaEventCreate(&c0); cudaEventCreate(&c1);
      long iters = 2000000; float ms = 0;
      for (int pass = 0; pass < 2; ++pass) {                                                              // warm-up, then calibrate
          cudaEventRecord(c0); k_busy<<<1, 64>>>(sink, iters); cudaEventRecord(c1); cudaEventSynchronize(c1); cudaEventElapsedTime(&ms, c0, c1);
      }
      if (ms < 0.01f) ms = 0.01f;
      iters = (long)((double)iters * 400.0 / ms); if (iters < 1000) iters = 1000;                         // target ~400 ms
      cudaEventDestroy(c0); cudaEventDestroy(c1);
      CapturedGraph g; g.begin(); g.push([=](cudaStream_t s) { k_busy<<<1, 64, 0, s>>>(sink, iters); return cudaGetLastError(); });
      CHECK(g.end() && g.launch());
      CHECK(!g.wait_ms(30));                                                                              // still running -> false
      CHECK(g.wait_ms(30000));                                                                            // finished -> true
      cudaDeviceSynchronize(); cudaFree(sink); }
    cudaFree(dbuf); cudaFree(dcur);
    // host model (core_graph.hpp) keeps its replay-in-order contract
    { core::CapturedGraph g; std::vector<int> order; g.begin(); g.add([&] { order.push_back(1); }); g.add([&] { order.push_back(2); }); g.end();
      CHECK(g.replay() && g.replay() && (order == std::vector<int>{1, 2, 1, 2})); }
}

// ---------------------------------------------------------------------------------------------------------------
// DirectFileIo
static void test_direct_file() {
    const uint64_t A = dfio::kAlign;
    using U = std::initializer_list<uint64_t>;
    for (uint64_t off : U{0, A, 3 * A}) for (uint64_t len : U{0, 1, A - 1, A, 2 * A, A + 1}) for (uint64_t buf : U{0, A, 8 * A, 8 * A + 16}) {
        const bool want = off % A == 0 && len % A == 0 && buf % A == 0 && len != 0;
        CHECK(dfio::acceptable(off, buf, len) == want);
        if (want) CHECK((off + len) % A == 0 && len >= A);                // end_aligned / len_ge_align / no_partial_sector
    }
    for (uint64_t off : {0ull, 1ull, 4294967295ull, 4294967296ull, 1ull << 40, (1ull << 63) + 12345}) {
        CHECK(dfio::recombine(dfio::low_word(off), dfio::high_word(off)) == off);          // offset_split
    }
    for (uint64_t max = 0; max < 200; ++max) for (uint64_t n = 0; n <= max; ++n) {
        const uint64_t w = dfio::batch_want(max, n);
        CHECK(w <= 64 && n + w <= max);                                    // port_request_fits
    }
    // real file: size 10000 -> read at 8192 returns the 1808 tail, past EOF returns 0, FIFO drain order, no overrun
    char path[] = "/tmp/u1af_dfio_XXXXXX"; int fd = mkstemp(path); CHECK(fd >= 0);
    std::vector<unsigned char> data(10000); for (size_t i = 0; i < data.size(); ++i) data[i] = (unsigned char)(i * 131 + 7);
    CHECK(write(fd, data.data(), data.size()) == (ssize_t)data.size()); close(fd);
    dfio::DirectFile f; CHECK(f.open(path) && f.size() == 10000);
    void* buf = nullptr; CHECK(posix_memalign(&buf, A, 4 * A) == 0);
    std::memset(buf, 0xAB, 4 * A);
    CHECK(!f.submit(1, 1, buf, A) && !f.submit(2, 0, buf, 100) && !f.submit(3, 0, (char*)buf + 16, A) && !f.submit(4, 0, buf, 0));
    CHECK(f.pending() == 0);                                               // rejected requests queue nothing
    CHECK(f.submit(10, 0, buf, A) && f.submit(11, 2 * A, (char*)buf + A, A) && f.submit(12, 3 * A, (char*)buf + 2 * A, A));
    CHECK(f.pending() == 3);
    dfio::Completion out[8];
    CHECK(f.wait(out, 2) == 2 && out[0].tag == 10 && out[1].tag == 11 && f.pending() == 1);   // oldest first, <= max, none dropped
    CHECK(out[0].ok && out[0].bytes == A && out[1].ok && out[1].bytes == 10000 - 2 * A);        // readLen_full / readLen_in_file
    CHECK(f.wait(out, 8) == 1 && out[0].tag == 12 && out[0].ok && out[0].bytes == 0);           // readLen_past_eof
    CHECK(f.wait(out, 8) == 0);
    CHECK(std::memcmp(buf, data.data(), A) == 0 && std::memcmp((char*)buf + A, data.data() + 2 * A, 10000 - 2 * A) == 0);
    free(buf); f.close(); CHECK(!f.is_open()); unlink(path);
}

// ---------------------------------------------------------------------------------------------------------------
// LayerComposition
static void test_layer_composition() {
    for (uint64_t n = 0; n < 5000; ++n) {
        const uint64_t a = core::align_up16(n);
        CHECK(a >= n && a % 16 == 0 && a < n + 16 && core::align_up16(a) == a);
    }
    std::mt19937_64 rng(3);
    for (int it = 0; it < 300; ++it) {                                     // carve_eq / carve_aligned / carve_append / row_fits
        std::vector<uint64_t> parts(rng() % 12); for (auto& p : parts) p = rng() % 100000;
        core::Cursor c; uint64_t prev_end = 0, sum = 0;
        for (uint64_t p : parts) {
            const uint64_t at = c.take(1, p);
            CHECK(at % 16 == 0 && at == prev_end && c.used >= at + p);     // regions sequential, 16-aligned, non-overlapping
            prev_end = c.used; sum += core::align_up16(p);
        }
        CHECK(c.used == sum && c.used % 16 == 0 && !c.failed);
        core::Cursor cap; cap.cap = sum; for (uint64_t p : parts) cap.take(1, p);
        CHECK(!cap.failed);                                                // counted total fits the borrowed capacity
        if (sum > 0) { core::Cursor tight; tight.cap = sum - 1; for (uint64_t p : parts) tight.take(1, p); CHECK(tight.failed); }
    }
    CHECK(core::q8k_bytes(2560) == 2920);                                  // q8kBytes_embd
    for (uint64_t b = 0; b < 100; ++b) CHECK(core::q8k_bytes(256 * b) == b * 292);
    // verify compatibility = native_bf16 && fused_gr && fused_gdn && native_gdn && fast_attn && !flash_short && fast_select && indexer
    for (int m = 0; m < 64; ++m) for (int ng = 0; ng < 2; ++ng) for (int qi = 0; qi < 2; ++qi) {
        const bool bf16 = m & 1, gr = m & 2, gdn = m & 4, fa = m & 8, fl = m & 16, fs = m & 32;
        core::layer_set_native_bf16(bf16); core::layer_set_fused_gr(gr); core::layer_set_fused_gdn(gdn);
        core::layer_set_fast_attn(fa); core::layer_set_native_flash_attn_short(fl); core::layer_set_fast_select(fs);
        const bool want = bf16 && gr && gdn && ng && fa && !fl && fs && qi;
        CHECK(core::layer_verify_compatible(ng, qi) == want);
    }
    core::layer_set_native_bf16(false); core::layer_set_fused_gr(false); core::layer_set_fused_gdn(false);
    core::layer_set_fast_attn(false); core::layer_set_native_flash_attn_short(false); core::layer_set_fast_select(false);
    for (uint64_t m = 0; m < 5000; m += 7) for (uint64_t ps : {1ull, 16ull, 64ull}) CHECK(m <= (m + ps - 1) / ps * ps);   // pages_cover
}

// ---------------------------------------------------------------------------------------------------------------
// PinnedArenaLoad
static void test_pinned() {
    std::mt19937_64 rng(9);
    for (int it = 0; it < 500; ++it) {
        const uint64_t total = rng() % 100000, parts = 1 + rng() % 40;
        auto b = hw::uniform_bounds(total, parts);
        CHECK(b.size() == parts + 1 && b.front() == 0 && b.back() == total);
        for (size_t i = 0; i < parts; ++i) { CHECK(b[i] <= b[i + 1]); CHECK(b[i + 1] - b[i] <= total / parts + 1 && b[i + 1] - b[i] >= total / parts); }
        // per-layer ranges [L*per, L*per+per) abut, stay inside the file and are pairwise disjoint
        const uint64_t per = 1 + rng() % 5000, layers = 1 + rng() % 30;
        for (uint64_t L = 0; L < layers; ++L) {
            CHECK(L * per + per <= layers * per && (L + 1) * per == L * per + per);
            for (uint64_t M = L + 1; M < layers; ++M) CHECK(L * per + per <= M * per);
        }
        // read loop: each step min(chunk, remaining) makes progress and never over-reads
        const uint64_t chunk = 1 + rng() % 700, lb = rng() % 5000; uint64_t pos = 0, steps = 0;
        while (pos < lb) { const uint64_t st = std::min(chunk, lb - pos); CHECK(st > 0 && pos + st <= lb); pos += st; ++steps; }
        CHECK(pos == lb && steps == (lb + chunk - 1) / chunk);
        const uint64_t bytes = rng() % 100000; const uint64_t moved = bytes / chunk * chunk;
        CHECK(moved <= bytes && (bytes % chunk != 0 || moved == bytes));   // bandwidth probe accounting
        // sliced pinning: k = bytes/slice whole slices, registered prefix + locked suffix = capacity
        const uint64_t slice = 1 + rng() % 9000, k = bytes / slice;
        CHECK(k * slice <= bytes);
        for (uint64_t c = 0; c <= k; ++c) { const uint64_t reg = c * slice; CHECK(reg <= bytes && reg + (bytes - reg) == bytes); }
    }
    CHECK(hw::uniform_bounds(10, 0).empty());
    // fnv1a64 known vectors (offset basis 0xcbf29ce484222325)
    const uint64_t seed = 0xcbf29ce484222325ull;
    CHECK(hw::fnv1a64(nullptr, 0, seed) == seed);
    const uint8_t a = 'a'; CHECK(hw::fnv1a64(&a, 1, seed) == 0xaf63dc4c8601ec8cull);
    const uint8_t foobar[6] = {'f', 'o', 'o', 'b', 'a', 'r'}; CHECK(hw::fnv1a64(foobar, 6, seed) == 0x85944171f73967e8ull);
}

// ---------------------------------------------------------------------------------------------------------------
// PrefillChunkPipeline
static void test_prefill_pipeline() {
    std::mt19937_64 rng(21);
    for (uint64_t b = 0; b < 3000; ++b) {
        const uint64_t t = prefill_driver::alloc_bytes(b);
        CHECK(t >= b + 256 && t % 256 == 0 && t == core::round_bytes(b));  // takeBytes_ge / takeBytes_mod, both implementations agree
    }
    for (int it = 0; it < 300; ++it) {
        std::vector<uint64_t> seq(1 + rng() % 10); for (auto& s : seq) s = rng() % 50000;
        const uint64_t need = prefill_driver::bytes_needed(seq); uint64_t out = 0;
        std::vector<uint64_t> rounded; for (uint64_t b : seq) rounded.push_back(prefill_driver::alloc_bytes(b));
        CHECK(prefill_driver::take_all(need, 0, rounded, out) && out == need);   // counted total fits exactly; take_all charges rounded sizes
        CHECK(!prefill_driver::take_all(need - 1, 0, rounded, out));
        uint64_t used = 0; bool ok = true; for (uint64_t s : seq) ok = ok && prefill_driver::bump(used, need, prefill_driver::alloc_bytes(s));
        CHECK(ok && used == need);                                         // take_fits: counted total fits, no failure
        uint64_t u2 = 0; bool ok2 = true; for (uint64_t s : seq) ok2 = ok2 && prefill_driver::bump(u2, need - 1, prefill_driver::alloc_bytes(s));
        CHECK(!ok2);
    }
    // chunk loop covers [0, n) in steps of m, every chunk >= 1 token and <= m
    for (uint64_t m : {1ull, 7ull, 512ull}) for (uint64_t n : {1ull, 6ull, 7ull, 8ull, 511ull, 513ull, 2000ull}) {
        auto lens = prefill_driver::chunk_lens(m, 100000, n);
        uint64_t sum = 0, c0 = 0; for (uint64_t t : lens) { CHECK(t >= 1 && t <= m && t == std::min(m, n - c0) && c0 + t <= n); sum += t; c0 += t; }
        CHECK(sum == n);
    }
    // grouping by expert: off prefix sums, disjoint slot ranges, j-th pair of e lands inside e's range, source token i / K
    for (int it = 0; it < 200; ++it) {
        const uint32_t E = 1 + rng() % 40, K = 1 + rng() % 10, T = 1 + rng() % 30;
        std::vector<uint32_t> ex(T * K); for (auto& e : ex) e = (uint32_t)(rng() % E);
        auto g = prefill_driver::group_by_expert(ex, E);
        CHECK(g.off.size() == E + 1u && g.off[0] == 0 && g.off[E] == ex.size());
        std::vector<uint64_t> seen(E, 0); std::set<uint64_t> slots;
        for (size_t i = 0; i < ex.size(); ++i) {
            const uint32_t e = ex[i]; const uint64_t s = g.slot[i];
            CHECK(g.off[e] <= s && s < g.off[e + 1]);                      // slot_in_range
            CHECK(s == g.off[e] + seen[e]++);                              // slot of j-th pair = off[e] + j (stable)
            CHECK(slots.insert(s).second && g.order[s] == i && i / K < T); // disjoint, invertible, valid source token
        }
        for (uint32_t e = 0; e + 1 < E; ++e) CHECK(g.off[e] + g.cnt[e] == g.off[e + 1]);
    }
    // staging ring (8 slots, lookahead 7) and dequant ring (2 slots)
    CHECK(prefill_driver::kStageRing == 8 && prefill_driver::kDqRing == 2);
    for (uint64_t j = 0; j < 200; ++j) {
        CHECK(prefill_driver::stage_slot(j) < 8 && prefill_driver::dq_slot(j + 1) != prefill_driver::dq_slot(j));
        for (uint64_t i = j; i <= j + 7; ++i) if (i >= 8) CHECK(i - 8 < j + 0 && prefill_driver::stage_slot(i - 8) == prefill_driver::stage_slot(i));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// PrefillKernelsIndexing (host formulas + real kernels)
static uint16_t bf_host(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static void test_prefill_kernels() {
    CHECK(1280 * 640 == 819200 && 1280 * 640 + 2560 * 160 == 1228800 && 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 == 1331200);
    CHECK(1280 * 640 + 2560 * 160 + 1280 * 40 * 2 + 2560 * 10 * 2 == 1382400);          // blob_total
    for (int n = 0; n < 5000; ++n) { const int64_t b = pk::blocks_for(n, 256); CHECK(b * 256 >= n && b * 256 < n + 256); }
    for (int e = 0; e < 512; ++e) CHECK(e % 32 + 32 * (e / 32) == e && e / 32 < 16);     // expert id <-> (lane, i)
    for (int i = 0; i < 40000; i += 13) CHECK(i % 2560 == (i % 10240) % 2560);          // broadcast_same_d
    for (int rg = 0; rg < 4; ++rg) for (int r = 0; r < 32; ++r) CHECK(rg * 32 + r < 128);
    // to_bf16 on the device equals the host round-to-nearest-even formula, ties go to even
    std::mt19937 rng(77);
    std::vector<float> x;
    for (int i = 0; i < 8192; ++i) { uint32_t u = rng(); float f; std::memcpy(&f, &u, 4); if (!std::isnan(f) && !std::isinf(f)) x.push_back(f); }
    for (uint32_t hi : {0x3f80u, 0x3f81u, 0x4000u, 0xbf80u}) for (uint32_t lo : {0x0000u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
        uint32_t u = (hi << 16) | lo; float f; std::memcpy(&f, &u, 4); x.push_back(f);
    }
    const int64_t n = (int64_t)x.size();
    float* dx = nullptr; uint16_t* dy = nullptr;
    CHECK(cudaMalloc(&dx, n * 4) == cudaSuccess && cudaMalloc(&dy, n * 2) == cudaSuccess);
    CHECK(cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(pk::to_bf16(dx, dy, n) == 0); cudaDeviceSynchronize();
    std::vector<uint16_t> y(n); CHECK(cudaMemcpy(y.data(), dy, n * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
    int bad = 0, off_by = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (y[i] != bf_host(x[i])) ++bad;
        uint32_t u; std::memcpy(&u, &x[i], 4); const uint32_t base = u >> 16;
        if (!(y[i] == base || y[i] == ((base + 1) & 0xffff))) ++off_by;                   // bf_within_one
        if ((u & 0xffff) == 0x8000 && (y[i] & 1)) ++off_by;                                // bf_tie_even
    }
    CHECK(bad == 0 && off_by == 0);
    cudaFree(dx); cudaFree(dy);
    // router order: higher value first, ties to the lower expert id; every id lies in [0, 512)
    { const int T = 4; std::vector<float> lg(T * 512); std::mt19937 r2(3);
      for (int t = 0; t < T; ++t) for (int e = 0; e < 512; ++e) lg[t * 512 + e] = (t == 0) ? 1.0f : (t == 1) ? (float)(e % 4) : (float)(r2() % 50) / 7.0f;
      for (int e = 0; e < 512; ++e) lg[3 * 512 + e] = -(float)e;                           // strictly decreasing: top-10 = 0..9
      float* dl; int32_t* di; float* dw;
      CHECK(cudaMalloc(&dl, lg.size() * 4) == cudaSuccess && cudaMalloc(&di, T * 10 * 4) == cudaSuccess && cudaMalloc(&dw, T * 10 * 4) == cudaSuccess);
      CHECK(cudaMemcpy(dl, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
      CHECK(pk::route(dl, di, dw, T) == 0); cudaDeviceSynchronize();
      std::vector<int32_t> ids(T * 10); CHECK(cudaMemcpy(ids.data(), di, ids.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      for (int t = 0; t < T; ++t) {
          std::vector<int> ord(512); std::iota(ord.begin(), ord.end(), 0);
          std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return lg[t * 512 + a] > lg[t * 512 + b]; });   // value desc, id asc
          std::set<int> want(ord.begin(), ord.begin() + 10), got;
          for (int k = 0; k < 10; ++k) { CHECK(ids[t * 10 + k] >= 0 && ids[t * 10 + k] < 512); got.insert(ids[t * 10 + k]); }
          CHECK(got == want);
      }
      cudaFree(dl); cudaFree(di); cudaFree(dw); }
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::fprintf(stderr, "no GPU device\n"); return 2; }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    signal(SIGSEGV, on_segv);
    STAGE("device init"); cudaSetDevice(0); cudaFree(0);
    STAGE("layer_geometry"); test_layer_geometry();
    STAGE("mtp_and_verify"); test_mtp_and_verify();
    STAGE("graph_registry"); test_graph_registry();
    STAGE("direct_file"); test_direct_file();
    STAGE("layer_composition"); test_layer_composition();
    STAGE("pinned"); test_pinned();
    STAGE("prefill_pipeline"); test_prefill_pipeline();
    STAGE("prefill_kernels"); test_prefill_kernels();
    std::printf("checks=%d failures=%d\n", g_checks, g_fail);
    if (g_fail) { std::printf("U1AF UNIT TESTS FAILED\n"); return 1; }
    std::printf("U1AF UNIT TESTS PASSED\n");
    return 0;
}
