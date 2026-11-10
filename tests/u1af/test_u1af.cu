// test_u1af.cu - invariant tests for unit u1af344a465 (layer geometry, graph registry, MTP chain, session token loop,
// verify window, direct file IO, layer composition, pinned arena, prefill pipeline, prefill kernel indexing).
// Every CHECK states a LAYER 1 theorem of the matching lean/u1af module; any failure makes the binary exit non-zero.
#include <cuda_runtime.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
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
#include <thread>
#include <vector>
#include <chrono>

#include "baloon/core_graph.hpp"
#include "baloon/core_graph_dev.hpp"
#include "baloon/core_layer.hpp"
#include "baloon/core_pinned_hw.hpp"
#include "baloon/core_verify.hpp"
#include "baloon/platform_io.hpp"
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
    CHECK(pk::HK == 16 && pk::HV == 48 && pk::S == 128 && pk::C == pk::HK * pk::S + pk::HK * pk::S + pk::HV * pk::S);   // conv_channels (q|k|v widths)
    std::set<int> gs, qs;
    for (int l = 0; l < 48; ++l) {
        const int g = core::gdn_point_at(l), q = core::qsa_point_at(l);
        if (core::is_qsa_layer(l)) { CHECK(g == -1 && q >= 0 && q < 12 && qs.insert(q).second); CHECK(q == l / 4); }
        else { CHECK(q == -1 && g >= 0 && g < 36 && gs.insert(g).second); CHECK(g == l - l / 4); }  // injective, in range
        CHECK(session_schedule::is_qsa((uint64_t)l) == core::is_qsa_layer(l));
        if (!core::is_qsa_layer(l)) CHECK((int)session_schedule::gdn_index((uint64_t)l) == g);
    }
    CHECK(gs.size() == 36 && qs.size() == 12);
    // q|gate halves of q_full: the real split_q kernel puts head h's first 256 lanes into q and the next 256 into gate
    { const int T = 3, HQ = 24, HD = 256; const int64_t nin = (int64_t)T * HQ * 2 * HD, nout = (int64_t)T * HQ * HD;
      std::vector<float> in(nin), q(nout, -1.f), g(nout, -1.f); for (int64_t k = 0; k < nin; ++k) in[k] = (float)k;
      float *di, *dq, *dg; CHECK(cudaMalloc(&di, nin * 4) == cudaSuccess && cudaMalloc(&dq, nout * 4) == cudaSuccess && cudaMalloc(&dg, nout * 4) == cudaSuccess);
      CHECK(cudaMemcpy(di, in.data(), nin * 4, cudaMemcpyHostToDevice) == cudaSuccess);
      CHECK(pk::split_q(di, dq, dg, T) == 0); cudaDeviceSynchronize();
      CHECK(cudaMemcpy(q.data(), dq, nout * 4, cudaMemcpyDeviceToHost) == cudaSuccess && cudaMemcpy(g.data(), dg, nout * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      int bad = 0;
      for (int t = 0; t < T; ++t) for (int h = 0; h < HQ; ++h) for (int d = 0; d < HD; ++d) {
          const int64_t o = ((int64_t)t * HQ + h) * HD + d, base = ((int64_t)t * HQ + h) * 2 * HD;
          if (q[o] != in[base + d] || g[o] != in[base + HD + d]) ++bad;
      }
      CHECK(bad == 0 && pk::split_q(nullptr, dq, dg, T) == -1 && pk::split_q(di, dq, dg, 0) == -1);
      cudaFree(di); cudaFree(dq); cudaFree(dg); }
    // prefix timings: differencing isolates one stage
    { std::vector<int64_t> P = {0, 5, 9, 14, 20, 27};   // prefix timings P[0..5], P[0] = 0
      auto st = session_loop::stage_times(P);                              // differencing isolates one stage
      CHECK(st.size() == 5 && st[0] == 5 && st[1] == 4 && st[2] == 5 && st[3] == 6 && st[4] == 7);
      auto st2 = session_schedule::stage_ms(5, 9, 14, 20, 27); CHECK(st2 == st); }

    // SessionTokenLoop: launches, dump layout, exec index bounds
    for (uint64_t stride : {1ull, 7ull, 2560ull, 10240ull}) for (uint64_t n : {1ull, 48ull}) for (uint64_t l = 0; l < n; ++l) {
        CHECK(session_loop::slot_base(stride, l) + stride <= session_loop::dump_total(stride, n));    // dump_slot_in_bounds
        CHECK(session_loop::slot_base(stride, l) + stride == session_loop::slot_base(stride, l + 1));  // slots abut
        CHECK(session_schedule::dump_floats(n, stride) == (n + 1) * stride);
    }
    for (uint64_t T = 1; T <= 8; ++T) CHECK(session_loop::exec_index_ok(8, T));
    CHECK(!session_loop::exec_index_ok(9, 1) && !session_loop::exec_index_ok(8, 9));
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
    CHECK((uint64_t)pk::kExperts * (uint64_t)pk::kBlobBytes == 707788800ull);          // expert file = 512 blobs
    for (uint64_t mt = 3; mt <= 8; ++mt) for (uint64_t j = 1; j + 2 <= mt; ++j) CHECK(mtp::rows_disjoint(mt, j));

    // acceptance through the REAL core::Verifier: accept_len = longest prefix with window[a+1] == greedy[a]
    core::Verifier::Config cfg; cfg.max_t = 8; cfg.n_layers = 4; cfg.groups = 1; cfg.row_floats = 4;
    core::Verifier ver; CHECK(ver.init(cfg) && ver.capture() && ver.capture_commit());
    { std::vector<int32_t> w(9, 0); CHECK(!ver.record_window(w.data(), 0) && !ver.record_window(w.data(), 9) && !ver.record_window(nullptr, 2)); }
    std::mt19937 rng(5);
    int hist[9] = {0};
    for (int it = 0; it < 3000; ++it) {
        const uint64_t T = 1 + rng() % 8, nd = T - 1;
        std::vector<int32_t> w(T), greedy(T);
        for (auto& x : w) x = (int32_t)(rng() % 3); for (auto& x : greedy) x = (int32_t)(rng() % 3);
        if (it % 5 == 0) for (uint64_t k = 0; k < nd; ++k) greedy[k] = w[k + 1];   // force fully-agreeing windows
        CHECK(ver.record_window(w.data(), T) && ver.T() == T);
        const uint64_t a = ver.accept_len(greedy);
        uint64_t want = 0; while (want < nd && w[want + 1] == greedy[want]) ++want;   // independent oracle
        CHECK(a == want && verify::accepted_ok(T, a));
        if (a < nd) CHECK(w[a + 1] != greedy[a]);                                    // stops exactly at the first mismatch
        if (it % 5 == 0) CHECK(a == nd);                                             // all agree -> all drafts accepted
        ++hist[a];
        std::vector<int32_t> ent; CHECK(ver.commit(a, 1000, ent));
        CHECK(ent == verify::commit_vector(1000, (int64_t)a + 1, 8) && ent[0] == (int32_t)a + 1 && ent[1] == (int32_t)a);
        for (int t = 0; t < 8; ++t) CHECK(ent[2 + t] == (t <= (int)a ? 1000 + t : -1));
        CHECK(!ver.commit(T, 1000, ent) && ent.empty());                             // a must be <= T-1
        CHECK(ver.final_R(a) != nullptr && ver.final_R(T) == nullptr);
        std::vector<int32_t> shortg(greedy.begin(), greedy.begin() + (greedy.size() > 1 ? 1 : 0));   // short greedy list never over-accepts
        CHECK(ver.accept_len(shortg) <= shortg.size());
    }
    for (int a = 0; a < 8; ++a) CHECK(hist[a] > 0);                                  // every acceptance length actually exercised
    { core::Verifier un; std::vector<int32_t> w(2, 1), e; CHECK(!un.record_window(w.data(), 2) && !un.commit(0, 0, e) && un.accept_len(w) == 0); }
    // schedule + flag: one step per (layer, group), flag rises monotonically to the last ring value, never lowers
    for (uint64_t G : {1ull, 2ull}) {
        core::Verifier v2; core::Verifier::Config c2; c2.max_t = 8; c2.n_layers = 6; c2.groups = G; CHECK(v2.init(c2) && v2.capture());
        std::vector<int32_t> w(5, 1); CHECK(v2.record_window(w.data(), 5));
        uint64_t last = 0, steps = 0; bool mono = true, cover = true; std::vector<uint64_t> seen(5, 0);
        const uint64_t n = v2.run([&](const core::Verifier::Step& st) {
            mono = mono && st.ring == last + 1 && v2.flag() == last; last = st.ring; ++steps;
            for (uint64_t r = st.row_begin; r < st.row_end; ++r) seen[r]++;
        });
        CHECK(n == 6 * G && steps == n && mono && v2.flag() == 6 * G);
        for (uint64_t r = 0; r < 5; ++r) cover = cover && seen[r] == 6; CHECK(cover);   // every row visited once per layer
        v2.raise_flag(3); CHECK(v2.flag() == 6 * G);
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

// platform::DirectFile (src/platform/direct_file.cpp): worker-thread implementation
static void test_platform_direct_file() {
    using platform::DirectFile; const uint64_t A = dfio::kAlign;
    char path[] = "/tmp/u1af_pdf_XXXXXX"; int fd = mkstemp(path); CHECK(fd >= 0);
    std::vector<unsigned char> data(10000); for (size_t i = 0; i < data.size(); ++i) data[i] = (unsigned char)(i * 131 + 7);
    CHECK(write(fd, data.data(), data.size()) == (ssize_t)data.size()); close(fd);
    // alloc_aligned: kAlign-aligned, zero length -> nullptr
    { void* p = DirectFile::alloc_aligned(1); CHECK(p && (uintptr_t)p % A == 0); std::memset(p, 1, A); DirectFile::free_aligned(p);
      CHECK(DirectFile::alloc_aligned(0) == nullptr); }
    int64_t t0 = platform::now_us(), t1 = platform::now_us(); CHECK(t1 >= t0);
    DirectFile f; CHECK(!f.is_open() && !f.open(nullptr) && !f.open("/nonexistent/u1af_missing"));
    uint8_t* buf = (uint8_t*)DirectFile::alloc_aligned(4 * A); CHECK(buf != nullptr);
    CHECK(!f.submit(1, 0, buf, A));                                          // closed file rejects
    CHECK(f.open(path) && f.is_open() && f.size() == 10000);
    std::memset(buf, 0xAB, 4 * A);
    CHECK(!f.submit(1, 1, buf, A) && !f.submit(2, 0, buf, 100) && !f.submit(3, 0, buf + 16, A) && !f.submit(4, 0, buf, 0));
    CHECK(f.in_flight() == 0);                                               // rejected requests are never queued
    platform::Completion out[8];
    CHECK(f.wait(out, 8, 20) == 0);                                          // nothing pending: bounded wait returns 0
    CHECK(f.submit(10, 0, buf, A) && f.submit(11, 2 * A, buf + A, A) && f.submit(12, 3 * A, buf + 2 * A, A));
    CHECK(f.in_flight() <= 3 && f.in_flight() >= 1);
    // drain: completions arrive oldest-first, each wait <= max, none dropped or duplicated
    std::vector<platform::Completion> got; int rounds = 0;
    while (got.size() < 3 && rounds++ < 200) { const uint64_t n = f.wait(out, 2, 1000); CHECK(n <= 2); for (uint64_t i = 0; i < n; ++i) got.push_back(out[i]); }
    CHECK(got.size() == 3 && got[0].tag == 10 && got[1].tag == 11 && got[2].tag == 12 && f.in_flight() == 0);
    CHECK(got[0].ok && got[0].bytes == A && got[1].ok && got[1].bytes == 10000 - 2 * A && got[2].ok && got[2].bytes == 0);   // full / tail / past EOF
    CHECK(std::memcmp(buf, data.data(), A) == 0 && std::memcmp(buf + A, data.data() + 2 * A, 10000 - 2 * A) == 0);
    CHECK(f.wait(out, 8, 10) == 0);
    // wake(): a blocked wait returns promptly with 0 completions; sticky when issued before the wait
    { const int64_t a0 = platform::now_us(); std::thread th([&] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); f.wake(); });
      const uint64_t n = f.wait(out, 8, -1); const int64_t dt = platform::now_us() - a0; th.join();
      CHECK(n == 0 && dt < 5000000);
      f.wake(); CHECK(f.wait(out, 8, -1) == 0);
      CHECK(f.wait(out, 8, 10) == 0); }                                      // wake was consumed
    // many in flight: a wait never exceeds max and every tag is delivered exactly once
    { const int N = 16; std::vector<uint8_t*> bs; for (int i = 0; i < N; ++i) { bs.push_back((uint8_t*)DirectFile::alloc_aligned(A)); CHECK(f.submit(100 + i, (uint64_t)(i % 2) * A, bs.back(), A)); }
      std::set<uint64_t> tags; uint64_t prev = 99; bool fifo = true; int r = 0;
      while (tags.size() < (size_t)N && r++ < 500) { const uint64_t n = f.wait(out, 3, 1000); CHECK(n <= 3); for (uint64_t i = 0; i < n; ++i) { tags.insert(out[i].tag); fifo = fifo && out[i].tag == prev + 1; prev = out[i].tag; CHECK(out[i].ok && out[i].bytes == A); } }
      CHECK(tags.size() == (size_t)N && fifo && f.in_flight() == 0);
      CHECK(std::memcmp(bs[1], data.data() + A, A) == 0 && std::memcmp(bs[2], data.data(), A) == 0);
      for (auto b : bs) DirectFile::free_aligned(b); }
    f.close(); CHECK(!f.is_open() && !f.submit(1, 0, buf, A) && f.size() == 0);
    CHECK(f.open(path) && f.size() == 10000); f.close();                     // re-open after close
    DirectFile::free_aligned(buf); unlink(path);
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
    }
    CHECK(hw::uniform_bounds(10, 0).empty());
    // fnv1a64 known vectors (offset basis 0xcbf29ce484222325)
    const uint64_t seed = 0xcbf29ce484222325ull;
    CHECK(hw::fnv1a64(nullptr, 0, seed) == seed);
    const uint8_t a = 'a'; CHECK(hw::fnv1a64(&a, 1, seed) == 0xaf63dc4c8601ec8cull);
    const uint8_t foobar[6] = {'f', 'o', 'o', 'b', 'a', 'r'}; CHECK(hw::fnv1a64(foobar, 6, seed) == 0x85944171f73967e8ull);
    // chunked hashing: the running value as next seed equals the hash of the concatenation
    { std::vector<uint8_t> z(1000); for (size_t i = 0; i < z.size(); ++i) z[i] = (uint8_t)(i * 7 + 1);
      uint64_t h = hw::kRefFnvSeed; for (size_t o = 0; o < z.size(); o += 300) h = hw::fnv1a64(z.data() + o, std::min<size_t>(300, z.size() - o), h);
      CHECK(h == hw::fnv1a64(z.data(), z.size(), hw::kRefFnvSeed)); }

    // real arena + parallel load: layers abut, each is loaded by exactly one worker, per-layer hash == hash of that layer's bytes
    const uint64_t layers = 6, bpl = 3, blob = 4096 + 8, chunk = 1000, total = layers * bpl * blob;
    auto src = [](uint64_t off) { return (uint8_t)((off * 2654435761ull >> 7) ^ off); };
    hw::Reader rd = [&](uint64_t off, uint64_t n, uint8_t* dst) { for (uint64_t i = 0; i < n; ++i) dst[i] = src(off + i); return true; };
    for (unsigned workers : {1u, 3u, 8u}) {
        hw::PinnedArena arena; CHECK(arena.reserve(total, 8192) && arena.data() && arena.bytes() == total && arena.backing() != hw::Backing::None);
        CHECK(arena.registered_bytes() + arena.locked_bytes() == total && arena.registered_bytes() % arena.slice_bytes() == 0);
        auto r = hw::load_experts(arena, rd, layers, bpl, blob, chunk, workers);
        CHECK(r.ok && r.bytes == total && r.hashes.size() == layers && r.claimed_by.size() == layers);
        for (uint64_t L = 0; L < layers; ++L) {
            const uint64_t b0 = L * bpl * blob, n = bpl * blob; bool same = true; std::vector<uint8_t> ref(n);
            for (uint64_t i = 0; i < n; ++i) { ref[i] = src(b0 + i); same = same && arena.data()[b0 + i] == ref[i]; }
            CHECK(same && r.hashes[L] == hw::fnv1a64(ref.data(), n, hw::kRefFnvSeed) && r.claimed_by[L] < workers);
        }
        // a failing read latches and the load reports failure
        std::atomic<int> calls{0}; hw::Reader bad = [&](uint64_t, uint64_t, uint8_t*) { return ++calls < 4; };
        hw::PinnedArena a2; CHECK(a2.reserve(total) && !hw::load_experts(a2, bad, layers, bpl, blob, chunk, workers).ok);
        // ranges packed back to back from arena offset 0
        std::vector<std::pair<uint64_t, uint64_t>> rg = {{100, 500}, {5000, 1}, {7, 2048}};
        auto rr = hw::load_experts_ranges(arena, rd, rg, 300); uint64_t at = 0; bool ok = rr.ok && rr.hashes.size() == rg.size();
        for (size_t k = 0; k < rg.size() && ok; ++k) { std::vector<uint8_t> ref(rg[k].second); for (uint64_t i = 0; i < ref.size(); ++i) ref[i] = src(rg[k].first + i);
            ok = ok && std::memcmp(arena.data() + at, ref.data(), ref.size()) == 0 && rr.hashes[k] == hw::fnv1a64(ref.data(), ref.size(), hw::kRefFnvSeed); at += ref.size(); }
        CHECK(ok);
        if (workers == 8) {   // stream to the device, then verify on the GPU against host chunk hashes
            hw::load_experts(arena, rd, layers, bpl, blob, chunk, 2);
            void* dev = nullptr; CHECK(cudaMalloc(&dev, total) == cudaSuccess);
            auto bw = hw::stream_bandwidth(arena, dev, 4096, 2); CHECK(bw.ok && bw.passes == 2 && bw.bytes == 2 * (total / 4096 * 4096) && bw.gbps > 0);
            std::vector<uint64_t> dh; CHECK(hw::device_chunk_hashes(dev, total, 4096, dh) && dh.size() == total / 4096);
            bool eq = true; for (size_t c = 0; c < dh.size(); ++c) eq = eq && dh[c] == hw::fnv1a64(arena.data() + c * 4096, 4096, hw::kRefFnvSeed);
            CHECK(eq); cudaFree(dev);
        }
        arena.release(); CHECK(arena.data() == nullptr && arena.bytes() == 0);
    }
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
    CHECK(pk::kDownOff == 819200 && pk::kGuScaleOff == 1228800 && pk::kDownScaleOff == 1331200 && pk::kDownScaleOff + 2560 * 10 * 2 == pk::kBlobBytes);   // blob planes tile the blob
    for (int n = 0; n < 5000; ++n) { const int64_t b = pk::blocks_for(n, 256); CHECK(b * 256 >= n && b * 256 < n + 256); }
    // gr_broadcast on the device: out[t][c][d] = e[t][d] (same d for every stream c)
    { const int T = 3; std::vector<float> e(T * pk::N), o((size_t)T * pk::D, -1.f); for (size_t k = 0; k < e.size(); ++k) e[k] = (float)k * 0.5f;
      float *de, *dout; CHECK(cudaMalloc(&de, e.size() * 4) == cudaSuccess && cudaMalloc(&dout, o.size() * 4) == cudaSuccess);
      CHECK(cudaMemcpy(de, e.data(), e.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess && pk::gr_broadcast(de, dout, T) == 0); cudaDeviceSynchronize();
      CHECK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      int bad = 0; for (int t = 0; t < T; ++t) for (int c = 0; c < pk::HC; ++c) for (int d = 0; d < pk::N; ++d) if (o[((size_t)t * pk::HC + c) * pk::N + d] != e[(size_t)t * pk::N + d]) ++bad;
      CHECK(bad == 0 && pk::gr_broadcast(nullptr, dout, T) == -1); cudaFree(de); cudaFree(dout); }
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
    STAGE("platform_direct_file"); test_platform_direct_file();
    STAGE("layer_composition"); test_layer_composition();
    STAGE("pinned"); test_pinned();
    STAGE("prefill_pipeline"); test_prefill_pipeline();
    STAGE("prefill_kernels"); test_prefill_kernels();
    std::printf("checks=%d failures=%d\n", g_checks, g_fail);
    if (g_fail) { std::printf("U1AF UNIT TESTS FAILED\n"); return 1; }
    std::printf("U1AF UNIT TESTS PASSED\n");
    return 0;
}
