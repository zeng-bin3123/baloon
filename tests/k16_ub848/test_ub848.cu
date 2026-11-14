// test_ub848.cu - unit ub848524604 smoke test.
// Tests: session constants (constexpr, header-only); spec controller choose/observe (inline);
//        suffix drafter append/propose (inline); verifier accept_len (needs verify.o);
//        GPU verify-window kernels (window_ids, dense_steps, gdn_conv_commit).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "baloon/core_session.hpp"
#include "baloon/spec_controller.hpp"
#include "baloon/suffix_drafter.hpp"
#include "baloon/core_verify.hpp"
#include "baloon/verify_kernels.hpp"

static long g_ok = 0, g_fail = 0;
#define CHECK(c) do { if (c) { ++g_ok; } else { ++g_fail; \
    std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA %d (%s) at %s:%d\n", (int)e_, cudaGetErrorString(e_), \
                __FILE__, __LINE__); std::exit(2); } } while (0)

// ---- CPU reference: session constants (constexpr/inline, no .cpp linkage needed) -----------

static void test_session_constants() {
    using namespace baloon::core;
    // SESSION_STATE_ALIGN must be 256 (from the theory: align_up(n, 256))
    CHECK(SESSION_STATE_ALIGN == 256);
    // align_up(n, 256): rounds up to next multiple of 256
    CHECK(align_up(0) == 0);
    CHECK(align_up(1) == 256);
    CHECK(align_up(255) == 256);
    CHECK(align_up(256) == 256);
    CHECK(align_up(257) == 512);
    CHECK(align_up(512) == 512);
    // gdn_state_floats is a constexpr constant from the header
    CHECK(gdn_state_floats() > 0);
    // Session init: session_bytes = align256(sum) + 4096
    std::vector<uint64_t> parts = {100, 200, 300};
    uint64_t sb = session_bytes(parts);
    CHECK(sb >= 100 + 200 + 300 + 4096);
    CHECK(sb % 256 == 0);
    // session_init carves regions correctly
    Session s;
    CHECK(session_init(s, parts, 48));
    CHECK(s.bytes == sb);
    CHECK(s.n_layers == 48);
    for (int i = 0; i < 3; ++i) CHECK(s.off[i] % 256 == 0);
}

// ---- CPU reference: spec controller ---------------------------------------------------

static void test_spec_controller() {
    using namespace baloon::spec;
    Controller ctrl;
    // K_MAX must be 8
    CHECK(K_MAX == 8);
    // Default MTP priors are 0.86
    for (uint64_t i = 0; i < K_MAX; ++i) CHECK(std::abs(ctrl.mtp_p[i] - 0.86) < 1e-9);
    // choose with no drafters returns no-draft
    Choice c0 = ctrl.choose(0, 0, 0);
    CHECK(c0.drafter == Drafter::None);
    CHECK(c0.rate > 0.0);
    // choose with MTP drafts returns positive rate
    Choice c1 = ctrl.choose(4, 0, 0);
    CHECK(c1.rate > 0.0);
    // expected(None, k) = 1.0 (no speculation, always one token)
    CHECK(std::abs(ctrl.expected(Source::None, 3) - 1.0) < 1e-9);
    // expected(Mtp, 1) > 1.0 (speculation adds expected drafts)
    CHECK(ctrl.expected(Source::Mtp, 1) > 1.0);
    // observe with full accept: all hits, prior moves up
    ctrl.observe(Drafter::Mtp, 4, 4);
    CHECK(ctrl.mtp_p[0] > 0.86);
    // observe with zero accept: miss at position 0, prior moves down
    Controller ctrl2;
    ctrl2.observe(Drafter::Mtp, 4, 0);
    CHECK(ctrl2.mtp_p[0] < 0.86);
    // CostModel: default table is all 1.0
    CostModel cm;
    CHECK(cm.step_ms(0) == 1.0);
    CHECK(cm.step_ms(8) == 1.0);
}

// ---- CPU reference: suffix drafter ----------------------------------------------------

static void test_suffix_drafter() {
    using namespace baloon::suffix;
    SuffixDrafter sd(64, 3, 16);
    CHECK(sd.min_match() == 3);
    CHECK(sd.max_match() == 16);
    // After fewer than 3 tokens: no proposal
    sd.append(10); sd.append(20);
    CHECK(sd.propose(8).empty());
    // Build a repeated pattern: 10 20 30 40 50, then repeat 10 20 30
    sd.append(30); sd.append(40); sd.append(50);
    sd.append(10); sd.append(20); sd.append(30);
    // "10 20 30" matches at positions [0,1,2] with continuation [40,50]
    auto p = sd.propose(8);
    CHECK(!p.empty());
    CHECK(p[0] == 40);
    // reset clears history
    sd.reset();
    CHECK(sd.size() == 0);
    CHECK(sd.propose(8).empty());
    // key_of forces low bit on
    CHECK((key_of(0) & 1) == 1);
    CHECK((key_of(0xFFFE) & 1) == 1);
    // mix produces different values for different tokens
    CHECK(mix(0xcbf29ce484222325ull, 1) != mix(0xcbf29ce484222325ull, 2));
}

// ---- CPU reference: verifier accept_len and commit ------------------------------------

static void test_verifier_cpu() {
    using namespace baloon::core;
    Verifier v;
    Verifier::Config cfg;
    cfg.max_t = 4; cfg.n_layers = 2; cfg.groups = 1;
    cfg.plan_cap = 8; cfg.staging_cap = 16; cfg.row_floats = 64;
    CHECK(v.init(cfg));
    // Record window: 1 anchor (token 5) + 2 drafts (10, 20)
    int32_t tokens[3] = {5, 10, 20};
    CHECK(v.record_window(tokens, 3));
    // accept_len: drafts match greedy [10, 20, x] → a = 2
    std::vector<int32_t> greedy = {10, 20, 99};
    uint64_t a = v.accept_len(greedy);
    CHECK(a == 2);
    // accept_len: first draft mismatches → a = 0
    std::vector<int32_t> greedy2 = {99, 20, 99};
    CHECK(v.accept_len(greedy2) == 0);
    // commit: 2 accepted, pos0=100 → n_keep=3, entries=[3,2,100,101,102,...]
    std::vector<int32_t> entries;
    CHECK(v.commit(2, 100, entries));
    CHECK((int)entries.size() >= 5);
    CHECK(entries[0] == 3);
    CHECK(entries[1] == 2);
    CHECK(entries[2] == 100);
    CHECK(entries[3] == 101);
    CHECK(entries[4] == 102);
}

// ---- GPU: window_ids and dense_steps kernels ------------------------------------------

static void test_gpu_kernels() {
    // window_ids: n_kv=10, window=4 → start=6, width=4, ids=[6,7,8,9]
    int32_t *d_ids, *d_width;
    CUOK(cudaMalloc(&d_ids, 8 * sizeof(int32_t)));
    CUOK(cudaMalloc(&d_width, sizeof(int32_t)));
    CUOK(cudaMemset(d_ids, 0xFF, 8 * sizeof(int32_t)));
    int r = baloon::verify_kernels::window_ids(d_ids, d_width, 10, 4, nullptr);
    CUOK(cudaDeviceSynchronize());
    CHECK(r == 0);
    int32_t h_ids[8]; int32_t h_width;
    CUOK(cudaMemcpy(h_ids, d_ids, 4 * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUOK(cudaMemcpy(&h_width, d_width, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK(h_width == 4);
    CHECK(h_ids[0] == 6); CHECK(h_ids[1] == 7);
    CHECK(h_ids[2] == 8); CHECK(h_ids[3] == 9);

    // dense_steps: T=2 windows starting at c=100 → 2 rows of 4 ints each
    int32_t *d_steps;
    CUOK(cudaMalloc(&d_steps, 2 * 4 * sizeof(int32_t)));
    CUOK(cudaMemset(d_steps, 0, 2 * 4 * sizeof(int32_t)));
    r = baloon::verify_kernels::dense_steps(d_steps, 100, 2, nullptr);
    CUOK(cudaDeviceSynchronize());
    CHECK(r == 0);
    int32_t h_steps[8];
    CUOK(cudaMemcpy(h_steps, d_steps, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost));
    // row 0: c=100 → [100, 101, 25, 101] (101/4=25)
    CHECK(h_steps[0] == 100); CHECK(h_steps[1] == 101);
    CHECK(h_steps[2] == 25);  CHECK(h_steps[3] == 101);
    // row 1: c=101 → [101, 102, 25, 102] (102/4=25)
    CHECK(h_steps[4] == 101); CHECK(h_steps[5] == 102);
    CHECK(h_steps[6] == 25);  CHECK(h_steps[7] == 102);

    cudaFree(d_ids); cudaFree(d_width); cudaFree(d_steps);
}

// ---- GPU: gdn_conv_commit -------------------------------------------------------------

static void test_gdn_conv_commit() {
    // hist [3][C] all zeros; x [2][C] filled 1..2C
    // After commit(n=2): hist = [hist[2], x[0], x[1]] = [0-row, 1..C, C+1..2C]
    const int C = 16, n = 2;
    std::vector<float> h_hist(3 * C, 0.0f), h_x(n * C);
    for (int i = 0; i < n * C; ++i) h_x[i] = (float)(i + 1);
    float *d_hist, *d_x;
    CUOK(cudaMalloc(&d_hist, 3 * C * sizeof(float)));
    CUOK(cudaMalloc(&d_x, n * C * sizeof(float)));
    CUOK(cudaMemcpy(d_hist, h_hist.data(), 3 * C * sizeof(float), cudaMemcpyHostToDevice));
    CUOK(cudaMemcpy(d_x, h_x.data(), n * C * sizeof(float), cudaMemcpyHostToDevice));
    int r = baloon::verify_kernels::gdn_conv_commit(d_x, d_hist, n, C, nullptr);
    CUOK(cudaDeviceSynchronize());
    CHECK(r == 0);
    std::vector<float> h_out(3 * C);
    CUOK(cudaMemcpy(h_out.data(), d_hist, 3 * C * sizeof(float), cudaMemcpyDeviceToHost));
    // hist[0] = old hist[2] = 0 (chan 0)
    CHECK(std::abs(h_out[0 * C + 0] - 0.0f) < 1e-5f);
    // hist[1] = x[0] = x[0*16+0] = 1 (chan 0)
    CHECK(std::abs(h_out[1 * C + 0] - 1.0f) < 1e-5f);
    // hist[2] = x[1] = x[1*16+0] = 17 (chan 0)
    CHECK(std::abs(h_out[2 * C + 0] - (float)(C + 1)) < 1e-5f);
    cudaFree(d_hist); cudaFree(d_x);
}

int main() {
    int ngpu = 0;
    CUOK(cudaGetDeviceCount(&ngpu));
    if (ngpu == 0) { std::printf("no GPU found\n"); return 2; }
    CUOK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUOK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s\n", prop.name);

    test_session_constants();
    test_spec_controller();
    test_suffix_drafter();
    test_verifier_cpu();
    test_gpu_kernels();
    test_gdn_conv_commit();

    std::printf("ok=%ld  fail=%ld\n", g_ok, g_fail);
    if (g_fail == 0) {
        std::printf("U16_UB848 GPU TESTS PASSED\n");
        return 0;
    }
    std::printf("U16_UB848 GPU TESTS FAILED\n");
    return 1;
}
