// test_uf4e.cu - invariants of SharedExpertBlock (lean/uf4e/SharedExpertBlock.lean) against the real host layout and GPU kernels.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include "baloon/kg_kernels.hpp"
#include "baloon/shared_expert_batch.hpp"
#include "baloon/shared_expert_layout.hpp"

namespace se = baloon::shared_expert;
static long checks = 0, fails = 0;
static void check(bool ok, const char* what) { ++checks; if (!ok) { ++fails; std::printf("  VIOLATED: %s\n", what); } }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
template <class T> static T* dup(const std::vector<T>& v) { T* d; cudaMalloc(&d, v.size() * sizeof(T)); cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }

static void host_invariants() {
    // align16_mod / align16_ge, exhaustively over a range
    for (uint64_t n = 0; n < 5000; ++n) { check(se::align16(n) % 16 == 0, "align16_mod"); check(se::align16(n) >= n, "align16_ge"); check(se::align16(n) < n + 16, "align16 tight"); }
    // offsets aligned + scratch_end + monotone, for per-token form (n_tok = 1) over many n_ff
    for (uint64_t ff = 32; ff <= 4096; ff += 32) {
        check(se::off_up(1, ff) % 16 == 0 && se::off_q0(1, ff) % 16 == 0 && se::off_qk(1, ff) % 16 == 0 && se::off_gate_slots(1, ff) % 16 == 0, "offsets_aligned");
        check(se::off_gate_slots(1, ff) + 32 == se::scratch_bytes(1, ff), "scratch_end");
        check(se::off_up(1, ff) < se::off_q0(1, ff) && se::off_q0(1, ff) <= se::off_qk(1, ff) && se::off_qk(1, ff) <= se::off_gate_slots(1, ff), "regions ordered");
        const uint64_t theory = se::align16(ff * 4) * 2 + se::align16(ff / 32 * 34) + se::align16(ff / 256 * 292) + 32;
        check(theory == se::scratch_bytes(1, ff), "scratchBytes formula");
    }
    check(se::scratch_bytes(1, 640) == 6432, "scratch_640 == 6432");
    check(640 % 32 == 0 && 640 % 256 != 0 && !se::q8k_usable(640), "down_q8k_impossible_640");
    check(se::q8k_usable(512), "q8k usable at 512");
    check(se::act_for(4, 1) == se::Act::Q8_K && se::act_for(4, 0) == se::Act::Q8_0 && se::act_for(2, 1) == se::Act::Q8_0, "activation format selection");
    for (uint64_t n = 0; n < 20; ++n) check(se::valid_tokens(n) == (1 <= n && n <= 8), "tokensOk_iff");
    for (uint64_t k = 0; k < 100; ++k) check(se::valid_combine_k(k) == (0 < k && k <= 64), "kOk");
    check(se::valid_combine_k(10) && !se::valid_combine_k(65), "kOk_cases");
    check(baloon::kg::shared_expert_scratch_bytes(1, 256, 640) == 6432, "kg scratch bytes agree with theory");
    check(baloon::kg::shared_expert_scratch_bytes(0, 256, 640) == 0, "zero tokens -> zero scratch");
}

static void gpu_invariants() {
    using namespace baloon::kg;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    // ---- moe_combine: acc = shared + sum w_e part_e (shared added plain), k guards
    const int E = 128, K = 10;
    std::vector<float> parts((size_t)K * E), w(K), sh(E), y(E);
    for (auto& v : parts) v = U(rng); for (auto& v : w) v = U(rng); for (auto& v : sh) v = U(rng);
    float *dp = dup(parts), *dw = dup(w), *ds = dup(sh), *dy; cudaMalloc(&dy, E * 4);
    check(moe_combine(dp, dw, ds, dy, E, K, 1, 0) == cudaSuccess, "combine k=10 accepted");
    cudaMemcpy(y.data(), dy, E * 4, cudaMemcpyDeviceToHost);
    double worst = 0, worst0 = 0;
    for (int j = 0; j < E; ++j) { double a = 0; for (int e = 0; e < K; ++e) a += (double)w[e] * parts[(size_t)e * E + j]; worst = std::fmax(worst, std::fabs(a + sh[j] - y[j])); }
    check(worst < 1e-5, "combine_eq: routed sum + shared");
    check(moe_combine(dp, dw, nullptr, dy, E, K, 0, 0) == cudaSuccess, "combine without shared");
    cudaMemcpy(y.data(), dy, E * 4, cudaMemcpyDeviceToHost);
    for (int j = 0; j < E; ++j) { double a = 0; for (int e = 0; e < K; ++e) a += (double)w[e] * parts[(size_t)e * E + j]; worst0 = std::fmax(worst0, std::fabs(a - y[j])); }
    check(worst0 < 1e-5, "no-shared == routed sum");
    // shared passes straight through: scaling routed weights to zero leaves exactly the shared output
    std::vector<float> w0(K, 0.f); float* dw0 = dup(w0);
    check(moe_combine(dp, dw0, ds, dy, E, K, 1, 0) == cudaSuccess, "combine zero weights");
    cudaMemcpy(y.data(), dy, E * 4, cudaMemcpyDeviceToHost);
    check(std::memcmp(y.data(), sh.data(), E * 4) == 0, "combine_nil: output is exactly the shared output");
    // guard k > 64 (parts buffer big enough so a missing guard would not fault)
    std::vector<float> bigp((size_t)65 * E, 0.f), bigw(65, 1.f); float *dbp = dup(bigp), *dbw = dup(bigw);
    check(moe_combine(dbp, dbw, ds, dy, E, 65, 1, 0) != cudaSuccess, "kOk: k=65 refused");
    check(moe_combine(dbp, dbw, ds, dy, E, 64, 1, 0) == cudaSuccess, "k=64 accepted");
    check(moe_combine(dp, dw, ds, dy, E, 0, 1, 0) != cudaSuccess, "k=0 refused");
    cudaDeviceSynchronize();
    // ---- batched combine kernel (shared_expert_batch.cu): each token row equals the single-token combine
    {
        const int T = 5;
        std::vector<float> bp((size_t)T * K * E), bw((size_t)T * K), bs((size_t)T * E), by((size_t)T * E);
        for (auto& v : bp) v = U(rng); for (auto& v : bw) v = U(rng); for (auto& v : bs) v = U(rng);
        float *dbp2 = dup(bp), *dbw2 = dup(bw), *dbs2 = dup(bs), *dby2; cudaMalloc(&dby2, by.size() * 4);
        for (int hs = 0; hs < 2; ++hs) {
            check(moe_combine_rows(dbp2, dbw2, hs ? dbs2 : nullptr, dby2, T, E, K, hs, 0) == cudaSuccess, "rows combine accepted");
            cudaDeviceSynchronize();
            cudaMemcpy(by.data(), dby2, by.size() * 4, cudaMemcpyDeviceToHost);
            double wr = 0;
            for (int t = 0; t < T; ++t) for (int j = 0; j < E; ++j) {
                double a = hs ? bs[(size_t)t * E + j] : 0.0;
                for (int e = 0; e < K; ++e) a += (double)bw[(size_t)t * K + e] * bp[((size_t)t * K + e) * E + j];
                wr = std::fmax(wr, std::fabs(a - by[(size_t)t * E + j]));
            }
            check(wr < 1e-5, "rows combine == shared + routed sum per token");
        }
        std::vector<float> bz((size_t)T * K, 0.f); float* dbz = dup(bz);
        moe_combine_rows(dbp2, dbz, dbs2, dby2, T, E, K, 1, 0);
        cudaMemcpy(by.data(), dby2, by.size() * 4, cudaMemcpyDeviceToHost);
        check(std::memcmp(by.data(), bs.data(), by.size() * 4) == 0, "rows combine_nil: exactly the shared output");
        check(moe_combine_rows(dbp2, dbw2, dbs2, dby2, T, E, 65, 1, 0) != cudaSuccess, "rows: k=65 refused");
        check(moe_combine_rows(dbp2, dbw2, dbs2, dby2, T, E, 0, 1, 0) != cudaSuccess, "rows: k=0 refused");
        check(moe_combine_rows(dbp2, dbw2, dbs2, dby2, 0, E, K, 1, 0) != cudaSuccess, "rows: 0 tokens refused");
        check(moe_combine_rows(dbp2, dbw2, nullptr, dby2, T, E, K, 1, 0) != cudaSuccess, "rows: missing shared refused");
    }
    // ---- shared expert: h = silu(x Wg) * (x Wu); out = (h Wd) * sigmoid(x . wgi)
    const int NE = 256, NF = 640;
    std::vector<uint16_t> wg((size_t)NF * NE), wu((size_t)NF * NE), wd((size_t)NE * NF), wgi(NE);
    for (auto& v : wg) v = f2bf(U(rng) * 0.2f); for (auto& v : wu) v = f2bf(U(rng) * 0.2f);
    for (auto& v : wd) v = f2bf(U(rng) * 0.2f); for (auto& v : wgi) v = f2bf(U(rng));
    uint16_t *dg = dup(wg), *du = dup(wu), *dd = dup(wd), *dgi = dup(wgi);
    const size_t sb = shared_expert_scratch_bytes(8, NE, NF);
    void* scr; cudaMalloc(&scr, sb);
    std::vector<float> xs((size_t)8 * NE); for (auto& v : xs) v = U(rng);
    auto host_ref = [&](const float* x, std::vector<double>& out) {
        std::vector<double> h(NF); out.assign(NE, 0.0);
        for (int f = 0; f < NF; ++f) { double g = 0, u = 0; for (int c = 0; c < NE; ++c) { g += (double)bf2f(wg[(size_t)f * NE + c]) * x[c]; u += (double)bf2f(wu[(size_t)f * NE + c]) * x[c]; } h[f] = g / (1 + std::exp(-g)) * u; }
        double s = 0; for (int c = 0; c < NE; ++c) s += (double)bf2f(wgi[c]) * x[c]; const double gate = 1 / (1 + std::exp(-s));
        for (int o = 0; o < NE; ++o) { double a = 0; for (int f = 0; f < NF; ++f) a += (double)bf2f(wd[(size_t)o * NF + f]) * h[f]; out[o] = a * gate; }
    };
    for (int native = 0; native < 2; ++native) {
        shared_expert_set_native_bf16(native != 0);
        const double tol = native ? 3e-2 : 1e-2;
        float *dx = dup(xs), *dout; cudaMalloc(&dout, (size_t)8 * NE * 4);
        check(shared_expert_multi(dx, 8, NE, NF, dg, du, dd, dgi, dout, scr, sb, 0) == cudaSuccess, "multi n=8 accepted");
        cudaDeviceSynchronize();
        std::vector<float> o8((size_t)8 * NE); cudaMemcpy(o8.data(), dout, o8.size() * 4, cudaMemcpyDeviceToHost);
        double num = 0, den = 0;
        for (int t = 0; t < 8; ++t) { std::vector<double> r; host_ref(&xs[(size_t)t * NE], r); for (int o = 0; o < NE; ++o) { num += std::fabs(r[o] - o8[(size_t)t * NE + o]); den += std::fabs(r[o]); } }
        std::printf("  native=%d shared_expert rel-L1 vs fp64 reference: %.3e\n", native, (double)(num / den));
        check(num / den < tol, "shared expert matches silu(gate)*up -> down -> scalar sigmoid gate");
        // single-token entry point equals the batched result row by row
        float* d1; cudaMalloc(&d1, NE * 4);
        for (int t = 0; t < 8; t += 3) {
            check(shared_expert(dx + (size_t)t * NE, NE, NF, dg, du, dd, dgi, d1, scr, sb, 0) == cudaSuccess, "single accepted");
            std::vector<float> o1(NE); cudaMemcpy(o1.data(), d1, NE * 4, cudaMemcpyDeviceToHost);
            double dm = 0; for (int o = 0; o < NE; ++o) dm = std::fmax(dm, std::fabs(o1[o] - o8[(size_t)t * NE + o]));
            check(dm < 1e-4, "batch row == single token");
        }
        // guards: tokens 0 and 9 refused, short scratch refused
        check(shared_expert_multi(dx, 0, NE, NF, dg, du, dd, dgi, dout, scr, sb, 0) != cudaSuccess, "tokensOk: 0 refused");
        check(shared_expert_multi(dx, 9, NE, NF, dg, du, dd, dgi, dout, scr, sb, 0) != cudaSuccess, "tokensOk: 9 refused");
        check(shared_expert_multi(dx, 8, NE, NF, dg, du, dd, dgi, dout, scr, sb - 16, 0) != cudaSuccess, "short scratch refused");
        cudaFree(dx); cudaFree(dout); cudaFree(d1);
    }
    shared_expert_set_native_bf16(false);
    // sigmoid gate scales: negate x -> gate changes, output linear in gate; check gate extremes via huge gate weight
    std::vector<uint16_t> big(NE, f2bf(8.f)); uint16_t* dbig = dup(big);
    std::vector<float> xp(NE, 1.f); float *dxp = dup(xp), *dop; cudaMalloc(&dop, NE * 4);
    shared_expert(dxp, NE, NF, dg, du, dd, dbig, dop, scr, sb, 0);     // gate -> 1
    std::vector<float> a(NE); cudaMemcpy(a.data(), dop, NE * 4, cudaMemcpyDeviceToHost);
    std::vector<uint16_t> neg(NE, f2bf(-8.f)); uint16_t* dneg = dup(neg);
    shared_expert(dxp, NE, NF, dg, du, dd, dneg, dop, scr, sb, 0);     // gate -> 0
    std::vector<float> b(NE); cudaMemcpy(b.data(), dop, NE * 4, cudaMemcpyDeviceToHost);
    double ma = 0, mb = 0; for (int o = 0; o < NE; ++o) { ma = std::fmax(ma, std::fabs(a[o])); mb = std::fmax(mb, std::fabs(b[o])); }
    check(ma > 1e-3 && mb < 1e-6 * (1 + ma), "scalar sigmoid gate saturates the shared output");
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("no GPU device\n"); return 2; }
    cudaSetDevice(0);
    host_invariants();
    gpu_invariants();
    std::printf("uf4e: checks=%ld violations=%ld\n", checks, fails);
    if (checks == 0 || fails) { std::printf("uf4e: FAIL\n"); return 1; }
    std::printf("uf4e: PASS\n");
    return 0;
}
