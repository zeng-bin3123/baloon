// Invariants of the PRODUCTION grouped-expert launcher (src/kernels/gpu/iq_quant.cu), checked on the device:
//  * native_expert_scratch_bytes == iqdot::native_scratch_bytes (the theory formula 3*align256(n_ff*4)+align256((n_ff/32)*36) plus the
//    q8_1 image of the input) and therefore >= the theory's grouped_scratch_bytes(1, n_ff);
//  * the launcher accepts exactly that many bytes and rejects one byte less; it never writes past them;
//  * the carved regions hold what the theory says: q8_1(x), gate/up planes, h = silu(gate)*up, q8_1(h), and the result equals the
//    host evaluation quantize -> gate/up -> SwiGLU -> quantize -> down.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/f16_bits.hpp"
#include "baloon/iq_dot_geometry.hpp"
#include "baloon/kg_kernels.hpp"
#include "baloon/scalar_dequantizers.hpp"

using namespace baloon;
using namespace baloon::kg;
namespace fb = baloon::f16bits;

static int g_fail = 0, g_checks = 0;
static std::mt19937_64 rng(777);
static float rf(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
static void ok(bool c, const char* what) { ++g_checks; if (!c) { ++g_fail; std::printf("FAIL: %s\n", what); } }
static bool sync_ok() { return cudaDeviceSynchronize() == cudaSuccess; }

static uint64_t al(uint64_t v) { return (v + 255) / 256 * 256; }

// host q8_1: d = amax/127 (f16), q = round-half-away(v / d), 36-byte blocks, plus the dequantized activation d_f16 * q.
static void q8_host(const std::vector<float>& x, std::vector<uint8_t>& q, std::vector<float>& xd) {
    const size_t nb = x.size() / 32; q.assign(nb * 36, 0); xd.assign(x.size(), 0.f);
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0; for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f;
        int sum = 0;
        for (int i = 0; i < 32; ++i) { const int v = (int)std::roundf(x[b * 32 + i] * id); sum += v; q[b * 36 + 4 + i] = (uint8_t)(int8_t)v; xd[b * 32 + i] = (float)v; }
        const uint16_t dh = fb::f16_from_f32(d), sh = fb::f16_from_f32(d * (float)sum);
        q[b * 36] = dh & 0xff; q[b * 36 + 1] = dh >> 8; q[b * 36 + 2] = sh & 0xff; q[b * 36 + 3] = sh >> 8;
        const float df = fb::f32_from_f16(dh);
        for (int i = 0; i < 32; ++i) xd[b * 32 + i] *= df;
    }
}

static void check_formula() {
    const int es[] = {32, 64, 96, 256, 1024, 2560, 4096};
    const int fs[] = {32, 64, 96, 512, 1536, 2048, 8192};
    bool all = true, dom = true, tight = true;
    for (int e : es) for (int f : fs) {
        const uint64_t prod = native_expert_scratch_bytes(e, f);
        all = all && prod == iqdot::native_scratch_bytes((uint64_t)e, (uint64_t)f);
        dom = dom && prod >= iqdot::grouped_scratch_bytes(1, (uint64_t)f) && prod >= 3ull * f * 4;
        tight = tight && prod == al((uint64_t)(e / 32) * 36) + iqdot::grouped_scratch_bytes(1, (uint64_t)f);
    }
    ok(all, "native_expert_scratch_bytes == iqdot::native_scratch_bytes (production vs theory) for every sweep point");
    ok(dom, "production scratch dominates the theory formula (three float planes + q8_1 image of the intermediate)");
    ok(tight, "production scratch == q8_1 image of the input + theory grouped_scratch_bytes(1, n_ff)");
    ok(native_expert_scratch_bytes(64, 96) == 256 + 3 * 512 + 256, "scratch(64, 96) literal: 256 + 3*512 + 256");
    ok(native_expert_scratch_bytes(32, 32) == 256 + 3 * 256 + 256, "scratch(32, 32) literal: 256 + 3*256 + 256");
}

static void check_launcher() {
    const Fmt f = Fmt::IQ4_NL; const int n_embd = 64, n_ff = 96, B = 18; ExpertLayout L;
    ok(native_expert_layout(f, n_embd, n_ff, &L), "layout");
    const iqdot::ExpertLayout T = iqdot::expert_layout(20, 20, n_embd, n_ff);
    ok(L.up_off == T.up_off && L.down_off == T.down_off && L.total == T.bytes, "production layout == theory layout");
    std::vector<uint8_t> blob(L.total);
    for (size_t b = 0; b < L.total / B; ++b) {
        uint8_t* p = &blob[b * B]; const uint16_t d = fb::f16_from_f32(rf(0.01f, 0.2f)); p[0] = d & 0xff; p[1] = d >> 8;
        for (int i = 2; i < B; ++i) p[i] = (uint8_t)(rng() & 0xff);
    }
    std::vector<float> x(n_embd); for (auto& v : x) v = rf(-1, 1);
    const size_t need = native_expert_scratch_bytes(n_embd, n_ff), guard = 512;
    uint8_t *db = nullptr, *scr = nullptr; float *dx = nullptr, *dout = nullptr;
    cudaMalloc(&db, blob.size()); cudaMalloc(&scr, need + guard); cudaMalloc(&dx, n_embd * 4); cudaMalloc(&dout, n_embd * 4);
    cudaMemcpy(db, blob.data(), blob.size(), cudaMemcpyHostToDevice); cudaMemcpy(dx, x.data(), n_embd * 4, cudaMemcpyHostToDevice);
    IqGrids none;
    cudaMemset(scr, 0xA5, need + guard);
    ok(native_expert_grouped(f, db, dx, n_embd, n_ff, dout, scr, need - 1, none) == cudaErrorInvalidValue, "one byte less than the scratch size is rejected");
    ok(native_expert_grouped(f, db, dx, n_embd, n_ff, dout, scr, 0, none) == cudaErrorInvalidValue, "zero scratch bytes rejected");
    std::vector<uint8_t> chk(need + guard); cudaMemcpy(chk.data(), scr, need + guard, cudaMemcpyDeviceToHost);
    bool untouched = true; for (uint8_t c : chk) untouched = untouched && c == 0xA5;
    ok(untouched, "a rejected call writes nothing into the scratch");
    ok(native_expert_grouped(f, db, dx, n_embd, n_ff, dout, scr, need, none) == cudaSuccess && sync_ok(), "exactly the scratch size is accepted");
    cudaMemcpy(chk.data(), scr, need + guard, cudaMemcpyDeviceToHost);
    bool guard_ok = true; for (size_t i = need; i < need + guard; ++i) guard_ok = guard_ok && chk[i] == 0xA5;
    ok(guard_ok, "the launcher never writes past native_expert_scratch_bytes");

    // regions
    const uint64_t o_xq = 0, o_gt = al((uint64_t)(n_embd / 32) * 36), o_up = o_gt + al((uint64_t)n_ff * 4), o_h = o_up + al((uint64_t)n_ff * 4),
                   o_hq = o_h + al((uint64_t)n_ff * 4), o_end = o_hq + al((uint64_t)(n_ff / 32) * 36);
    ok(o_end == need, "carved regions end exactly at the scratch size");
    std::vector<uint8_t> xq; std::vector<float> xd; q8_host(x, xq, xd);
    ok(std::memcmp(&chk[o_xq], xq.data(), xq.size()) == 0, "region 0 holds the q8_1 image of x");
    auto plane = [&](uint64_t off, int n) { std::vector<float> v(n); std::memcpy(v.data(), &chk[off], (size_t)n * 4); return v; };
    auto gt = plane(o_gt, n_ff), up = plane(o_up, n_ff), h = plane(o_h, n_ff);
    auto mv = [&](size_t off, int rows, int cols, const std::vector<float>& xdq) {
        std::vector<float> o(rows);
        for (int r = 0; r < rows; ++r) {
            double a = 0; const size_t rb = (size_t)(cols / 32) * B;
            for (int b = 0; b < cols / 32; ++b) { float y[32]; sdq::deq_iq4_nl(&blob[off + r * rb + (size_t)b * B], y); for (int i = 0; i < 32; ++i) a += (double)y[i] * xdq[b * 32 + i]; }
            o[r] = (float)a;
        }
        return o;
    };
    auto g = mv(L.gate_off, n_ff, n_embd, xd), u = mv(L.up_off, n_ff, n_embd, xd);
    bool gu = true, hh = true;
    for (int i = 0; i < n_ff; ++i) {
        gu = gu && std::fabs(gt[i] - g[i]) <= 1e-3 * std::fabs(g[i]) + 1e-3 && std::fabs(up[i] - u[i]) <= 1e-3 * std::fabs(u[i]) + 1e-3;
        const double s = (double)gt[i] / (1.0 + std::exp(-(double)gt[i]));
        hh = hh && std::fabs(h[i] - (float)(s * up[i])) <= 1e-4 * std::fabs(s * up[i]) + 1e-5;
    }
    ok(gu, "gate and up planes = dequantized rows . dequantized q8_1 activation");
    ok(hh, "h plane = silu(gate) * up");
    std::vector<uint8_t> hq; std::vector<float> hd; q8_host(h, hq, hd);
    ok(std::memcmp(&chk[o_hq], hq.data(), hq.size()) == 0, "last region holds the q8_1 image of h");
    auto o = mv(L.down_off, n_embd, n_ff, hd);
    std::vector<float> out(n_embd); cudaMemcpy(out.data(), dout, n_embd * 4, cudaMemcpyDeviceToHost);
    bool res = true; for (int i = 0; i < n_embd; ++i) res = res && std::fabs(out[i] - o[i]) <= 1e-3 * std::fabs(o[i]) + 1e-3;
    ok(res, "result = down . q8_1(h)");
    cudaFree(db); cudaFree(scr); cudaFree(dx); cudaFree(dout);
}

int main() {
    int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("no GPU\n"); return 2; }
    cudaSetDevice(0);
    check_formula();
    check_launcher();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    if (g_fail) return 1;
    std::printf("U5E0 PROD TESTS PASSED\n");
    return 0;
}
