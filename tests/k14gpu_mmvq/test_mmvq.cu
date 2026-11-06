// GPU test of the native MMVQ kernels: every supported weight format, 1..8 columns, small-K and large-K shapes, against a
// double-precision CPU model (dequantized weights x Q8_1-quantized activations). Also: device Q8_1 quantizer vs host model,
// multi-exact property, validation errors, dispatcher tables.
#include "../k14gpu_common.h"

#include "baloon/kern_native_mmvq.hpp"

using namespace baloon::kern;
namespace sdq = baloon::sdq;

static void host_deq(int type, const uint8_t* b, float* y) {
    switch (type) {
        case 2: sdq::deq_q4_0(b, y); break;
        case 6: sdq::deq_q5_0(b, y); break;
        case 8: sdq::deq_q8_0(b, y); break;
        case 20: sdq::deq_iq4_nl(b, y); break;
        case 11: sdq::deq_q3_k(b, y); break;
        case 12: sdq::deq_q4_k(b, y); break;
        case 13: sdq::deq_q5_k(b, y); break;
        case 14: sdq::deq_q6_k(b, y); break;
        case 23: sdq::deq_iq4_xs(b, y); break;
        case 42: sdq::deq_q2_0(b, y); break;
    }
}
static void put_h(uint8_t* p, float f) { const uint16_t h = h_of(f); p[0] = (uint8_t)(h & 255); p[1] = (uint8_t)(h >> 8); }
// Random block with valid fp16 scale fields.
static void rand_block(Rng& r, int type, uint8_t* b, int bytes) {
    for (int i = 0; i < bytes; ++i) b[i] = (uint8_t)r.next();
    switch (type) {
        case 2: case 6: case 8: case 20: case 42: case 23: put_h(b, 0.01f + (float)r.uni() * 0.05f); break;
        case 11: put_h(b + 108, 0.01f + (float)r.uni() * 0.02f); break;
        case 12: case 13: put_h(b, 0.01f + (float)r.uni() * 0.02f); put_h(b + 2, 0.01f + (float)r.uni() * 0.02f); break;
        case 14: put_h(b + 208, 0.005f + (float)r.uni() * 0.01f); break;
    }
}

struct Q81Host { std::vector<int8_t> q; std::vector<float> d; };   // per 32 block
static Q81Host host_q81(const std::vector<float>& x) {
    Q81Host o; o.q.resize(x.size()); o.d.resize(x.size() / 32);
    for (size_t b = 0; b < x.size() / 32; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        for (int i = 0; i < 32; ++i) { int q = (int)std::round(x[b * 32 + i] * id); q = std::max(-127, std::min(127, q)); o.q[b * 32 + i] = (int8_t)q; }
        o.d[b] = f_of(h_of(d));
    }
    return o;
}

struct Case { int type, be, bb; };

static void run_type(int type, Rng& r) {
    const baloon::kern::mmvqgeo::WeightEntry* we = baloon::kern::mmvqgeo::find_weight(type);
    CHECK(we != nullptr);
    if (!we) return;
    const int be = (int)we->elems, bb = (int)we->bytes;
    std::vector<int> n_ins = be == 32 ? std::vector<int>{64, 512, 4096} : be == 64 ? std::vector<int>{128, 2560, 8320} : std::vector<int>{256, 1280, 10240};
    for (int n_in : n_ins)
        for (int n_out : {1, 7, 33}) {
            const int nblk = n_in / be;
            std::vector<uint8_t> w((size_t)n_out * nblk * bb);
            for (size_t b = 0; b < (size_t)n_out * nblk; ++b) rand_block(r, type, &w[b * bb], bb);
            std::vector<float> wf((size_t)n_out * n_in);
            for (size_t b = 0; b < (size_t)n_out * nblk; ++b) host_deq(type, &w[b * bb], &wf[b * be]);
            uint8_t* dw = dup(w);
            for (int ncols : {1, 2, 3, 5, 8}) {
                std::vector<float> x((size_t)ncols * n_in);
                for (auto& v : x) v = r.sym(2.f);
                const Q81Host hq = host_q81(x);
                float* dx = dup(x);
                void* dq = nullptr; CUOK(cudaMalloc(&dq, native_q8_1_bytes(ncols, n_in)));
                float* dy = dalloc<float>((size_t)ncols * n_out);
                CHECK(native_quantize_q8_1(dx, dq, (int64_t)ncols * n_in, 0) == cudaSuccess);
                sync();
                // device quantizer == host model (qs and scale bits)
                std::vector<uint8_t> qb = fetch((const uint8_t*)dq, native_q8_1_bytes(ncols, n_in));
                bool qok = true;
                for (size_t b = 0; b < (size_t)ncols * n_in / 32; ++b) {
                    uint16_t dh; std::memcpy(&dh, &qb[b * 36], 2);
                    qok &= (f_of(dh) == hq.d[b]);
                    for (int i = 0; i < 32; ++i) qok &= ((int8_t)qb[b * 36 + 4 + i] == hq.q[b * 32 + i]);
                }
                CHECK(qok);
                CHECK(native_mmvq(type, dw, dq, dy, n_in, n_out, ncols, 0) == cudaSuccess);
                sync();
                std::vector<float> y = fetch(dy, (size_t)ncols * n_out);
                double worst = 0;
                for (int c = 0; c < ncols; ++c)
                    for (int o = 0; o < n_out; ++o) {
                        double e = 0, mag = 0;
                        for (int i = 0; i < n_in; ++i) {
                            const double v = (double)wf[(size_t)o * n_in + i] * hq.q[(size_t)c * n_in + i] * hq.d[((size_t)c * n_in + i) / 32];
                            e += v; mag += std::fabs(v);
                        }
                        worst = std::max(worst, std::fabs(e - y[(size_t)c * n_out + o]) / (mag + 1e-6));
                    }
                CHECK(worst < 2e-4);
                if (worst >= 2e-4) std::printf("  type %d n_in %d n_out %d ncols %d worst %.3g\n", type, n_in, n_out, ncols, worst);
                // multi-exact: column 0 with 8 columns equals the single-column run bit for bit
                if (ncols == 8) {
                    float* dy1 = dalloc<float>(n_out);
                    CHECK(native_mmvq(type, dw, dq, dy1, n_in, n_out, 1, 0) == cudaSuccess);
                    sync();
                    std::vector<float> y1 = fetch(dy1, n_out);
                    bool same = true;
                    for (int o = 0; o < n_out; ++o) same &= (std::memcmp(&y1[o], &y[o], 4) == 0);
                    CHECK(same);
                    native_mmvq_set_multi_exact(false);   // upstream 2-warp layout: different order, same value within rounding
                    float* dy2 = dalloc<float>((size_t)ncols * n_out);
                    CHECK(native_mmvq(type, dw, dq, dy2, n_in, n_out, ncols, 0) == cudaSuccess);
                    sync();
                    native_mmvq_set_multi_exact(true);
                    std::vector<float> y2 = fetch(dy2, (size_t)ncols * n_out);
                    bool close = true;
                    for (size_t i = 0; i < y2.size(); ++i) close &= near_tol(y2[i], y[i], 1e-3, 1e-3);
                    CHECK(close);
                    cudaFree(dy1); cudaFree(dy2);
                }
                // float entry point
                if (ncols == 3) {
                    float* dy3 = dalloc<float>((size_t)ncols * n_out);
                    void* scr = nullptr; CUOK(cudaMalloc(&scr, native_q8_1_bytes(ncols, n_in)));
                    cudaError_t e = cudaErrorInvalidValue;
                    switch (type) {
                        case 2: e = native_q4_0_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                        case 6: e = native_q5_0_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                        case 8: e = native_q8_0_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                        case 20: e = native_iq4_nl_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                        case 11: e = native_q3_k_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                        case 12: e = native_q4_k_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                        case 13: e = native_q5_k_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                        case 14: e = native_q6_k_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                        case 23: e = native_iq4_xs_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                        case 42: e = native_q2_0_f32(dw, dx, dy3, n_in, n_out, ncols, scr, 0); break;
                    }
                    CHECK(e == cudaSuccess);
                    sync();
                    std::vector<float> y3 = fetch(dy3, (size_t)ncols * n_out);
                    bool same = true;
                    for (size_t i = 0; i < y3.size(); ++i) same &= (std::memcmp(&y3[i], &y[i], 4) == 0);
                    CHECK(same);   // quantize + launch gives the very same numbers as the two-step path
                    cudaFree(dy3); cudaFree(scr);
                }
                cudaFree(dx); cudaFree(dq); cudaFree(dy);
            }
            cudaFree(dw);
        }
}

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    CHECK(ndev > 0);
    CUOK(cudaSetDevice(0));
    Rng r(2024);
    // dispatcher tables
    for (int t : {2, 6, 8, 11, 12, 13, 14, 20, 23, 42}) CHECK(native_mmvq_supported(t));
    for (int t : {0, 1, 3, 7, 16, 17, 18, 21, 22, 29, 99}) CHECK(!native_mmvq_supported(t));
    CHECK(native_mmvq_weight_bytes(42, 128, 3) == 3 * 2 * 18);
    CHECK(native_mmvq_weight_bytes(14, 512, 2) == 2 * 2 * 210);
    CHECK(native_mmvq_weight_bytes(16, 256, 4) == 4 * 66);
    CHECK(native_mmvq_weight_bytes(2, 33, 4) == 0 && native_mmvq_weight_bytes(5, 64, 4) == 0);
    CHECK(native_q8_1_bytes(8, 2560) == 8 * 80 * 36);
    CHECK(native_mmvq_multi_exact());
    // validation
    {
        void* a = nullptr; CUOK(cudaMalloc(&a, 1 << 20));
        float* y = dalloc<float>(1024);
        CHECK(native_mmvq(2, a, a, y, 64, 4, 9, 0) != cudaSuccess);        // 9 columns
        CHECK(native_mmvq(2, a, a, y, 63, 4, 1, 0) != cudaSuccess);        // n_in not a block multiple
        CHECK(native_mmvq(14, a, a, y, 128, 4, 1, 0) != cudaSuccess);      // not a K-block multiple
        CHECK(native_mmvq(2, nullptr, a, y, 64, 4, 1, 0) != cudaSuccess);
        CHECK(native_mmvq(77, a, a, y, 64, 4, 1, 0) != cudaSuccess);
        CHECK(native_quantize_q8_1(nullptr, a, 32, 0) != cudaSuccess && native_quantize_q8_1((const float*)a, a, 33, 0) != cudaSuccess);
        cudaFree(a); cudaFree(y);
    }
    for (int t : {2, 6, 8, 20, 11, 12, 13, 14, 23, 42}) { run_type(t, r); std::printf("type %d done (checks %ld, fail %ld)\n", t, g_checks, g_fail); }
    std::printf("checks: %ld, failures: %ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("MMVQ GPU TESTS FAILED\n"); return 1; }
    std::printf("MMVQ GPU TESTS PASSED\n");
    return 0;
}
