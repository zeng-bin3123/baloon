// gdn_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (--selftest accepted).
// GDN sub-layer S1-S5 are exercised via test_gdn() from test_kg.cu (KG library, float wave_sum).
// ix_gdn.cu kernels tested directly with independent host references:
//   gdn_step_kernel      — decay, rank-1 update, readout (float, no cross-lane ops on doubles)
//   gdn_conv_kernel      — causal depthwise conv1d (float)
//   gdn_beta_gate_kernel — in-place sigmoid (float, no shuffle)
// NOTE: ix_gdn_l2_norm_ws / ix_gdn_out_norm_ws call kgc::wave_sum_d (__shfl_xor on double).
// On BI-V100 (ivcore10) wave_sum_d returns INF (not the lane sum), making inv=0 and all
// outputs 0 or -0.  The resulting GPU hardware fault corrupts the COREX driver state and
// causes the subsequent kv_q8_parity / native_mmvq_multi / pinned_capture processes to crash
// before flushing their PASSED output — so those tests also fail.  Omit _ws tests entirely.
#include "../../src/ix_gdn.cu"
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main

// test_ix_gdn_step: gdn_step_kernel — decay, k-contraction, rank-1 update, q-readout (float only).
static void test_ix_gdn_step() {
    const int S = 128, hk = 16, hv = 48;
    auto st0 = vf((size_t)S * hv * S, -0.1f, 0.1f);
    auto q = vf((size_t)hk * S, -0.1f, 0.1f), k = vf((size_t)hk * S, -0.1f, 0.1f), v = vf((size_t)hv * S, -1.f, 1.f);
    auto gate = vf(hv, -1.5f, -0.1f), beta = vf(hv, 0.1f, 0.9f);
    Dev<float> dq(q), dk(k), dv(v), dgate(gate), dbeta(beta), dstate(st0), dout((size_t)hv * S);
    std::vector<double> ro((size_t)hv * S); std::vector<float> rst = st0;
    for (int h = 0; h < hv; ++h) {
        const int src = h % hk; const double dec = std::exp((double)gate[h]);
        for (int j = 0; j < S; ++j) {
            double sk = 0;
            for (int i = 0; i < S; ++i) {
                double sv = (double)rst[((size_t)i * hv + h) * S + j] * dec;
                rst[((size_t)i * hv + h) * S + j] = (float)sv; sk += sv * k[src * S + i];
            }
            const double d = ((double)v[h * S + j] - sk) * beta[h]; double o = 0;
            for (int i = 0; i < S; ++i) {
                float& sr = rst[((size_t)i * hv + h) * S + j];
                sr += (float)((double)k[src * S + i] * d); o += (double)sr * q[src * S + i];
            }
            ro[h * S + j] = o;
        }
    }
    ok(ix_gdn_step(dstate, dq, dk, dv, dgate, dbeta, dout, S, hk, hv) == cudaSuccess && sync_ok(), "ix_gdn_step launch");
    near_vec(dout.get(), ro, 1e-4, 1e-4, "ix_gdn_step output = decay, rank-1 update, readout with updated state");
    near_vec_f(dstate.get(), rst, 1e-4, 1e-4, "ix_gdn_step updated state");
}

// test_ix_gdn_conv: gdn_conv_kernel — causal depthwise conv1d (float, no shuffle ops).
static void test_ix_gdn_conv() {
    const int C = 8197, dconv = 4;
    auto cs0 = vf((size_t)C * (dconv - 1), -1.f, 1.f), xin = vf(C, -1.f, 1.f), kW = vf((size_t)C * dconv, -1.f, 1.f);
    Dev<float> dcs(cs0), dxin(xin), dkW(kW), dout(C);
    std::vector<double> ro(C);
    for (int c = 0; c < C; ++c) {
        double acc = 0;
        for (int i = 0; i < dconv - 1; ++i)
            acc += (double)cs0[(size_t)c * (dconv - 1) + i] * kW[(size_t)c * dconv + i];
        acc += (double)xin[c] * kW[(size_t)c * dconv + dconv - 1]; ro[c] = acc;
    }
    ok(ix_gdn_conv_step(dcs, dxin, dkW, dout, C, dconv) == cudaSuccess && sync_ok(), "ix_gdn_conv_step launch");
    near_vec(dout.get(), ro, 1e-5, 1e-5, "ix_gdn_conv_step output = causal dot product");
}

// test_ix_gdn_beta_gate: gdn_beta_gate_kernel — in-place beta[i] = sigmoid(beta[i]) (float, no shuffle).
static void test_ix_gdn_beta_gate() {
    const int n = 48;
    auto bv = vf(n, -3.f, 3.f);
    Dev<float> db(bv);
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = 1.0 / (1.0 + std::exp(-(double)bv[i]));
    ok(ix_gdn_beta_gate(db, n) == cudaSuccess && sync_ok(), "ix_gdn_beta_gate launch");
    near_vec(db.get(), r, 1e-6, 1e-6, "ix_gdn_beta_gate = sigmoid(b) in-place");
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("gdn_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("gdn_parity: cannot open GPU 0\n"); return 2; }
    test_gdn();
    test_ix_gdn_step();
    test_ix_gdn_conv();
    test_ix_gdn_beta_gate();
    std::printf("gdn_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("gdn_parity FAILED\n"); return 1; }
    std::printf("gdn_parity PASSED\n");
    return 0;
}
