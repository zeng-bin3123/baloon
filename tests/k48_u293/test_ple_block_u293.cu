// test_ple_block_u293.cu - unit u293ad55355: launches the production kernels of src/cu/ple_block.cu (ple_block, ple_history_advance)
// and checks the PleBlockKernels LAYER 1 statements against an independent double-precision host model:
//   conv taps 0,3,6,9 (row-fastest state), grouped norm, gate = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6))), broadcast i%N / i/N,
//   three-way add, stream sum + SiLU, bf16 projection, history shift, scratch size 223152, overlap rejection.
#include "../k14gpu_common.h"

#include "baloon/kern_ple.hpp"

using namespace baloon::kern;

static double host_gate(double s) {
    const double m = std::fabs(s) > 1e-6 ? std::fabs(s) : 1e-6;
    const double sg = s > 0 ? 1.0 : (s < 0 ? -1.0 : 0.0);
    return 1.0 / (1.0 + std::exp(-sg * std::sqrt(m)));
}

static void run_case(int seed, float gate_scale, bool zero_row0, bool postops) {
    Rng r(seed);
    const int NE_ = 2560, HC = 4, HCD = 10240, NH = 9;
    std::vector<float> convw((size_t)4 * HCD), normw(NE_), hist((size_t)NH * HCD), newrow(HCD), hidden(NE_);
    std::vector<uint16_t> gatew((size_t)HC * NE_), projw((size_t)NE_ * NE_);
    for (auto& v : convw) v = r.sym(0.4f);
    for (auto& v : normw) v = 1.f + r.sym(0.1f);
    for (auto& v : hist) v = r.sym(1.f);
    for (auto& v : newrow) v = r.sym(1.f);
    for (auto& v : hidden) v = r.sym(1.f);
    for (auto& v : gatew) v = baloon::f16bits::round_f32_to_bf16_bits(r.sym(gate_scale));
    if (zero_row0) for (int j = 0; j < NE_; ++j) gatew[j] = 0;            // s = 0 -> sgn 0 -> gate 0.5
    for (auto& v : projw) v = baloon::f16bits::round_f32_to_bf16_bits(r.sym(0.03f));

    std::vector<double> conv(HCD), nrm(HCD), s3(HCD), gate(HC), lg(HC);
    for (int c = 0; c < HCD; ++c) {
        double a = 0;
        for (int k = 0; k < 4; ++k) {
            const int row = 9 - (3 - k) * 3;
            a += (double)convw[k + 4 * c] * (row == 9 ? newrow[c] : hist[row + 9 * c]);
        }
        conv[c] = a;
    }
    for (int s = 0; s < HC; ++s) {
        double ss = 0; for (int i = 0; i < NE_; ++i) ss += conv[s * NE_ + i] * conv[s * NE_ + i];
        const double rr = 1.0 / std::sqrt(ss / NE_ + 1e-6);
        for (int i = 0; i < NE_; ++i) nrm[s * NE_ + i] = conv[s * NE_ + i] * rr * normw[i];
        double t = 0; for (int j = 0; j < NE_; ++j) t += (double)baloon::f16bits::bf16_bits_to_f32(gatew[(size_t)s * NE_ + j]) * hidden[j];
        lg[s] = t; gate[s] = host_gate(t);
    }
    for (int i = 0; i < HCD; ++i) s3[i] = (hidden[i % NE_] * gate[i / NE_] + nrm[i]) + newrow[i];
    std::vector<double> act(NE_);
    for (int j = 0; j < NE_; ++j) { double s = 0; for (int k = 0; k < HC; ++k) s += s3[k * NE_ + j]; act[j] = s / (1.0 + std::exp(-s)); }

    float* dcw = dup(convw); float* dnw = dup(normw); float* dh = dup(hist); float* dn = dup(newrow); float* dhid = dup(hidden);
    uint16_t* dgw = dup(gatew); uint16_t* dpw = dup(projw);
    float* dout = dalloc<float>(NE_); float* dgate = dalloc<float>(HC);
    void* scratch = nullptr; CUOK(cudaMalloc(&scratch, ple_block_scratch_bytes()));
    PleWeights w; w.conv_w = dcw; w.norm_w = dnw; w.gate_w = dgw; w.proj_w = dpw;
    PleOut po; po.out = dout; po.gate = dgate;
    ple_set_native_bf16(false); ple_set_native_postops(postops);
    CHECK(ple_block(w, dn, dh, dhid, po, scratch, 0) == cudaSuccess);
    sync();
    const auto o = fetch(dout, NE_), g = fetch(dgate, HC);
    for (int s = 0; s < HC; ++s) {
        CHECK(near_tol(g[s], gate[s], 1e-5, 1e-4));
        if (!near_tol(g[s], gate[s], 1e-5, 1e-4)) std::printf("  gate[%d]=%g want %g (dot=%g)\n", s, g[s], gate[s], lg[s]);
    }
    if (zero_row0) CHECK(near_tol(g[0], 0.5, 1e-6, 0));
    if (gate_scale >= 0.1f) {   // large dots: the signed-sqrt gate must differ clearly from plain sigmoid for at least one stream
        bool differs = false;
        for (int s = 0; s < HC; ++s) differs |= std::fabs(gate[s] - 1.0 / (1.0 + std::exp(-lg[s]))) > 0.05;
        CHECK(differs);
    }
    bool ok = true;
    for (int oo = 0; oo < NE_; ++oo) {
        double e = postops ? hidden[oo] : 0.0;
        for (int j = 0; j < NE_; ++j) e += (double)baloon::f16bits::bf16_bits_to_f32(projw[(size_t)oo * NE_ + j]) * act[j];
        ok &= near_tol(o[oo], e, 5e-3, 5e-3);
    }
    CHECK(ok);
    // history unchanged by the block; advance shifts left and appends the new row
    CHECK(fetch(dh, hist.size()) == hist);
    CHECK(ple_history_advance(dh, dn, 0) == cudaSuccess);
    sync();
    const auto h2 = fetch(dh, hist.size());
    bool hok = true;
    for (int c = 0; c < HCD; ++c) {
        for (int rr = 0; rr + 1 < NH; ++rr) hok &= (h2[rr + 9 * c] == hist[rr + 1 + 9 * c]);
        hok &= (h2[8 + 9 * c] == newrow[c]);
    }
    CHECK(hok);
    // overlap / null rejection
    CHECK(ple_block(w, dn, dh, (const float*)scratch, po, scratch, 0) != cudaSuccess);
    CHECK(ple_block(w, dn, dh, dhid, po, (void*)dhid, 0) != cudaSuccess);
    CHECK(ple_history_advance(dh, dh, 0) != cudaSuccess);
    PleWeights nw = w; nw.gate_w = nullptr;
    CHECK(ple_block(nw, dn, dh, dhid, po, scratch, 0) != cudaSuccess);
    cudaFree(dcw); cudaFree(dnw); cudaFree(dh); cudaFree(dn); cudaFree(dhid); cudaFree(dgw); cudaFree(dpw); cudaFree(dout); cudaFree(dgate); cudaFree(scratch);
}

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { std::printf("no device\n"); return 2; }
    CHECK(ple_block_available());
    CHECK(ple_block_scratch_bytes() == 223152);
    run_case(1, 0.03f, false, false);     // small dots
    run_case(2, 0.30f, false, true);      // large dots (|s| >> 1) with residual post-op
    run_case(3, 0.30f, true, false);      // exact-zero dot for stream 0
    std::printf("%ld checks, %ld failures\n", g_checks, g_fail);
    if (g_fail == 0) { std::printf("U293 PLE BLOCK TESTS PASSED\n"); return 0; }
    return 1;
}
