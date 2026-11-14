// test_u41a_gpu.cu — unit u41a85163f4 GPU kernel-launch + result-correctness test.
// Opens a GPU device, calls ple_block() (launches conv_kernel gnorm_kernel gate_kernel
// bcast_kernel add3_kernel to_bf16_kernel bf16_gemv_kernel) and ple_history_advance()
// (history_advance_kernel), then compares GPU outputs with CPU reference values computed
// for the same inputs.  The test FAILS if any skipped/no-op kernel would cause a mismatch.
//
// CPU reference trace (see comments in main):
//   gate ≈ 0.73106,  out ≈ 11.924  for all elements
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "baloon/kern_ple.hpp"
#include "baloon/kern_geometry.hpp"

using namespace baloon::kern;
using namespace plegeo;

static long g_ok = 0, g_fail = 0;
#define CHECK(c) do { if (c) { ++g_ok; } else { ++g_fail; \
    std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA %d at %s:%d\n", (int)e_, __FILE__, __LINE__); std::exit(2); } } while (0)

template<class T> static T* dalloc(size_t n) {
    T* p = nullptr; CUOK(cudaMalloc(&p, (n ? n : 1) * sizeof(T))); return p;
}
template<class T> static std::vector<T> stoh(const T* p, size_t n) {
    std::vector<T> v(n);
    CUOK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static void sync() { CUOK(cudaGetLastError()); CUOK(cudaDeviceSynchronize()); }

// Encode float → bf16 bits (host-side, same rounding as the GPU to_bf16 / bf16_bits helpers)
static uint16_t f32_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    // round-to-nearest-even: add 0x7FFF + lsb of bit 16
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}
// Decode bf16 bits → float (host-side)
static float bf16_bits_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// CPU reference sigmoid(sign(t) * sqrt(max(|t|, 1e-6)))
static double cpu_gate_sigmoid(double t) {
    double m = std::fabs(t) < 1e-6 ? 1e-6 : std::fabs(t);
    double sign = t > 0.0 ? 1.0 : (t < 0.0 ? -1.0 : 0.0);
    return 1.0 / (1.0 + std::exp(-sign * std::sqrt(m)));
}
// CPU silu(x) = x/(1+exp(-x))
static double cpu_silu(double x) { return x / (1.0 + std::exp(-x)); }

int main() {
    // Open a GPU device (required by the kernel probe check)
    CUOK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUOK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s\n", prop.name);

    // Geometry (from kern_geometry.hpp, namespace baloon::plegeo)
    const int NE  = (int)NG_EMBD;    // 2560
    const int NHC = (int)NG_HC;      // 4
    const int HCD = (int)NG_HC_DIM;  // 10240
    const int NH  = (int)NG_HIST;    // 9
    const int CW  = (int)CONV_KERNEL; // 4

    // ---- Exact bf16 values used for gate_w and proj_w -------------------------
    // gate_w / proj_w = bf16(1/NE) for every element.
    // We reconstruct the exact float so the CPU reference matches the GPU precisely.
    const uint16_t gw_bits = f32_to_bf16_bits(1.0f / (float)NE);
    const uint16_t pw_bits = f32_to_bf16_bits(1.0f / (float)NE);
    const double gw_f = (float)bf16_bits_to_f32(gw_bits); // exact bf16 value ≈ 1/2560
    const double pw_f = (float)bf16_bits_to_f32(pw_bits);

    // ---- CPU reference trace --------------------------------------------------
    // conv_w:  only tap 3 (new_row tap) = 1.0, rest 0  → conv[i] = new_row[i] = 1.0
    const double conv_val = 1.0;

    // gnorm_kernel: rms_scale = 1 / sqrt(sum_of_sq/NE + 1e-6) on conv = all conv_val
    const double gnorm_sumsq = NE * conv_val * conv_val;
    const double gnorm_scale = 1.0 / std::sqrt(gnorm_sumsq / NE + 1e-6);
    const double nrm_val = conv_val * gnorm_scale * 1.0; // norm_w = 1

    // gate_kernel: t = sum_j gw_f * hidden[j] = gw_f * NE (hidden = all 1)
    const double t_ref = gw_f * NE;
    const double gate_ref = cpu_gate_sigmoid(t_ref);

    // bcast_kernel: bc[i] = hidden[i%NE] * gate[i/NE] = 1 * gate_ref
    const double bc_val = 1.0 * gate_ref;

    // add3_kernel: s3[i] = bc[i] + nrm[i] + new_row[i] = bc + nrm + 1
    const double s3_val = bc_val + nrm_val + 1.0;

    // to_bf16_kernel: r[j] = silu(sum_{s} s3[s*NE+j]) = silu(NHC * s3_val)
    const double r_ref = cpu_silu((float)NHC * s3_val);

    // bf16_gemv_kernel: out[o] = hidden[o] + dot(proj_w[o], r)
    //   dot = sum_j pw_f * r_ref = pw_f * NE * r_ref
    const double dot_ref = pw_f * NE * r_ref;
    const double out_ref = dot_ref;  // native_postops=false: no residual add

    std::printf("ref: conv=%.4g nrm=%.4g gate=%.5g bc=%.4g s3=%.4g r=%.4g out=%.4g\n",
        (float)conv_val, (float)nrm_val, (float)gate_ref,
        (float)bc_val, (float)s3_val, (float)r_ref, (float)out_ref);

    // ---- Build host-side weights ----------------------------------------------
    // conv_w: layout k + 4*c; only tap k=3 = 1.0 per channel c
    std::vector<float> h_conv_w((size_t)CW * HCD, 0.0f);
    for (int c = 0; c < HCD; ++c) h_conv_w[3 + 4 * c] = 1.0f;

    std::vector<float>    h_norm_w(NE, 1.0f);
    std::vector<uint16_t> h_gate_w((size_t)NHC * NE, gw_bits);
    std::vector<uint16_t> h_proj_w((size_t)NE * NE, pw_bits);
    std::vector<float>    h_hidden(NE, 1.0f);
    std::vector<float>    h_new_row(HCD, 1.0f);
    std::vector<float>    h_hist((size_t)NH * HCD, 0.0f);

    // ---- Upload to GPU --------------------------------------------------------
    float*    d_conv_w = dalloc<float>(h_conv_w.size());
    float*    d_norm_w = dalloc<float>(h_norm_w.size());
    uint16_t* d_gate_w = dalloc<uint16_t>(h_gate_w.size());
    uint16_t* d_proj_w = dalloc<uint16_t>(h_proj_w.size());
    float*    d_hidden = dalloc<float>(h_hidden.size());
    float*    d_new_row = dalloc<float>(h_new_row.size());
    float*    d_hist   = dalloc<float>(h_hist.size());
    float*    d_out    = dalloc<float>(NE);
    float*    d_gate   = dalloc<float>(NHC);
    void*     d_scratch = nullptr;
    CUOK(cudaMalloc(&d_scratch, ple_block_scratch_bytes()));

#define UPL(d,h,T) CUOK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice))
    UPL(d_conv_w,  h_conv_w,  float);
    UPL(d_norm_w,  h_norm_w,  float);
    UPL(d_gate_w,  h_gate_w,  uint16_t);
    UPL(d_proj_w,  h_proj_w,  uint16_t);
    UPL(d_hidden,  h_hidden,  float);
    UPL(d_new_row, h_new_row, float);
    UPL(d_hist,    h_hist,    float);
#undef UPL

    // Zero the output buffers so NOP kernels (which never write them) are detected.
    CUOK(cudaMemset(d_scratch, 0, ple_block_scratch_bytes()));  // stale scratch → stale rvec → false positive
    CUOK(cudaMemset(d_gate, 0, NHC * sizeof(float)));
    CUOK(cudaMemset(d_out,  0, NE  * sizeof(float)));

    // ---- Run ple_block --------------------------------------------------------
    PleWeights w;
    w.conv_w = d_conv_w; w.norm_w = d_norm_w; w.gate_w = d_gate_w; w.proj_w = d_proj_w;
    PleOut po; po.out = d_out; po.gate = d_gate;
    ple_set_native_bf16(false); ple_set_native_postops(false);
    CUOK(ple_block(w, d_new_row, d_hist, d_hidden, po, d_scratch, 0));
    sync();

    // ---- Explicit magnitude guards (NOP → 0 after memset; real → non-zero) --------
    // gate_kernel:   gate ≈ 0.731 → must be > 0.1
    // bf16_gemv_kernel: out ≈ 10.9 → must be > 1.0

    // ---- Check gate[s] ≈ gate_ref for all streams ----------------------------
    auto gate_h = stoh(d_gate, NHC);
    for (int s = 0; s < NHC; ++s) CHECK(gate_h[s] > 0.1f);  // NOP → 0, real → 0.731
    for (int s = 0; s < NHC; ++s) {
        double diff = std::fabs((float)gate_h[s] - gate_ref);
        CHECK(diff < 0.05);
        if (diff >= 0.05)
            std::printf("  gate[%d]=%.5g ref=%.5g diff=%.5g\n",
                s, (float)gate_h[s], (float)gate_ref, (float)diff);
    }

    // ---- Check out[o] ≈ out_ref for all elements -----------------------------
    auto out_h = stoh(d_out, NE);
    CHECK(out_h[0] > 1.0f);  // NOP → 0, real → ~10.9
    int out_bad = 0;
    for (int o = 0; o < NE; ++o) {
        double diff = std::fabs((float)out_h[o] - out_ref);
        if (diff >= 0.5) {
            if (out_bad < 5)
                std::printf("  out[%d]=%.4g ref=%.4g diff=%.4g\n",
                    o, (float)out_h[o], (float)out_ref, (float)diff);
            ++out_bad;
        }
    }
    CHECK(out_bad == 0);
    std::printf("gate_ref=%.5g gpu=%.5g  out_ref=%.4g gpu[0]=%.4g\n",
        (float)gate_ref, (float)gate_h[0], (float)out_ref, (float)out_h[0]);

    // ---- ple_history_advance (history_advance_kernel) ------------------------
    CUOK(ple_history_advance(d_hist, d_new_row, 0));
    sync();
    auto hist2 = stoh(d_hist, (size_t)NH * HCD);
    // After advance, last row of channel 0: state_index(NH-1, 0) = NH-1
    CHECK(std::fabs(hist2[state_index(NH - 1, 0)] - 1.0f) < 1e-5f);

    std::printf("ok=%ld fail=%ld\n", g_ok, g_fail);
    if (g_fail == 0) std::printf("U41A GPU TESTS PASSED\n");
    return g_fail ? 1 : 0;
}
