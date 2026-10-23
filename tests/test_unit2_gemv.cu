// tests/test_unit2_gemv.cu — Smoke test for unit 2 GEMV CUDA kernels
//
// Tests:
//   1. ix_s_gemv (s_family_gemv) — structured GEMV GPU vs host reference
//   2. ix_s2_gemv_q8 (s2_gemv_q8_cuda) — S2 × Q8_0 GPU kernel
//   3. ix_s2_gemv_quads (s2_gemv_quads_kernel) — amortized quads GPU kernel
//   4. ix_bf16_gemv_split (bf16_gemv_kernels) — split BF16 GEMV GPU kernel
//   5. ix_mmvq_q6k (native_gguf_mmvq) — Q6_K MMVQ GPU kernel

#include "strata/kernels/cuda/ix_gemv_ext.h"
#include "strata/kernels/cuda/ix_kernels.h"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++tests_passed; } \
    else { ++tests_failed; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

// ─── Helpers ──────────────────────────────────────────────────

static float rand_f32() {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

// Check two float arrays are within tolerance
static bool approx_equal(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        float denom = fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (denom > 1e-6f) {
            if (diff / denom > tol) {
                fprintf(stderr, "  mismatch at [%d]: %.6f vs %.6f (rel=%.4e)\n",
                        i, a[i], b[i], diff / denom);
                return false;
            }
        } else if (diff > tol) {
            fprintf(stderr, "  mismatch at [%d]: %.6f vs %.6f (abs=%.4e)\n",
                    i, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════
// Test 1: ix_s_gemv — Structured GEMV GPU vs host reference
// ═══════════════════════════════════════════════════════════════

static void test_s_family_gemv() {
    printf("\n--- Test 1: ix_s_gemv (s_family_gemv) ---\n");

    const int n_in = 256;
    const int n_out = 8;
    const int code_bits = 4;
    const int group_elems = 32;

    // Generate random data
    std::vector<uint16_t> h_x(n_in);
    for (int i = 0; i < n_in; i++) {
        h_x[i] = strata::kernels::f16_from_f32(rand_f32());
    }

    // Codes: 4-bit, 2 per byte
    const int code_bytes = (n_in * code_bits + 7) / 8;
    std::vector<uint8_t> h_codes(n_out * code_bytes);
    for (auto& c : h_codes) c = rand() & 0xFF;

    // Scales: n_out × (n_in / group_elems)
    const int groups = n_in / group_elems;
    std::vector<float> h_scales(n_out * groups);
    for (auto& s : h_scales) s = rand_f32() * 0.1f;

    // Host reference
    strata::kernels::SForm form;
    form.code_bits = code_bits;
    form.code_bias = 0;
    form.group_elems = group_elems;
    form.codebook = strata::kernels::Codebook::Affine;
    form.has_offset = false;

    std::vector<float> h_y_ref(n_out);
    strata::kernels::s_gemv(h_x.data(), h_codes.data(), h_scales.data(),
                            nullptr, h_y_ref.data(), n_in, n_out, form);

    // GPU
    uint16_t* d_x; uint8_t* d_codes; float* d_scales; float* d_y;
    cudaMalloc(&d_x, n_in * sizeof(uint16_t));
    cudaMalloc(&d_codes, h_codes.size());
    cudaMalloc(&d_scales, h_scales.size() * sizeof(float));
    cudaMalloc(&d_y, n_out * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), n_in * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice);

    ix_s_gemv(d_x, d_codes, d_scales, nullptr, d_y,
              n_in, n_out,
              code_bits, /*code_bias=*/0, /*codebook_id=*/0,
              group_elems, /*has_offset=*/0,
              0);
    cudaDeviceSynchronize();

    std::vector<float> h_y_gpu(n_out);
    cudaMemcpy(h_y_gpu.data(), d_y, n_out * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = approx_equal(h_y_ref.data(), h_y_gpu.data(), n_out, 1e-3f);
    CHECK(ok, "ix_s_gemv matches host reference");

    cudaFree(d_x); cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_y);
}

// ═══════════════════════════════════════════════════════════════
// Test 2: ix_s2_gemv_q8 — S2 × Q8_0
// ═══════════════════════════════════════════════════════════════

static void test_s2_gemv_q8() {
    printf("\n--- Test 2: ix_s2_gemv_q8 (s2_gemv_q8_cuda) ---\n");

    const int n_in = 256;  // must be multiple of 32
    const int n_out = 4;
    const int code_bits = 2;
    const int group_elems = 64;
    const int code_bias = -1;

    // Q8_0 activation: 34 bytes per 32 elements
    const int n_blocks = n_in / 32;
    std::vector<uint8_t> h_x_q8(n_blocks * 34);
    // Also need FP16 version for host reference
    std::vector<uint16_t> h_x_fp16(n_in);

    for (int b = 0; b < n_blocks; b++) {
        float d = 0.01f + 0.005f * (rand() % 10);
        // Write d as fp16 using our host-safe converter
        uint16_t hd = strata::kernels::f16_from_f32(d);
        memcpy(h_x_q8.data() + b * 34, &hd, 2);
        int8_t* qs = (int8_t*)(h_x_q8.data() + b * 34 + 2);
        // Recover the actual fp16 value for reference computation
        float d_actual = strata::kernels::f32_from_f16(hd);
        for (int j = 0; j < 32; j++) {
            qs[j] = (int8_t)((rand() % 255) - 127);
            h_x_fp16[b * 32 + j] = strata::kernels::f16_from_f32(d_actual * (float)qs[j]);
        }
    }

    // 2-bit codes
    const int code_bytes = n_in / 4;
    std::vector<uint8_t> h_codes(n_out * code_bytes);
    for (auto& c : h_codes) c = rand() & 0xFF;

    // Scales
    const int groups = n_in / group_elems;
    std::vector<float> h_scales(n_out * groups);
    for (auto& s : h_scales) s = rand_f32() * 0.1f;

    // Host reference using s_gemv with FP16 activation
    strata::kernels::SForm form;
    form.code_bits = code_bits;
    form.code_bias = code_bias;
    form.group_elems = group_elems;
    form.codebook = strata::kernels::Codebook::Affine;
    form.has_offset = false;

    std::vector<float> h_y_ref(n_out);
    strata::kernels::s_gemv(h_x_fp16.data(), h_codes.data(), h_scales.data(),
                            nullptr, h_y_ref.data(), n_in, n_out, form);

    // GPU
    uint8_t* d_x; uint8_t* d_codes; float* d_scales; float* d_y;
    cudaMalloc(&d_x, h_x_q8.size());
    cudaMalloc(&d_codes, h_codes.size());
    cudaMalloc(&d_scales, h_scales.size() * sizeof(float));
    cudaMalloc(&d_y, n_out * sizeof(float));

    cudaMemcpy(d_x, h_x_q8.data(), h_x_q8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice);

    ix_s2_gemv_q8(d_x, d_codes, d_scales, (float*)d_y,
                  n_in, n_out, code_bias, group_elems, 0);
    cudaDeviceSynchronize();

    std::vector<float> h_y_gpu(n_out);
    cudaMemcpy(h_y_gpu.data(), d_y, n_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Tolerance is larger because Q8_0 dequant introduces rounding
    bool ok = approx_equal(h_y_ref.data(), h_y_gpu.data(), n_out, 5e-2f);
    CHECK(ok, "ix_s2_gemv_q8 approximately matches host reference");

    cudaFree(d_x); cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_y);
}

// ═══════════════════════════════════════════════════════════════
// Test 3: ix_s2_gemv_quads — Amortized quads
// ═══════════════════════════════════════════════════════════════

static void test_s2_gemv_quads() {
    printf("\n--- Test 3: ix_s2_gemv_quads (s2_gemv_quads_kernel) ---\n");

    const int n_in = 256;
    const int n_out = 4;
    const int code_bits = 2;
    const int group_elems = 64;
    const int code_bias = -1;

    std::vector<uint16_t> h_x(n_in);
    for (int i = 0; i < n_in; i++) {
        h_x[i] = strata::kernels::f16_from_f32(rand_f32());
    }

    const int code_bytes = n_in / 4;
    std::vector<uint8_t> h_codes(n_out * code_bytes);
    for (auto& c : h_codes) c = rand() & 0xFF;

    const int groups = n_in / group_elems;
    std::vector<float> h_scales(n_out * groups);
    for (auto& s : h_scales) s = rand_f32() * 0.1f;

    // Host reference
    strata::kernels::SForm form;
    form.code_bits = code_bits;
    form.code_bias = code_bias;
    form.group_elems = group_elems;
    form.codebook = strata::kernels::Codebook::Affine;
    form.has_offset = false;

    std::vector<float> h_y_ref(n_out);
    strata::kernels::s_gemv(h_x.data(), h_codes.data(), h_scales.data(),
                            nullptr, h_y_ref.data(), n_in, n_out, form);

    // GPU
    uint16_t* d_x; uint8_t* d_codes; float* d_scales; float* d_y;
    cudaMalloc(&d_x, n_in * sizeof(uint16_t));
    cudaMalloc(&d_codes, h_codes.size());
    cudaMalloc(&d_scales, h_scales.size() * sizeof(float));
    cudaMalloc(&d_y, n_out * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), n_in * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice);

    ix_s2_gemv_quads(d_x, d_codes, d_scales, (float*)d_y,
                     n_in, n_out, code_bias, group_elems,
                     /*threads_per_row=*/64, 0);
    cudaDeviceSynchronize();

    std::vector<float> h_y_gpu(n_out);
    cudaMemcpy(h_y_gpu.data(), d_y, n_out * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = approx_equal(h_y_ref.data(), h_y_gpu.data(), n_out, 1e-3f);
    CHECK(ok, "ix_s2_gemv_quads matches host reference");

    cudaFree(d_x); cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_y);
}

// ═══════════════════════════════════════════════════════════════
// Test 4: ix_bf16_gemv_split — BF16 GEMV with split
// ═══════════════════════════════════════════════════════════════

static void test_bf16_gemv_split() {
    printf("\n--- Test 4: ix_bf16_gemv_split (bf16_gemv_kernels) ---\n");

    const int n_in = 256;
    const int n_out = 4;

    // Generate BF16 data
    std::vector<uint16_t> h_x(n_in);
    std::vector<uint16_t> h_w(n_out * n_in);

    for (int i = 0; i < n_in; i++) {
        float v = rand_f32();
        uint32_t bits;
        memcpy(&bits, &v, 4);
        h_x[i] = (uint16_t)(bits >> 16);  // FP32 → BF16 truncation
    }
    for (int i = 0; i < n_out * n_in; i++) {
        float v = rand_f32() * 0.1f;
        uint32_t bits;
        memcpy(&bits, &v, 4);
        h_w[i] = (uint16_t)(bits >> 16);
    }

    // Host reference: manual BF16 → FP32 → dot
    std::vector<float> h_y_ref(n_out);
    for (int o = 0; o < n_out; o++) {
        float dot = 0.0f;
        for (int i = 0; i < n_in; i++) {
            uint32_t xb = (uint32_t)h_x[i] << 16;
            uint32_t wb = (uint32_t)h_w[o * n_in + i] << 16;
            float xf, wf;
            memcpy(&xf, &xb, 4);
            memcpy(&wf, &wb, 4);
            dot += xf * wf;
        }
        h_y_ref[o] = dot;
    }

    // GPU
    uint16_t* d_x; uint16_t* d_w; float* d_y;
    cudaMalloc(&d_x, n_in * sizeof(uint16_t));
    cudaMalloc(&d_w, n_out * n_in * sizeof(uint16_t));
    cudaMalloc(&d_y, n_out * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), n_in * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), n_out * n_in * sizeof(uint16_t), cudaMemcpyHostToDevice);

    ix_bf16_gemv_split(d_x, d_w, d_y, n_in, n_out,
                       /*threads_per_row=*/128, 0);
    cudaDeviceSynchronize();

    std::vector<float> h_y_gpu(n_out);
    cudaMemcpy(h_y_gpu.data(), d_y, n_out * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = approx_equal(h_y_ref.data(), h_y_gpu.data(), n_out, 1e-3f);
    CHECK(ok, "ix_bf16_gemv_split matches host reference");

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_y);
}

// ═══════════════════════════════════════════════════════════════
// Test 5: ix_mmvq_q6k — Q6_K MMVQ
// ═══════════════════════════════════════════════════════════════

static void test_mmvq_q6k() {
    printf("\n--- Test 5: ix_mmvq_q6k (native_gguf_mmvq) ---\n");

    // Q6_K: 210 bytes per 256 elements
    // Just test that the kernel runs without crashing (no host reference for Q6_K)
    const int n_in = 256;
    const int n_out = 2;

    // Q6_K weight blocks
    const int n_sb = n_in / 256;
    std::vector<uint8_t> h_w(n_out * n_sb * 210, 0);
    // Set d (last 2 bytes of Q6_K_Block) to a small fp16
    for (int o = 0; o < n_out; o++) {
        for (int s = 0; s < n_sb; s++) {
            uint8_t* blk = h_w.data() + (o * n_sb + s) * 210;
            // ql[128] = all zeros (codes = 0)
            // qh[64]  = all zeros
            // scales[16] = small values
            for (int i = 0; i < 16; i++) blk[128 + 64 + i] = 1;
            // d at offset 208 (half) — use host-safe conversion
            uint16_t hd = strata::kernels::f16_from_f32(0.01f);
            memcpy(blk + 208, &hd, 2);
        }
    }

    // Q8_1 activation
    const int n_q8_blocks = n_in / 32;
    std::vector<uint8_t> h_act(n_q8_blocks * 36, 0);
    for (int b = 0; b < n_q8_blocks; b++) {
        uint8_t* blk = h_act.data() + b * 36;
        // ds: half2 = {d, sum} packed as 4 bytes
        // half2 layout: low half = d, high half = sum
        uint16_t hd = strata::kernels::f16_from_f32(0.01f);
        uint16_t hs = strata::kernels::f16_from_f32(0.0f);
        memcpy(blk, &hd, 2);
        memcpy(blk + 2, &hs, 2);
        // qs: small values
        int8_t* qs = (int8_t*)(blk + 4);
        for (int j = 0; j < 32; j++) qs[j] = (int8_t)(j - 16);
    }

    void* d_w; void* d_act; float* d_y;
    cudaMalloc(&d_w, h_w.size());
    cudaMalloc(&d_act, h_act.size());
    cudaMalloc(&d_y, n_out * sizeof(float));

    cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_act, h_act.data(), h_act.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, n_out * sizeof(float));

    ix_mmvq_q6k(d_w, d_act, (float*)d_y, n_in, n_out, 0);
    cudaError_t err = cudaDeviceSynchronize();

    CHECK(err == cudaSuccess, "ix_mmvq_q6k runs without CUDA error");

    std::vector<float> h_y(n_out);
    cudaMemcpy(h_y.data(), d_y, n_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Just check the output is finite
    bool finite = true;
    for (int i = 0; i < n_out; i++) {
        if (!isfinite(h_y[i])) { finite = false; break; }
    }
    CHECK(finite, "ix_mmvq_q6k output is finite");

    printf("  q6k output: [%.6f, %.6f]\n", h_y[0], h_y[1]);

    cudaFree(d_w); cudaFree(d_act); cudaFree(d_y);
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main() {
    printf("=== strata unit 2 GEMV smoke test ===\n");

    // Check CUDA device
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        printf("No CUDA devices found, skipping GPU tests.\n");
        printf("\n=== Results: 0 passed, 0 failed (skipped) ===\n");
        return 0;
    }
    printf("Found %d CUDA device(s)\n", dev_count);
    cudaSetDevice(0);

    srand(42);

    test_s_family_gemv();
    test_s2_gemv_q8();
    test_s2_gemv_quads();
    test_bf16_gemv_split();
    test_mmvq_q6k();

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}

