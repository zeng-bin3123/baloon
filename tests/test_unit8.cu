// test_unit8.cu — Unit 8 tests: f16 conversions, activation quantizer,
//                  elementwise kernels, elementwise ops, native cuda ops
//
// Tests GPU kernels for type conversions, quantization, and elementwise ops.
// Compile: clang++ -x cuda --cuda-gpu-arch=ivcore10 ...
// Run: CUDA_VISIBLE_DEVICES=0 ./test_unit8

#include "strata/kernels/cuda/ix_kernels.h"
#include "strata/kernels/cuda/ix_native_ops.h"
#include "strata/kernels/cuda/ix_warp64.cuh"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s at %s:%d\n", msg, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ─── Test 1: F32 → F16 → F32 roundtrip ─────────────────────────────
static void test_f32_f16_roundtrip() {
    printf("  test_f32_f16_roundtrip... ");

    const int n = 1024;
    std::vector<float> h_src(n);
    for (int i = 0; i < n; i++) {
        h_src[i] = (float)(i - 512) * 0.01f;  // range [-5.12, 5.11]
    }

    float* d_src;
    uint16_t* d_f16;
    float* d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_f16, n * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_dst, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // F32 → F16
    ix_cvt_f32_to_f16(d_src, d_f16, n, stream);
    // F16 → F32
    ix_cvt_f16_to_f32(d_f16, d_dst, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_dst(n);
    CHECK_CUDA(cudaMemcpy(h_dst.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++) {
        // FP16 has ~3 digits of precision; allow relative error
        float expected = h_src[i];
        float actual = h_dst[i];
        float err = fabsf(actual - expected);
        float threshold = fabsf(expected) * 0.002f + 0.001f;  // relative + absolute tolerance
        if (err > threshold) {
            printf("idx %d: expected %.6f, got %.6f (err=%.6f)\n", i, expected, actual, err);
            ok = false;
            break;
        }
    }

    cudaFree(d_src);
    cudaFree(d_f16);
    cudaFree(d_dst);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "F32→F16→F32 roundtrip error too large");
    printf("OK\n");
}

// ─── Test 2: BF16 → F32 conversion ─────────────────────────────────
static void test_bf16_f32_conversion() {
    printf("  test_bf16_f32_conversion... ");

    const int n = 256;
    // Create BF16 values: top 16 bits of known floats
    std::vector<float> h_expected(n);
    std::vector<uint16_t> h_bf16(n);
    for (int i = 0; i < n; i++) {
        float val = (float)(i - 128) * 0.5f;
        h_expected[i] = val;
        uint32_t bits;
        memcpy(&bits, &val, sizeof(float));
        h_bf16[i] = (uint16_t)(bits >> 16);
        // Re-derive expected from truncated BF16
        uint32_t bf16_expanded = (uint32_t)h_bf16[i] << 16;
        memcpy(&h_expected[i], &bf16_expanded, sizeof(float));
    }

    uint16_t* d_bf16;
    float* d_f32;
    CHECK_CUDA(cudaMalloc(&d_bf16, n * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_f32, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_bf16, h_bf16.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ix_cvt_bf16_to_f32(d_bf16, d_f32, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_result(n);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_f32, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (h_result[i] != h_expected[i]) {
            printf("idx %d: expected %f, got %f\n", i, h_expected[i], h_result[i]);
            ok = false;
            break;
        }
    }

    cudaFree(d_bf16);
    cudaFree(d_f32);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "BF16→F32 conversion mismatch");
    printf("OK\n");
}

// ─── Test 3: FP16 → BF16 conversion ────────────────────────────────
static void test_f16_bf16_conversion() {
    printf("  test_f16_bf16_conversion... ");

    const int n = 128;
    // Create FP16 values, convert to BF16, check via F32 intermediary
    std::vector<float> h_vals(n);
    std::vector<uint16_t> h_f16(n);
    for (int i = 0; i < n; i++) {
        h_vals[i] = (float)(i - 64) * 0.1f;
        h_f16[i] = strata::kernels::f16_from_f32(h_vals[i]);
    }

    uint16_t* d_f16;
    uint16_t* d_bf16;
    CHECK_CUDA(cudaMalloc(&d_f16, n * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_bf16, n * sizeof(uint16_t)));
    CHECK_CUDA(cudaMemcpy(d_f16, h_f16.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ix_cvt_f16_to_bf16(d_f16, d_bf16, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<uint16_t> h_bf16_result(n);
    CHECK_CUDA(cudaMemcpy(h_bf16_result.data(), d_bf16, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // Verify: convert BF16 back to F32 and compare with original F16→F32
    bool ok = true;
    for (int i = 0; i < n; i++) {
        float f16_as_f32 = strata::kernels::f32_from_f16(h_f16[i]);
        uint32_t bf16_expanded = (uint32_t)h_bf16_result[i] << 16;
        float bf16_as_f32;
        memcpy(&bf16_as_f32, &bf16_expanded, sizeof(float));

        // BF16 has less precision (7 mantissa bits vs 10 for FP16)
        // but the range is larger, so small values may lose precision
        float err = fabsf(bf16_as_f32 - f16_as_f32);
        float threshold = fabsf(f16_as_f32) * 0.02f + 0.01f;
        if (err > threshold) {
            printf("idx %d: f16_f32=%.6f, bf16_f32=%.6f, err=%.6f\n",
                   i, f16_as_f32, bf16_as_f32, err);
            ok = false;
            break;
        }
    }

    cudaFree(d_f16);
    cudaFree(d_bf16);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "F16→BF16 conversion error too large");
    printf("OK\n");
}

// ─── Test 4: Activation Q8_1 quantize → dequant roundtrip ──────────
static void test_q8_1_roundtrip() {
    printf("  test_q8_1_roundtrip... ");

    const int n = 256;  // must be multiple of 32
    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(i - 128) * 0.01f;
    }

    float* d_input;
    void* d_q8;
    float* d_output;
    // Q8_1 block: 36 bytes per 32 elements
    const int n_blocks = n / 32;
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q8, n_blocks * 36));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Quantize
    ix_activation_quantize_q8_1(d_input, d_q8, n, stream);
    // Dequantize
    ix_activation_dequant_q8_1(d_q8, d_output, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_output(n);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Q8_1 quantization has ~1/127 relative error per block
    bool ok = true;
    for (int i = 0; i < n; i++) {
        float err = fabsf(h_output[i] - h_input[i]);
        float amax = 0.0f;
        int block_start = (i / 32) * 32;
        for (int j = block_start; j < block_start + 32 && j < n; j++) {
            float a = fabsf(h_input[j]);
            if (a > amax) amax = a;
        }
        float threshold = amax / 127.0f * 1.5f + 0.001f;
        if (err > threshold) {
            printf("idx %d: expected %.6f, got %.6f (err=%.6f, thresh=%.6f)\n",
                   i, h_input[i], h_output[i], err, threshold);
            ok = false;
            break;
        }
    }

    cudaFree(d_input);
    cudaFree(d_q8);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "Q8_1 roundtrip error too large");
    printf("OK\n");
}

// ─── Test 5: SiLU in-place on GPU ──────────────────────────────────
static void test_silu_inplace() {
    printf("  test_silu_inplace... ");

    const int n = 512;
    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(i - 256) * 0.02f;
    }

    float* d_x;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ix_ew_silu_inplace(d_x, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_output(n);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++) {
        float x = h_input[i];
        float expected = x / (1.0f + expf(-x));
        float err = fabsf(h_output[i] - expected);
        if (err > 0.001f) {
            printf("idx %d: expected %.6f, got %.6f\n", i, expected, h_output[i]);
            ok = false;
            break;
        }
    }

    cudaFree(d_x);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "SiLU inplace mismatch");
    printf("OK\n");
}

// ─── Test 6: Scale in-place on GPU ─────────────────────────────────
static void test_scale_inplace() {
    printf("  test_scale_inplace... ");

    const int n = 256;
    std::vector<float> h_input(n, 2.0f);
    const float scale = 0.5f;

    float* d_x;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ix_ew_scale_inplace(d_x, n, scale, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_output(n);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_output[i] - 1.0f) > 0.001f) {
            printf("idx %d: expected 1.0, got %f\n", i, h_output[i]);
            ok = false;
            break;
        }
    }

    cudaFree(d_x);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "Scale inplace mismatch");
    printf("OK\n");
}

// ─── Test 7: Add in-place on GPU ───────────────────────────────────
static void test_add_inplace() {
    printf("  test_add_inplace... ");

    const int n = 256;
    std::vector<float> h_dst(n, 3.0f);
    std::vector<float> h_src(n, 7.0f);

    float *d_dst, *d_src;
    CHECK_CUDA(cudaMalloc(&d_dst, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_src, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_dst, h_dst.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ix_ew_add_inplace(d_dst, d_src, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_result(n);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_result[i] - 10.0f) > 0.001f) {
            printf("idx %d: expected 10.0, got %f\n", i, h_result[i]);
            ok = false;
            break;
        }
    }

    cudaFree(d_dst);
    cudaFree(d_src);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "Add inplace mismatch");
    printf("OK\n");
}

// ─── Test 8: RMSNorm weighted on GPU ───────────────────────────────
static void test_rms_norm_weighted() {
    printf("  test_rms_norm_weighted... ");

    const int rows = 2;
    const int cols = 128;
    const float eps = 1e-5f;

    // x = all 1.0, w = all 1.0 → after rms_norm, output should be 1.0
    std::vector<float> h_x(rows * cols, 1.0f);
    std::vector<float> h_w(cols, 1.0f);

    float *d_x, *d_w;
    CHECK_CUDA(cudaMalloc(&d_x, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ix_ew_rms_norm_weighted(d_x, d_w, rows, cols, eps, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_result(rows * cols);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_x, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    // RMS of all-ones: sqrt(mean(1²) + eps) ≈ 1.0, so output ≈ 1.0 * 1.0 / 1.0 = 1.0
    bool ok = true;
    for (int i = 0; i < rows * cols; i++) {
        if (fabsf(h_result[i] - 1.0f) > 0.01f) {
            printf("idx %d: expected ~1.0, got %f\n", i, h_result[i]);
            ok = false;
            break;
        }
    }

    cudaFree(d_x);
    cudaFree(d_w);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "RMSNorm weighted mismatch");
    printf("OK\n");
}

// ─── Test 9: Embedding gather on GPU ───────────────────────────────
static void test_embedding_gather() {
    printf("  test_embedding_gather... ");

    const int n = 64;
    const int code_bits = 8;
    const int code_bias = 0;
    const int group_elems = 16;

    // Create codes: sequential 0, 1, 2, ...
    std::vector<uint8_t> h_codes(n);
    for (int i = 0; i < n; i++) h_codes[i] = (uint8_t)(i % 8);

    // Scales: 2.0 for all groups
    const int n_groups = (n + group_elems - 1) / group_elems;
    std::vector<float> h_scales(n_groups, 2.0f);
    std::vector<float> h_offsets(n_groups, 1.0f);

    uint8_t* d_codes;
    float *d_scales, *d_offsets, *d_out;
    CHECK_CUDA(cudaMalloc(&d_codes, n));
    CHECK_CUDA(cudaMalloc(&d_scales, n_groups * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_offsets, n_groups * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_codes, h_codes.data(), n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scales, h_scales.data(), n_groups * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets.data(), n_groups * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ix_ew_embedding_gather(d_codes, d_scales, d_offsets, n, code_bits,
                           code_bias, group_elems, d_out, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_result(n);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++) {
        float expected = (float)(i % 8) * 2.0f + 1.0f;
        if (fabsf(h_result[i] - expected) > 0.001f) {
            printf("idx %d: expected %.1f, got %f\n", i, expected, h_result[i]);
            ok = false;
            break;
        }
    }

    cudaFree(d_codes);
    cudaFree(d_scales);
    cudaFree(d_offsets);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "Embedding gather mismatch");
    printf("OK\n");
}

// ─── Test 10: Fused SiLU + Q8_1 quantize ──────────────────────────
static void test_silu_quantize_q8_1() {
    printf("  test_silu_quantize_q8_1... ");

    const int n = 128;  // multiple of 32
    std::vector<float> h_gate(n), h_up(n);
    for (int i = 0; i < n; i++) {
        h_gate[i] = (float)(i - 64) * 0.05f;
        h_up[i] = 1.0f;
    }

    float *d_gate, *d_up;
    void* d_q8;
    float* d_deq;
    const int n_blocks = n / 32;
    CHECK_CUDA(cudaMalloc(&d_gate, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_up, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_q8, n_blocks * 36));
    CHECK_CUDA(cudaMalloc(&d_deq, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_gate, h_gate.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_up, h_up.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ix_silu_quantize_q8_1(d_gate, d_up, d_q8, n, stream);
    ix_activation_dequant_q8_1(d_q8, d_deq, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_result(n);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_deq, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: silu(gate) * up ≈ result (with quantization error)
    bool ok = true;
    for (int i = 0; i < n; i++) {
        float g = h_gate[i];
        float expected = g / (1.0f + expf(-g)) * h_up[i];
        float err = fabsf(h_result[i] - expected);
        // Quantization error scales with block max
        float threshold = 0.1f + fabsf(expected) * 0.05f;
        if (err > threshold) {
            printf("idx %d: expected %.6f, got %.6f (err=%.6f)\n",
                   i, expected, h_result[i], err);
            ok = false;
            break;
        }
    }

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_q8);
    cudaFree(d_deq);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "Fused SiLU+Q8_1 error too large");
    printf("OK\n");
}

// ─── Test 11: Doorbell ring (GPU atomic increment) ─────────────────
static void test_doorbell_ring() {
    printf("  test_doorbell_ring... ");

    uint32_t* d_seq;
    CHECK_CUDA(cudaMalloc(&d_seq, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_seq, 0, sizeof(uint32_t)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Ring 5 times
    for (int i = 0; i < 5; i++) {
        ix_ew_doorbell_ring(d_seq, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    uint32_t h_seq;
    CHECK_CUDA(cudaMemcpy(&h_seq, d_seq, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_seq);
    cudaStreamDestroy(stream);

    TEST_ASSERT(h_seq == 5, "Doorbell ring count mismatch");
    printf("OK (seq=%u)\n", h_seq);
}

// ─── Test 12: Copy from mapped ─────────────────────────────────────
static void test_copy_from_mapped() {
    printf("  test_copy_from_mapped... ");

    const int n = 256;
    std::vector<float> h_src(n);
    for (int i = 0; i < n; i++) h_src[i] = (float)i;

    float *d_src, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_src, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dst, 0, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    ix_ew_copy_from_mapped(d_dst, d_src, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_result(n);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (h_result[i] != h_src[i]) {
            printf("idx %d: expected %f, got %f\n", i, h_src[i], h_result[i]);
            ok = false;
            break;
        }
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "Copy from mapped mismatch");
    printf("OK\n");
}

// ─── Main ─────────────────────────────────────────────────────────
int main() {
    printf("=== infccl Unit 8: Native CUDA Ops Tests (BI-V100) ===\n\n");

    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    printf("GPUs detected: %d\n", n_gpus);
    if (n_gpus > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("  GPU 0: %s (SM %d.%d, %.1f GB)\n\n",
               prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / 1e9);
    }

    cudaSetDevice(0);

    printf("Correctness tests:\n");
    test_f32_f16_roundtrip();
    test_bf16_f32_conversion();
    test_f16_bf16_conversion();
    test_q8_1_roundtrip();
    test_silu_inplace();
    test_scale_inplace();
    test_add_inplace();
    test_rms_norm_weighted();
    test_embedding_gather();
    test_silu_quantize_q8_1();
    test_doorbell_ring();
    test_copy_from_mapped();

    printf("\n=== ALL UNIT 8 TESTS PASSED ===\n");
    return 0;
}

