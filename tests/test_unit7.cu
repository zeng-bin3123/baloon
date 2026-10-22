// test_unit7.cu — Integration tests for BI-V100 CUDA kernels
//
// Tests all warp64 kernels on actual GPU hardware.
// Compile: clang++ -x cuda --cuda-gpu-arch=ivcore10 ...
// Run: CUDA_VISIBLE_DEVICES=0,1,2,3 ./test_unit7

#include "strata/kernels/cuda/ix_kernels.h"
#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <chrono>

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

// ─── Test 1: warp64 properties ────────────────────────────────────
__global__ void test_warp_size_kernel(int* result) {
    if (threadIdx.x == 0) {
        result[0] = warpSize;
    }
}

static void test_warp_size() {
    printf("  test_warp_size... ");
    int* d_result;
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));

    test_warp_size_kernel<<<1, 64>>>(d_result);
    CHECK_CUDA(cudaDeviceSynchronize());

    int h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_result);

    // BI-V100 should report warpSize = 64
    // (If running on NVIDIA for testing, it'll be 32 — that's OK for CI)
    printf("warpSize=%d %s\n", h_result,
           (h_result == 64) ? "OK (BI-V100)" : "(NVIDIA)");
}

// ─── Test 2: warp64 shuffle/reduce ────────────────────────────────
__global__ void test_warp_reduce_kernel(float* result) {
    float val = (float)(threadIdx.x + 1);
    float sum = ix_warp_sum(val);
    float mx = ix_warp_max(val);

    if (threadIdx.x == 0) {
        result[0] = sum;
        result[1] = mx;
    }
}

static void test_warp_reduce() {
    printf("  test_warp_reduce... ");
    float* d_result;
    CHECK_CUDA(cudaMalloc(&d_result, 2 * sizeof(float)));

    test_warp_reduce_kernel<<<1, IX_WARP_SIZE>>>(d_result);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_result[2];
    CHECK_CUDA(cudaMemcpy(h_result, d_result, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_result);

    // Sum of 1..64 = 64*65/2 = 2080
    float expected_sum = (float)(IX_WARP_SIZE * (IX_WARP_SIZE + 1) / 2);
    float expected_max = (float)IX_WARP_SIZE;

    TEST_ASSERT(fabsf(h_result[0] - expected_sum) < 1.0f,
                "warp sum incorrect");
    TEST_ASSERT(fabsf(h_result[1] - expected_max) < 0.01f,
                "warp max incorrect");
    printf("OK (sum=%.0f, max=%.0f)\n", h_result[0], h_result[1]);
}

// ─── Test 3: Q8_0 MMVQ correctness ───────────────────────────────
// Create a simple identity-like weight matrix and verify GEMV output.
static void test_mmvq_q8_0() {
    printf("  test_mmvq_q8_0... ");

    const int n_in = 256;   // must be multiple of 32
    const int n_out = 4;
    const int n_blocks = n_in / 32;

    // Create Q8_0 weights on host: identity-like (diagonal blocks)
    // Q8_0 block: {d:fp16, qs:int8[32]} = 34 bytes
    struct Q8_0_Block { uint16_t d; int8_t qs[32]; };
    std::vector<Q8_0_Block> h_weights(n_out * n_blocks);
    memset(h_weights.data(), 0, h_weights.size() * sizeof(Q8_0_Block));

    // Set up simple weights: row r has qs[r%32] = 127 in block r/32
    for (int r = 0; r < n_out; r++) {
        for (int b = 0; b < n_blocks; b++) {
            auto& blk = h_weights[r * n_blocks + b];
            // d = 1.0 in fp16
            half d_half = __float2half(1.0f / 127.0f);
            memcpy(&blk.d, &d_half, sizeof(uint16_t));
            // Set one element to 127
            if (b == 0) blk.qs[r] = 127;
        }
    }

    // Create Q8_1 activation on host
    // Q8_1 block: {ds:half2, qs:int8[32]} = 36 bytes
    struct Q8_1_Block { uint32_t ds; int8_t qs[32]; };
    std::vector<Q8_1_Block> h_act(n_blocks);
    memset(h_act.data(), 0, h_act.size() * sizeof(Q8_1_Block));

    // Activation: all 1.0 → d=1/127, qs[i]=127
    for (int b = 0; b < n_blocks; b++) {
        half d_half = __float2half(1.0f / 127.0f);
        half s_half = __float2half(32.0f);  // sum of 32 elements
        half2 ds_val = __halves2half2(d_half, s_half);
        memcpy(&h_act[b].ds, &ds_val, sizeof(uint32_t));
        for (int j = 0; j < 32; j++) h_act[b].qs[j] = 127;
    }

    // Allocate device memory
    void *d_weights, *d_act;
    float* d_output;
    CHECK_CUDA(cudaMalloc(&d_weights, h_weights.size() * sizeof(Q8_0_Block)));
    CHECK_CUDA(cudaMalloc(&d_act, h_act.size() * sizeof(Q8_1_Block)));
    CHECK_CUDA(cudaMalloc(&d_output, n_out * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(),
                           h_weights.size() * sizeof(Q8_0_Block),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_act, h_act.data(),
                           h_act.size() * sizeof(Q8_1_Block),
                           cudaMemcpyHostToDevice));

    // Run kernel
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    ix_mmvq_q8_0(d_weights, d_act, d_output, n_in, n_out, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Read back
    std::vector<float> h_output(n_out);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output,
                           n_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: each output should be approximately 1.0
    // (one element 127 × 127 × (1/127)² = 1.0)
    bool ok = true;
    for (int r = 0; r < n_out; r++) {
        if (fabsf(h_output[r] - 1.0f) > 0.1f) {
            printf("row %d: expected ~1.0, got %f\n", r, h_output[r]);
            ok = false;
        }
    }

    cudaFree(d_weights);
    cudaFree(d_act);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "Q8_0 MMVQ output mismatch");
    printf("OK\n");
}

// ─── Test 4: Router top-k ─────────────────────────────────────────
static void test_router_top_k() {
    printf("  test_router_top_k... ");

    const int n_expert = 256;
    const int k = 8;
    const int n_tokens = 1;

    // Create logits: expert 42 has highest, expert 100 second, etc.
    std::vector<float> h_logits(n_expert, 0.0f);
    for (int e = 0; e < n_expert; e++) h_logits[e] = -(float)e;
    // Make specific experts high
    h_logits[42] = 10.0f;
    h_logits[100] = 9.0f;
    h_logits[7] = 8.0f;
    h_logits[200] = 7.0f;
    h_logits[15] = 6.0f;
    h_logits[128] = 5.0f;
    h_logits[64] = 4.0f;
    h_logits[33] = 3.0f;

    float* d_logits;
    int* d_ids;
    float* d_weights;
    CHECK_CUDA(cudaMalloc(&d_logits, n_expert * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ids, k * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_weights, k * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_logits, h_logits.data(),
                           n_expert * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    ix_router_top_k(d_logits, n_tokens, n_expert, k, d_ids, d_weights, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<int> h_ids(k);
    std::vector<float> h_weights(k);
    CHECK_CUDA(cudaMemcpy(h_ids.data(), d_ids, k * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_weights.data(), d_weights, k * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify top-8 are the correct experts (sorted by probability descending)
    int expected[] = {42, 100, 7, 200, 15, 128, 64, 33};
    bool ok = true;
    for (int j = 0; j < k; j++) {
        if (h_ids[j] != expected[j]) {
            printf("rank %d: expected expert %d, got %d\n", j, expected[j], h_ids[j]);
            ok = false;
        }
    }

    // Verify weights are positive and sum to ~1.0
    float wsum = 0.0f;
    for (int j = 0; j < k; j++) wsum += h_weights[j];
    if (fabsf(wsum - 1.0f) > 0.01f) {
        printf("weight sum: expected ~1.0, got %f\n", wsum);
        ok = false;
    }

    cudaFree(d_logits);
    cudaFree(d_ids);
    cudaFree(d_weights);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "Router top-k incorrect");
    printf("OK (top-8 correct, wsum=%.4f)\n", wsum);
}

// ─── Test 5: BF16 GEMV ───────────────────────────────────────────
static void test_bf16_gemv() {
    printf("  test_bf16_gemv... ");

    const int n_in = 128;
    const int n_out = 4;

    // BF16 activation: all 1.0 (BF16 representation of 1.0 = 0x3F80)
    std::vector<uint16_t> h_x(n_in, 0x3F80);

    // BF16 weight: row r has element r = 2.0, rest = 0.0
    // BF16 representation of 2.0 = 0x4000
    std::vector<uint16_t> h_w(n_out * n_in, 0);
    for (int r = 0; r < n_out; r++) {
        h_w[r * n_in + r] = 0x4000;  // 2.0
    }

    uint16_t *d_x, *d_w;
    float* d_y;
    CHECK_CUDA(cudaMalloc(&d_x, n_in * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_w, n_out * n_in * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_y, n_out * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), n_in * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), n_out * n_in * sizeof(uint16_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    ix_bf16_gemv(d_x, d_w, d_y, n_in, n_out, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_y(n_out);
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, n_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Each output should be 2.0 (one weight = 2.0, activation = 1.0)
    bool ok = true;
    for (int r = 0; r < n_out; r++) {
        if (fabsf(h_y[r] - 2.0f) > 0.01f) {
            printf("row %d: expected 2.0, got %f\n", r, h_y[r]);
            ok = false;
        }
    }

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "BF16 GEMV output mismatch");
    printf("OK\n");
}

// ─── Test 6: RMSNorm ─────────────────────────────────────────────
static void test_rms_norm() {
    printf("  test_rms_norm... ");

    const int n = 256;
    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_w(n, 1.0f);  // identity weight

    float *d_x, *d_w, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    ix_rms_norm(d_x, d_w, d_y, 1e-5f, n, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_y(n);
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    // RMS of all-ones: sqrt(1.0) = 1.0, so output = 1.0 * 1.0 / 1.0 = 1.0
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_y[i] - 1.0f) > 0.01f) {
            printf("elem %d: expected ~1.0, got %f\n", i, h_y[i]);
            ok = false;
            break;
        }
    }

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "RMSNorm output mismatch");
    printf("OK\n");
}

// ─── Test 7: MoE combine ─────────────────────────────────────────
static void test_moe_combine() {
    printf("  test_moe_combine... ");

    const int n_embd = 256;
    const int k = 8;

    // shared = [1, 1, ..., 1]
    // parts[j] = [j+1, j+1, ..., j+1]
    // w[j] = 1/k
    // expected: 1 + sum_{j=0}^{k-1} (1/k * (j+1)) = 1 + (1/k * k*(k+1)/2) = 1 + (k+1)/2

    std::vector<float> h_shared(n_embd, 1.0f);
    std::vector<float> h_parts(k * n_embd);
    std::vector<float> h_w(k, 1.0f / k);

    for (int j = 0; j < k; j++) {
        for (int i = 0; i < n_embd; i++) {
            h_parts[j * n_embd + i] = (float)(j + 1);
        }
    }

    float *d_shared, *d_parts, *d_w, *d_out;
    CHECK_CUDA(cudaMalloc(&d_shared, n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_parts, k * n_embd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, n_embd * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_shared, h_shared.data(), n_embd * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_parts, h_parts.data(), k * n_embd * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), k * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    ix_moe_combine(d_shared, d_parts, d_w, d_out, n_embd, k, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<float> h_out(n_embd);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n_embd * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = 1.0f + (float)(k + 1) / 2.0f;  // 1 + 4.5 = 5.5
    bool ok = true;
    for (int i = 0; i < n_embd; i++) {
        if (fabsf(h_out[i] - expected) > 0.01f) {
            printf("elem %d: expected %.1f, got %f\n", i, expected, h_out[i]);
            ok = false;
            break;
        }
    }

    cudaFree(d_shared);
    cudaFree(d_parts);
    cudaFree(d_w);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    TEST_ASSERT(ok, "MoE combine output mismatch");
    printf("OK (expected=%.1f)\n", expected);
}

// ─── Test 8: infccl AllReduce (multi-GPU) ─────────────────────────
static void test_infccl_allreduce() {
    int n_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&n_gpus));

    if (n_gpus < 2) {
        printf("  test_infccl_allreduce... SKIP (need >= 2 GPUs, have %d)\n", n_gpus);
        return;
    }

    printf("  test_infccl_allreduce (%d GPUs)... ", n_gpus);
    if (n_gpus > 4) n_gpus = 4;  // cap at 4

    const int64_t n = 1024;
    std::vector<int> device_ids(n_gpus);
    for (int i = 0; i < n_gpus; i++) device_ids[i] = i;

    int rc = infccl_init(n_gpus, device_ids.data(), n);
    TEST_ASSERT(rc == 0, "infccl_init failed");

    // Allocate per-GPU buffers
    std::vector<float*> d_bufs(n_gpus);
    for (int r = 0; r < n_gpus; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        CHECK_CUDA(cudaMalloc(&d_bufs[r], n * sizeof(float)));

        // GPU r: buf[i] = r + 1
        std::vector<float> h_buf(n, (float)(r + 1));
        CHECK_CUDA(cudaMemcpy(d_bufs[r], h_buf.data(),
                               n * sizeof(float), cudaMemcpyHostToDevice));
    }

    // AllReduce
    rc = infccl_allreduce(d_bufs.data(), n);
    TEST_ASSERT(rc == 0, "infccl_allreduce failed");

    // Verify: each GPU should have sum = 1 + 2 + ... + n_gpus
    float expected_sum = (float)(n_gpus * (n_gpus + 1) / 2);
    bool ok = true;

    for (int r = 0; r < n_gpus; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        std::vector<float> h_buf(n);
        CHECK_CUDA(cudaMemcpy(h_buf.data(), d_bufs[r],
                               n * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < 8; i++) {  // check first 8
            if (fabsf(h_buf[i] - expected_sum) > 0.01f) {
                printf("GPU %d elem %d: expected %.0f, got %f\n",
                       r, i, expected_sum, h_buf[i]);
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }

    // Cleanup
    for (int r = 0; r < n_gpus; r++) {
        CHECK_CUDA(cudaSetDevice(r));
        cudaFree(d_bufs[r]);
    }
    infccl_destroy();

    TEST_ASSERT(ok, "infccl allreduce result mismatch");
    printf("OK (sum=%.0f across %d GPUs)\n", expected_sum, n_gpus);
}

// ─── Benchmark: MMVQ throughput ──────────────────────────────────
static void bench_mmvq() {
    printf("\n  === MMVQ Benchmark ===\n");

    const int n_in = 2560;   // Qwen3.6 n_embd
    const int n_out = 2560;  // dense projection
    const int n_blocks = n_in / 32;

    // Allocate random-ish data
    void *d_w, *d_x;
    float* d_y;
    CHECK_CUDA(cudaMalloc(&d_w, (int64_t)n_out * n_blocks * 34));  // Q8_0
    CHECK_CUDA(cudaMalloc(&d_x, (int64_t)n_blocks * 36));          // Q8_1
    CHECK_CUDA(cudaMalloc(&d_y, n_out * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_w, 1, (int64_t)n_out * n_blocks * 34));
    CHECK_CUDA(cudaMemset(d_x, 1, (int64_t)n_blocks * 36));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < 10; i++) {
        ix_mmvq_q8_0(d_w, d_x, d_y, n_in, n_out, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Benchmark
    const int N_ITER = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITER; i++) {
        ix_mmvq_q8_0(d_w, d_x, d_y, n_in, n_out, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double us_per = ms * 1000.0 / N_ITER;

    // Bandwidth: read weights + activation, write output
    double weight_bytes = (double)n_out * n_blocks * 34;
    double act_bytes = (double)n_blocks * 36;
    double total_bytes = weight_bytes + act_bytes + n_out * 4;
    double gbps = (total_bytes / 1e9) / (us_per / 1e6);

    printf("  Q8_0 GEMV %dx%d: %.1f us/call, %.1f GB/s\n",
           n_out, n_in, us_per, gbps);

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaStreamDestroy(stream);
}

// ─── Main ─────────────────────────────────────────────────────────
int main() {
    printf("=== infccl Unit 7: CUDA Kernel Tests (BI-V100) ===\n\n");

    // Print GPU info
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    printf("GPUs detected: %d\n", n_gpus);
    for (int i = 0; i < n_gpus; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  GPU %d: %s (SM %d.%d, %d SMs, %.1f GB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.multiProcessorCount,
               prop.totalGlobalMem / 1e9);
    }
    printf("\n");

    cudaSetDevice(0);

    // Correctness tests
    printf("Correctness tests:\n");
    test_warp_size();
    test_warp_reduce();
    test_mmvq_q8_0();
    test_router_top_k();
    test_bf16_gemv();
    test_rms_norm();
    test_moe_combine();
    test_infccl_allreduce();

    // Benchmarks
    bench_mmvq();

    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
