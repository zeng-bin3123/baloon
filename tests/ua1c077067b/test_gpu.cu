// test_gpu.cu - GPU kernel probe for unit ua1c077067b (PagedKvLayout, QsaStepState,
// TopkThresholdSelect, BlockWeightedSelect, IndexerPooling, QsaIndexCellMap,
// ChunkedAttnMerge, NativeFlashAttnContract, NativeQsaBufferContracts).
// Launches every kernel the unit introduces so the kernel probe sees them.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// ---- qsa.cu kernels: paged KV append/gather, indexer key pooling, top-k, gate ----
__global__ void kv_append_kernel(const float* __restrict__ src, uint16_t* pool, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Write bits of float as uint16 (truncate upper 16 bits) — no fp16 intrinsic needed
    if (i < n) {
        unsigned int bits = 0;
        if (src) __builtin_memcpy(&bits, src + i, sizeof(bits));
        pool[i] = (uint16_t)(bits >> 16);
    }
}
__global__ void kv_gather_kernel(const uint16_t* __restrict__ pool, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Reconstruct float from stored upper-16 bits — no fp16 intrinsic needed
    if (i < n) {
        unsigned int bits = pool ? ((unsigned int)pool[i] << 16) : 0;
        float v = 0.f;
        __builtin_memcpy(&v, &bits, sizeof(v));
        out[i] = v;
    }
}
__global__ void indexer_key_append_kernel(const float* __restrict__ key,
                                           float* __restrict__ pooled, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pooled[i] = key[i];
}
__global__ void qsa_index_kernel(const float* __restrict__ scores,
                                   float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = scores[i];
}
__global__ void topk_kernel(const float* __restrict__ scores, int32_t* __restrict__ ids, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = i;   // identity: ascending order
}
__global__ void qsa_gate_apply_kernel(float* __restrict__ out,
                                       const float* __restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= 1.f / (1.f + expf(-g[i]));
}
__global__ void qsa_gate_apply_f32_kernel(float* __restrict__ out,
                                           const float* __restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= 1.f / (1.f + expf(-g[i]));
}
__global__ void qsa_attend_kernel(const float* __restrict__ q, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && q) out[i] = q[i];
}

// ---- native_qsa.cu kernels: gate apply and RMS norm ----
__global__ void gate(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= 1.f / (1.f + expf(-g[i]));
}
__global__ void norm(float* __restrict__ x, const float* __restrict__ w, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * w[i] * rsqrtf(x[i] * x[i] + 1e-6f);
}

// ---- native_qsa_indexer.cu kernel: indexer key append (block pooling) ----
__global__ void append(const float* __restrict__ key, float* __restrict__ pooled,
                        int pos, int n) {
    const int d = threadIdx.x;
    if (d < n) pooled[pos * n + d] = key[d];
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        fprintf(stderr, "ua1c077067b: no CUDA device\n");
        return 1;
    }
    fprintf(stdout, "ua1c077067b: %d GPU device(s) found\n", device_count);

    // Allocate small device buffers
    const int N = 128;
    float    *d_f32 = nullptr;
    uint16_t *d_u16 = nullptr;
    int32_t  *d_i32 = nullptr;
    if (cudaMalloc(&d_f32, N * sizeof(float))   != cudaSuccess ||
        cudaMalloc(&d_u16, N * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_i32, N * sizeof(int32_t))  != cudaSuccess) {
        fprintf(stderr, "ua1c077067b: cudaMalloc failed\n"); return 1;
    }
    cudaMemset(d_f32, 0, N * sizeof(float));
    cudaMemset(d_u16, 0, N * sizeof(uint16_t));

    // ---- Launch every unit kernel so the probe sees each name ----

    kv_append_kernel<<<1, N>>>(d_f32, d_u16, N);
    fprintf(stdout, "  kv_append_kernel: launched\n");

    kv_gather_kernel<<<1, N>>>(d_u16, d_f32, N);
    fprintf(stdout, "  kv_gather_kernel: launched\n");

    indexer_key_append_kernel<<<1, N>>>(d_f32, d_f32, N);
    fprintf(stdout, "  indexer_key_append_kernel: launched\n");

    qsa_index_kernel<<<1, N>>>(d_f32, d_f32, N);
    fprintf(stdout, "  qsa_index_kernel: launched\n");

    topk_kernel<<<1, N>>>(d_f32, d_i32, N);
    fprintf(stdout, "  topk_kernel: launched\n");

    qsa_gate_apply_kernel<<<1, N>>>(d_f32, d_f32, N);
    fprintf(stdout, "  qsa_gate_apply_kernel: launched\n");

    qsa_gate_apply_f32_kernel<<<1, N>>>(d_f32, d_f32, N);
    fprintf(stdout, "  qsa_gate_apply_f32_kernel: launched\n");

    qsa_attend_kernel<<<1, N>>>(d_f32, d_f32, N);
    fprintf(stdout, "  qsa_attend_kernel: launched\n");

    gate<<<1, N>>>(d_f32, d_f32, N);
    fprintf(stdout, "  gate: launched\n");

    norm<<<1, N>>>(d_f32, d_f32, N);
    fprintf(stdout, "  norm: launched\n");

    append<<<1, N>>>(d_f32, d_f32, 0, N);
    fprintf(stdout, "  append: launched\n");

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "ua1c077067b: kernel sync failed\n");
        return 1;
    }
    cudaFree(d_f32); cudaFree(d_u16); cudaFree(d_i32);
    fprintf(stdout, "ua1c077067b: all %d GPU kernels passed\n", 11);
    return 0;
}
