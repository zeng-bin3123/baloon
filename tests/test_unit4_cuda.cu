// test_unit4_cuda.cu -- Integration tests for unit 4 CUDA kernels (BI-V100)
#include "strata/kernels/cuda/ix_unit4_ops.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static int g_pass = 0, g_fail = 0;
#define CHECK_CUDA(c) do{cudaError_t e=(c);if(e!=cudaSuccess){printf("  CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));g_fail++;return;}}while(0)

static void test_gr_write() {
    printf("  [1] GR write (zero inject = plain add) ... ");
    const int ne = 8, hc = 4;
    std::vector<float> hR(hc * ne, 1.0f), hbo(ne, 2.0f), hinj(hc, 0.0f);
    float *dR, *dbo, *dinj, *dout;
    CHECK_CUDA(cudaMalloc(&dR, hc*ne*4)); CHECK_CUDA(cudaMalloc(&dbo, ne*4));
    CHECK_CUDA(cudaMalloc(&dinj, hc*4)); CHECK_CUDA(cudaMalloc(&dout, hc*ne*4));
    CHECK_CUDA(cudaMemcpy(dR, hR.data(), hc*ne*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dbo, hbo.data(), ne*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dinj, hinj.data(), hc*4, cudaMemcpyHostToDevice));
    ix_gr_write(dR, dbo, dinj, ne, hc, dout, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> ho(hc * ne);
    CHECK_CUDA(cudaMemcpy(ho.data(), dout, hc*ne*4, cudaMemcpyDeviceToHost));
    // w = 2*sigmoid(0/4) = 2*0.5 = 1.0, so R'[c,d] = 1 + 2*1 = 3.0
    if (fabsf(ho[0] - 3.0f) > 1e-4f) { printf("FAIL: got %.4f exp 3.0\n", ho[0]); g_fail++; }
    else { printf("PASS\n"); g_pass++; }
    cudaFree(dR); cudaFree(dbo); cudaFree(dinj); cudaFree(dout);
}

static void test_quantize_q8_0() {
    printf("  [2] Q8_0 quantize/dequant round-trip ... ");
    const int n = 64;
    std::vector<float> hx(n);
    for (int i = 0; i < n; i++) hx[i] = (float)(i - 32) * 0.1f;
    float *dx, *dout;
    uint8_t *dq;
    int n_blocks = (n + 31) / 32;
    int q_bytes = n_blocks * 34;
    CHECK_CUDA(cudaMalloc(&dx, n*4)); CHECK_CUDA(cudaMalloc(&dq, q_bytes));
    CHECK_CUDA(cudaMalloc(&dout, n*4));
    CHECK_CUDA(cudaMemcpy(dx, hx.data(), n*4, cudaMemcpyHostToDevice));
    ix_quantize_q8_0(dx, dq, n, 0);
    ix_dequant_q8_0(dq, dout, n, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> ho(n);
    CHECK_CUDA(cudaMemcpy(ho.data(), dout, n*4, cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ho[i] - hx[i]);
        if (err > max_err) max_err = err;
    }
    // Q8_0 with 127 levels should have max error < 0.03 for this range
    if (max_err > 0.05f) { printf("FAIL: max_err=%.4f\n", max_err); g_fail++; }
    else { printf("PASS (max_err=%.4f)\n", max_err); g_pass++; }
    cudaFree(dx); cudaFree(dq); cudaFree(dout);
}

static void test_fused_gdn() {
    printf("  [3] Fused GDN step+norm ... ");
    const int nh = 2, S = 4;
    std::vector<float> state(nh*S*S, 0.0f);
    std::vector<float> q(nh*S, 1.0f), k(nh*S, 1.0f), v(nh*S, 1.0f);
    std::vector<float> gate(nh, 0.5f), beta(nh, 1.0f), z(nh*S, 0.0f);
    std::vector<float> gamma(nh*S, 1.0f);
    float *ds,*dq,*dk,*dv,*dg,*db,*dz,*dgm,*dr;
    CHECK_CUDA(cudaMalloc(&ds, nh*S*S*4)); CHECK_CUDA(cudaMalloc(&dq, nh*S*4));
    CHECK_CUDA(cudaMalloc(&dk, nh*S*4)); CHECK_CUDA(cudaMalloc(&dv, nh*S*4));
    CHECK_CUDA(cudaMalloc(&dg, nh*4)); CHECK_CUDA(cudaMalloc(&db, nh*4));
    CHECK_CUDA(cudaMalloc(&dz, nh*S*4)); CHECK_CUDA(cudaMalloc(&dgm, nh*S*4));
    CHECK_CUDA(cudaMalloc(&dr, nh*S*4));
    CHECK_CUDA(cudaMemcpy(ds, state.data(), nh*S*S*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dq, q.data(), nh*S*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dk, k.data(), nh*S*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dv, v.data(), nh*S*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dg, gate.data(), nh*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db, beta.data(), nh*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dz, z.data(), nh*S*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dgm, gamma.data(), nh*S*4, cudaMemcpyHostToDevice));
    ix_fused_gdn_step_norm(ds,dq,dk,dv,dg,db,dz,dgm,1e-5f,dr,nh,S,0);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> hr(nh*S);
    CHECK_CUDA(cudaMemcpy(hr.data(), dr, nh*S*4, cudaMemcpyDeviceToHost));
    // With z=0, sigmoid(0)=0.5, result should be non-zero
    bool ok = false;
    for (int i = 0; i < nh*S; i++) if (hr[i] != 0.0f) ok = true;
    // With zero initial state, after one step state gets populated
    if (!ok) { printf("FAIL: all zeros\n"); g_fail++; }
    else { printf("PASS (r[0]=%.6f)\n", hr[0]); g_pass++; }
    cudaFree(ds);cudaFree(dq);cudaFree(dk);cudaFree(dv);
    cudaFree(dg);cudaFree(db);cudaFree(dz);cudaFree(dgm);cudaFree(dr);
}

int main() {
    printf("=== infccl Unit 4: CUDA Kernel Tests (BI-V100) ===\n\n");
    int dc=0; cudaGetDeviceCount(&dc);
    printf("GPUs detected: %d\n",dc);
    if(!dc){printf("No GPU.\n");return 1;}
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("  GPU 0: %s\n\n",p.name);
    test_gr_write(); test_quantize_q8_0(); test_fused_gdn();
    printf("\n=== Results: %d passed, %d failed ===\n",g_pass,g_fail);
    return g_fail>0?1:0;
}
