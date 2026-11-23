// moe_parity.cu — GPU parity tests for the MoE FFN block (uintegration01529aaa82b9).
// Five sentences of the MoE spec, each covered by one test function:
//   S1 test_moe_router_logits   : mlp.gate.weight × x → router logits
//   S2 test_moe_top8_select     : top-8 expert selection via native_router_top10
//   S3 test_moe_shared_expert   : gate_proj/up_proj→SwiGLU→down_proj via shared_expert_multi
//   S4 test_moe_shared_gate     : scalar sigmoid gate on the shared expert output
//   S5 test_moe_combine         : sum all contributions via moe_combine
// Exit 0 on PASS.
#include "baloon/kg_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

static int g_pass = 0, g_fail = 0;
#define MOE_CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){                  \
    std::fprintf(stderr,"CUDA err %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); \
    std::exit(2); } } while(0)
#define MOE_T(cond) do { if(cond){++g_pass;}else{++g_fail;                        \
    std::fprintf(stderr,"FAIL %s:%d\n",__FILE__,__LINE__);} } while(0)

static uint16_t f32_to_bf16(float f){uint32_t u;memcpy(&u,&f,4);return(uint16_t)(u>>16);}
static float bf16_to_f32(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
static float sigmoidf(float x){return 1.f/(1.f+expf(-x));}
static float siluf(float x){return x*sigmoidf(x);}

template<typename T>
static T* dev_alloc(size_t n){T*p=nullptr;MOE_CK(cudaMalloc((void**)&p,n*sizeof(T)));return p;}
template<typename T>
static void dev_put(T*d,const std::vector<T>&h){MOE_CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<typename T>
static std::vector<T> dev_get(T*d,size_t n){
    MOE_CK(cudaDeviceSynchronize());
    std::vector<T>h(n);MOE_CK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return h;}

// ---------------------------------------------------------------------------
// S1: multiplies hidden state by mlp.gate.weight (BF16) to get router logits
// ---------------------------------------------------------------------------
static void test_moe_router_logits() {
    const int NE = 16, NH = 32;   // small but representative shapes
    // Build BF16 weight matrix gate_w[NE, NH] and f32 input x[NH]
    std::vector<uint16_t> gw(NE * NH);
    std::vector<float> x(NH);
    for (int i = 0; i < NE * NH; ++i) gw[i] = f32_to_bf16(0.1f * ((i % 7) - 3));
    for (int i = 0; i < NH; ++i)      x[i]  = 0.5f * (i % 5 + 1);

    // Host reference: logits[e] = sum_j bf16(gw[e*NH+j]) * x[j]
    std::vector<float> ref(NE, 0.f);
    for (int e = 0; e < NE; ++e)
        for (int j = 0; j < NH; ++j)
            ref[e] += bf16_to_f32(gw[e * NH + j]) * x[j];

    uint16_t* d_gw = dev_alloc<uint16_t>(NE * NH);
    float*    d_x  = dev_alloc<float>(NH);
    float*    d_lg = dev_alloc<float>(NE);
    dev_put(d_gw, gw); dev_put(d_x, x);
    MOE_CK(baloon::kg::moe_router_logits(d_gw, d_x, d_lg, NE, NH));
    auto got = dev_get(d_lg, NE);
    for (int e = 0; e < NE; ++e)
        MOE_T(fabsf(got[e] - ref[e]) < 1e-3f);
    // bad-arg guard
    MOE_T(baloon::kg::moe_router_logits(nullptr, d_x, d_lg, NE, NH) == cudaErrorInvalidValue);
    MOE_T(baloon::kg::moe_router_logits(d_gw, d_x, d_lg, 0, NH)    == cudaErrorInvalidValue);
    cudaFree(d_gw); cudaFree(d_x); cudaFree(d_lg);
    printf("moe_parity: test_moe_router_logits done\n");
}

// ---------------------------------------------------------------------------
// S2: selects top-8 experts (kernels in u12d8dd5102) via native_router_top10
// ---------------------------------------------------------------------------
static void test_moe_top8_select() {
    const int NE = 256, K = 8;
    std::vector<float> logits(NE);
    for (int i = 0; i < NE; ++i) logits[i] = 0.1f * (i % 13) - 0.6f;

    float* d_lg = dev_alloc<float>(NE);
    int32_t* d_ids = dev_alloc<int32_t>(K);
    float* d_wt = dev_alloc<float>(K);
    dev_put(d_lg, logits);
    MOE_CK(baloon::kg::native_router_top10(d_lg, 1, NE, K, d_ids, d_wt));
    auto ids = dev_get(d_ids, K);
    auto wts = dev_get(d_wt, K);
    // All ids must be in [0, NE) and unique
    std::vector<int32_t> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < K; ++i) MOE_T(ids[i] >= 0 && ids[i] < NE);
    for (int i = 1; i < K; ++i) MOE_T(sorted[i] != sorted[i-1]);
    // Weights sum ≈ 1 (renorm)
    float wsum = 0.f; for (float w : wts) wsum += w;
    MOE_T(fabsf(wsum - 1.f) < 1e-4f);
    // bad-arg guard
    MOE_T(baloon::kg::native_router_top10(nullptr, 1, NE, K, d_ids, d_wt) == cudaErrorInvalidValue);
    cudaFree(d_lg); cudaFree(d_ids); cudaFree(d_wt);
    printf("moe_parity: test_moe_top8_select done\n");
}

// ---------------------------------------------------------------------------
// S3: runs the shared expert (gate_proj, up_proj → SwiGLU → down_proj)
// ---------------------------------------------------------------------------
static void test_moe_shared_expert() {
    const int NH = 16, NF = 8;   // n_embd=16, n_ff=8
    // Build BF16 weights
    std::vector<uint16_t> gate_w(NF * NH), up_w(NF * NH), down_w(NH * NF), sg_w(NH);
    for (int i = 0; i < NF * NH; ++i) {
        gate_w[i] = f32_to_bf16(0.1f * ((i % 5) - 2));
        up_w[i]   = f32_to_bf16(0.1f * ((i % 7) - 3));
    }
    for (int i = 0; i < NH * NF; ++i) down_w[i] = f32_to_bf16(0.1f * ((i % 11) - 5));
    // scalar gate weight: all BF16(1) so sigmoid(dot(x,sg_w))=sigmoid(sum(x))
    for (int i = 0; i < NH; ++i) sg_w[i] = f32_to_bf16(1.f);
    std::vector<float> x(NH);
    for (int i = 0; i < NH; ++i) x[i] = 0.5f * (i % 4 + 1);

    // Host reference: gate_proj=gate_w*x, up_proj=up_w*x; h=silu(gate)*up; out=down*h
    std::vector<float> gp(NF, 0.f), up(NF, 0.f), h(NF), ref(NH, 0.f);
    for (int o = 0; o < NF; ++o)
        for (int j = 0; j < NH; ++j) {
            gp[o] += bf16_to_f32(gate_w[o * NH + j]) * x[j];
            up[o] += bf16_to_f32(up_w  [o * NH + j]) * x[j];
        }
    for (int o = 0; o < NF; ++o) h[o] = siluf(gp[o]) * up[o];
    for (int o = 0; o < NH; ++o)
        for (int j = 0; j < NF; ++j)
            ref[o] += bf16_to_f32(down_w[o * NF + j]) * h[j];
    // scale by scalar gate = sigmoid(dot(x, sg_w)) = sigmoid(sum(x))
    float dot_sg = 0.f; for (float v : x) dot_sg += v;
    float gate_val = sigmoidf(dot_sg);
    for (float& v : ref) v *= gate_val;

    size_t sc = baloon::kg::shared_expert_scratch_bytes(1, NH, NF);
    uint16_t *d_gw=dev_alloc<uint16_t>(NF*NH), *d_uw=dev_alloc<uint16_t>(NF*NH);
    uint16_t *d_dw=dev_alloc<uint16_t>(NH*NF), *d_sw=dev_alloc<uint16_t>(NH);
    float *d_x=dev_alloc<float>(NH), *d_out=dev_alloc<float>(NH);
    void  *d_sc=nullptr; MOE_CK(cudaMalloc(&d_sc, sc));
    dev_put(d_gw, gate_w); dev_put(d_uw, up_w); dev_put(d_dw, down_w);
    dev_put(d_sw, sg_w); dev_put(d_x, x);
    MOE_CK(baloon::kg::shared_expert(d_x, NH, NF, d_gw, d_uw, d_dw, d_sw, d_out, d_sc, sc));
    auto got = dev_get(d_out, NH);
    for (int o = 0; o < NH; ++o) MOE_T(fabsf(got[o] - ref[o]) < 2e-3f);
    // bad-arg guard
    MOE_T(baloon::kg::shared_expert(nullptr, NH, NF, d_gw, d_uw, d_dw, d_sw, d_out, d_sc, sc)
          == cudaErrorInvalidValue);
    cudaFree(d_gw); cudaFree(d_uw); cudaFree(d_dw); cudaFree(d_sw);
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_sc);
    printf("moe_parity: test_moe_shared_expert done\n");
}

// ---------------------------------------------------------------------------
// S4: gates the shared contribution with shared_expert_gate (scalar sigmoid)
// ---------------------------------------------------------------------------
static void test_moe_shared_gate() {
    // Verify that shared_expert output is scaled by sigmoid(dot(x, shared_expert_gate.weight)).
    const int NH = 16, NF = 8;
    // Use all-zero gate_proj/up_proj/down_proj weights → down output is 0 before gate.
    // Then gate_scalar_w with one nonzero entry forces a known sigmoid value.
    std::vector<uint16_t> gate_w(NF*NH,0), up_w(NF*NH,0), down_w(NH*NF,0), sg_w(NH,0);
    // down_w row 0: all 1 so out[0] = sum(h) = 0 before gate → still 0 (down of 0 = 0)
    // We need nonzero h: use gate_w[0,0]=1 and up_w[0,0]=1 so h[0]=silu(x[0])*x[0]
    gate_w[0] = f32_to_bf16(1.f); up_w[0] = f32_to_bf16(1.f);
    for (int j = 0; j < NF; ++j) down_w[j] = f32_to_bf16(1.f);  // down_w[0, j]=1
    // scalar gate weight: sg_w[0]=2 → dot(x,sg_w)=2*x[0]
    sg_w[0] = f32_to_bf16(2.f);
    std::vector<float> x(NH, 0.f); x[0] = 1.f;

    float h0 = siluf(1.f) * 1.f;
    float down0 = h0;  // out[0] = down_w[0,:]*h = 1*h0 = h0
    float gate_scalar = sigmoidf(2.f * x[0]);  // sigmoid(dot(x, sg_w))
    float ref0 = down0 * gate_scalar;

    size_t sc = baloon::kg::shared_expert_scratch_bytes(1, NH, NF);
    uint16_t *d_gw=dev_alloc<uint16_t>(NF*NH), *d_uw=dev_alloc<uint16_t>(NF*NH);
    uint16_t *d_dw=dev_alloc<uint16_t>(NH*NF), *d_sw=dev_alloc<uint16_t>(NH);
    float *d_x=dev_alloc<float>(NH), *d_out=dev_alloc<float>(NH);
    void  *d_sc=nullptr; MOE_CK(cudaMalloc(&d_sc, sc));
    dev_put(d_gw, gate_w); dev_put(d_uw, up_w); dev_put(d_dw, down_w);
    dev_put(d_sw, sg_w); dev_put(d_x, x);
    MOE_CK(baloon::kg::shared_expert(d_x, NH, NF, d_gw, d_uw, d_dw, d_sw, d_out, d_sc, sc));
    auto got = dev_get(d_out, NH);
    // out[0] must equal h0 * sigmoid(2*x[0])
    MOE_T(fabsf(got[0] - ref0) < 1e-4f);
    cudaFree(d_gw); cudaFree(d_uw); cudaFree(d_dw); cudaFree(d_sw);
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_sc);
    printf("moe_parity: test_moe_shared_gate done\n");
}

// ---------------------------------------------------------------------------
// S5: sums all contributions via moe_combine
// ---------------------------------------------------------------------------
static void test_moe_combine() {
    const int NE = 4, NH = 8;
    // parts[NE, NH], weights[NE], shared[NH]
    std::vector<float> parts(NE * NH), weights(NE), shared(NH), ref(NH, 0.f);
    for (int e = 0; e < NE; ++e) weights[e] = 1.f / NE;
    for (int i = 0; i < NE * NH; ++i) parts[i] = (float)(i + 1) * 0.1f;
    for (int j = 0; j < NH; ++j)      shared[j] = (float)(j + 1) * 0.5f;
    // reference: y[j] = sum_e(w[e]*parts[e][j]) + shared[j]
    for (int j = 0; j < NH; ++j) {
        for (int e = 0; e < NE; ++e) ref[j] += weights[e] * parts[e * NH + j];
        ref[j] += shared[j];
    }

    float *d_parts=dev_alloc<float>(NE*NH), *d_wt=dev_alloc<float>(NE);
    float *d_sh=dev_alloc<float>(NH), *d_y=dev_alloc<float>(NH);
    dev_put(d_parts, parts); dev_put(d_wt, weights); dev_put(d_sh, shared);
    MOE_CK(baloon::kg::moe_combine(d_parts, d_wt, d_sh, d_y, NH, NE, 1));
    auto got = dev_get(d_y, NH);
    for (int j = 0; j < NH; ++j) MOE_T(fabsf(got[j] - ref[j]) < 1e-4f);
    // without shared: y[j] = sum_e(w[e]*parts[e][j])
    std::fill(ref.begin(), ref.end(), 0.f);
    for (int j = 0; j < NH; ++j)
        for (int e = 0; e < NE; ++e) ref[j] += weights[e] * parts[e * NH + j];
    MOE_CK(baloon::kg::moe_combine(d_parts, d_wt, nullptr, d_y, NH, NE, 0));
    got = dev_get(d_y, NH);
    for (int j = 0; j < NH; ++j) MOE_T(fabsf(got[j] - ref[j]) < 1e-4f);
    // bad-arg guard
    MOE_T(baloon::kg::moe_combine(nullptr, d_wt, d_sh, d_y, NH, NE, 1) == cudaErrorInvalidValue);
    cudaFree(d_parts); cudaFree(d_wt); cudaFree(d_sh); cudaFree(d_y);
    printf("moe_parity: test_moe_combine done\n");
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        printf("moe_parity: no GPU\n"); return 2;
    }
    MOE_CK(cudaSetDevice(0));
    test_moe_router_logits();
    test_moe_top8_select();
    test_moe_shared_expert();
    test_moe_shared_gate();
    test_moe_combine();
    printf("moe_parity: pass=%d fail=%d\n", g_pass, g_fail);
    if (g_fail) { printf("moe_parity FAILED\n"); return 1; }
    printf("moe_parity selftest: ok\n"); return 0;
}

