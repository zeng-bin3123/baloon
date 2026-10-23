// ix_router_top10.cu -- MoE router: softmax + stable top-k + renormalize (BI-V100)
// Stable argsort by rank. 512 experts, single block. Renorm 2^-14 floor.

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {
constexpr int MX = 512;

__global__ void ix_router_top10_kernel(
    const float* __restrict__ logits, int ne, int k,
    int32_t* __restrict__ ids, float* __restrict__ wts) {
    __shared__ float sp[MX]; __shared__ float swm[8];
    const int tid = threadIdx.x;
    float l = (tid < ne) ? logits[tid] : -1e30f;
    float wm = l;
    for(int o=IX_WARP_SIZE/2;o>0;o>>=1){float t=ix_shfl_xor(wm,o);if(t>wm)wm=t;}
    int wid=tid/IX_WARP_SIZE, lane=tid%IX_WARP_SIZE;
    if(lane==0) swm[wid]=wm; __syncthreads();
    if(tid==0){float m=swm[0];for(int w=1;w<8;w++)if(swm[w]>m)m=swm[w];swm[0]=m;} __syncthreads();
    float gm=swm[0];
    float e=(tid<ne)?expf(l-gm):0.0f; sp[tid]=e; __syncthreads();
    __shared__ float ss;
    if(tid==0){float s=0;for(int i=0;i<ne;i++)s+=sp[i];ss=s;} __syncthreads();
    float inv=1.0f/ss;
    if(tid<ne) sp[tid]*=inv; __syncthreads();
    if(tid<ne){float my=sp[tid];int rank=0;
        for(int f=0;f<ne;f++){float pf=sp[f];if(pf>my||(pf==my&&f<tid))rank++;}
        if(rank<k){ids[rank]=tid;wts[rank]=my;}}
    __syncthreads();
    if(tid==0){float ws=0;for(int i=0;i<k;i++)ws+=wts[i];
        float d=ws;if(d<6.103515625e-05f)d=6.103515625e-05f;float ri=1.0f/d;
        for(int i=0;i<k;i++)wts[i]*=ri;}
}
} // anon
extern "C" {
void ix_router_top10(const float* logits, int ne, int k, int32_t* ids, float* wts, cudaStream_t stream) {
    ix_router_top10_kernel<<<1, MX, 0, stream>>>(logits, ne, k, ids, wts);
}
} // extern C
}} // ns
