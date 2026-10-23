// ix_qsa_select.cu -- QSA block scoring + top-k for BI-V100 (ivcore10)

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {
constexpr int QR = 4;

__global__ void ix_block_scores_kernel(
    const float* __restrict__ kp, const float* __restrict__ q, const int32_t* __restrict__ step,
    int nkv, int nh, int nkvh, int hd, int mb, float* __restrict__ sc) {
    const int b=(int)blockIdx.x; if(b>=mb) return;
    const int tid=threadIdx.x, gqa=nh/nkvh;
    int rp=(b+1)*QR-1; if(rp>=nkv) rp=nkv-1;
    float bs=0;
    for(int h=0;h<nh;h++){int kvh=h/gqa;
        float dot=0; for(int d=tid;d<hd;d+=(int)blockDim.x) dot+=kp[((int64_t)rp*nkvh+kvh)*hd+d]*q[(int64_t)h*hd+d];
        for(int o=IX_WARP_SIZE/2;o>0;o>>=1) dot+=ix_shfl_xor(dot,o);
        __shared__ float sd[8]; int wid=tid/IX_WARP_SIZE,lane=tid%IX_WARP_SIZE;
        if(lane==0)sd[wid]=dot; __syncthreads();
        if(tid==0){float s=0;int nw=((int)blockDim.x+IX_WARP_SIZE-1)/IX_WARP_SIZE;for(int w=0;w<nw;w++)s+=sd[w];if(s>0)bs+=s;}
        __syncthreads();}
    if(tid==0) sc[b]=bs;
}

__global__ void ix_block_topk_kernel(const float* __restrict__ sc, int mb, int cap, int32_t* __restrict__ ids) {
    if(threadIdx.x!=0) return;
    float ts[128]; int32_t ti[128];
    for(int i=0;i<cap;i++){ts[i]=-1e30f;ti[i]=-1;}
    for(int b=0;b<mb;b++){float s=sc[b];
        if(s>ts[cap-1]){int p=cap-1;while(p>0&&s>ts[p-1]){ts[p]=ts[p-1];ti[p]=ti[p-1];p--;}ts[p]=s;ti[p]=b;}}
    for(int i=0;i<cap;i++) ids[i]=ti[i];
}
} // anon
extern "C" {
void ix_block_scores(const float* kp, const float* q, const int32_t* step,
    int nkv, int nh, int nkvh, int hd, int mb, float* sc, cudaStream_t stream) {
    ix_block_scores_kernel<<<mb, 256, 0, stream>>>(kp,q,step,nkv,nh,nkvh,hd,mb,sc);
}
void ix_block_topk(const float* sc, int mb, int cap, int32_t* ids, cudaStream_t stream) {
    ix_block_topk_kernel<<<1, 1, 0, stream>>>(sc, mb, cap, ids);
}
} // extern C
}} // ns
