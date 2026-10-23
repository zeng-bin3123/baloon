// ix_kv_q8.cu -- INT8 KV cache append/gather for BI-V100 (ivcore10)
// Per-64-group quantize via fp16 scale round-trip. Paged storage.

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {
constexpr int KVG = 64;

__global__ void ix_kv_append_q8_kernel(
    int8_t* __restrict__ kq, int8_t* __restrict__ vq,
    uint16_t* __restrict__ ks, uint16_t* __restrict__ vs,
    const int32_t* __restrict__ tab, const int32_t* __restrict__ step,
    const float* __restrict__ kc, const float* __restrict__ vc,
    int kvh, int hd, int ps) {
    const int head = (int)blockIdx.x, grp = (int)blockIdx.y, tid = threadIdx.x;
    if (head >= kvh) return;
    const int gs = grp * KVG; if (gs >= hd) return;
    const int ge = (gs + KVG < hd) ? gs + KVG : hd, gsz = ge - gs;
    const int pos = step[0], page = tab[pos / ps];
    const int64_t row = (int64_t)page * kvh * ps + (int64_t)head * ps + (pos % ps);
    const int ng = (hd + KVG - 1) / KVG;
    __shared__ float sh[4];
    // K amax
    float am = 0; for (int i = tid; i < gsz; i += (int)blockDim.x) { float v = kc[head*hd+gs+i]; float a = v<0?-v:v; if(a>am) am=a; }
    for (int o=IX_WARP_SIZE/2;o>0;o>>=1){float t=ix_shfl_xor(am,o);if(t>am)am=t;}
    int wid=tid/IX_WARP_SIZE,lane=tid%IX_WARP_SIZE;
    if(lane==0)sh[wid]=am; __syncthreads();
    if(tid==0){float m=sh[0];for(int w=1;w<2;w++)if(sh[w]>m)m=sh[w];sh[0]=m;} __syncthreads();
    am=sh[0]; float sc=am/127.0f; uint16_t s16=__half_as_ushort(__float2half(sc));
    float sst=__half2float(__ushort_as_half(s16)); float inv=(sst>0)?1.0f/sst:0.0f;
    for(int i=tid;i<gsz;i+=(int)blockDim.x){float v=kc[head*hd+gs+i];int q=(int)roundf(v*inv);q=q<-127?-127:(q>127?127:q);kq[row*hd+gs+i]=(int8_t)q;}
    if(tid==0) ks[row*ng+grp]=s16;
    // V amax
    am=0; for(int i=tid;i<gsz;i+=(int)blockDim.x){float v=vc[head*hd+gs+i];float a=v<0?-v:v;if(a>am)am=a;}
    for(int o=IX_WARP_SIZE/2;o>0;o>>=1){float t=ix_shfl_xor(am,o);if(t>am)am=t;}
    if(lane==0)sh[wid]=am; __syncthreads();
    if(tid==0){float m=sh[0];for(int w=1;w<2;w++)if(sh[w]>m)m=sh[w];sh[0]=m;} __syncthreads();
    am=sh[0]; sc=am/127.0f; s16=__half_as_ushort(__float2half(sc));
    sst=__half2float(__ushort_as_half(s16)); inv=(sst>0)?1.0f/sst:0.0f;
    for(int i=tid;i<gsz;i+=(int)blockDim.x){float v=vc[head*hd+gs+i];int q=(int)roundf(v*inv);q=q<-127?-127:(q>127?127:q);vq[row*hd+gs+i]=(int8_t)q;}
    if(tid==0) vs[row*ng+grp]=s16;
}

__global__ void ix_kv_gather_q8_kernel(
    const int8_t* __restrict__ kq, const int8_t* __restrict__ vq,
    const uint16_t* __restrict__ ks, const uint16_t* __restrict__ vs,
    const int32_t* __restrict__ tab, const int32_t* __restrict__ ids,
    int kvh, int hd, int ps, int ns,
    uint16_t* __restrict__ ko, uint16_t* __restrict__ vo) {
    const int sel=(int)blockIdx.z, head=(int)blockIdx.x;
    const int bd=(int)blockIdx.y*blockDim.x*4+threadIdx.x*4;
    if(sel>=ns||head>=kvh||bd>=hd) return;
    const int pos=ids[sel], page=tab[pos/ps];
    const int64_t row=(int64_t)page*kvh*ps+(int64_t)head*ps+(pos%ps);
    const int ng=(hd+KVG-1)/KVG, grp=bd/KVG;
    float ksc=__half2float(__ushort_as_half(ks[row*ng+grp])), vsc=__half2float(__ushort_as_half(vs[row*ng+grp]));
    const int64_t db=row*hd+bd, ob=((int64_t)sel*kvh+head)*hd+bd;
    int cnt=hd-bd; if(cnt>4) cnt=4;
    for(int i=0;i<cnt;i++){ko[ob+i]=__half_as_ushort(__float2half((float)kq[db+i]*ksc));vo[ob+i]=__half_as_ushort(__float2half((float)vq[db+i]*vsc));}
}
} // anon
extern "C" {
void ix_kv_append_q8(int8_t* kq, int8_t* vq, uint16_t* ks, uint16_t* vs,
    const int32_t* tab, const int32_t* step, const float* kc, const float* vc,
    int kvh, int hd, int ps, cudaStream_t stream) {
    dim3 grid(kvh, (hd+KVG-1)/KVG);
    ix_kv_append_q8_kernel<<<grid, 128, 0, stream>>>(kq,vq,ks,vs,tab,step,kc,vc,kvh,hd,ps);
}
void ix_kv_gather_q8(const int8_t* kq, const int8_t* vq, const uint16_t* ks, const uint16_t* vs,
    const int32_t* tab, const int32_t* ids, int kvh, int hd, int ps, int ns,
    uint16_t* ko, uint16_t* vo, cudaStream_t stream) {
    dim3 grid(kvh, (hd+64*4-1)/(64*4), ns);
    ix_kv_gather_q8_kernel<<<grid, 64, 0, stream>>>(kq,vq,ks,vs,tab,ids,kvh,hd,ps,ns,ko,vo);
}
} // extern C
}} // ns
