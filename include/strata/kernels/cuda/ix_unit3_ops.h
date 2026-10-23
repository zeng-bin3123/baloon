// ix_unit3_ops.h -- Unit 3 CUDA kernel APIs (RoPE, KV Q8, router, QSA select, sampler)
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
void ix_rope_neox(float* x, const int32_t* pos, const float* cos_tab, const float* sin_tab,
                  int n_head, int head_dim, int n_rot, int max_pos, cudaStream_t stream);
void ix_rope_neox_batch(float* x, const int32_t* pos, const float* cos_tab, const float* sin_tab,
                        int T, int n_head, int head_dim, int n_rot, int max_pos, cudaStream_t stream);
void ix_kv_append_q8(int8_t* kq, int8_t* vq, uint16_t* ks, uint16_t* vs,
    const int32_t* tab, const int32_t* step, const float* kc, const float* vc,
    int kvh, int hd, int ps, cudaStream_t stream);
void ix_kv_gather_q8(const int8_t* kq, const int8_t* vq, const uint16_t* ks, const uint16_t* vs,
    const int32_t* tab, const int32_t* ids, int kvh, int hd, int ps, int ns,
    uint16_t* ko, uint16_t* vo, cudaStream_t stream);
void ix_router_top10(const float* logits, int ne, int k, int32_t* ids, float* wts, cudaStream_t stream);
void ix_block_scores(const float* kp, const float* q, const int32_t* step,
    int nkv, int nh, int nkvh, int hd, int mb, float* sc, cudaStream_t stream);
void ix_block_topk(const float* sc, int mb, int cap, int32_t* ids, cudaStream_t stream);
void ix_greedy_argmax(const float* lg, int nv, int32_t* out, cudaStream_t stream);
void ix_sampler(float* lg, int nv, float temp, float tp, int tk, float pr, float pf, float pp,
    const int32_t* hist, int hl, uint64_t seed, int32_t* out, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
