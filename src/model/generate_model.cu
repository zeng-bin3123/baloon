// generate_model.cu - real-model backend for the hybrid GDN / full-attention / MoE checkpoint (HF safetensors, bf16).
// Loads every weight from MODEL_DIR, splits the 40 layers evenly over the visible GPUs, and runs greedy decoding.
// `--bench` measures prefill and decode throughput from real forward passes and prints
//     prefill: X tok/s
//     decode: Y tok/s
// Before timing anything it runs an invariant check (chunked prefill must agree with token-by-token prefill) and refuses to
// print numbers if logits are non-finite or the check fails.
//
// Architecture (read from config.json, asserted at load):
//   x += Attn(RMSNorm(x)); x += MoE(RMSNorm(x)); every 4th layer is full attention, the others gated delta-net (GDN).
//   RMSNorm (input/post/final/q_norm/k_norm): y = x * rsqrt(mean(x^2)+eps) * (1 + w).   GDN output norm: weight used as is.
//   Full attention: q_proj -> per head [query(256) | gate(256)], per-head q/k RMSNorm, neox RoPE on the first 64 dims
//     (theta 1e7; with identical t/h/w positions the interleaved mrope reduces to plain RoPE), GQA 16:2, out = o_proj(attn * sigmoid(gate)).
//   GDN: in_proj_qkv (q16*128 | k16*128 | v32*128) -> causal depthwise conv(4)+silu -> l2norm(q,k), q *= 128^-0.5,
//     g = -exp(A_log)*softplus(a+dt_bias), beta = sigmoid(b), S = S*exp(g); S += k (x) beta*(v - S^T k); o = S^T q,
//     then RMSNorm(o)*w*silu(z) and out_proj. Key head of value head h is h/2.
//   MoE: softmax over 256 experts, top-8, renormalised; fused gate_up_proj [E,2*512,2048], down_proj [E,2048,512];
//     plus shared expert scaled by sigmoid(shared_expert_gate(x)).
#include <cuda_runtime.h>
#include "ix_compat.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

namespace {
constexpr int H = 2048, NL = 40, VOC = 248320, NE = 256, TOPK = 8, MI = 512;
constexpr int NH = 16, NKV = 2, HD = 256, ROT = 64;
constexpr int LKH = 16, LVH = 32, LD = 128, QKV = 8192, VALD = 4096;
constexpr int NBM = 8;           // tokens per chunk
constexpr float EPS = 1e-6f;
constexpr float ROPE_THETA = 1.0e7f;

// ============================================================ kernels
__device__ __forceinline__ float siluf(float x) { return x / (1.f + expf(-x)); }

__device__ float bsum(float v, float* red, int nt) {
    red[threadIdx.x] = v; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if ((int)threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
    float r = red[0]; __syncthreads(); return r;
}
__device__ float bmax(float v, float* red, int nt) {
    red[threadIdx.x] = v; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if ((int)threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
    float r = red[0]; __syncthreads(); return r;
}

// y[b*ystride + row] (+)= W[row,:] . x[b,:]   one wave (64 lanes) per row, 4 rows per block, up to NBM columns
__global__ void k_gemv(const uint16_t* __restrict__ W, const float* __restrict__ x, float* y, int rows, int cols, int n, int ystride, int accum) {
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const int row = blockIdx.x * 4 + wave;
    if (row >= rows) return;
    const uint32_t* w = (const uint32_t*)(W + (size_t)row * cols);
    float acc[NBM];
#pragma unroll
    for (int b = 0; b < NBM; ++b) acc[b] = 0.f;
    for (int c = lane; c < cols / 2; c += 64) {
        const uint32_t v = w[c];
        const float w0 = __uint_as_float(v << 16), w1 = __uint_as_float(v & 0xffff0000u);
#pragma unroll
        for (int b = 0; b < NBM; ++b)
            if (b < n) { const float2 xv = *(const float2*)(x + (size_t)b * cols + 2 * c); acc[b] += w0 * xv.x + w1 * xv.y; }
    }
#pragma unroll
    for (int b = 0; b < NBM; ++b) acc[b] = ix_warp_reduce_sum(acc[b]);
    if (lane == 0) {
        for (int b = 0; b < n; ++b) { float* p = y + (size_t)b * ystride + row; *p = accum ? *p + acc[b] : acc[b]; }
    }
}

__global__ void k_embed(const uint16_t* E, const int* tok, float* x) {
    const int t = blockIdx.x; const uint16_t* r = E + (size_t)tok[t] * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) x[(size_t)t * H + i] = __uint_as_float(((uint32_t)r[i]) << 16);
}

__global__ void k_rmsnorm(const float* x, const float* w, float* out, int dim, int plus1) {
    __shared__ float red[256];
    const float* xr = x + (size_t)blockIdx.x * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += 256) s += xr[i] * xr[i];
    s = bsum(s, red, 256);
    const float r = rsqrtf(s / dim + EPS);
    for (int i = threadIdx.x; i < dim; i += 256) out[(size_t)blockIdx.x * dim + i] = xr[i] * r * (plus1 ? 1.f + w[i] : w[i]);
}

__global__ void k_gdn_conv(float* qkv, const float* cw, float* cs, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= QKV) return;
    float s0 = cs[c * 3], s1 = cs[c * 3 + 1], s2 = cs[c * 3 + 2];
    const float w0 = cw[c * 4], w1 = cw[c * 4 + 1], w2 = cw[c * 4 + 2], w3 = cw[c * 4 + 3];
    for (int t = 0; t < n; ++t) {
        const float xin = qkv[(size_t)t * QKV + c];
        const float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xin;
        qkv[(size_t)t * QKV + c] = siluf(y);
        s0 = s1; s1 = s2; s2 = xin;
    }
    cs[c * 3] = s0; cs[c * 3 + 1] = s1; cs[c * 3 + 2] = s2;
}

__global__ void k_gdn_step(const float* qkv, const float* a_in, const float* b_in, const float* Alog, const float* dtb, float* S, float* out, int n) {
    const int h = blockIdx.x, j = threadIdx.x, kh = h / 2;
    __shared__ float qs[LD], ks[LD], red[LD];
    float* Sh = S + (size_t)h * LD * LD;
    for (int t = 0; t < n; ++t) {
        const float* base = qkv + (size_t)t * QKV;
        const float qv = base[kh * LD + j], kv = base[LKH * LD + kh * LD + j], vv = base[2 * LKH * LD + h * LD + j];
        const float qn = qv * rsqrtf(bsum(qv * qv, red, LD) + EPS);
        const float kn = kv * rsqrtf(bsum(kv * kv, red, LD) + EPS);
        qs[j] = qn * 0.08838834764831845f;   // 128^-0.5
        ks[j] = kn;
        __syncthreads();
        const float av = a_in[t * LVH + h] + dtb[h];
        const float sp = av > 20.f ? av : log1pf(expf(av));
        const float decay = expf(-expf(Alog[h]) * sp);
        const float beta = 1.f / (1.f + expf(-b_in[t * LVH + h]));
        float kvm = 0.f;
        for (int i = 0; i < LD; ++i) { const float s = Sh[i * LD + j] * decay; Sh[i * LD + j] = s; kvm += s * ks[i]; }
        const float d = (vv - kvm) * beta;
        float o = 0.f;
        for (int i = 0; i < LD; ++i) { const float s = Sh[i * LD + j] + ks[i] * d; Sh[i * LD + j] = s; o += s * qs[i]; }
        out[(size_t)t * VALD + h * LD + j] = o;
        __syncthreads();
    }
}

__global__ void k_gdn_norm_gate(float* o, const float* z, const float* nw) {
    __shared__ float red[LD];
    const size_t base = (size_t)blockIdx.x * LD;   // (t*32 + h)
    const float v = o[base + threadIdx.x];
    const float ss = bsum(v * v, red, LD);
    o[base + threadIdx.x] = v * rsqrtf(ss / LD + EPS) * nw[threadIdx.x] * siluf(z[base + threadIdx.x]);
}

// grid (n, NH + NKV), block 256. blocks < NH: query heads (norm+rope -> qo); blocks >= NH: kv heads (norm+rope k, copy v -> cache)
__global__ void k_attn_prep(const float* qfull, const float* kb, const float* vb, const float* qnw, const float* knw,
                            float* qo, float* Kc, float* Vc, int p0) {
    __shared__ float red[256], sm[256];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x, pos = p0 + t;
    const bool isq = hh < NH;
    const float v = isq ? qfull[(size_t)t * NH * 2 * HD + hh * 2 * HD + d] : kb[(size_t)t * NKV * HD + (hh - NH) * HD + d];
    const float ss = bsum(v * v, red, 256);
    const float xn = v * rsqrtf(ss / HD + EPS) * (1.f + (isq ? qnw[d] : knw[d]));
    sm[d] = xn; __syncthreads();
    float r = xn;
    if (d < ROT) {
        const int i = d & 31;
        const float ang = (float)pos * powf(ROPE_THETA, -(2.f * i) / (float)ROT);
        float sn, cs; sincosf(ang, &sn, &cs);
        r = d < 32 ? sm[d] * cs - sm[d + 32] * sn : sm[d] * cs + sm[d - 32] * sn;
    }
    if (isq) qo[(size_t)t * NH * HD + hh * HD + d] = r;
    else {
        const int kvh = hh - NH;
        Kc[((size_t)pos * NKV + kvh) * HD + d] = r;
        Vc[((size_t)pos * NKV + kvh) * HD + d] = vb[(size_t)t * NKV * HD + kvh * HD + d];
    }
}

// grid (n, NH), block 256, dynamic smem (p0+n) floats
__global__ void k_attn(const float* q, const float* qfull, const float* Kc, const float* Vc, float* out, int p0) {
    extern __shared__ float sc[];
    __shared__ float qs[256], red[256];
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, pos = p0 + t, kvh = h / (NH / NKV);
    const int wave = tid >> 6, lane = tid & 63;
    qs[tid] = q[(size_t)t * NH * HD + h * HD + tid]; __syncthreads();
    for (int s = wave; s <= pos; s += 4) {
        const float* kp = Kc + ((size_t)s * NKV + kvh) * HD;
        float a = 0.f;
        for (int d = lane; d < HD; d += 64) a += qs[d] * kp[d];
        a = ix_warp_reduce_sum(a);
        if (lane == 0) sc[s] = a * 0.0625f;   // 256^-0.5
    }
    __syncthreads();
    float m = -1e30f;
    for (int s = tid; s <= pos; s += 256) m = fmaxf(m, sc[s]);
    m = bmax(m, red, 256);
    float sum = 0.f;
    for (int s = tid; s <= pos; s += 256) { const float e = expf(sc[s] - m); sc[s] = e; sum += e; }
    sum = bsum(sum, red, 256);   // also orders the sc[] writes before the reads below
    float acc = 0.f;
    for (int s = 0; s <= pos; ++s) acc += sc[s] * Vc[((size_t)s * NKV + kvh) * HD + tid];
    acc /= sum;
    const float g = qfull[(size_t)t * NH * 2 * HD + h * 2 * HD + HD + tid];
    out[(size_t)t * NH * HD + h * HD + tid] = acc / (1.f + expf(-g));
}

__global__ void k_router(const float* rl, int* idx, float* wt) {
    const int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float p[NE], red[256];
    const float v = rl[t * NE + tid];
    const float m = bmax(v, red, 256);
    const float e = expf(v - m);
    const float s = bsum(e, red, 256);
    p[tid] = e / s; __syncthreads();
    if (tid == 0) {
        int si[TOPK]; float sel[TOPK]; float tot = 0.f;
        for (int k = 0; k < TOPK; ++k) {
            int bi = 0; float bv = -1.f;
            for (int i = 0; i < NE; ++i) if (p[i] > bv) { bv = p[i]; bi = i; }
            si[k] = bi; sel[k] = bv; tot += bv; p[bi] = -2.f;
        }
        for (int k = 0; k < TOPK; ++k) { idx[t * TOPK + k] = si[k]; wt[t * TOPK + k] = sel[k] / tot; }
    }
}

// grid (MI/4, n*TOPK): eh[(t,k), r] = silu(gate_r . x) * (up_r . x)
__global__ void k_moe_gu(const uint16_t* GU, const float* xn, const int* idx, float* eh) {
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const int r = blockIdx.x * 4 + wave, by = blockIdx.y, t = by / TOPK;
    if (r >= MI) return;
    const size_t e = (size_t)idx[by];
    const uint32_t* wg = (const uint32_t*)(GU + (e * 2 * MI + r) * H);
    const uint32_t* wu = (const uint32_t*)(GU + (e * 2 * MI + MI + r) * H);
    const float* x = xn + (size_t)t * H;
    float g = 0.f, u = 0.f;
    for (int c = lane; c < H / 2; c += 64) {
        const float2 xv = *(const float2*)(x + 2 * c);
        const uint32_t a = wg[c], b = wu[c];
        g += __uint_as_float(a << 16) * xv.x + __uint_as_float(a & 0xffff0000u) * xv.y;
        u += __uint_as_float(b << 16) * xv.x + __uint_as_float(b & 0xffff0000u) * xv.y;
    }
    g = ix_warp_reduce_sum(g); u = ix_warp_reduce_sum(u);
    if (lane == 0) eh[(size_t)by * MI + r] = siluf(g) * u;
}

// grid (H/4, n*TOPK): ebuf[(t,k), row] = down[e][row,:] . eh[(t,k),:]
__global__ void k_moe_down(const uint16_t* DN, const float* eh, const int* idx, float* ebuf) {
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const int row = blockIdx.x * 4 + wave, by = blockIdx.y;
    if (row >= H) return;
    const size_t e = (size_t)idx[by];
    const uint32_t* w = (const uint32_t*)(DN + (e * H + row) * MI);
    const float* x = eh + (size_t)by * MI;
    float a = 0.f;
    for (int c = lane; c < MI / 2; c += 64) {
        const float2 xv = *(const float2*)(x + 2 * c);
        const uint32_t v = w[c];
        a += __uint_as_float(v << 16) * xv.x + __uint_as_float(v & 0xffff0000u) * xv.y;
    }
    a = ix_warp_reduce_sum(a);
    if (lane == 0) ebuf[(size_t)by * H + row] = a;
}

__global__ void k_silu_mul(const float* g, const float* u, float* o, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) o[i] = siluf(g[i]) * u[i];
}

__global__ void k_moe_combine(float* x, const float* ebuf, const float* wt, const float* sh_o, const float* sgate) {
    const int t = blockIdx.x;
    const float sg = 1.f / (1.f + expf(-sgate[t]));
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float a = 0.f;
        for (int k = 0; k < TOPK; ++k) a += wt[t * TOPK + k] * ebuf[((size_t)t * TOPK + k) * H + i];
        x[(size_t)t * H + i] += a + sg * sh_o[(size_t)t * H + i];
    }
}

// ============================================================ safetensors
struct Tensor { std::string dtype; std::vector<int64_t> shape; const uint8_t* p = nullptr; size_t bytes = 0;
    size_t numel() const { size_t n = 1; for (auto d : shape) n *= (size_t)d; return n; } };
std::map<std::string, Tensor> g_t;

struct JS {
    const char *p, *e;
    void ws() { while (p < e && isspace((unsigned char)*p)) ++p; }
    bool eat(char c) { ws(); if (p < e && *p == c) { ++p; return true; } return false; }
    std::string str() { ws(); ++p; std::string r; while (p < e && *p != '"') { if (*p == '\\') ++p; r += *p++; } ++p; return r; }
    int64_t num() { ws(); char* q; long long v = strtoll(p, &q, 10); p = q; return v; }
    std::vector<int64_t> arr() {
        std::vector<int64_t> r; eat('['); ws();
        if (*p == ']') { ++p; return r; }
        for (;;) { r.push_back(num()); ws(); if (*p == ',') { ++p; continue; } ++p; break; }
        return r;
    }
    void skip() {
        ws();
        if (*p == '"') { str(); }
        else if (*p == '{' || *p == '[') {
            const char o = *p, c = o == '{' ? '}' : ']'; ++p; ws();
            if (*p == c) { ++p; return; }
            for (;;) { if (o == '{') { str(); eat(':'); } skip(); ws(); if (*p == ',') { ++p; continue; } ++p; break; }
        } else { while (p < e && *p != ',' && *p != '}' && *p != ']') ++p; }
    }
};

void load_shard(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror(path.c_str()); exit(2); }
    struct stat st; fstat(fd, &st);
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); exit(2); }
    close(fd);
    const uint8_t* base = (const uint8_t*)m;
    uint64_t n; memcpy(&n, base, 8);
    const uint8_t* data = base + 8 + n;
    JS j{(const char*)base + 8, (const char*)base + 8 + n};
    j.eat('{');
    for (;;) {
        const std::string key = j.str(); j.eat(':');
        if (key == "__metadata__") j.skip();
        else {
            Tensor t; std::vector<int64_t> off;
            j.eat('{');
            for (;;) {
                const std::string k2 = j.str(); j.eat(':');
                if (k2 == "dtype") t.dtype = j.str();
                else if (k2 == "shape") t.shape = j.arr();
                else if (k2 == "data_offsets") off = j.arr();
                else j.skip();
                if (j.eat(',')) continue;
                j.eat('}'); break;
            }
            if (off.size() != 2) { fprintf(stderr, "bad header for %s\n", key.c_str()); exit(2); }
            t.p = data + off[0]; t.bytes = (size_t)(off[1] - off[0]);
            g_t[key] = t;
        }
        if (j.eat(',')) continue;
        break;
    }
}

const Tensor& get(const std::string& name) {
    auto it = g_t.find(name);
    if (it == g_t.end()) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(2); }
    return it->second;
}

float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf((float)(m + 1024), (int)e - 25);
    return s ? -v : v;
}

void need_shape(const std::string& name, const std::vector<int64_t>& s) {
    const Tensor& t = get(name);
    if (t.shape != s) { fprintf(stderr, "tensor %s has unexpected shape\n", name.c_str()); exit(2); }
}

uint16_t* up_bf16(const std::string& name, const std::vector<int64_t>& shape) {
    need_shape(name, shape);
    const Tensor& t = get(name);
    if (t.dtype != "BF16") { fprintf(stderr, "tensor %s is %s, expected BF16\n", name.c_str(), t.dtype.c_str()); exit(2); }
    void* d; CK(cudaMalloc(&d, t.bytes));
    const size_t chunk = (size_t)64 << 20;
    for (size_t o = 0; o < t.bytes; o += chunk) CK(cudaMemcpy((char*)d + o, t.p + o, std::min(chunk, t.bytes - o), cudaMemcpyHostToDevice));
    return (uint16_t*)d;
}

float* up_f32(const std::string& name, size_t count) {
    const Tensor& t = get(name);
    if (t.numel() != count) { fprintf(stderr, "tensor %s has %zu elements, expected %zu\n", name.c_str(), t.numel(), count); exit(2); }
    std::vector<float> h(count);
    for (size_t i = 0; i < count; ++i) {
        if (t.dtype == "BF16") { uint16_t b; memcpy(&b, t.p + 2 * i, 2); uint32_t u = (uint32_t)b << 16; memcpy(&h[i], &u, 4); }
        else if (t.dtype == "F16") { uint16_t b; memcpy(&b, t.p + 2 * i, 2); h[i] = h2f(b); }
        else if (t.dtype == "F32") memcpy(&h[i], t.p + 4 * i, 4);
        else { fprintf(stderr, "tensor %s: unsupported dtype %s\n", name.c_str(), t.dtype.c_str()); exit(2); }
    }
    float* d; CK(cudaMalloc(&d, count * 4)); CK(cudaMemcpy(d, h.data(), count * 4, cudaMemcpyHostToDevice));
    return d;
}

// ============================================================ model state
struct LW {
    bool full = false;
    float *in_ln = 0, *post_ln = 0;
    uint16_t *qkv = 0, *z = 0, *a = 0, *b = 0, *outp = 0; float *conv = 0, *Alog = 0, *dtb = 0, *gnorm = 0;
    uint16_t *q = 0, *k = 0, *v = 0, *o = 0; float *qn = 0, *kn = 0;
    uint16_t *gate = 0, *gu = 0, *dn = 0, *sg = 0, *su = 0, *sd = 0, *seg = 0;
    float *S = 0, *cs = 0, *Kc = 0, *Vc = 0;
};
struct Dev {
    int id = 0;
    float *x, *xn, *big, *big2, *kb, *vb, *ab, *bb, *gdn_o, *qo, *attn_o, *rl, *wt, *eh, *ebuf, *shg, *shu, *sho, *sgate;
    int *idx, *tok;
};
LW g_L[NL];
Dev g_D[8];
int g_ngpu = 1, g_maxctx = 0;
uint16_t *g_embed = 0, *g_head = 0; float* g_fnorm = 0; float* g_logits = 0;
int gpu_of(int l) { return l * g_ngpu / NL; }

template <class T> void dalloc(T*& p, size_t count) { CK(cudaMalloc((void**)&p, count * sizeof(T))); }

void gemv(const uint16_t* W, const float* x, float* y, int rows, int cols, int n, int ystride, int accum) {
    k_gemv<<<(rows + 3) / 4, 256>>>(W, x, y, rows, cols, n, ystride, accum);
}

void layer_fwd(int l, int n, int p0) {
    const LW& L = g_L[l]; Dev& d = g_D[gpu_of(l)];
    CK(cudaSetDevice(d.id));
    k_rmsnorm<<<n, 256>>>(d.x, L.in_ln, d.xn, H, 1);
    if (L.full) {
        gemv(L.q, d.xn, d.big, NH * 2 * HD, H, n, NH * 2 * HD, 0);
        gemv(L.k, d.xn, d.kb, NKV * HD, H, n, NKV * HD, 0);
        gemv(L.v, d.xn, d.vb, NKV * HD, H, n, NKV * HD, 0);
        k_attn_prep<<<dim3(n, NH + NKV), 256>>>(d.big, d.kb, d.vb, L.qn, L.kn, d.qo, L.Kc, L.Vc, p0);
        k_attn<<<dim3(n, NH), 256, (size_t)(p0 + n) * sizeof(float)>>>(d.qo, d.big, L.Kc, L.Vc, d.attn_o, p0);
        gemv(L.o, d.attn_o, d.x, H, NH * HD, n, H, 1);
    } else {
        gemv(L.qkv, d.xn, d.big, QKV, H, n, QKV, 0);
        gemv(L.z, d.xn, d.big2, VALD, H, n, VALD, 0);
        gemv(L.a, d.xn, d.ab, LVH, H, n, LVH, 0);
        gemv(L.b, d.xn, d.bb, LVH, H, n, LVH, 0);
        k_gdn_conv<<<QKV / 256, 256>>>(d.big, L.conv, L.cs, n);
        k_gdn_step<<<LVH, LD>>>(d.big, d.ab, d.bb, L.Alog, L.dtb, L.S, d.gdn_o, n);
        k_gdn_norm_gate<<<n * LVH, LD>>>(d.gdn_o, d.big2, L.gnorm);
        gemv(L.outp, d.gdn_o, d.x, H, VALD, n, H, 1);
    }
    k_rmsnorm<<<n, 256>>>(d.x, L.post_ln, d.xn, H, 1);
    gemv(L.gate, d.xn, d.rl, NE, H, n, NE, 0);
    k_router<<<n, 256>>>(d.rl, d.idx, d.wt);
    k_moe_gu<<<dim3(MI / 4, n * TOPK), 256>>>(L.gu, d.xn, d.idx, d.eh);
    k_moe_down<<<dim3(H / 4, n * TOPK), 256>>>(L.dn, d.eh, d.idx, d.ebuf);
    gemv(L.sg, d.xn, d.shg, MI, H, n, MI, 0);
    gemv(L.su, d.xn, d.shu, MI, H, n, MI, 0);
    k_silu_mul<<<(n * MI + 255) / 256, 256>>>(d.shg, d.shu, d.shg, n * MI);
    gemv(L.sd, d.shg, d.sho, H, MI, n, H, 0);
    gemv(L.seg, d.xn, d.sgate, 1, H, n, 1, 0);
    k_moe_combine<<<n, 256>>>(d.x, d.ebuf, d.wt, d.sho, d.sgate);
}

void hop(int da, const float* pa, int db, float* pb, size_t bytes) {
    static std::vector<char> st; st.resize(bytes);
    CK(cudaSetDevice(da)); CK(cudaGetLastError()); CK(cudaMemcpy(st.data(), pa, bytes, cudaMemcpyDeviceToHost));
    CK(cudaSetDevice(db)); CK(cudaMemcpy(pb, st.data(), bytes, cudaMemcpyHostToDevice));
}

// Runs n tokens at positions p0..p0+n-1; if logits != nullptr writes the logits row of the last token.
void forward(const int* toks, int n, int p0, std::vector<float>* logits) {
    if (n < 1 || n > NBM || p0 + n > g_maxctx) { fprintf(stderr, "forward: bad n/p0\n"); exit(1); }
    Dev& d0 = g_D[0];
    CK(cudaSetDevice(d0.id));
    CK(cudaMemcpy(d0.tok, toks, n * sizeof(int), cudaMemcpyHostToDevice));
    k_embed<<<n, 256>>>(g_embed, d0.tok, d0.x);
    for (int l = 0; l < NL; ++l) {
        if (l > 0 && gpu_of(l) != gpu_of(l - 1)) hop(gpu_of(l - 1), g_D[gpu_of(l - 1)].x, gpu_of(l), g_D[gpu_of(l)].x, (size_t)n * H * 4);
        layer_fwd(l, n, p0);
    }
    Dev& dl = g_D[gpu_of(NL - 1)];
    CK(cudaSetDevice(dl.id));
    if (logits) {
        k_rmsnorm<<<1, 256>>>(dl.x + (size_t)(n - 1) * H, g_fnorm, dl.xn, H, 1);
        gemv(g_head, dl.xn, g_logits, VOC, H, 1, VOC, 0);
        logits->resize(VOC);
        CK(cudaMemcpy(logits->data(), g_logits, (size_t)VOC * 4, cudaMemcpyDeviceToHost));
    }
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
}

void reset_state() {
    for (int l = 0; l < NL; ++l) {
        if (g_L[l].full) continue;
        CK(cudaSetDevice(g_D[gpu_of(l)].id));
        CK(cudaMemset(g_L[l].S, 0, (size_t)LVH * LD * LD * 4));
        CK(cudaMemset(g_L[l].cs, 0, (size_t)QKV * 3 * 4));
    }
}

// ============================================================ loading
int cfg_int(const std::string& txt, const char* key) {
    const std::string k = std::string("\"") + key + "\":";
    const size_t p = txt.find(k);
    if (p == std::string::npos) { fprintf(stderr, "config.json: key %s missing\n", key); exit(2); }
    return atoi(txt.c_str() + p + k.size());
}

void load_model(const std::string& dir, int maxctx) {
    g_maxctx = maxctx;
    FILE* f = fopen((dir + "/config.json").c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot read %s/config.json\n", dir.c_str()); exit(2); }
    std::string txt; char buf[4096]; size_t r;
    while ((r = fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, r);
    fclose(f);
    // strip whitespace so "key": value matches
    std::string c; for (char ch : txt) if (!isspace((unsigned char)ch)) c += ch;
    struct { const char* k; int v; } chk[] = {{"hidden_size", H}, {"num_hidden_layers", NL}, {"vocab_size", VOC}, {"num_experts", NE},
        {"num_experts_per_tok", TOPK}, {"moe_intermediate_size", MI}, {"shared_expert_intermediate_size", MI}, {"num_attention_heads", NH},
        {"num_key_value_heads", NKV}, {"head_dim", HD}, {"linear_num_key_heads", LKH}, {"linear_num_value_heads", LVH},
        {"linear_key_head_dim", LD}, {"linear_value_head_dim", LD}, {"linear_conv_kernel_dim", 4}, {"full_attention_interval", 4}};
    for (auto& e : chk) if (cfg_int(c, e.k) != e.v) { fprintf(stderr, "config mismatch: %s = %d, this backend is built for %d\n", e.k, cfg_int(c, e.k), e.v); exit(2); }

    DIR* dp = opendir(dir.c_str());
    if (!dp) { perror("opendir"); exit(2); }
    std::vector<std::string> shards;
    while (dirent* de = readdir(dp)) { const std::string n = de->d_name; if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors") shards.push_back(dir + "/" + n); }
    closedir(dp);
    std::sort(shards.begin(), shards.end());
    for (auto& s : shards) load_shard(s);

    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    g_ngpu = std::min(ndev, 4);
    if (g_ngpu < 1) { fprintf(stderr, "no GPU\n"); exit(2); }
    fprintf(stderr, "loading %zu tensors from %zu shards onto %d GPU(s)\n", g_t.size(), shards.size(), g_ngpu);

    for (int g = 0; g < g_ngpu; ++g) {
        Dev& d = g_D[g]; d.id = g; CK(cudaSetDevice(g));
        dalloc(d.x, (size_t)NBM * H); dalloc(d.xn, (size_t)NBM * H); dalloc(d.big, (size_t)NBM * NH * 2 * HD); dalloc(d.big2, (size_t)NBM * VALD);
        dalloc(d.kb, (size_t)NBM * NKV * HD); dalloc(d.vb, (size_t)NBM * NKV * HD); dalloc(d.ab, (size_t)NBM * LVH); dalloc(d.bb, (size_t)NBM * LVH);
        dalloc(d.gdn_o, (size_t)NBM * VALD); dalloc(d.qo, (size_t)NBM * NH * HD); dalloc(d.attn_o, (size_t)NBM * NH * HD);
        dalloc(d.rl, (size_t)NBM * NE); dalloc(d.wt, (size_t)NBM * TOPK); dalloc(d.idx, (size_t)NBM * TOPK);
        dalloc(d.eh, (size_t)NBM * TOPK * MI); dalloc(d.ebuf, (size_t)NBM * TOPK * H);
        dalloc(d.shg, (size_t)NBM * MI); dalloc(d.shu, (size_t)NBM * MI); dalloc(d.sho, (size_t)NBM * H); dalloc(d.sgate, (size_t)NBM);
        dalloc(d.tok, (size_t)NBM);
    }
    const std::string P = "model.language_model.";
    CK(cudaSetDevice(0));
    g_embed = up_bf16(P + "embed_tokens.weight", {VOC, H});
    for (int l = 0; l < NL; ++l) {
        LW& L = g_L[l]; CK(cudaSetDevice(gpu_of(l)));
        const std::string p = P + "layers." + std::to_string(l) + ".";
        L.full = g_t.count(p + "self_attn.q_proj.weight") > 0;
        if (L.full != (l % 4 == 3)) { fprintf(stderr, "layer %d: layer type disagrees with full_attention_interval\n", l); exit(2); }
        L.in_ln = up_f32(p + "input_layernorm.weight", H);
        L.post_ln = up_f32(p + "post_attention_layernorm.weight", H);
        if (L.full) {
            L.q = up_bf16(p + "self_attn.q_proj.weight", {NH * 2 * HD, H});
            L.k = up_bf16(p + "self_attn.k_proj.weight", {NKV * HD, H});
            L.v = up_bf16(p + "self_attn.v_proj.weight", {NKV * HD, H});
            L.o = up_bf16(p + "self_attn.o_proj.weight", {H, NH * HD});
            L.qn = up_f32(p + "self_attn.q_norm.weight", HD);
            L.kn = up_f32(p + "self_attn.k_norm.weight", HD);
            dalloc(L.Kc, (size_t)g_maxctx * NKV * HD); dalloc(L.Vc, (size_t)g_maxctx * NKV * HD);
        } else {
            L.qkv = up_bf16(p + "linear_attn.in_proj_qkv.weight", {QKV, H});
            L.z = up_bf16(p + "linear_attn.in_proj_z.weight", {VALD, H});
            L.a = up_bf16(p + "linear_attn.in_proj_a.weight", {LVH, H});
            L.b = up_bf16(p + "linear_attn.in_proj_b.weight", {LVH, H});
            L.outp = up_bf16(p + "linear_attn.out_proj.weight", {H, VALD});
            L.conv = up_f32(p + "linear_attn.conv1d.weight", (size_t)QKV * 4);
            L.Alog = up_f32(p + "linear_attn.A_log", LVH);
            L.dtb = up_f32(p + "linear_attn.dt_bias", LVH);
            L.gnorm = up_f32(p + "linear_attn.norm.weight", LD);
            dalloc(L.S, (size_t)LVH * LD * LD); dalloc(L.cs, (size_t)QKV * 3);
        }
        L.gate = up_bf16(p + "mlp.gate.weight", {NE, H});
        L.gu = up_bf16(p + "mlp.experts.gate_up_proj", {NE, 2 * MI, H});
        L.dn = up_bf16(p + "mlp.experts.down_proj", {NE, H, MI});
        L.sg = up_bf16(p + "mlp.shared_expert.gate_proj.weight", {MI, H});
        L.su = up_bf16(p + "mlp.shared_expert.up_proj.weight", {MI, H});
        L.sd = up_bf16(p + "mlp.shared_expert.down_proj.weight", {H, MI});
        L.seg = up_bf16(p + "mlp.shared_expert_gate.weight", {1, H});
        if (l % 8 == 7) fprintf(stderr, "  layer %d/%d loaded\n", l + 1, NL);
    }
    CK(cudaSetDevice(gpu_of(NL - 1)));
    g_fnorm = up_f32(P + "norm.weight", H);
    g_head = up_bf16("lm_head.weight", {VOC, H});
    dalloc(g_logits, (size_t)VOC);
}
}  // namespace

// ============================================================ driver
namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// argmax with lowest-index tie-break; returns -1 if any logit is non-finite.
int argmax_checked(const std::vector<float>& v) {
    int best = 0;
    for (int i = 0; i < (int)v.size(); ++i) { if (!std::isfinite(v[i])) return -1; if (v[i] > v[best]) best = i; }
    return best;
}

void prefill(const std::vector<int>& toks, int chunk, std::vector<float>& logits) {
    const int P = (int)toks.size();
    for (int p0 = 0; p0 < P; p0 += chunk) {
        const int n = std::min(chunk, P - p0);
        forward(&toks[p0], n, p0, p0 + n == P ? &logits : nullptr);
    }
}

std::vector<int> make_prompt(int n) {
    std::vector<int> t(n); uint32_t s = 12345u;
    for (int i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u; t[i] = 1000 + (int)((s >> 8) % 100000u); }
    return t;
}
}  // namespace

int main(int argc, char** argv) {
    std::string model = "/tmp/model"; int P = 512, G = 64; bool bench = false; std::vector<int> ids;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) P = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) G = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) { ids.clear(); for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) ids.push_back(atoi(t)); P = (int)ids.size(); }
        else if (!strcmp(argv[i], "--bench")) bench = true;
        else { fprintf(stderr, "usage: generate_model --model DIR [--bench] [--prompt N] [--gen N]\n"); return 2; }
    }
    if (P < 1 || G < 1 || P + G + 8 > 8192) { fprintf(stderr, "need 1 <= prompt, gen and prompt+gen+8 <= 8192\n"); return 2; }
    (void)bench;
    const int maxctx = std::max(P + G + 8, 64);

    auto t0 = Clock::now();
    load_model(model, maxctx);
    fprintf(stderr, "load: %.1f s\n", since(t0));

    // ---- invariant: chunked prefill == token-by-token prefill (same weights, same math, different batching)
    {
        const std::vector<int> pr = make_prompt(24);
        std::vector<float> a, b;
        reset_state(); prefill(pr, NBM, a);
        reset_state(); prefill(pr, 1, b);
        const int ia = argmax_checked(a), ib = argmax_checked(b);
        if (ia < 0 || ib < 0) { fprintf(stderr, "selfcheck FAILED: non-finite logits\n"); return 4; }
        double num = 0, den = 0;
        for (int i = 0; i < VOC; ++i) { const double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        const double rel = std::sqrt(num / (den + 1e-30));
        fprintf(stderr, "selfcheck: chunked vs sequential prefill rel L2 = %.3e, argmax %d vs %d\n", rel, ia, ib);
        if (ia != ib || rel > 1e-2) { fprintf(stderr, "selfcheck FAILED\n"); return 5; }
    }

    const std::vector<int> prompt = ids.empty() ? make_prompt(P) : ids;
    std::vector<float> logits;
    reset_state(); prefill(prompt, NBM, logits);   // warm-up pass over the same shapes
    reset_state();

    t0 = Clock::now();
    prefill(prompt, NBM, logits);
    const double t_pre = since(t0);
    int cur = argmax_checked(logits);
    if (cur < 0) { fprintf(stderr, "non-finite logits after prefill\n"); return 4; }

    std::vector<int> gen;
    t0 = Clock::now();
    for (int g = 0; g < G; ++g) {
        gen.push_back(cur);
        forward(&cur, 1, P + g, &logits);
        cur = argmax_checked(logits);
        if (cur < 0) { fprintf(stderr, "non-finite logits at decode step %d\n", g); return 4; }
    }
    const double t_dec = since(t0);

    printf("generated:");
    for (int i = 0; i < (int)gen.size(); ++i) printf(" %d", gen[i]);
    printf("\n");
    printf("prefill: %.2f tok/s\n", P / t_pre);
    printf("decode: %.2f tok/s\n", G / t_dec);
    return 0;
}
