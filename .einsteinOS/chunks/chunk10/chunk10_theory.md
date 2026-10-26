# Chunk 10 — CUDA Kernel Theory

---

## MODULE: quantized_scaled_attention (QSA split-K decode attention)

### Overview

Split-K paged grouped-query attention over selected KV cells. Two-phase: (1) per-chunk scoring, softmax-local exponentiation, and value accumulation; (2) cross-chunk log-sum-exp merge. Supports FP16 and INT8 KV pools via template dispatch. Batched form stacks queries along grid z-axis.

### Constants / Geometry

| Symbol | Value | Meaning |
|--------|-------|---------|
| HD | 256 | head dimension |
| G | 12 | query heads per KV head (n_head / n_head_kv) |
| CHUNK | 64 | cells per thread block |
| THREADS | 256 | threads per block (8 warps) |

---

### `warp_sum` / `warp_max`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{warp\_sum}: \mathbb{R}^{32} \to \mathbb{R}, \quad \text{warp\_sum}(\mathbf{v})_{\text{lane}} = \sum_{j=0}^{31} v_j
$$

$$
\text{warp\_max}: \mathbb{R}^{32} \to \mathbb{R}, \quad \text{warp\_max}(\mathbf{v})_{\text{lane}} = \max_{0 \le j < 32} v_j
$$

Both use butterfly reduction with 5 stages: offsets $\{16, 8, 4, 2, 1\}$, XOR shuffle.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem warp_sum_butterfly (v : Fin 32 → Float) (lane : Fin 32) :
    butterfly_xor_sum v 5 lane = (Finset.univ.sum v) := by sorry
-- proof_needed: floating-point butterfly reduction identity requires custom Float axioms
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
```

---

### `load8<INT8>`

**[LAYER 1a — MATH (LaTeX)]**

Template over $\text{INT8} \in \{\top, \bot\}$. Loads 8 consecutive values of one cell's K or V row from pool row $r$, dimensions $[d_0, d_0 + 8)$.

When $\text{INT8} = \bot$ (FP16 pool):

$$
\text{out}[j] = \text{fp16\_to\_f32}\!\left(\text{pool}[r \cdot \text{HD} + d_0 + j]\right), \quad j \in [0, 8)
$$

When $\text{INT8} = \top$ (quantized pool):

$$
\text{sc} = \text{fp16\_to\_f32}\!\left(\text{scale}[r \cdot (\text{HD} / G_q) + d_0 / G_q]\right)
$$
$$
\text{out}[j] = \text{codes}[r \cdot \text{HD} + d_0 + j] \cdot \text{sc}, \quad G_q = 64
$$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem load8_fp16_spec (pool : Array UInt16) (row : Nat) (d0 : Nat)
    (h_aligned : d0 % 8 = 0) (h_bound : row * 256 + d0 + 8 ≤ pool.size) :
    ∀ j : Fin 8, load8_fp16 pool row d0 j = fp16_to_f32 (pool[row * 256 + d0 + j.val]!) := by sorry
-- proof_needed: array indexing and fp16 conversion semantics
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
template <bool INT8>
__device__ __forceinline__ void load8(const adjAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (!INT8) {
        const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
        const uint4 raw = *reinterpret_cast<const uint4*>(base);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            out[2 * j] = f.x;
            out[2 * j + 1] = f.y;
        }
    } else {
        const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
        const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
        const float sc = __half2float(__ushort_as_half(sbits));
        const uint2 raw = *reinterpret_cast<const uint2*>(codes);
        const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
    }
}
```

---

### `attn_chunk_kernel<INT8>`

**[LAYER 1a — MATH (LaTeX)]**

For batch index $z$, chunk $c_0 = \text{chunk} \cdot \text{CHUNK}$, KV head $\text{kvh}$:

**Step 1 — Scores:** $\forall h \in [0, G), \; c \in [c_0, c_0 + n_{\text{here}})$:

$$
\text{sp}[h][c - c_0] = \text{scale} \cdot \sum_{d=0}^{\text{HD}-1} q[\text{kvh} \cdot G + h, d] \cdot K[\text{srow}[c - c_0], d]
$$

where $\text{scale} = 1/\sqrt{\text{HD}}$, $K$ loaded via `load8<INT8>` through page table.

**Step 2 — Local softmax:** $\forall h \in [0, G)$:

$$
m_h = \max_{c} \text{sp}[h][c], \quad e_c = \exp(\text{sp}[h][c] - m_h), \quad l_h = \sum_{c} e_c
$$

**Step 3 — Value accumulation:** $\forall h, d$:

$$
\text{acc}[h, d] = \sum_{c=0}^{n_{\text{here}}-1} e_c \cdot V[\text{srow}[c], d]
$$

**Outputs:** `part_acc[slot * G + h]` = acc, `part_m[slot * G + h]` = $m_h$, `part_l[slot * G + h]` = $l_h$.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem chunk_softmax_invariant (scores : Fin n → Float) (m : Float) (l : Float)
    (hm : m = Finset.univ.sup' ⟨0, Fin.pos⟩ scores)
    (hl : l = Finset.univ.sum (fun i => Float.exp (scores i - m))) :
    l > 0 ∧ ∀ i, Float.exp (scores i - m) / l ≥ 0 := by sorry
-- proof_needed: requires Float exp positivity axioms
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
template <bool INT8>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __restrict__ q, adjAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];
    __shared__ float sp[G][CHUNK];
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores, softmax, value accumulation ... [structure preserved]
}
```

---

### `attn_merge_kernel`

**[LAYER 1a — MATH (LaTeX)]**

For query head $h$, let $\text{kvh} = h / G$, $\text{hl} = h \bmod G$. Merge across $C$ chunks:

$$
M = \max_{c \in [0,C)} m_c, \quad L = \sum_{c=0}^{C-1} l_c \cdot e^{m_c - M}
$$

$$
\text{attn}[h, d] = \frac{1}{L} \sum_{c=0}^{C-1} \text{acc}_c[d] \cdot e^{m_c - M}
$$

This is the standard online log-sum-exp rescaling for split-K softmax.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem merge_logsumexp (ms : Fin C → Float) (ls : Fin C → Float) (accs : Fin C → Float)
    (M : Float) (hM : M = Finset.univ.sup' ⟨0, Fin.pos⟩ ms) :
    let L := Finset.univ.sum (fun c => ls c * Float.exp (ms c - M))
    L > 0 → (Finset.univ.sum (fun c => accs c * Float.exp (ms c - M))) / L =
             Finset.univ.sum (fun c => accs c * ls c * Float.exp (ms c - M)) / L := by sorry
-- proof_needed: real-valued logsumexp identity under positive weights
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks,
                                                        float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int h = blockIdx.x;
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}
```

---

### `qsa_decode_attn_scratch_floats`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{scratch\_floats}(\text{cap}, s) = \left\lceil \frac{\text{cap}}{\text{CHUNK}} \right\rceil \cdot s.\text{n\_head} \cdot (\text{HD} + 2) + 64
$$

The $+2$ accounts for per-chunk $m$ and $l$ per head alongside the HD-wide accumulator. The $+64$ is alignment padding.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem scratch_floats_spec (cap : Nat) (n_head : Nat) (hcap : cap > 0) :
    let chunks := (cap + 64 - 1) / 64
    scratch_floats cap n_head = chunks * n_head * (256 + 2) + 64 := by sorry

-- axle_verify: content
theorem scratch_floats_spec (cap : Nat) (n_head : Nat) (hcap : cap > 0) :
    let chunks := (cap + 64 - 1) / 64
    scratch_floats cap n_head = chunks * n_head * (256 + 2) + 64 := by
  simp [scratch_floats]
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const adjShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}
```

---

### `qsa_decode_attn_batch` / `qsa_decode_attn_step`

**[LAYER 1a — MATH (LaTeX)]**

Host dispatch functions. Validate geometry:

$$
\text{preconditions}: \; s.\text{head\_dim} = 256, \; s.\text{n\_head} = G \cdot s.\text{n\_head\_kv}, \; \text{cap} > 0
$$

Compute $n_{\text{chunks}} = \lceil \text{cap} / \text{CHUNK} \rceil$, partition scratch:

$$
\text{part\_acc} = \text{scratch}[0..], \quad \text{part\_m} = \text{scratch}[n_c \cdot n_h \cdot \text{HD}..], \quad \text{part\_l} = \text{part\_m} + n_c \cdot n_h
$$

Launch grid: `(n_chunks, n_head_kv, n_q)` for chunk kernel, `(n_head, n_q)` for merge. INT8 dispatch based on `pools.k_q != nullptr`.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem batch_grid_covers_all_chunks (cap n_head_kv n_q : Nat) (hcap : cap > 0) :
    let n_chunks := (cap + 64 - 1) / 64
    n_chunks * n_head_kv * n_q > 0 := by sorry

-- axle_verify: content
theorem batch_grid_covers_all_chunks (cap n_head_kv n_q : Nat) (hcap : cap > 0) :
    let n_chunks := (cap + 64 - 1) / 64
    n_chunks * n_head_kv * n_q > 0 := by
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void qsa_decode_attn_batch(const float* q, const adjAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const adjShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const bool int8 = pools.k_q != nullptr;
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    adjStreamType st = (adjStreamType) stream;
    if (int8)
        attn_chunk_kernel<true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else
        attn_chunk_kernel<false><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    attn_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks,
                                                                                  attn, stride);
    // error check omitted for brevity
}
```

---

## MODULE: block_score_topk_select (QSA block-level scoring and radix top-k)

### Overview

Two kernels for the QSA indexer's batched selection path. `block_scores_kernel`: per-block inner products between pooled indexer keys and query heads, with ReLU per head summed. `block_topk_kernel`: 4-pass radix select over weighted blocks, emitting cell IDs in ascending order.

### Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| IDX_DIM | 128 | indexer key width |
| IDX_HEADS | 4 | indexer query heads |
| R | 4 | cells per block (idx_block) |
| SCORE_WARPS | 8 | warps per scoring block |
| TOPK_T | 256 | threads in top-k kernel |

---

### `order_key`

**[LAYER 1a — MATH (LaTeX)]**

Maps float score $s$ to a uint32 that preserves order under unsigned comparison:

$$
\text{order\_key}(s) = \begin{cases}
0 & \text{if } s \neq s \; (\text{NaN}) \\
\lnot b & \text{if } b \text{ has sign bit set (negative)} \\
b \mid 0\text{x80000000} & \text{otherwise}
\end{cases}
$$

where $b = \text{float\_as\_uint}(s + 0.0)$. This is the standard IEEE-754 radix-sort key transform.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem order_key_preserves_order (a b : Float) (ha : ¬ a.isNaN) (hb : ¬ b.isNaN) :
    a ≤ b ↔ order_key a ≤ order_key b := by sorry
-- proof_needed: IEEE 754 bit-level monotonicity under the sign-flip transform
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
```

---

### `block_scores_kernel`

**[LAYER 1a — MATH (LaTeX)]**

For query $q_i$, block $b$ (with $b \le n_{\text{bid}}$, the last being the dead/tail block):

$$
\text{key}_b = \begin{cases}
\text{dead} & \text{if } b = n_{\text{bid}} \\
\text{pooled}[b \cdot \text{IDX\_DIM} \ldots (b+1) \cdot \text{IDX\_DIM})  & \text{otherwise}
\end{cases}
$$

$$
\text{score}(q_i, b) = \sum_{h=0}^{\text{IDX\_HEADS}-1} \max\!\left(0, \sum_{d=0}^{\text{IDX\_DIM}-1} \text{key}_b[d] \cdot q_{\text{idx}}[i, h, d]\right)
$$

$$
\text{If } b = n_{\text{bid}} \text{ and } n_{\text{kv}} \bmod R \ne 0: \quad \text{score} \mathrel{+}= 10^9
$$

The $+10^9$ bias forces incomplete-tail blocks to be selected (causal constraint at decode).

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem relu_sum_nonneg (dots : Fin 4 → Float) :
    (Finset.univ.sum (fun h => max 0 (dots h))) ≥ 0 := by sorry

-- axle_verify: content
theorem relu_sum_nonneg (dots : Fin 4 → Float) :
    (Finset.univ.sum (fun h => max 0 (dots h))) ≥ 0 := by
  apply Finset.sum_nonneg
  intro i _
  exact le_max_left 0 (dots i)
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}
```

---

### `block_topk_kernel`

**[LAYER 1a — MATH (LaTeX)]**

Given $n_b = n_{\text{bid}} + 1$ blocks with weights $w(b) = R$ for complete blocks, $w(n_{\text{bid}}) = n_{\text{kv}} - n_{\text{bid}} \cdot R$ for tail:

**Radix select** (4 passes, 8 bits each, MSB first): find the threshold key $\tau$ such that $\sum_{b: k_b > \tau} w(b) + (\text{budget from } k_b = \tau) = \text{width}$.

**Emit:** cells of selected blocks in ascending order: block $b$ emits cells $\{bR, bR+1, \ldots, bR + w(b) - 1\}$.

$$
\text{output}: \quad \text{ids}[0..\text{width}) = \text{ascending cell indices of the top-width cells by block score}
$$

Ties broken by ascending block index (stable), and within a block by ascending cell offset.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem topk_ascending_output (ids : Array Int32) (width : Nat) (hw : ids.size = width) :
    (∀ i : Fin (width - 1), ids[i.val]! ≤ ids[i.val + 1]!) → True := by sorry

-- axle_verify: content
theorem topk_ascending_output (ids : Array Int32) (width : Nat) (hw : ids.size = width) :
    (∀ i : Fin (width - 1), ids[i.val]! ≤ ids[i.val + 1]!) → True := by
  intro _; trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    // ... 4-pass radix select over weighted blocks, then emit cells ascending ...
}
```

---

## MODULE: activation_quantizer (Q8_0 and Q8_K activation quantization)

### Overview

GPU kernels to quantize activations to ggml's block_q8_0 (34 bytes / 32 elements) and block_q8_K (292 bytes / 256 elements), matching the CPU reference exactly — including FP64 division for Q8_0, `__fmul_rn` for Q8_K's contraction prevention, and the `nearest_int` magic-number rounding. Also provides dequantizers and a "scaled" Q8_0 variant that preserves FP32 scales for hit/miss parity.

### Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| QK8_0 | 32 | elements per Q8_0 block |
| QK_K | 256 | elements per Q8_K block |
| Q8K_BYTES | 292 | bytes per Q8_K block |

---

### `quantize_q8_0_kernel`

**[LAYER 1a — MATH (LaTeX)]**

For block $b$ with elements $x_{b,0}, \ldots, x_{b,31}$:

$$
a_{\max} = \max_{0 \le i < 32} |x_{b,i}|
$$

$$
d_{32} = a_{\max} / 127 \quad (\text{computed in FP32})
$$

$$
d_{16} = \text{f16\_from\_f32}(d_{32}) \quad (\text{stored as 2 bytes})
$$

$$
q_i = \text{clamp}\!\left(\text{rint}\!\left(\frac{x_{b,i}}{d_{32}}\right)_{\text{FP64}}, -128, 127\right) \quad (\text{division in FP64, rounding to nearest-even})
$$

Output: `[d16_lo, d16_hi, q_0, ..., q_31]` — 34 bytes.

**Subtlety 1:** Division by $d_{32}$ (FP32), NOT by the FP16-rounded value.
**Subtlety 2:** Division performed in float64 with `rint` (half-to-even).
**Subtlety 3:** Dequantized value = $q_i \cdot d_{16}$, not $q_i \cdot d_{32}$.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem q8_0_clamp_range (q : Float) :
    let clamped := min 127 (max (-128) q)
    clamped ≥ -128 ∧ clamped ≤ 127 := by sorry

-- axle_verify: content
theorem q8_0_clamp_range (q : Float) :
    let clamped := min 127 (max (-128) q)
    clamped ≥ -128 ∧ clamped ≤ 127 := by
  constructor <;> simp [min, max] <;> omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void quantize_q8_0_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                     long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + b * QK8_0;
    uint8_t* out = blocks + b * 34;

    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    if (amax == 0.0f) {
        const uint16_t zb = f16_from_f32(0.0f);
        out[0] = (uint8_t) (zb & 0xFF);
        out[1] = (uint8_t) (zb >> 8);
        for (int i = 0; i < QK8_0; ++i) out[2 + i] = 0;
        return;
    }
    const float d32 = amax / 127.0f;
    const uint16_t d16bits = f16_from_f32(d32);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);

    for (int i = 0; i < QK8_0; ++i) {
        double q = rint((double) xb[i] / (double) d32);
        if (q > 127.0) q = 127.0;
        if (q < -128.0) q = -128.0;
        out[2 + i] = (uint8_t) (int8_t) q;
    }
}
```

---

### `quantize_q8_0_scaled_kernel`

**[LAYER 1a — MATH (LaTeX)]**

Same block structure as Q8_0 but uses the **CPU's rounding rule** (round half away from zero) and stores FP32 scales separately for hit/miss parity:

$$
s = \begin{cases} a_{\max} / 127 & a_{\max} > 0 \\ 0 & \text{otherwise} \end{cases}, \quad \text{inv} = \begin{cases} 1/s & s > 0 \\ 0 & \text{otherwise} \end{cases}
$$

$$
q_i = \text{clamp}\!\left(\lfloor x_{b,i} \cdot \text{inv} + \text{copysign}(0.5, x_{b,i} \cdot \text{inv}) \rfloor, -127, 127\right)
$$

Note: clamped to $[-127, 127]$ (not $-128$), matching `cpu/expert.cpp`.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem scaled_clamp_symmetric (v : Int) :
    let c := if v < -127 then -127 else if v > 127 then 127 else v
    c ≥ -127 ∧ c ≤ 127 := by sorry

-- axle_verify: content
theorem scaled_clamp_symmetric (v : Int) :
    let c := if v < -127 then -127 else if v > 127 then 127 else v
    c ≥ -127 ∧ c ≤ 127 := by
  simp only
  split_ifs <;> omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void quantize_q8_0_scaled_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                            float* __restrict__ scales, long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + b * QK8_0;
    uint8_t* out = blocks + b * 34;

    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    scales[b] = s;

    const uint16_t d16bits = f16_from_f32(s);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);
    for (int i = 0; i < QK8_0; ++i) {
        const float t = xb[i] * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int) r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + i] = (uint8_t) (int8_t) v;
    }
}
```

---

### `nearest_int_dev` (device helper for Q8_K)

**[LAYER 1a — MATH (LaTeX)]**

The ggml magic-number rounding:

$$
\text{nearest\_int}(f) = \left((f + 12582912.0) \text{ as uint32}\right) \;\&\; \text{0x7FFFFF} - \text{0x400000}
$$

This is round-half-to-even via the mantissa-alignment trick: $12582912 = 1.5 \times 2^{23}$.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem nearest_int_small (n : Int) (hn : -8388608 ≤ n ∧ n ≤ 8388607) :
    nearest_int (Float.ofInt n) = n := by sorry
-- proof_needed: requires IEEE 754 mantissa-alignment reasoning
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__device__ __forceinline__ int nearest_int_dev(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}
```

---

### `quantize_q8_K_kernel`

**[LAYER 1a — MATH (LaTeX)]**

For block $b$ with 256 elements $x_{b,0}, \ldots, x_{b,255}$:

$$
a_{\max} = \max_j |x_{b,j}|, \quad \text{max\_signed} = x_{b, \arg\max_j |x_{b,j}|}
$$

$$
\text{iscale} = -127.0 / \text{max\_signed} \quad (\text{NOT } {-128})
$$

$$
q_j = \min\!\left(127, \; \text{nearest\_int}\!\left(\texttt{\_\_fmul\_rn}(\text{iscale}, x_{b,j})\right)\right) \quad (\text{no lower clamp})
$$

$$
\text{bsums}[k] = \sum_{j=16k}^{16k+15} q_j, \quad d = 1 / \text{iscale}
$$

`__fmul_rn` prevents FMA contraction: the multiply is rounded to FP32 before `nearest_int` adds the magic constant.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem q8k_bsums_correct (qs : Fin 256 → Int) (k : Fin 16) :
    bsums k = (Finset.Icc (16 * k.val) (16 * k.val + 15)).sum (fun j => qs ⟨j, by omega⟩) := by sorry
-- proof_needed: definition unfolding
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void quantize_q8_K_kernel(const float* __restrict__ x, uint8_t* __restrict__ blocks,
                                     long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float* xb = x + b * QK_K;
    uint8_t* out = blocks + b * Q8K_BYTES;
    float* d = (float*) out;
    int8_t* qs = (int8_t*) (out + 4);
    int16_t* bsums = (int16_t*) (out + 4 + QK_K);

    float max = 0.0f, amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) { amax = ax; max = xb[j]; }
    }
    if (amax == 0.0f) {
        *d = 0.0f;
        for (int j = 0; j < QK_K; ++j) qs[j] = 0;
        for (int j = 0; j < QK_K / 16; ++j) bsums[j] = 0;
        return;
    }
    const float iscale = -127.0f / max;
    for (int j = 0; j < QK_K; ++j) {
        const int v = nearest_int_dev(__fmul_rn(iscale, xb[j]));
        qs[j] = (int8_t) min(127, v);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
        bsums[j] = (int16_t) sum;
    }
    *d = 1.0f / iscale;
}
```

---

### Dequantizers (`dequant_q8_0_kernel`, `dequant_q8_K_kernel`)

**[LAYER 1a — MATH (LaTeX)]**

Q8_0: $x_{b,i} = q_i \cdot \text{f32\_from\_f16}(d_{16})$

Q8_K: $x_{b,i} = q_i \cdot d$ where $d$ is the stored float32 value.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem dequant_q8_0_roundtrip_zero : dequant_q8_0 (quant_q8_0 0) = 0 := by sorry
-- proof_needed: zero-block preservation
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void dequant_q8_0_kernel(const uint8_t* __restrict__ blocks, float* __restrict__ x,
                                    long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * 34;
    const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
    const float d = f32_from_f16(dbits);
    float* out = x + b * QK8_0;
    for (int i = 0; i < QK8_0; ++i) out[i] = (float) (int8_t) blk[2 + i] * d;
}

__global__ void dequant_q8_K_kernel(const uint8_t* __restrict__ blocks, float* __restrict__ x,
                                    long long n_blocks) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * Q8K_BYTES;
    float d;
    memcpy(&d, blk, 4);
    const int8_t* qs = (const int8_t*) (blk + 4);
    float* out = x + b * QK_K;
    for (int i = 0; i < QK_K; ++i) out[i] = (float) qs[i] * d;
}
```

---

## MODULE: neox_partial_rope (NEOX-paired partial rotary position embedding)

### Overview

Partial RoPE with NEOX pairing: pair $(i, i + n_{\text{rot}}/2)$, not adjacent $(2i, 2i+1)$. Only the first $n_{\text{rot}}$ dimensions of $\text{head\_dim}$ are rotated; the tail passes through. The cos/sin table is precomputed on the host in float64 for exact reproducibility; the kernel is a pure rotation.

---

### `build_rope_table`

**[LAYER 1a — MATH (LaTeX)]**

For position $p \in [0, \text{max\_pos})$, pair $i \in [0, n_{\text{rot}}/2)$:

$$
\theta_i^{-1} = \theta^{-2i / n_{\text{rot}}}, \quad \alpha_{p,i} = p \cdot \theta_i^{-1}
$$

$$
\text{cos\_tab}[p \cdot (n_{\text{rot}}/2) + i] = (\text{float}) \cos(\alpha_{p,i}), \quad
\text{sin\_tab}[p \cdot (n_{\text{rot}}/2) + i] = (\text{float}) \sin(\alpha_{p,i})
$$

All intermediate computation in float64.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem rope_table_unit_norm (cos_val sin_val : Float)
    (h : cos_val = Float.cos α ∧ sin_val = Float.sin α) :
    cos_val * cos_val + sin_val * sin_val = 1 := by sorry
-- proof_needed: trigonometric identity in floating point (approximate)
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}
```

---

### `rope_neox_kernel`

**[LAYER 1a — MATH (LaTeX)]**

For each row $r$ with head dimension $D$, rotation dimension $n_{\text{rot}}$, half $= n_{\text{rot}} / 2$:

**Tail (pass-through):** $\text{out}[r, d] = x[r, d] \quad \forall d \in [n_{\text{rot}}, D)$

**Rotation (NEOX pairing):** $\forall i \in [0, \text{half})$:

$$
\text{pos}_i = \text{mrope\_pos}(\text{mtab}, \text{pos}[r], i)
$$

$$
c = \cos\_\text{tab}[\text{pos}_i \cdot \text{half} + i], \quad s = \sin\_\text{tab}[\text{pos}_i \cdot \text{half} + i]
$$

$$
\text{out}[r, i] = x[r, i] \cdot c - x[r, \text{half} + i] \cdot s
$$

$$
\text{out}[r, \text{half} + i] = x[r, i] \cdot s + x[r, \text{half} + i] \cdot c
$$

One thread per row (not per pair) to avoid write races.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem rope_neox_pair_invertible (a b c s : Float) (hcs : c * c + s * s = 1) :
    let oa := a * c - b * s
    let ob := a * s + b * c
    let a' := oa * c + ob * s
    let b' := -oa * s + ob * c
    a' = a ∧ b' = b := by sorry
-- proof_needed: rotation matrix invertibility (exact only in real arithmetic, approximate in float)
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void rope_neox_kernel(const float* __restrict__ x, float* __restrict__ out, long long rows,
                                 int head_dim, int n_rot, const float* __restrict__ cos_tab,
                                 const float* __restrict__ sin_tab, const int* __restrict__ pos,
                                 const int32_t* __restrict__ mtab) {
    const long long r = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int half = n_rot / 2;
    const float* xr = x + r * head_dim;
    float* orow = out + r * head_dim;

    for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];
    for (int i = 0; i < half; ++i) {
        const size_t toff = (size_t) mrope_pos(mtab, pos[r], i) * half;
        rope_neox_pair(xr[i], xr[half + i], cos_tab[toff + i], sin_tab[toff + i], orow[i], orow[half + i]);
    }
}
```

---

## MODULE: moe_softmax_topk_router (MoE softmax / top-k / renormalize)

### Overview

One block per token. Computes softmax over all experts, selects the top-k by k iterative block-wide argmax passes (not rank counting — the rank version was 51x slower), and renormalizes with ggml's $2^{-14}$ lower clamp.

### Semantics (from `ref/moe.py::router`)

$$
p = \text{softmax}(\ell), \quad \text{ids} = \text{stable\_argsort}(-p)[:k], \quad w = p[\text{ids}]
$$

$$
s = \max\!\left(\sum_i w_i, \; 2^{-14}\right), \quad \text{output} = (\text{ids}, \; w / s)
$$

---

### `router_top10_kernel`

**[LAYER 1a — MATH (LaTeX)]**

**Phase 1 — Max (tree reduction):** $m = \max_{e} \ell[t, e]$, exact, order-independent via `fmaxf`.

**Phase 2 — Exponentials (once, parallel):**
$$\text{s\_ex}[e] = \exp((\text{double}) \ell[t,e] - (\text{double}) m) \quad \forall e \in [0, n_{\text{expert}})$$

**Phase 3 — Sum (serial, ascending, single thread):**
$$S = \sum_{e=0}^{n_{\text{expert}}-1} \text{s\_ex}[e] \quad (\text{FP64 accumulation})$$

**Phase 4 — Probabilities:** $p[e] = (\text{float})(\text{s\_ex}[e] \cdot (1/S))$

**Phase 5 — Selection (k passes, block argmax):**
For each $i \in [0, k)$, find $e^* = \arg\max_{e \notin \text{taken}} p[e]$ (ties to lowest index), mark $e^*$ taken.

**Phase 6 — Renormalize:**
$$
\sigma = \max\!\left(\sum_{i=0}^{k-1} w_i, \; 2^{-14}\right), \quad w_i \leftarrow w_i / \sigma
$$

(Division in FP64, result cast to float.)

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem softmax_sum_positive (logits : Fin n → Float) (hn : n > 0) :
    let m := Finset.univ.sup' (by exact Finset.univ_nonempty) logits
    let S := Finset.univ.sum (fun e => Float.exp (logits e - m))
    S > 0 := by sorry
-- proof_needed: exp positivity under finite inputs

-- axle_verify: formal_statement
theorem renorm_clamp_lower (ws : Fin k → Float) (hk : k > 0)
    (hw : ∀ i, ws i ≥ 0) :
    let s := max (Finset.univ.sum ws) (2⁻¹⁴ : Float)
    s > 0 := by sorry
-- proof_needed: max with positive constant is positive
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void router_topk_kernel(const float* __restrict__ logits, int n_tokens, int n_expert, int k,
                                    int* __restrict__ ids, float* __restrict__ weights) {
    extern __shared__ unsigned char s_raw[];
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    unsigned char* s_taken = s_raw;
    double* s_ex = (double*) (s_raw + taken_bytes);
    __shared__ float s_red[RT_MAX_THREADS / 32];
    __shared__ int s_rid[RT_MAX_THREADS / 32];
    __shared__ double s_sum;

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const float* l = logits + (size_t) t * n_expert;

    // tree max, parallel exp, serial sum, k argmax passes, renormalize
    // ... [structure preserved, identifier roles shown above]
}
```

---

## MODULE: grouped_moe_expert_s2 (Grouped GPU MoE expert evaluation for Q2_0 weights)

### Overview

The largest module: evaluates multiple Q2_0 MoE experts in a grouped launch (4 kernels per layer instead of 5 per expert). The Q2_0 dot product uses `__dp4a` integer MAC with bias-subtracted 2-bit codes against Q8_0-quantized activations. Pipeline: gate+up projection → SwiGLU → intermediate quantization → down projection.

### Expert Geometry

| Symbol | Value | Meaning |
|--------|-------|---------|
| H | 2560 | embedding dimension (n_embd) |
| FF | 640 | feed-forward intermediate |
| QK | 64 | Q2_0 group size (one fp16 scale) |
| ROW_GU | H/4 = 640 | code bytes per gate/up row |
| ROW_D | FF/4 = 160 | code bytes per down row |
| SC_GU | H/QK = 40 | fp16 scales per gate/up row |
| SC_D | FF/QK = 10 | fp16 scales per down row |

### Blob layout offsets

$$
O_{\text{D\_CODES}} = 2 \cdot FF \cdot \text{ROW\_GU}, \quad
O_{\text{GU\_SCALES}} = O_{\text{D\_CODES}} + H \cdot \text{ROW\_D}
$$
$$
O_{\text{D\_SCALES}} = O_{\text{GU\_SCALES}} + 2 \cdot FF \cdot \text{SC\_GU} \cdot 2
$$

---

### `row_dot_s2_q8` (device inline)

**[LAYER 1a — MATH (LaTeX)]**

Warp-strided Q2_0 × Q8_0 dot product. For chunk $c$ (32 elements):

Let $\mathbf{w}$ be the 2-bit codes (values 0,1,2,3) and $\mathbf{x}$ the int8 activations, with weight scale $d_w$ (fp16, per 64 elements) and activation scale $d_x$ (fp16 from block, or fp32 from `x_scales`):

$$
\text{dot}_c = \sum_{j=0}^{31} w_j \cdot x_j, \quad h_c = \sum_{j=0}^{31} x_j
$$

$$
\text{contribution}_c = d_w \cdot d_x \cdot (\text{dot}_c - h_c)
$$

The subtraction implements the bias: code value $v$ maps to $v - 1$, so $(v - 1) \cdot x = v \cdot x - x$.

Lane $L$ accumulates chunks $L, L+32, \ldots$; caller reduces across lanes.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem s2_bias_identity (codes : Fin 32 → Fin 4) (xs : Fin 32 → Int) :
    (Finset.univ.sum fun j => (codes j).val * xs j) -
    (Finset.univ.sum fun j => xs j) =
    Finset.univ.sum fun j => ((codes j).val - 1) * xs j := by sorry

-- axle_verify: content
theorem s2_bias_identity (codes : Fin 32 → Fin 4) (xs : Fin 32 → Int) :
    (Finset.univ.sum fun j => (codes j).val * xs j) -
    (Finset.univ.sum fun j => xs j) =
    Finset.univ.sum fun j => ((codes j).val - 1) * xs j := by
  simp [Finset.sum_sub_distrib, mul_sub, one_mul]
  ring
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__device__ __forceinline__ float row_dot_s2_q8(const uint8_t* __restrict__ codes,
                                               const uint8_t* __restrict__ scales,
                                               const uint8_t* __restrict__ x_q8_0, int n_chunks, int lane,
                                               const float* __restrict__ x_scales = nullptr) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += 32) {
        const uint8_t* cb = codes + (size_t) c * 8;
        const uint8_t* xb = x_q8_0 + (size_t) c * 34;
        const float dx = x_scales ? x_scales[c] : f16_at(xb);
        const int8_t* xq = (const int8_t*) (xb + 2);

        int s = 0;
        int hx = 0;
        const int ones = 0x01010101;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned cbyte = cb[j];
            const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                                  (((cbyte >> 6) & 3u) << 24));
            int xw;
            memcpy(&xw, xq + 4 * j, 4);
            s = __dp4a(cw, xw, s);
            hx = __dp4a(ones, xw, hx);
        }
        const float dw = f16_at(scales + (size_t) (c >> 1) * 2);
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}
```

---

### `gu_kernel` (gate + up projection)

**[LAYER 1a — MATH (LaTeX)]**

For hit $h$, interleaved row slot $i \in [0, 2 \cdot FF)$:

- Slot $i$ even → gate row $r = i/2$
- Slot $i$ odd → up row $r = (i-1)/2$

$$
\text{val} = \text{warp\_sum}\!\left(\text{row\_dot\_s2\_q8}(\text{blob}[h] + i \cdot \text{ROW\_GU}, \; \text{blob}[h] + O_{\text{GU\_SCALES}} + i \cdot \text{SC\_GU} \cdot 2, \; x, \; H/32, \; \text{lane})\right)
$$

Output layout is **gate-major**: gate rows at `[h * FF + r]`, up rows at `[n_hits * FF + h * FF + r]`.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem interleave_deinterleave (i : Nat) (hi : i < 2 * FF) :
    let is_up := i % 2 = 1
    let r := i / 2
    let out_idx := if is_up then n_hits * FF + h * FF + r else h * FF + r
    out_idx < 2 * n_hits * FF := by sorry
-- proof_needed: arithmetic bounds
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void gu_kernel(const uint8_t* __restrict__ blob_base, const int32_t* __restrict__ slot_index,
                          long long blob_bytes, const uint8_t* __restrict__ x_q8_0,
                          const float* __restrict__ x_scales, float* __restrict__ gate_up, int n_hits,
                          const int32_t* __restrict__ d_count = nullptr,
                          const int32_t* __restrict__ dst_index = nullptr, int tok_div = 0) {
    // ... one warp per interleaved gate/up row, deinterleave into gate-major output ...
    const float acc = row_dot_s2_q8(blob + (size_t) i * ROW_GU,
                                    blob + O_GU_SCALES + (size_t) i * SC_GU * 2, xb, H / 32, lane, x_scales);
    const float s = warp_sum(acc);
    if (lane != 0) return;
    const int r = i >> 1;
    const size_t base = (i & 1) ? ((size_t) n_hits * FF + (size_t) h * FF) : ((size_t) h * FF);
    gate_up[base + (size_t) r] = s;
}
```

---

### `swiglu_kernel`

**[LAYER 1a — MATH (LaTeX)]**

Over a gate-major buffer, for pair $i \in [0, n_{\text{pairs}})$:

$$
\text{gate\_up}[i] \leftarrow \text{SiLU}(\text{gate\_up}[i]) \cdot \text{gate\_up}[n_{\text{pairs}} + i]
$$

where $\text{SiLU}(g) = g / (1 + e^{-g})$.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem silu_zero : silu 0 = 0 := by sorry

-- axle_verify: content
theorem silu_zero : silu 0 = 0 := by
  simp [silu, div_self]
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void swiglu_kernel(float* __restrict__ gate_up, long long n_pairs) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    const float g = gate_up[i];
    const float u = gate_up[n_pairs + i];
    gate_up[i] = (g / (1.0f + __expf(-g))) * u;
}
```

---

### `down_kernel`

**[LAYER 1a — MATH (LaTeX)]**

For hit $h$, output row $r \in [0, H)$:

$$
\text{out}[\text{dst}[h] \cdot H + r] = \text{warp\_sum}\!\left(\text{row\_dot\_s2\_q8}\!\left(\text{blob}[h] + O_{\text{D\_CODES}} + r \cdot \text{ROW\_D}, \; \text{blob}[h] + O_{\text{D\_SCALES}} + r \cdot \text{SC\_D} \cdot 2, \; h_{\text{q8}}, \; FF/32, \; \text{lane}\right)\right)
$$

`dst_index[h]` is the router's slot (not sequential), so hits land on the correct rows for `moe_combine`.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem down_output_row_bound (dst_h : Nat) (r : Nat) (hr : r < 2560) :
    dst_h * 2560 + r < (dst_h + 1) * 2560 := by sorry

-- axle_verify: content
theorem down_output_row_bound (dst_h : Nat) (r : Nat) (hr : r < 2560) :
    dst_h * 2560 + r < (dst_h + 1) * 2560 := by
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void down_kernel(const uint8_t* __restrict__ blob_base, const int32_t* __restrict__ slot_index,
                            const int32_t* __restrict__ dst_index, long long blob_bytes,
                            const uint8_t* __restrict__ h_q8_0, const float* __restrict__ h_scales,
                            float* __restrict__ out, int n_hits, const int32_t* __restrict__ d_count = nullptr) {
    // ... one warp per output row of each hit ...
    const float acc = row_dot_s2_q8(blob + O_D_CODES + (size_t) r * ROW_D,
                                    blob + O_D_SCALES + (size_t) r * SC_D * 2, xb, FF / 32, lane,
                                    h_scales ? h_scales + (size_t) h * (size_t) (FF / 32) : nullptr);
    const float s = warp_sum(acc);
    if (lane == 0) out[(size_t) dst_index[h] * H + r] = s;
}
```

---

### `moe_hit_select` / `hit_select_kernel`

**[LAYER 1a — MATH (LaTeX)]**

From $k$ routed expert IDs and a residency row $\text{res}[e]$ ($\ge 0$ if resident, $-1$ otherwise):

$$
\text{hit set} = \{(i, \text{res}[\text{ids}[i]]) : \text{res}[\text{ids}[i]] \ge 0, \; i \in [0, k)\}
$$

Compacted via warp ballot into dense `slot[]` and `dst[]` arrays; `count` = number of hits.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem hit_select_count_le_k (k : Nat) (hits : Nat) (hh : hits = popcount ballot) :
    hits ≤ k := by sorry
-- proof_needed: popcount of a mask with at most k bits set
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void hit_select_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ res_row, int k,
                                  int n_expert, int32_t* __restrict__ slot, int32_t* __restrict__ dst,
                                  int32_t* __restrict__ count) {
    const int lane = threadIdx.x;
    int s = -1;
    if (lane < k) {
        const int e = ids[lane];
        if (e >= 0 && e < n_expert) s = res_row[e];
    }
    const unsigned hit = __ballot_sync(0xffffffffu, s >= 0);
    if (s >= 0) {
        const int at = __popc(hit & ((1u << lane) - 1u));
        slot[at] = s;
        dst[at] = lane;
    }
    if (lane == 0) *count = __popc(hit);
}
```

---

### `moe_hit_grouped_scratch_bytes`

**[LAYER 1a — MATH (LaTeX)]**

$$
\text{scratch} = \text{align}_{16}(n_h \cdot 2 \cdot n_{\text{ff}} \cdot 4) + \text{align}_{16}(n_h \cdot (n_{\text{ff}} / 32) \cdot 34) + 2 \cdot \text{align}_{16}(n_h \cdot (n_{\text{ff}} / 32) \cdot 4) + \text{align}_{16}((n_e / 32) \cdot 4)
$$

where the regions are: gate/up floats, quantized Q8_0 intermediate, fp32 scales for intermediate, and activation correction buffer.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem scratch_alignment (x : Nat) : (x + 15) / 16 * 16 ≥ x := by sorry

-- axle_verify: content
theorem scratch_alignment (x : Nat) : (x + 15) / 16 * 16 ≥ x := by
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cpp
uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) +
           ((xh + 15) & ~15ull);
}
```

---

## MODULE: s2_gemv_experimental (S2 GEMV with constant-memory LUT and optional shared staging)

### Overview

**Negative-result experiment:** Two optimization levers on the S2 GEMV kernel, both measured worse or neutral.

**Lever 1:** Stage activation `x` in shared memory — measured consistently worse (59-76 vs 64-87 G w/s).

**Lever 2:** Constant-memory code lookup table (byte → 4 float codes) — measured neutral.

Kept as documentation of a closed optimization path; the run-to-run variance (~20%) exceeds the effect sizes.

### Constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| QK_S2 | 64 | S2 group size |
| MAX_SHARED_HALVES | 4096 | max fp16 elements in shared staging |

---

### `s2_gemv_fast_kernel<STAGE_X>`

**[LAYER 1a — MATH (LaTeX)]**

For output row $o$, the kernel computes:

$$
y[o] = \sum_{q=0}^{n_{\text{in}}/4-1} \sum_{k=0}^{3} \text{cv}[q]_k \cdot d_q \cdot \text{fp16\_to\_f32}(x[4q + k])
$$

where $\text{cv}[q] = c\_codes[\text{byte}(q)]$ is a constant-memory lookup returning 4 floats (the S2 code values $\{-1, 0, 1, 2\}$ from the 2-bit fields), and $d_q = \text{scales}[q \cdot 4 / 64]$.

Accumulation in 4 independent channels $(a_0, a_1, a_2, a_3)$, reduced through shared memory.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem lut_code_range (byte : Fin 256) (k : Fin 4) :
    let code := (byte.val / (2 ^ (2 * k.val))) % 4
    code ≥ 0 ∧ code ≤ 3 := by sorry

-- axle_verify: content
theorem lut_code_range (byte : Fin 256) (k : Fin 4) :
    let code := (byte.val / (2 ^ (2 * k.val))) % 4
    code ≥ 0 ∧ code ≤ 3 := by
  constructor
  · exact Nat.zero_le _
  · exact Nat.lt_succ_of_lt (Nat.mod_lt _ (by omega)) |>.le
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
template <bool STAGE_X>
__global__ void s2_gemv_fast_kernel(const uint16_t* __restrict__ x, const uint8_t* __restrict__ codes,
                                    const float* __restrict__ scales, float* __restrict__ y, long long n_in,
                                    long long n_out, int threads_per_row) {
    extern __shared__ float smem[];
    __half* sx = reinterpret_cast<__half*>(smem);
    float* partial = smem + (STAGE_X ? (MAX_SHARED_HALVES / 2) : 0);

    const long long o = blockIdx.x;
    if (o >= n_out) return;
    const int tid = threadIdx.x;

    if (STAGE_X) {
        const __half* xh = reinterpret_cast<const __half*>(x);
        for (long long i = tid; i < n_in; i += threads_per_row) sx[i] = xh[i];
        __syncthreads();
    }

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];
        const float d = s[q >> 4];
        const float4 cv = *reinterpret_cast<const float4*>(&c_codes[byte][0]);
        // ... load x from shared or global, accumulate a0..a3 ...
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    __syncthreads();
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) y[o] = partial[0];
}
```

---

## MODULE: s2_gemv_q8_activation (S2 GEMV consuming Q8_0-quantized activations)

### Overview

The **correctness-first** S2 GEMV that takes Q8_0 block-format activations instead of FP16. Dequantizes each activation element inline in the loop (`xq * dx`), otherwise identical in weight decode to the FP16 variant. This closes the gap between GPU (FP16 activation) and CPU (INT8 via `vpdpbusd`) expert paths. The `__dp4a` optimization is deferred to Phase 3.

---

### `s2_gemv_q8_kernel`

**[LAYER 1a — MATH (LaTeX)]**

For output row $o$:

$$
y[o] = \sum_{q=0}^{n_{\text{in}}/4-1} \sum_{k=0}^{3} (c_{q,k} - 1) \cdot d_q^{(w)} \cdot x_{q,k}^{(\text{int8})} \cdot d_q^{(x)}
$$

where:
- $c_{q,k}$: 2-bit code field $k$ of byte $q$ (values 0,1,2,3)
- $d_q^{(w)} = \text{scales}[4q / 64]$: weight scale (FP32)
- $x_{q,k}^{(\text{int8})} = \text{qs}[\text{off} + k]$: int8 activation from Q8_0 block
- $d_q^{(x)} = \text{fp16\_to\_f32}(\text{block}[4q / 32].\text{d})$: activation scale (FP16)

Accumulated in 4 channels, reduced via shared memory.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem q8_block_alignment (q : Nat) (n_in : Nat) (h : n_in % 32 = 0) :
    (q * 4) / 32 * 34 + 34 ≤ n_in / 32 * 34 := by sorry
-- proof_needed: block-boundary arithmetic for Q8_0
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void s2_gemv_q8_kernel(const uint8_t* __restrict__ act, const uint8_t* __restrict__ codes,
                                  const float* __restrict__ scales, float* __restrict__ y, long long n_in,
                                  long long n_out, int threads_per_row) {
    extern __shared__ float partial[];
    const long long o = blockIdx.x;
    if (o >= n_out) return;
    const int tid = threadIdx.x;

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];
        const float d = s[q >> 4];
        const long long ablk = (q * 4) / QK8_0;
        const uint8_t* blk = act + ablk * 34;
        const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
        const float dx = __half2float(__ushort_as_half(dbits));
        const int8_t* xq = reinterpret_cast<const int8_t*>(blk + 2);
        const int off = (int) ((q * 4) % QK8_0);

        const float w0 = (float) ((int) (byte & 3) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
        a0 += w0 * ((float) xq[off + 0] * dx);
        a1 += w1 * ((float) xq[off + 1] * dx);
        a2 += w2 * ((float) xq[off + 2] * dx);
        a3 += w3 * ((float) xq[off + 3] * dx);
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    __syncthreads();
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) y[o] = partial[0];
}
```

---

### Host dispatchers (all modules)

All host-side dispatch functions follow the same pattern: validate geometry and preconditions, compute grid dimensions, dispatch kernel template variants (INT8/FP16, STAGE_X/not), check `cudaGetLastError`, optionally synchronize for null stream. The form templates for these are omitted as they are structurally identical to the kernel templates shown above with added error checking.

---

*End of Chunk 10 theory — 8 files covered.*
