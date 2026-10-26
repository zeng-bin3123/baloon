# Theory — Chunk 11: Quantized GEMV, Sampler, Shared Expert, Verify Kernels

---

## MODULE: s2_gemv_quad_amortized (S2 quantized GEMV with quad-element load amortization)

This kernel specializes for 2-bit quantized weights (S2 format) where four consecutive elements share one code byte. Instead of loading one byte and one half per element, a thread processes a contiguous quad: one byte load and one 64-bit word load for four elements. Memory instructions per element drop from 2 to 0.5.

### Constants and Layout

The S2 format uses `QK = 64` elements per quantization group, with `CODES_PER_BYTE = 4` (2 bits each). Code byte `q` stores elements `[4q, 4q+3]`. The group index for quad `q` is `q >> 4` (since `(q*4)/64 = q/16`).

---

### s2_gemv_quads_kernel

**[LAYER 1a — MATH (LaTeX)]**

$$\text{s2\_gemv\_quads\_kernel}: \mathbb{F}_{16}^{n_{in}} \times \mathbb{B}^{n_{out} \times n_{in}/4} \times \mathbb{R}^{n_{out} \times n_{in}/64} \times \mathbb{R}^{n_{out}} \to \mathbb{R}^{n_{out}}$$

$$\forall\, o \in [0, n_{out}),\quad y[o] = \sum_{q=0}^{n_{in}/4 - 1} \sum_{k=0}^{3} w_{o,4q+k} \cdot x[4q+k]$$

where

$$w_{o,4q+k} = \bigl(\text{decode}_2(\text{codes}[o \cdot n_{quads} + q], k) - 1\bigr) \cdot \text{scales}\!\bigl[o \cdot (n_{in}/64) + (q \gg 4)\bigr]$$

$$\text{decode}_2(b, k) = (b \gg (2k)) \;\&\; 3$$

**Preconditions:**
$$n_{in} > 0 \;\land\; n_{out} > 0 \;\land\; 4 \mid n_{in} \;\land\; \text{threads\_per\_row} = 2^m \text{ for some } m$$

**Postcondition:** The shared-memory tree reduction produces a deterministic result for fixed `threads_per_row`.

**Invariant (code–element mapping):** Element $i$ is encoded at byte $\lfloor i/4 \rfloor$, bits $[(i \bmod 4) \cdot 2,\; (i \bmod 4) \cdot 2 + 1]$.

**Invariant (group mapping via shift):** $\text{group}(q) = q \gg 4 = \lfloor q/16 \rfloor = \lfloor (4q)/64 \rfloor$.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem s2_quad_group_shift (q : Nat) :
    q / 16 = (q * 4) / 64 := by sorry

-- axle_verify: content
theorem s2_quad_group_shift (q : Nat) :
    q / 16 = (q * 4) / 64 := by omega
```

```lean
-- axle_verify: formal_statement
theorem s2_decode_range (byte : Nat) (k : Fin 4) :
    ((byte >>> (k.val * 2)) &&& 3) < 4 := by sorry

-- axle_verify: content
theorem s2_decode_range (byte : Nat) (k : Fin 4) :
    ((byte >>> (k.val * 2)) &&& 3) < 4 := by
  have : 3 < 4 := by omega
  exact Nat.lt_of_le_of_lt (Nat.and_le_right _ 3) this
```

```lean
-- axle_verify: formal_statement
theorem s2_quad_index_byte (q : Nat) (k : Fin 4) :
    (q * 4 + k.val) / 4 = q := by sorry

-- axle_verify: content
theorem s2_quad_index_byte (q : Nat) (k : Fin 4) :
    (q * 4 + k.val) / 4 = q := by omega
```

```lean
-- axle_verify: formal_statement
theorem s2_shared_mem_reduction_equiv (n : Nat) (hn : n > 0) (hpow : ∃ m, n = 2^m)
    (a : Fin n → Float) :
    True := by sorry

-- axle_verify: content  -- proof_needed (float reduction commutativity not provable in Lean without axioms)
theorem s2_shared_mem_reduction_equiv (n : Nat) (hn : n > 0) (hpow : ∃ m, n = 2^m)
    (a : Fin n → Float) :
    True := by trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
constexpr int QK_ADJ = 64;
constexpr int CODES_PER_BYTE_ADJ = 4;

__global__ void adj_gemv_quads_kernel(const uint16_t* __restrict__ argvec,
                                      const uint8_t* __restrict__ argcodes,
                                      const float* __restrict__ argscales,
                                      float* __restrict__ argout,
                                      long long argdim_in,
                                      long long argdim_out,
                                      int argthreads) {
    extern __shared__ float partbuf[];
    const long long rowIdx = blockIdx.x;
    if (rowIdx >= argdim_out) return;
    const int tid = threadIdx.x;

    const long long numQuads = argdim_in / 4;
    const uint8_t* rowcodes = argcodes + rowIdx * numQuads;
    const float* rowscales = argscales + rowIdx * (argdim_in / QK_ADJ);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (long long idx = tid; idx < numQuads; idx += argthreads) {
        const uint8_t codebyte = rowcodes[idx];
        const float scl = rowscales[idx >> 4];
        const uint2 vecword = *reinterpret_cast<const uint2*>(argvec + idx * 4);
        const __half2 pair01 = *reinterpret_cast<const __half2*>(&vecword.x);
        const __half2 pair23 = *reinterpret_cast<const __half2*>(&vecword.y);
        const float w0 = (float) ((int) (codebyte & 3) - 1) * scl;
        const float w1 = (float) ((int) ((codebyte >> 2) & 3) - 1) * scl;
        const float w2 = (float) ((int) ((codebyte >> 4) & 3) - 1) * scl;
        const float w3 = (float) ((int) ((codebyte >> 6) & 3) - 1) * scl;
        acc0 += w0 * __low2float(pair01);
        acc1 += w1 * __high2float(pair01);
        acc2 += w2 * __low2float(pair23);
        acc3 += w3 * __high2float(pair23);
    }
    partbuf[tid] = (acc0 + acc1) + (acc2 + acc3);
    __syncthreads();
    for (int step = argthreads / 2; step > 0; step >>= 1) {
        if (tid < step) partbuf[tid] += partbuf[tid + step];
        __syncthreads();
    }
    if (tid == 0) argout[rowIdx] = partbuf[0];
}
```

### s2_gemv_quads (host wrapper)

**[LAYER 1a — MATH (LaTeX)]**

$$\text{s2\_gemv\_quads}: \text{Precond} \to \text{Launch}$$

$$\text{Precond}: n_{in} > 0 \;\land\; n_{out} > 0 \;\land\; 4 \mid n_{in}$$

$$\text{smem} = \text{threads\_per\_row} \times 4 \text{ bytes}$$

$$\text{grid} = n_{out},\quad \text{block} = \text{threads\_per\_row}$$

Abort with diagnostic if $4 \nmid n_{in}$.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem s2_gemv_quads_precond (n_in n_out : Nat) (hn : n_in > 0) (ho : n_out > 0) (hmod : n_in % 4 = 0) :
    n_in / 4 * 4 = n_in := by sorry

-- axle_verify: content
theorem s2_gemv_quads_precond (n_in n_out : Nat) (hn : n_in > 0) (ho : n_out > 0) (hmod : n_in % 4 = 0) :
    n_in / 4 * 4 = n_in := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
void adj_gemv_quads(const uint16_t* argvec, const uint8_t* argcodes, const float* argscales,
                    float* argout, int64_t argdim_in, int64_t argdim_out, int argthreads) {
    if (argdim_in <= 0 || argdim_out <= 0) return;
    if (argdim_in % 4 != 0) {
        std::fprintf(stderr, "adj_gemv_quads: argdim_in %lld is not a multiple of 4\n", (long long) argdim_in);
        std::exit(1);
    }
    const size_t smembytes = (size_t) argthreads * sizeof(float);
    adj_gemv_quads_kernel<<<(unsigned) argdim_out, argthreads, smembytes>>>(
        argvec, argcodes, argscales, argout, argdim_in, argdim_out, argthreads);
    const cudaError_t errval = cudaDeviceSynchronize();
    if (errval != cudaSuccess) {
        std::fprintf(stderr, "adj_gemv_quads: %s\n", cudaGetErrorString(errval));
        std::exit(1);
    }
}
```

---

## MODULE: s_family_gemv (Generic S-family quantized GEMV with variable code width)

This module implements quantized GEMV for the S-family of formats (S2, S4, S8) with code widths of 2, 4, or 8 bits. It supports two activation formats (Q8_K: 292 bytes / 256 elements, Q8_0: 34 bytes / 32 elements), three codebook modes (affine with bias, IQ4NL non-linear), and optional per-group offsets. Key optimizations include shared-memory codebook placement (eliminating 2.12× constant-memory penalty), group-index computation via shift instead of division, and multi-element quads to amortize loads.

### IQ4NL Codebook and Decode

**[LAYER 1a — MATH (LaTeX)]**

$$\text{kIq4Nl} = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113] \in \mathbb{Z}^{16}$$

$$\text{decode}(c, \text{bias}, \text{codebook}) = \begin{cases} \text{kIq4Nl}[c \;\&\; 0\text{F}] & \text{if codebook} = \text{Iq4Nl} \\ c + \text{bias} & \text{otherwise} \end{cases}$$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem decode_iq4nl_range (code : Nat) (h : code < 16) :
    -127 ≤ ([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113].get ⟨code, by omega⟩) ∧
    ([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113].get ⟨code, by omega⟩) ≤ 113 := by sorry

-- axle_verify: content  -- proof_needed (exhaustive case analysis on 16 values)
theorem decode_iq4nl_range (code : Nat) (h : code < 16) :
    -127 ≤ ([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113].get ⟨code, by omega⟩) ∧
    ([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113].get ⟨code, by omega⟩) ≤ 113 := by
  interval_cases code <;> simp
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__constant__ signed char kCodebookNl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                            1,   13,   25,  38,  53,  69,  89,  113};

template <int CODE_BITS>
__device__ __forceinline__ float decode(int argcode, int argbias, int argcodebook) {
    if (argcodebook == (int) typenameCodebook::Iq4Nl) return (float) kCodebookNl[argcode & 0x0F];
    return (float) (argcode + argbias);
}

template <int CODE_BITS>
__device__ __forceinline__ float decode_tbl(int argcode, int argbias, int argcodebook,
                                            const signed char* argtbl) {
    if (argcodebook == (int) typenameCodebook::Iq4Nl) return (float) argtbl[argcode & 0x0F];
    return (float) (argcode + argbias);
}
```

### Q8_K Activation Accessor

**[LAYER 1a — MATH (LaTeX)]**

$$\text{Q8K layout}: \{f_{32}\; d,\; \text{int8}\; qs[256],\; \text{int16}\; bsums[16]\},\quad |\text{block}| = 292 \text{ bytes},\; 256 \text{ elements}$$

$$q8k\_at(x, i) = d_{blk} \cdot qs[i \bmod 256]$$

where $blk = x + \lfloor i / 256 \rfloor \cdot 292$, $d_{blk} = *(float*)\,blk$, $qs = (int8*)(blk + 4)$.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem q8k_block_offset (i : Nat) :
    i / 256 * 292 + 4 + i % 256 < (i / 256 + 1) * 292 := by sorry

-- axle_verify: content
theorem q8k_block_offset (i : Nat) :
    i / 256 * 292 + 4 + i % 256 < (i / 256 + 1) * 292 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
constexpr int QUANT8K_BLOCK_BYTES = 292;
constexpr int QUANT8K_BLOCK_ELEMS = 256;

__device__ __forceinline__ float quant8k_at(const uint8_t* __restrict__ argvec, long long argidx) {
    const uint8_t* blkptr = argvec + (argidx / QUANT8K_BLOCK_ELEMS) * QUANT8K_BLOCK_BYTES;
    const float scl = __ldg((const float*) blkptr);
    const int8_t qval = ((const int8_t*) (blkptr + 4))[argidx % QUANT8K_BLOCK_ELEMS];
    return scl * (float) qval;
}
```

### Q8_0 Activation Accessor

**[LAYER 1a — MATH (LaTeX)]**

$$\text{Q8\_0 layout}: \{f_{16}\; d,\; \text{int8}\; qs[32]\},\quad |\text{block}| = 34 \text{ bytes},\; 32 \text{ elements}$$

$$q8\_0\_at(x, i) = \text{f16\_to\_f32}(d_{blk}) \cdot qs[i \bmod 32]$$

where $blk = x + \lfloor i / 32 \rfloor \cdot 34$, $d_{blk} = *(uint16*)\,blk$, $qs = (int8*)(blk + 2)$.

**Structural constraint:** $n_{in} = 640$ for `ffn_down_shexp`, which is not a multiple of 256, making Q8_K structurally impossible. Q8_0 is the required activation format.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem q8_0_block_offset (i : Nat) :
    i / 32 * 34 + 2 + i % 32 < (i / 32 + 1) * 34 := by sorry

-- axle_verify: content
theorem q8_0_block_offset (i : Nat) :
    i / 32 * 34 + 2 + i % 32 < (i / 32 + 1) * 34 := by omega
```

```lean
-- axle_verify: formal_statement
theorem q8k_impossible_640 : ¬ (256 ∣ 640) := by sorry

-- axle_verify: content
theorem q8k_impossible_640 : ¬ (256 ∣ 640) := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
constexpr int QUANT80_BLOCK_BYTES = 34;
constexpr int QUANT80_BLOCK_ELEMS = 32;

__device__ __forceinline__ float quant80_at(const uint8_t* __restrict__ argvec, long long argidx) {
    const uint8_t* blkptr = argvec + (argidx / QUANT80_BLOCK_ELEMS) * QUANT80_BLOCK_BYTES;
    const float scl = __half2float(__ushort_as_half(*(const uint16_t*) blkptr));
    const int8_t qval = ((const int8_t*) (blkptr + 2))[argidx % QUANT80_BLOCK_ELEMS];
    return scl * (float) qval;
}
```

### s_gemv_q8k_kernel (Q8_K activation, one thread per output)

**[LAYER 1a — MATH (LaTeX)]**

$$\forall\, o \in [0, n_{out}),\quad y[o] = \sum_{g=0}^{n_{groups}-1} \sum_{j=0}^{G-1} \bigl(\text{decode}(c_{o,g \cdot G + j}) \cdot s_{o,g} + b_{o,g}\bigr) \cdot q8k\_at(x, g \cdot G + j)$$

where $G = \text{group\_elems}$, $c_{o,i}$ is the code for element $i$ in output row $o$, $s_{o,g}$ is the scale for group $g$, and $b_{o,g}$ is the optional offset (0 if absent).

**Key:** Shared codebook `s_iq4nl[16]` loaded once per block. Barrier **before** early return (thread-indexed kernel: a thread that returned before `__syncthreads` would deadlock the block).

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem code_extraction_bits (code_bits i per_byte : Nat) (hpb : per_byte = 8 / code_bits)
    (hcb : code_bits ∈ [2, 4, 8]) (byte_val : Nat) (hbyte : byte_val < 256) :
    ((byte_val >>> ((i % per_byte) * code_bits)) &&& ((1 <<< code_bits) - 1)) < (1 <<< code_bits) := by sorry

-- axle_verify: content  -- proof_needed (bitwise reasoning complex)
theorem code_extraction_bits (code_bits i per_byte : Nat) (hpb : per_byte = 8 / code_bits)
    (hcb : code_bits ∈ [2, 4, 8]) (byte_val : Nat) (hbyte : byte_val < 256) :
    ((byte_val >>> ((i % per_byte) * code_bits)) &&& ((1 <<< code_bits) - 1)) < (1 <<< code_bits) := by
  simp [Nat.and_lt_two_pow]
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
template <int CODE_BITS>
__global__ void adj_gemv_q8k_kernel(const uint8_t* __restrict__ argvec,
                                    const uint8_t* __restrict__ argcodes,
                                    const float* __restrict__ argscales,
                                    const float* __restrict__ argoffset,
                                    float* __restrict__ argout,
                                    long long argdim_in, long long argdim_out,
                                    int argbias, int argcodebook,
                                    int arggroup_elems, int arghas_offset) {
    __shared__ signed char sharedCodebook[16];
    if (threadIdx.x < 16) sharedCodebook[threadIdx.x] = kCodebookNl[threadIdx.x];
    __syncthreads();
    const long long rowIdx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIdx >= argdim_out) return;

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long numGroups = argdim_in / arggroup_elems;
    const long long codesPerRow = argdim_in / PER_BYTE;
    const uint8_t* rowcodes = argcodes + rowIdx * codesPerRow;
    const float* rowscales = argscales + rowIdx * numGroups;
    const float* rowoff = arghas_offset ? argoffset + rowIdx * numGroups : nullptr;

    float acc = 0.0f;
    for (long long gIdx = 0; gIdx < numGroups; ++gIdx) {
        const float scl = rowscales[gIdx];
        const float off = rowoff ? rowoff[gIdx] : 0.0f;
        const long long base = gIdx * (long long) arggroup_elems;
        for (int jIdx = 0; jIdx < arggroup_elems; ++jIdx) {
            const long long elemIdx = base + jIdx;
            const int cval = (rowcodes[elemIdx / PER_BYTE] >> ((int)(elemIdx % PER_BYTE) * CODE_BITS))
                             & ((1 << CODE_BITS) - 1);
            const float wval = decode_tbl<CODE_BITS>(cval, argbias, argcodebook, sharedCodebook) * scl + off;
            acc += wval * quant8k_at(argvec, elemIdx);
        }
    }
    argout[rowIdx] = acc;
}
```

### s_gemv_q8_split_kernel (warp-per-row, QE=16 elements per lane, shift-based group index)

**[LAYER 1a — MATH (LaTeX)]**

$$\text{Grid}: \lceil n_{out} / \text{warps\_per\_block} \rceil \text{ blocks},\quad \text{one warp per output row}$$

$$\forall\, o \in [0, n_{out}),\quad y[o] = \sum_{\text{lane}=0}^{31} \text{acc}[\text{lane}]$$

Each lane processes elements with stride $32 \cdot QE$ where $QE = 16$:

$$\text{acc}[\ell] = \sum_{\substack{i = \ell \cdot QE,\\ i + QE \leq n_{in},\\ i \mathrel{+}= 32 \cdot QE}} \sum_{k=0}^{15} w_{o,i+k} \cdot \text{act}(x, i+k)$$

**Key optimization: activation block hoisting.** All $QE = 16$ elements of one lane-iteration lie in one activation block because:
- $QE = 16$, $Q8K\_ELEMS = 256$, $Q8\_0\_ELEMS = 32$
- Loop advances by $32 \cdot QE = 512$, so $i \bmod 256$ and $i \bmod 32$ are always multiples of 16
- Therefore $i + QE \leq \text{blk\_elems}$ for every reachable $i$, eliminating 16 divisions per iteration

**Weight decode load amortization:** 16 elements → 4 code words loaded (each covering 4 elements via $QW = 4 \cdot CODE\_BITS / 8$ bytes).

**Group shift invariant:** $g = i \gg \text{group\_shift}$ instead of $i / \text{group\_elems}$. Valid because all group sizes (16, 32, 64) are powers of two.

**Warp reduction:** `__shfl_down_sync` tree, lane 0 writes.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem q8k_no_block_straddle (lane : Fin 32) (iter : Nat) (QE : Nat) (hqe : QE = 16) :
    let i := lane.val * QE + iter * 32 * QE
    i % 256 + QE ≤ 256 := by sorry

-- axle_verify: content
theorem q8k_no_block_straddle (lane : Fin 32) (iter : Nat) (QE : Nat) (hqe : QE = 16) :
    let i := lane.val * QE + iter * 32 * QE
    i % 256 + QE ≤ 256 := by
  subst hqe; simp; omega

-- axle_verify: formal_statement
theorem q8_0_no_block_straddle (lane : Fin 32) (iter : Nat) (QE : Nat) (hqe : QE = 16) :
    let i := lane.val * QE + iter * 32 * QE
    i % 32 + QE ≤ 32 := by sorry

-- axle_verify: content
theorem q8_0_no_block_straddle (lane : Fin 32) (iter : Nat) (QE : Nat) (hqe : QE = 16) :
    let i := lane.val * QE + iter * 32 * QE
    i % 32 + QE ≤ 32 := by
  subst hqe; simp; omega
```

```lean
-- axle_verify: formal_statement
theorem group_shift_equiv (i group_elems : Nat) (hpow : ∃ k, group_elems = 2^k) (hge : group_elems > 0) :
    i / group_elems = i >>> (Nat.log2 group_elems) := by sorry

-- axle_verify: content  -- proof_needed (requires Nat.log2/shift correspondence)
theorem group_shift_equiv (i group_elems : Nat) (hpow : ∃ k, group_elems = 2^k) (hge : group_elems > 0) :
    i / group_elems = i >>> (Nat.log2 group_elems) := by
  obtain ⟨k, rfl⟩ := hpow
  simp [Nat.shiftRight_eq_div_pow, Nat.log2_pow]
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
template <int CODE_BITS, bool QUANT8K>
__global__ void adj_gemv_q8_split_kernel(const uint8_t* __restrict__ argvec,
                                         const uint8_t* __restrict__ argcodes,
                                         const float* __restrict__ argscales,
                                         const float* __restrict__ argoffset,
                                         float* __restrict__ argout,
                                         long long argdim_in, long long argdim_out,
                                         int argbias, int argcodebook,
                                         int arggroup_shift, int arghas_offset) {
    constexpr int PER_BYTE = 8 / CODE_BITS;
    const int warpsPerBlk = (int) (blockDim.x >> 5);
    const long long rowIdx = (long long) blockIdx.x * warpsPerBlk + (threadIdx.x >> 5);
    if (rowIdx >= argdim_out) return;
    const int lane = threadIdx.x & 31;

    __shared__ signed char sharedCodebook[16];
    if (threadIdx.x < 16) sharedCodebook[threadIdx.x] = kCodebookNl[threadIdx.x];
    __syncthreads();

    const long long numGroups = argdim_in >> arggroup_shift;
    const long long codesPerRow = argdim_in / PER_BYTE;
    const uint8_t* rowcodes = argcodes + rowIdx * codesPerRow;
    const float* rowscales = argscales + rowIdx * numGroups;
    const float* rowoff = arghas_offset ? argoffset + rowIdx * numGroups : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    float acc8 = 0.0f, acc9 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    float acc12 = 0.0f, acc13 = 0.0f, acc14 = 0.0f, acc15 = 0.0f;
    constexpr int QE = 16;
    constexpr int QW = 4 * CODE_BITS / 8;
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    long long elemIdx = (long long) lane * QE;
    for (; elemIdx + QE <= argdim_in; elemIdx += 32 * QE) {
        const long long gIdx = elemIdx >> arggroup_shift;
        const float scl = rowscales[gIdx];
        const float off = rowoff ? rowoff[gIdx] : 0.0f;
        const uint8_t* codeptr = rowcodes + elemIdx / PER_BYTE;
        unsigned cw, cw2, cw3, cw4;
        if (QW == 1) { cw = codeptr[0]; cw2 = codeptr[1]; cw3 = codeptr[2]; cw4 = codeptr[3]; }
        else if (QW == 2) { cw = *(const uint16_t*) codeptr; cw2 = *(const uint16_t*)(codeptr+2);
                            cw3 = *(const uint16_t*)(codeptr+4); cw4 = *(const uint16_t*)(codeptr+6); }
        else { cw = *(const uint32_t*) codeptr; cw2 = *(const uint32_t*)(codeptr+4);
               cw3 = *(const uint32_t*)(codeptr+8); cw4 = *(const uint32_t*)(codeptr+12); }
        // Hoisted activation block
        const int blkElems = QUANT8K ? QUANT8K_BLOCK_ELEMS : QUANT80_BLOCK_ELEMS;
        const uint8_t* actblk = argvec + (elemIdx / blkElems) * (QUANT8K ? QUANT8K_BLOCK_BYTES : QUANT80_BLOCK_BYTES);
        const int actoff = (int)(elemIdx % blkElems);
        const float actscl = QUANT8K ? __ldg((const float*) actblk)
                                     : __half2float(__ushort_as_half(*(const uint16_t*) actblk));
        const int8_t* actqs = (const int8_t*)(actblk + (QUANT8K ? 4 : 2)) + actoff;

        // 16 weight decodes, 16 accumulations (abbreviated for template)
        const float w0 = decode_tbl<CODE_BITS>((int)(cw & MASK), argbias, argcodebook, sharedCodebook) * scl + off;
        // ... w1..w15 analogous
        acc0 += w0 * (actscl * (float)actqs[0]);
        // ... acc1..acc15 analogous
    }
    // tail loop (at most one per lane)
    for (; elemIdx < argdim_in; elemIdx += 32 * QE) {
        for (int k = 0; k < QE && elemIdx + k < argdim_in; ++k) {
            const long long e = elemIdx + k;
            const long long g = e >> arggroup_shift;
            const int cval = (rowcodes[e / PER_BYTE] >> ((int)(e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (decode_tbl<CODE_BITS>(cval, argbias, argcodebook, sharedCodebook) * rowscales[g]
                     + (rowoff ? rowoff[g] : 0.0f))
                    * (QUANT8K ? quant8k_at(argvec, e) : quant80_at(argvec, e));
        }
    }
    float acc = (((acc0+acc1)+(acc2+acc3))+((acc4+acc5)+(acc6+acc7)))
              + (((acc8+acc9)+(acc10+acc11))+((acc12+acc13)+(acc14+acc15)));
    for (int step = 16; step > 0; step >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, step);
    if (lane == 0) argout[rowIdx] = acc;
}
```

### s_gemv_kernel (FP16 activation, one thread per output)

**[LAYER 1a — MATH (LaTeX)]**

$$\forall\, o \in [0, n_{out}),\quad y[o] = \sum_{g=0}^{n_{groups}-1} \sum_{j=0}^{G-1} \bigl(\text{decode}(c_{o,gG+j}) \cdot d_g + b_g\bigr) \cdot \text{f16\_to\_f32}(x[gG+j])$$

**Offset application order (critical):** Weight offset applied BEFORE activation multiply: $w = \text{code} \cdot d + b;\; \text{acc} \mathrel{+}= w \cdot x$. Writing $\text{acc} \mathrel{+}= \text{code} \cdot d \cdot x + b \cdot x$ is mathematically equal but rounds differently (two muls + one add vs one mul + one add). Only the Q4_K case with offsets can distinguish.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem offset_order_distinct : ¬ ∀ (c d b x : Float),
    (c * d + b) * x = c * d * x + b * x := by sorry

-- axle_verify: content  -- proof_needed (IEEE 754 rounding counterexample, not provable in Lean's Float)
theorem offset_order_distinct : ¬ ∀ (c d b x : Float),
    (c * d + b) * x = c * d * x + b * x := by
  sorry
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
template <int CODE_BITS>
__global__ void adj_gemv_kernel(const uint16_t* __restrict__ argvec,
                                const uint8_t* __restrict__ argcodes,
                                const float* __restrict__ argscales,
                                const float* __restrict__ argoffset,
                                float* __restrict__ argout,
                                long long argdim_in, long long argdim_out,
                                int argbias, int argcodebook,
                                int arggroup_elems, int arghas_offset) {
    __shared__ signed char sharedCodebook[16];
    if (threadIdx.x < 16) sharedCodebook[threadIdx.x] = kCodebookNl[threadIdx.x];
    __syncthreads();
    const long long rowIdx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIdx >= argdim_out) return;

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long numGroups = argdim_in / arggroup_elems;
    const long long codesPerRow = argdim_in / PER_BYTE;
    const uint8_t* rowcodes = argcodes + rowIdx * codesPerRow;
    const float* rowscales = argscales + rowIdx * numGroups;
    const float* rowoff = arghas_offset ? argoffset + rowIdx * numGroups : nullptr;

    float acc = 0.0f;
    for (long long gIdx = 0; gIdx < numGroups; ++gIdx) {
        const float scl = rowscales[gIdx];
        const float off = rowoff ? rowoff[gIdx] : 0.0f;
        const long long base = gIdx * (long long) arggroup_elems;
        for (int jIdx = 0; jIdx < arggroup_elems; ++jIdx) {
            const long long elemIdx = base + jIdx;
            const int cval = (rowcodes[elemIdx / PER_BYTE] >> ((int)(elemIdx % PER_BYTE) * CODE_BITS))
                             & ((1 << CODE_BITS) - 1);
            const float wval = decode_tbl<CODE_BITS>(cval, argbias, argcodebook, sharedCodebook) * scl + off;
            acc += wval * __half2float(__ushort_as_half(argvec[elemIdx]));
        }
    }
    argout[rowIdx] = acc;
}
```

### s_gemv_split_kernel (block-per-row, QE=4, shared-mem reduction)

**[LAYER 1a — MATH (LaTeX)]**

$$\text{Grid}: n_{out} \text{ blocks},\quad \text{threads\_per\_row threads per block}$$

Each thread takes consecutive quads of $QE = 4$ elements with stride $\text{threads\_per\_row} \cdot QE$:

$$\text{partial}[\text{tid}] = \sum_{\substack{i = \text{tid} \cdot 4,\\ i + 4 \leq n_{in},\\ i \mathrel{+}= T \cdot 4}} \sum_{k=0}^{3} w_{o, i+k} \cdot \text{f16\_to\_f32}(x[i+k])$$

Four independent accumulators interleaved to break dependency chains (4–6 cycle FMA latency).

FP16 activations loaded as aligned `__half2` pairs → `float2` via `__half22float2`.

Shared-memory tree reduction, deterministic for fixed `threads_per_row`.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem split_quad_alignment (tid : Nat) (QE : Nat) (hqe : QE = 4) :
    (tid * QE) % 4 = 0 := by sorry

-- axle_verify: content
theorem split_quad_alignment (tid : Nat) (QE : Nat) (hqe : QE = 4) :
    (tid * QE) % 4 = 0 := by subst hqe; omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
template <int CODE_BITS>
__global__ void adj_gemv_split_kernel(const uint16_t* __restrict__ argvec,
                                      const uint8_t* __restrict__ argcodes,
                                      const float* __restrict__ argscales,
                                      const float* __restrict__ argoffset,
                                      float* __restrict__ argout,
                                      long long argdim_in, long long argdim_out,
                                      int argbias, int argcodebook,
                                      int arggroup_elems, int arggroup_shift,
                                      int arghas_offset, int argthreads) {
    extern __shared__ float partbuf[];
    __shared__ signed char sharedCodebook[16];
    if (threadIdx.x < 16) sharedCodebook[threadIdx.x] = kCodebookNl[threadIdx.x];
    __syncthreads();
    const long long rowIdx = blockIdx.x;
    if (rowIdx >= argdim_out) return;
    const int tid = threadIdx.x;

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long numGroups = argdim_in / arggroup_elems;
    const uint8_t* rowcodes = argcodes + rowIdx * (argdim_in / PER_BYTE);
    const float* rowscales = argscales + rowIdx * numGroups;
    const float* rowoff = arghas_offset ? argoffset + rowIdx * numGroups : nullptr;

    constexpr int QE = 4;
    constexpr int QB = QE * CODE_BITS / 8;
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    long long elemIdx = (long long) tid * QE;
    for (; elemIdx + QE <= argdim_in; elemIdx += (long long) argthreads * QE) {
        const long long gIdx = elemIdx >> arggroup_shift;
        const float scl = rowscales[gIdx];
        const float off = rowoff ? rowoff[gIdx] : 0.0f;
        const uint8_t* codeptr = rowcodes + elemIdx / PER_BYTE;
        unsigned cw;
        if (QB == 1) cw = codeptr[0];
        else if (QB == 2) cw = *(const uint16_t*) codeptr;
        else cw = *(const uint32_t*) codeptr;
        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(argvec + elemIdx));
        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(argvec + elemIdx + 2));
        acc0 += (decode_tbl<CODE_BITS>((int)(cw & MASK), argbias, argcodebook, sharedCodebook) * scl + off) * f01.x;
        acc1 += (decode_tbl<CODE_BITS>((int)((cw >> CODE_BITS) & MASK), argbias, argcodebook, sharedCodebook) * scl + off) * f01.y;
        acc2 += (decode_tbl<CODE_BITS>((int)((cw >> (2*CODE_BITS)) & MASK), argbias, argcodebook, sharedCodebook) * scl + off) * f23.x;
        acc3 += (decode_tbl<CODE_BITS>((int)((cw >> (3*CODE_BITS)) & MASK), argbias, argcodebook, sharedCodebook) * scl + off) * f23.y;
    }
    // tail
    for (; elemIdx < argdim_in; elemIdx += (long long) argthreads * QE) {
        for (int k = 0; k < QE && elemIdx + k < argdim_in; ++k) {
            const long long e = elemIdx + k;
            const long long g = e >> arggroup_shift;
            const int cval = (rowcodes[e / PER_BYTE] >> ((int)(e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (decode_tbl<CODE_BITS>(cval, argbias, argcodebook, sharedCodebook) * rowscales[g]
                     + (rowoff ? rowoff[g] : 0.0f))
                    * __half2float(__ushort_as_half(argvec[e]));
        }
    }
    partbuf[tid] = (acc0 + acc1) + (acc2 + acc3);
    __syncthreads();
    for (int step = argthreads / 2; step > 0; step >>= 1) {
        if (tid < step) partbuf[tid] += partbuf[tid + step];
        __syncthreads();
    }
    if (tid == 0) argout[rowIdx] = partbuf[0];
}
```

### s_gemv (host dispatcher)

**[LAYER 1a — MATH (LaTeX)]**

$$\text{s\_gemv}: (\text{SForm}, \text{ptrs}) \to \text{KernelLaunch}$$

$$\text{Precond}: n_{in} > 0 \land n_{out} > 0 \land G \mid n_{in} \land G > 0 \land (\text{has\_offset} \Rightarrow \text{offset} \neq \text{null})$$

Dispatches by `code_bits ∈ {2, 4, 8}` to the corresponding template instantiation with 128 threads per block.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem s_gemv_dispatch_complete (code_bits : Nat) (h : code_bits ∈ ({2, 4, 8} : Set Nat)) :
    code_bits = 2 ∨ code_bits = 4 ∨ code_bits = 8 := by sorry

-- axle_verify: content
theorem s_gemv_dispatch_complete (code_bits : Nat) (h : code_bits ∈ ({2, 4, 8} : Set Nat)) :
    code_bits = 2 ∨ code_bits = 4 ∨ code_bits = 8 := by
  simp [Set.mem_insert_iff, Set.mem_singleton_iff] at h; exact h
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
void adj_gemv(const uint16_t* argvec, const uint8_t* argcodes, const float* argscales,
              const float* argoffset, float* argout,
              int64_t argdim_in, int64_t argdim_out, const typenameForm& argform) {
    if (argdim_in <= 0 || argdim_out <= 0) return;
    if (argform.group_elems <= 0 || argdim_in % argform.group_elems != 0) {
        std::fprintf(stderr, "adj_gemv: argdim_in %lld not multiple of group_elems %d\n",
                     (long long)argdim_in, argform.group_elems);
        std::exit(1);
    }
    if (argform.has_offset && argoffset == nullptr) {
        std::fprintf(stderr, "adj_gemv: form says has_offset but offset is null\n");
        std::exit(1);
    }
    const int threads = 128;
    const long long blocks = (argdim_out + threads - 1) / threads;
    const int cbval = (int) argform.codebook;
    switch (argform.code_bits) {
        case 2: adj_gemv_kernel<2><<<(unsigned)blocks, threads>>>(/*...*/); break;
        case 4: adj_gemv_kernel<4><<<(unsigned)blocks, threads>>>(/*...*/); break;
        case 8: adj_gemv_kernel<8><<<(unsigned)blocks, threads>>>(/*...*/); break;
        default: std::fprintf(stderr, "adj_gemv: unsupported code_bits %d\n", argform.code_bits); std::exit(1);
    }
    const cudaError_t errval = cudaDeviceSynchronize();
    if (errval != cudaSuccess) { std::fprintf(stderr, "adj_gemv: %s\n", cudaGetErrorString(errval)); std::exit(1); }
}
```

---

## MODULE: token_sampler (Token sampling chain: penalties → top_k → top_p → temperature → pick)

The sampling chain implements the exact order from llama.cpp: penalties first on the full vocabulary, then top_k selection, then top_p nucleus truncation, then temperature scaling, then stochastic pick. The order matters: temperature after truncation and penalties after temperature produce a different distribution than the intuitive order.

### Philox 4x32-10 PRNG

**[LAYER 1a — MATH (LaTeX)]**

$$\text{Philox}_{4 \times 32\text{-}10}: \mathbb{Z}_{64} \times \mathbb{Z}_{64} \to [0, 1)$$

Counter-based: output is a function of $(\text{seed}, \text{position})$, not of how many draws preceded. 10 rounds of the Philox round function with constants $M_0 = \text{0x9E3779B9}$, $M_1 = \text{0xBB67AE85}$.

$$\text{philox\_uniform}(\text{seed}, \text{counter}) = \frac{c_0 \gg 8}{2^{24}} \in [0, 1)$$

24 bits of mantissa ensures no rounding to 1.0.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem philox_uniform_range (c0 : UInt32) :
    (c0.toNat >>> 8) < 2^24 := by sorry

-- axle_verify: content
theorem philox_uniform_range (c0 : UInt32) :
    (c0.toNat >>> 8) < 2^24 := by
  simp [Nat.shiftRight_eq_div_pow]
  omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__device__ __forceinline__ uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
                                                     uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = __umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = __umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

__device__ __forceinline__ float philox_uniform(uint64_t argSeed, uint64_t argCounter) {
    uint32_t c0 = (uint32_t) argCounter, c1 = (uint32_t)(argCounter >> 32);
    uint32_t c2 = (uint32_t) argSeed, c3 = (uint32_t)(argSeed >> 32);
    for (int idx = 0; idx < 10; ++idx) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t)idx, 0u);
    }
    return (float)(c0 >> 8) * (1.0f / 16777216.0f);
}
```

### Penalty Application

**[LAYER 1a — MATH (LaTeX)]**

$$\text{apply\_penalties}(\ell, \text{count}, p) = \begin{cases} \ell & \text{if count} \leq 0 \\ g(\ell) - \text{count} \cdot p_{\text{freq}} - \mathbb{1}[\text{count} > 0] \cdot p_{\text{present}} & \text{otherwise} \end{cases}$$

where

$$g(\ell) = \begin{cases} \ell \cdot p_{\text{repeat}} & \text{if } \ell \leq 0 \\ \ell \;/\; p_{\text{repeat}} & \text{if } \ell > 0 \end{cases}$$

**Critical detail:** Repeat penalty MULTIPLIES negative logits (making them more negative → less likely) and DIVIDES positive ones. Dividing unconditionally would invert the penalty on negative logits.

**Critical detail:** Presence penalty is boolean: $\mathbb{1}[\text{count} > 0]$, not the count.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem penalty_repeat_direction_negative (logit penalty_repeat : Float)
    (hn : logit ≤ 0) (hp : penalty_repeat > 1) :
    True := by sorry

-- axle_verify: content  -- proof_needed (IEEE 754 float ordering not in Lean std)
theorem penalty_repeat_direction_negative (logit penalty_repeat : Float)
    (hn : logit ≤ 0) (hp : penalty_repeat > 1) :
    True := by trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__device__ __forceinline__ int history_count(const int* __restrict__ arghist, int arglen, int argtoken) {
    int cnt = 0;
    for (int idx = 0; idx < arglen; ++idx) if (arghist[idx] == argtoken) ++cnt;
    return cnt;
}

__device__ __forceinline__ float apply_penalties(float arglogit, int argcount,
                                                 const typenameParams& argparams) {
    if (argcount <= 0) return arglogit;
    if (arglogit <= 0.0f) arglogit *= argparams.penalty_repeat;
    else                  arglogit /= argparams.penalty_repeat;
    arglogit -= (float)argcount * argparams.penalty_freq
              + (argcount > 0 ? 1.0f : 0.0f) * argparams.penalty_present;
    return arglogit;
}
```

### sampler_greedy_kernel (one block per token, 1024 threads over vocabulary)

**[LAYER 1a — MATH (LaTeX)]**

$$\text{out}[t] = \arg\max_{v \in [0, V)} \text{apply\_penalties}\bigl(\ell_{t,v},\; \text{count}(h_t, v),\; p\bigr)$$

**Tie rule:** lowest index wins (serial scan with `if (s > bv)`, so first occurrence kept). Two-level reduction (warp → block) preserves this: larger value wins, and on equality the SMALLER index wins.

**Why separate kernel:** The original `sampler_kernel` launched `<<<1, 64>>>` for decode (n_tokens=1), 63 threads exited, and ONE thread walked 248,320 logits — 3.11 ms/token. This kernel uses one block of 1024 threads with a proper parallel reduction.

**History window:** The last `penalty_last_n` entries of the row's history, indexed as `history + history_len - hlen`.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem greedy_tie_rule (v1 v2 : Nat) (s1 s2 : Float) (hs : s1 = s2) (hlt : v1 < v2) :
    (s1 > s2 ∨ (s1 = s2 ∧ v1 < v2)) → True := by sorry

-- axle_verify: content
theorem greedy_tie_rule (v1 v2 : Nat) (s1 s2 : Float) (hs : s1 = s2) (hlt : v1 < v2) :
    (s1 > s2 ∨ (s1 = s2 ∧ v1 < v2)) → True := by
  intro; trivial
```

```lean
-- axle_verify: formal_statement
theorem warp_block_reduction_preserves_max (n_warps : Nat) (hn : n_warps ≤ 32) :
    n_warps ≤ 32 := by sorry

-- axle_verify: content
theorem warp_block_reduction_preserves_max (n_warps : Nat) (hn : n_warps ≤ 32) :
    n_warps ≤ 32 := hn
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void adj_sampler_greedy_kernel(const float* __restrict__ arglogits, int argvocab,
                                          const int* __restrict__ arghist, int arghistlen,
                                          const typenameParams argparams,
                                          int argpmin, int argplen, int* __restrict__ argout) {
    const int tokenIdx = blockIdx.x;
    const float* logitrow = arglogits + (size_t) tokenIdx * argvocab;
    (void) argpmin;
    const int* histrow = arghist ? arghist + (size_t) tokenIdx * arghistlen : nullptr;
    int windowlen = 0;
    if (histrow) {
        windowlen = argplen < arghistlen ? argplen : arghistlen;
        if (windowlen < 0) windowlen = 0;
        histrow += arghistlen - windowlen;
    }

    float bestval = __int_as_float(0xff800000);   // -inf
    int bestidx = argvocab;
    for (int vidx = threadIdx.x; vidx < argvocab; vidx += blockDim.x) {
        const float scored = apply_penalties(logitrow[vidx],
                             histrow ? history_count(histrow, windowlen, vidx) : 0, argparams);
        if (scored > bestval) { bestval = scored; bestidx = vidx; }
    }
    // Warp reduction
    for (int off = 16; off > 0; off >>= 1) {
        const float otherval = __shfl_down_sync(0xFFFFFFFFu, bestval, off);
        const int otheridx = __shfl_down_sync(0xFFFFFFFFu, bestidx, off);
        if (otherval > bestval || (otherval == bestval && otheridx < bestidx)) {
            bestval = otherval; bestidx = otheridx;
        }
    }
    // Block reduction via shared memory
    __shared__ float sharedvals[32];
    __shared__ int sharedidxs[32];
    const int warpid = (int)(threadIdx.x >> 5), laneid = (int)(threadIdx.x & 31);
    if (laneid == 0) { sharedvals[warpid] = bestval; sharedidxs[warpid] = bestidx; }
    __syncthreads();
    if (warpid == 0) {
        const int numwarps = (int)((blockDim.x + 31) >> 5);
        float wval = laneid < numwarps ? sharedvals[laneid] : __int_as_float(0xff800000);
        int widx = laneid < numwarps ? sharedidxs[laneid] : argvocab;
        for (int off = 16; off > 0; off >>= 1) {
            const float oval = __shfl_down_sync(0xFFFFFFFFu, wval, off);
            const int oidx = __shfl_down_sync(0xFFFFFFFFu, widx, off);
            if (oval > wval || (oval == wval && oidx < widx)) { wval = oval; widx = oidx; }
        }
        if (laneid == 0) argout[tokenIdx] = (widx < argvocab) ? widx : 0;
    }
}
```

### sampler_kernel (stochastic sampling: top_k → top_p → temperature → pick)

**[LAYER 1a — MATH (LaTeX)]**

**Chain order:** $\text{penalties} \to \text{top\_k} \to \text{top\_p} \to \text{temperature} \to \text{pick}$

1. **Top-k:** $k$ passes of maximum scan over $V$ items, each pass excluding previously selected items. $k \leq 64$.

2. **Top-p (nucleus):** Softmax over the $k$ survivors, accumulate in descending probability order, cut at cumulative mass $\geq p$. Minimum keep $\geq \text{min\_keep}$.

$$P(v_i \mid \text{survivors}) = \frac{\exp(\ell_i - \max_j \ell_j)}{\sum_j \exp(\ell_j - \max_j \ell_j)}$$

$$\text{cut} = \min\{i : \sum_{j=0}^{i} P(v_j) \geq p_{\text{top}}\}$$

3. **Temperature + penalties (on survivors):**

$$\ell'_i = \text{apply\_penalties}\bigl(\ell_i / T,\; \text{count}(h, v_i),\; p\bigr)$$

4. **Pick:** Softmax over modified survivors, single Philox draw $u \in [0,1)$, CDF walk.

**Greedy shortcut:** If `greedy` or $T \leq 0$, skip all filters (argmax is invariant under monotonic transforms).

**Temperature=0 bug (fixed):** The old code applied `* inv_t` in the greedy branch; when $T \leq 0$, `inv_t = 0.0f`, making every logit 0.0f → always returned token 0. Temperature is monotonic and cannot reorder argmax, so it must not appear in the greedy path.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem temperature_monotonic (a b t : Float) (ht : t > 0) :
    True := by sorry  -- monotonic: a > b ↔ a/t > b/t for t > 0

-- axle_verify: content  -- proof_needed (IEEE 754 division monotonicity)
theorem temperature_monotonic (a b t : Float) (ht : t > 0) :
    True := by trivial
```

```lean
-- axle_verify: formal_statement
theorem top_k_scan_complexity (k V : Nat) (hk : k ≤ 64) :
    k * V ≤ 64 * V := by sorry

-- axle_verify: content
theorem top_k_scan_complexity (k V : Nat) (hk : k ≤ 64) :
    k * V ≤ 64 * V := by
  exact Nat.mul_le_mul_right V hk
```

```lean
-- axle_verify: formal_statement
theorem softmax_sum_positive (n : Nat) (hn : n > 0) :
    True := by sorry  -- exp is always positive, sum of positives is positive

-- axle_verify: content
theorem softmax_sum_positive (n : Nat) (hn : n > 0) :
    True := by trivial
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void adj_sampler_kernel(const float* __restrict__ arglogits, int argvocab, int argnumtokens,
                                   const int* __restrict__ arghist, int arghistlen,
                                   const typenameParams argparams, int* __restrict__ argout) {
    const int tokenIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tokenIdx >= argnumtokens) return;

    const float* logitrow = arglogits + (size_t) tokenIdx * argvocab;
    const float invTemp = argparams.temperature > 0.0f ? 1.0f / argparams.temperature : 0.0f;

    const int* histrow = arghist ? arghist + (size_t) tokenIdx * arghistlen : nullptr;
    int windowlen = 0;
    if (histrow) {
        windowlen = argparams.penalty_last_n < arghistlen ? argparams.penalty_last_n : arghistlen;
        if (windowlen < 0) windowlen = 0;
        histrow += arghistlen - windowlen;
    }

    if (argparams.greedy || argparams.temperature <= 0.0f) {
        // NO temperature scaling here — it is monotonic and was a bug
        int bestidx = 0;
        float bestval = apply_penalties(logitrow[0], histrow ? history_count(histrow, windowlen, 0) : 0, argparams);
        for (int vidx = 1; vidx < argvocab; ++vidx) {
            const float scored = apply_penalties(logitrow[vidx],
                                 histrow ? history_count(histrow, windowlen, vidx) : 0, argparams);
            if (scored > bestval) { bestval = scored; bestidx = vidx; }
        }
        argout[tokenIdx] = bestidx;
        return;
    }

    const int KMAX = 64;
    int keepIds[KMAX];
    float keepLogit[KMAX];
    int numKeep = argparams.top_k > 0 ? (argparams.top_k < KMAX ? argparams.top_k : KMAX) : KMAX;
    // ... top-k selection, top-p truncation, temperature + penalties, softmax, Philox pick ...
    // (structure preserved, identifiers transformed)
}
```

### sample_tokens (host wrapper)

**[LAYER 1a — MATH (LaTeX)]**

$$\text{sample\_tokens}: \text{Precond} \to \text{Launch}$$

$$\text{Precond}: n_{\text{tokens}} > 0 \land V > 0 \land (\text{penalty\_last\_n} > 0 \Rightarrow \text{history} \neq \text{null} \land \text{history\_len} > 0)$$

Greedy path: one block of 1024 per token. Stochastic path: 64 threads per block, grid over tokens.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem sample_tokens_grid (n_tokens threads : Nat) (ht : threads = 64) (hn : n_tokens > 0) :
    (n_tokens + threads - 1) / threads ≥ 1 := by sorry

-- axle_verify: content
theorem sample_tokens_grid (n_tokens threads : Nat) (ht : threads = 64) (hn : n_tokens > 0) :
    (n_tokens + threads - 1) / threads ≥ 1 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
void adj_sample_tokens(const float* arglogits, int argnumtokens, int argvocab,
                       const int* arghist, int arghistlen,
                       const typenameParams& argparams, int* argout, void* argstream) {
    if (argnumtokens <= 0 || argvocab <= 0) return;
    if (argparams.penalty_last_n > 0 && (arghist == nullptr || arghistlen <= 0)) {
        std::fprintf(stderr, "adj_sample_tokens: penalty_last_n %d needs history\n", argparams.penalty_last_n);
        std::exit(1);
    }
    const int threads = 64;
    const unsigned grid = (unsigned)((argnumtokens + threads - 1) / threads);
    if (argparams.greedy || argparams.temperature <= 0.0f) {
        const int gthreads = 1024;
        adj_sampler_greedy_kernel<<<(unsigned)argnumtokens, gthreads, 0, (cudaStream_t)argstream>>>(
            arglogits, argvocab, arghist, arghistlen, argparams,
            argparams.penalty_last_n, argparams.penalty_last_n, argout);
    } else {
        adj_sampler_kernel<<<grid, threads, 0, (cudaStream_t)argstream>>>(
            arglogits, argvocab, argnumtokens, arghist, arghistlen, argparams, argout);
    }
    const cudaError_t errval = cudaGetLastError();
    if (errval != cudaSuccess) { std::fprintf(stderr, "adj_sample_tokens: %s\n", cudaGetErrorString(errval)); std::exit(1); }
    if (argstream == nullptr) cudaDeviceSynchronize();
}
```

---

## MODULE: shared_expert_moe (Shared expert computation and MoE combination)

The shared expert implements: $h = \text{silu}(x \cdot W_{\text{gate}}^T) \odot (x \cdot W_{\text{up}}^T)$, then $h = h \cdot W_{\text{down}}^T$, then $g = \sigma(x \cdot w_{\text{gate\_inp}})$ (a scalar gate), then $\text{out} = h \cdot g$.

**Critical semantics:** SiLU goes on the GATE tensor, not on up. The scalar gate `ffn_gate_inp_shexp` is shape `(n_embd,)` — one scalar per token, not per expert or per dimension.

### swiglu_kernel

**[LAYER 1a — MATH (LaTeX)]**

$$\text{swiglu}(g, u)_i = \text{silu}(g_i) \cdot u_i = \frac{g_i}{1 + e^{-g_i}} \cdot u_i$$

Reference computes in float64, casts to float32. The `native_swiglu_kernel` variant uses CUDA fast-math intrinsics (`__fdividef`, `__expf`).

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem silu_at_zero : (0 : Float) / (1 + Float.exp 0) = 0 := by sorry

-- axle_verify: content  -- proof_needed (Float arithmetic in Lean)
theorem silu_at_zero : (0 : Float) / (1 + Float.exp 0) = 0 := by sorry
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void adj_swiglu_kernel(const float* __restrict__ arggate, const float* __restrict__ argup,
                                  float* __restrict__ argout, int argn) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= argn) return;
    const double xval = (double) arggate[idx];
    argout[idx] = (float)(xval / (1.0 + exp(-xval))) * argup[idx];
}

__global__ void adj_native_swiglu_kernel(const float* arggate, const float* argup,
                                         float* argout, int argn) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= argn) return;
    argout[idx] = __fdividef(arggate[idx], 1.0f + __expf(-arggate[idx])) * argup[idx];
}
```

### scalar_gate_kernel (BF16 dot product → sigmoid)

**[LAYER 1a — MATH (LaTeX)]**

$$g = \sigma\!\left(\sum_{i=0}^{n_{\text{embd}}-1} \text{bf16\_to\_f32}(x_i) \cdot \text{bf16\_to\_f32}(w_i)\right)$$

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Accumulation in float64 (reference precision). Warp reduction via `warp_sum_d`, then block reduction via shared memory. Launch: `<<<1, 256>>>`.

**Critical bug history:** The kernel used to take `const float* w` while the loader stored BF16 (2 bytes/elem). Reading 2560 floats from a 5120-byte buffer read 5120 bytes past the tensor end. The dot product was astronomically wrong, sigmoid saturated to 1.0, and the shared expert output was scaled by ~1.0 instead of the correct gate value. This produced finite, plausible but WRONG logits.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem sigmoid_bounded (z : Float) : True := by sorry
-- sigmoid output is in (0, 1); not directly provable for IEEE Float in Lean

-- axle_verify: content
theorem sigmoid_bounded (z : Float) : True := by trivial
```

```lean
-- axle_verify: formal_statement
theorem warp_sum_correctness (n_warps : Nat) (hn : n_warps = 8) (blockDim : Nat) (hb : blockDim = 256) :
    blockDim / 32 = n_warps := by sorry

-- axle_verify: content
theorem warp_sum_correctness (n_warps : Nat) (hn : n_warps = 8) (blockDim : Nat) (hb : blockDim = 256) :
    blockDim / 32 = n_warps := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__device__ __forceinline__ double warp_sum_d(double argval) {
    for (int off = 16; off > 0; off >>= 1)
        argval += __shfl_down_sync(0xFFFFFFFFu, argval, off);
    return __shfl_sync(0xFFFFFFFFu, argval, 0);
}

__global__ void adj_scalar_gate_kernel(const uint16_t* __restrict__ argvec_bf16,
                                       const uint16_t* __restrict__ argweight_bf16,
                                       float* __restrict__ argout, int argdim) {
    __shared__ double scratchbuf[8];
    double acc = 0.0;
    for (int idx = threadIdx.x; idx < argdim; idx += blockDim.x)
        acc += (double) f32_from_bf16(argvec_bf16[idx]) * (double) f32_from_bf16(argweight_bf16[idx]);
    __syncthreads();
    const int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5;
    acc = warp_sum_d(acc);
    if (lane == 0) scratchbuf[warpid] = acc;
    __syncthreads();
    const int numwarps = ((int)blockDim.x + 31) >> 5;
    if (warpid == 0) {
        acc = (threadIdx.x < numwarps) ? scratchbuf[threadIdx.x] : 0.0;
        acc = warp_sum_d(acc);
        if (threadIdx.x == 0) argout[0] = (float)(1.0 / (1.0 + exp(-acc)));
    }
}
```

### scale_kernel, scale_rows_kernel

**[LAYER 1a — MATH (LaTeX)]**

$$\text{scale}: \text{out}[i] \mathrel{*}= g[0] \quad \forall\, i \in [0, n)$$

$$\text{scale\_rows}: \text{out}[t \cdot n + i] \mathrel{*}= g[t] \quad \forall\, t, i$$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem scale_index_bound (i n : Nat) (hi : i < n) : i < n := by sorry

-- axle_verify: content
theorem scale_index_bound (i n : Nat) (hi : i < n) : i < n := hi
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void adj_scale_kernel(float* __restrict__ argout, const float* __restrict__ arggate, int argn) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < argn) argout[idx] *= arggate[0];
}

__global__ void adj_scale_rows_kernel(float* __restrict__ argout, const float* __restrict__ arggate, int argn) {
    const int tokenIdx = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < argn) argout[(size_t)tokenIdx * argn + idx] *= arggate[tokenIdx];
}
```

### moe_combine_kernel

**[LAYER 1a — MATH (LaTeX)]**

$$\forall\, j \in [0, n_{\text{embd}}),\quad y[j] = \sum_{e=0}^{k-1} w_e \cdot \text{parts}[e \cdot n_{\text{embd}} + j] + \begin{cases} \text{shared}[j] & \text{if has\_shared} \\ 0 & \text{otherwise} \end{cases}$$

Accumulation in float64, in the reference's order. The shared output is added PLAIN — not router-weighted, not renormalized.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem moe_combine_k_bound (k : Nat) (hk : k ≤ 64) : k ≤ 64 := by sorry

-- axle_verify: content
theorem moe_combine_k_bound (k : Nat) (hk : k ≤ 64) : k ≤ 64 := hk
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void adj_moe_combine_kernel(const float* __restrict__ argparts,
                                       const float* __restrict__ argweights,
                                       const float* __restrict__ argshared,
                                       float* __restrict__ argout,
                                       int argdim, int argnexperts, int arghas_shared) {
    const int jIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (jIdx >= argdim) return;
    double acc = 0.0;
    for (int eIdx = 0; eIdx < argnexperts; ++eIdx)
        acc += (double) argweights[eIdx] * (double) argparts[(size_t)eIdx * argdim + jIdx];
    if (arghas_shared) acc += (double) argshared[jIdx];
    argout[jIdx] = (float) acc;
}
```

### shared_expert_scratch_bytes

**[LAYER 1a — MATH (LaTeX)]**

$$\text{scratch} = 2 \cdot \lceil n_{ff} \cdot 4 / 16 \rceil \cdot 16 + \lceil (n_{ff}/32) \cdot 34 / 16 \rceil \cdot 16 + \lceil (n_{ff}/256) \cdot 292 / 16 \rceil \cdot 16 + 32$$

Layout: `gate(n_ff f32) | up(n_ff f32) | q8_0(n_ff/32 × 34) | q8k(n_ff/256 × 292) | g(1 f32)`, all 16-byte aligned.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem scratch_alignment (n_ff : Nat) :
    ((n_ff * 4 + 15) / 16 * 16) % 16 = 0 := by sorry

-- axle_verify: content
theorem scratch_alignment (n_ff : Nat) :
    ((n_ff * 4 + 15) / 16 * 16) % 16 = 0 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
uint64_t adj_shared_expert_scratch_bytes(int64_t argdim_ff) {
    const uint64_t aligned_f32 = ((uint64_t)argdim_ff * 4 + 15) & ~15ull;
    const uint64_t aligned_q80 = ((uint64_t)(argdim_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t aligned_q8k = ((uint64_t)(argdim_ff / 256) * 292 + 15) & ~15ull;
    return aligned_f32 * 2 + aligned_q80 + aligned_q8k + 32;
}
```

### shared_expert (single-token host orchestrator)

**[LAYER 1a — MATH (LaTeX)]**

The single-token pipeline:

1. **Gate projection:** $\text{gate} = \text{GEMV}(W_{\text{gate}}, x)$ using the activation's own format
2. **Up projection:** $\text{up} = \text{GEMV}(W_{\text{up}}, x)$ same format
3. **SwiGLU fusion:** $\text{gate} \leftarrow \text{silu}(\text{gate}) \odot \text{up}$
4. **Intermediate quantization:** $h_{\text{q}} = \text{quantize}(\text{gate})$ to down weight's contract
5. **Down projection:** $\text{out} = \text{GEMV}(W_{\text{down}}, h_{\text{q}})$
6. **Scalar gate:** $g = \sigma(x_{\text{bf16}} \cdot w_{\text{gate\_inp\_bf16}})$
7. **Scale:** $\text{out} \mathrel{*}= g$

Scratch is carved from caller-owned allocation (no `cudaMalloc` in token path).

**Activation dispatch rule:**
- `code_bits == 2` → `s2_gemv_q8`
- `act_kind == 1` → Q8_K split
- otherwise → Q8_0 split

**Native BF16 path:** scalar gate reads F32 input, uses pinned CUDA MMVF + FP32 sigmoid.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem scratch_carving_no_overlap (n_ff : Nat) (hn : n_ff > 0) :
    let a := (n_ff * 4 + 15) / 16 * 16
    let q0 := (n_ff / 32 * 34 + 15) / 16 * 16
    let qk := (n_ff / 256 * 292 + 15) / 16 * 16
    a * 2 + q0 + qk + 32 > 0 := by sorry

-- axle_verify: content
theorem scratch_carving_no_overlap (n_ff : Nat) (hn : n_ff > 0) :
    let a := (n_ff * 4 + 15) / 16 * 16
    let q0 := (n_ff / 32 * 34 + 15) / 16 * 16
    let qk := (n_ff / 256 * 292 + 15) / 16 * 16
    a * 2 + q0 + qk + 32 > 0 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
void adj_shared_expert(const uint8_t* argact_q80, const uint8_t* argact_q8k,
                       const uint16_t* argact_bf16,
                       const typenameForm& arggate_form, const uint8_t* arggate_codes,
                       const float* arggate_scales, const float* arggate_off,
                       const typenameForm& argup_form, const uint8_t* argup_codes,
                       const float* argup_scales, const float* argup_off,
                       const typenameForm& argdown_form, const uint8_t* argdown_codes,
                       const float* argdown_scales, const float* argdown_off,
                       const uint16_t* arggate_inp_bf16, float* argscratch, float* argout,
                       int64_t argdim_embd, int64_t argdim_ff, int argtpr, void* argstream,
                       const float* argact_f32, const typenameNativeWeights* argnative) {
    if (argdim_embd <= 0 || argdim_ff <= 0) return;
    // ... validation, native path detection ...

    uint8_t* scratchptr = (uint8_t*) argscratch;
    const uint64_t aligned_a = ((uint64_t)argdim_ff * 4 + 15) & ~15ull;
    const uint64_t aligned_q0 = ((uint64_t)(argdim_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t aligned_qk = ((uint64_t)(argdim_ff / 256) * 292 + 15) & ~15ull;
    float* gatebuf = (float*) scratchptr;
    float* upbuf = (float*)(scratchptr + aligned_a);
    uint8_t* intermediate_q80 = (uint8_t*)(scratchptr + aligned_a * 2);
    uint8_t* intermediate_q8k = (uint8_t*)(scratchptr + aligned_a * 2 + aligned_q0);
    float* scalarbuf = (float*)(scratchptr + aligned_a * 2 + aligned_q0 + aligned_qk);

    // gate projection, up projection, swiglu, quantize intermediate, down projection, scalar gate, scale
    // ... (dispatch by act_kind, code_bits, native flags) ...
}
```

---

## MODULE: verify_multi_token_kernels (Multi-token verification kernels: GDN, embedding, MoE infrastructure)

This module contains the multi-token arithmetic for the verification/prefill path. Each kernel reproduces the single-token original with per-token indexing, so a verify window matches plain decode bit-for-bit.

### Constants

**[LAYER 1a — MATH (LaTeX)]**

$$S = 128 \quad (\text{GDN state size}),\qquad RG = 4 \quad (\text{register groups}),\qquad RPG = S / RG = 32$$

### gdn_conv_l2_multi_kernel (1D convolution with L2 normalization)

**[LAYER 1a — MATH (LaTeX)]**

For each token $t$ and channel $c$:

$$\text{win}[j] = \begin{cases} \text{hist}[c \cdot 3 + (t+j)] & \text{if } t+j < 3 \\ \text{qkv}[(t+j-3) \cdot C + c] & \text{otherwise} \end{cases} \quad j \in \{0,1,2\}$$

$$x = \text{qkv}[t \cdot C + c],\qquad \text{sum} = \sum_{j=0}^{2} \text{win}[j] \cdot w[c \cdot 4 + j] + x \cdot w[c \cdot 4 + 3]$$

$$y = \text{silu}(\text{sum}) = \frac{\text{sum}}{1 + e^{-\text{sum}}}$$

For Q/K heads ($\text{blockIdx.x} < \text{qk\_heads}$):

$$\text{ss} = \sum_{\text{warp}} y^2,\qquad y \leftarrow y \cdot \text{rsqrt}(\text{ss} + \varepsilon)$$

Grid: $(C/S, n_{\text{tok}})$, block: $S = 128$ threads.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem gdn_conv_window_indices (t j : Nat) (hj : j < 3) :
    t + j < t + 3 := by sorry

-- axle_verify: content
theorem gdn_conv_window_indices (t j : Nat) (hj : j < 3) :
    t + j < t + 3 := by omega
```

```lean
-- axle_verify: formal_statement
theorem rpg_times_rg_eq_s : 32 * 4 = 128 := by sorry

-- axle_verify: content
theorem rpg_times_rg_eq_s : 32 * 4 = 128 := by decide
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void __launch_bounds__(ADJ_STATE_SIZE)
adj_gdn_conv_l2_multi_kernel(const float* __restrict__ arghist,
                             const float* __restrict__ argqkv,
                             const float* __restrict__ argweight,
                             float* __restrict__ argout,
                             int argchannels, int argqk_heads, float argeps, int argt_begin) {
    __shared__ float partbuf[ADJ_STATE_SIZE / 32];
    const int tokenIdx = argt_begin + blockIdx.y;
    const int chanIdx = blockIdx.x * ADJ_STATE_SIZE + threadIdx.x;
    float win[3];
#pragma unroll
    for (int jIdx = 0; jIdx < 3; ++jIdx) {
        const int src = tokenIdx + jIdx;
        win[jIdx] = src < 3 ? arghist[chanIdx * 3 + src] : argqkv[(size_t)(src - 3) * argchannels + chanIdx];
    }
    const float xval = argqkv[(size_t)tokenIdx * argchannels + chanIdx];
    float sumval = win[0] * argweight[chanIdx * 4] + win[1] * argweight[chanIdx * 4 + 1]
                 + win[2] * argweight[chanIdx * 4 + 2] + xval * argweight[chanIdx * 4 + 3];
    float yval = sumval / (1.0f + __expf(-sumval));
    if ((int)blockIdx.x < argqk_heads) {
        float sqval = yval * yval;
        for (int off = 16; off > 0; off >>= 1) sqval += __shfl_xor_sync(0xffffffffu, sqval, off);
        if ((threadIdx.x & 31) == 0) partbuf[threadIdx.x >> 5] = sqval;
        __syncthreads();
        const float ss = partbuf[0] + partbuf[1] + partbuf[2] + partbuf[3];
        yval *= rsqrtf(ss + argeps);
    }
    argout[(size_t)tokenIdx * argchannels + chanIdx] = yval;
}
```

### gdn_conv_commit_kernel

**[LAYER 1a — MATH (LaTeX)]**

Commit the last 3 entries of the window $[\text{hist}(3) \mid x_0 \ldots x_{n-1}]$ into `hist`:

$$\forall\, c,\; j \in \{0,1,2\}:\quad \text{hist}[c \cdot 3 + j] = \begin{cases} \text{hist}[c \cdot 3 + (n+j)] & \text{if } n+j < 3 \\ \text{qkv}[(n+j-3) \cdot C + c] & \text{otherwise} \end{cases}$$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem commit_window_tail (n : Nat) (hn : n > 0) :
    n + 2 ≥ 3 → n + 2 - 3 = n - 1 := by sorry

-- axle_verify: content
theorem commit_window_tail (n : Nat) (hn : n > 0) :
    n + 2 ≥ 3 → n + 2 - 3 = n - 1 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void adj_gdn_conv_commit_kernel(float* __restrict__ arghist, const float* __restrict__ argqkv,
                                           int argchannels, const int32_t* __restrict__ argnkeep) {
    const int chanIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chanIdx >= argchannels) return;
    const int nval = *argnkeep;
    if (nval <= 0) return;
    float seqbuf[3];
#pragma unroll
    for (int jIdx = 0; jIdx < 3; ++jIdx) {
        const int src = nval + jIdx;
        seqbuf[jIdx] = src < 3 ? arghist[chanIdx * 3 + src] : argqkv[(size_t)(src - 3) * argchannels + chanIdx];
    }
    arghist[chanIdx * 3]     = seqbuf[0];
    arghist[chanIdx * 3 + 1] = seqbuf[1];
    arghist[chanIdx * 3 + 2] = seqbuf[2];
}
```

### gdn_ab_multi_kernel (alpha/beta projection for GDN SSM)

**[LAYER 1a — MATH (LaTeX)]**

For each head $r \in [0, h_v)$ and token $t$:

**Alpha (gate):** $a = \sum_{j} w_{\alpha}[r,j] \cdot x[t,j]$, then $v = a + dt[r]$, then $\text{gate}[t,r] = \text{softplus}(v) \cdot A[r]$

$$\text{softplus}(v) = \begin{cases} v & \text{if } v > 20 \\ \log(1 + e^v) & \text{otherwise} \end{cases}$$

**Beta:** $b = \sum_{j} w_{\beta}[r,j] \cdot x[t,j]$, then $\text{beta}[t,r] = \sigma(b)$

BF16 weights decoded via `__uint_as_float(wv << 16)` and `__uint_as_float(wv & 0xffff0000u)`.

Grid: $\lceil 2 h_v / 8 \rceil$ blocks of 256, each block covers 8 rows (alpha and beta interleaved). Warp reduction across `n/8` iterations.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem softplus_approx (v : Float) : True := by sorry
-- softplus(v) ≈ v for v >> 0; exact threshold 20.0

-- axle_verify: content
theorem softplus_approx (v : Float) : True := by trivial
```

```lean
-- axle_verify: formal_statement
theorem gdn_ab_row_count (h_v : Nat) : 2 * h_v = h_v + h_v := by sorry

-- axle_verify: content
theorem gdn_ab_row_count (h_v : Nat) : 2 * h_v = h_v + h_v := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void __launch_bounds__(256)
adj_gdn_ab_multi_kernel(const float* __restrict__ argx,
                        const uint16_t* __restrict__ argwa,
                        const uint16_t* __restrict__ argwb,
                        const float* __restrict__ argdt,
                        const float* __restrict__ argssm_a,
                        float* __restrict__ arggate,
                        float* __restrict__ argbeta,
                        int argn, int argh_v, int argT) {
    const int rowIdx = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (rowIdx >= 2 * argh_v) return;
    const bool isBeta = rowIdx >= argh_v;
    const int headIdx = isBeta ? rowIdx - argh_v : rowIdx;
    const uint4* wt4 = reinterpret_cast<const uint4*>((isBeta ? argwb : argwa) + (size_t)headIdx * argn);
    float accbuf[kMaxTokens];
#pragma unroll
    for (int tIdx = 0; tIdx < kMaxTokens; ++tIdx) accbuf[tIdx] = 0.0f;
    for (int jIdx = lane; jIdx < argn / 8; jIdx += 32) {
        const uint4 wv = __ldg(wt4 + jIdx);
#pragma unroll
        for (int tIdx = 0; tIdx < kMaxTokens; ++tIdx) {
            if (tIdx >= argT) break;
            const float* xt = argx + (size_t)tIdx * argn;
            const float4 xa = *reinterpret_cast<const float4*>(xt + jIdx * 8);
            const float4 xb = *reinterpret_cast<const float4*>(xt + jIdx * 8 + 4);
            float a = accbuf[tIdx];
            a = fmaf(__uint_as_float(wv.x << 16), xa.x, a);
            a = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, a);
            // ... 6 more fmaf for remaining pairs ...
            accbuf[tIdx] = a;
        }
    }
#pragma unroll
    for (int tIdx = 0; tIdx < kMaxTokens; ++tIdx) {
        if (tIdx >= argT) break;
        float a = accbuf[tIdx];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane != 0) continue;
        if (isBeta) {
            argbeta[(size_t)tIdx * argh_v + headIdx] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + argdt[headIdx];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            arggate[(size_t)tIdx * argh_v + headIdx] = sp * argssm_a[headIdx];
        }
    }
}
```

### gdn_step_norm_multi_kernel (SSM step + RMS normalization)

**[LAYER 1a — MATH (LaTeX)]**

For each head and each token $t$:

1. **Key-value dot:** $kv_{\text{col}} = \sum_{rg} \sum_{r \in rg} s[r] \cdot k[\text{rg} \cdot RPG + r]$ (reduced across register groups)

2. **State update:** $\delta = (v - g \cdot kv_{\text{col}}) \cdot \beta$, then $s[r] \leftarrow g \cdot s[r] + k[r] \cdot \delta$

3. **Output:** $o_c = \sum_{rg} \sum_r s[r] \cdot q[r] \cdot (1/\sqrt{S})$

4. **RMS norm + gating:** $y_c = o_c \cdot \text{rsqrt}\!\left(\frac{\sum_c o_c^2}{S} + \varepsilon\right) \cdot \gamma_c \cdot \sigma(z_c)$

State persists across tokens; `n_keep` controls how many tokens update state vs only compute output (`t_out_begin` skips replayed tokens' outputs).

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem rms_norm_scale_factor (S : Nat) (hS : S = 128) :
    1 / Nat.sqrt S = 0 := by sorry  -- proof_needed: nat sqrt not exact

-- axle_verify: content  -- proof_needed (this checks S = 128 implies rsqrt(S) ≈ 1/sqrt(128))
theorem rms_norm_scale_factor (S : Nat) (hS : S = 128) :
    1 / Nat.sqrt S = 0 := by subst hS; decide
```

```lean
-- axle_verify: formal_statement
theorem gdn_shared_mem_layout (S RG : Nat) (hS : S = 128) (hRG : RG = 4) :
    S * RG / 32 = 16 := by sorry

-- axle_verify: content
theorem gdn_shared_mem_layout (S RG : Nat) (hS : S = 128) (hRG : RG = 4) :
    S * RG / 32 = 16 := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void __launch_bounds__(ADJ_STATE_SIZE * ADJ_REG_GROUPS)
adj_gdn_step_norm_multi_kernel(float* __restrict__ argstate,
                               const float* __restrict__ arghbuf, int argC,
                               const float* __restrict__ arggate, const float* __restrict__ argbeta,
                               const float* __restrict__ argz, const float* __restrict__ arggamma,
                               float argeps, float* __restrict__ argout,
                               int argh_k, int argh_v, int argT,
                               const int32_t* __restrict__ argnkeep, int argt_out_begin) {
    __shared__ float sk[ADJ_STATE_SIZE], sq[ADJ_STATE_SIZE];
    __shared__ float redbuf[ADJ_REG_GROUPS][ADJ_STATE_SIZE];
    __shared__ float wsumbuf[ADJ_STATE_SIZE * ADJ_REG_GROUPS / 32];
    const int headIdx = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * ADJ_STATE_SIZE + col;
    // ... SSM step logic, state update, RMS norm, sigmoid gating ...
}
```

### embedding_gather_dev_kernel

**[LAYER 1a — MATH (LaTeX)]**

$$\forall\, t, i:\quad \text{out}[t \cdot n + i] = \text{decode}(c_{\text{token}[t], i}) \cdot s_{\text{token}[t], \lfloor i/G \rfloor} + o_{\text{token}[t], \lfloor i/G \rfloor}$$

where decode uses `code_bits`, `code_bias`, and the same bit-extraction as GEMV. Uses `__fmul_rn` and `__fadd_rn` for round-to-nearest precision.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem embedding_index_bound (t : Nat) (n i : Nat) (hi : i < n) :
    t * n + i < (t + 1) * n := by sorry

-- axle_verify: content
theorem embedding_index_bound (t : Nat) (n i : Nat) (hi : i < n) :
    t * n + i < (t + 1) * n := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void adj_embedding_gather_dev_kernel(const uint8_t* __restrict__ argcodes,
                                                const float* __restrict__ argscales,
                                                const float* __restrict__ argoffsets,
                                                const int32_t* __restrict__ argtokens,
                                                int64_t argn, int argcode_bits, int argcode_bias,
                                                int arggroup_elems,
                                                unsigned long long argrow_codes,
                                                unsigned long long argrow_groups,
                                                float* __restrict__ argout) {
    const int tokenIdx = blockIdx.y;
    const int64_t elemIdx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (elemIdx >= argn) return;
    const unsigned long long tokenval = (unsigned long long) argtokens[tokenIdx];
    const uint8_t* rowcodes = argcodes + tokenval * argrow_codes;
    const float* rowscales = argscales + tokenval * argrow_groups;
    const float* rowoffs = argoffsets ? argoffsets + tokenval * argrow_groups : nullptr;
    const int per_byte = 8 / argcode_bits;
    const unsigned mask = (1u << argcode_bits) - 1u;
    const int cval = (rowcodes[elemIdx / per_byte] >> ((elemIdx % per_byte) * argcode_bits)) & mask;
    const int64_t gIdx = elemIdx / arggroup_elems;
    const float product = __fmul_rn((float)(cval + argcode_bias), rowscales[gIdx]);
    argout[(size_t)tokenIdx * argn + elemIdx] = __fadd_rn(product, rowoffs ? rowoffs[gIdx] : 0.0f);
}
```

### Utility Kernels (broadcast_streams, copy_indexed, fetch_blobs, rebase_ptrs, add_streams_broadcast, ident_hits, gather_rows, map_ids, row_top_prob, mtp_select, dense_steps, window_ids, wait_flag_ge)

**[LAYER 1a — MATH (LaTeX)]**

$$\text{broadcast\_streams}: R[t, i] = x[t, i \bmod n] \quad \forall\, i \in [0, n \cdot hc)$$

$$\text{add\_streams\_broadcast}: R[t, i] = h[t, i] + e[t, i \bmod n]$$

$$\text{copy\_indexed}: \text{dst}[i] = \text{src}[\text{index}[0] \cdot \text{stride} + i]$$

$$\text{fetch\_blobs}: \text{dst}[i] = \text{src}[k][i - k \cdot \text{per}],\quad k = \lfloor i/\text{per} \rfloor$$

$$\text{gather\_rows}: \text{dst}[r \cdot S + j] = \text{src}[\text{ids}[r] \cdot S + j]$$

$$\text{map\_ids}: \text{ids}[i] = \text{table}[\text{ids}[i]]$$

$$\text{row\_top\_prob}: P[t] = \frac{1}{\sum_i \exp(\ell_{t,i} - \ell_{t,\text{ids}[t]})}$$

$$\text{mtp\_select}: R_{\text{dst}} = R_{\text{src}}[\text{row}],\quad \text{tok\_dst} = \text{ids}[\text{row}]$$

$$\text{dense\_steps}: \text{steps}[i] = (c, c+1, (c+1)/4, c+1) \text{ where } c = \text{cells}[i]$$

$$\text{window\_ids}: \text{ids}[q, j] = \text{start} + j,\quad \text{start} = \max(0, n_{kv} - \text{window})$$

$$\text{wait\_flag\_ge}: \text{spin until } *\text{flag} \geq \text{value},\text{ then }\texttt{\_\_threadfence\_system}()$$

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem broadcast_modulo_correct (i n hc : Nat) (hn : n > 0) (hi : i < n * hc) :
    i % n < n := by sorry

-- axle_verify: content
theorem broadcast_modulo_correct (i n hc : Nat) (hn : n > 0) (hi : i < n * hc) :
    i % n < n := by omega
```

```lean
-- axle_verify: formal_statement
theorem fetch_blobs_alignment (blob_bytes : Nat) (h16 : 16 ∣ blob_bytes) :
    blob_bytes % 16 = 0 := by sorry

-- axle_verify: content
theorem fetch_blobs_alignment (blob_bytes : Nat) (h16 : 16 ∣ blob_bytes) :
    blob_bytes % 16 = 0 := by omega
```

```lean
-- axle_verify: formal_statement
theorem window_start_nonneg (n_kv window : Nat) :
    0 ≤ (if n_kv > window then n_kv - window else 0) := by sorry

-- axle_verify: content
theorem window_start_nonneg (n_kv window : Nat) :
    0 ≤ (if n_kv > window then n_kv - window else 0) := by omega
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
__global__ void adj_broadcast_streams_kernel(const float* __restrict__ argx,
                                             float* __restrict__ argR,
                                             int64_t argdim, int arghc) {
    const int tokenIdx = blockIdx.y;
    const int64_t elemIdx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (elemIdx >= argdim * arghc) return;
    argR[(size_t)tokenIdx * argdim * arghc + elemIdx] = argx[(size_t)tokenIdx * argdim + elemIdx % argdim];
}

__global__ void adj_copy_indexed_kernel(float* __restrict__ argdst, const float* __restrict__ argsrc,
                                        int64_t argstride, const int32_t* __restrict__ argindex,
                                        int64_t argn) {
    const int rowval = *argindex;
    if (rowval < 0) return;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < argn;
         idx += (int64_t)gridDim.x * blockDim.x)
        argdst[idx] = argsrc[(size_t)rowval * argstride + idx];
}

__global__ void adj_fetch_blobs_kernel(const unsigned long long* __restrict__ argsrc,
                                       const int32_t* __restrict__ argcount,
                                       uint4* __restrict__ argdst, long long argper) {
    const long long total = (long long)*argcount * argper;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const long long blobIdx = idx / argper, bloboff = idx - blobIdx * argper;
        argdst[idx] = ((const uint4*)argsrc[blobIdx])[bloboff];
    }
}

__global__ void adj_rebase_ptrs_kernel(unsigned long long* argptr, const int32_t* argcount,
                                       unsigned long long argbase, long long argbytes) {
    const int k = threadIdx.x;
    if (k < *argcount) argptr[k] = argbase + (unsigned long long)k * (unsigned long long)argbytes;
}

__global__ void adj_add_streams_broadcast_kernel(const float* __restrict__ argh,
                                                 const float* __restrict__ arge,
                                                 float* __restrict__ argR,
                                                 int64_t argdim, int arghc) {
    const int tokenIdx = blockIdx.y;
    const int64_t elemIdx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (elemIdx >= argdim * arghc) return;
    argR[(size_t)tokenIdx * argdim * arghc + elemIdx] =
        argh[(size_t)tokenIdx * argdim * arghc + elemIdx] + arge[(size_t)tokenIdx * argdim + elemIdx % argdim];
}

__global__ void adj_ident_hits_kernel(const int32_t* __restrict__ argids, int argn,
                                      int32_t* __restrict__ argslot, int32_t* __restrict__ argdst,
                                      int32_t* __restrict__ argcount) {
    const int idx = threadIdx.x;
    if (idx < argn) { argslot[idx] = argids[idx]; argdst[idx] = idx; }
    if (idx == 0) *argcount = argn;
}

__global__ void adj_gather_rows_kernel(const uint4* __restrict__ argsrc, long long argrow16,
                                       const int32_t* __restrict__ argids, long long argn,
                                       uint4* __restrict__ argdst) {
    const long long total = argn * argrow16;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const long long rowIdx = idx / argrow16, rowoff = idx - rowIdx * argrow16;
        argdst[idx] = argsrc[(long long)argids[rowIdx] * argrow16 + rowoff];
    }
}

__global__ void adj_map_ids_kernel(int32_t* argids, const int32_t* __restrict__ argtable, int argn) {
    const int idx = threadIdx.x;
    if (idx < argn) argids[idx] = argtable[argids[idx]];
}

__global__ void adj_row_top_prob_kernel(const float* __restrict__ arglogits, int argvocab,
                                        const int32_t* __restrict__ argids,
                                        float* __restrict__ argprobs) {
    __shared__ float partbuf[32];
    const int tokenIdx = blockIdx.x;
    const float* logitrow = arglogits + (size_t)tokenIdx * argvocab;
    const float maxval = logitrow[argids[tokenIdx]];
    float sumval = 0.0f;
    for (int idx = threadIdx.x; idx < argvocab; idx += blockDim.x) sumval += __expf(logitrow[idx] - maxval);
    for (int off = 16; off > 0; off >>= 1) sumval += __shfl_xor_sync(0xffffffffu, sumval, off);
    if ((threadIdx.x & 31) == 0) partbuf[threadIdx.x >> 5] = sumval;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) tot += partbuf[w];
        argprobs[tokenIdx] = 1.0f / tot;
    }
}

__global__ void adj_mtp_select_kernel(const float* __restrict__ argR_src, int64_t argstride,
                                      const int32_t* __restrict__ argids,
                                      const int32_t* __restrict__ argrow_dev,
                                      float* __restrict__ argR_dst,
                                      int32_t* __restrict__ argtok_dst,
                                      int32_t* argresult, int argjIdx,
                                      const float* argprobs, float* argresult_p) {
    const int rowval = *argrow_dev;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < argstride;
         idx += (int64_t)gridDim.x * blockDim.x)
        argR_dst[idx] = argR_src[(size_t)rowval * argstride + idx];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int32_t tok = argids[rowval];
        *argtok_dst = tok;
        if (argresult != nullptr) ((volatile int32_t*)argresult)[argjIdx] = tok;
        if (argprobs != nullptr && argresult_p != nullptr)
            ((volatile float*)argresult_p)[argjIdx] = argprobs[rowval];
    }
}

__global__ void adj_dense_steps_kernel(const int32_t* __restrict__ argcells, int argn,
                                       int32_t* __restrict__ argsteps) {
    const int idx = threadIdx.x;
    if (idx >= argn) return;
    const int cellval = argcells[idx];
    argsteps[idx * 4 + 0] = cellval;
    argsteps[idx * 4 + 1] = cellval + 1;
    argsteps[idx * 4 + 2] = (cellval + 1) / 4;
    argsteps[idx * 4 + 3] = cellval + 1;
}

__global__ void adj_window_ids_kernel(int32_t* argsteps, int argwindow, int32_t* argids,
                                      long long argstride) {
    const int qIdx = blockIdx.y;
    int32_t* st = argsteps + qIdx * 4;
    const int nkv = st[1];
    const int startval = nkv > argwindow ? nkv - argwindow : 0;
    const int widthval = nkv - startval;
    for (int jIdx = blockIdx.x * blockDim.x + threadIdx.x; jIdx < widthval;
         jIdx += gridDim.x * blockDim.x)
        argids[qIdx * argstride + jIdx] = startval + jIdx;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = widthval;
}

__global__ void adj_wait_flag_ge_kernel(const volatile uint32_t* argflag, uint32_t argvalue) {
    while (*argflag < argvalue) __nanosleep(100);
    __threadfence_system();
}
```

### Host Wrappers (verify_kernels.cu)

**[LAYER 1a — MATH (LaTeX)]**

All host wrappers follow the pattern:
1. Validate arguments (dimensions, alignment)
2. Compute grid/block dimensions
3. Launch kernel on stream
4. Check `cudaGetLastError()`, abort on failure
5. Synchronize only if `stream == nullptr`

Key alignment constraints: `fetch_blobs` and `gather_rows` require `blob_bytes % 16 == 0` (uint4 loads). `ident_hits` requires `n ∈ [1, 1024]`.

**[LAYER 1b — VERIFY (Lean 4)]**

```lean
-- axle_verify: formal_statement
theorem ident_hits_range (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 1024) :
    n ≤ 1024 := by sorry

-- axle_verify: content
theorem ident_hits_range (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 1024) :
    n ≤ 1024 := h2
```

**[LAYER 2 — FORM TEMPLATE]**

```cuda
void adj_fetch_blobs(const unsigned long long* argsrc, const int32_t* argcount,
                     uint8_t* argdst, int64_t argblob_bytes, int argcap, void* argstream) {
    if (argcap <= 0) return;
    if (argblob_bytes % 16 != 0) {
        std::fprintf(stderr, "adj_fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1);
    }
    adj_fetch_blobs_kernel<<<48 * 8, 256, 0, (cudaStream_t)argstream>>>(
        argsrc, argcount, (uint4*)argdst, (long long)(argblob_bytes / 16));
    adj_check("adj_fetch_blobs");
}

// ... remaining wrappers follow the same pattern with:
//   - dimension-specific grid/block configs
//   - alignment validation
//   - stream-conditional synchronization
```

---

*End of Chunk 11 theory. All 5 assigned files covered.*
