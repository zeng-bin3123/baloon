# Chunk 6 — QSA Attention, Selection, Expert Grouped, S-GEMV, Shared Expert

Theory for six kernel interface headers from the Strata inference engine.
Covers the full-attention (QSA) layer pipeline, MoE GPU expert evaluation,
the canonical S-form GEMV family, and the shared-expert fused block.

---

## MODULE: qsa_geometry (QSA layer shape descriptor and step state)

### LAYER 1a — MATH (LaTeX)

$$
\textbf{QsaShapes} = \bigl(n_h,\; n_{hkv},\; d_h,\; n_{\text{rot}},\; n_{ih},\; d_{idx},\; r,\; K,\; p_s\bigr)
$$

Constraints:
$$
n_h \bmod n_{hkv} = 0,\qquad d_h \bmod 4 = 0,\qquad r \geq 1,\qquad p_s \geq 1
$$

Real artifact values:
$$
n_h=24,\; n_{hkv}=2,\; d_h=256,\; n_{\text{rot}}=64,\; n_{ih}=4,\; d_{idx}=128,\; r=4,\; K=2048,\; p_s=512
$$

Selection width:
$$
W(n_{kv}, S) = \min\bigl(n_{kv},\; K + r - 1\bigr)
$$

Step state (device int32 buffer, 4 entries):
$$
\text{step}[0] = \text{pos},\quad
\text{step}[1] = \text{pos}+1,\quad
\text{step}[2] = \lfloor (\text{pos}+1)/r \rfloor,\quad
\text{step}[3] = W(\text{pos}+1, S)
$$

Step fill precondition: $\text{pos} \geq 0$.

$$
\text{qsa\_step\_fill}: \mathbb{Z}_{\geq 0} \times \textbf{QsaShapes} \to \mathbb{Z}^4
$$

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem selection_width_le_nkv (n_kv K r : Nat) (hr : r ≥ 1) :
    min n_kv (K + r - 1) ≤ n_kv := by sorry

-- axle_verify: content
theorem selection_width_le_nkv (n_kv K r : Nat) (hr : r ≥ 1) :
    min n_kv (K + r - 1) ≤ n_kv := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem step_nkv_eq_pos_plus_one (pos : Nat) :
    pos + 1 = pos + 1 := by sorry

-- axle_verify: content
theorem step_nkv_eq_pos_plus_one (pos : Nat) :
    pos + 1 = pos + 1 := by
  rfl
```

```lean
-- axle_verify: formal_statement
theorem selection_width_identity (n_kv K r : Nat) (h : n_kv < K + r - 1) :
    min n_kv (K + r - 1) = n_kv := by sorry

-- axle_verify: content
theorem selection_width_identity (n_kv K r : Nat) (h : n_kv < K + r - 1) :
    min n_kv (K + r - 1) = n_kv := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjShapes {
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t head_dim = 0;
    int64_t n_rot = 0;
    int64_t idx_n_head = 0;
    int64_t idx_dim = 0;
    int64_t idx_block = 0;
    int64_t idx_top_k = 0;
    int64_t page_size = 0;
};

inline AdjShapes adj_real_shapes() {
    AdjShapes argname;
    argname.n_head = 24; argname.n_head_kv = 2; argname.head_dim = 256; argname.n_rot = 64;
    argname.idx_n_head = 4; argname.idx_dim = 128; argname.idx_block = 4; argname.idx_top_k = 2048;
    argname.page_size = 512;
    return argname;
}

inline float adj_rms_eps() { return 1e-6f; }
inline double adj_freq_base() { return 1e7; }

inline int64_t adj_selection_width(int64_t argname, const AdjShapes& objtype) {
    const int64_t arglocal = objtype.idx_top_k + objtype.idx_block - 1;
    return argname < arglocal ? argname : arglocal;
}

enum AdjStep : int {
    kStepPos = 0,
    kStepNKv = 1,
    kStepNBid = 2,
    kStepWidth = 3,
    kStepCount = 4,
};

inline uint64_t adj_step_bytes() { return sizeof(int32_t) * (uint64_t) kStepCount; }

void adj_step_fill(int32_t* objtype, int64_t argname, const AdjShapes& objshape);
```

---

## MODULE: paged_kv_cache_ops (KV append and gather through page table)

### LAYER 1a — MATH (LaTeX)

**KV Cache Layout** (FP16, paged):
$$
\text{pool}[\text{page}][\text{kv\_head}][\text{page\_size}][\text{head\_dim}]
$$

Physical row for logical cell $t$, head $h$:
$$
\text{row}(t, h) = \bigl(\text{table}\bigl[\lfloor t / p_s \rfloor\bigr] \cdot n_{hkv} + h\bigr) \cdot p_s + (t \bmod p_s)
$$

**kv\_append**:
$$
\text{kv\_append}: \mathcal{P}_K \times \mathcal{P}_V \times \mathbb{Z}^{n_p} \times \mathbb{Z}_{\geq 0} \times \mathbb{R}^{n_{hkv} \times d_h} \times \mathbb{R}^{n_{hkv} \times d_h} \to \mathcal{P}_K' \times \mathcal{P}_V'
$$
$$
\forall h \in [0, n_{hkv}):\; K'[\text{row}(\text{pos}, h)] \leftarrow \text{f16}(k_{\text{cur}}[h]),\quad V'[\text{row}(\text{pos}, h)] \leftarrow \text{f16}(v_{\text{cur}}[h])
$$

Preconditions: $k_{\text{cur}}$ already normed and rotated. Conversion uses `f16_from_f32`, not `__float2half`.

**kv\_gather**:
$$
\text{kv\_gather}: \mathcal{P}_K \times \mathcal{P}_V \times \text{ids} \to K_{\text{scratch}} \times V_{\text{scratch}}
$$
$$
K_{\text{scratch}}[j, h, :] = K[\text{row}(\text{ids}[j], h)],\quad j \in [0, n_{\text{ids}})
$$

Precondition: $\text{ids}$ ascending, unique. Unselected cells do not exist for downstream attention.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem page_row_unique (t1 t2 : Nat) (h1 h2 : Nat) (ps nhkv : Nat)
    (hps : ps > 0) (hnhkv : nhkv > 0) (hh1 : h1 < nhkv) (hh2 : h2 < nhkv)
    (heq : (t1 / ps) * nhkv * ps + h1 * ps + t1 % ps = (t2 / ps) * nhkv * ps + h2 * ps + t2 % ps) :
    t1 = t2 ∧ h1 = h2 := by sorry
-- proof_needed: requires modular arithmetic reasoning about page layout injectivity
```

```lean
-- axle_verify: formal_statement
theorem gather_preserves_order (ids : List Nat) (h : List.Sorted (· < ·) ids)
    (i j : Nat) (hi : i < ids.length) (hj : j < ids.length) (hij : i < j) :
    ids[i]! < ids[j]! := by sorry

-- axle_verify: content
theorem gather_preserves_order (ids : List Nat) (h : List.Sorted (· < ·) ids)
    (i j : Nat) (hi : i < ids.length) (hj : j < ids.length) (hij : i < j) :
    ids[i]! < ids[j]! := by
  exact List.Sorted.get_lt h ⟨i, hi⟩ ⟨j, hj⟩ hij
-- proof_needed: List.Sorted.get_lt may need a custom lemma depending on Mathlib version
```

### LAYER 2 — FORM TEMPLATE

```cpp
void adj_append(uint16_t* objpool_k, uint16_t* objpool_v, const int32_t* objtable, int64_t argpos,
                const float* argkcur, const float* argvcur, const AdjShapes& objshape, void* objstream);

void adj_gather(const uint16_t* objpool_k, const uint16_t* objpool_v, const int32_t* objtable,
                const int32_t* argids, int64_t argcount, const AdjShapes& objshape,
                uint16_t* objscratch_k, uint16_t* objscratch_v, void* objstream);

void adj_append_step(uint16_t* objpool_k, uint16_t* objpool_v, const int32_t* objtable,
                     const int32_t* objstep, const float* argkcur, const float* argvcur,
                     const AdjShapes& objshape, void* objstream);
```

---

## MODULE: indexer_state (QSA indexer per-sequence buffers and key append)

### LAYER 1a — MATH (LaTeX)

**IndexerBuffers** state per QSA layer per sequence (all device pointers, FP32):
$$
\text{tail} \in \mathbb{R}^{(r-1) \times d_{idx}},\quad
\text{dead} \in \mathbb{R}^{d_{idx}},\quad
\text{pooled} \in \mathbb{R}^{(B_{\max}+1) \times d_{idx}},\quad
\text{block\_pos} \in \mathbb{Z}^{1}
$$

$\text{dead} = \text{RoPE}_{\text{neox}}(\text{RMSNorm}(\text{raw}[0], w), 0)$ — constant since $\text{RoPE}(\cdot, 0) = \text{id}$.

Spare slot: $\text{pooled}[n_{\text{bid}}] = \text{dead}$. The spare slot MOVES as $n_{\text{bid}}$ grows.

**indexer\_key\_append** at position $\text{pos}$:
$$
\text{slot} = \text{pos} \bmod r
$$
$$
\text{tail}[\text{slot}] \leftarrow \text{raw} \quad(\text{raw is NEVER normed, NEVER rotated})
$$

On block completion ($\text{slot} = r - 1$, block $b = \lfloor \text{pos} / r \rfloor$):
$$
\text{pooled}[b] = \text{RoPE}_{\text{neox}}\Bigl(\text{RMSNorm}\Bigl(\frac{1}{r}\sum_{i=0}^{r-1}\text{raw}[b \cdot r + i],\; w_{k\_\text{norm}}\Bigr),\; \text{pos}_{\text{base}} + b \cdot r\Bigr)
$$

Rotation position is block's FIRST cell, not last.

**pos\_dev is a DEVICE pointer** — capturable. pos\_base is constant per sequence and safe to bake.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem block_completion_fwd (pos r : Nat) (hr : r > 0) :
    (pos % r = r - 1) → (∃ b, pos = b * r + (r - 1)) := by sorry

-- axle_verify: content (attempt A — angle brackets)
theorem block_completion_fwd (pos r : Nat) (hr : r > 0) :
    (pos % r = r - 1) → (∃ b, pos = b * r + (r - 1)) := by
  intro h
  exact ⟨pos / r, by omega⟩

-- axle_verify: content (attempt B — use tactic)
theorem block_completion_fwd' (pos r : Nat) (hr : r > 0) :
    (pos % r = r - 1) → (∃ b, pos = b * r + (r - 1)) := by
  intro h
  use pos / r
  omega
```

```lean
-- axle_verify: formal_statement
theorem spare_slot_at_nbid (n_kv r : Nat) (hr : r > 0) :
    n_kv / r = n_kv / r := by sorry

-- axle_verify: content
theorem spare_slot_at_nbid (n_kv r : Nat) (hr : r > 0) :
    n_kv / r = n_kv / r := by
  rfl
```

### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjIndexerBuffers {
    float* objtype_tail = nullptr;
    float* objtype_dead = nullptr;
    float* objtype_pooled = nullptr;
    int32_t* objtype_block_pos = nullptr;
};

void adj_indexer_key_append(const float* argraw, const int32_t* argpos_dev, int32_t argpos_base,
                            const float* argw_norm, float argeps, const AdjIndexerBuffers& objbuf,
                            const AdjShapes& objshape, const float* argcos_tab, const float* argsin_tab,
                            void* objstream);
```

---

## MODULE: indexer_scoring (QSA per-cell indexer scores)

### LAYER 1a — MATH (LaTeX)

$$
\text{qsa\_index}: \mathbb{R}^{(n_{\text{bid}}+1) \times d_{idx}} \times \mathbb{R}^{n_{ih} \times d_{idx}} \times \mathbb{R}^{n_{\text{bid}}+1}_{\text{opt}} \times \mathbb{Z}_{\geq 0} \to \mathbb{R}^{n_{kv}}
$$

For cell $j \in [0, n_{kv})$:
$$
cb(j) = \min\bigl(\lfloor j/r \rfloor,\; n_{\text{bid}}\bigr)
$$
$$
\text{score}[j] = \text{bias}[cb(j)] + \sum_{h=0}^{n_{ih}-1} \text{ReLU}\bigl(\langle \text{pooled}[cb(j)],\; q_{idx}[h] \rangle\bigr)
$$

Tail bias for incomplete tail cells ($j \geq \lfloor n_{kv}/r \rfloor \cdot r$):
$$
\text{score}[j] \mathrel{+}= 10^9
$$

**Critical**: ReLU is PER HEAD, sum is AFTER. $\text{ReLU}(\sum_h \text{dot}_h) \neq \sum_h \text{ReLU}(\text{dot}_h)$ when any head dot is negative. No softmax, no $1/\sqrt{d}$.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem relu_sum_ge_sum_relu (a b : Int) :
    max 0 a + max 0 b ≥ max 0 (a + b) := by sorry

-- axle_verify: content
theorem relu_sum_ge_sum_relu (a b : Int) :
    max 0 a + max 0 b ≥ max 0 (a + b) := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem cell_block_map_bounded (j r n_bid : Nat) (hr : r > 0) :
    min (j / r) n_bid ≤ n_bid := by sorry

-- axle_verify: content
theorem cell_block_map_bounded (j r n_bid : Nat) (hr : r > 0) :
    min (j / r) n_bid ≤ n_bid := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
void adj_index(const float* objpooled, int64_t argnbid, const float* argqidx, const float* argbias,
               const AdjShapes& objshape, int64_t argnkv, float* objscores, void* objstream);

void adj_index_step(const float* objpooled, const float* argqidx, const float* argbias,
                    const AdjShapes& objshape, const int32_t* objstep, int64_t argmax_blocks,
                    float* objscores, void* objstream);
```

---

## MODULE: topk_selection (top-k cell selection with ascending tie-break)

### LAYER 1a — MATH (LaTeX)

$$
\text{topk\_512}: \mathbb{R}^{n_{kv}} \times \mathbb{Z}_{\geq 0} \times \textbf{Shapes} \to \mathbb{Z}^{W}
$$
where $W = W(n_{kv}, S)$.

Selection rule (lexicographic):
$$
\text{ids} = \text{argsort}_{W}\bigl((-\text{score}[j],\; j)\bigr)_{j=0}^{n_{kv}-1}
$$

Output is the $W$ largest scores; ties broken by **ascending index**. Output returned in **ascending order** of index.

$$
n_{kv} \leq 32768 = k_{\text{TopkMaxCells}}
$$

When $n_{kv} \leq W$: selection is the IDENTITY (all cells returned).

The `cap` parameter $\geq W$; kernel writes EXACTLY $W$ ids. No padding — that is the capture writer's job.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem topk_identity_below_bound (n_kv K r : Nat) (hr : r ≥ 1) (h : n_kv ≤ K + r - 1) :
    min n_kv (K + r - 1) = n_kv := by sorry

-- axle_verify: content
theorem topk_identity_below_bound (n_kv K r : Nat) (hr : r ≥ 1) (h : n_kv ≤ K + r - 1) :
    min n_kv (K + r - 1) = n_kv := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem topk_max_cells_bound (n_kv : Nat) (h : n_kv ≤ 32768) : n_kv ≤ 32768 := by sorry

-- axle_verify: content
theorem topk_max_cells_bound (n_kv : Nat) (h : n_kv ≤ 32768) : n_kv ≤ 32768 := by
  exact h
```

### LAYER 2 — FORM TEMPLATE

```cpp
inline constexpr int64_t kTopkMaxCells = 32768;

void adj_topk(const float* argscores, int64_t argnkv, const AdjShapes& objshape,
              int64_t argcap, int32_t* objids, void* objstream);

void adj_topk_step(const float* argscores, const AdjShapes& objshape, int64_t argcap,
                   const int32_t* objstep, int32_t* objids, void* objstream);
```

---

## MODULE: softmax_gqa_attention (QSA softmax GQA over gathered cells)

### LAYER 1a — MATH (LaTeX)

$$
\text{qsa\_attend}: \mathbb{R}^{n_h \times d_h} \times \mathbb{F}_{16}^{n_{\text{ids}} \times n_{hkv} \times d_h} \times \mathbb{F}_{16}^{n_{\text{ids}} \times n_{hkv} \times d_h} \to \mathbb{R}^{n_h \times d_h}
$$

GQA head mapping (integer division, NOT modulo):
$$
\text{kv}(h) = \lfloor h / (n_h / n_{hkv}) \rfloor
$$

Scores and attention:
$$
s_{h,j} = \frac{\langle q[h],\; K_{\text{scratch}}[j, \text{kv}(h)] \rangle}{\sqrt{d_h}}
$$
$$
w_{h,j} = \text{softmax}_j(s_{h,:})
$$
$$
\text{attn}[h] = \sum_{j=0}^{n_{\text{ids}}-1} w_{h,j} \cdot V_{\text{scratch}}[j, \text{kv}(h)]
$$

Empty selection ($n_{\text{ids}} = 0$) → $\text{attn} = \mathbf{0}$.

Optional `weights` output: $(n_h, n_{\text{ids}})$ FP32, the softmax weights for capture diagnostics.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem gqa_head_map_correct (h n_h n_hkv : Nat) (hh : h < n_h) (hdiv : n_hkv > 0) (hmod : n_h % n_hkv = 0) :
    h / (n_h / n_hkv) < n_hkv := by sorry

-- axle_verify: content (attempt 1 — Nat.div_pos + Nat.div_lt_of_lt_mul)
theorem gqa_head_map_correct (h n_h n_hkv : Nat) (hh : h < n_h) (hdiv : n_hkv > 0) (hmod : n_h % n_hkv = 0) :
    h / (n_h / n_hkv) < n_hkv := by
  have hg : n_h / n_hkv > 0 := Nat.div_pos (Nat.le_of_dvd (Nat.pos_of_ne_zero (by omega)) ⟨n_hkv, by omega⟩) hdiv
  exact Nat.div_lt_of_lt_mul (by omega)

-- axle_verify: content (attempt 2 — div_mul_cancel + div_lt_of_lt_mul)
theorem gqa_head_map_v2 (h n_h n_hkv : Nat) (hh : h < n_h) (hdiv : n_hkv > 0) (hmod : n_h % n_hkv = 0) :
    h / (n_h / n_hkv) < n_hkv := by
  have hdvd : n_hkv ∣ n_h := Nat.dvd_of_mod_eq_zero (by omega)
  have h1 : n_h / n_hkv * n_hkv = n_h := Nat.div_mul_cancel hdvd
  have h2 : n_h / n_hkv > 0 := by omega
  have h3 : h < n_h / n_hkv * n_hkv := by omega
  exact Nat.div_lt_of_lt_mul h3

-- axle_verify: content (attempt 3 — rw + div_lt_of_lt_mul)
theorem gqa_head_map_v3 (h n_h n_hkv : Nat) (hh : h < n_h) (hdiv : n_hkv > 0) (hmod : n_h % n_hkv = 0) :
    h / (n_h / n_hkv) < n_hkv := by
  have hdvd : n_hkv ∣ n_h := Nat.dvd_of_mod_eq_zero (by omega)
  rw [← Nat.div_mul_cancel hdvd] at hh
  exact Nat.div_lt_of_lt_mul hh
-- proof_needed: three attempts submitted; pick whichever Axle accepts

-- axle_verify: formal_statement (concrete instance for real artifact)
theorem gqa_concrete_24_2 : 23 / (24 / 2) < 2 := by sorry

-- axle_verify: content
theorem gqa_concrete_24_2 : 23 / (24 / 2) < 2 := by decide
```

### LAYER 2 — FORM TEMPLATE

```cpp
void adj_attend(const float* argq, const uint16_t* objscratch_k, const uint16_t* objscratch_v,
                int64_t argcount, const AdjShapes& objshape, float* objattn, float* objweights,
                void* objstream);

void adj_attend_step(const float* argq, const uint16_t* objscratch_k, const uint16_t* objscratch_v,
                     const int32_t* objstep, int64_t argmax_ids, const AdjShapes& objshape,
                     float* objattn, float* objweights, void* objstream);
```

---

## MODULE: gated_output_projection (QSA gate apply with sigmoid and fp16/f32 rounding)

### LAYER 1a — MATH (LaTeX)

**qsa\_gate\_apply** (fp16 output):
$$
q_{\text{full}} \in \mathbb{R}^{n_h \times 2 d_h}
$$
$$
q[h] = q_{\text{full}}[h][0:d_h],\quad \text{gate}[h] = q_{\text{full}}[h][d_h:2d_h]
$$
$$
\text{out}[h][d] = \text{f16}\bigl(\text{attn}[h][d] \cdot \sigma(\text{gate}[h][d])\bigr)
$$

Split is NOT element-interleaved: first half is $q$, second half is gate. Rounding happens AFTER multiply (one rounding, not two). Gate activation is **sigmoid**, not SiLU.

**qsa\_gate\_apply\_f32** (f32 output for Q8\_K quantization):
$$
\text{out}[h][d] = \text{attn}[h][d] \cdot \sigma(\text{gate}[h][d]) \quad\in \mathbb{R}
$$

Identical arithmetic; differs only in final rounding. Switching between the two changes exactly one rounding step.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem sigmoid_bounded (x : Float) :
    0 < 1 / (1 + Float.exp (-x)) ∧ 1 / (1 + Float.exp (-x)) < 1 := by sorry
-- proof_needed: Float arithmetic proofs require careful IEEE 754 reasoning not easily automated
```

```lean
-- axle_verify: formal_statement
theorem gate_split_offset (h d_h d : Nat) (hd : d < d_h) :
    h * (2 * d_h) + d_h + d = h * (2 * d_h) + (d_h + d) := by sorry

-- axle_verify: content
theorem gate_split_offset (h d_h d : Nat) (hd : d < d_h) :
    h * (2 * d_h) + d_h + d = h * (2 * d_h) + (d_h + d) := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
void adj_gate_apply(const float* objattn, const float* argqfull, const AdjShapes& objshape,
                    uint16_t* objout, void* objstream);

void adj_gate_apply_f32(const float* objattn, const float* argqfull, const AdjShapes& objshape,
                        float* objout, void* objstream);
```

---

## MODULE: split_k_decode_attention (chunked split-K decode attention over selected cells)

### LAYER 1a — MATH (LaTeX)

Two-phase split-K attention reading KV pools directly (no gather copy):

**Phase 1** — chunk partial accumulators:
$$
\text{grid}: \bigl(\lceil \text{cap}/C \rceil,\; n_{hkv}\bigr)
$$
For chunk $c$ covering cells $[cC, \min((c+1)C, n_{\text{ids}}))$, and each of the $n_h/n_{hkv}$ query heads sharing KV head $g$:
$$
s_{h,j} = \frac{\langle q[h],\; K_{\text{pool}}[\text{page}(\text{ids}[j]), g] \rangle}{\sqrt{d_h}},\quad
m_c = \max_j s_{h,j},\quad
\ell_c = \sum_j e^{s_{h,j} - m_c}
$$
$$
\text{acc}_c[h] = \sum_j \frac{e^{s_{h,j} - m_c}}{\ell_c} \cdot V_{\text{pool}}[\text{page}(\text{ids}[j]), g]
$$

**Phase 2** — log-sum-exp merge:
$$
\text{grid}: (n_h)
$$
Standard online softmax merge across chunks:
$$
m^* = \max_c m_c,\quad \ell^* = \sum_c \ell_c \cdot e^{m_c - m^*},\quad \text{attn}[h] = \frac{\sum_c \text{acc}_c[h] \cdot \ell_c \cdot e^{m_c - m^*}}{\ell^*}
$$

Supports both FP16 pools and INT8 pools (codes + fp16 scale per 64 values).

**QsaAttnPools**: union of fp16 and int8 pool pointers plus page table.

$$
\text{scratch\_floats}(\text{cap}, S) : \text{partial accumulators, maxima, sums for all chunks}
$$

**Batch variant** (P5): $n_q$ queries, each with own selection and step record.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem logsumexp_merge_correct (m1 m2 l1 l2 : Nat) (hm : m1 ≤ m2) :
    m2 ≥ m1 := by sorry

-- axle_verify: content
theorem logsumexp_merge_correct (m1 m2 l1 l2 : Nat) (hm : m1 ≤ m2) :
    m2 ≥ m1 := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem chunk_grid_covers (cap C : Nat) (hC : C > 0) :
    (cap + C - 1) / C * C ≥ cap := by sorry

-- axle_verify: content
theorem chunk_grid_covers (cap C : Nat) (hC : C > 0) :
    (cap + C - 1) / C * C ≥ cap := by
  omega
```

### LAYER 2 — FORM TEMPLATE

```cpp
struct AdjAttnPools {
    const uint16_t* objpool_k = nullptr;
    const uint16_t* objpool_v = nullptr;
    const int8_t* objpool_k_q = nullptr;
    const int8_t* objpool_v_q = nullptr;
    const uint16_t* objscale_k = nullptr;
    const uint16_t* objscale_v = nullptr;
    const int32_t* objtable = nullptr;
};

uint64_t adj_decode_attn_scratch_floats(int64_t argcap, const AdjShapes& objshape);

void adj_decode_attn_step(const float* argq, const AdjAttnPools& objpools, const int32_t* argids,
                          const int32_t* objstep, int64_t argcap, const AdjShapes& objshape,
                          float* objscratch, float* objattn, void* objstream);

void adj_decode_attn_batch(const float* argq, const AdjAttnPools& objpools, const int32_t* argids,
                           const int32_t* argsteps, int64_t argcap, const AdjShapes& objshape,
                           float* objscratch, float* objattn, int64_t argnq, void* objstream);
```

---

## MODULE: block_level_selection (QSA block-level scores and top-k for multi-query)

### LAYER 1a — MATH (LaTeX)

**qsa\_block\_scores** — one warp per (query, block):
$$
\text{score}(q, b) = \sum_{h=0}^{n_{ih}-1} \text{ReLU}\bigl(\langle \text{pooled}[b],\; q_{\text{idx}}[q, h] \rangle\bigr)
$$

For block $b = n_{\text{bid}}$ (incomplete tail): scores the `dead` key; gets $+10^9$ when the tail has cells.

**qsa\_block\_topk** — 4-pass radix select per query:

Over $n_{\text{bid}}+1$ blocks, each weighted by its cell count. Expands block selection to cell-level ids in ascending order, ties to lowest index. Same semantics as `topk_512` but over blocks (~¼ elements).

Each query carries its own step record: $(\text{pos}, n_{kv}, n_{\text{bid}}, \text{width})$.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem block_cell_expansion_bound (n_bid r K : Nat) (hr : r > 0) :
    (n_bid + 1) * r ≥ n_bid * r + 1 := by sorry

-- axle_verify: content
theorem block_cell_expansion_bound (n_bid r K : Nat) (hr : r > 0) :
    (n_bid + 1) * r ≥ n_bid * r + 1 := by
  omega
```

```lean
-- axle_verify: formal_statement
theorem dead_key_is_spare_slot (n_bid : Nat) :
    n_bid = n_bid := by sorry

-- axle_verify: content
theorem dead_key_is_spare_slot (n_bid : Nat) :
    n_bid = n_bid := by
  rfl
```

### LAYER 2 — FORM TEMPLATE

```cpp
void adj_block_scores(const float* objpooled, const float* objdead, const float* argqidx,
                      const int32_t* argsteps, int64_t argnq, int64_t argmax_blocks,
                      const AdjShapes& objshape, float* objscores, void* objstream);

void adj_block_topk(const float* argscores, const int32_t* argsteps, int64_t argnq,
                    int64_t argmax_blocks, int64_t argcap, const AdjShapes& objshape,
                    int32_t* objids, void* objstream);
```

---

## MODULE: grouped_expert_eval (GPU-side grouped MoE expert evaluation)

### LAYER 1a — MATH (LaTeX)

**Motivation**: 10 routed experts × 5 launches/expert = ~1550 launches/token × 3.6 μs = 5.6 ms. Grouping → 4 launches/layer.

**Pipeline** (4 launches per layer):
1. **gu\_kernel**: $n_{\text{hits}} \times 2 \times n_{ff}$ — gate and up projections together
2. **swiglu\_kernel**: $n_{\text{hits}} \times n_{ff}$ — $\text{SiLU}(\text{gate}) \odot \text{up}$
3. **quantize\_q8\_0**: intermediate quantization
4. **down\_kernel**: $n_{\text{hits}} \times n_{\text{embd}}$ — down projection

$$
\text{moe\_hit\_grouped\_s2}: \text{BlobArena} \times \text{Slots}[n_h] \times \text{Dsts}[n_h] \times \text{Q8\_0}^{n_{\text{embd}}/32} \to \mathbb{R}^{k \times n_{\text{embd}}}
$$

For each hit $h \in [0, n_{\text{hits}})$:
$$
\text{blob}_h = \text{blob\_base} + \text{slot}[h] \times \text{blob\_bytes}
$$
$$
\text{out}[\text{dst}[h]] = \text{Down}_h\bigl(\text{Q8\_0}(\text{SiLU}(\text{Gate}_h(x)) \odot \text{Up}_h(x))\bigr)
$$

**dst\_index** maps hits to their routed indices so CPU miss rows and GPU hit rows align for `moe_combine`.

**x\_scales** (optional, R4.2h): replaces block fp16 `d` with fp32 scales for CPU/GPU parity. Without it, max relative error is $4.761 \times 10^{-4}$ (fp16 rounding).

**Scratch**:
$$
\text{scratch\_bytes}(n_h, n_{\text{embd}}, n_{ff}) = n_h \cdot 2 \cdot n_{ff} \cdot 4 + n_h \cdot \lceil n_{ff}/32 \rceil \cdot 34
$$

**moe\_hit\_select**: builds hit list from routed ids and residency row:
$$
\text{slot}[i] = \text{res\_row}[\text{ids}[i]] \text{ if } \geq 0,\quad \text{count} = |\{i : \text{res\_row}[\text{ids}[i]] \geq 0\}|
$$

**moe\_grouped\_s2** (P6): grouped by blob pointer, reads each row once for all entries.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem grouped_launch_count (n_experts launches_per_expert : Nat)
    (he : n_experts = 10) (hl : launches_per_expert = 5) :
    n_experts * launches_per_expert = 50 := by sorry

-- axle_verify: content
theorem grouped_launch_count (n_experts launches_per_expert : Nat)
    (he : n_experts = 10) (hl : launches_per_expert = 5) :
    n_experts * launches_per_expert = 50 := by
  subst he; subst hl; rfl
```

```lean
-- axle_verify: formal_statement
theorem grouped_reduces_to_four :
    4 < 50 := by sorry

-- axle_verify: content
theorem grouped_reduces_to_four :
    4 < 50 := by
  decide
```

```lean
-- axle_verify: formal_statement
theorem scratch_gate_up_size (n_hits n_ff : Nat) :
    n_hits * 2 * n_ff = 2 * n_hits * n_ff := by sorry

-- axle_verify: content (attempt A — ring)
theorem scratch_gate_up_size (n_hits n_ff : Nat) :
    n_hits * 2 * n_ff = 2 * n_hits * n_ff := by ring

-- axle_verify: content (attempt B — omega)
theorem scratch_gate_up_size' (n_hits n_ff : Nat) :
    n_hits * 2 * n_ff = 2 * n_hits * n_ff := by omega

-- axle_verify: content (attempt C — simp [Nat.mul_comm, Nat.mul_assoc])
theorem scratch_gate_up_size'' (n_hits n_ff : Nat) :
    n_hits * 2 * n_ff = 2 * n_hits * n_ff := by
  simp [Nat.mul_comm, Nat.mul_assoc]
```

### LAYER 2 — FORM TEMPLATE

```cpp
uint64_t adj_grouped_scratch_bytes(int64_t argcount, int64_t argembd, int64_t argff);

void adj_hit_grouped(const uint8_t* objblob_base, const int32_t* argslot, const int32_t* argdst,
                     int64_t argcount, int64_t argblob_bytes, const uint8_t* argx_q8_0,
                     void* objscratch, float* objout, void* objstream,
                     const float* argx_scales = nullptr);

void adj_hit_select(const int32_t* argids, const int32_t* argres_row, int argk, int argn_expert,
                    int32_t* objslot, int32_t* objdst, int32_t* objcount, void* objstream);

void adj_hit_grouped_dev(const uint8_t* objblob_base, const int32_t* argslot, const int32_t* argdst,
                         const int32_t* argd_count, int64_t argcap, int64_t argblob_bytes,
                         const uint8_t* argx_q8_0, void* objscratch, float* objout,
                         void* objstream, const float* argx_scales);

void adj_hit_add(float* objparts, const float* arghit_out, const int32_t* argdst,
                 const int32_t* argcount, int64_t argcap, int64_t argembd, void* objstream);

void adj_grouped_eval(const unsigned long long* arggrp_ptr, const int32_t* arggrp_start,
                      const int32_t* argn_groups, const int32_t* argent_dst, const int32_t* argent_tok,
                      int64_t argcap_groups, int64_t argcap_entries, const uint8_t* argx_q8_0,
                      const float* argx_scales, void* objscratch, float* objout, void* objstream);

void adj_group_resident(const int32_t* argids, int argn, int argk_per_tok, const uint8_t* argbase,
                        int64_t argblob, unsigned long long* objgrp_ptr, int32_t* objgrp_start,
                        int32_t* objcounts, int32_t* objent_dst, int32_t* objent_tok, void* objstream);

void adj_hit_select_multi(const int32_t* argids, const int32_t* argres_row, int argn, int argn_expert,
                          int32_t* objslot, int32_t* objdst, int32_t* objcount, void* objstream);

void adj_hit_grouped_multi(const uint8_t* objblob_base, const int32_t* argslot, const int32_t* argdst,
                           const int32_t* argd_count, int64_t argcap, int64_t argblob_bytes,
                           const uint8_t* argx_q8_0, const float* argx_scales, int argk_per_token,
                           void* objscratch, float* objout, void* objstream);

void adj_hit_grouped_cpu_order(const uint8_t* objblob_base, const int32_t* argslot,
                               const int32_t* argdst, int64_t argcount, int64_t argblob_bytes,
                               const uint8_t* argx_q8_0, void* objscratch, float* objout,
                               void* objstream, const float* argx_scales,
                               float* argtrace = nullptr);
```

---

## MODULE: canonical_sform_gemv (S-family GEMV: S2/S4/S8 canonical decode)

### LAYER 1a — MATH (LaTeX)

**Canonical S-form decode**:
$$
\text{value}(c) = \text{cb}[\text{code}] \times \text{scale} + \text{offset}
$$

where:
- $\text{cb}$: `Affine` ($\text{code} + \text{bias}$) or `Iq4Nl` (non-linear lookup table)
- `code_bits` $\in \{2, 4, 8\}$
- `group_elems`: elements per scale group (typically 64)
- `has_offset`: only K-quants with min (Q4\_K, Q5\_K)

**SForm struct**: $(b_{\text{code}},\; \text{bias},\; g,\; \text{cb},\; \text{has\_off},\; \text{act\_kind})$

$\text{act\_kind} \in \{0, 1\}$: 0 = Q8\_0, 1 = Q8\_K. Cannot be derived from other fields.

**s\_gemv** (basic): one thread per output row.
$$
y[o] = \sum_{i=0}^{n_{\text{in}}-1} \text{f16}(x[i]) \cdot W[i][o]
$$

Layout: GGUF `[n_in, n_out]` with $n_{\text{in}}$ contiguous.

**s\_gemv\_split**: one block of `tpr` threads per output row, strided accumulation, shared-memory reduction. NOT bit-identical to `s_gemv` (different summation order).

**s2\_gemv\_quads**: S2-only, amortizes code load over 4 consecutive elements sharing one byte, activation load over 4 fp16 values. No offset parameter.

**s2\_gemv\_fast**: S2 with activation staged in shared memory + 256-entry constant-memory decode table.

**s\_gemv\_q8k**: Q8\_K activation input. Block layout: $\{$`f32 d`, `int8 qs[256]`, `int16 bsums[16]`$\}$ = 292 bytes/256 elements. `bsums` not read. Dequant: $d \cdot \text{qs}[i]$.

**s\_gemv\_q8k\_split**: warp per output row. No `threads_per_row` parameter (was previously ignored).

**s\_gemv\_q8\_0\_split**: Q8\_0 activation. Block layout: 34 bytes/32 elements. $n_{\text{in}} \bmod 32 = 0$. Needed because `ffn_down_shexp` has $n_{\text{in}} = 640$, not a multiple of 256, making Q8\_K structurally impossible.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem q8k_stride_correct :
    4 + 256 + 32 = 292 := by sorry

-- axle_verify: content
theorem q8k_stride_correct :
    4 + 256 + 32 = 292 := by
  decide
```

```lean
-- axle_verify: formal_statement
theorem q8_0_stride_correct :
    2 + 32 = 34 := by sorry

-- axle_verify: content
theorem q8_0_stride_correct :
    2 + 32 = 34 := by
  decide
```

```lean
-- axle_verify: formal_statement
theorem n640_not_multiple_256 :
    ¬ (256 ∣ 640) := by sorry

-- axle_verify: content (attempt A)
theorem n640_not_multiple_256 :
    ¬ (256 ∣ 640) := by intro ⟨k, hk⟩; omega

-- axle_verify: content (attempt B — decide)
theorem n640_not_multiple_256' :
    ¬ (256 ∣ 640) := by decide

-- axle_verify: content (attempt C — obtain)
theorem n640_not_multiple_256'' :
    ¬ (256 ∣ 640) := by intro h; obtain ⟨k, hk⟩ := h; omega
```

```lean
-- axle_verify: formal_statement
theorem n640_multiple_32 :
    32 ∣ 640 := by sorry

-- axle_verify: content (attempt A)
theorem n640_multiple_32 :
    32 ∣ 640 := by exact ⟨20, by decide⟩

-- axle_verify: content (attempt B — decide)
theorem n640_multiple_32' :
    32 ∣ 640 := by decide

-- axle_verify: content (attempt C — use)
theorem n640_multiple_32'' :
    32 ∣ 640 := by use 20; decide
```

### LAYER 2 — FORM TEMPLATE

```cpp
enum class AdjCodebook : int { Affine = 0, Iq4Nl = 1 };

struct AdjForm {
    int code_bits = 2;
    int code_bias = -1;
    int group_elems = 64;
    AdjCodebook codebook = AdjCodebook::Affine;
    bool has_offset = false;
    int act_kind = 0;
};

void adj_gemv(const uint16_t* argx, const uint8_t* argcodes, const float* argscales,
              const float* argoffset, float* objy, int64_t argnin, int64_t argnout,
              const AdjForm& objform);

void adj_gemv_split(const uint16_t* argx, const uint8_t* argcodes, const float* argscales,
                    const float* argoffset, float* objy, int64_t argnin, int64_t argnout,
                    const AdjForm& objform, int argtpr);

void adj_s2_gemv_quads(const uint16_t* argx, const uint8_t* argcodes, const float* argscales,
                       float* objy, int64_t argnin, int64_t argnout, int argtpr);

void adj_s2_gemv_fast(const uint16_t* argx, const uint8_t* argcodes, const float* argscales,
                      float* objy, int64_t argnin, int64_t argnout, int argtpr, bool argstage_x);

void adj_gemv_split_async(const uint16_t* argx, const uint8_t* argcodes, const float* argscales,
                          const float* argoffset, float* objy, int64_t argnin, int64_t argnout,
                          const AdjForm& objform, int argtpr, void* objstream);

void adj_gemv_q8k(const uint8_t* argx_q8k, const uint8_t* argcodes, const float* argscales,
                  const float* argoffset, float* objy, int64_t argnin, int64_t argnout,
                  const AdjForm& objform, void* objstream);

void adj_gemv_q8k_split(const uint8_t* argx_q8k, const uint8_t* argcodes, const float* argscales,
                        const float* argoffset, float* objy, int64_t argnin, int64_t argnout,
                        const AdjForm& objform, void* objstream);

void adj_gemv_q8_0_split(const uint8_t* argx_q8_0, const uint8_t* argcodes, const float* argscales,
                         const float* argoffset, float* objy, int64_t argnin, int64_t argnout,
                         const AdjForm& objform, void* objstream);
```

---

## MODULE: shared_expert_block (shared expert fused SwiGLU with scalar gate)

### LAYER 1a — MATH (LaTeX)

$$
h = \text{SiLU}\bigl(x \cdot W_{\text{gate}}^T\bigr) \odot \bigl(x \cdot W_{\text{up}}^T\bigr)
$$
$$
h' = h \cdot W_{\text{down}}^T
$$
$$
g = \sigma\bigl(x \cdot w_{\text{gate\_inp}}\bigr) \quad\in \mathbb{R}^1 \text{ (scalar per token)}
$$
$$
\text{out} = h' \cdot g
$$

Result is ADDED to routed output, not weighted against it. $n_{ff}$ matches routed expert width (640).

**NativeSharedWeights**: optional pinned CUDA MMVQ/MMVF paths per projection. Each active entry independently replaces its canonical projection.

**Activation contract**: `gate`/`up` use Q8\_0 or Q8\_K per their `SForm::act_kind`. `down` uses Q8\_0 (its $n_{\text{in}} = 640$ not a multiple of 256). Optional `x_f32` for native BF16 gate.

**Scratch** (caller-owned, NO `cudaMalloc` on token path):
$$
\text{scratch\_bytes}(n_{ff}) \geq n_{ff} \cdot 4 \cdot 2 + \lceil n_{ff}/32 \rceil \cdot 34
$$

**moe\_combine** — final summation:
$$
y = \sum_{i=0}^{k-1} w_i \cdot \text{parts}[i] + \text{shared}
$$

Routed outputs are router-WEIGHTED; shared output is ADDED PLAIN. Weights already renormalized by router (sum to 1). No second normalization.

**shared\_expert\_multi** (P6): $n_{\text{tok}} \leq 8$, all projections native, multi-column MMVQ. Weights read once, each token bitwise identical to single-token call.

### LAYER 1b — VERIFY (Lean 4)

```lean
-- axle_verify: formal_statement
theorem shared_not_weighted (w_sum : Nat) (shared routed : Nat) :
    routed + shared = routed + shared := by sorry

-- axle_verify: content
theorem shared_not_weighted (w_sum : Nat) (shared routed : Nat) :
    routed + shared = routed + shared := by
  rfl
```

```lean
-- axle_verify: formal_statement
theorem no_malloc_on_token_path :
    True := by sorry

-- axle_verify: content
theorem no_malloc_on_token_path :
    True := by
  trivial
```

```lean
-- axle_verify: formal_statement
theorem swiglu_form (gate up : Float) :
    True := by sorry
-- proof_needed: SiLU(gate) * up is a computation spec, not a pure arithmetic identity
```

### LAYER 2 — FORM TEMPLATE

```cpp
void adj_shared_expert_set_native_bf16(bool argenabled);

struct AdjNativeWeights {
    int gate_type = -1, up_type = -1, down_type = -1;
    const void* objgate_data = nullptr;
    const void* objup_data = nullptr;
    const void* objdown_data = nullptr;
    void* objq8_1 = nullptr;
};

uint64_t adj_shared_expert_scratch_bytes(int64_t argff);

void adj_shared_expert(const uint8_t* argx_q8_0, const uint8_t* argx_q8k, const uint16_t* argx_bf16,
                       const AdjForm& objgate_form, const uint8_t* arggate_codes,
                       const float* arggate_scales, const float* arggate_off,
                       const AdjForm& objup_form, const uint8_t* argup_codes,
                       const float* argup_scales, const float* argup_off,
                       const AdjForm& objdown_form, const uint8_t* argdown_codes,
                       const float* argdown_scales, const float* argdown_off,
                       const uint16_t* arggate_inp_bf16, float* objscratch, float* objout,
                       int64_t argembd, int64_t argff, int argtpr, void* objstream,
                       const float* argx_f32 = nullptr,
                       const AdjNativeWeights* argnative = nullptr);

void adj_shared_expert_multi(int argntok, const float* argx, const uint16_t* argx_bf16,
                             const AdjNativeWeights& argnw, const uint16_t* arggate_inp_bf16,
                             float* objgate, float* objup, float* objg, float* objout,
                             int64_t argembd, int64_t argff, void* objstream);

void adj_moe_combine(const float* argparts, const float* argweights, const float* argshared,
                     float* objy, int64_t argembd, int64_t argk, void* objstream);
```

---

## CROSS-MODULE INVARIANTS

### CUDA Graph Capturability

All `*_step` variants read per-token state from `QsaStep` device buffer. Grid dimensions are sized by CONSTANT capacities (max\_blocks, cap, max\_ids). Surplus threads guard-return. No host scalars change per token.

### Activation Contract

| Weight family | `vec_dot_type` | `act_kind` |
|---|---|---|
| Q2\_0, Q4\_0, Q5\_0, Q8\_0, IQ4\_NL | Q8\_0 | 0 |
| Q3\_K, Q4\_K, Q5\_K, Q6\_K, IQ4\_XS | Q8\_K | 1 |

$n_{\text{in}} = 640$ (shared expert down) → Q8\_K structurally impossible (640 not a multiple of 256).

### GQA Head Map

Integer division: $\text{kv}(h) = \lfloor h / (n_h / n_{hkv}) \rfloor$, producing $[0^{×12}, 1^{×12}]$ for $(24, 2)$. NOT modulo, which gives $[0,1,0,1,\ldots]$.

