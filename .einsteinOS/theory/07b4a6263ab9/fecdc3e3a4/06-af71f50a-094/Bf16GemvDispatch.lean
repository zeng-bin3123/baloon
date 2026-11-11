/-!
# MODULE: bf16_gemv_dispatch
SOURCES: src/kernels/cuda/bf16_gemv.cu
The BF16 x BF16 -> FP32 GEMV: the host-side choice between the naive (thread-per-row) and the
warp-per-row kernel, the launch geometry of each, the `threads_per_row` guard of the split entry point,
and the shared-memory / warp-shuffle tree reductions both cooperative kernels end with.

## Source semantics
* `bf16_gemv` returns immediately unless `n_in > 0 ∧ n_out > 0`; it launches the warp kernel iff
  `n_out ≥ 64`, otherwise the naive kernel.  Block size is always `THREADS = 256`.
* Warp kernel: `warps = 256/32 = 8`, `grid = ⌈n_out/8⌉`, row `o = blockIdx*8 + (tid >> 5)`,
  lane `tid & 31`; lane `i` strides the reduction axis by 32; lane 0 writes `y[o]`.
* Naive kernel: `grid = ⌈n_out/256⌉`, row `o = blockIdx*256 + tid`.
* `bf16_gemv_split`: `tpr = 32` selects the warp path; otherwise `tpr` must be a positive power of
  two (`tpr & (tpr-1) == 0`), one block per row, and the block reduces `scratch[0..tpr)` by halving.
* Both reductions (shared-memory tree with `off = tpr/2, …, 1` and `__shfl_down_sync` with
  `off = 16, …, 1`) compute, at index/lane 0, `scratch[t] += scratch[t+off]` for `t < off` —
  the "halving" step.  On exact (integer) values the result equals the full sum.

## Form template identifiers
* `chooseGemv`  ↔ `bf16_gemv` branch (`n_out >= 64`)
* `warpGrid`, `naiveGrid` ↔ the two `grid` computations
* `warpRow`     ↔ `blockIdx.x * warps_per_block + (threadIdx.x >> 5)`
* `tprAccepted` ↔ the `bf16_gemv_split` guard
* `halve`, `tree` ↔ one / all steps of the `for (off = tpr >> 1; off > 0; off >>= 1)` reduction
-/

namespace Strata.Bf16Gemv

-- § launch constants and dispatch

/-- Threads per block in every `bf16_gemv*` launch. -/
def THREADS : Nat := 256

/-- Warp width. -/
def WARP : Nat := 32

/-- The three device kernels in the file. -/
inductive Kernel
  | naive
  | warp
  | split
  deriving DecidableEq, Repr

/-- Host dispatch of `bf16_gemv` (for a shape that passed the `n_in, n_out > 0` early return). -/
def chooseGemv (n_out : Nat) : Kernel := if 64 ≤ n_out then .warp else .naive

/-- Ceiling division as the source writes it, `(a + b - 1) / b`. -/
def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Grid of the warp-per-row launch: one block per 8 rows. -/
def warpGrid (n_out : Nat) : Nat := ceilDiv n_out (THREADS / WARP)

/-- Grid of the naive launch: one block per 256 rows. -/
def naiveGrid (n_out : Nat) : Nat := ceilDiv n_out THREADS

/-- Output row owned by thread `tid` of block `blk` in the warp kernel. -/
def warpRow (blk tid : Nat) : Nat := blk * (THREADS / WARP) + (tid >>> 5)

/-- The warp kernel is chosen exactly when there are at least 64 output rows. -/
theorem chooseGemv_warp_iff (n : Nat) : chooseGemv n = .warp ↔ 64 ≤ n := by
  unfold chooseGemv
  by_cases h : 64 ≤ n <;> simp [h]

/-- Below the threshold the (uncoalesced) naive kernel is the one launched. -/
theorem chooseGemv_naive_iff (n : Nat) : chooseGemv n = .naive ↔ n < 64 := by
  unfold chooseGemv
  by_cases h : 64 ≤ n <;> simp [h] <;> omega

/-- The source's justification of the threshold: at 64 rows the warp kernel launches 8 blocks, and at
128 rows 16 blocks, out of 48 SMs. -/
theorem warpGrid_at_threshold : warpGrid 64 = 8 ∧ warpGrid 128 = 16 := by decide

/-- Whenever the warp kernel is chosen it launches at least 8 blocks. -/
theorem warpGrid_ge_eight (n : Nat) (h : chooseGemv n = .warp) : 8 ≤ warpGrid n := by
  have h64 : 64 ≤ n := (chooseGemv_warp_iff n).1 h
  unfold warpGrid ceilDiv THREADS WARP
  omega

-- § thread → row mapping

/-- `tid >> 5` is the warp index and `tid & 31` the lane. -/
theorem warp_lane_decomp (tid : Nat) : tid >>> 5 = tid / 32 ∧ tid &&& 31 = tid % 32 := by
  refine ⟨by simp [Nat.shiftRight_eq_div_pow], ?_⟩
  simpa using Nat.and_two_pow_sub_one_eq_mod tid 5

/-- Every output row `o < n_out` is owned by lane 0 of some warp inside the launched grid. -/
theorem warpRow_covers (n_out o : Nat) (ho : o < n_out) :
    ∃ blk, blk < warpGrid n_out ∧ ∃ tid, tid < THREADS ∧ tid % 32 = 0 ∧ warpRow blk tid = o := by
  refine ⟨o / 8, ?_, (o % 8) * 32, ?_, ?_, ?_⟩
  · unfold warpGrid ceilDiv THREADS WARP; omega
  · unfold THREADS; omega
  · omega
  · unfold warpRow THREADS WARP
    rw [(warp_lane_decomp _).1]
    omega

/-- No two warps of the launch own the same row: `warpRow` is injective on (block, warp index). -/
theorem warpRow_injective (b₁ b₂ t₁ t₂ : Nat) (h₁ : t₁ < THREADS) (h₂ : t₂ < THREADS)
    (h : warpRow b₁ t₁ = warpRow b₂ t₂) : b₁ = b₂ ∧ t₁ / 32 = t₂ / 32 := by
  unfold warpRow THREADS WARP at *
  rw [(warp_lane_decomp _).1, (warp_lane_decomp _).1] at h
  omega

/-- The naive grid covers every row: row `o` is thread `o % 256` of block `o / 256`. -/
theorem naive_covers (n_out o : Nat) (ho : o < n_out) :
    o / THREADS < naiveGrid n_out ∧ (o / THREADS) * THREADS + o % THREADS = o := by
  unfold naiveGrid ceilDiv THREADS
  omega

-- § strided reduction axis

/-- In a loop `for (i = t; i < n; i += s)` every index `i` is visited by thread `i % s` at
iteration `i / s` — this is the lane stride of the warp kernel (`s = 32`) and the thread stride of the
split kernel (`s = tpr`). -/
theorem stride_visit (s i : Nat) (hs : 0 < s) : i % s < s ∧ i = i % s + (i / s) * s := by
  refine ⟨Nat.mod_lt _ hs, ?_⟩
  have h := Nat.div_add_mod i s
  rw [Nat.mul_comm] at h
  omega

/-- …and that visit is unique: two (thread, iteration) pairs reaching the same index coincide. -/
theorem stride_unique (s t₁ t₂ j₁ j₂ : Nat) (h₁ : t₁ < s) (h₂ : t₂ < s)
    (h : t₁ + j₁ * s = t₂ + j₂ * s) : t₁ = t₂ ∧ j₁ = j₂ := by
  have hm₁ : (t₁ + j₁ * s) % s = t₁ := by rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h₁]
  have hm₂ : (t₂ + j₂ * s) % s = t₂ := by rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h₂]
  have ht : t₁ = t₂ := by rw [← hm₁, ← hm₂, h]
  subst ht
  have hs : 0 < s := Nat.lt_of_le_of_lt (Nat.zero_le _) h₁
  have : j₁ * s = j₂ * s := by omega
  exact ⟨rfl, Nat.eq_of_mul_eq_mul_right hs this⟩

-- § the split guard

/-- The `bf16_gemv_split` guard: 32 selects the warp path, otherwise a positive power of two. -/
def tprAccepted (t : Nat) : Bool := t == 32 || (0 < t && (t &&& (t - 1)) == 0)

/-- Every power of two passes the guard (`2^k & (2^k - 1) = 2^k mod 2^k = 0`). -/
theorem tprAccepted_pow_two (k : Nat) : tprAccepted (2 ^ k) = true := by
  have hpos : 0 < 2 ^ k := Nat.two_pow_pos k
  have hand : 2 ^ k &&& (2 ^ k - 1) = 0 := by
    rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_self]
  simp [tprAccepted, hpos, hand]

/-- Zero and non-powers of two are refused rather than rounded (examples from the guard's domain). -/
theorem tprAccepted_rejects : tprAccepted 0 = false ∧ tprAccepted 48 = false ∧
    tprAccepted 96 = false ∧ tprAccepted 100 = false := by decide

/-- The shared scratch of the split kernel is `tpr * 4` bytes; for `tpr ≤ 1024` that is at most 4 KiB. -/
theorem split_smem_bound (tpr : Nat) (h : tpr ≤ 1024) : tpr * 4 ≤ 4096 := by omega

-- § tree reduction

/-- Pairwise sum of two lists. -/
theorem sum_zipWith_add : ∀ (a b : List Int), a.length = b.length →
    (List.zipWith (· + ·) a b).sum = a.sum + b.sum
  | [], [], _ => by simp
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | x :: xs, y :: ys, h => by
    have ih := sum_zipWith_add xs ys (by simpa using h)
    simp only [List.zipWith_cons_cons, List.sum_cons, ih]
    omega

/-- One reduction step on `2*m` slots: slot `t < m` becomes `s[t] + s[t+m]`. -/
def halve (l : List Int) (m : Nat) : List Int := List.zipWith (· + ·) (l.take m) (l.drop m)

/-- A halving step preserves the total. -/
theorem halve_sum (l : List Int) (m : Nat) (h : l.length = 2 * m) : (halve l m).sum = l.sum := by
  have hlen : (l.take m).length = (l.drop m).length := by
    simp [List.length_take, List.length_drop]; omega
  unfold halve
  rw [sum_zipWith_add _ _ hlen, ← List.sum_append, List.take_append_drop]

/-- A halving step on `2*m` slots leaves `m` live slots. -/
theorem halve_length (l : List Int) (m : Nat) (h : l.length = 2 * m) : (halve l m).length = m := by
  unfold halve
  simp [List.length_zipWith, List.length_take, List.length_drop]
  omega

/-- The full reduction on `2^k` slots: offsets `2^(k-1), …, 2, 1`. -/
def tree : Nat → List Int → List Int
  | 0, l => l
  | k + 1, l => tree k (halve l (2 ^ k))

/-- **Tree reduction is exact**: on `2^k` slots of exact values, slot 0 ends holding the full sum.
With `k = log2 tpr` this is the split kernel's shared-memory reduction; with `k = 5` it is the
`__shfl_down_sync` loop of the warp kernel read at lane 0. -/
theorem tree_eq_sum : ∀ (k : Nat) (l : List Int), l.length = 2 ^ k → tree k l = [l.sum]
  | 0, l, h => by
    match l, h with
    | [x], _ => simp [tree]
  | k + 1, l, h => by
    have h2 : l.length = 2 * 2 ^ k := by rw [h, Nat.pow_succ, Nat.mul_comm]
    have hl : (halve l (2 ^ k)).length = 2 ^ k := halve_length l _ h2
    have ih := tree_eq_sum k (halve l (2 ^ k)) hl
    simp only [tree]
    rw [ih, halve_sum l _ h2]

/-- The warp reduction (`off = 16, 8, 4, 2, 1` over 32 lanes) delivers the sum of all 32 lane partials
at lane 0. -/
theorem warp_reduce_lane0 (l : List Int) (h : l.length = 32) : tree 5 l = [l.sum] :=
  tree_eq_sum 5 l h

end Strata.Bf16Gemv
