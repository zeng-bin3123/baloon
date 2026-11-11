/-!
# MODULE: bf16_gemv_dispatch
SOURCES: src/kernels/cuda/bf16_gemv.cu
The BF16 GEMV `y[o] = Σ_i bf16(x[i]) * bf16(w[o*n_in + i])` with three kernels (thread-per-row "naive",
warp-per-row, block-split) and two host dispatchers.  This module models the dispatch decisions, the
thread→row/lane index maps, and proves that every kernel's work decomposition (strided lanes, shared-memory
tree) computes the same exact sum as the naive row walk.

## Source semantics
* `bf16_gemv` returns without launching when `n_in ≤ 0 ∨ n_out ≤ 0`; otherwise it uses the warp kernel iff
  `n_out ≥ 64`, else the naive kernel.  Grids: `⌈n_out/8⌉` blocks of 256 (warp), `⌈n_out/256⌉` (naive).
* Warp kernel: row `o = blockIdx*8 + tid/32`, lane `tid % 32` takes `i ≡ lane (mod 32)`, then a shuffle sum.
* `bf16_gemv_split`: `tpr = 32` selects the warp kernel; otherwise `tpr` must be a positive power of two
  (`tpr & (tpr-1) = 0`), one block per row, thread `t` takes `i ≡ t (mod tpr)`, then a shared-memory tree with
  offsets `tpr/2, tpr/4, …, 1`.
* Arithmetic is modelled EXACTLY over `Int` (products already formed).  Real FP32 reassociation is not exact;
  that is why the source checks the warp/split kernels against the naive one with a tolerance.

## Form template identifiers
`bf16_gemv` → `dispatch`; `bf16_gemv_warp_kernel` → `Kernel.warp`; `bf16_gemv_naive_kernel` → `Kernel.naive`;
`bf16_gemv_split_kernel` → `Kernel.split`; `THREADS` → `256`; `n_in, n_out` → `nIn, nOut`;
`threads_per_row` → `tpr`; per-lane partial → `ssum`; shared tree → `treeReduce`; row sum → `rsum`.
-/

namespace Strata.Bf16GemvDispatch

-- § exact sums

/-- `rsum f n = f 0 + … + f (n-1)` over `Int`: the naive kernel's row walk with exact arithmetic. -/
def rsum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => rsum f n + f n

/-- Unfolding step of `rsum`. -/
theorem rsum_succ (f : Nat → Int) (n : Nat) : rsum f (n + 1) = rsum f n + f n := rfl

/-- `rsum` only looks at indices below its bound. -/
theorem rsum_congr {f g : Nat → Int} {n : Nat} (h : ∀ i, i < n → f i = g i) : rsum f n = rsum g n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [rsum_succ, rsum_succ, ih (fun i hi => h i (by omega)), h n (by omega)]

/-- The sum of the zero function is zero. -/
theorem rsum_zero_fn (n : Nat) : rsum (fun _ => (0 : Int)) n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [rsum_succ, ih]; rfl

/-- `rsum` is additive in the summand. -/
theorem rsum_add (f g : Nat → Int) (n : Nat) :
    rsum (fun i => f i + g i) n = rsum f n + rsum g n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [rsum_succ, ih]; omega

/-- Splitting a range `[0, m+n)` into `[0, m)` and `[m, m+n)`. -/
theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n := by
  induction n with
  | zero => exact (Int.add_zero _).symm
  | succ n ih => rw [← Nat.add_assoc, rsum_succ, ih, rsum_succ]; omega

/-- Summing an indicator over `s < k` picks out the one matching index, if it is in range. -/
theorem rsum_indicator (c : Nat) (a : Int) (k : Nat) :
    rsum (fun s => if c = s then a else 0) k = if c < k then a else 0 := by
  induction k with
  | zero => simp [rsum]
  | succ k ih =>
    rw [rsum_succ, ih]
    by_cases h1 : c < k
    · have h2 : c ≠ k := by omega
      simp [h1, h2, show c < k + 1 by omega]
    · by_cases h3 : c = k
      · subst h3; simp
      · simp [h1, h3, show ¬ c < k + 1 by omega]

-- § strided lanes

/-- The partial sum a lane/thread `s` of a `k`-wide team accumulates: the indices `i < n` with `i % k = s`. -/
def ssum (f : Nat → Int) (s k n : Nat) : Int :=
  rsum (fun i => if i % k = s then f i else 0) n

/-- One more index adds `f n` to exactly the lane `n % k`. -/
theorem ssum_succ (f : Nat → Int) (s k n : Nat) :
    ssum f s k (n + 1) = ssum f s k n + (if n % k = s then f n else 0) := rfl

/-- **Strided partition.**  The lane partial sums of a `k`-wide team add up to the full row sum: the warp
kernel (`k = 32`) and the split kernel (`k = tpr`) lose and duplicate no element. -/
theorem strided_partition (f : Nat → Int) (k n : Nat) (hk : 0 < k) :
    rsum (fun s => ssum f s k n) k = rsum f n := by
  induction n with
  | zero => exact rsum_zero_fn k
  | succ n ih =>
    rw [rsum_congr (fun s _ => ssum_succ f s k n), rsum_add, ih, rsum_indicator, rsum_succ]
    simp [Nat.mod_lt n hk]

-- § shared-memory tree

/-- One tree level: threads `t < off` add the partner at `t + off`. -/
def treeStep (a : Nat → Int) (off : Nat) : Nat → Int :=
  fun t => if t < off then a t + a (t + off) else a t

/-- The tree loop `for off = 2^(k-1) … 1`, reading slot 0 at the end. -/
def treeReduce : Nat → (Nat → Int) → Int
  | 0, a => a 0
  | k + 1, a => treeReduce k (treeStep a (2 ^ k))

/-- One level halves the live width and preserves the sum. -/
theorem treeStep_sum (a : Nat → Int) (off : Nat) :
    rsum (treeStep a off) off = rsum a (off + off) := by
  rw [rsum_split, ← rsum_add]
  exact rsum_congr (fun t ht => by simp [treeStep, ht, Nat.add_comm t off])

/-- **Tree correctness.**  Over `2^k` slots the tree leaves the exact sum in slot 0, for every power-of-two
`tpr` the split dispatcher accepts. -/
theorem treeReduce_eq (k : Nat) : ∀ a, treeReduce k a = rsum a (2 ^ k) := by
  induction k with
  | zero =>
    intro a
    show a 0 = 0 + a 0
    omega
  | succ k ih =>
    intro a
    rw [treeReduce, ih, treeStep_sum, Nat.pow_succ, Nat.mul_two]

/-- Every tree read `t + off` with `t < off ≤ tpr/2` stays inside the `tpr` shared slots. -/
theorem tree_read_in_bounds (t off tpr : Nat) (ht : t < off) (hoff : 2 * off ≤ tpr) : t + off < tpr := by
  omega

/-- **Split kernel end to end** (exact model): partials by strided lanes, then the tree, equal the row sum. -/
theorem split_kernel_exact (f : Nat → Int) (K n : Nat) :
    treeReduce K (fun s => ssum f s (2 ^ K) n) = rsum f n := by
  rw [treeReduce_eq]
  exact strided_partition f (2 ^ K) n (Nat.two_pow_pos K)

-- § dispatch

/-- The three kernels in the file. -/
inductive Kernel
  | naive
  | warp
  | split
  deriving DecidableEq, Repr

/-- `bf16_gemv`'s choice (`none` = early return, nothing launched). -/
def dispatch (nIn nOut : Int) : Option Kernel :=
  if nIn ≤ 0 ∨ nOut ≤ 0 then none else if 64 ≤ nOut then some .warp else some .naive

/-- `bf16_gemv_split`'s choice: `none` = early return; `some (Except.error ..)` = the host refusal
(`exit(1)`) for a `tpr` that is not 32 and not a positive power of two. -/
def dispatchSplit (nIn nOut : Int) (tpr : Nat) : Option (Except String Kernel) :=
  if nIn ≤ 0 ∨ nOut ≤ 0 then none
  else if tpr = 32 then some (.ok .warp)
  else if tpr = 0 ∨ tpr &&& (tpr - 1) ≠ 0 then some (.error "threads_per_row must be a power of two")
  else some (.ok .split)

/-- Empty shapes launch nothing. -/
theorem dispatch_empty (nIn nOut : Int) (h : nIn ≤ 0 ∨ nOut ≤ 0) : dispatch nIn nOut = none := by
  simp [dispatch, h]

/-- The coalesced warp kernel is chosen exactly from 64 output rows up (8 blocks of 8 warps). -/
theorem dispatch_warp_iff (nIn nOut : Int) (hi : 0 < nIn) (ho : 0 < nOut) :
    dispatch nIn nOut = some .warp ↔ 64 ≤ nOut := by
  have hne : ¬ (nIn ≤ 0 ∨ nOut ≤ 0) := by omega
  by_cases h : 64 ≤ nOut <;> simp [dispatch, hne, h]

/-- Below 64 rows the naive (uncoalesced) kernel is still used: the branch the source flags to revisit. -/
theorem dispatch_naive_small (nIn nOut : Int) (hi : 0 < nIn) (ho : 0 < nOut) (hs : nOut < 64) :
    dispatch nIn nOut = some .naive := by
  have hne : ¬ (nIn ≤ 0 ∨ nOut ≤ 0) := by omega
  have hs' : ¬ (64 ≤ nOut) := by omega
  simp [dispatch, hne, hs']

/-- Every power of two `2^k` passes the host's `tpr & (tpr-1) == 0` test. -/
theorem pow2_passes (k : Nat) : (2 ^ k) &&& (2 ^ k - 1) = 0 := by
  rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_self]

/-- `tpr = 32` always selects the warp path; 48 and 0 are refused; 64 uses the block-split kernel. -/
theorem dispatchSplit_examples :
    dispatchSplit 640 48 32 = some (.ok .warp) ∧
    (∃ m, dispatchSplit 640 48 48 = some (.error m)) ∧
    (∃ m, dispatchSplit 640 48 0 = some (.error m)) ∧
    dispatchSplit 640 48 64 = some (.ok .split) := by
  exact ⟨rfl, ⟨_, rfl⟩, ⟨_, rfl⟩, rfl⟩

/-- The host does not bound `tpr` above: 2048 is accepted although a CUDA block holds at most 1024 threads,
so that request only fails at launch (caught by `finish`, which exits). -/
theorem dispatchSplit_accepts_2048 : dispatchSplit 640 48 2048 = some (.ok .split) := rfl

-- § grids and index maps

/-- Warp-kernel grid: `⌈nOut / 8⌉` blocks. -/
def warpGrid (nOut : Nat) : Nat := (nOut + 7) / 8

/-- Naive-kernel grid: `⌈nOut / 256⌉` blocks. -/
def naiveGrid (nOut : Nat) : Nat := (nOut + 255) / 256

/-- Warp row map `o = blockIdx * 8 + tid / 32`. -/
def warpRow (blk tid : Nat) : Nat := blk * 8 + tid / 32

/-- The warp grid has a warp for every row. -/
theorem warpGrid_covers (nOut : Nat) : nOut ≤ warpGrid nOut * 8 := by
  unfold warpGrid; omega

/-- The naive grid has a thread for every row. -/
theorem naiveGrid_covers (nOut : Nat) : nOut ≤ naiveGrid nOut * 256 := by
  unfold naiveGrid; omega

/-- **Every row is owned by a warp of the grid**: row `o` is warp `o % 8` of block `o / 8`. -/
theorem warpRow_surj (nOut o : Nat) (ho : o < nOut) :
    ∃ blk tid, blk < warpGrid nOut ∧ tid < 256 ∧ tid % 32 = 0 ∧ warpRow blk tid = o := by
  refine ⟨o / 8, (o % 8) * 32, ?_, ?_, ?_, ?_⟩
  · unfold warpGrid; omega
  · omega
  · omega
  · unfold warpRow; omega

/-- **No two warps share a row**: the (block, warp) pair is determined by the row. -/
theorem warpRow_inj (b1 t1 b2 t2 : Nat) (h1 : t1 < 256) (h2 : t2 < 256)
    (h : warpRow b1 t1 = warpRow b2 t2) : b1 = b2 ∧ t1 / 32 = t2 / 32 := by
  unfold warpRow at h; omega

/-- The source's sizing note: 64 rows are 8 warp blocks, 128 rows are 16. -/
theorem warpGrid_threshold : warpGrid 64 = 8 ∧ warpGrid 128 = 16 := by decide

/-- Lane coalescing: at a fixed iteration the 32 lanes of a warp read 32 consecutive elements of the row. -/
theorem warp_lanes_consecutive (rowBase it lane : Nat) :
    (rowBase + it * 32 + (lane + 1)) = (rowBase + it * 32 + lane) + 1 := by
  omega

end Strata.Bf16GemvDispatch
