/-!
# MODULE: bf16_mmvf_block_select
SOURCES: src/kernels/cuda/native_bf16.cu
The FP32-activation BF16 GEMV adapted from llama.cpp's `mmvf`: one block per output row, each thread walking
PAIRS of elements (one `uint32` of two bf16 weights against one `float2` of activations).  This module models
the block-size search `mmvf_block_size`, proves it always lands on a case of the launch `switch` and is optimal
among the candidates, and models the host argument/alignment contract and the pair decomposition.

## Source semantics
* `mmvf_block_size(n)`: start at `(32, ⌈n/64⌉)`; for `c = 64, 96, …, 256`, replace the best when
  `⌈n/(2c)⌉` is STRICTLY smaller (so ties keep the smaller block).
* The `switch` has cases 32, 64, …, 256 and no `default`: a value outside that set would launch nothing.
* Host contract: `n_in > 0`, `n_in` even, `n_in ≤ INT_MAX`, `0 < n_out ≤ INT_MAX`; `x` 8-byte aligned (read as
  `float2`), `w` and `y` 4-byte aligned; violations throw `invalid_argument`.
* Pair `p` covers elements `2p` (low half of the weight word) and `2p+1` (high half), little-endian.
* Cross-warp step: `BLOCK_SIZE/32 ≤ 8` warp sums fit the 32-entry `partials` array.

## Form template identifiers
`mmvf_block_size` → `blockSize`; candidate loop → `candidates`/`stepBest`; iteration count → `iters`;
`bf16_gemv_fp32_mmvf` argument checks → `argsOk`, `alignOk`; `weights2[pair]` halves → `lo16`, `hi16`.
-/

namespace Strata.Bf16MmvfBlockSelect

-- § the block-size search

/-- Candidate block sizes after the initial 32: `for (candidate = 64; candidate <= 256; candidate += 32)`. -/
def candidates : List Nat := [64, 96, 128, 160, 192, 224, 256]

/-- Per-thread pair iterations for a block of `c` threads: `(n + 2c - 1) / (2c)`. -/
def iters (n c : Nat) : Nat := (n + 2 * c - 1) / (2 * c)

/-- One loop step: adopt `c` only on a strict improvement. -/
def stepBest (n : Nat) (acc : Nat × Nat) (c : Nat) : Nat × Nat :=
  if iters n c < acc.2 then (c, iters n c) else acc

/-- The initial state `(32, (n + 63) / 64)`. -/
def init (n : Nat) : Nat × Nat := (32, (n + 63) / 64)

/-- Final `(best, best_iterations)`. -/
def search (n : Nat) : Nat × Nat := candidates.foldl (stepBest n) (init n)

/-- `mmvf_block_size`. -/
def blockSize (n : Nat) : Nat := (search n).1

/-- The cases of the launch `switch`. -/
def switchCases : List Nat := [32, 64, 96, 128, 160, 192, 224, 256]

/-- Generic invariant rule for the fold. -/
theorem fold_inv (n : Nat) (P : Nat × Nat → Prop) :
    ∀ (cs : List Nat) (acc : Nat × Nat), P acc → (∀ a c, c ∈ cs → P a → P (stepBest n a c)) →
      P (cs.foldl (stepBest n) acc) := by
  intro cs
  induction cs with
  | nil => intro acc h0 _; exact h0
  | cons c rest ih =>
    intro acc h0 hs
    exact ih (stepBest n acc c) (hs acc c (List.mem_cons_self) h0)
      (fun a c' hc' ha => hs a c' (List.mem_cons_of_mem _ hc') ha)

/-- A step never increases the best iteration count. -/
theorem stepBest_le (n : Nat) (a : Nat × Nat) (c : Nat) : (stepBest n a c).2 ≤ a.2 := by
  unfold stepBest
  by_cases h : iters n c < a.2 <;> simp [h] <;> omega

/-- After a step, the best is no worse than the candidate just tried. -/
theorem stepBest_le_cand (n : Nat) (a : Nat × Nat) (c : Nat) : (stepBest n a c).2 ≤ iters n c := by
  unfold stepBest
  by_cases h : iters n c < a.2 <;> simp [h] <;> omega

/-- The fold is monotone non-increasing in the iteration count. -/
theorem fold_le_init (n : Nat) (cs : List Nat) (acc : Nat × Nat) :
    (cs.foldl (stepBest n) acc).2 ≤ acc.2 :=
  fold_inv n (fun a => a.2 ≤ acc.2) cs acc (Nat.le_refl _)
    (fun a c _ ha => Nat.le_trans (stepBest_le n a c) ha)

/-- Every candidate the fold visits bounds the final iteration count. -/
theorem fold_le_each (n : Nat) :
    ∀ (cs : List Nat) (acc : Nat × Nat) (c : Nat), c ∈ cs → (cs.foldl (stepBest n) acc).2 ≤ iters n c := by
  intro cs
  induction cs with
  | nil => intro _ _ hc; cases hc
  | cons c0 rest ih =>
    intro acc c hc
    rcases List.mem_cons.mp hc with h | h
    · subst h
      exact Nat.le_trans (fold_le_init n rest _) (stepBest_le_cand n acc c)
    · exact ih _ c h

/-- **Exhaustive switch.**  The selected block size is always one of the `switch` cases, so the launch is
never silently skipped. -/
theorem blockSize_mem_switch (n : Nat) : blockSize n ∈ switchCases := by
  have h := fold_inv n (fun a => a.1 = 32 ∨ a.1 ∈ candidates) candidates (init n) (Or.inl rfl)
    (fun a c hc ha => by
      unfold stepBest
      by_cases hlt : iters n c < a.2
      · simp only [hlt, ite_true]; exact Or.inr hc
      · simp only [hlt]; exact ha)
  rcases h with h | h
  · unfold blockSize search; rw [h]; simp [switchCases]
  · unfold blockSize search; simp only [switchCases, candidates] at h ⊢; simp_all

/-- The bookkeeping pair stays consistent: the recorded count is the iteration count of the recorded size. -/
theorem search_consistent (n : Nat) : (search n).2 = iters n (search n).1 := by
  apply fold_inv n (fun a => a.2 = iters n a.1)
  · simp only [init, iters]; omega
  · intro a c _ ha
    unfold stepBest
    by_cases hlt : iters n c < a.2
    · simp only [hlt, ite_true]
    · simp only [hlt, ite_false]; exact ha

/-- **Optimality.**  The chosen block needs no more iterations than any candidate, including 32. -/
theorem blockSize_optimal (n c : Nat) (hc : c ∈ switchCases) : iters n (blockSize n) ≤ iters n c := by
  rw [blockSize, ← search_consistent]
  rcases List.mem_cons.mp hc with h | h
  · subst h
    have := fold_le_init n candidates (init n)
    simp only [search, init, iters] at this ⊢
    omega
  · exact fold_le_each n candidates (init n) c h

/-- Concrete selections at the model's shapes (`n_embd = 2560`, expert width 640). -/
theorem blockSize_examples : blockSize 2560 = 256 ∧ blockSize 640 = 160 ∧ blockSize 64 = 32 := by
  decide

/-- Ties keep the smaller block: at `n = 640` the 160-, 192-, 224- and 256-thread blocks all need 2 pair
iterations and 160 is chosen. -/
theorem blockSize_tie_640 :
    iters 640 160 = 2 ∧ iters 640 192 = 2 ∧ iters 640 256 = 2 ∧ blockSize 640 = 160 := by
  decide

/-- Cross-warp reduction fits: every case is a whole number of warps and at most 8 ≤ 32 partial slots. -/
theorem partials_fit : ∀ b ∈ switchCases, b % 32 = 0 ∧ b / 32 ≤ 32 := by decide

-- § host contract

/-- `INT_MAX`. -/
def intMax : Int := 2147483647

/-- The size checks of `bf16_gemv_fp32_mmvf`. -/
def argsOk (nIn nOut : Int) : Prop :=
  0 < nIn ∧ nIn % 2 = 0 ∧ nIn ≤ intMax ∧ 0 < nOut ∧ nOut ≤ intMax

/-- The pointer checks: `x` read as `float2` (8 B), `w` as `uint32` (4 B), `y` written as `float` (4 B). -/
def alignOk (x w y : Nat) : Prop := x ≠ 0 ∧ w ≠ 0 ∧ y ≠ 0 ∧ x % 8 = 0 ∧ w % 4 = 0 ∧ y % 4 = 0

/-- An odd `n_in` is refused: the pair loop would drop the last element. -/
theorem odd_refused (nIn nOut : Int) (h : nIn % 2 = 1) : ¬ argsOk nIn nOut := by
  intro hk; unfold argsOk at hk; omega

/-- With an even `n_in` the pairs `0 … n_in/2 - 1` cover every element exactly once. -/
theorem pairs_cover (nIn i : Nat) (he : nIn % 2 = 0) (hi : i < nIn) : i / 2 < nIn / 2 ∧ 2 * (i / 2) + i % 2 = i := by
  omega

/-- Every `float2` read `x + 8p` stays 8-byte aligned given an aligned base. -/
theorem float2_aligned (x p : Nat) (hx : x % 8 = 0) : (x + 8 * p) % 8 = 0 := by omega

/-- Row `o`'s `uint32` reads stay 4-byte aligned: the row starts at `w + 2*n_in*o` bytes and `n_in` is even. -/
theorem row_words_aligned (w nIn o p : Nat) (hw : w % 4 = 0) (he : nIn % 2 = 0) :
    (w + 2 * (nIn * o) + 4 * p) % 4 = 0 := by
  obtain ⟨k, rfl⟩ : ∃ k, nIn = 2 * k := ⟨nIn / 2, by omega⟩
  rw [Nat.mul_assoc]
  omega

-- § the pair word

/-- Low bf16 of a weight word (element `2p`). -/
def lo16 (wd : Nat) : Nat := wd % 65536

/-- High bf16 of a weight word (element `2p+1`): `weight >> 16`, truncated to 16 bits. -/
def hi16 (wd : Nat) : Nat := wd / 65536 % 65536

/-- Little-endian pairing: the word assembled from elements `e0, e1` hands them back in order. -/
theorem pair_roundtrip (e0 e1 : Nat) (h0 : e0 < 65536) (h1 : e1 < 65536) :
    lo16 (e0 + 65536 * e1) = e0 ∧ hi16 (e0 + 65536 * e1) = e1 := by
  unfold lo16 hi16; constructor <;> omega

/-- `(uint16_t) (weight >> 16)` agrees with `hi16` (the shift-then-truncate the kernel performs). -/
theorem hi16_shift (wd : Nat) : (wd >>> 16) % 65536 = hi16 wd := by
  rw [Nat.shiftRight_eq_div_pow]; rfl

end Strata.Bf16MmvfBlockSelect
