/-!
# MODULE: router_top10_selection
SOURCES: src/kernels/cuda/native_router.cu, src/kernels/cuda/router_top10.cu
MoE router: softmax over all experts, k passes of block-wide argmax (ties to the lowest index), gather,
renormalise with the 2^-14 lower clamp. Covers the selection order, lane/expert coverage and the
shared-memory / thread-count arithmetic.

## Source semantics
* A candidate (ov, oi) replaces the current best (bv, bi) iff ov > bv or (ov == bv and oi < bi).
* Scanning ascending with strict `>` keeps the lowest index on a tie (stable descending argsort).
* Native path: 512 experts as 32 lanes x 16 values, expert = lane + 32 * i.
* Generic path: taken-mask padded to 16 bytes first, then n doubles, then n floats.
* threads = round_up_32(min(n_expert, 512)); renorm divisor is max(sum, 2^-14).

## Form template identifiers
beats <- the replace-if predicate in the shuffle reductions, takenBytes <- taken_bytes,
threadsFor <- `threads` in router_top10, clampSum <- fmax(s, 2^-14).
-/

namespace Strata.Router

-- § argmax order

/-- Candidate `(ov, oi)` replaces `(bv, bi)`. -/
def beats (ov oi bv bi : Nat) : Prop := bv < ov ∨ (ov = bv ∧ oi < bi)

/-- No candidate beats itself. -/
theorem beats_irrefl (v i : Nat) : ¬ beats v i v i := by
  unfold beats; omega

/-- Beating is asymmetric. -/
theorem beats_asymm {a ai b bi : Nat} (h : beats a ai b bi) : ¬ beats b bi a ai := by
  unfold beats at *; omega

/-- Beating is transitive, so the reduction order does not matter. -/
theorem beats_trans {a ai b bi c ci : Nat} (h1 : beats a ai b bi) (h2 : beats b bi c ci) :
    beats a ai c ci := by
  unfold beats at *; omega

/-- Distinct experts are always comparable. -/
theorem beats_total {v i w j : Nat} (h : i ≠ j) : beats v i w j ∨ beats w j v i := by
  unfold beats; omega

/-- Equal probabilities: the lower index wins and the higher cannot. -/
theorem tie_prefers_lower {v i j : Nat} (h : i < j) : beats v i v j ∧ ¬ beats v j v i := by
  unfold beats; omega

-- § expert coverage

/-- Every expert id below 512 is `lane + 32 * i` for exactly one lane < 32 and i < 16. -/
theorem expert_covered {e : Nat} (h : e < 512) :
    ∃ lane i : Nat, lane < 32 ∧ i < 16 ∧ e = lane + i * 32 :=
  ⟨e % 32, e / 32, by omega, by omega, by omega⟩

/-- That decomposition is unique. -/
theorem expert_unique {l1 i1 l2 i2 : Nat} (h1 : l1 < 32) (h2 : l2 < 32)
    (h : l1 + i1 * 32 = l2 + i2 * 32) : l1 = l2 ∧ i1 = i2 := by omega

-- § shared memory and threads

/-- Taken-mask size, padded to 16 bytes. -/
def takenBytes (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Padding never shrinks the mask. -/
theorem takenBytes_ge (n : Nat) : n ≤ takenBytes n := by unfold takenBytes; omega

/-- Padding adds fewer than 16 bytes. -/
theorem takenBytes_lt (n : Nat) : takenBytes n < n + 16 := by unfold takenBytes; omega

/-- The padded size keeps the following double array 16-byte aligned. -/
theorem takenBytes_aligned (n : Nat) : takenBytes n % 16 = 0 := by unfold takenBytes; omega

/-- Threads per block: `min(n, 512)` rounded up to a whole warp. -/
def threadsFor (n : Nat) : Nat := (min n 512 + 31) / 32 * 32

/-- At least one full warp for a non-empty expert set. -/
theorem threadsFor_ge_warp {n : Nat} (h : 1 ≤ n) : 32 ≤ threadsFor n := by
  unfold threadsFor; omega

/-- Never past the 512-thread cap. -/
theorem threadsFor_le (n : Nat) : threadsFor n ≤ 512 := by unfold threadsFor; omega

/-- A whole number of warps. -/
theorem threadsFor_warps (n : Nat) : threadsFor n % 32 = 0 := by unfold threadsFor; omega

/-- It covers every expert up to the cap. -/
theorem threadsFor_cover (n : Nat) : min n 512 ≤ threadsFor n := by unfold threadsFor; omega

/-- Dynamic shared bytes for the real 512 experts: mask + doubles + floats. -/
theorem real_smem : takenBytes 512 + 512 * 8 + 512 * 4 = 6656 := by decide

/-- k passes of 512 compares versus the old all-pairs rank loop. -/
theorem work_counts : 10 * 512 = 5120 ∧ 512 * 512 = 262144 := by decide

-- § renormalisation clamp

/-- Divisor of the weight renormalisation. -/
def clampSum (s eps : Nat) : Nat := max s eps

/-- The divisor never falls below the clamp. -/
theorem clampSum_ge_eps (s eps : Nat) : eps ≤ clampSum s eps := by unfold clampSum; omega

/-- The divisor never falls below the sum. -/
theorem clampSum_ge_sum (s eps : Nat) : s ≤ clampSum s eps := by unfold clampSum; omega

/-- Above the clamp the divisor is the sum itself. -/
theorem clampSum_of_ge {s eps : Nat} (h : eps ≤ s) : clampSum s eps = s := by
  unfold clampSum; omega

end Strata.Router
