/-!
# MODULE: moe_router_top10
MoE router selection: stable descending top-k (ties by ascending index), renormalization with a lower clamp,
and shared-memory / thread-count geometry of the launcher.

## Source semantics
ids = stable argsort(-p)[:k]; w = gather(p, ids); s = max(sum w, 2^-14); out = w / s.
The k-pass block argmax scans ascending with strict `>`, so the lowest index wins ties; warp merges use
`ov > bv || (ov == bv && oi < bi)`. The taken-mask is first in shared memory (16-byte padded).

## Form template identifiers
- `better` ↔ the merge predicate; `takenBytes` ↔ `taken_bytes`; `launchThreads` ↔ `threads`
- `clampDen` ↔ `fmax(s, 2^-14)`
-/

namespace Strata.MoeRouterTop10

-- § the selection order

/-- Candidate `(v, i)` beats `(v', i')`: larger value, or equal value and smaller index. -/
def better (a b : Nat × Nat) : Prop := a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

/-- `better` is decidable (a Presburger formula on naturals). -/
instance (a b : Nat × Nat) : Decidable (better a b) := by
  unfold better; infer_instance

/-- `better` is irreflexive. -/
theorem better_irrefl (a : Nat × Nat) : ¬ better a a := by
  unfold better; omega

/-- `better` is transitive. -/
theorem better_trans (a b c : Nat × Nat) (h1 : better a b) (h2 : better b c) : better a c := by
  unfold better at *; omega

/-- Distinct-index candidates are always comparable: the order is total, so the argmax is unique. -/
theorem better_total (a b : Nat × Nat) (h : a.2 ≠ b.2) : better a b ∨ better b a := by
  unfold better; omega

/-- On equal values the smaller index wins. -/
theorem better_tie (v i j : Nat) (h : i < j) : better (v, i) (v, j) := by
  unfold better; omega

/-- A merge step keeps `a` iff it is not beaten by `b`; merging is symmetric on distinct indices. -/
def merge (a b : Nat × Nat) : Nat × Nat := if better b a then b else a

/-- The merge result is never beaten by either input (distinct indices). -/
theorem merge_best (a b : Nat × Nat) (h : a.2 ≠ b.2) :
    ¬ better b (merge a b) ∧ ¬ better a (merge a b) := by
  unfold merge
  by_cases hh : better b a
  · rw [if_pos hh]
    have h2 : ¬ better a b := by
      unfold better at *; omega
    exact ⟨better_irrefl b, h2⟩
  · rw [if_neg hh]
    exact ⟨hh, better_irrefl a⟩

-- § renormalization

/-- Clamped denominator, in units of 2^-14 (so the lower bound is 1). -/
def clampDen (s : Nat) : Nat := max s 1

/-- The renormalization denominator is never zero. -/
theorem clampDen_pos (s : Nat) : 0 < clampDen s := by
  unfold clampDen; omega

/-- The clamp does not change denominators that are already large enough. -/
theorem clampDen_id (s : Nat) (h : 1 ≤ s) : clampDen s = s := by
  unfold clampDen; omega

-- § launcher geometry

/-- Taken-mask bytes: one per expert, padded to 16. -/
def takenBytes (n : Nat) : Nat := (n + 15) / 16 * 16

/-- The padded mask covers every expert. -/
theorem takenBytes_ge (n : Nat) : n ≤ takenBytes n := by
  unfold takenBytes; omega

/-- The padding is 16-byte aligned, so the following double array is aligned. -/
theorem takenBytes_aligned (n : Nat) : takenBytes n % 16 = 0 := by
  unfold takenBytes; omega

/-- The padding adds fewer than 16 bytes. -/
theorem takenBytes_lt (n : Nat) : takenBytes n < n + 16 := by
  unfold takenBytes; omega

/-- Total dynamic shared bytes. -/
def smemBytes (n : Nat) : Nat := takenBytes n + n * 8 + n * 4

/-- The doubles start 16-byte aligned. -/
theorem smem_double_aligned (n : Nat) : takenBytes n % 16 = 0 := takenBytes_aligned n

/-- The float array begins right after the doubles, 8-byte aligned. -/
theorem smem_float_off_aligned (n : Nat) : (takenBytes n + n * 8) % 8 = 0 := by
  have := takenBytes_aligned n
  omega

/-- Launch thread count: `min(n, 512)` rounded up to a whole warp. -/
def launchThreads (n : Nat) : Nat := (min n 512 + 31) / 32 * 32

/-- At least a full warp for any positive expert count. -/
theorem launchThreads_ge_warp (n : Nat) (h : 0 < n) : 32 ≤ launchThreads n := by
  unfold launchThreads; omega

/-- Never more than 512 threads. -/
theorem launchThreads_le (n : Nat) : launchThreads n ≤ 512 := by
  unfold launchThreads; omega

/-- A whole number of warps. -/
theorem launchThreads_warps (n : Nat) : launchThreads n % 32 = 0 := by
  unfold launchThreads; omega

/-- With 512 experts the launch is 512 threads and the mask takes 512 bytes. -/
theorem geometry_512 : launchThreads 512 = 512 ∧ takenBytes 512 = 512 := by
  decide

end Strata.MoeRouterTop10
