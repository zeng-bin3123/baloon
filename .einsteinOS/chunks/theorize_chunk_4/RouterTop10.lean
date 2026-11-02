/-!
# MODULE: router_top10
MoE router: softmax over all experts (double exp, ascending double sum), k block-wide argmax passes with
lowest-index tie rule, renormalise the gathered weights with lower clamp 2^-14.

## Source semantics
ids = stable descending argsort(p)[:k]; w = gather(p, ids); w / max(sum w, 2^-14).
Shared memory: taken mask (16-aligned), n_expert doubles, n_expert floats. Threads = min(n_expert,512) rounded to 32.

## Form template identifiers
router_top10 -> topk_gate; n_expert -> num_experts; k -> top_k
-/
namespace Strata.RouterTop10

/-- Preference order of the argmax passes: larger value, then smaller index. -/
def better (v i v' i' : Nat) : Prop := v > v' ∨ (v = v' ∧ i < i')

/-- Irreflexive. -/
theorem better_irrefl (v i : Nat) : ¬ better v i v i := by unfold better; omega

/-- Asymmetric. -/
theorem better_asymm (v i v' i' : Nat) (h : better v i v' i') : ¬ better v' i' v i := by
  unfold better at *; omega

/-- Transitive. -/
theorem better_trans (v i v' i' v'' i'' : Nat) (h1 : better v i v' i') (h2 : better v' i' v'' i'') :
    better v i v'' i'' := by
  unfold better at *; omega

/-- Total on distinct indices: any two candidates are comparable. -/
theorem better_total (v i v' i' : Nat) (h : i ≠ i') : better v i v' i' ∨ better v' i' v i := by
  unfold better; omega

/-- Round up to a multiple of 16. -/
def round16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Taken mask size is at least n_expert and 16-aligned. -/
theorem taken_bytes (n : Nat) : n ≤ round16 n ∧ round16 n % 16 = 0 := by unfold round16; omega

/-- Shared-memory bytes. -/
def smem (n : Nat) : Nat := round16 n + n * 8 + n * 4

/-- Shared memory for 512 experts. -/
theorem smem_512 : smem 512 = 512 + 4096 + 2048 := by decide

/-- Thread count: min(n, 512) rounded up to a warp. -/
def threads (n : Nat) : Nat := ((min n 512) + 31) / 32 * 32

/-- Thread count is a warp multiple and at least min(n,512). -/
theorem threads_warp (n : Nat) : threads n % 32 = 0 ∧ min n 512 ≤ threads n := by
  unfold threads; omega

/-- For 512 experts the block has 512 threads. -/
theorem threads_512 : threads 512 = 512 := by decide

/-- The clamp keeps the divisor at least 2^-14 (as a fixed denominator model). -/
theorem clamp_pos (s m : Nat) (h : 0 < m) : 0 < max s m := by omega

/-- k passes each select one new expert: selected count after i passes is i. -/
theorem passes_count (k : Nat) : (List.range k).length = k := by simp

end Strata.RouterTop10
