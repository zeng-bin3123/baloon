/-!
# MODULE: moe_router_topk
Softmax / top-k / renormalise half of the MoE router. Selection is k passes of a block-wide argmax whose
reduction keeps the larger value and, on equality, the smaller index.

## Source semantics
* The candidate merge is a total-order maximum (larger value, then smaller index), so any reduction tree gives
  the same winner.
* The taken-mask is padded to a multiple of 16 bytes so the following shared arrays stay aligned.
* The renormaliser divides by `max(sum, 2^-14)`, so the divisor is never below the clamp.

## Form template identifiers
* `router_top10_kernel` -> `topk_select_kernel`; `s_taken` -> `taken_mask`; `taken_bytes` -> `mask_pad`
* `(bv, bi)` -> `(best_val, best_idx)`; `n_expert` -> `n_cand`; `k` -> `n_pick`
-/

namespace Strata.RouterTopK

-- § candidate merge

/-- Merge of two (value, index) candidates: larger value wins, ties go to the smaller index. -/
def merge (a b : Nat × Nat) : Nat × Nat :=
  if a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2) then a else b

/-- The merge is commutative, so lane order in the shuffle tree does not matter. -/
theorem merge_comm (a b : Nat × Nat) : merge a b = merge b a := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge]
  split_ifs <;> first | rfl | (simp only [Prod.mk.injEq]; omega)

/-- The merge is associative, so a warp tree and a block tree agree with a serial scan. -/
theorem merge_assoc (a b c : Nat × Nat) : merge (merge a b) c = merge a (merge b c) := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  obtain ⟨cv, ci⟩ := c
  simp only [merge]
  split_ifs <;> first | rfl | (simp only [Prod.mk.injEq] at *; omega) | (exfalso; simp only at *; omega)

/-- The merge is idempotent. -/
theorem merge_self (a : Nat × Nat) : merge a a = a := by
  simp only [merge]
  split_ifs <;> rfl

/-- The merge result is one of its inputs. -/
theorem merge_mem (a b : Nat × Nat) : merge a b = a ∨ merge a b = b := by
  simp only [merge]
  split_ifs
  · exact Or.inl rfl
  · exact Or.inr rfl

/-- The merge result's value is at least the left value. -/
theorem merge_ge_left (a b : Nat × Nat) : a.1 ≤ (merge a b).1 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge]
  split_ifs <;> simp only at * <;> omega

/-- The merge result's value is at least the right value. -/
theorem merge_ge_right (a b : Nat × Nat) : b.1 ≤ (merge a b).1 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge]
  split_ifs <;> simp only at * <;> omega

-- § sentinel

/-- The sentinel index `n_expert` loses to a real candidate of equal value. -/
theorem sentinel_loses (n v i : Nat) (hi : i < n) : merge (v, i) (v, n) = (v, i) := by
  simp only [merge]
  split_ifs with h
  · rfl
  · exact absurd (Or.inr ⟨by trivial, hi⟩) h

-- § shared-memory layout

/-- Padded size of the taken mask. -/
def maskPad (n : Nat) : Nat := (n + 15) / 16 * 16

/-- The padded mask covers every expert. -/
theorem maskPad_ge (n : Nat) : n ≤ maskPad n := by
  unfold maskPad; omega

/-- The padded mask is a multiple of 16 bytes. -/
theorem maskPad_mod (n : Nat) : maskPad n % 16 = 0 := by
  unfold maskPad; omega

/-- The padding adds fewer than 16 bytes. -/
theorem maskPad_lt (n : Nat) : maskPad n < n + 16 := by
  unfold maskPad; omega

/-- Dynamic shared memory: mask, then n doubles, then n floats. -/
def smemBytes (n : Nat) : Nat := maskPad n + n * 8 + n * 4

/-- The double array starts at a multiple of 16, hence 8-aligned. -/
theorem double_base_aligned (n : Nat) : maskPad n % 8 = 0 := by
  unfold maskPad; omega

/-- The float array starts at a multiple of 8 because the doubles are a multiple of 8. -/
theorem float_base_aligned (n : Nat) : (maskPad n + n * 8) % 8 = 0 := by
  unfold maskPad; omega

/-- The total covers all three regions. -/
theorem smem_covers (n : Nat) : maskPad n + n * 8 + n * 4 = smemBytes n := rfl

-- § renormalisation

/-- The clamped divisor. -/
def clampDen (s lo : Nat) : Nat := max s lo

/-- The divisor is at least the clamp. -/
theorem clampDen_ge (s lo : Nat) : lo ≤ clampDen s lo := by
  unfold clampDen; omega

/-- A positive clamp keeps the divisor positive, so the division is defined. -/
theorem clampDen_pos (s lo : Nat) (h : 0 < lo) : 0 < clampDen s lo := by
  unfold clampDen; omega

/-- Above the clamp the divisor is the sum itself. -/
theorem clampDen_of_ge (s lo : Nat) (h : lo ≤ s) : clampDen s lo = s := by
  unfold clampDen; omega

/-- Below the clamp the divisor is the clamp. -/
theorem clampDen_of_lt (s lo : Nat) (h : s < lo) : clampDen s lo = lo := by
  unfold clampDen; omega

-- § launch configuration

/-- Threads per block: capped at 512, rounded up to a whole warp. -/
def launchThreads (n : Nat) : Nat := ((if n < 512 then n else 512) + 31) / 32 * 32

/-- The block is a multiple of the warp size. -/
theorem launchThreads_mod (n : Nat) : launchThreads n % 32 = 0 := by
  unfold launchThreads; split_ifs <;> omega

/-- The block never exceeds 512 threads. -/
theorem launchThreads_le (n : Nat) : launchThreads n ≤ 512 := by
  unfold launchThreads; split_ifs <;> omega

/-- Every positive expert count gets at least one warp. -/
theorem launchThreads_pos (n : Nat) (h : 0 < n) : 32 ≤ launchThreads n := by
  unfold launchThreads; split_ifs <;> omega

/-- The warp-count array of 16 floats is large enough for 512 threads. -/
theorem warps_fit (n : Nat) : (launchThreads n + 31) / 32 ≤ 16 := by
  have := launchThreads_le n; omega

end Strata.RouterTopK
