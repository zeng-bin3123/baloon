/-!
# MODULE: router_top10_selection_semantics
MoE router: softmax over all experts, stable top-k (ties by ascending index) via `k` block-wide argmax passes,
renormalisation of the gathered weights with a lower clamp `2^-14`.

## Source semantics
* Selection pass: scan `e` ascending keeping a strict `>` improvement ⇒ the lowest index wins ties; masked
  (already taken) experts are skipped; the sentinel index `n_expert` loses to every real index.
* Weights are `w_i / max(Σ w, 2^-14)`; the clamp guarantees a positive denominator.
* Launch: threads = min(n_expert, 512) rounded up to a warp; shared memory is
  `round16(n) + 8 n + 4 n` bytes; `k ≤ 64`.
* Pass count `k * n` versus the old rank-by-counting `n * n` comparisons.

## Form template identifiers
- `router_top10_kernel`  → `softmax_topk_renorm_kernel`
- `s_taken`              → `taken_mask`
- `RT_MAX_THREADS`       → `max_block_threads`
- `6.103515625e-05`      → `renorm_clamp`
-/

namespace Strata.RouterTop10

-- § argmax with lowest-index tie-break

/-- Index of the first maximum of `p` among positions `0 .. n-1` (position 0 for `n = 0`). -/
def am (p : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => if p (am p n) < p n then n else am p n

/-- The argmax is an in-range index. -/
theorem am_le (p : Nat → Nat) (n : Nat) : am p (n + 1) ≤ n := by
  induction n with
  | zero => simp [am]
  | succ n ih =>
    show (if p (am p (n + 1)) < p (n + 1) then n + 1 else am p (n + 1)) ≤ n + 1
    split <;> omega

/-- No element exceeds the argmax's value. -/
theorem am_max (p : Nat → Nat) (n : Nat) : ∀ j, j ≤ n → p j ≤ p (am p (n + 1)) := by
  induction n with
  | zero =>
    intro j hj
    have h0 : j = 0 := by omega
    subst h0
    simp [am]
  | succ n ih =>
    intro j hj
    show p j ≤ p (if p (am p (n + 1)) < p (n + 1) then n + 1 else am p (n + 1))
    by_cases h : p (am p (n + 1)) < p (n + 1)
    · rw [if_pos h]
      rcases Nat.lt_or_ge j (n + 1) with hj' | hj'
      · have h1 := ih j (by omega)
        omega
      · have h2 : j = n + 1 := by omega
        subst h2
        exact Nat.le_refl _
    · rw [if_neg h]
      rcases Nat.lt_or_ge j (n + 1) with hj' | hj'
      · exact ih j (by omega)
      · have h2 : j = n + 1 := by omega
        subst h2
        omega

/-- Every index before the argmax is strictly smaller: ties go to the LOWEST index. -/
theorem am_first (p : Nat → Nat) (n : Nat) : ∀ j, j < am p (n + 1) → p j < p (am p (n + 1)) := by
  induction n with
  | zero =>
    intro j hj
    simp [am] at hj
  | succ n ih =>
    intro j hj
    have e : am p (n + 1 + 1) = if p (am p (n + 1)) < p (n + 1) then n + 1 else am p (n + 1) := rfl
    rw [e] at hj ⊢
    by_cases h : p (am p (n + 1)) < p (n + 1)
    · rw [if_pos h] at hj ⊢
      have h1 := am_max p n j (by omega)
      omega
    · rw [if_neg h] at hj ⊢
      exact ih j hj

/-- Warp-shuffle combine of two candidates `(value, index)`: higher value wins, equal values prefer the lower index. -/
def comb (v i w j : Nat) : Nat × Nat :=
  if w > v ∨ (w = v ∧ j < i) then (w, j) else (v, i)

/-- The combine is commutative on distinct indices, so the butterfly order of lanes cannot matter. -/
theorem comb_comm (v i w j : Nat) (hij : i ≠ j) : comb v i w j = comb w j v i := by
  unfold comb
  by_cases h1 : w > v ∨ (w = v ∧ j < i) <;> by_cases h2 : v > w ∨ (v = w ∧ i < j)
  · exfalso
    omega
  · rw [if_pos h1, if_neg h2]
  · rw [if_neg h1, if_pos h2]
  · exfalso
    omega

/-- The sentinel `n_expert` loses to every real index `e < n_expert` at equal value. -/
theorem sentinel_loses (n e v : Nat) (he : e < n) : comb v e v n = (v, e) := by
  unfold comb
  rw [if_neg]
  omega

-- § renormalisation clamp

/-- Denominator after the clamp (over a scaled numerator domain). -/
def clampDen (s eps : Nat) : Nat := max s eps

/-- The clamp keeps the denominator at least `eps`, hence positive when `eps` is. -/
theorem clampDen_pos (s eps : Nat) (h : 0 < eps) : 0 < clampDen s eps := by
  unfold clampDen
  exact Nat.lt_of_lt_of_le h (Nat.le_max_right _ _)

/-- And at least the plain sum, so every renormalised weight is at most its raw share. -/
theorem clampDen_ge_sum (s eps : Nat) : s ≤ clampDen s eps := Nat.le_max_left _ _

-- § launch geometry

/-- Block size: `n_expert` rounded up to a whole warp is at least `n_expert` and warp-aligned. -/
theorem threads_rounding (t : Nat) : t ≤ (t + 31) / 32 * 32 ∧ (t + 31) / 32 * 32 % 32 = 0 := by omega

/-- For the 512-expert model the block is 512 threads, 16 warps, matching `s_red[RT_MAX_THREADS / 32]`. -/
theorem block_512 : (min 512 512 + 31) / 32 * 32 = 512 ∧ 512 / 32 = 16 := by decide

/-- The taken-mask region is 16-byte rounded, so the following `double` array is 8-byte aligned. -/
theorem taken_alignment (n : Nat) : n ≤ (n + 15) / 16 * 16 ∧ ((n + 15) / 16 * 16) % 8 = 0 := by omega

/-- Following arrays: `n` doubles, then `n` floats, keep 4-byte alignment. -/
theorem prob_array_alignment (n : Nat) : ((n + 15) / 16 * 16 + 8 * n) % 4 = 0 := by omega

/-- Shared memory for 512 experts: 512 + 4096 + 2048 = 6656 bytes, within 48 KB. -/
theorem smem_512 : (512 + 15) / 16 * 16 + 512 * 8 + 512 * 4 = 6656 ∧ 6656 ≤ 48 * 1024 := by decide

/-- `k ≤ 64` guard and `n_expert ≤ 512 * 64` guard accept the real configuration (512 experts, k = 10). -/
theorem real_config_accepted : 10 ≤ 64 ∧ 512 ≤ 512 * 64 := by decide

-- § work comparison

/-- Selection work: `k` passes over `n` experts versus `n^2` rank comparisons: 5,120 against 262,144. -/
theorem selection_work : 10 * 512 = 5120 ∧ 512 * 512 = 262144 ∧ 262144 / 5120 = 51 := by decide

/-- The k-pass selection does strictly less work than rank-by-counting whenever `k < n`. -/
theorem passes_cheaper (k n : Nat) (h : k < n) (hn : 0 < n) : k * n < n * n :=
  Nat.mul_lt_mul_of_pos_right h hn

/-- Output slot of selection pass `i` of token `t`: `t * k + i`, unique per (token, pass). -/
theorem out_slot_injective (k t i t' i' : Nat) (hi : i < k) (hi' : i' < k)
    (h : t * k + i = t' * k + i') : t = t' ∧ i = i' := by
  have hpos : 0 < k := by omega
  have h1 : (t * k + i) / k = t := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hi, Nat.zero_add]
  have h2 : (t' * k + i') / k = t' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hi', Nat.zero_add]
  have ht : t = t' := by
    calc t = (t * k + i) / k := h1.symm
      _ = (t' * k + i') / k := by rw [h]
      _ = t' := h2
  subst ht
  exact ⟨rfl, by omega⟩

end Strata.RouterTop10
