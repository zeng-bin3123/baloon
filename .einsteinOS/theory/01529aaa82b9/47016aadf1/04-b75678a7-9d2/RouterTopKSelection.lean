/-!
# MODULE: router_top_k_selection
Selection order, clamp and launch sizing of the MoE router in `router_top10.cu`.

## Source semantics
* Top-k is `k` passes of a block-wide argmax; candidates are combined by "larger value, ties by smaller index".
* The renormalisation divisor is `max(sum, 2^-14)`, hence positive.
* Launch: threads = min(n_expert, 512) rounded up to a multiple of 32; shared memory is a 16-byte-rounded taken
  mask, then `n_expert` doubles, then `n_expert` floats.

## Form template identifiers
bv/bi -> best_value / best_index; s_taken -> taken_mask; s_p -> probabilities; s_ex -> exponentials;
n_expert -> expert_count; k -> top_k.
-/

namespace RouterTopKSelection

-- § the candidate order

/-- `a` beats `b`: larger value, or equal value and smaller index. -/
abbrev Better (a b : Nat × Nat) : Prop := a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

/-- No candidate beats itself. -/
theorem better_irrefl (a : Nat × Nat) : ¬ Better a a := by
  unfold Better; omega

/-- The order is asymmetric. -/
theorem better_asymm {a b : Nat × Nat} (h : Better a b) : ¬ Better b a := by
  unfold Better at *; omega

/-- The order is transitive. -/
theorem better_trans {a b c : Nat × Nat} (h1 : Better a b) (h2 : Better b c) : Better a c := by
  unfold Better at *; omega

/-- Candidates with distinct indices are comparable. -/
theorem better_total {a b : Nat × Nat} (h : a.2 ≠ b.2) : Better a b ∨ Better b a := by
  unfold Better; omega

/-- The shuffle-combine step: keep `b` only if it beats `a`. -/
def pick (a b : Nat × Nat) : Nat × Nat := if Better b a then b else a

/-- The combine step is commutative on candidates with distinct indices, so the reduction tree order is
irrelevant. -/
theorem pick_comm (a b : Nat × Nat) (h : a.2 ≠ b.2) : pick a b = pick b a := by
  unfold pick
  by_cases hb : Better b a
  · have hn : ¬ Better a b := better_asymm hb
    rw [if_pos hb, if_neg hn]
  · have ha : Better a b := (better_total h).resolve_right hb
    rw [if_neg hb, if_pos ha]

/-- The winner of a combine beats or equals both inputs' index choice: it is one of the inputs. -/
theorem pick_mem (a b : Nat × Nat) : pick a b = a ∨ pick a b = b := by
  unfold pick; split <;> simp

/-- The sentinel index `n_expert` loses to any real index at equal value. -/
theorem sentinel_loses (v i n : Nat) (h : i < n) : Better (v, i) (v, n) := by
  unfold Better; right; exact ⟨rfl, h⟩

-- § clamp and renormalisation

/-- The clamped divisor is at least the lower bound (model over naturals). -/
theorem clamp_ge (s c : Nat) : c ≤ max s c := Nat.le_max_right s c

/-- The clamped divisor is positive when the bound is. -/
theorem clamp_pos (s c : Nat) (hc : 0 < c) : 0 < max s c := Nat.lt_of_lt_of_le hc (clamp_ge s c)

/-- The clamp never lowers a sum that is already large. -/
theorem clamp_id (s c : Nat) (h : c ≤ s) : max s c = s := Nat.max_eq_left h

-- § launch sizing

/-- Round up to a multiple of 32. -/
def round32 (t : Nat) : Nat := (t + 31) / 32 * 32

/-- The rounded thread count is at least the request. -/
theorem round32_ge (t : Nat) : t ≤ round32 t := by unfold round32; omega

/-- It is at least one warp for a positive request. -/
theorem round32_warp (t : Nat) (h : 0 < t) : 32 ≤ round32 t := by unfold round32; omega

/-- A request of at most 512 rounds to at most 512. -/
theorem round32_le (t : Nat) (h : t ≤ 512) : round32 t ≤ 512 := by unfold round32; omega

/-- Round up to a multiple of 16. -/
def round16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- The taken mask size covers the experts. -/
theorem taken_covers (n : Nat) : n ≤ round16 n := by unfold round16; omega

/-- The doubles array that follows the mask is 8-byte aligned. -/
theorem doubles_aligned (n : Nat) : round16 n % 8 = 0 := by unfold round16; omega

/-- The floats array after `n` doubles is 4-byte aligned. -/
theorem floats_aligned (n : Nat) : (round16 n + n * 8) % 4 = 0 := by unfold round16; omega

/-- Total shared bytes decompose as mask + doubles + floats. -/
theorem smem_decomp (n : Nat) : round16 n + n * 8 + n * 4 = round16 n + n * 12 := by omega

/-- Every expert is covered by the block-strided loop `e = tid; e += nt`. -/
theorem strided_cover (e nt : Nat) (h : 0 < nt) : e % nt < nt ∧ e % nt + nt * (e / nt) = e := by
  refine ⟨Nat.mod_lt _ h, ?_⟩
  have := Nat.div_add_mod e nt
  omega

end RouterTopKSelection
