/-!
# MODULE: native_router_top10
SOURCES: src/kernels/cuda/native_router.cu
Single-block CUDA router: softmax over 512 F32 logits spread over 32 lanes x 16 slots, ten rounds of
arg-max with a deterministic tie rule, then renormalisation of the ten selected weights.

## Source semantics
* Expert `e` lives at lane `e % 32`, slot `e / 32` (`e = lane + 32 * slot`).
* A candidate beats another when its value is larger, or equal with a smaller expert id.
* The destination spans (`logits` 2048 B, `ids` 40 B, `weights` 40 B) must be 4-byte aligned, non-null,
  non-wrapping and pairwise disjoint (half-open interval test).
* The selected-weight sum is clamped below by 6.103515625e-5 before the reciprocal is taken.

## Form template identifiers
logits -> `scores`; ids -> `selected_ids`; weights -> `selected_weights`; `route` -> `router_topk_kernel`;
`valid` -> `span_valid`; `overlap` -> `span_overlap`; 512 -> `n_expert`; 10 -> `k_out`.
-/

namespace Ref.NativeRouter

-- § span validity and overlap

/-- Model of `valid(p, bytes)`: non-null, 4-byte aligned, and `bytes <= UINTPTR_MAX - address`. -/
def SpanValid (addr bytes maxAddr : Nat) : Prop :=
  addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ maxAddr - addr

/-- A valid span cannot wrap past the top of the address space. -/
theorem spanValid_no_wrap (addr bytes maxAddr : Nat) (h : SpanValid addr bytes maxAddr)
    (hle : addr ≤ maxAddr) : addr + bytes ≤ maxAddr := by
  obtain ⟨_, _, h3⟩ := h
  omega

/-- A valid span is 4-byte aligned. -/
theorem spanValid_aligned (addr bytes maxAddr : Nat) (h : SpanValid addr bytes maxAddr) :
    addr % 4 = 0 := by
  obtain ⟨_, h2, _⟩ := h
  exact h2

/-- Model of `overlap(a, an, b, bn)`: the half-open interval test used by the launcher. -/
def Overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- The overlap test is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : Overlap a an b bn ↔ Overlap b bn a an := by
  unfold Overlap
  omega

/-- Spans that touch end to end do not overlap. -/
theorem overlap_adjacent (a an bn : Nat) : ¬ Overlap a an (a + an) bn := by
  unfold Overlap
  omega

/-- If the first span ends before the second begins, they are disjoint. -/
theorem overlap_of_lt (a an b bn : Nat) (h : a + an ≤ b) : ¬ Overlap a an b bn := by
  unfold Overlap
  omega

/-- A non-empty span overlaps itself. -/
theorem overlap_self (a n : Nat) (hn : 0 < n) : Overlap a n a n := by
  unfold Overlap
  omega

/-- Quirk of the formula: a zero-length span strictly inside another one is reported as overlapping. -/
theorem overlap_zero_length_quirk : Overlap 5 0 3 10 := by
  unfold Overlap
  omega

-- § geometry

/-- The 512 experts are exactly 32 lanes times 16 slots. -/
theorem expert_count : (512 : Nat) = 32 * 16 := by decide

/-- The pinned block geometry is 32 x 8 = 256 threads, of which row zero (32 threads) is active. -/
theorem block_threads : (32 : Nat) * 8 = 256 := by decide

/-- Ten output ranks fit in one warp, so `lane < 10` can hold one weight each. -/
theorem outputs_fit_warp : (10 : Nat) ≤ 32 := by decide

/-- `lane + 32 * slot` is a valid expert id. -/
theorem expert_lt (lane slot : Nat) (hl : lane < 32) (hs : slot < 16) : lane + 32 * slot < 512 := by
  omega

/-- The owning lane of `lane + 32 * slot` is `lane`. -/
theorem expert_lane (lane slot : Nat) (hl : lane < 32) : (lane + 32 * slot) % 32 = lane := by
  omega

/-- The slot of `lane + 32 * slot` is `slot`. -/
theorem expert_slot (lane slot : Nat) (hl : lane < 32) : (lane + 32 * slot) / 32 = slot := by
  omega

/-- Every expert id decomposes into a lane and a slot. -/
theorem expert_roundtrip (e : Nat) (he : e < 512) :
    e % 32 + 32 * (e / 32) = e ∧ e / 32 < 16 ∧ e % 32 < 32 := by
  omega

/-- Exactly one lane owns a given expert, so the `ids[rank]` write has a single writer. -/
theorem unique_owner (e l l' : Nat) (h : e % 32 = l) (h' : e % 32 = l') : l = l' := by
  omega

-- § tie-breaking order

/-- `Beats v i w j`: candidate `(v, i)` is preferred to `(w, j)`: larger value, or equal and smaller id. -/
abbrev Beats (v i w j : Nat) : Prop := v > w ∨ (v = w ∧ i < j)

/-- No candidate beats itself. -/
theorem beats_irrefl (v i : Nat) : ¬ Beats v i v i := by
  unfold Beats
  omega

/-- The preference order is asymmetric. -/
theorem beats_asymm (v i w j : Nat) : Beats v i w j → ¬ Beats w j v i := by
  intro h1 h2
  unfold Beats at h1 h2
  omega

/-- The preference order is transitive. -/
theorem beats_trans (a i b j c k : Nat) : Beats a i b j → Beats b j c k → Beats a i c k := by
  intro h1 h2
  unfold Beats at h1 h2 ⊢
  omega

/-- Two candidates with different ids are always comparable, so the arg-max is deterministic. -/
theorem beats_total (v w i j : Nat) (h : i ≠ j) : Beats v i w j ∨ Beats w j v i := by
  unfold Beats
  omega

/-- The butterfly combine step used in the shuffle reduction. -/
def pick (a b : Nat × Nat) : Nat × Nat :=
  if Beats b.1 b.2 a.1 a.2 then b else a

/-- Combining is commutative for candidates with distinct ids, so the shuffle order cannot matter. -/
theorem pick_comm (a b : Nat × Nat) (h : a.2 ≠ b.2) : pick a b = pick b a := by
  unfold pick
  by_cases h1 : Beats b.1 b.2 a.1 a.2
  · have h2 : ¬ Beats a.1 a.2 b.1 b.2 := beats_asymm _ _ _ _ h1
    rw [if_pos h1, if_neg h2]
  · have h2 : Beats a.1 a.2 b.1 b.2 := by
      rcases beats_total a.1 b.1 a.2 b.2 h with h3 | h3
      · exact h3
      · exact absurd h3 h1
    rw [if_neg h1, if_pos h2]

-- § weight normalisation

/-- The clamped selected-weight sum is at least the clamp constant. -/
theorem clamp_ge (s m : Nat) : m ≤ max s m := by omega

/-- With a positive clamp constant the denominator is never zero. -/
theorem clamp_pos (s m : Nat) (hm : 0 < m) : 0 < max s m := by omega

end Ref.NativeRouter
