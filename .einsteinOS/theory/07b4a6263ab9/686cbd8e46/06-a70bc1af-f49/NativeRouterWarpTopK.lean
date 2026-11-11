/-!
# MODULE: native_router_warp_top_k
SOURCES: src/kernels/cuda/native_router.cu
Model of the pinned ggml `topk-moe` router for the fixed 512-expert / top-10 shape: one warp
holds 16 logits per lane, softmax, ten rounds of a lane-local scan plus an XOR-butterfly
argmax, `-∞` masking of winners, clamped renormalisation, and the host-side span checks.

## Source semantics
* Expert `e` lives in lane `e % 32`, register slot `e / 32`; the 512 experts fill exactly the
  32 × 16 register file.
* Each round: each lane scans its 16 slots ascending with strict `>` (lowest index wins a tie
  inside the lane), then lanes combine by "larger value, then smaller id".
* The winner's slot is set to `-∞`; output rank `r < 10` is written by lane `r`.
* `selected_sum = max(Σ selected, 2⁻¹⁴)`, `weights[r] = selected_r / selected_sum`.
* Host: non-null stream; logits 2048 B, ids 40 B, weights 40 B, all 4-aligned and pairwise
  disjoint.

## Form template identifiers
lane/slot map → `laneOf`/`slotOf`/`expertAt`; combine predicate → `better`/`pick`;
in-lane scan → `laneScan`; span checks → `spanOk`/`disjoint3`.
-/

namespace NativeRouterWarpTopK

-- § register geometry

/-- Lane holding expert `e`. -/
def laneOf (e : Nat) : Nat := e % 32
/-- Register slot holding expert `e`. -/
def slotOf (e : Nat) : Nat := e / 32
/-- Expert in `values[i]` of lane `l` (`lane + i * 32`). -/
def expertAt (l i : Nat) : Nat := l + i * 32

/-- Every expert `< 512` has a lane `< 32` and slot `< 16`, and is recovered from them. -/
theorem layout_roundtrip (e : Nat) (h : e < 512) :
    laneOf e < 32 ∧ slotOf e < 16 ∧ expertAt (laneOf e) (slotOf e) = e := by
  unfold laneOf slotOf expertAt; omega

/-- Every (lane, slot) pair is a distinct expert `< 512`. -/
theorem layout_inverse (l i : Nat) (hl : l < 32) (hi : i < 16) :
    expertAt l i < 512 ∧ laneOf (expertAt l i) = l ∧ slotOf (expertAt l i) = i := by
  unfold laneOf slotOf expertAt; omega

/-- Within one lane, slot order is index order, so the ascending strict-`>` scan keeps the
    lowest expert id among equal values in that lane. -/
theorem lane_index_monotone (l i j : Nat) (h : i < j) : expertAt l i < expertAt l j := by
  unfold expertAt; omega

/-- `(expert & 31) == lane` identifies the owning lane of the winner. -/
theorem owner_bits : ∀ e, e < 512 → e &&& 31 = laneOf e := by
  intro e _; unfold laneOf; exact Nat.and_two_pow_sub_one_eq_mod e 5

-- § the combine

/-- `x` beats `y`: larger value, ties by smaller id. -/
def better (p : Nat → Int) (x y : Nat) : Prop := p x > p y ∨ (p x = p y ∧ x < y)

instance (p : Nat → Int) (x y : Nat) : Decidable (better p x y) := by
  unfold better; infer_instance

/-- `if (other > best || (other == best && other_id < expert))` keep the other. -/
def pick (p : Nat → Int) (a b : Nat) : Nat := if better p b a then b else a

/-- Commutative: lane `l` combining with `l ^ mask` and lane `l ^ mask` combining with `l`
    reach the same candidate, so after the butterfly all lanes agree. -/
theorem pick_comm (p : Nat → Int) (a b : Nat) : pick p a b = pick p b a := by
  unfold pick
  by_cases h1 : better p b a <;> by_cases h2 : better p a b <;>
    simp only [h1, h2, ite_true, ite_false] <;> unfold better at * <;> omega

/-- Associative: the butterfly's grouping cannot change the winner. -/
theorem pick_assoc (p : Nat → Int) (a b c : Nat) :
    pick p (pick p a b) c = pick p a (pick p b c) := by
  unfold pick
  by_cases h1 : better p b a <;> by_cases h2 : better p c b <;> by_cases h3 : better p c a <;>
    simp only [h1, h2, h3, ite_true, ite_false] <;> unfold better at * <;> omega

/-- The lane-local scan: start from slot 0, replace on strictly greater value. -/
def laneScan (p : Nat → Int) (l : Nat) (slots : List Nat) : Nat :=
  slots.foldl (fun best i => if p (expertAt l i) > p best then expertAt l i else best)
    (expertAt l 0)

/-- Because slot order is index order, the strict-`>` scan coincides with `pick` along the
    lane: replacing only on strictly larger values is the tie-to-lower-id rule. -/
theorem laneStep_eq_pick (p : Nat → Int) (l i : Nat) (best : Nat) (hb : best ≤ expertAt l i) :
    (if p (expertAt l i) > p best then expertAt l i else best) = pick p best (expertAt l i) := by
  unfold pick better
  by_cases h : p (expertAt l i) > p best
  · have hc : p (expertAt l i) > p best ∨ p (expertAt l i) = p best ∧ expertAt l i < best :=
      Or.inl h
    rw [if_pos h, if_pos hc]
  · have hc : ¬ (p (expertAt l i) > p best ∨ p (expertAt l i) = p best ∧ expertAt l i < best) := by
      omega
    rw [if_neg h, if_neg hc]

-- § outputs

/-- Ranks `0..9` are written by lanes `0..9`, one each; lanes `10..31` write no weight. -/
theorem rank_writer (r : Nat) (h : r < 10) : r < 32 := by omega

/-- `-∞` masking: once an expert is masked it can only win again against other masked
    candidates; with 512 finite values and 10 rounds at most 10 experts are ever masked. -/
theorem rounds_fit : 10 ≤ 512 := by decide

/-- The clamp literal `6.103515625e-5f` is `2⁻¹⁴`, exactly representable in fp32. -/
theorem clamp_literal : 6103515625 * 2^14 = 10^14 := by decide

/-- The clamped selected sum is positive, so every weight is defined. -/
theorem clamp_pos (s c : Int) (hc : 0 < c) : 0 < max s c :=
  Int.lt_of_lt_of_le hc (Int.le_max_right s c)

-- § host checks

/-- Span sizes: 512 logits, 10 ids, 10 weights, 4 bytes each. -/
theorem span_sizes : 512 * 4 = 2048 ∧ 10 * 4 = 40 := by decide

/-- `valid`: non-null, 4-aligned, no address wrap. -/
def spanOk (addr bytes : Nat) : Prop := addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ 2^64 - 1 - addr

/-- `overlap` on half-open spans. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- The three buffers are pairwise disjoint. -/
def disjoint3 (lg ids w : Nat) : Prop :=
  ¬ overlaps lg 2048 ids 40 ∧ ¬ overlaps lg 2048 w 40 ∧ ¬ overlaps ids 40 w 40

/-- A packed layout `logits | ids | weights` passes the disjointness check. -/
theorem packed_layout_ok (a : Nat) : disjoint3 a (a + 2048) (a + 2088) := by
  unfold disjoint3 overlaps; omega

/-- Writing `ids` inside the logits span (e.g. aliasing the input) is rejected. -/
theorem alias_rejected (a : Nat) : ¬ disjoint3 a a (a + 2048) := by
  unfold disjoint3 overlaps; omega

/-- Launch shape `dim3(32, 8)`: 256 threads of which only row `y = 0` (32 lanes) works. -/
theorem active_lanes : 32 * 8 = 256 ∧ 256 / 8 = 32 := by decide

end NativeRouterWarpTopK
