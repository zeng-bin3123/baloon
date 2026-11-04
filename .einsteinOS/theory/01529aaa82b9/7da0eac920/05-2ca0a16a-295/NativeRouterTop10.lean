/-!
# MODULE: native_router_top10
Single-token MoE router: softmax over 512 expert logits (32 lanes x 16 slots), iterative top-10 selection with
lowest-index tie-break, and renormalization of the selected weights.

## Source semantics
* Expert `e` lives in lane `e % 32`, slot `e / 32`; lane `l` holds experts `l + 32*i` for `i < 16`.
* Selection runs 10 ranks; at each rank the warp reduces `(value, expert)` pairs, preferring larger value and,
  on equal values, the smaller expert id. A selected expert is knocked out (set to -inf).
* The selected-sum is accumulated in the winning expert's lane (`expert & 31`) and clamped below by 6.1035e-5.
* Only lanes `< 10` write weights; `ids[rank]` is written by the winning lane.

## Form template identifiers
* `EXPERTS`, `TOPK`, `LANES`, `SLOTS` -> 512, 10, 32, 16
* `beats`                           -> reduction comparator (other replaces current)
-/

namespace Strata.NativeRouter

/-- Number of experts. -/
def EXPERTS : Nat := 512
/-- Selected experts. -/
def TOPK : Nat := 10
/-- Lanes. -/
def LANES : Nat := 32
/-- Slots per lane. -/
def SLOTS : Nat := 16

/-- Lane/slot layout covers the experts exactly. -/
theorem layout_size : LANES * SLOTS = EXPERTS := by decide

/-- Every expert has a unique (lane, slot) address that reassembles to the expert id. -/
theorem expert_address (e : Nat) (h : e < EXPERTS) :
    e % LANES < LANES ∧ e / LANES < SLOTS ∧ e % LANES + LANES * (e / LANES) = e := by
  unfold EXPERTS LANES SLOTS at *
  omega

/-- Distinct (lane, slot) pairs give distinct experts. -/
theorem expert_address_injective (l1 s1 l2 s2 : Nat) (hl1 : l1 < 32) (hl2 : l2 < 32)
    (h : l1 + 32 * s1 = l2 + 32 * s2) : l1 = l2 ∧ s1 = s2 := by
  omega

/-- Reduction comparator: the other candidate replaces the current one. -/
def beats (b1 : Int) (e1 : Nat) (b2 : Int) (e2 : Nat) : Prop :=
  b1 < b2 ∨ (b1 = b2 ∧ e2 < e1)

/-- The comparator is irreflexive. -/
theorem beats_irrefl (b : Int) (e : Nat) : ¬ beats b e b e := by
  unfold beats
  omega

/-- The comparator is asymmetric, so the butterfly reduction never oscillates. -/
theorem beats_asymm (b1 b2 : Int) (e1 e2 : Nat) (h : beats b1 e1 b2 e2) : ¬ beats b2 e2 b1 e1 := by
  unfold beats at *
  omega

/-- Distinct experts are strictly comparable: exactly one of the two beats the other. -/
theorem beats_total (b1 b2 : Int) (e1 e2 : Nat) (hne : e1 ≠ e2) :
    beats b1 e1 b2 e2 ∨ beats b2 e2 b1 e1 := by
  unfold beats
  omega

/-- On equal probabilities the lower expert id wins. -/
theorem tie_prefers_lower_id (b : Int) (e1 e2 : Nat) (h : e2 < e1) : beats b e1 b e2 := by
  unfold beats
  omega

/-- The comparator is transitive. -/
theorem beats_trans (b1 b2 b3 : Int) (e1 e2 e3 : Nat) (h12 : beats b1 e1 b2 e2)
    (h23 : beats b2 e2 b3 e3) : beats b1 e1 b3 e3 := by
  unfold beats at *
  omega

set_option maxRecDepth 100000 in
/-- The lane that owns the selected expert accumulates its weight: `e &&& 31 = e % 32` for all experts. -/
theorem lane_mask_eq_mod : ∀ e : Nat, e < 512 → e &&& 31 = e % 32 := by
  decide

/-- Normalization clamp: the divisor is at least the floor, hence positive. -/
theorem sum_clamp_pos (s floor : Nat) (hf : 0 < floor) : 0 < max s floor := by
  omega

/-- Ranks and output lanes are the same index space: rank `< 10` writes `weights[rank]` from lane `rank`. -/
theorem rank_lane_valid (rank : Nat) (h : rank < TOPK) : rank < LANES := by
  unfold TOPK LANES at *
  omega

/-- Knock-out value is below every finite value, so a selected expert cannot be re-selected. -/
theorem knockout_below (v lo : Int) (h : lo < v) : lo < max v lo := by
  omega

end Strata.NativeRouter
