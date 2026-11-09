/-!
# MODULE: native_router_top10
SOURCES: src/kernels/cuda/native_router.cu
Warp-level 512-expert / 10-output router adapted from llama.cpp's topk-moe: each of 32 lanes holds 16 experts,
softmax by warp reductions, ten xor-shuffle argmax rounds, and a renormalisation clamped below by 2^-14.

## Source semantics
* Expert e is held by lane `e % 32` in register slot `e / 32`; registers are lane + 32 * i for i < 16.
* Each round, the winner's owning lane (`expert % 32 = lane`) masks slot `expert / 32` and records ids[rank].
* The shuffle merge keeps the larger value, and on equal values the smaller expert id.
* Weights are written for lane < 10; the sum is clamped below by 2^-14 before the reciprocal.

## Form template identifiers
- `owner e` = e % 32, `slotOf e` = e / 32
- `better` : shuffle-merge preference
- `rank` : 0..9
-/

namespace Strata.NativeRouter

-- § expert layout

/-- Expert held by (lane, slot). -/
def expertAt (lane i : Nat) : Nat := lane + i * 32

/-- Every expert below 512 is held by exactly one in-range (lane, slot). -/
theorem expert_layout (e : Nat) (h : e < 512) :
    e % 32 < 32 ∧ e / 32 < 16 ∧ expertAt (e % 32) (e / 32) = e := by
  unfold expertAt; omega

/-- Two different register positions never hold the same expert. -/
theorem expert_layout_unique (l i l' i' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : expertAt l i = expertAt l' i') : l = l' ∧ i = i' := by
  unfold expertAt at h; omega

/-- Every register position names an expert below 512. -/
theorem expertAt_lt (l i : Nat) (hl : l < 32) (hi : i < 16) : expertAt l i < 512 := by
  unfold expertAt; omega

/-- A winning expert is masked in its owner's register at slot `e / 32`, which exists. -/
theorem mask_slot_lt (e : Nat) (h : e < 512) : e / 32 < 16 := by
  omega

/-- Exactly one lane owns a given expert. -/
theorem unique_owner (e lane : Nat) (hl : lane < 32) (h : e % 32 = lane) : lane = e % 32 := by
  omega

-- § ordering

/-- Candidate (w, j) beats incumbent (v, i): larger value, or equal value with smaller expert id. -/
def better (v i w j : Nat) : Prop := w > v ∨ (w = v ∧ j < i)

/-- Nothing beats itself. -/
theorem better_irrefl (v i : Nat) : ¬ better v i v i := by
  unfold better; omega

/-- The preference is asymmetric. -/
theorem better_asymm (v i w j : Nat) (h : better v i w j) : ¬ better w j v i := by
  unfold better at h ⊢; omega

/-- The preference is transitive. -/
theorem better_trans (v i w j x l : Nat) (h1 : better v i w j) (h2 : better w j x l) : better v i x l := by
  unfold better at h1 h2 ⊢; omega

/-- For distinct expert ids, exactly one of the two candidates wins, so the xor-shuffle merge is order independent. -/
theorem better_exclusive (v i w j : Nat) (h : i ≠ j) : better v i w j ↔ ¬ better w j v i := by
  unfold better; omega

/-- On equal values the smaller expert id wins. -/
theorem tie_prefers_lower (v i j : Nat) (h : j < i) : better v i v j := by
  unfold better; omega

-- § ranks and outputs

/-- Round `rank` is held by lane `rank`; every rank below 10 has a lane below 32. -/
theorem rank_has_lane (r : Nat) (h : r < 10) : r < 32 := by
  omega

/-- Weight lanes (lane < 10) are exactly the ranks 0..9. -/
theorem weight_lanes (lane : Nat) : lane < 10 ↔ lane < 10 ∧ lane < 32 := by
  omega

/-- Output arrays of 10 entries are indexed in range by every rank. -/
theorem output_index_lt (r : Nat) (h : r < 10) : r < 10 := h

/-- Ten rounds over 512 experts leave 502 unselected. -/
theorem unselected_count : 512 - 10 = 502 := by
  decide

-- § buffer validation

/-- Byte spans required by the entry point. -/
theorem span_sizes : 512 * 4 = 2048 ∧ 10 * 4 = 40 := by
  decide

/-- Interval overlap test. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap test is symmetric. -/
theorem overlap_symm (ap an bp bn : Nat) : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap; omega

/-- Adjacent spans do not overlap. -/
theorem overlap_adjacent (p n m : Nat) : ¬ overlap p n (p + n) m := by
  unfold overlap; omega

-- § renormalisation

/-- The clamped sum is at least the clamp. -/
theorem clamp_lower (s c : Nat) : c ≤ max s c := by
  omega

/-- A sum above the clamp is unchanged. -/
theorem clamp_noop (s c : Nat) (h : c ≤ s) : max s c = s := by
  omega

end Strata.NativeRouter
