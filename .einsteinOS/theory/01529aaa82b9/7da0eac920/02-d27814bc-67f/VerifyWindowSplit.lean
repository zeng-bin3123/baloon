/-!
# MODULE: verify_window_split
The speculative-decoding verify window: splitting the window's tokens into two groups, the ring numbering that
interleaves the groups across layers, the GPU-plan staging layout, the monotone flag, and the commit row table.

## Source semantics
* A window of `T ≥ 2` tokens is cut at `(T+1)/2`: group 0 is `[0, (T+1)/2)`, group 1 is `[(T+1)/2, T)`. The groups
  tile `[0, T)` and are both non-empty.
* Ring number for (layer `l`, group `g`) is `l*G + g + 1`; layer and group are recoverable by `/ G` and `% G`,
  and consecutive (layer, group) steps get consecutive ring numbers.
* The plan buffer places 64-bit pointer arrays at an even 32-bit offset, after the counts/start/dst/tok arrays and
  before `start2`; `post` recomputes the same offsets as `init`.
* Flag B only rises (`raise_flag` is a max).
* A commit row of `n_keep` tokens stores positions `pos0 + t` for kept `t`, `-1` otherwise.
* Staging blobs are split evenly between the two groups of a split window and never overlap.

## Form template identifiers
* `Verifier::record_window` group bounds -> `gBegin`, `gEnd`
* ring numbers -> `ring`
* `plan_i32_`, `ptr_off` -> `i32Count`, `ptrOff`, `planSize`
* `Verifier::raise_flag` -> `raise`
* `Verifier::commit` rows -> `commitPos`
* `Verifier::set_plan_slot` staging -> `stagingPer`
-/

namespace Strata.Verify

-- § token groups

/-- First token of group `g` in a split window. -/
def gBegin (T g : Nat) : Nat := if g = 0 then 0 else (T + 1) / 2

/-- One past the last token of group `g` in a split window. -/
def gEnd (T g : Nat) : Nat := if g = 0 then (T + 1) / 2 else T

/-- Group sizes add up to the window. -/
theorem group_sizes (T : Nat) : (gEnd T 0 - gBegin T 0) + (gEnd T 1 - gBegin T 1) = T := by
  simp [gEnd, gBegin]
  omega

/-- The groups abut with no gap and no overlap. -/
theorem groups_abut (T : Nat) : gEnd T 0 = gBegin T 1 := by
  simp [gEnd, gBegin]

/-- Group 0 starts at token 0 and group 1 ends at `T`. -/
theorem groups_cover (T : Nat) : gBegin T 0 = 0 ∧ gEnd T 1 = T := by
  simp [gBegin, gEnd]

/-- For a window of at least 2 tokens both groups are non-empty. -/
theorem groups_nonempty (T : Nat) (h : 2 ≤ T) : gBegin T 0 < gEnd T 0 ∧ gBegin T 1 < gEnd T 1 := by
  simp [gBegin, gEnd]
  omega

/-- Group 0 is never smaller than group 1. -/
theorem group0_ge_group1 (T : Nat) : gEnd T 1 - gBegin T 1 ≤ gEnd T 0 - gBegin T 0 := by
  simp [gEnd, gBegin]
  omega

/-- Each group's tokens lie inside the window. -/
theorem group_in_window (T g : Nat) : gBegin T g ≤ gEnd T g ∧ gEnd T g ≤ T := by
  unfold gBegin gEnd
  by_cases h : g = 0 <;> simp [h] <;> omega

-- § ring numbering

/-- Ring number of (layer, group) with `G` groups per layer. -/
def ring (G l g : Nat) : Nat := l * G + g + 1

/-- Layer recovered from a ring number. -/
theorem ring_layer (G l g : Nat) (hG : 0 < G) (hg : g < G) : (ring G l g - 1) / G = l := by
  unfold ring
  have h : l * G + g + 1 - 1 = g + l * G := by omega
  rw [h, Nat.add_mul_div_right _ _ hG, Nat.div_eq_of_lt hg]
  simp

/-- Group recovered from a ring number. -/
theorem ring_group (G l g : Nat) (hG : 0 < G) (hg : g < G) : (ring G l g - 1) % G = g := by
  unfold ring
  have h : l * G + g + 1 - 1 = g + l * G := by omega
  rw [h, Nat.add_mul_mod_self_right]
  exact Nat.mod_eq_of_lt hg

/-- Distinct (layer, group) pairs never share a ring number. -/
theorem ring_inj (G l g l' g' : Nat) (hG : 0 < G) (hg : g < G) (hg' : g' < G)
    (h : ring G l g = ring G l' g') : l = l' ∧ g = g' := by
  constructor
  · have a := ring_layer G l g hG hg
    have b := ring_layer G l' g' hG hg'
    rw [h] at a
    omega
  · have a := ring_group G l g hG hg
    have b := ring_group G l' g' hG hg'
    rw [h] at a
    omega

/-- Within a layer, the next group gets the next ring number. -/
theorem ring_next_group (G l g : Nat) : ring G l (g + 1) = ring G l g + 1 := by
  unfold ring; omega

/-- From the last group of a layer to the first of the next, ring numbers are consecutive. -/
theorem ring_next_layer (G l : Nat) (hG : 0 < G) : ring G (l + 1) 0 = ring G l (G - 1) + 1 := by
  unfold ring
  have h : (l + 1) * G = l * G + G := Nat.succ_mul l G
  omega

/-- Ring numbers are positive. -/
theorem ring_pos (G l g : Nat) : 0 < ring G l g := by
  unfold ring; omega

/-- The final ring number is `n_layers * G`. -/
theorem ring_last (G n : Nat) (hG : 0 < G) (hn : 0 < n) : ring G (n - 1) (G - 1) = n * G := by
  unfold ring
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have h : (m + 1) * G = m * G + G := Nat.succ_mul m G
  simp
  omega

-- § plan buffer layout

/-- Count of 32-bit words before the pointer arrays: counts(4) + start(cap+1) + dst(cap) + tok(cap). -/
def i32Count (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- 32-bit offset of the pointer arrays, rounded up to even so 64-bit loads are 8-byte aligned. -/
def ptrOff (cap : Nat) : Nat := (i32Count cap + 1) / 2 * 2

/-- Total 32-bit words of one plan slot. -/
def planSize (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The pointer offset is even. -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer arrays begin after the counts/start/dst/tok arrays. -/
theorem ptrOff_ge (cap : Nat) : i32Count cap ≤ ptrOff cap := by
  unfold ptrOff; omega

/-- At most one word of padding. -/
theorem ptrOff_lt (cap : Nat) : ptrOff cap < i32Count cap + 2 := by
  unfold ptrOff; omega

/-- The `start2` array ends inside the plan slot. -/
theorem start2_fits (cap : Nat) : ptrOff cap + 4 * cap + (cap + 1) ≤ planSize cap := by
  unfold planSize; omega

/-- `post` recomputes the offset from `4 + (capx+1) + 2*capx`; it equals `init`'s. -/
theorem ptrOff_consistent (cap : Nat) : ((4 + (cap + 1) + 2 * cap) + 1) / 2 * 2 = ptrOff cap := by
  unfold ptrOff i32Count; omega

/-- The two plan slots (one per group) do not overlap: slot `g` starts at `g * planSize`. -/
theorem plan_slots_disjoint (cap : Nat) : planSize cap + 0 ≤ 1 * planSize cap := by
  omega

-- § monotone flag

/-- `raise_flag`: only ever raises. -/
def raise (cur v : Nat) : Nat := if cur < v then v else cur

/-- The flag never decreases. -/
theorem raise_ge_cur (cur v : Nat) : cur ≤ raise cur v := by
  unfold raise; by_cases h : cur < v <;> simp [h] <;> omega

/-- The flag reaches at least the requested value. -/
theorem raise_ge_val (cur v : Nat) : v ≤ raise cur v := by
  unfold raise; by_cases h : cur < v <;> simp [h] <;> omega

/-- Late arrivals of earlier layers cannot lower the flag. -/
theorem raise_comm (a b c : Nat) : raise (raise a b) c = raise (raise a c) b := by
  unfold raise
  by_cases h1 : a < b <;> by_cases h2 : a < c <;> by_cases h3 : b < c <;> by_cases h4 : c < b <;>
    simp [h1, h2, h3, h4] <;> omega

/-- Raising twice to the same value is the same as once. -/
theorem raise_idem (cur v : Nat) : raise (raise cur v) v = raise cur v := by
  unfold raise
  by_cases h : cur < v <;> simp [h]

-- § commit rows

/-- Commit table entry for token `t`: its position if kept, `-1` otherwise. -/
def commitPos (pos0 nKeep t : Nat) : Int := if t < nKeep then ((pos0 + t : Nat) : Int) else -1

/-- Kept tokens carry non-negative consecutive positions. -/
theorem commitPos_kept (pos0 nKeep t : Nat) (h : t < nKeep) :
    commitPos pos0 nKeep t = ((pos0 + t : Nat) : Int) := by
  simp [commitPos, h]

/-- Dropped tokens are marked `-1`. -/
theorem commitPos_dropped (pos0 nKeep t : Nat) (h : nKeep ≤ t) : commitPos pos0 nKeep t = -1 := by
  have : ¬ t < nKeep := by omega
  simp [commitPos, this]

/-- Consecutive kept tokens have consecutive positions. -/
theorem commitPos_succ (pos0 nKeep t : Nat) (h : t + 1 < nKeep) :
    commitPos pos0 nKeep (t + 1) = commitPos pos0 nKeep t + 1 := by
  have h1 : t < nKeep := by omega
  simp [commitPos, h, h1]
  omega

/-- The last kept index `n_keep - 1` is itself a kept token. -/
theorem last_kept (nKeep : Nat) (h : 1 ≤ nKeep) : nKeep - 1 < nKeep := by
  omega

-- § staging between groups

/-- Staging blobs per group. -/
def stagingPer (G k : Nat) : Nat := if G = 2 then k / 2 else k

/-- A group's staging range fits inside the staging area and ranges of different groups do not overlap. -/
theorem staging_fits (G k g : Nat) (hG : G = 1 ∨ G = 2) (hg : g < G) :
    g * stagingPer G k + stagingPer G k ≤ k := by
  unfold stagingPer
  rcases hG with rfl | rfl
  · have : g = 0 := by omega
    subst this; simp
  · have : g = 0 ∨ g = 1 := by omega
    rcases this with rfl | rfl <;> simp <;> omega

end Strata.Verify
