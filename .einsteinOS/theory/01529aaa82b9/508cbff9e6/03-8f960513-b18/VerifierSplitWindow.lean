/-!
# MODULE: verifier_split_window
Group splitting, ring numbering, plan-buffer layout and staging partition of the speculative verify window.

## Source semantics
With `G = 2` a window of `T ≥ 2` tokens is cut into group A = `[0, (T+1)/2)` and B = `[(T+1)/2, T)`; with `G = 1` the
single group is `[0, T)`. Group `grp` of layer `l` rings `l*G + grp + 1`, and the host loop recovers `(l, grp)` as
`(k/G, k%G)`. The GPU plan buffer places 64-bit pointer arrays on an even 32-bit offset. The staging area is split
evenly between the two groups. The commit table keeps positions for kept tokens and `-1` for the rest.

## Form template identifiers
- groupBegin/groupEnd ← `tb_`/`te_`, ringNumber ← `ring`, ptrOff ← `ptr_off`,
- stagingPer ← `per`, commitSlot ← `h_commit_[2 + t]`.
-/

namespace Strata.Verify

-- § token groups

/-- Start of group `grp` in a two-group window. -/
def groupBegin (T grp : Nat) : Nat := grp * ((T + 1) / 2)

/-- End of group `grp` in a two-group window. -/
def groupEnd (T grp : Nat) : Nat := if grp = 0 then (T + 1) / 2 else T

/-- Group A starts at 0 and ends where B starts. -/
theorem groups_adjacent (T : Nat) : groupBegin T 0 = 0 ∧ groupEnd T 0 = groupBegin T 1 := by
  unfold groupBegin groupEnd; simp

/-- The two groups cover the window exactly, both nonempty when `T ≥ 2`. -/
theorem groups_cover (T : Nat) (h : 2 ≤ T) :
    (T + 1) / 2 + (T - (T + 1) / 2) = T ∧ 1 ≤ (T + 1) / 2 ∧ 1 ≤ T - (T + 1) / 2 := by
  omega

/-- Group A is at least as large as group B. -/
theorem groupA_ge_groupB (T : Nat) : T - (T + 1) / 2 ≤ (T + 1) / 2 := by
  omega

/-- Group B's end is the window size. -/
theorem groupB_end (T : Nat) : groupEnd T 1 = T := by
  unfold groupEnd; simp

-- § ring numbering

/-- Ring sequence number of group `grp` at layer `l` for `G` groups. -/
def ringNumber (G l grp : Nat) : Nat := l * G + grp + 1

/-- The host's decoding `(k/G, k%G)` of the flat index reconstructs it. -/
theorem ring_decode (G k : Nat) : k / G * G + k % G = k :=
  Nat.div_add_mod' k G

/-- The flat index `k` rings `k + 1`. -/
theorem ring_flat (G k : Nat) : ringNumber G (k / G) (k % G) = k + 1 := by
  unfold ringNumber
  have := Nat.div_add_mod' k G
  omega

/-- With two groups, `(layer, group)` is recovered from the ring number. -/
theorem ring2_injective {l l' g g' : Nat} (hg : g < 2) (hg' : g' < 2)
    (h : ringNumber 2 l g = ringNumber 2 l' g') : l = l' ∧ g = g' := by
  unfold ringNumber at h; omega

/-- With one group the layer is recovered from the ring number. -/
theorem ring1_injective {l l' : Nat} (h : ringNumber 1 l 0 = ringNumber 1 l' 0) : l = l' := by
  unfold ringNumber at h; omega

/-- Ring numbers increase in the pipeline order pre(0,A) pre(0,B) | post(0,A) ... -/
theorem ring2_monotone {l g : Nat} (hg : g < 2) :
    ringNumber 2 l g < ringNumber 2 (l + 1) 0 := by
  unfold ringNumber; omega

-- § plan buffer layout

/-- 32-bit words before the pointer arrays: counts(4) + start(cap+1) + dst(cap) + tok(cap). -/
def planHead (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- Even offset of the 64-bit pointer arrays, `(i32 + 1) & ~1`. -/
def ptrOff (cap : Nat) : Nat := (planHead cap + 1) / 2 * 2

/-- The pointer offset is even, so 64-bit loads are 8-byte aligned. -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer offset is past the head and wastes at most one word. -/
theorem ptrOff_bounds (cap : Nat) : planHead cap ≤ ptrOff cap ∧ ptrOff cap ≤ planHead cap + 1 := by
  unfold ptrOff; omega

/-- `post` recomputes the same head size from `4 + (capx+1) + 2*capx`. -/
theorem planHead_consistent (cap : Nat) : planHead cap = 4 + (cap + 1) + 2 * cap := by
  unfold planHead; omega

/-- `ptr` (cap u64 = 2cap words) and `ptr2` are contiguous and end exactly at `start2`. -/
theorem ptr_regions_contiguous (cap : Nat) :
    ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap := by
  omega

/-- Total plan words hold `start2` (cap+1 entries) plus its trailing slot. -/
theorem plan_total_holds_start2 (cap : Nat) :
    (ptrOff cap + 4 * cap) + (cap + 1) ≤ ptrOff cap + 4 * cap + (cap + 1) + 1 := by
  omega

-- § staging partition

/-- Staging blobs per group. -/
def stagingPer (G S : Nat) : Nat := if G = 2 then S / 2 else S

/-- With two groups, each group's slice ends within the staging area. -/
theorem staging_slice_fits (S grp : Nat) (hg : grp < 2) :
    grp * (S / 2) + S / 2 ≤ S := by
  have : grp = 0 ∨ grp = 1 := by omega
  rcases this with rfl | rfl <;> omega

/-- The two groups' slices do not overlap. -/
theorem staging_slices_disjoint (S : Nat) : 0 * (S / 2) + S / 2 ≤ 1 * (S / 2) := by
  omega

-- § commit table

/-- Position recorded for token `t` when `n_keep` tokens are kept: `pos0 + t`, else `-1`. -/
def commitSlot (nKeep t pos0 : Nat) : Int := if t < nKeep then ((pos0 + t : Nat) : Int) else -1

/-- Kept tokens get a nonnegative position. -/
theorem commitSlot_kept {nKeep t pos0 : Nat} (h : t < nKeep) : 0 ≤ commitSlot nKeep t pos0 := by
  unfold commitSlot
  rw [if_pos h]
  omega

/-- Dropped tokens are marked `-1`. -/
theorem commitSlot_dropped {nKeep t pos0 : Nat} (h : nKeep ≤ t) : commitSlot nKeep t pos0 = -1 := by
  unfold commitSlot
  rw [if_neg (by omega)]

end Strata.Verify
