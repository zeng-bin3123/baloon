/-!
# MODULE: verify_window
SOURCES: src/core/verify.cpp
Speculative verify window: split into two token groups, per-(layer, group) ring numbering, staging split between
groups, monotone flag B, and the empty-plan fallback.

## Source semantics
* With `G = 2`, group A is `[0, (T+1)/2)` and group B is `[(T+1)/2, T)`; together they cover `[0, T)`.
* Ring number is `l * G + grp + 1`, strictly increasing in execution order, and decodes back to `(l, grp)`.
* Staging blobs are halved per group; the halves are disjoint.
* Flag B only rises (CAS-max).

## Form template identifiers
tb_/te_ → grpBegin/grpEnd ; ring → ringNo ; raise_flag → raiseFlag ; per → stagingPer
-/

namespace Strata.Verify

-- § token groups

/-- First token of a group. -/
def grpBegin (T : Nat) : Nat → Nat
  | 0 => 0
  | _ + 1 => (T + 1) / 2

/-- One past the last token of a group. -/
def grpEnd (T G : Nat) : Nat → Nat
  | 0 => if G = 2 then (T + 1) / 2 else T
  | _ + 1 => T

/-- In a split window the groups abut. -/
theorem groups_abut (T : Nat) : grpEnd T 2 0 = grpBegin T 1 := by
  simp [grpEnd, grpBegin]

/-- In a split window the two groups together cover exactly `T` tokens. -/
theorem groups_cover (T : Nat) :
    (grpEnd T 2 0 - grpBegin T 0) + (grpEnd T 2 1 - grpBegin T 1) = T := by
  simp [grpEnd, grpBegin]
  omega

/-- With at least two tokens, both groups are non-empty. -/
theorem groups_nonempty (T : Nat) (h : 2 ≤ T) :
    grpBegin T 0 < grpEnd T 2 0 ∧ grpBegin T 1 < grpEnd T 2 1 := by
  simp [grpEnd, grpBegin]
  omega

/-- Unsplit window: the single group is the whole window. -/
theorem group_unsplit (T : Nat) : grpBegin T 0 = 0 ∧ grpEnd T 1 0 = T := by
  simp [grpEnd, grpBegin]

-- § ring numbering

/-- Ring number of layer `l`, group `grp` in a split window. -/
def ringNo (l grp : Nat) : Nat := l * 2 + grp + 1

/-- Distinct `(layer, group)` pairs have distinct ring numbers. -/
theorem ringNo_inj (l g l' g' : Nat) (hg : g < 2) (hg' : g' < 2) (h : ringNo l g = ringNo l' g') :
    l = l' ∧ g = g' := by
  unfold ringNo at h
  omega

/-- Execution order is ring order: A then B within a layer, then the next layer. -/
theorem ringNo_order (l : Nat) : ringNo l 0 < ringNo l 1 ∧ ringNo l 1 < ringNo (l + 1) 0 := by
  unfold ringNo
  omega

/-- The host's loop index `k = l*G + grp` decodes back to the layer and group. -/
theorem ring_decode (l grp : Nat) (hg : grp < 2) : (l * 2 + grp) / 2 = l ∧ (l * 2 + grp) % 2 = grp := by
  omega

/-- The wait value for step `k` is `k + 1`, equal to `ringNo`. -/
theorem ring_want (l grp : Nat) : l * 2 + grp + 1 = ringNo l grp := rfl

-- § staging split

/-- Staging blobs available to each group in a split window. -/
def stagingPer (S : Nat) : Nat := S / 2

/-- The two groups' halves fit in the staging area. -/
theorem staging_halves_fit (S : Nat) : 2 * stagingPer S ≤ S := by
  unfold stagingPer; omega

/-- Group B's slice begins where group A's ends. -/
theorem staging_disjoint (S : Nat) : 0 + stagingPer S = 1 * stagingPer S := by
  omega

-- § monotone flag

/-- CAS-max raise of a flag. -/
def raiseFlag (cur v : Nat) : Nat := if cur < v then v else cur

/-- The flag never decreases. -/
theorem raiseFlag_ge_cur (cur v : Nat) : cur ≤ raiseFlag cur v := by
  unfold raiseFlag; split <;> omega

/-- The flag reaches at least the requested value. -/
theorem raiseFlag_ge_val (cur v : Nat) : v ≤ raiseFlag cur v := by
  unfold raiseFlag; split <;> omega

/-- Raising is idempotent, so a late host function of an earlier layer cannot lower it. -/
theorem raiseFlag_idem (cur v : Nat) : raiseFlag (raiseFlag cur v) v = raiseFlag cur v := by
  unfold raiseFlag
  by_cases h : cur < v <;> simp [h]

/-- A late, smaller raise is a no-op once a larger value has landed. -/
theorem raiseFlag_late (cur v : Nat) (h : v ≤ cur) : raiseFlag cur v = cur := by
  unfold raiseFlag; split <;> omega

-- § commit mask

/-- Position written to commit slot `t`: kept tokens carry their position, the rest `-1`. -/
def commitPos (pos0 nKeep t : Nat) : Int := if t < nKeep then (pos0 + t : Nat) else -1

/-- Kept tokens carry a non-negative position. -/
theorem commitPos_kept (pos0 nKeep t : Nat) (h : t < nKeep) : 0 ≤ commitPos pos0 nKeep t := by
  unfold commitPos
  split <;> omega

/-- Dropped tokens are marked `-1`. -/
theorem commitPos_dropped (pos0 nKeep t : Nat) (h : nKeep ≤ t) : commitPos pos0 nKeep t = -1 := by
  have : ¬ t < nKeep := by omega
  simp [commitPos, this]

end Strata.Verify
