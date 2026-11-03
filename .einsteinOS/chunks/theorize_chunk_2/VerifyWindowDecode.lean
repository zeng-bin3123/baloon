/-!
# MODULE: verify_window_decode
Speculative verification window: T draft tokens are processed as one captured graph, optionally split
into two token groups so CPU expert work of one group overlaps GPU work of the other; commit then keeps a prefix.

## Source semantics
- With G = 2 the groups are A = [0, (T+1)/2) and B = [(T+1)/2, T); with G = 1 one group covers [0, T).
- The ring for (layer l, group g) is l*G + g + 1, matching the host's k-th wait with k = l*G + g.
- commit: positions pos0 + t for t < n_keep, and -1 for the rest.

## Form template identifiers
tb_, te_ -> groupBegin, groupEnd ; ring -> ringNo ; h_commit_[2+t] -> commitPos
-/

namespace Strata.Verify

-- § token groups

/-- Start of group `grp` for window size `T`. -/
def groupBegin (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- End of group `grp` for `G` groups. -/
def groupEnd (T G grp : Nat) : Nat := if grp = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- With two groups the sizes add up to T. -/
theorem split_sizes_sum (T : Nat) :
    (groupEnd T 2 0 - groupBegin T 0) + (groupEnd T 2 1 - groupBegin T 1) = T := by
  simp [groupEnd, groupBegin]
  omega

/-- Group A is at least as large as group B. -/
theorem split_A_ge_B (T : Nat) : groupEnd T 2 1 - groupBegin T 1 ≤ groupEnd T 2 0 - groupBegin T 0 := by
  simp [groupEnd, groupBegin]
  omega

/-- With one group, group 0 is the whole window. -/
theorem single_group_whole (T : Nat) : groupEnd T 1 0 - groupBegin T 0 = T := by
  simp [groupEnd, groupBegin]

-- § ring numbering

/-- Ring number for layer `l`, group `grp`. -/
def ringNo (l G grp : Nat) : Nat := l * G + grp + 1

/-- Round trip of the host's counter k = l*G + grp. -/
theorem div_mod_roundtrip (l G grp : Nat) (hG : 0 < G) (hg : grp < G) :
    (l * G + grp) / G = l ∧ (l * G + grp) % G = grp := by
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hG, Nat.div_eq_of_lt hg]
    simp
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hg]

/-- The graph's ring number equals the host's expected value k+1. -/
theorem ringNo_eq (l G grp : Nat) : ringNo l G grp = (l * G + grp) + 1 := rfl

/-- Rings are distinct across (layer, group) pairs of a window. -/
theorem ringNo_inj (l1 l2 G g1 g2 : Nat) (hg1 : g1 < G) (hg2 : g2 < G)
    (h : ringNo l1 G g1 = ringNo l2 G g2) : l1 = l2 ∧ g1 = g2 := by
  unfold ringNo at h
  have hG : 0 < G := by omega
  have a := div_mod_roundtrip l1 G g1 hG hg1
  have b := div_mod_roundtrip l2 G g2 hG hg2
  have h' : l1 * G + g1 = l2 * G + g2 := by omega
  rw [h'] at a
  exact ⟨by omega, by omega⟩

-- § commit

/-- Position written to the commit vector for slot t. -/
def commitPos (pos0 nKeep t : Int) : Int := if t < nKeep then pos0 + t else -1

/-- Kept slots carry consecutive positions. -/
theorem commitPos_kept (pos0 nKeep t : Int) (h : t < nKeep) : commitPos pos0 nKeep t = pos0 + t := by
  simp [commitPos, h]

/-- Dropped slots carry -1. -/
theorem commitPos_dropped (pos0 nKeep t : Int) (h : ¬ t < nKeep) : commitPos pos0 nKeep t = -1 := by
  simp [commitPos, h]

/-- The commit count is valid only in 1..last window size. -/
def commitCountOk (nKeep lastT : Nat) : Prop := 1 ≤ nKeep ∧ nKeep ≤ lastT

/-- A valid commit leaves at least one token for the PLE history shift (index nKeep-1). -/
theorem commit_hist_index (n lastT : Nat) (h : commitCountOk n lastT) : n - 1 < lastT := by
  unfold commitCountOk at h; omega

end Strata.Verify
