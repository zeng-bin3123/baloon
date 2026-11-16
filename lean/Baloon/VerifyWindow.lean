-- Baloon.VerifyWindow: speculative-decode verify-window token groups, ring numbering
-- and split-window stream ordering invariants.
-- Independent implementation proving the same mathematical facts as Ref.VerifyWindow.

namespace Baloon.VerifyWindow

def groupBegin (T grp : Nat) : Nat := (if grp = 0 then 0 else (T + 1) / 2)
def groupEnd (T G grp : Nat) : Nat := (if grp = 0 then (if G = 2 then (T + 1) / 2 else T) else T)

theorem groups_tile (T : Nat) :
    groupBegin T 0 = 0 ∧ groupEnd T 2 0 = groupBegin T 1 ∧ groupEnd T 2 1 = T := by { simp [groupBegin, groupEnd] }

theorem groups_sizes (T : Nat) :
    (groupEnd T 2 0 - groupBegin T 0) + (groupEnd T 2 1 - groupBegin T 1) = T := by
  simp only [groupBegin, groupEnd]; simp; omega

theorem groups_nonempty (T : Nat) (hT : 2 ≤ T) :
    groupBegin T 0 < groupEnd T 2 0 ∧ groupBegin T 1 < groupEnd T 2 1 := by
  simp only [groupBegin, groupEnd]; simp; omega

theorem single_group (T : Nat) : groupBegin T 0 = 0 ∧ groupEnd T 1 0 = T := by { simp [groupBegin, groupEnd] }

def ringValue (l G grp : Nat) : Nat := (l * G + grp + 1)

theorem ring_host_matches (l G grp : Nat) : (l * G + grp) + 1 = ringValue l G grp := by { unfold ringValue; rfl }

theorem ring_recover_layer (l grp : Nat) (h : grp < 2) : (l * 2 + grp) / 2 = l := by { omega }
theorem ring_recover_group (l grp : Nat) (h : grp < 2) : (l * 2 + grp) % 2 = grp := by { omega }

theorem ring_strict (l grp l' grp' : Nat) (h : grp < 2) (h' : grp' < 2) (hl : l < l') :
    ringValue l 2 grp < ringValue l' 2 grp' := by { unfold ringValue; omega }

def preIdx (l g : Nat) : Nat := (if l = 0 then g else 4 * l - 1 + 2 * g)
def postIdx (l g : Nat) : Nat := (2 + 4 * l + 2 * g)

theorem pre_before_post (l g : Nat) (_hg : g < 2) : preIdx l g < postIdx l g := by
  unfold preIdx postIdx; split <;> omega

theorem post_before_next_pre (l g : Nat) (_hg : g < 2) : postIdx l g < preIdx (l + 1) g := by
  unfold preIdx postIdx; simp; omega

end Baloon.VerifyWindow
