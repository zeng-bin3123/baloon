/-!
# MODULE: verify_window_schedule
Scheduling arithmetic of the speculative verify window (`verify.cpp`): the split into two token groups, the
ring/flag numbering, flag-B monotone raise, the mapped GPU-plan layout, staging and plan slabs per group, and
commit row bookkeeping.

## Source semantics
* With `G = 2` the window `[0,T)` is cut into `A = [0,(T+1)/2)` and `B = [(T+1)/2, T)`; they tile `[0,T)` and are
  both non-empty for `T >= 2`.
* Ring number of `(layer, group)` is `layer*G + group + 1`; the host loop recovers `(layer, group)` from
  `k = layer*G + group` by `k / G` and `k % G`.
* `raise_flag` only raises: it equals the maximum of the old value and the request, and raises commute.
* Plan layout: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1);
  the 64-bit regions start on an even index and everything ends within `plan_i32`.
* The two groups' plan slabs and staging halves are disjoint and inside their allocations.
* `commit(n_keep)` marks rows `t < n_keep` with `pos0 + t` and the rest with `-1`.

## Form template identifiers
* `tb_[2]`, `te_[2]` -> `tb`, `te`; `ring` -> `ringOf`; `raise_flag` -> `raise`
* `ptr_off`, `plan_i32_` -> `ptrOff`, `planI32`; `staging_ + grp*per*max_blob` -> `stageOff`
* `h_commit_[2+t]` -> `commitRow`
-/

namespace VerifyWindowSchedule

-- § token groups

/-- Group begin: group 0 starts at 0, group 1 at `(T+1)/2`. -/
def tb (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- Group end for `G` groups: group 0 ends at `(T+1)/2` when split, else at `T`; group 1 ends at `T`. -/
def te (T G grp : Nat) : Nat := if grp = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- The two groups of a split window tile `[0, T)`: group 0 ends where group 1 begins, group 1 ends at `T`. -/
theorem groups_tile (T : Nat) :
    tb T 0 = 0 ∧ te T 2 0 = tb T 1 ∧ te T 2 1 = T := by
  simp [tb, te]

/-- Both groups of a split window are non-empty for `T >= 2`. -/
theorem groups_nonempty (T : Nat) (h : 2 ≤ T) :
    tb T 0 < te T 2 0 ∧ tb T 1 < te T 2 1 := by
  simp [tb, te]
  omega

/-- Group sizes add up to `T`. -/
theorem group_sizes (T : Nat) : (te T 2 0 - tb T 0) + (te T 2 1 - tb T 1) = T := by
  simp [tb, te]
  omega

/-- Without a split the single group is the whole window. -/
theorem single_group (T : Nat) : tb T 0 = 0 ∧ te T 1 0 = T := by
  simp [tb, te]

-- § ring numbering

/-- Ring (sequence) number of `(layer, group)`. -/
def ringOf (G layer grp : Nat) : Nat := layer * G + grp + 1

/-- With `G` in `{1, 2}` the host's `k / G`, `k % G` recovers `(layer, group)` from `k = layer*G + group`. -/
theorem ring_decode (G layer grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    (layer * G + grp) / G = layer ∧ (layer * G + grp) % G = grp := by
  rcases hG with rfl | rfl <;> omega

/-- Ring numbers are strictly increasing in the loop order `k = layer*G + group`. -/
theorem ringOf_eq (G layer grp : Nat) : ringOf G layer grp = (layer * G + grp) + 1 := rfl

/-- Distinct `(layer, group)` pairs have distinct ring numbers when `G` is 1 or 2. -/
theorem ring_inj (G l1 g1 l2 g2 : Nat) (hG : G = 1 ∨ G = 2) (h1 : g1 < G) (h2 : g2 < G)
    (h : ringOf G l1 g1 = ringOf G l2 g2) : l1 = l2 ∧ g1 = g2 := by
  unfold ringOf at h
  rcases hG with rfl | rfl <;> omega

/-- The last ring number is `n_layers * G`, the number of rings the loop waits for. -/
theorem ring_last (G n : Nat) (hn : 1 ≤ n) (hG : 1 ≤ G) :
    ringOf G (n - 1) (G - 1) = n * G := by
  unfold ringOf
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  obtain ⟨g, rfl⟩ : ∃ g, G = g + 1 := ⟨G - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  ring

-- § flag B only rises

/-- The effect of the compare-exchange loop: raise to `v` only if currently below. -/
def raise (cur v : Nat) : Nat := if cur < v then v else cur

/-- A raise never lowers the flag. -/
theorem raise_ge (cur v : Nat) : cur ≤ raise cur v := by
  unfold raise; split <;> omega

/-- After a raise the flag is at least the requested value. -/
theorem raise_ge_req (cur v : Nat) : v ≤ raise cur v := by
  unfold raise; split <;> omega

/-- A raise equals the maximum: nothing else is ever written. -/
theorem raise_eq_max (cur v : Nat) : raise cur v = max cur v := by
  unfold raise; split <;> omega

/-- Late host functions commute: raising by `a` then `b` equals raising by `b` then `a`. -/
theorem raise_comm (c a b : Nat) : raise (raise c a) b = raise (raise c b) a := by
  simp only [raise_eq_max]
  omega

/-- Raising to a value already reached is a no-op. -/
theorem raise_idem (cur v : Nat) (h : v ≤ cur) : raise cur v = cur := by
  unfold raise; split <;> omega

-- § the GPU plan layout (i32 indices)

/-- Index just after the three i32 arrays: `4 + (cap+1) + cap + cap`. -/
def planHead (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- Start of the 64-bit regions, rounded up to an even i32 index. -/
def ptrOff (cap : Nat) : Nat := (planHead cap + 1) / 2 * 2

/-- Total i32 slots of one plan: ptr and ptr2 take `2*cap` each, then `start2` has `cap+1`, plus one. -/
def planI32 (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The 64-bit pointer regions start on an even index (8-byte alignment of the u64 arrays). -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer region never overlaps the three i32 arrays before it. -/
theorem ptrOff_ge_head (cap : Nat) : planHead cap ≤ ptrOff cap := by
  unfold ptrOff; omega

/-- Padding is at most one slot. -/
theorem ptrOff_le_head_succ (cap : Nat) : ptrOff cap ≤ planHead cap + 1 := by
  unfold ptrOff; omega

/-- `start2` (cap + 1 entries after `ptr`, `ptr2`) ends inside the plan. -/
theorem start2_in_plan (cap : Nat) : ptrOff cap + 4 * cap + (cap + 1) ≤ planI32 cap := by
  unfold planI32; omega

/-- The two groups' plan slabs of the mapped buffer (`plan_i32` i32 each) fit in `2 * plan_i32`. -/
theorem plan_slab_host (p grp : Nat) (h : grp < 2) : grp * p + p ≤ 2 * p := by
  interval_cases grp <;> omega

/-- The device plan copy uses stride `plan_i32 + 16` and fits in `2 * (plan_i32 + 16)`. -/
theorem plan_slab_dev (p grp : Nat) (h : grp < 2) : grp * (p + 16) + p ≤ 2 * (p + 16) := by
  interval_cases grp <;> omega

-- § staging halves

/-- Number of staging blobs per group: half in a split window, all otherwise. -/
def stagePer (G k : Nat) : Nat := if G = 2 then k / 2 else k

/-- Each group's staging range `[grp*per, grp*per + per)` fits in the `k` staging blobs. -/
theorem stage_in_bounds (G k grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    grp * stagePer G k + stagePer G k ≤ k := by
  rcases hG with rfl | rfl
  · have : grp = 0 := by omega
    subst this
    simp [stagePer]
  · unfold stagePer
    simp only [if_true]
    interval_cases grp <;> omega

/-- The two staging halves are disjoint in a split window. -/
theorem stage_disjoint (k : Nat) : 0 * stagePer 2 k + stagePer 2 k ≤ 1 * stagePer 2 k := by
  omega

-- § commit rows

/-- Row `t` of the commit array: the position of an accepted token, else `-1`. -/
def commitRow (pos0 : Int) (nKeep t : Nat) : Int := if t < nKeep then pos0 + t else -1

/-- Accepted rows carry their position. -/
theorem commitRow_keep (pos0 : Int) (nKeep t : Nat) (h : t < nKeep) : commitRow pos0 nKeep t = pos0 + t := by
  simp [commitRow, h]

/-- Rows past the accepted prefix are marked `-1`. -/
theorem commitRow_drop (pos0 : Int) (nKeep t : Nat) (h : nKeep ≤ t) : commitRow pos0 nKeep t = -1 := by
  have : ¬ t < nKeep := by omega
  simp [commitRow, this]

/-- The last-kept index `n_keep - 1` is a valid window row when `1 <= n_keep <= T`. -/
theorem last_kept_valid (nKeep T : Nat) (h1 : 1 ≤ nKeep) (h2 : nKeep ≤ T) : nKeep - 1 < T := by
  omega

end VerifyWindowSchedule
