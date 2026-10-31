/-!
# MODULE: verify_window_schedule
Token grouping, ring numbering and commit bookkeeping of the speculative verify window.

## Source semantics
* With `G = 2` the `T` window tokens are cut into group A = `[0, (T+1)/2)` and group B = `[(T+1)/2, T)`; with
  `G = 1` there is a single group `[0, T)`. The groups tile `[0, T)` with no overlap.
* The ring for layer `l`, group `grp` is `l * G + grp + 1`; the host loop iterates `k = 0 .. n_layers*G - 1`
  with `l = k / G`, `grp = k % G`, `want = k + 1`. Ring numbers are the same sequence as `want`.
* The staging area is split `kStagingBlobs / 2` per group when `G = 2`.
* `commit(n_keep)` stages `commit[0] = n_keep`, `commit[1] = n_keep - 1`, and for `t < max_t` position
  `pos0 + t` if `t < n_keep` else `-1`; valid iff `1 <= n_keep <= last_t`.
* Flag B only rises (compare-and-swap max).

## Form template identifiers
* `groupBegin`, `groupEnd`   : `tb_`, `te_`
* `ringOf`                   : `l * G + grp + 1`
* `commitPos`                : committed position or -1
-/

namespace Strata.Verify

-- § token groups

/-- Group begin. -/
def groupBegin (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- Group end, for `G` groups. -/
def groupEnd (T G grp : Nat) : Nat := if grp = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- Group size. -/
def groupSize (T G grp : Nat) : Nat := groupEnd T G grp - groupBegin T grp

/-- With one group, the single group is the whole window. -/
theorem single_group (T : Nat) : groupBegin T 0 = 0 ∧ groupEnd T 1 0 = T := by
  unfold groupBegin groupEnd
  simp

/-- With two groups, A ends exactly where B begins. -/
theorem groups_abut (T : Nat) : groupEnd T 2 0 = groupBegin T 1 := by
  unfold groupEnd groupBegin
  simp

/-- With two groups, B ends at `T`. -/
theorem second_group_end (T : Nat) : groupEnd T 2 1 = T := by
  unfold groupEnd
  simp

/-- The two groups' sizes add up to `T`. -/
theorem two_groups_cover (T : Nat) : groupSize T 2 0 + groupSize T 2 1 = T := by
  unfold groupSize groupEnd groupBegin
  simp <;> omega

/-- Group A is the larger or equal half. -/
theorem first_group_ge (T : Nat) : groupSize T 2 1 ≤ groupSize T 2 0 := by
  unfold groupSize groupEnd groupBegin
  simp <;> omega

/-- A group bound never exceeds `T`. -/
theorem group_end_le (T G grp : Nat) : groupEnd T G grp ≤ T := by
  unfold groupEnd
  split_ifs <;> omega

/-- Group begin never exceeds group end for two groups. -/
theorem begin_le_end (T grp : Nat) (h : grp < 2) : groupBegin T grp ≤ groupEnd T 2 grp := by
  unfold groupBegin groupEnd
  have : grp = 0 ∨ grp = 1 := by omega
  rcases this with rfl | rfl <;> simp <;> omega

-- § ring numbering

/-- Ring value for layer `l`, group `grp`. -/
def ringOf (l G grp : Nat) : Nat := l * G + grp + 1

/-- The host loop's `want` for iteration `k`. -/
def wantOf (k : Nat) : Nat := k + 1

/-- Decoding `k = l * G + grp` recovers `l`. -/
theorem decode_layer (l G grp : Nat) (hG : 0 < G) (hg : grp < G) : (l * G + grp) / G = l := by
  rw [Nat.mul_comm, Nat.mul_add_div hG]
  simp [Nat.div_eq_of_lt hg]

/-- Decoding `k = l * G + grp` recovers `grp`. -/
theorem decode_group (l G grp : Nat) (hG : 0 < G) (hg : grp < G) : (l * G + grp) % G = grp := by
  rw [Nat.mul_comm, Nat.mul_add_mod]
  exact Nat.mod_eq_of_lt hg

/-- The ring of `(l, grp)` equals the loop's `want` for `k = l * G + grp`: host and graph number alike. -/
theorem ring_eq_want (l G grp : Nat) : ringOf l G grp = wantOf (l * G + grp) := by
  unfold ringOf wantOf; omega

/-- Rings are strictly increasing in `k`, so every flag wait is satisfied by a later ring too. -/
theorem want_strict (a b : Nat) (h : a < b) : wantOf a < wantOf b := by
  unfold wantOf; omega

/-- Distinct (layer, group) pairs get distinct rings. -/
theorem ring_injective (l1 g1 l2 g2 G : Nat) (hG : 0 < G) (h1 : g1 < G) (h2 : g2 < G)
    (h : ringOf l1 G g1 = ringOf l2 G g2) : l1 = l2 ∧ g1 = g2 := by
  have e : l1 * G + g1 = l2 * G + g2 := by unfold ringOf at h; omega
  have a := decode_layer l1 G g1 hG h1
  have b := decode_layer l2 G g2 hG h2
  have c := decode_group l1 G g1 hG h1
  have d := decode_group l2 G g2 hG h2
  rw [e] at a c
  exact ⟨by rw [← a, b], by rw [← c, d]⟩

/-- Total rings in a window: `n_layers * G`; the last want is `n_layers * G`. -/
theorem last_ring (nLayers G : Nat) (hn : 0 < nLayers) (hG : 0 < G) :
    ringOf (nLayers - 1) G (G - 1) = nLayers * G := by
  unfold ringOf
  have h1 : nLayers = (nLayers - 1) + 1 := by omega
  conv_rhs => rw [h1]
  rw [Nat.succ_mul]
  omega

-- § staging split

/-- Blobs of staging per group. -/
def stagingPer (staging G : Nat) : Nat := if G = 2 then staging / 2 else staging

/-- The two halves fit in the staging area. -/
theorem staging_halves_fit (staging : Nat) : stagingPer staging 2 + stagingPer staging 2 ≤ staging := by
  unfold stagingPer
  simp <;> omega

/-- Group `grp`'s staging offset (in blobs) is `grp * per`; group 1 starts after group 0's share. -/
theorem staging_disjoint (per : Nat) : 0 * per + per ≤ 1 * per := by omega

-- § commit

/-- Position committed at index `t`, or `-1`. -/
def commitPos (pos0 nKeep t : Nat) : Int := if t < nKeep then ((pos0 + t : Nat) : Int) else -1

/-- Valid commit arguments. -/
def commitOk (nKeep lastT : Nat) : Prop := 1 ≤ nKeep ∧ nKeep ≤ lastT

/-- A committed slot has the expected position. -/
theorem commitPos_kept (pos0 nKeep t : Nat) (h : t < nKeep) : commitPos pos0 nKeep t = ((pos0 + t : Nat) : Int) := by
  unfold commitPos
  simp [h]

/-- A dropped slot is `-1`. -/
theorem commitPos_dropped (pos0 nKeep t : Nat) (h : ¬ t < nKeep) : commitPos pos0 nKeep t = -1 := by
  unfold commitPos
  simp [h]

/-- The last kept token is at index `n_keep - 1`, which is `commit[1]`. -/
theorem commit_last_index (nKeep : Nat) (h : 1 ≤ nKeep) : nKeep - 1 < nKeep := by omega

/-- A valid commit keeps at most the window's tokens. -/
theorem commit_within_window (nKeep lastT maxT : Nat) (h : commitOk nKeep lastT) (hl : lastT ≤ maxT) :
    nKeep ≤ maxT := by
  obtain ⟨_, h2⟩ := h
  omega

-- § flag B monotone

/-- One CAS-max step on flag B. -/
def raise (cur value : Nat) : Nat := if cur < value then value else cur

/-- Raising never lowers the flag. -/
theorem raise_ge (cur value : Nat) : cur ≤ raise cur value := by
  unfold raise; split <;> omega

/-- After raising, the flag is at least `value`. -/
theorem raise_ge_value (cur value : Nat) : value ≤ raise cur value := by
  unfold raise; split <;> omega

/-- Raise is commutative on results: an earlier layer's late host function cannot undo a later raise. -/
theorem raise_comm (cur a b : Nat) : raise (raise cur a) b = raise (raise cur b) a := by
  unfold raise
  split_ifs <;> omega

end Strata.Verify
