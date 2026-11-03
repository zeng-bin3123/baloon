/-!
# MODULE: verify_window_groups
Index arithmetic of the speculative-decoding verify window: the two-group split of the window's tokens, the
per-(layer, group) ring numbers the host waits for, the mapped GPU-plan offsets and the commit bookkeeping.

## Source semantics
- With `split` and `T >= 2` the window is cut into group A = `[0, (T+1)/2)` and group B = `[(T+1)/2, T)`;
  otherwise there is one group `[0, T)`.
- Ring `k = l * G + grp` is awaited as `k + 1`; `l = k / G` and `grp = k % G` recover the pair.
- The plan's pointer region starts at an even `int32` offset: `(i32 + 1) & ~1`.
- A commit of `n_keep` tokens keeps positions `pos0 .. pos0 + n_keep - 1` and `n_keep - 1` indexes the last kept row.

## Form template identifiers
- `groups`, `teA`, `tbB` <- `G`, `te_[0]`, `tb_[1]` in `record_window`
- `ringNo`        <- `ring = l * G + grp + 1`
- `ptrOff`        <- `(i32 + 1) & ~1ll`
- `commitPos`     <- `h_commit_[2 + t]`
-/

namespace Strata.Verify

/-- Number of token groups. -/
def groups (split : Bool) (T : Nat) : Nat := if split = true ∧ 2 ≤ T then 2 else 1

/-- End of group A. -/
def teA (G T : Nat) : Nat := if G = 2 then (T + 1) / 2 else T

/-- Start of group B. -/
def tbB (T : Nat) : Nat := (T + 1) / 2

/-- Ring number awaited for `(l, grp)`. -/
def ringNo (l G grp : Nat) : Nat := l * G + grp + 1

/-- Even `int32` offset of the pointer region. -/
def ptrOff (i32 : Nat) : Nat := (i32 + 1) / 2 * 2

/-- The cache position committed at window row `t` (none past `n_keep`). -/
def commitPos (pos0 nKeep t : Nat) : Option Nat := if t < nKeep then some (pos0 + t) else none

/-- There is always one or two groups. -/
theorem groups_cases (split : Bool) (T : Nat) : groups split T = 1 ∨ groups split T = 2 := by
  unfold groups
  split <;> simp

/-- The two groups of a split window partition the `T` tokens. -/
theorem groups_partition (T : Nat) : (teA 2 T - 0) + (T - tbB T) = T := by
  unfold teA tbB
  rw [if_pos rfl]
  omega

/-- Group A of a split window is the larger half. -/
theorem groupA_ge_B (T : Nat) : T - tbB T ≤ teA 2 T := by
  unfold teA tbB
  rw [if_pos rfl]
  omega

/-- Both groups are non-empty when the window holds at least two tokens. -/
theorem groups_nonempty (T : Nat) (h : 2 ≤ T) : 0 < teA 2 T ∧ 0 < T - tbB T := by
  unfold teA tbB
  rw [if_pos rfl]
  omega

/-- A single-group window is the whole range. -/
theorem single_group (T : Nat) : teA 1 T = T := by
  unfold teA
  simp

/-- `(k / G, k % G)` recovers the ring index. -/
theorem ring_roundtrip (G k : Nat) : (k / G) * G + k % G = k := by
  have h := Nat.div_add_mod k G
  rw [Nat.mul_comm] at h
  exact h

/-- The group index is below `G`. -/
theorem ring_grp_lt (G k : Nat) (hG : 0 < G) : k % G < G := Nat.mod_lt k hG

/-- Ring numbers of distinct `(l, grp)` pairs differ. -/
theorem ringNo_inj (G l grp l' grp' : Nat) (hg : grp < G) (hg' : grp' < G)
    (h : ringNo l G grp = ringNo l' G grp') : l = l' ∧ grp = grp' := by
  unfold ringNo at h
  have h1 : (l * G + grp) / G = l := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hg]; simp
  have h2 : (l' * G + grp') / G = l' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hg']; simp
  have h3 : l * G + grp = l' * G + grp' := by omega
  have h4 : l = l' := by rw [← h1, ← h2, h3]
  subst h4
  omega

/-- Ring numbers start at 1 and increase in launch order. -/
theorem ringNo_pos (l G grp : Nat) : 0 < ringNo l G grp := by
  unfold ringNo; omega

/-- The pointer region offset is even. -/
theorem ptrOff_even (i : Nat) : ptrOff i % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer region begins at or after the `int32` header. -/
theorem ptrOff_ge (i : Nat) : i ≤ ptrOff i := by
  unfold ptrOff; omega

/-- It wastes at most one `int32`. -/
theorem ptrOff_le (i : Nat) : ptrOff i ≤ i + 1 := by
  unfold ptrOff; omega

/-- Rows kept by a commit carry consecutive positions. -/
theorem commitPos_some (pos0 nKeep t : Nat) (h : t < nKeep) : commitPos pos0 nKeep t = some (pos0 + t) := by
  unfold commitPos
  simp [h]

/-- Rows past `n_keep` are marked unused. -/
theorem commitPos_none (pos0 nKeep t : Nat) (h : nKeep ≤ t) : commitPos pos0 nKeep t = none := by
  unfold commitPos
  have : ¬ (t < nKeep) := by omega
  simp [this]

/-- The last kept row index is below the keep count. -/
theorem commit_last_lt (nKeep : Nat) (h : 1 ≤ nKeep) : nKeep - 1 < nKeep := by
  omega

/-- A window that fits the context never writes a position at or past it. -/
theorem window_in_context (pos0 T maxCells t : Nat) (h : pos0 + T ≤ maxCells) (ht : t < T) :
    pos0 + t < maxCells := by
  omega

end Strata.Verify
