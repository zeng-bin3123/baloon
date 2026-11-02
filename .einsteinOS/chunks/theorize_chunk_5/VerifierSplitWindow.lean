/-!
# MODULE: verifier_split_window
Token-group splitting, ring numbering, staging partition, commit rows and monotone flags of the
speculative-decoding verify window.

## Source semantics
* With two groups, group A is `[0, (T+1)/2)` and group B the rest; they partition `[0, T)` and A is never smaller.
* Ring number of (layer `l`, group `g`) is `l*G + g + 1`; stepping `k = l*G + g` decodes back via `k / G`, `k % G`.
* The staging area is split evenly between groups; each group's slice fits inside it.
* Commit rows hold `pos0 + t` for kept tokens and `-1`/none otherwise; the kept-count index is below the count.
* Flag B only rises (compare-and-raise is a max).
* The plan's pointer offset is rounded to an even number of int32 words.

## Form template identifiers
* `tb_`, `te_` → `tb`, `te`
* `ring = l*G + grp + 1` → `ring`
* `per = G == 2 ? kStagingBlobs/2 : kStagingBlobs` → `perGroup`
* `h_commit_[2+t]` → `commitAt`
* `raise_flag` → `raiseFlag`
* `ptr_off = (i32+1) & ~1` → `ptrOff`
-/

namespace Strata.Verify

-- § token groups

/-- Start of group `g` in a window of `T` tokens. -/
def tb (T g : Nat) : Nat := if g = 0 then 0 else (T + 1) / 2

/-- End of group `g` when there are `G` groups. -/
def te (G T g : Nat) : Nat := if g = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- Two groups tile the window: A ends where B begins. -/
theorem groups_abut (T : Nat) : te 2 T 0 = tb T 1 := by
  simp [te, tb]

/-- Group B ends at the window's end. -/
theorem groupB_end (G T : Nat) : te G T 1 = T := by
  simp [te]

/-- The two group sizes add up to `T`. -/
theorem group_sizes (T : Nat) : (te 2 T 0 - tb T 0) + (te 2 T 1 - tb T 1) = T := by
  simp [te, tb]
  omega

/-- The first group is never smaller than the second. -/
theorem groupA_ge_B (T : Nat) : te 2 T 1 - tb T 1 ≤ te 2 T 0 - tb T 0 := by
  simp [te, tb]
  omega

/-- A single group is the whole window. -/
theorem single_group (T : Nat) : tb T 0 = 0 ∧ te 1 T 0 = T := by
  simp [tb, te]

-- § ring numbers

/-- Ring number of (layer, group). -/
def ring (l G grp : Nat) : Nat := l * G + grp + 1

/-- Decoding a step index and re-encoding gives the step's ring number. -/
theorem ring_decode (k G : Nat) : ring (k / G) G (k % G) = k + 1 := by
  unfold ring
  have h := Nat.div_add_mod k G
  have h2 : k / G * G = G * (k / G) := Nat.mul_comm _ _
  omega

/-- Ring numbers are strictly increasing in the step index. -/
theorem ring_mono (G k : Nat) : ring (k / G) G (k % G) < ring ((k + 1) / G) G ((k + 1) % G) := by
  rw [ring_decode, ring_decode]
  omega

-- § staging partition

/-- Staging blobs per group. -/
def perGroup (G S : Nat) : Nat := if G = 2 then S / 2 else S

/-- Group slices stay inside the staging area. -/
theorem group_slice_fits (S grp : Nat) (hg : grp < 2) : grp * perGroup 2 S + perGroup 2 S ≤ S := by
  unfold perGroup
  have : grp = 0 ∨ grp = 1 := by omega
  rcases this with rfl | rfl <;> simp <;> omega

/-- With one group the whole area is its slice. -/
theorem single_slice (S : Nat) : perGroup 1 S = S := by
  simp [perGroup]

-- § commit rows

/-- Commit row for token `t` when `n_keep` tokens are kept. -/
def commitAt (nKeep pos0 t : Nat) : Option Nat := if t < nKeep then some (pos0 + t) else none

/-- Kept tokens commit their absolute position. -/
theorem commitAt_kept (nKeep pos0 t : Nat) (h : t < nKeep) : commitAt nKeep pos0 t = some (pos0 + t) := by
  unfold commitAt; rw [if_pos h]

/-- Dropped tokens commit nothing. -/
theorem commitAt_dropped (nKeep pos0 t : Nat) (h : nKeep ≤ t) : commitAt nKeep pos0 t = none := by
  unfold commitAt
  rw [if_neg (by omega)]

/-- The "last kept" index `n_keep - 1` is a kept token. -/
theorem lastKept_lt (nKeep : Nat) (h : 1 ≤ nKeep) : nKeep - 1 < nKeep := by
  omega

-- § flags and plan layout

/-- Compare-and-raise on a monotone flag. -/
def raiseFlag (cur v : Nat) : Nat := if cur < v then v else cur

/-- Raising never lowers the flag. -/
theorem raiseFlag_ge_cur (cur v : Nat) : cur ≤ raiseFlag cur v := by
  unfold raiseFlag; split <;> omega

/-- After raising, the flag has reached the requested value. -/
theorem raiseFlag_ge_val (cur v : Nat) : v ≤ raiseFlag cur v := by
  unfold raiseFlag; split <;> omega

/-- Raising is exactly a maximum. -/
theorem raiseFlag_eq_max (cur v : Nat) : raiseFlag cur v = max cur v := by
  unfold raiseFlag; split <;> omega

/-- Raising is idempotent: a late host function cannot lower a later layer's value. -/
theorem raiseFlag_idem (cur v : Nat) : raiseFlag (raiseFlag cur v) v = raiseFlag cur v := by
  simp only [raiseFlag_eq_max]; omega

/-- Pointer offset rounded up to an even word count. -/
def ptrOff (i32 : Nat) : Nat := (i32 + 1) / 2 * 2

/-- The offset is even, so the 64-bit pointers are 8-byte aligned. -/
theorem ptrOff_even (i32 : Nat) : ptrOff i32 % 2 = 0 := by
  unfold ptrOff; omega

/-- The offset leaves room for the int32 prefix. -/
theorem le_ptrOff (i32 : Nat) : i32 ≤ ptrOff i32 := by
  unfold ptrOff; omega

end Strata.Verify
