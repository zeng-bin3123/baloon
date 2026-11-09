/-!
# MODULE: verify_window
SOURCES: src/core/verify.cpp
The speculative-decoding verify window: token-group split, ring/flag numbering, monotone flag B, the GPU plan
layout in mapped memory, staging split and the commit rows.

## Source semantics
* With `groups = 2` tokens split into A = `[0, (T+1)/2)` and B = `[(T+1)/2, T)`; otherwise one group.
* Ring `k` (0-based) is layer `k / G`, group `k % G`, and waits for value `k + 1 = l*G + grp + 1`.
* Flag B only rises (compare-and-swap maximum), so out-of-order host functions are harmless.
* Plan layout: `counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)`.
* `commit(n_keep)`: `h_commit = [n_keep, n_keep-1, pos0+t for t<n_keep else -1 ...]`.

## Form template identifiers
tb/te ↦ `tb_[2]`, `te_[2]`; ring ↦ `l*G+grp+1`; raise ↦ `raise_flag`; ptrOff ↦ `(i32+1)&~1`.
-/
namespace Strata.VerifyWindow

-- § token groups

/-- Group begin. -/
def tb (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- Group end for `G` groups. -/
def te (G T grp : Nat) : Nat := if grp = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- With two groups the groups tile `[0,T)`: A ends where B begins. -/
theorem groups_tile (T : Nat) : te 2 T 0 = tb T 1 ∧ tb T 0 = 0 ∧ te 2 T 1 = T := by
  simp [te, tb]

/-- Sizes of the two groups add to T. -/
theorem groups_size (T : Nat) : (te 2 T 0 - tb T 0) + (te 2 T 1 - tb T 1) = T := by
  simp [te, tb]; omega

/-- With `T >= 2` both groups are non-empty. -/
theorem groups_nonempty (T : Nat) (h : 2 ≤ T) : 0 < te 2 T 0 - tb T 0 ∧ 0 < te 2 T 1 - tb T 1 := by
  simp [te, tb]; omega

/-- Group A is at least as large as group B (ceil split). -/
theorem groups_balanced (T : Nat) : te 2 T 1 - tb T 1 ≤ te 2 T 0 - tb T 0 := by
  simp [te, tb]; omega

/-- A single group is the whole window. -/
theorem single_group (T : Nat) : tb T 0 = 0 ∧ te 1 T 0 = T := by simp [tb, te]

-- § ring numbering

/-- The value the `k`-th ring waits for is `l*G + grp + 1` with `l = k/G`, `grp = k%G`. -/
theorem ring_numbering (k G : Nat) : G * (k / G) + k % G + 1 = k + 1 := by
  have h := Nat.div_add_mod k G
  omega

/-- The group index is a valid group. -/
theorem ring_group_lt (k G : Nat) (hG : 0 < G) : k % G < G := Nat.mod_lt _ hG

/-- Rings of consecutive `(l, grp)` pairs are consecutive. -/
theorem ring_consecutive (G l grp : Nat) (h : grp + 1 < G) :
    G * l + (grp + 1) + 1 = (G * l + grp + 1) + 1 := by omega

-- § flag B

/-- `raise_flag`: only ever raises. -/
def raise (cur v : Nat) : Nat := max cur v

/-- Raising never lowers the flag. -/
theorem raise_ge_cur (cur v : Nat) : cur ≤ raise cur v := by unfold raise; omega

/-- Raising reaches the requested value. -/
theorem raise_ge_val (cur v : Nat) : v ≤ raise cur v := by unfold raise; omega

/-- Late host functions commute: order of arrival does not matter. -/
theorem raise_comm (c a b : Nat) : raise (raise c a) b = raise (raise c b) a := by
  unfold raise; omega

/-- Replaying the same raise is idempotent. -/
theorem raise_idem (c v : Nat) : raise (raise c v) v = raise c v := by unfold raise; omega

-- § plan layout

/-- Offset of the pointer array: the i32 prefix rounded up to even. -/
def ptrOff (cap : Nat) : Nat := ((4 + (cap + 1) + cap + cap) + 1) / 2 * 2

/-- The pointer array is 8-byte aligned in i32 units (even offset). -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by unfold ptrOff; omega

/-- The pad is at most one i32 and the pointer array starts after `tok`. -/
theorem ptrOff_bounds (cap : Nat) :
    4 + (cap + 1) + cap + cap ≤ ptrOff cap ∧ ptrOff cap ≤ 4 + (cap + 1) + cap + cap + 1 := by
  unfold ptrOff; omega

/-- Total i32 count of one plan slot. -/
def planI32 (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- `ptr` and `ptr2` (2 i32 per u64, `cap` each) end exactly where `start2` begins. -/
theorem ptr2_ends_at_start2 (cap : Nat) : ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap := by
  omega

/-- `start2` (cap+1 entries) fits strictly inside the plan slot. -/
theorem start2_fits (cap : Nat) : ptrOff cap + 4 * cap + (cap + 1) < planI32 cap := by
  unfold planI32; omega

-- § staging split

/-- Blobs of staging per group. -/
def stagePer (G S : Nat) : Nat := if G = 2 then S / 2 else S

/-- Each group's staging window lies inside the staging area. -/
theorem stage_window_fits (G S grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    grp * stagePer G S + stagePer G S ≤ S := by
  rcases hG with rfl | rfl
  · have : grp = 0 := by omega
    subst this
    simp [stagePer]
  · have : grp = 0 ∨ grp = 1 := by omega
    rcases this with rfl | rfl <;> simp [stagePer] <;> omega

/-- The two windows of a split window are disjoint. -/
theorem stage_windows_disjoint (S : Nat) : 0 * stagePer 2 S + stagePer 2 S ≤ 1 * stagePer 2 S := by
  omega

-- § commit rows

/-- Entry `t` of the commit array (`-1` marks a discarded token). -/
def commitEntry (pos0 : Int) (nKeep t : Nat) : Int := if t < nKeep then pos0 + t else -1

/-- Kept tokens carry consecutive positions. -/
theorem commit_kept (pos0 : Int) (nKeep t : Nat) (h : t < nKeep) : commitEntry pos0 nKeep t = pos0 + t := by
  unfold commitEntry; rw [if_pos h]

/-- Dropped tokens are marked `-1`. -/
theorem commit_dropped (pos0 : Int) (nKeep t : Nat) (h : nKeep ≤ t) : commitEntry pos0 nKeep t = -1 := by
  unfold commitEntry; rw [if_neg (by omega)]

/-- The second header word (`n_keep - 1`) addresses the last kept token. -/
theorem commit_last_index (nKeep : Nat) (h : 1 ≤ nKeep) : nKeep - 1 < nKeep := by omega

end Strata.VerifyWindow
