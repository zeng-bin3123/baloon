/-!
# MODULE: window_verifier
Arithmetic contracts of `src/core/verify.cpp`: the split verify window (two token groups), ring numbering,
the GPU plan buffer layout, staging halves, the monotone flag B and draft acceptance.

## Source semantics
* With `G = 2` tokens `[0,T)` split at `(T+1)/2`; with `G = 1` one group `[0,T)`. Ring of `(layer l, group g)`
  is `l*G + g + 1` and equals `k + 1` for the host's loop index `k = l*G + g`.
* Plan buffer: `counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)`;
  pointer area starts at an even int32 index so u64 pointers are 8-byte aligned.
* Staging blobs are split in halves for `G = 2`; flag B only rises.
* Acceptance `a` counts matching drafts, `a <= T-1`, hence `commit(a+1)` has `1 <= a+1 <= T`.

## Form template identifiers
* `tb`, `te0` <- `tb_[]`, `te_[]`
* `ptrOff`, `planI32` <- `ptr_off`, `plan_i32_`
* `raise` <- `Verifier::raise_flag` (final state of the CAS loop)
* `acceptFrom` <- the `while (a < T-1 && window[a+1] == outv[a])` loop
-/

namespace Strata.WindowVerifier

-- § token groups

/-- Group begin: group 0 at 0, group 1 at `(T+1)/2`. -/
def tb (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- End of group 0: `(T+1)/2` when split, else `T`. -/
def te0 (T G : Nat) : Nat := if G = 2 then (T + 1) / 2 else T

/-- Split: group 0 ends at the midpoint. -/
theorem te0_split (T : Nat) : te0 T 2 = (T + 1) / 2 := by
  simp [te0]

/-- Unsplit: group 0 is the whole window. -/
theorem te0_single (T : Nat) : te0 T 1 = T := by
  simp [te0]

/-- Both groups of a split window are non-empty when `T >= 2`, and they tile `[0,T)`. -/
theorem split_groups_tile (T : Nat) (h : 2 ≤ T) :
    0 < (T + 1) / 2 ∧ 0 < T - (T + 1) / 2 ∧ (T + 1) / 2 + (T - (T + 1) / 2) = T := by
  omega

-- § ring numbering

/-- The ring awaited for `(layer, group)` equals the host loop's `want = k + 1` where `k = layer*G + group`. -/
theorem ring_matches_loop (k G : Nat) (hG : 0 < G) : (k / G) * G + k % G + 1 = k + 1 := by
  have h := Nat.div_add_mod k G
  rw [Nat.mul_comm (k / G) G]
  omega

/-- A group index is below `G` and the layer/group decomposition covers `n*G` rings. -/
theorem ring_in_range (k G n : Nat) (hG : 0 < G) (hk : k < n * G) : k / G < n := by
  exact (Nat.div_lt_iff_lt_mul hG).2 hk

-- § plan layout

/-- Even-aligned start of the pointer area (`ptr_off`). -/
def ptrOff (cap : Nat) : Nat := (4 + (cap + 1) + cap + cap + 1) / 2 * 2

/-- Total int32 count of one plan slot (`plan_i32_`). -/
def planI32 (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The pointer area start is even (u64 pointers are 8-byte aligned). -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer area starts after `tok` and wastes at most one int32. -/
theorem ptrOff_bounds (cap : Nat) :
    4 + (cap + 1) + cap + cap ≤ ptrOff cap ∧ ptrOff cap ≤ 4 + (cap + 1) + cap + cap + 1 := by
  unfold ptrOff; omega

/-- `start2` ends inside the plan slot. -/
theorem plan_start2_fits (cap : Nat) : ptrOff cap + 4 * cap + (cap + 1) ≤ planI32 cap := by
  unfold planI32; omega

/-- The device-side recomputation of `ptr_off` in `post` agrees with the host layout. -/
theorem ptrOff_device_agree (cap : Nat) : (4 + (cap + 1) + 2 * cap + 1) / 2 * 2 = ptrOff cap := by
  unfold ptrOff; omega

/-- Host plan slots 0 and 1 (each `p` int32) fit in the `2p + 16` allocation. -/
theorem host_plan_fits (p : Nat) : 0 * p + p ≤ 2 * p + 16 ∧ 1 * p + p ≤ 2 * p + 16 := by
  omega

/-- Device plan slots 0 and 1 (stride `p+16`, copy length `p`) fit in `2*(p+16)`. -/
theorem device_plan_fits (p : Nat) :
    0 * (p + 16) + p ≤ 2 * (p + 16) ∧ 1 * (p + 16) + p ≤ 2 * (p + 16) := by
  omega

-- § staging halves

/-- Blobs per group. -/
def perGroup (K G : Nat) : Nat := if G = 2 then K / 2 else K

/-- A split window's two halves fit in the staging area. -/
theorem staging_split_fits (K : Nat) : 2 * perGroup K 2 ≤ K := by
  unfold perGroup; split <;> omega

/-- Group 1's half starts where group 0's ends. -/
theorem staging_adjacent (K : Nat) : 0 * perGroup K 2 + perGroup K 2 = 1 * perGroup K 2 := by
  omega

-- § flag B

/-- Final value of the CAS loop of `raise_flag`. -/
def raise (cur v : Nat) : Nat := if cur < v then v else cur

/-- The flag never goes down. -/
theorem raise_ge_cur (cur v : Nat) : cur ≤ raise cur v := by
  unfold raise; split <;> omega

/-- The flag reaches the requested value. -/
theorem raise_ge_val (cur v : Nat) : v ≤ raise cur v := by
  unfold raise; split <;> omega

/-- `raise` is the maximum, so late host functions of earlier layers cannot lower it. -/
theorem raise_eq_max (cur v : Nat) : raise cur v = max cur v := by
  unfold raise; split <;> omega

/-- `raise` is monotone in the current value. -/
theorem raise_mono (c c' v : Nat) (h : c ≤ c') : raise c v ≤ raise c' v := by
  unfold raise; split <;> split <;> omega

-- § acceptance

/-- The acceptance loop with fuel: extend while the next draft matches. -/
def acceptFrom (T : Nat) (ok : Nat → Bool) : Nat → Nat → Nat
  | a, 0 => a
  | a, f + 1 => if a + 1 < T then (if ok a then acceptFrom T ok (a + 1) f else a) else a

/-- The accepted count never reaches `T`, so `a + 1 <= T`. -/
theorem accept_bound (T : Nat) (ok : Nat → Bool) (fuel : Nat) :
    ∀ a, a + 1 ≤ T → acceptFrom T ok a fuel + 1 ≤ T := by
  induction fuel with
  | zero => intro a h; exact h
  | succ f ih =>
    intro a h
    show (if a + 1 < T then (if ok a then acceptFrom T ok (a + 1) f else a) else a) + 1 ≤ T
    by_cases h1 : a + 1 < T
    · rw [if_pos h1]
      by_cases h2 : ok a = true
      · rw [if_pos h2]; exact ih (a + 1) (by omega)
      · rw [if_neg h2]; exact h
    · rw [if_neg h1]; exact h

/-- The accepted count only grows from its start. -/
theorem accept_ge (T : Nat) (ok : Nat → Bool) (fuel : Nat) :
    ∀ a, a ≤ acceptFrom T ok a fuel := by
  induction fuel with
  | zero => intro a; exact Nat.le_refl a
  | succ f ih =>
    intro a
    show a ≤ (if a + 1 < T then (if ok a then acceptFrom T ok (a + 1) f else a) else a)
    by_cases h1 : a + 1 < T
    · rw [if_pos h1]
      by_cases h2 : ok a = true
      · rw [if_pos h2]; have := ih (a + 1); omega
      · rw [if_neg h2]
    · rw [if_neg h1]

/-- `commit(a+1)` is called with `1 <= a+1 <= T`. -/
theorem commit_range (T a : Nat) (hT : 1 ≤ T) (ha : a + 1 ≤ T) : 1 ≤ a + 1 ∧ a + 1 ≤ T := by
  omega

end Strata.WindowVerifier
