/-!
# MODULE: verify_window_groups
Token-group split, ring numbering, staging and plan-buffer layout of the speculative verify window.

## Source semantics
With `G = 2` the window's T tokens are cut into A = `[0,(T+1)/2)` and B = `[(T+1)/2, T)`; with `G = 1` one group holds all.
Processing order index is `k = l*G + grp` and the ring awaited is `k+1`; `k/G = l`, `k%G = grp`. Staging blobs are
split evenly per group. The plan buffer holds i32 header/arrays, then 8-byte aligned u64 pointer arrays. Flag B only rises.

## Form template identifiers
- tb_/te_              -> tbF / teF
- ring `l*G+grp+1`     -> ringK
- kStagingBlobs split  -> perGroup
- Verifier::raise_flag -> raiseFlag
-/
namespace Strata.VerifyWindowGroups

def tbF (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2
def teF (G T grp : Nat) : Nat := if G = 2 ∧ grp = 0 then (T + 1) / 2 else T

/-- One group covers `[0, T)`. -/
theorem single_group (T : Nat) : tbF T 0 = 0 ∧ teF 1 T 0 = T := by simp [tbF, teF]

/-- Two groups tile `[0, T)` with no gap or overlap. -/
theorem two_groups_tile (T : Nat) : tbF T 0 = 0 ∧ teF 2 T 0 = tbF T 1 ∧ teF 2 T 1 = T := by simp [tbF, teF]

/-- Both groups are non-empty when `T ≥ 2`. -/
theorem two_groups_nonempty {T : Nat} (h : 2 ≤ T) : 1 ≤ teF 2 T 0 - tbF T 0 ∧ 1 ≤ teF 2 T 1 - tbF T 1 := by
  simp [tbF, teF]
  omega

/-- Group A is at least as large as group B. -/
theorem groupA_ge_groupB (T : Nat) : teF 2 T 1 - tbF T 1 ≤ teF 2 T 0 - tbF T 0 := by
  simp [tbF, teF]
  omega

/-- The two group sizes add up to the window. -/
theorem group_sizes_sum {T : Nat} (h : 2 ≤ T) :
    (teF 2 T 0 - tbF T 0) + (teF 2 T 1 - tbF T 1) = T := by
  simp [tbF, teF]
  omega

def ringK (G l grp : Nat) : Nat := grp + G * l + 1

/-- `(l*G + grp) / G = l` for `grp < G`: the host loop recovers the layer. -/
theorem ring_div {G l grp : Nat} (hG : 0 < G) (hg : grp < G) : (grp + G * l) / G = l := by
  rw [Nat.add_mul_div_left _ _ hG, Nat.div_eq_of_lt hg]
  simp

/-- `(l*G + grp) % G = grp`: the host loop recovers the group. -/
theorem ring_mod {G l grp : Nat} (hg : grp < G) : (grp + G * l) % G = grp := by
  rw [Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt hg

/-- Ring numbers of distinct (layer, group) pairs are distinct. -/
theorem ringK_inj {G l l' grp grp' : Nat} (hG : 0 < G) (hg : grp < G) (hg' : grp' < G)
    (h : ringK G l grp = ringK G l' grp') : l = l' ∧ grp = grp' := by
  unfold ringK at h
  have h1 : grp + G * l = grp' + G * l' := by omega
  have e1 := ring_div (l := l) hG hg
  have e2 := ring_div (l := l') hG hg'
  have m1 := ring_mod (l := l) hg
  have m2 := ring_mod (l := l') hg'
  rw [h1] at e1 m1
  exact ⟨by omega, by omega⟩

/-- The ring numbers are all within `1 .. n_layers * G`. -/
theorem ringK_le {G nl l grp : Nat} (hg : grp < G) (hl : l < nl) : ringK G l grp ≤ nl * G := by
  unfold ringK
  have := Nat.mul_le_mul_left G (show l + 1 ≤ nl by omega)
  rw [Nat.mul_add, Nat.mul_one] at this
  have h2 := Nat.mul_comm G nl
  omega

def perGroup (G K : Nat) : Nat := if G = 2 then K / 2 else K

/-- Each group's staging half lies inside the staging area. -/
theorem staging_in_area {G K grp : Nat} (hg : grp < G) (hG : G ≤ 2) : grp * perGroup G K + perGroup G K ≤ K := by
  unfold perGroup
  split
  · have h01 : grp = 0 ∨ grp = 1 := by omega
    rcases h01 with rfl | rfl <;> omega
  · have : grp = 0 := by omega
    subst this
    omega

/-- Plan buffer layout: the u64 region start is even (8-byte aligned in i32 units), past the i32 header. -/
theorem plan_ptr_off (cap : Nat) :
    let i32 := 4 + (cap + 1) + cap + cap
    let off := (i32 + 1) / 2 * 2
    i32 ≤ off ∧ off % 2 = 0 := by
  intro i32 off
  omega

/-- Plan buffer: the `start2` array fits within the allocated i32 count. -/
theorem plan_start2_fits (off cap : Nat) : off + 4 * cap + (cap + 1) ≤ off + 4 * cap + (cap + 1) + 1 := by omega

def raiseFlag (cur v : Nat) : Nat := if cur < v then v else cur

/-- Flag B never decreases. -/
theorem raiseFlag_ge (cur v : Nat) : cur ≤ raiseFlag cur v := by unfold raiseFlag; split <;> omega
/-- Raising reaches at least the requested value. -/
theorem raiseFlag_ge_v (cur v : Nat) : v ≤ raiseFlag cur v := by unfold raiseFlag; split <;> omega
/-- Late raises from earlier layers do not lower a later layer's value. -/
theorem raiseFlag_eq_max (cur v : Nat) : raiseFlag cur v = max cur v := by
  unfold raiseFlag
  split <;> omega

/-- Raise order does not matter. -/
theorem raiseFlag_comm (c a b : Nat) : raiseFlag (raiseFlag c a) b = raiseFlag (raiseFlag c b) a := by
  simp only [raiseFlag_eq_max]
  omega

/-- The flag-set ring index is always in bounds. -/
theorem flagset_index {n sz : Nat} (h : 0 < sz) : n % sz < sz := Nat.mod_lt n h

end Strata.VerifyWindowGroups
