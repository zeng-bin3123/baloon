-- IndexerPooling.lean
-- Formal model of src/kernels/cuda/qsa.cu and native_qsa_indexer.cu indexer key pooling.
namespace Baloon.Indexer

theorem decomp_unique {a b a' b' c : Nat} (hb : b < c) (hb' : b' < c)
    (h : a * c + b = a' * c + b') : a = a' ∧ b = b' := by
  rcases Nat.lt_trichotomy a a' with hlt | heq | hgt
  · have h1 : (a + 1) * c ≤ a' * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h1; omega
  · subst heq; omega
  · have h1 : (a' + 1) * c ≤ a * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h1; omega

def completes (r pos : Nat) : Prop := pos % r = r - 1

theorem completes_next (r pos : Nat) (hr : 0 < r) (h : completes r pos) :
    (pos + 1) % r = 0 ∧ (pos + 1) / r = pos / r + 1 := by
  unfold completes at h
  have e1 := Nat.div_add_mod pos r
  have e2 := Nat.div_add_mod (pos + 1) r
  have m2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.mul_comm] at e1 e2
  have key : (pos / r + 1) * r + 0 = (pos + 1) / r * r + (pos + 1) % r := by
    rw [Nat.succ_mul]; omega
  obtain ⟨h1, h2⟩ := decomp_unique hr m2 key
  exact ⟨h2.symm, h1.symm⟩

theorem noncomplete_same_block (r pos : Nat) (hr : 0 < r) (h : pos % r < r - 1) :
    (pos + 1) / r = pos / r ∧ (pos + 1) % r = pos % r + 1 := by
  have e1 := Nat.div_add_mod pos r
  have e2 := Nat.div_add_mod (pos + 1) r
  have m2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.mul_comm] at e1 e2
  have hlt : pos % r + 1 < r := by omega
  have key : pos / r * r + (pos % r + 1) = (pos + 1) / r * r + (pos + 1) % r := by omega
  obtain ⟨h1, h2⟩ := decomp_unique hlt m2 key
  exact ⟨h1.symm, h2.symm⟩

theorem block_first_cell (r pos : Nat) (h : completes r pos) : pos / r * r = pos - (r - 1) := by
  unfold completes at h
  have e1 := Nat.div_add_mod pos r; rw [Nat.mul_comm] at e1; omega

def tailRow (idxDim slot d : Nat) : Nat := slot * idxDim + d

theorem tailRow_lt (r idxDim slot d : Nat) (hs : slot < r - 1) (hd : d < idxDim) :
    tailRow idxDim slot d < (r - 1) * idxDim := by
  unfold tailRow
  have h1 : (slot + 1) * idxDim ≤ (r - 1) * idxDim := Nat.mul_le_mul_right idxDim (Nat.succ_le_of_lt hs)
  rw [Nat.succ_mul] at h1; omega

def blockPos (r posBase pos : Nat) : Nat := posBase + pos / r * r

theorem blockPos_le (r posBase pos : Nat) : blockPos r posBase pos ≤ posBase + pos := by
  unfold blockPos; have := Nat.div_mul_le_self pos r; omega

def nativeRopePos (posBase pos : Nat) : Nat := if pos = 0 then 0 else posBase + 4 * (pos / 4)

theorem nativeRopePos_aligned (posBase pos : Nat) (h : posBase % 4 = 0) :
    nativeRopePos posBase pos % 4 = 0 := by
  unfold nativeRopePos; split <;> omega

def nativeWrites (pos : Nat) : Prop := pos = 0 ∨ pos % 4 = 3

theorem native_pooled_in_bounds (pos maxCells : Nat) (hp : pos < maxCells) (hw : nativeWrites pos) :
    pos / 4 + 1 < maxCells / 4 + 1 + 1 ∧ (pos ≠ 0 → pos / 4 + 1 < maxCells / 4 + 1) := by
  unfold nativeWrites at hw; constructor; · omega
  · intro hne; rcases hw with h | h; · exact absurd h hne; · omega

theorem native_guard_needed : ¬ (∀ pos maxCells : Nat, pos < maxCells → pos / 4 + 1 < maxCells / 4 + 1) := by
  intro h; have := h 4 5 (by decide); omega

end Baloon.Indexer
