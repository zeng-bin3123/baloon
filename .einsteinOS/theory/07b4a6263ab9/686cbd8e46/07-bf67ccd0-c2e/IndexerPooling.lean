/-!
# MODULE: indexer_pooling
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/native_qsa_indexer.cu
`indexer_key_append` / native `append`: store the raw indexer key into an `r-1` row tail, and when a
cell COMPLETES a block, pool the block's `r` raw keys (mean → rms_norm → weight), rotate the pooled
row at the block's first position, write it to row `b = pos / r`, and move the constant spare key to
row `b + 1`.

## Source semantics
* `slot = pos % r`; slots `< r-1` go to the tail, slot `r-1` completes block `b = pos / r`.
* A completion means `(pos + 1) % r = 0` and the completed-block count becomes `b + 1`.
* The tail never needs more than `r-1` rows; the block's first cell is `b*r = pos - (r-1)`.
* The rotation position is `pos_base + b*r` (block's first cell); native: `0` at `pos = 0`, and with
  `pos_base % 4 = 0` it is 4-aligned.
* Native bounds: writes to pooled row `b + 1` happen only on completion, and then `b + 1 ≤ max/4`
  fits the `max/4 + 1` row buffer even when `max % 4 ≠ 0`.

## Form template identifiers
`idx_block`/`R` → `r`, `slot` → `pos % r`, `tail` rows → `tailRow`, `dead` → spare, `block_pos` →
`blockPos`, `rope_pos` → `nativeRopePos`, `max_cells` → `maxCells`.
-/

namespace Strata.Indexer

-- § helper

/-- Mixed-radix decompositions with digit `< c` are unique. -/
theorem decomp_unique {a b a' b' c : Nat} (hb : b < c) (hb' : b' < c)
    (h : a * c + b = a' * c + b') : a = a' ∧ b = b' := by
  rcases Nat.lt_trichotomy a a' with hlt | heq | hgt
  · have h1 : (a + 1) * c ≤ a' * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h1
    omega
  · subst heq; omega
  · have h1 : (a' + 1) * c ≤ a * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h1
    omega

-- § slots and completion

/-- A cell completes a block iff it is in the last slot. -/
def completes (r pos : Nat) : Prop := pos % r = r - 1

/-- A completing cell makes `n_kv` block-aligned and bumps the completed-block count. -/
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

/-- A non-completing cell leaves the completed-block count unchanged. -/
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

/-- The block's first cell, `b*r`, is `pos - (r-1)` at a completion. -/
theorem block_first_cell (r pos : Nat) (h : completes r pos) : pos / r * r = pos - (r - 1) := by
  unfold completes at h
  have e1 := Nat.div_add_mod pos r
  rw [Nat.mul_comm] at e1
  omega

/-- Tail element offset `slot * idx_dim + d`. -/
def tailRow (idxDim slot d : Nat) : Nat := slot * idxDim + d

/-- The `r - 1` row tail is enough: stored slots never index past it. -/
theorem tailRow_lt (r idxDim slot d : Nat) (hs : slot < r - 1) (hd : d < idxDim) :
    tailRow idxDim slot d < (r - 1) * idxDim := by
  unfold tailRow
  have h1 : (slot + 1) * idxDim ≤ (r - 1) * idxDim := Nat.mul_le_mul_right idxDim (Nat.succ_le_of_lt hs)
  rw [Nat.succ_mul] at h1
  omega

/-- `block_pos`: the rotation position of the completed row (a POSITION, not a cell index). -/
def blockPos (r posBase pos : Nat) : Nat := posBase + pos / r * r

/-- The rotation position never exceeds the completing position (shifted by the base). -/
theorem blockPos_le (r posBase pos : Nat) : blockPos r posBase pos ≤ posBase + pos := by
  unfold blockPos
  have := Nat.div_mul_le_self pos r
  omega

-- § native fixed geometry (R = 4)

/-- Native rotation position: `0` for the spare at `pos = 0`, else `pos_base + 4*b`. -/
def nativeRopePos (posBase pos : Nat) : Nat := if pos = 0 then 0 else posBase + 4 * (pos / 4)

/-- With the host-enforced `pos_base % 4 = 0`, native rotation positions are block-aligned. -/
theorem nativeRopePos_aligned (posBase pos : Nat) (h : posBase % 4 = 0) : nativeRopePos posBase pos % 4 = 0 := by
  unfold nativeRopePos
  split <;> omega

/-- The native kernel writes past its first barrier only at `pos = 0` or at a completion. -/
def nativeWrites (pos : Nat) : Prop := pos = 0 ∨ pos % 4 = 3

/-- Every pooled-row write of the native kernel (rows `b` and `b+1`) fits the `max/4 + 1` buffer. -/
theorem native_pooled_in_bounds (pos maxCells : Nat) (hp : pos < maxCells) (hw : nativeWrites pos) :
    pos / 4 + 1 < maxCells / 4 + 1 + 1 ∧ (pos ≠ 0 → pos / 4 + 1 < maxCells / 4 + 1) := by
  unfold nativeWrites at hw
  constructor
  · omega
  · intro hne
    rcases hw with h | h
    · exact absurd h hne
    · omega

/-- Without the completion guard a tail position could address one row past the buffer. -/
theorem native_guard_needed : ¬ (∀ pos maxCells : Nat, pos < maxCells → pos / 4 + 1 < maxCells / 4 + 1) := by
  intro h
  have := h 4 5 (by decide)
  omega

end Strata.Indexer
