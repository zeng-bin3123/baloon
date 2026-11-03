/-!
# MODULE: expert_cache_residency
VRAM expert-cache slot storage and residency table (`ExpertCache`): per-layer slot ranges, no-eviction
admission, slot addressing, 256-byte aligned variable slots, and the fill-size clamp.

## Source semantics
* `layer_slot_range` gives layer `l` the half-open range `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last
  layer takes the remainder, so the ranges tile `0..slots` exactly.
* `admit` never evicts: a pair gets the next free slot of its range (per-layer) or of the global counter, or is refused.
* `device_slot(slot)` is only defined for `0 ≤ slot < slots`, and the slot (plus its blob) lies inside the arena.
* `open_sized` aligns every slot to 256 bytes; `fill_slot` clamps the copy size into `(0, blob]`.

## Form template identifiers
`slots_` -> nSlots, `n_layers_` -> nl, `n_expert_` -> ne, `blob_` -> blob, `layer_next_[l]` -> next,
`kNotResident` -> notResident, `layer_slot_range` -> slotLo / slotHi.
-/

namespace Strata.ExpertCache

-- § alignment of variable-size slots

/-- Slot sizes are rounded up to a multiple of 256 bytes. -/
def align256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- Aligned size is at least the requested size. -/
theorem align256_ge (b : Nat) : b ≤ align256 b := by unfold align256; omega

/-- Aligned size is a multiple of 256. -/
theorem align256_mod (b : Nat) : align256 b % 256 = 0 := by unfold align256; omega

/-- Alignment wastes fewer than 256 bytes. -/
theorem align256_lt (b : Nat) : align256 b < b + 256 := by unfold align256; omega

/-- Sum of aligned slot sizes (the single allocation of `open_sized`). -/
def total : List Nat → Nat
  | [] => 0
  | b :: bs => align256 b + total bs

/-- Offset of slot `i`: the sum of the aligned sizes before it. -/
def off : List Nat → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | b :: bs, i + 1 => align256 b + off bs i

/-- Every slot offset is 256-byte aligned. -/
theorem off_aligned : ∀ (l : List Nat) (i : Nat), off l i % 256 = 0
  | _, 0 => by simp [off]
  | [], _ + 1 => by simp [off]
  | b :: bs, i + 1 => by
      have h := off_aligned bs i
      have h2 := align256_mod b
      simp only [off]
      omega

/-- Slot offsets never exceed the arena size. -/
theorem off_le_total : ∀ (l : List Nat) (i : Nat), off l i ≤ total l
  | _, 0 => by simp [off]
  | [], _ + 1 => by simp [off, total]
  | b :: bs, i + 1 => by
      have h := off_le_total bs i
      simp only [off, total]
      omega

/-- Slot `i+1` starts exactly where slot `i`'s aligned extent ends. -/
theorem off_succ : ∀ (l : List Nat) (i : Nat), i < l.length →
    off l (i + 1) = off l i + align256 (l.getD i 0)
  | [], i, h => by simp at h
  | b :: bs, 0, _ => by simp [off, List.getD_cons_zero]
  | b :: bs, i + 1, h => by
      have h' : i < bs.length := by simpa using h
      have := off_succ bs i h'
      simp only [off, List.getD_cons_succ]
      omega

-- § layer ranges

/-- Lower end of layer `l`'s slot range. -/
def slotLo (nl s l : Nat) : Nat :=
  if nl = 0 ∨ s = 0 ∨ nl ≤ l then 0 else l * (s / nl)

/-- Upper end (exclusive) of layer `l`'s slot range; the last layer takes the remainder. -/
def slotHi (nl s l : Nat) : Nat :=
  if nl = 0 ∨ s = 0 ∨ nl ≤ l then 0 else if l = nl - 1 then s else (l + 1) * (s / nl)

/-- An out-of-range layer owns the empty range. -/
theorem slotRange_oob (nl s l : Nat) (h : nl ≤ l) : slotLo nl s l = 0 ∧ slotHi nl s l = 0 := by
  unfold slotLo slotHi; simp [h]

/-- The first layer starts at slot 0. -/
theorem slotLo_first (nl s : Nat) (h : 0 < nl) (hs : 0 < s) : slotLo nl s 0 = 0 := by
  have c : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ 0) := by omega
  unfold slotLo; rw [if_neg c]; simp

/-- No range extends past the slot count. -/
theorem slotHi_le (nl s l : Nat) (h : l < nl) (hs : 0 < s) : slotHi nl s l ≤ s := by
  have c : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ l) := by omega
  unfold slotHi
  rw [if_neg c]
  by_cases hl : l = nl - 1
  · simp [hl]
  · rw [if_neg hl]
    have h1 : (l + 1) * (s / nl) ≤ nl * (s / nl) := Nat.mul_le_mul_right _ (by omega)
    have h2 : nl * (s / nl) ≤ s := Nat.mul_div_le s nl
    omega

/-- The last layer's range ends exactly at the slot count: no slot is orphaned by the division. -/
theorem slotHi_last (nl s : Nat) (h : 0 < nl) (hs : 0 < s) : slotHi nl s (nl - 1) = s := by
  have c : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ nl - 1) := by omega
  unfold slotHi; simp [c]

/-- Adjacent layers' ranges are contiguous, so the ranges tile `0..slots`. -/
theorem slot_contiguous (nl s l : Nat) (h : l + 1 < nl) (hs : 0 < s) :
    slotHi nl s l = slotLo nl s (l + 1) := by
  have c1 : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ l) := by omega
  have c2 : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ l + 1) := by omega
  have c3 : ¬ (l = nl - 1) := by omega
  unfold slotHi slotLo
  rw [if_neg c1, if_neg c3, if_neg c2]

/-- Each range is well formed (`lo ≤ hi`). -/
theorem slotLo_le_hi (nl s l : Nat) (h : l < nl) (hs : 0 < s) : slotLo nl s l ≤ slotHi nl s l := by
  have c : ¬ (nl = 0 ∨ s = 0 ∨ nl ≤ l) := by omega
  unfold slotLo slotHi
  rw [if_neg c, if_neg c]
  by_cases hl : l = nl - 1
  · rw [if_pos hl]
    have h1 : l * (s / nl) ≤ nl * (s / nl) := Nat.mul_le_mul_right _ (by omega)
    have h2 : nl * (s / nl) ≤ s := Nat.mul_div_le s nl
    omega
  · rw [if_neg hl]
    exact Nat.mul_le_mul_right _ (Nat.le_succ l)

-- § admission (no eviction)

/-- Per-layer admission: take `next` if the layer's quota is not full. -/
def admitStep (next hi : Nat) : Option (Nat × Nat) :=
  if hi ≤ next then none else some (next, next + 1)

/-- An admitted slot is below the ceiling, and the counter advances by one. -/
theorem admitStep_sound (next hi slot next' : Nat) (h : admitStep next hi = some (slot, next')) :
    slot < hi ∧ next' = slot + 1 ∧ slot = next := by
  unfold admitStep at h
  by_cases c : hi ≤ next
  · simp [c] at h
  · simp [c] at h
    omega

/-- A full quota refuses (no eviction). -/
theorem admitStep_full (next hi : Nat) (h : hi ≤ next) : admitStep next hi = none := by
  unfold admitStep; simp [h]

/-- The counter never passes the ceiling when it started within it. -/
theorem admitStep_bounded (next hi slot next' : Nat) (hn : next ≤ hi)
    (h : admitStep next hi = some (slot, next')) : next' ≤ hi := by
  have := admitStep_sound next hi slot next' h
  omega

-- § addressing

/-- The residency table index `layer*ne + expert` is in range for valid coordinates. -/
theorem residency_idx_lt (nl ne layer expert : Nat) (hl : layer < nl) (he : expert < ne) :
    layer * ne + expert < nl * ne := by
  have h := Nat.mul_le_mul_right ne (show layer + 1 ≤ nl by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- A valid slot plus its blob lies inside the arena. -/
theorem device_slot_in_arena (slots blob slot : Nat) (h : slot < slots) :
    slot * blob + blob ≤ slots * blob := by
  have h1 := Nat.mul_le_mul_right blob (show slot + 1 ≤ slots by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- A positive slot count and blob size give a positive allocation. -/
theorem want_pos (n blob : Nat) (hn : 0 < n) (hb : 0 < blob) : 0 < n * blob := Nat.mul_pos hn hb

/-- The VRAM check: a request larger than the free figure is refused before allocation. -/
def vramOk (free want : Nat) : Bool := !(decide (free < want))

/-- If the check passes, the request fits in free VRAM. -/
theorem vramOk_fits (free want : Nat) (h : vramOk free want = true) : want ≤ free := by
  unfold vramOk at h
  simp at h
  omega

/-- The copy size of `fill_slot`: the request if it is in `(0, blob]`, otherwise the whole blob. -/
def fillBytes (bytes blob : Int) : Int := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- The copy size is always positive and never exceeds the slot's blob size. -/
theorem fillBytes_bounds (bytes blob : Int) (hb : 0 < blob) :
    0 < fillBytes bytes blob ∧ fillBytes bytes blob ≤ blob := by
  unfold fillBytes
  split <;> omega

end Strata.ExpertCache
