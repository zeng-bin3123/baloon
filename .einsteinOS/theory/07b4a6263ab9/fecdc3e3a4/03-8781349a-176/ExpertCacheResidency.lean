/-!
# MODULE: expert_cache_residency
SOURCES: src/core/expert_cache.cpp
The VRAM expert tier's residency table: compulsory-miss admission with no eviction, either from one
global counter or from per-layer slot ranges (R4.2g), plus slot addressing (uniform or 256-aligned
variable-size slots), fill-size clamping and the `STRP` profile reader's validation.

## Source semantics
* `layer_slot_range(l)`: `q = slots / n_layers`, `lo = l*q`, `hi = (l+1)*q` except the LAST layer,
  which takes everything up to `slots`; the ranges tile `[0, slots)` exactly.
* `admit` returns the existing slot when already resident (idempotent); otherwise the next free slot
  if one remains, else `kNotResident`.  There is no eviction: a resident slot never changes, and no
  two `(layer, expert)` pairs ever share a slot.
* `slot_of` / `admit` refuse out-of-range axes.
* `fill_slot` copies `bytes` if `0 < bytes ≤ blob`, else the whole `blob`.
* `open_sized` rounds every slot up to 256 bytes; offsets are the running sum.
* `read_expert_profile` refuses `n_ranked > slots` and any pair out of range.

## Form template identifiers
`ExpertCache` → `Cache`; `residency_` → `res`; `next_free_` → `next`; `slots_` → `slots`;
`admit` → `admit`; `layer_slot_range` → `lo`/`hi`; `fill_slot` byte count → `fillBytes`;
`open_sized` offsets → `slotOffsets`; `read_expert_profile` → `profileOk`.
-/

namespace Strata.ExpertCacheResidency

-- § per-layer slot ranges

/-- Lower end of layer `l`'s range. -/
def lo (slots nL l : Nat) : Nat := l * (slots / nL)

/-- Upper end of layer `l`'s range; the last layer takes the remainder. -/
def hi (slots nL l : Nat) : Nat := if l = nL - 1 then slots else (l + 1) * (slots / nL)

/-- The first layer starts at slot 0. -/
theorem lo_zero (s nL : Nat) : lo s nL 0 = 0 := by simp [lo]

/-- The last layer ends at `slots`: no slot is orphaned by the division. -/
theorem hi_last (s nL : Nat) : hi s nL (nL - 1) = s := by simp [hi]

/-- Consecutive ranges are contiguous: layer `l` ends where layer `l+1` begins. -/
theorem hi_eq_lo_succ (s nL l : Nat) (h : l + 1 < nL) : hi s nL l = lo s nL (l + 1) := by
  have : l ≠ nL - 1 := by omega
  simp [hi, lo, this]

/-- Every range start lies inside the arena. -/
theorem lo_le_slots (s nL l : Nat) (h : l < nL) : lo s nL l ≤ s := by
  unfold lo
  have h1 : l * (s / nL) ≤ nL * (s / nL) := Nat.mul_le_mul_right _ (Nat.le_of_lt h)
  have h2 := Nat.mul_div_le s nL
  omega

/-- Every range is well-formed (`lo ≤ hi`). -/
theorem lo_le_hi (s nL l : Nat) (h : l < nL) : lo s nL l ≤ hi s nL l := by
  unfold hi
  split
  · exact lo_le_slots s nL l h
  · unfold lo; exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- Every range end lies inside the arena. -/
theorem hi_le_slots (s nL l : Nat) (h : l < nL) : hi s nL l ≤ s := by
  unfold hi
  split
  · exact Nat.le_refl _
  · have hl : l + 1 < nL := by omega
    have := lo_le_slots s nL (l + 1) hl
    unfold lo at this; exact this

-- § admission (global counter)

/-- Residency table (flat key `layer * n_expert + expert` ↦ slot) and the global free pointer. -/
structure Cache where
  res : Nat → Option Nat
  next : Nat
  slots : Nat

/-- `ExpertCache::admit` on the global path: existing slot, next free slot, or `none` when full. -/
def admit (c : Cache) (key : Nat) : Option Nat × Cache :=
  match c.res key with
  | some s => (some s, c)
  | none =>
    if c.next < c.slots then
      (some c.next, { c with res := fun x => if x = key then some c.next else c.res x, next := c.next + 1 })
    else (none, c)

/-- The cache invariant: assigned slots are below `next ≤ slots`, and the table is injective. -/
def Inv (c : Cache) : Prop :=
  c.next ≤ c.slots ∧ (∀ x s, c.res x = some s → s < c.next) ∧
    (∀ x y s, c.res x = some s → c.res y = some s → x = y)

/-- An empty cache satisfies the invariant. -/
theorem inv_empty (slots : Nat) : Inv ⟨fun _ => none, 0, slots⟩ := by
  refine ⟨Nat.zero_le _, ?_, ?_⟩ <;> intros <;> contradiction

/-- `admit` preserves the invariant: no slot is ever shared, none exceeds the arena. -/
theorem admit_inv (c : Cache) (key : Nat) (h : Inv c) : Inv (admit c key).2 := by
  obtain ⟨hn, hlt, hinj⟩ := h
  unfold admit
  split
  · exact ⟨hn, hlt, hinj⟩
  · rename_i hk
    split
    · rename_i hfree
      refine ⟨by simp; omega, ?_, ?_⟩
      · intro x s hx
        simp only at hx ⊢
        split at hx
        · simp at hx; omega
        · have := hlt x s hx; omega
      · intro x y s hx hy
        simp only at hx hy
        split at hx <;> split at hy
        · rename_i a b; rw [a, b]
        · rename_i a b; simp at hx; subst hx; have := hlt y _ hy; omega
        · rename_i a b; simp at hy; subst hy; have := hlt x _ hx; omega
        · exact hinj x y s hx hy
    · exact ⟨hn, hlt, hinj⟩

/-- A returned slot is always inside the arena. -/
theorem admit_lt_slots (c : Cache) (key s : Nat) (h : Inv c) (ha : (admit c key).1 = some s) :
    s < c.slots := by
  obtain ⟨hn, hlt, _⟩ := h
  unfold admit at ha
  split at ha
  · rename_i s' hs; simp at ha; subst ha; have := hlt key s' hs; omega
  · split at ha
    · simp at ha; omega
    · contradiction

/-- Admission is idempotent: admitting a resident key returns its slot and changes nothing. -/
theorem admit_resident (c : Cache) (key s : Nat) (h : c.res key = some s) :
    admit c key = (some s, c) := by
  simp [admit, h]

/-- No eviction: a resident key keeps its slot after any admission. -/
theorem admit_keeps (c : Cache) (key x s : Nat) (h : c.res x = some s) :
    (admit c key).2.res x = some s := by
  unfold admit
  split
  · exact h
  · rename_i hk
    split
    · simp only
      split
      · rename_i hx; subst hx; rw [hk] at h; contradiction
      · exact h
    · exact h

/-- A full cache refuses every non-resident key and stays unchanged. -/
theorem admit_full (c : Cache) (key : Nat) (hk : c.res key = none) (hf : c.slots ≤ c.next) :
    admit c key = (none, c) := by
  simp [admit, hk, Nat.not_lt.mpr hf]

-- § per-layer admission

/-- R4.2g: admission bounded by the layer's own range; returns the slot and the new pointer. -/
def admitLayer (ptr hiL : Nat) : Option Nat × Nat :=
  if ptr < hiL then (some ptr, ptr + 1) else (none, ptr)

/-- A per-layer admission stays inside that layer's range `[lo, hi)`. -/
theorem admitLayer_in_range (ptr loL hiL s : Nat) (hp : loL ≤ ptr)
    (h : (admitLayer ptr hiL).1 = some s) : loL ≤ s ∧ s < hiL := by
  unfold admitLayer at h
  split at h
  · simp at h; omega
  · contradiction

/-- The layer pointer never passes the layer's upper end once it starts below it. -/
theorem admitLayer_ptr_le (ptr hiL : Nat) (h : ptr ≤ hiL) : (admitLayer ptr hiL).2 ≤ hiL := by
  unfold admitLayer; split <;> simp <;> omega

-- § slot addressing and fills

/-- `fill_slot`'s byte count: the request if it is in `(0, blob]`, else the whole blob. -/
def fillBytes (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never copies more than one slot. -/
theorem fillBytes_le (b blob : Nat) : fillBytes b blob ≤ blob := by
  unfold fillBytes; split <;> omega

/-- Round up to the 256-byte alignment `open_sized` uses. -/
def alignUp (x : Nat) : Nat := (x + 255) / 256 * 256

/-- An aligned slot holds its blob. -/
theorem le_alignUp (x : Nat) : x ≤ alignUp x := by unfold alignUp; omega

/-- Alignment wastes less than 256 bytes per slot. -/
theorem alignUp_lt (x : Nat) : alignUp x < x + 256 := by unfold alignUp; omega

/-- Aligned sizes are multiples of 256. -/
theorem alignUp_mod (x : Nat) : alignUp x % 256 = 0 := by unfold alignUp; omega

/-- Running offsets of `open_sized`: `off[0] = start`, `off[i+1] = off[i] + alignUp(bytes[i])`. -/
def slotOffsets (start : Nat) : List Nat → List Nat
  | [] => [start]
  | b :: bs => start :: slotOffsets (start + alignUp b) bs

/-- Every slot offset is 256-aligned when the arena start is. -/
theorem slotOffsets_aligned (bs : List Nat) : ∀ start, start % 256 = 0 →
    ∀ o ∈ slotOffsets start bs, o % 256 = 0 := by
  induction bs with
  | nil => intro start hs o ho; simp [slotOffsets] at ho; omega
  | cons b bs ih =>
    intro start hs o ho
    simp only [slotOffsets, List.mem_cons] at ho
    rcases ho with rfl | ho
    · exact hs
    · have := alignUp_mod b
      exact ih (start + alignUp b) (by omega) o ho

/-- The offset list has one more entry than there are slots (the end of the arena). -/
theorem slotOffsets_length (bs : List Nat) : ∀ start, (slotOffsets start bs).length = bs.length + 1 := by
  induction bs with
  | nil => intro; rfl
  | cons b bs ih => intro start; simp [slotOffsets, ih]

-- § profile validation

/-- `read_expert_profile`'s accept predicate on the header and ranked pairs. -/
def profileOk (nL nE want : Nat) (ranked : List (Nat × Nat)) : Bool :=
  decide (ranked.length ≤ want) && ranked.all (fun p => decide (p.1 < nL) && decide (p.2 < nE))

/-- An accepted profile never ranks more pairs than it has slots. -/
theorem profileOk_len (nL nE w : Nat) (r : List (Nat × Nat)) (h : profileOk nL nE w r = true) :
    r.length ≤ w := by
  simp only [profileOk, Bool.and_eq_true, decide_eq_true_eq] at h; exact h.1

/-- Every pair of an accepted profile is a valid `(layer, expert)`. -/
theorem profileOk_range (nL nE w : Nat) (r : List (Nat × Nat)) (h : profileOk nL nE w r = true) :
    ∀ p ∈ r, p.1 < nL ∧ p.2 < nE := by
  simp only [profileOk, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  intro p hp
  have := h.2 p hp
  simpa using this

end Strata.ExpertCacheResidency
