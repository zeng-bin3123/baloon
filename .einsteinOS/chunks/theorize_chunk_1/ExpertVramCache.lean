/-!
# MODULE: expert_vram_cache
VRAM-resident expert tier: slot arena, residency table, per-layer slot quotas, sized slot offsets, and the
bounded copy size of `fill_slot`.

## Source semantics
`admit` never evicts: a resident (layer, expert) keeps its slot forever; a free slot is handed out from one
counter (global) or from a per-layer counter bounded by the layer's own range `[l*q, (l+1)*q)` with `q = slots/n_layers`
(the last layer takes the remainder). Ranges tile `0..slots` exactly. Sized slots are 256-byte aligned.

## Form template identifiers
- ExpertCache::layer_slot_range -> layerLo / layerHi
- ExpertCache::admit            -> admitSlot / admitNext / admitRes
- ExpertCache::open_sized       -> align256 / offsetsFrom
- ExpertCache::fill_slot        -> fillBytes
-/
namespace Strata.ExpertVramCache

-- § per-layer slot ranges
def layerLo (slots nl l : Nat) : Nat := l * (slots / nl)
def layerHi (slots nl l : Nat) : Nat := if l = nl - 1 then slots else (l + 1) * (slots / nl)

/-- Layer 0's range starts at slot 0. -/
theorem layerLo_zero (slots nl : Nat) : layerLo slots nl 0 = 0 := by simp [layerLo]

/-- `n_layers` quotas of size `q` never exceed the slot count. -/
theorem layer_quota_le (slots nl : Nat) : nl * (slots / nl) ≤ slots := Nat.mul_div_le slots nl

/-- Every layer's range is well formed (lo ≤ hi). -/
theorem layerLo_le_layerHi {slots nl l : Nat} (h : l < nl) : layerLo slots nl l ≤ layerHi slots nl l := by
  unfold layerLo layerHi
  have hq := layer_quota_le slots nl
  split
  · have h1 : l * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    omega
  · exact Nat.mul_le_mul_right _ (by omega)

/-- Consecutive ranges abut: no slot is orphaned between layers. -/
theorem layerHi_eq_layerLo_succ {slots nl l : Nat} (h : l + 1 < nl) :
    layerHi slots nl l = layerLo slots nl (l + 1) := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]

/-- The last layer's range ends exactly at `slots`, so the ranges cover everything. -/
theorem layerHi_last {slots nl : Nat} (h : 0 < nl) : layerHi slots nl (nl - 1) = slots := by
  unfold layerHi
  simp

/-- Ranges of distinct layers are disjoint. -/
theorem layer_ranges_disjoint {slots nl l l' : Nat} (h : l < l') (h' : l' < nl) :
    layerHi slots nl l ≤ layerLo slots nl l' := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]
  exact Nat.mul_le_mul_right _ (by omega)

-- § admission without eviction
def admitSlot (next hi : Nat) : Option Nat := if next ≥ hi then none else some next
def admitNext (next hi : Nat) : Nat := if next ≥ hi then next else next + 1

/-- A granted slot lies below the ceiling. -/
theorem admitSlot_lt {next hi s : Nat} (h : admitSlot next hi = some s) : s < hi ∧ s = next := by
  unfold admitSlot at h
  split at h
  · simp at h
  · rename_i hn
    simp only [Option.some.injEq] at h
    omega

/-- A full counter refuses and does not move. -/
theorem admit_full {next hi : Nat} (h : next ≥ hi) : admitSlot next hi = none ∧ admitNext next hi = next := by
  unfold admitSlot admitNext
  simp [h]

/-- The counter only grows and never passes its ceiling. -/
theorem admitNext_mono (next hi : Nat) : next ≤ admitNext next hi ∧ (next ≤ hi → admitNext next hi ≤ hi) := by
  unfold admitNext
  split <;> omega

/-- Residency update: a key that is already resident is left untouched (no eviction). -/
def admitRes (res : Nat → Option Nat) (key : Nat) (s : Option Nat) : Nat → Option Nat :=
  fun k => if k = key ∧ res k = none then s else res k

/-- Nothing resident is ever displaced by an admission. -/
theorem admitRes_preserves {res : Nat → Option Nat} {key k v : Nat} {s : Option Nat}
    (hk : res k = some v) : admitRes res key s k = some v := by
  unfold admitRes
  split
  · rename_i hh
    rw [hh.2] at hk
    simp at hk
  · exact hk

-- § 256-byte aligned sized slots
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment never shrinks a slot. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega
/-- Aligned sizes are multiples of 256. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega
/-- Alignment wastes fewer than 256 bytes. -/
theorem align256_lt (n : Nat) : align256 n < n + 256 := by unfold align256; omega

def offsetsFrom (start : Nat) : List Nat → Nat
  | [] => start
  | b :: bs => offsetsFrom (start + align256 b) bs

/-- The arena is at least as big as the sum of the blobs. -/
theorem offsets_total_ge : ∀ (bs : List Nat) (start : Nat), start + bs.sum ≤ offsetsFrom start bs
  | [], start => by simp [offsetsFrom]
  | b :: bs, start => by
    have h := offsets_total_ge bs (start + align256 b)
    have := align256_ge b
    simp only [offsetsFrom, List.sum_cons]
    omega

/-- Every slot starts on a 256-byte boundary. -/
theorem offsets_total_mod : ∀ (bs : List Nat) (start : Nat), start % 256 = 0 → offsetsFrom start bs % 256 = 0
  | [], start, h => by simpa [offsetsFrom] using h
  | b :: bs, start, h => by
    show offsetsFrom (start + align256 b) bs % 256 = 0
    apply offsets_total_mod bs
    have := align256_mod b
    omega

-- § bounded fill and in-arena slot address
def fillBytes (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never copies more than one slot. -/
theorem fillBytes_le (bytes blob : Nat) : fillBytes bytes blob ≤ blob := by
  unfold fillBytes
  split <;> omega

/-- A uniform slot lies wholly inside the arena. -/
theorem uniform_slot_in_arena {slot slots blob : Nat} (h : slot < slots) : slot * blob + blob ≤ slots * blob := by
  have := Nat.mul_le_mul_right blob (show slot + 1 ≤ slots by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

end Strata.ExpertVramCache
