/-!
# MODULE: rope_neox_pairing
SOURCES: src/kernels/cuda/rope.cu, src/kernels/cuda/native_rope.cu
Partial NEOX rotary embedding: pair `(i, i + n_rot/2)` for `i < n_rot/2`, rotate by the angle of the
row's position, copy the untouched tail `[n_rot, head_dim)`; `build_rope_table` caches `cos/sin` per
`(position, pair)` at offset `p * half + i`.

## Source semantics
* NEOX pairing, not adjacent `(2i, 2i+1)`: every dimension `< head_dim` is touched by exactly one role —
  first of a pair (`d < half`), second of a pair (`half ≤ d < n_rot`), or tail copy.
* `out = (a·c − b·s, a·s + b·c)`; with `c² + s² = 1` it preserves `a² + b²` (exactly, over a ring);
  position 0 (`c = 1, s = 0`) is the identity, which is why the spare key needs no rotation.
* Table reads `p * half + i` with `p < max_pos`, `i < half` stay inside `max_pos * half`.
* `native_rope`'s pair thread `pair < width/2` either rotates (`pair < 32`, writes `pair`, `pair+32`) or
  copies (`2·pair`, `2·pair+1 ≥ 64`): the write sets are disjoint and cover `[0, width)`.

## Form template identifiers
`rope_neox_pair` → `rotPair`, `half = n_rot / 2` → `half`, `toff + i` → `tabIdx`, role of a dimension
→ `Role`, native pair thread → `nativeWrites`.
-/

namespace Strata.Rope

-- § rotation algebra

/-- `rope_neox_pair`: rotate `(a, b)` by `(c, s)`. -/
def rotPair (a b c s : Int) : Int × Int := (a * c - b * s, a * s + b * c)

/-- The rotation scales the squared norm by `c² + s²`. -/
theorem rotPair_norm (a b c s : Int) :
    (rotPair a b c s).1 * (rotPair a b c s).1 + (rotPair a b c s).2 * (rotPair a b c s).2 =
      (a * a + b * b) * (c * c + s * s) := by
  simp only [rotPair]
  grind

/-- With a unit `(c, s)` the rotation preserves the norm. -/
theorem rotPair_norm_unit (a b c s : Int) (h : c * c + s * s = 1) :
    (rotPair a b c s).1 * (rotPair a b c s).1 + (rotPair a b c s).2 * (rotPair a b c s).2 = a * a + b * b := by
  rw [rotPair_norm, h, Int.mul_one]

/-- Position 0 (`cos = 1`, `sin = 0`) is the identity: the spare key's skipped rotation is exact. -/
theorem rotPair_pos0 (a b : Int) : rotPair a b 1 0 = (a, b) := by
  simp [rotPair]

/-- Rotating by `(c, s)` then by `(c, -s)` with `c² + s² = 1` undoes the rotation. -/
theorem rotPair_inverse (a b c s : Int) (h : c * c + s * s = 1) :
    rotPair (rotPair a b c s).1 (rotPair a b c s).2 c (-s) = (a, b) := by
  simp only [rotPair, Prod.mk.injEq]
  constructor <;> grind

-- § dimension roles

/-- The role of dimension `d` under partial NEOX with `half = n_rot / 2`. -/
inductive Role where
  | first (i : Nat)
  | second (i : Nat)
  | tail
deriving DecidableEq

/-- The unique role of dimension `d`. -/
def role (half d : Nat) : Role :=
  if d < half then .first d else if d < 2 * half then .second (d - half) else .tail

/-- Every role names the dimension it came from: first `i` is `d = i`, second `i` is `d = i + half`. -/
theorem role_spec (half d : Nat) :
    (∃ i, role half d = .first i ∧ d = i ∧ i < half) ∨
    (∃ i, role half d = .second i ∧ d = i + half ∧ i < half) ∨
    (role half d = .tail ∧ 2 * half ≤ d) := by
  unfold role
  by_cases h1 : d < half
  · exact Or.inl ⟨d, by simp [h1], rfl, h1⟩
  · by_cases h2 : d < 2 * half
    · exact Or.inr (Or.inl ⟨d - half, by simp [h1, h2], by omega, by omega⟩)
    · exact Or.inr (Or.inr ⟨by simp [h1, h2], by omega⟩)

/-- NEOX partners differ by `half`, not by one: `(i, i + half)` is never the adjacent pair for `half > 1`. -/
theorem neox_not_adjacent (half i : Nat) (h : 1 < half) : i + half ≠ i + 1 := by
  omega

/-- The NEOX pairing is a bijection between `[0, half)` and `[half, 2*half)`. -/
theorem partner_bij (half i j : Nat) (hj : half ≤ j) (hj2 : j < 2 * half) :
    (i < half → half ≤ i + half ∧ i + half < 2 * half) ∧ (j - half < half ∧ j - half + half = j) := by
  omega

-- § table

/-- `build_rope_table` offset. -/
def tabIdx (half p i : Nat) : Nat := p * half + i

/-- Table reads stay inside the `max_pos * half` table. -/
theorem tabIdx_lt (half maxPos p i : Nat) (hp : p < maxPos) (hi : i < half) : tabIdx half p i < maxPos * half := by
  unfold tabIdx
  have h1 : (p + 1) * half ≤ maxPos * half := Nat.mul_le_mul_right half (Nat.succ_le_of_lt hp)
  rw [Nat.succ_mul] at h1
  omega

-- § native pair threads (n_rot = 64)

/-- The two dimensions written by native pair thread `pair`. -/
def nativeWrites (pair : Nat) : Nat × Nat := if pair < 32 then (pair, pair + 32) else (2 * pair, 2 * pair + 1)

/-- Rotating threads write below 64, copying threads at or above 64: no write race. -/
theorem nativeWrites_disjoint (p q : Nat) (hp : p < 32) (hq : 32 ≤ q) :
    (nativeWrites p).1 < 64 ∧ (nativeWrites p).2 < 64 ∧ 64 ≤ (nativeWrites q).1 ∧ 64 ≤ (nativeWrites q).2 := by
  unfold nativeWrites
  simp only [hp, if_true, Nat.not_lt.mpr hq, if_false]
  omega

/-- Every dimension of a 128- or 256-wide row is written by some pair thread `< width/2`. -/
theorem nativeWrites_cover (width d : Nat) (hw : width = 128 ∨ width = 256) (hd : d < width) :
    ∃ pair, pair < width / 2 ∧ ((nativeWrites pair).1 = d ∨ (nativeWrites pair).2 = d) := by
  by_cases h1 : d < 32
  · exact ⟨d, by omega, Or.inl (by simp [nativeWrites, h1])⟩
  · by_cases h2 : d < 64
    · refine ⟨d - 32, by omega, Or.inr ?_⟩
      have : d - 32 < 32 := by omega
      simp only [nativeWrites, this, if_true]
      omega
    · refine ⟨d / 2, by omega, ?_⟩
      have : ¬ d / 2 < 32 := by omega
      simp only [nativeWrites, this, if_false]
      omega

end Strata.Rope
