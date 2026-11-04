/-!
# MODULE: s2_expert_grouped
Blob geometry, integer-dot identity, gate/up interleave layout and hit-compaction contracts of the grouped
Q2_0 GPU expert.

## Source semantics
sum_j (c_j - 1) * x_j = sum_j c_j*x_j - sum_j x_j (the two dp4a sums). The blob holds 2*FF interleaved
gate/up code rows (slot i: gate row i/2 if even, up row i/2 if odd), then down codes, gate/up scales, down
scales. Outputs are gate-major: gate of hit h at h*FF+r, up at n_hits*FF + h*FF + r.

## Form template identifiers
- `H`=2560, `FF`=640 ↔ `H`, `FF`; `outIdx` ↔ gate/up output slot; `popBelow` ↔ `__popc(hit & ((1<<lane)-1))`
-/

namespace Strata.S2ExpertGrouped

-- § blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- Expert FF width. -/
def FF : Nat := 640

/-- Bytes of 2-bit codes per gate/up row. -/
theorem row_gu : H / 4 = 640 := by decide
/-- Bytes of 2-bit codes per down row. -/
theorem row_d : FF / 4 = 160 := by decide
/-- fp16 scales per gate/up row (one per 64 weights). -/
theorem sc_gu : H / 64 = 40 := by decide
/-- fp16 scales per down row. -/
theorem sc_d : FF / 64 = 10 := by decide

/-- Offset of down codes. -/
def oDCodes : Nat := 2 * FF * 640
/-- Offset of gate/up scales. -/
def oGuScales : Nat := oDCodes + H * 160
/-- Offset of down scales. -/
def oDScales : Nat := oGuScales + 2 * FF * 40 * 2
/-- Total blob size. -/
def blobBytes : Nat := oDScales + H * 10 * 2

/-- The four regions are ordered and the offsets have the expected values. -/
theorem blob_offsets :
    oDCodes = 819200 ∧ oGuScales = 1228800 ∧ oDScales = 1331200 ∧ blobBytes = 1382400 := by
  decide

/-- Regions are in increasing order. -/
theorem blob_ordered : oDCodes < oGuScales ∧ oGuScales < oDScales ∧ oDScales < blobBytes := by
  decide

-- § integer dot identity

/-- Dot of codes with activations, with the bias `-1` applied to each code. -/
def biasedDot : List (Int × Int) → Int
  | [] => 0
  | p :: ps => (p.1 - 1) * p.2 + biasedDot ps

/-- Plain dot of codes with activations. -/
def plainDot : List (Int × Int) → Int
  | [] => 0
  | p :: ps => p.1 * p.2 + plainDot ps

/-- Sum of activations (the `ones · x` dp4a). -/
def actSum : List (Int × Int) → Int
  | [] => 0
  | p :: ps => p.2 + actSum ps

/-- The expert identity: `Σ (c-1)x = Σ c·x − Σ x`, exactly in integers. -/
theorem biased_dot_identity (l : List (Int × Int)) :
    biasedDot l = plainDot l - actSum l := by
  induction l with
  | nil => simp [biasedDot, plainDot, actSum]
  | cons p ps ih =>
    simp only [biasedDot, plainDot, actSum, Int.sub_mul, Int.one_mul, ih]
    omega

-- § 2-bit unpacking

/-- Field 0 of a code byte. -/
def c0 (b : Nat) : Nat := b % 4
/-- Field 1. -/
def c1 (b : Nat) : Nat := b / 4 % 4
/-- Field 2. -/
def c2 (b : Nat) : Nat := b / 16 % 4
/-- Field 3. -/
def c3 (b : Nat) : Nat := b / 64 % 4

/-- The four 2-bit fields reassemble the byte (LSB-first packing). -/
theorem fields_reassemble (b : Nat) (h : b < 256) : c0 b + 4 * c1 b + 16 * c2 b + 64 * c3 b = b := by
  unfold c0 c1 c2 c3; omega

/-- Each field is a 2-bit value. -/
theorem fields_lt (b : Nat) : c0 b < 4 ∧ c1 b < 4 ∧ c2 b < 4 ∧ c3 b < 4 := by
  unfold c0 c1 c2 c3; omega

-- § gate/up layout

/-- Output slot of row-slot `i` of hit `h`, gate-major layout. -/
def outIdx (nHits h i : Nat) : Nat :=
  if i % 2 = 1 then nHits * 640 + h * 640 + i / 2 else h * 640 + i / 2

/-- Every output slot is inside the `2 * n_hits * FF` buffer. -/
theorem outIdx_lt (nHits h i : Nat) (hh : h < nHits) (hi : i < 1280) :
    outIdx nHits h i < 2 * nHits * 640 := by
  unfold outIdx; split <;> omega

/-- The layout is injective: no two (hit, row-slot) pairs share an output slot. -/
theorem outIdx_inj (nHits h i h' i' : Nat) (hh : h < nHits) (hh' : h' < nHits)
    (hi : i < 1280) (hi' : i' < 1280) (e : outIdx nHits h i = outIdx nHits h' i') :
    h = h' ∧ i = i' := by
  unfold outIdx at e
  split at e <;> split at e <;> omega

/-- Even slots land in the gate half `[0, n_hits*FF)`. -/
theorem outIdx_gate_half (nHits h i : Nat) (hh : h < nHits) (hi : i < 1280) (he : i % 2 = 0) :
    outIdx nHits h i < nHits * 640 := by
  unfold outIdx; split <;> omega

-- § hit compaction

/-- Number of set bits among lanes below `lane` in a boolean mask. -/
def popBelow : List Bool → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | b :: bs, n + 1 => (if b then 1 else 0) + popBelow bs n

/-- Compaction index of a hit lane is bounded by the lane. -/
theorem popBelow_le (m : List Bool) : ∀ n, popBelow m n ≤ n := by
  induction m with
  | nil => intro n; cases n <;> simp [popBelow]
  | cons b bs ih =>
    intro n
    cases n with
    | zero => simp [popBelow]
    | succ k =>
      have := ih k
      simp only [popBelow]
      split <;> omega

/-- Slot decode: a flat warp slot maps to a hit index below `n_hits`. -/
theorem slot_hit_lt (slot rows nHits : Nat) (h : slot < nHits * rows) (hr : 0 < rows) :
    slot / rows < nHits := by
  have h' : slot < rows * nHits := by simpa [Nat.mul_comm] using h
  exact Nat.div_lt_of_lt_mul h'

end Strata.S2ExpertGrouped
