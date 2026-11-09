/-!
# MODULE: gguf_writer_q2_contract
SOURCES: tools/gguf_writer.py
A minimal GGUF v3 writer including the `Q2_0` quantiser: block = 64 elements, fp16 scale, 16 bytes of 2-bit
codes, LSB first, `d = amax`, `code = clamp(round(w/d) + 1, 0, 3)`, symbol = `code - 1`.

## Source semantics
* The code is clamped into 0..3, so the symbol set is {-1, 0, +1, +2}.
* With `d = amax` the ratio `w/d` lies in [-1, 1], so code 3 (symbol +2) is never produced by the quantiser.
* Element `j` lives in byte `j/4` at bit `(j%4)*2`: a bijection from 0..63 onto 16 bytes x 4 slots.
* Re-quantising dequantised values at the same scale is the identity on codes (the fixed-point check).
* Tensor data and the header are padded to `alignment` (32); each tensor offset is the aligned running sum.

## Form template identifiers
- np.clip(rint(b/d)+1, 0, 3) → `clamp3`
- packed[j // 4] |= code[j] << ((j % 4) * 2) → `slotOf`
- pad = (-len) % alignment → `pad32`
- running tensor offset → `alignUp32`
-/

namespace Strata.GgufWriter

-- § the quantiser clamp

/-- `clamp(x, 0, 3)` on integers, returned as a natural code. -/
def clamp3 (x : Int) : Nat :=
  if x < 0 then 0 else if x > 3 then 3 else x.toNat

/-- Every produced code is a valid 2-bit code. -/
theorem clamp3_lt (x : Int) : clamp3 x < 4 := by
  unfold clamp3
  split <;> (try split) <;> omega

/-- In-range values are unchanged by the clamp. -/
theorem clamp3_id (x : Int) (h0 : 0 ≤ x) (h3 : x ≤ 3) : (clamp3 x : Int) = x := by
  unfold clamp3
  split <;> (try split) <;> omega

/-- The decoded symbol of any code lies in {-1, 0, 1, 2}. -/
theorem symbol_range (x : Int) : -1 ≤ (clamp3 x : Int) - 1 ∧ (clamp3 x : Int) - 1 ≤ 2 := by
  have h := clamp3_lt x
  omega

/-- With `d = amax`, `rint(w/d) ∈ [-1, 1]` and so the +2 level (code 3) is never emitted. -/
theorem plus_two_never_emitted (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 1) : clamp3 (r + 1) ≤ 2 := by
  unfold clamp3
  split <;> (try split) <;> omega

/-- Fixed point: decoding a code at unit scale and re-quantising returns the same code. -/
theorem requantise_fixed_point (c : Nat) (h : c < 4) : clamp3 (((c : Int) - 1) + 1) = c := by
  unfold clamp3
  split <;> (try split) <;> omega

-- § bit packing

/-- Element `j` of a block goes to (byte `j/4`, slot `j%4`). -/
def slotOf (j : Nat) : Nat × Nat := (j / 4, j % 4)

/-- The slot map covers 16 bytes x 4 slots and is invertible. -/
theorem slotOf_bijective (j : Nat) (h : j < 64) :
    (slotOf j).1 < 16 ∧ (slotOf j).2 < 4 ∧ 4 * (slotOf j).1 + (slotOf j).2 = j := by
  unfold slotOf
  omega

/-- Two distinct elements never share a (byte, slot). -/
theorem slotOf_injective (i j : Nat) (hi : i < 64) (hj : j < 64) (h : slotOf i = slotOf j) : i = j := by
  unfold slotOf at h
  have h1 : i / 4 = j / 4 := congrArg Prod.fst h
  have h2 : i % 4 = j % 4 := congrArg Prod.snd h
  omega

/-- A Q2_0 block is a 2-byte scale plus 16 code bytes (64 elements at 2 bits each). -/
theorem q2_0_block_bytes : 2 + 64 * 2 / 8 = 18 := by
  decide

-- § alignment

/-- `(-len) % 32` written without negation. -/
def pad32 (n : Nat) : Nat := (32 - n % 32) % 32

/-- Padding always lands on an alignment boundary. -/
theorem pad32_aligned (n : Nat) : (n + pad32 n) % 32 = 0 := by
  unfold pad32
  omega

/-- Round up to the 32-byte alignment (the tensor directory offset update). -/
def alignUp32 (n : Nat) : Nat := (n + 31) / 32 * 32

/-- The aligned offset never overlaps the previous tensor and wastes less than one unit. -/
theorem alignUp32_spec (n : Nat) : n ≤ alignUp32 n ∧ alignUp32 n < n + 32 ∧ alignUp32 n % 32 = 0 := by
  unfold alignUp32
  omega

/-- Next tensor offset is at least the previous offset plus its length. -/
theorem offsets_monotone (off len : Nat) : off + len ≤ alignUp32 (off + len) := by
  have h := alignUp32_spec (off + len)
  exact h.1

end Strata.GgufWriter
