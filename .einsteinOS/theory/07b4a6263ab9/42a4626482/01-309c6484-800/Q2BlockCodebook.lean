/-!
# MODULE: q2_block_codebook
SOURCES: include/strata/artifact/dequant.hpp, tools/gguf_writer.py, tools/canonical_xcheck.py
Integer-level model of the Q2_0 block (64 elements, fp16 scale, 16 bytes of 2-bit codes): code
extraction, the asymmetric codebook {-1,0,+1,+2}, the LSB-first byte/bit addressing, the writer's
quantizer, and the requantization fixed-point property used as the writer's self-check.

## Source semantics
* block = { fp16 d; uint8 qs[16] }, 64 elements, 18 bytes.
* element j lives in byte j/4 at bit offset (j%4)*2; code = (qs[j/4] >> (j%4)*2) & 3.
* value = (code - 1) * d; the `- 1` is an integer bias applied before the multiply.
* the writer picks d = amax and code = clamp(round(w/d) + 1, 0, 3); since |w| ≤ d the rounded
  symbol is in {-1,0,1}, so the writer never emits code 3.
* requantizing dequantized values reproduces the codes exactly, PROVIDED no code is 3 and at least
  one code is non-1 (otherwise amax changes). Code 3 breaks the fixed point.

## Form template identifiers
dequantize_q2_0 / dequantize_row_q2_0 / from_s2 → `q2Value`; qs byte addressing → `q2ByteIndex`,
`q2BitOffset`; code extraction → `q2Code`; quantize_q2_0 clamp → `q2Quantize`; to_s2 unpack
`codes[:, t::4]` → `q2Code` with j = 4*b + t; QK=64, BLOCK_BYTES=18 → `qk`, `blockBytes`.
-/

namespace Strata.Q2Block

-- § constants

/-- Elements per Q2_0 block. -/
def qk : Nat := 64
/-- Bytes per Q2_0 block: a 2-byte fp16 scale plus 16 code bytes. -/
def blockBytes : Nat := 18

/-- The block size is exactly the fp16 scale plus 2 bits per element. -/
theorem blockBytes_eq : blockBytes = 2 + qk * 2 / 8 := by decide

-- § addressing

/-- Byte holding element `j` (`j / 4`). -/
def q2ByteIndex (j : Nat) : Nat := j / 4
/-- Bit offset of element `j` inside its byte (`(j % 4) * 2`). -/
def q2BitOffset (j : Nat) : Nat := (j % 4) * 2

/-- Every element of a block addresses one of the 16 code bytes. -/
theorem byteIndex_lt (j : Nat) (hj : j < qk) : q2ByteIndex j < 16 := by
  unfold q2ByteIndex qk at *; omega

/-- Bit offsets are 0, 2, 4 or 6, so a 2-bit field never crosses a byte. -/
theorem bitOffset_le (j : Nat) : q2BitOffset j ≤ 6 := by
  unfold q2BitOffset; omega

/-- Addressing is injective: distinct elements use distinct (byte, bit) slots. -/
theorem address_injective (i j : Nat)
    (hb : q2ByteIndex i = q2ByteIndex j) (ho : q2BitOffset i = q2BitOffset j) : i = j := by
  unfold q2ByteIndex q2BitOffset at *; omega

/-- `to_s2`'s slice `codes[:, t::4]` for byte `b` is element `4*b + t`: the same addressing. -/
theorem slice_matches_address (b t : Nat) (ht : t < 4) :
    q2ByteIndex (4 * b + t) = b ∧ q2BitOffset (4 * b + t) = 2 * t := by
  unfold q2ByteIndex q2BitOffset; omega

-- § codes and values

/-- The 2-bit code of element `j` read from byte `byte`. -/
def q2Code (byte j : Nat) : Nat := (byte >>> q2BitOffset j) &&& 3

/-- Extracted codes are always in `0..3`. -/
theorem q2Code_lt (byte j : Nat) : q2Code byte j < 4 := by
  have h : q2Code byte j = (byte >>> q2BitOffset j) % 2 ^ 2 := by
    unfold q2Code
    exact Nat.and_two_pow_sub_one_eq_mod _ 2
  rw [h]; exact Nat.mod_lt _ (by decide)

/-- The integer symbol of a code: `code - 1`. -/
def q2Symbol (c : Nat) : Int := (c : Int) - 1

/-- Dequantized value in units of an integer scale `d`. -/
def q2Value (c : Nat) (d : Int) : Int := q2Symbol c * d

/-- The codebook is asymmetric: the four symbols are exactly -1, 0, +1, +2. -/
theorem symbol_set (c : Nat) (hc : c < 4) :
    q2Symbol c = -1 ∨ q2Symbol c = 0 ∨ q2Symbol c = 1 ∨ q2Symbol c = 2 := by
  unfold q2Symbol; omega

/-- Structural invariant of `--check`: every value is one of `-d, 0, d, 2d`. -/
theorem value_set (byte j : Nat) (d : Int) :
    let v := q2Value (q2Code byte j) d
    v = -d ∨ v = 0 ∨ v = d ∨ v = 2 * d := by
  have hc := q2Code_lt byte j
  have hs := symbol_set (q2Code byte j) hc
  simp only [q2Value]
  rcases hs with h | h | h | h <;> rw [h] <;> omega

/-- Decoding unsigned codes as `code * d` (no bias) is wrong on every element. -/
theorem unbiased_decode_differs (c : Nat) (d : Int) (hd : d ≠ 0) :
    q2Value c d ≠ (c : Int) * d := by
  unfold q2Value q2Symbol
  intro h
  have : ((c : Int) - 1) * d - (c : Int) * d = 0 := by omega
  have : -d = 0 := by
    have e : ((c : Int) - 1) * d - (c : Int) * d = -d := by
      rw [Int.sub_mul, Int.one_mul]; omega
    omega
  omega

-- § quantizer

/-- `clamp(r + 1, 0, 3)` applied to an integer rounded symbol `r`. -/
def q2Quantize (r : Int) : Nat := (max 0 (min 3 (r + 1))).toNat

/-- The quantizer always yields a valid 2-bit code. -/
theorem q2Quantize_lt (r : Int) : q2Quantize r < 4 := by
  unfold q2Quantize; omega

/-- With `d = amax`, the rounded symbol lies in `[-1, 1]`, so the writer never emits code 3. -/
theorem quantize_never_three (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 1) : q2Quantize r ≤ 2 := by
  unfold q2Quantize; omega

/-- On symbols the codebook can represent, quantization inverts `q2Symbol`. -/
theorem quantize_symbol (c : Nat) (hc : c < 4) : q2Quantize (q2Symbol c) = c := by
  unfold q2Quantize q2Symbol; omega

-- § requantization fixed point

/-- Exact division recovers the symbol from a dequantized value. -/
theorem value_div_scale (c : Nat) (d : Int) (hd : d ≠ 0) : q2Value c d / d = q2Symbol c := by
  unfold q2Value
  exact Int.mul_ediv_cancel _ hd

/-- Fixed point (writer self-check): when the requantization scale equals the original `d`
(true whenever some code is 0 or 2 and none is 3), each code is reproduced exactly. -/
theorem requantize_fixed (c : Nat) (hc : c < 4) (d : Int) (hd : d ≠ 0) :
    q2Quantize (q2Value c d / d) = c := by
  have h := value_div_scale c d hd
  rw [h]
  exact quantize_symbol c hc

/-- Codes `[1, 3]` with `d = 2` dequantize to `[0, 4]`; amax becomes 4, and code 3 requantizes
to code 2. The fixed point genuinely needs the "no code 3" precondition. -/
theorem code_three_breaks_fixed_point :
    let d : Int := 2
    let vals := [q2Value 1 d, q2Value 3 d]
    let dNew : Int := 4
    vals = [0, 4] ∧ (vals.map fun v => q2Quantize (v / dNew)) = [1, 2] := by
  decide

end Strata.Q2Block
