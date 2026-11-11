/-!
# MODULE: q2_zero_ternary_codec
SOURCES: include/strata/artifact/dequant.hpp, tools/gguf_writer.py, tools/canonical_xcheck.py
The Q2_0 block codec that carries every routed-expert weight: one fp16 scale `d` followed by 16
bytes holding 64 two-bit codes, least-significant pair first. This module models code extraction,
the `code - 1` symbol offset, the quantizer's clamp, and the S2 canonical unpacking.

## Source semantics
* A block is `{fp16 d; uint8 qs[16]}`: 64 elements in 18 bytes (`QK2_0 = 64`, not 128).
* Element `j` lives in byte `j / 4` at bit offset `(j % 4) * 2`; its code is `(qs[j/4] >> off) & 3`.
* The codebook is affine with an INTEGER bias: `value = (code - 1) * d`, so the representable set is
  `{-d, 0, d, 2d}`, never `{0, d, 2d, 3d}` (the `--check` structural invariant).
* The quantizer sets `code = clamp(round(w/d) + 1, 0, 3)`; re-quantizing a dequantized value is a
  fixed point (the writer's self-check).
* `to_s2` unpacks to one code per element with `codes[:, t::4] = (byte >> 2t) & 3`, which is the
  same bit address as the C++ loop.
Floating-point multiplication is abstracted as exact `Int` arithmetic on the scale; the theorems
are about the integer code path, which is where the documented bugs (bit order, sign, stride) live.

## Form template identifiers
`qs` → `codeBytes`, `dequantize_q2_0` → `q2Value`, `(code - 1)` → `q2Symbol`,
`quantize_q2_0` code step → `q2QuantCode`, packing of four codes → `pack4`, `QK2_0` → `q2BlockElems`,
18 → `q2BlockBytes`.
-/

namespace Strata.Q2Zero

-- § constants

/-- Elements per Q2_0 block. -/
def q2BlockElems : Nat := 64
/-- Bytes per Q2_0 block: a 2-byte fp16 scale plus 2 bits per element. -/
def q2BlockBytes : Nat := 2 + q2BlockElems * 2 / 8

/-- The Q2_0 block is 18 bytes, as `block_geometry(42)` and `BLOCK_BYTES` state. -/
theorem q2BlockBytes_eq : q2BlockBytes = 18 := by decide

/-- A block stores exactly four codes per byte: 16 code bytes for 64 elements. -/
theorem q2_code_bytes : q2BlockElems / 4 = 16 := by decide

-- § code extraction

/-- Byte index of element `j` inside the code area (`j / 4`). -/
def byteIndex (j : Nat) : Nat := j / 4
/-- Bit offset of element `j` inside its byte (`(j % 4) * 2`). -/
def bitOffset (j : Nat) : Nat := (j % 4) * 2
/-- The 2-bit code of element `j`, read from the code bytes `qs`. -/
def q2Code (qs : Nat → Nat) (j : Nat) : Nat := (qs (byteIndex j) >>> bitOffset j) &&& 3

/-- Every element of a block addresses one of the 16 code bytes. -/
theorem byteIndex_lt {j : Nat} (hj : j < 64) : byteIndex j < 16 := by
  unfold byteIndex; omega

/-- The bit offset is one of 0, 2, 4, 6, so a code never straddles a byte boundary. -/
theorem bitOffset_le_six (j : Nat) : bitOffset j ≤ 6 ∧ bitOffset j % 2 = 0 := by
  unfold bitOffset; omega

/-- The element-to-(byte, slot) map is injective: distinct elements read distinct bit pairs. -/
theorem address_injective {i j : Nat} (h1 : byteIndex i = byteIndex j)
    (h2 : bitOffset i = bitOffset j) : i = j := by
  unfold byteIndex bitOffset at *; omega

/-- A decoded code always lies in `0..3` (the `& 0x03` mask). -/
theorem q2Code_lt_four (qs : Nat → Nat) (j : Nat) : q2Code qs j < 4 := by
  have h : (qs (byteIndex j) >>> bitOffset j) &&& 3 ≤ 3 := Nat.and_le_right
  unfold q2Code; omega

-- § symbols and values

/-- The code-to-symbol map `code - 1`, an integer bias applied before the multiply. -/
def q2Symbol (c : Nat) : Int := (c : Int) - 1
/-- The dequantized value with an integer-modelled scale `d`. -/
def q2Value (d : Int) (c : Nat) : Int := q2Symbol c * d

/-- Codes `0..3` map to symbols `{-1, 0, +1, +2}`. -/
theorem q2Symbol_range {c : Nat} (hc : c < 4) :
    q2Symbol c = -1 ∨ q2Symbol c = 0 ∨ q2Symbol c = 1 ∨ q2Symbol c = 2 := by
  unfold q2Symbol; omega

/-- The `--check` structural invariant: each output is one of `{-d, 0, d, 2d}` for its own scale. -/
theorem q2Value_mem (d : Int) (qs : Nat → Nat) (j : Nat) :
    q2Value d (q2Code qs j) = -d ∨ q2Value d (q2Code qs j) = 0 ∨
    q2Value d (q2Code qs j) = d ∨ q2Value d (q2Code qs j) = 2 * d := by
  have hc := q2Code_lt_four qs j
  obtain h | h | h | h := q2Symbol_range hc
  · left; unfold q2Value; rw [h]; omega
  · right; left; unfold q2Value; rw [h]; omega
  · right; right; left; unfold q2Value; rw [h]; omega
  · right; right; right; unfold q2Value; rw [h]

/-- Decoding `code * d` instead of `(code - 1) * d` is off by exactly one scale step on EVERY
element: the failure mode `canonical_xcheck.py` exists to catch. -/
theorem missing_bias_off_by_d (d : Int) (c : Nat) : (c : Int) * d - q2Value d c = d := by
  unfold q2Value q2Symbol
  rw [Int.sub_mul, Int.one_mul]; omega

-- § quantizer

/-- The writer's code step: `clamp(round(w/d) + 1, 0, 3)` given the rounded ratio `r`. -/
def q2QuantCode (r : Int) : Int := max 0 (min 3 (r + 1))

/-- Quantized codes always fit in two bits. -/
theorem q2QuantCode_range (r : Int) : 0 ≤ q2QuantCode r ∧ q2QuantCode r ≤ 3 := by
  unfold q2QuantCode; omega

/-- A ratio already on the grid `{-1,0,1,2}` is quantized to the code whose symbol is itself. -/
theorem q2QuantCode_on_grid {r : Int} (h1 : -1 ≤ r) (h2 : r ≤ 2) : q2QuantCode r - 1 = r := by
  unfold q2QuantCode; omega

/-- Requantization fixed point: quantizing a dequantized symbol returns the original code. -/
theorem requant_fixed_point {c : Nat} (hc : c < 4) : q2QuantCode (q2Symbol c) = c := by
  unfold q2QuantCode q2Symbol; omega

/-- Ratios outside the grid saturate to the end codes rather than wrapping. -/
theorem q2QuantCode_saturates (r : Int) :
    (r ≤ -1 → q2QuantCode r = 0) ∧ (2 ≤ r → q2QuantCode r = 3) := by
  unfold q2QuantCode; omega

-- § packing

/-- Pack four 2-bit codes into one byte, least-significant first (the writer's `|=` loop). -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 ||| (c1 <<< 2) ||| (c2 <<< 4) ||| (c3 <<< 6)

/-- Unpacking slot `k` of a packed byte recovers the code written there, for every slot. -/
theorem pack4_roundtrip : ∀ c0, c0 < 4 → ∀ c1, c1 < 4 → ∀ c2, c2 < 4 → ∀ c3, c3 < 4 →
    (pack4 c0 c1 c2 c3 &&& 3 = c0) ∧ ((pack4 c0 c1 c2 c3 >>> 2) &&& 3 = c1) ∧
    ((pack4 c0 c1 c2 c3 >>> 4) &&& 3 = c2) ∧ ((pack4 c0 c1 c2 c3 >>> 6) &&& 3 = c3) := by
  unfold pack4; decide

/-- A packed byte fits in one byte. -/
theorem pack4_lt_256 : ∀ c0, c0 < 4 → ∀ c1, c1 < 4 → ∀ c2, c2 < 4 → ∀ c3, c3 < 4 →
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4; decide

/-- The S2 unpacking `codes[:, t::4] = (byte >> 2t) & 3` addresses the same bits as the C++ loop:
element `4b + t` is read from byte `b` at offset `2t`. -/
theorem s2_unpack_matches (qs : Nat → Nat) {b t : Nat} (ht : t < 4) :
    q2Code qs (4 * b + t) = (qs b >>> (2 * t)) &&& 3 := by
  have h1 : byteIndex (4 * b + t) = b := by unfold byteIndex; omega
  have h2 : bitOffset (4 * b + t) = 2 * t := by unfold bitOffset; omega
  unfold q2Code; rw [h1, h2]

-- § tensor size

/-- Bytes of a Q2_0 tensor with `n` elements (a whole number of blocks). -/
def q2TensorBytes (n : Nat) : Nat := n / q2BlockElems * q2BlockBytes

/-- One expert's role tensor (2560 x 640) is 460,800 bytes of Q2_0. -/
theorem expert_role_bytes : q2TensorBytes (2560 * 640) = 460800 := by decide

/-- Q2_0 costs 18*8 = 144 bits per 64 elements, i.e. 2.25 bits per weight. -/
theorem q2_bits_per_weight : q2BlockBytes * 8 * 4 = q2BlockElems * 9 := by decide

end Strata.Q2Zero
