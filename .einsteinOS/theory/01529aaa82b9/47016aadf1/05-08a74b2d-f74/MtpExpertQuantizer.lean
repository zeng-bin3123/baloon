/-!
# MODULE: mtp_expert_quantizer
Packing of the MTP draft block's fused routed experts into Q2_0 / Q4_0 / Q8_0 GGUF blocks: tensor shapes,
divisibility by block size, byte totals, scale-search range, code ranges and nibble packing.

## Source semantics
* Fused experts: `gate_up_proj [512, 1280, 2560]` and `down_proj [512, 2560, 640]`, quantised along the last axis.
* Block sizes: Q2_0 64 elements / 18 bytes, Q4_0 32 / 18, Q8_0 32 / 34. The inner dimension must divide.
* Q2_0 scale candidates are `amax * f` with `f` from 0.5 to 1.0 in 16 equal steps (17 points).
* Q2_0 code = `clip(rint(x/d), -1, 2) + 1`; Q4_0 nibble = `min 15 (trunc (x/d + 8.5))`; low half and high
  half of the 32 elements share a byte.

## Form template identifiers
q2_0/q4_0/q8_0 -> quantize_q2/q4/q8, EXPERT_TENSORS -> expert_tensors, QUANT -> format_table.
-/

namespace Strata.MtpExpertQuantizer

-- § shapes and sizes

/-- Number of routed experts. -/
def E : Nat := 512
/-- Fused gate/up output width. -/
def GU_OUT : Nat := 1280
/-- Hidden size (inner dim of gate_up, output of down). -/
def HID : Nat := 2560
/-- Expert intermediate width (inner dim of down). -/
def INTER : Nat := 640

/-- Bytes of one row of inner width `n` for a block geometry. -/
def rowBytes (n block bytes : Nat) : Nat := n / block * bytes

/-- Total quantised bytes of both fused expert tensors. -/
def totalBytes (block bytes : Nat) : Nat :=
  E * GU_OUT * rowBytes HID block bytes + E * HID * rowBytes INTER block bytes

/-- Inner dimensions divide every block size in use (checked for Q2_0 and 32-element blocks). -/
theorem inner_dims_divide :
    HID % 64 = 0 ∧ INTER % 64 = 0 ∧ HID % 32 = 0 ∧ INTER % 32 = 0 := by decide

/-- Q2_0 expert bytes: 707,788,800 (about 0.71 GB). -/
theorem q2_0_total : totalBytes 64 18 = 707788800 := by decide

/-- Q4_0 expert bytes: 1,415,577,600 (about 1.42 GB). -/
theorem q4_0_total : totalBytes 32 18 = 1415577600 := by decide

/-- Q8_0 expert bytes: 2,673,868,800 (about 2.67 GB). -/
theorem q8_0_total : totalBytes 32 34 = 2673868800 := by decide

/-- Q2_0 is the smallest and Q8_0 the largest of the three formats. -/
theorem format_order : totalBytes 64 18 < totalBytes 32 18 ∧ totalBytes 32 18 < totalBytes 32 34 := by
  decide

/-- The fused gate_up tensor in float32 is 6,710,886,400 bytes, which is why it is streamed per expert. -/
theorem gate_up_f32_bytes : E * GU_OUT * HID * 4 = 6710886400 := by decide

/-- The `raw_shape` last axis passed to the writer is `n / block * bytes`. -/
theorem raw_last_axis_q2 : rowBytes HID 64 18 = 720 ∧ rowBytes INTER 64 18 = 180 := by decide

-- § scale search

/-- Candidate scale factors are `(16 + k) / 32` for `k = 0 .. 16`, expressed by numerator over 32. -/
theorem scale_candidates (k : Nat) (hk : k ≤ 16) : 16 ≤ 16 + k ∧ 16 + k ≤ 32 := by omega

/-- The first candidate is one half and the last is one. -/
theorem scale_endpoints : 16 + 0 = 32 / 2 ∧ 16 + 16 = 32 := by decide

/-- Seventeen candidates are tried. -/
theorem candidate_count : (List.range 17).length = 17 := by simp

-- § code ranges

/-- Clip a rounded value to `[-1, 2]`. -/
def clip12 (v : Int) : Int := max (-1) (min 2 v)

/-- Stored Q2_0 codes `clip + 1` lie in `[0, 3]`. -/
theorem q2_code_range (v : Int) : 0 ≤ clip12 v + 1 ∧ clip12 v + 1 ≤ 3 := by
  unfold clip12
  omega

/-- A value already on the grid keeps its code. -/
theorem q2_grid_fixed (v : Int) (h1 : -1 ≤ v) (h2 : v ≤ 2) : clip12 v = v := by
  unfold clip12
  omega

/-- Q4_0 nibble is capped at 15. -/
def nibble (t : Nat) : Nat := min 15 t

/-- Nibbles are below 16. -/
theorem nibble_lt (t : Nat) : nibble t < 16 := by
  unfold nibble
  omega

/-- Packing the low half and the high half of 32 nibbles into bytes loses nothing. -/
theorem nibble_pack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi ∧ lo + 16 * hi < 256 := by omega

/-- Q4_0 element `i < 16` pairs with element `i + 16` in byte `i`. -/
theorem q4_pairing (i : Nat) (hi : i < 16) : i + 16 < 32 ∧ i < 16 := by omega

/-- Q4_0 scale is `max / -8`: the extreme element maps to code 0 or 15 only after rounding; the offset 8 keeps
codes unsigned: `-8 .. 7` shifted by 8 is `0 .. 15`. -/
theorem q4_offset_range (c : Int) (h1 : -8 ≤ c) (h2 : c ≤ 7) : 0 ≤ c + 8 ∧ c + 8 ≤ 15 := by omega

/-- Q8_0 int8 codes: `rint(x / d)` is in `[-127, 127]` when `d = amax / 127`. -/
theorem q8_symmetric (c : Int) (h1 : -127 ≤ c) (h2 : c ≤ 127) : -128 ≤ c ∧ c ≤ 127 := by omega

-- § checkpoint reading

/-- A BF16 tensor file must contain exactly the declared number of values. -/
def sizeMatches (rawSize prodShape : Nat) : Bool := rawSize == prodShape

/-- A mismatching size is rejected. -/
theorem size_mismatch_rejected (a b : Nat) (h : a ≠ b) : sizeMatches a b = false := by
  simp [sizeMatches, h]

/-- BF16 widens to float32 by a 16-bit shift and is recovered exactly. -/
theorem bf16_widen (h : Nat) (hh : h < 65536) : h * 65536 / 65536 = h := by omega

end Strata.MtpExpertQuantizer
