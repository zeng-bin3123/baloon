/-!
# MODULE: activation_contract
SOURCES: include/strata/core/layer.hpp, src/core/layer.cpp
Which quantized activation image a weight consumes (a property of the TENSOR, not its role), how the three
canonical planes of a quantized tensor are located, and how the embedding column decode addresses its codes.

## Source semantics
* `gemv_quantized`: native tensors use the FP32 input; otherwise a 2-bit form uses Q8_0; a legacy 4/8-bit form
  uses Q8_0; a K-quant uses Q8_K.  BOTH quantized images are produced because the pack mixes them per layer.
* `plane_ptrs`: codes | scales | offsets are contiguous in that order and add up to the tensor's bytes;
  `has_offset` holds exactly when the offset plane is non-empty.
* `embed_row`: the code of element i is the `CODE_BITS`-wide field at bit `(i % PER_BYTE) * CODE_BITS` of byte
  `i / PER_BYTE`.

## Form template identifiers
Act <- {block_q8_0, block_q8_K, FP32 native input}; actOf <- the dispatch in `gemv_quantized`;
PlaneSizes <- `WeightRef::{codes_bytes, scales_bytes, offset_bytes, bytes}`; perByte <- `8 / code_bits`.
-/

namespace Strata.ActivationContract

-- § activation dispatch

/-- The activation image a projection reads. -/
inductive Act
  | q8_0
  | q8k
  | native
  deriving DecidableEq, Repr

/-- `gemv_quantized`'s choice, from (native?, code_bits, wants_q8k). -/
def actOf (native : Bool) (codeBits : Nat) (wantsQ8K : Bool) : Act :=
  if native = true then Act.native
  else if codeBits = 2 then Act.q8_0
  else if wantsQ8K = true then Act.q8k
  else Act.q8_0

/-- A native tensor always takes the FP32 activation. -/
theorem actOf_native (c : Nat) (w : Bool) : actOf true c w = Act.native := by
  simp [actOf]

/-- An S2 (Q2_0) form takes Q8_0 whatever `wants_q8k` says. -/
theorem actOf_s2 (w : Bool) : actOf false 2 w = Act.q8_0 := by
  simp [actOf]

/-- A K-quant (not 2-bit, wants Q8_K) takes Q8_K. -/
theorem actOf_kquant (c : Nat) (hc : c ≠ 2) : actOf false c true = Act.q8k := by
  simp [actOf, hc]

/-- A legacy form that does not want Q8_K takes Q8_0. -/
theorem actOf_legacy (c : Nat) : actOf false c false = Act.q8_0 := by
  simp [actOf]

/-- A non-native projection never receives the FP32 image. -/
theorem actOf_not_native (c : Nat) (w : Bool) : actOf false c w ≠ Act.native := by
  unfold actOf
  by_cases h : c = 2
  · simp [h]
  · cases w <;> simp [h]

-- § plane layout

/-- The recorded plane sizes of one quantized tensor. -/
structure PlaneSizes where
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat

/-- What `plane_ptrs` insists on before it returns pointers. -/
def planesOk (p : PlaneSizes) (codeBits : Nat) (hasOffset : Bool) : Prop :=
  (codeBits = 2 ∨ codeBits = 4 ∨ codeBits = 8) ∧ p.codes ≠ 0 ∧ p.scales ≠ 0 ∧
    p.codes + p.scales + p.offset = p.bytes ∧ (hasOffset = true ↔ p.offset ≠ 0)

/-- Start of the scale plane (just after the codes). -/
def scalesStart (p : PlaneSizes) : Nat := p.codes

/-- Start of the offset plane (just after the scales). -/
def offsetStart (p : PlaneSizes) : Nat := p.codes + p.scales

/-- The three planes tile the tensor exactly: nothing is read past its end. -/
theorem planes_tile (p : PlaneSizes) (c : Nat) (h : Bool) (hok : planesOk p c h) :
    scalesStart p = p.codes ∧ offsetStart p + p.offset = p.bytes := by
  obtain ⟨_, _, _, hsum, _⟩ := hok
  unfold scalesStart offsetStart
  exact ⟨rfl, hsum⟩

/-- A validated tensor is non-empty. -/
theorem planes_nonempty (p : PlaneSizes) (c : Nat) (h : Bool) (hok : planesOk p c h) : 0 < p.bytes := by
  obtain ⟨_, hc, _, hsum, _⟩ := hok
  omega

/-- The two ways of spelling "has an offset plane" agree after validation. -/
theorem planes_offset_iff (p : PlaneSizes) (c : Nat) (h : Bool) (hok : planesOk p c h) :
    (h = true ↔ p.offset ≠ 0) := hok.2.2.2.2

/-- Reading the scale plane as 4-byte floats when it is really 2-byte halves doubles its size and breaks the
tiling; the check therefore refuses a plane list whose sum differs from the tensor. -/
theorem doubled_scales_rejected (p : PlaneSizes) (c : Nat) (h : Bool)
    (hbad : p.codes + p.scales + p.offset ≠ p.bytes) : ¬ planesOk p c h := by
  intro hok
  exact hbad hok.2.2.2.1

-- § embedding column decode

/-- Codes per byte for the three legal widths (`8 / code_bits`). -/
def perByte : Nat → Nat
  | 2 => 4
  | 4 => 2
  | 8 => 1
  | _ => 0

/-- `perByte` really is `8 / code_bits` on the legal widths. -/
theorem perByte_eq (b : Nat) (hb : b = 2 ∨ b = 4 ∨ b = 8) : perByte b = 8 / b := by
  rcases hb with rfl | rfl | rfl <;> rfl

/-- The bit shift of element i stays inside one byte, so the field never straddles two bytes. -/
theorem shift_lt (b i : Nat) (hb : b = 2 ∨ b = 4 ∨ b = 8) : (i % perByte b) * b < 8 := by
  rcases hb with rfl | rfl | rfl <;> simp [perByte] <;> omega

/-- The code field fits entirely in its byte: shift + width ≤ 8. -/
theorem field_fits (b i : Nat) (hb : b = 2 ∨ b = 4 ∨ b = 8) : (i % perByte b) * b + b ≤ 8 := by
  rcases hb with rfl | rfl | rfl <;> simp [perByte] <;> omega

/-- The byte holding element i lies inside the row when the row length is a multiple of `perByte`. -/
theorem byte_in_row (b i ne0 : Nat) (hb : b = 2 ∨ b = 4 ∨ b = 8) (hdiv : ne0 % perByte b = 0)
    (hi : i < ne0) : i / perByte b < ne0 / perByte b := by
  rcases hb with rfl | rfl | rfl <;> simp [perByte] at hdiv ⊢ <;> omega

end Strata.ActivationContract
