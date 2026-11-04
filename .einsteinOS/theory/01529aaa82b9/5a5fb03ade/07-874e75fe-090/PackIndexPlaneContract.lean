/-!
# MODULE: pack_index_plane_contract
`tools/pack_index.py`: flattens the pack manifest into the engine loader's index.  The engine form of a
quantized tensor differs from the pack by exactly the fp16->f32 scale widening; the activation-quant kind
cannot be derived from the S-form fields.

## Source semantics
* engine scale plane = groups * 4 B always; pack scale plane = groups * (2 if fp16 else 4).
* dst = src + widening; planes must sum to the span.
* BF16 / F16 sources are re-rounded to 2 B per element in the engine.
* `vec_dot_type`: legacy formats use Q8_0, K-quants and IQ4_XS use Q8_K.

## Form template identifiers
nGroups      -> ne1 * (ne0 / group_elems)
scales_fp16  -> scale width flag
act_kind     -> vec_dot_type
-/

namespace Strata.PackIndexPlaneContract

-- § plane sizes

/-- Pack-side scale plane bytes for `n` groups. -/
def packScales (n : Nat) (fp16 : Bool) : Nat := n * (if fp16 then 2 else 4)

/-- Engine-side scale plane bytes: always 4 per group. -/
def engineScales (n : Nat) : Nat := n * 4

/-- Widening cost of one tensor: fp16 planes grow by 2 bytes per group. -/
def widening (n : Nat) (fp16 : Bool) : Nat := if fp16 then n * 2 else 0

/-- The widening is exactly the engine/pack difference of the scale plane. -/
theorem widening_eq_diff (n : Nat) (fp16 : Bool) : widening n fp16 = engineScales n - packScales n fp16 := by
  cases fp16 <;> simp [widening, engineScales, packScales] <;> omega

/-- The engine form equals the pack form plus the widening (no offset plane). -/
theorem engine_eq_pack_plus_widen (codes n : Nat) (fp16 : Bool) :
    codes + engineScales n = codes + packScales n fp16 + widening n fp16 := by
  cases fp16 <;> simp [widening, engineScales, packScales] <;> omega

/-- With an offset plane, both planes widen and the equality still holds. -/
theorem engine_eq_pack_plus_widen_off (codes n : Nat) (sfp offp : Bool) :
    codes + engineScales n + engineScales n =
      codes + packScales n sfp + packScales n offp + widening n sfp + widening n offp := by
  cases sfp <;> cases offp <;> simp [widening, engineScales, packScales] <;> omega

/-- fp16 scales widen to exactly twice their pack size: the widening costs `2n` bytes. -/
theorem widen_cost (n : Nat) : widening n true = 2 * n := by
  simp [widening] <;> omega

/-- f32 scales need no widening. -/
theorem no_widen_f32 (n : Nat) : widening n false = 0 := by
  unfold widening; simp

/-- Misreading an fp16 plane as 4-byte-wide gives a plane twice the real size. -/
theorem wrong_width_doubles (n : Nat) : engineScales n = 2 * packScales n true := by
  simp [engineScales, packScales] <;> omega

-- § re-rounding

/-- Bytes of a 16-bit re-rounded tensor and of its f32 promotion. -/
theorem rerounding_saves (elems : Nat) : elems * 4 - elems * 2 = elems * 2 := by omega

/-- The engine size of a re-rounded float tensor is two bytes per element. -/
theorem rerounded_bytes (elems : Nat) : elems * 2 ≤ elems * 4 := by omega

-- § activation kind

/-- Quantized source types of the pack. -/
inductive QT
  | q2_0 | q3_k | q4_0 | q4_k | q5_0 | q5_k | q6_k | q8_0 | iq4_nl | iq4_xs
  deriving DecidableEq

/-- Activation quantization kind. -/
inductive Act
  | q8_0 | q8_k
  deriving DecidableEq

/-- `vec_dot_type` of each source type. -/
def vdt : QT → Act
  | .q2_0 => .q8_0 | .q4_0 => .q8_0 | .q5_0 => .q8_0 | .q8_0 => .q8_0 | .iq4_nl => .q8_0
  | .q3_k => .q8_k | .q4_k => .q8_k | .q5_k => .q8_k | .q6_k => .q8_k | .iq4_xs => .q8_k

/-- The S-form fields: (code_bits, code_bias, group_elems, iq4 codebook, has_offset). -/
def sform : QT → Nat × Int × Nat × Bool × Bool
  | .q2_0 => (2, -1, 64, false, false)
  | .q3_k => (4, -4, 16, false, false)
  | .q4_0 => (4, -8, 32, false, false)
  | .q4_k => (4, 0, 32, false, true)
  | .q5_0 => (8, -16, 32, false, false)
  | .q5_k => (8, 0, 32, false, true)
  | .q6_k => (8, -32, 16, false, false)
  | .q8_0 => (8, -128, 32, false, false)
  | .iq4_nl => (4, 0, 32, true, false)
  | .iq4_xs => (4, 0, 32, true, false)

/-- IQ4_NL and IQ4_XS are indistinguishable by S-form fields. -/
theorem iq4_same_sform : sform .iq4_nl = sform .iq4_xs := rfl

/-- They nevertheless need different activation kinds. -/
theorem iq4_diff_vdt : vdt .iq4_nl ≠ vdt .iq4_xs := by decide

/-- No function of the S-form fields can recover the activation kind. -/
theorem no_vdt_from_sform : ¬ ∃ f : Nat × Int × Nat × Bool × Bool → Act, ∀ t, f (sform t) = vdt t := by
  rintro ⟨f, hf⟩
  have h1 := hf .iq4_nl
  have h2 := hf .iq4_xs
  rw [← iq4_same_sform] at h2
  exact iq4_diff_vdt (h1.symm.trans h2)

/-- Q5_0 and Q5_K share bits and the 8-bit code plane but differ in `has_offset`. -/
theorem q5_differ_only_by_offset :
    (sform .q5_0).1 = (sform .q5_k).1 ∧ (sform .q5_0).2.2.2.2 ≠ (sform .q5_k).2.2.2.2 := by decide

end Strata.PackIndexPlaneContract
