/-!
# MODULE: pack_index_planes
SOURCES: tools/pack_index.py
The flat engine index derived from the pack manifest: per-tensor load KIND, the three engine-form
plane sizes (codes, scales widened to f32, offsets), the consistency checks between pack-form and
engine-form widths, the aligned engine pool, and the activation-type column.

## Source semantics
* kind: BF16 → BF16_IN_F32 (take the high 16 bits, 2 B/elem); F16 stored promoted → F16_IN_F32
  (real RNE narrowing, 2 B/elem); F16 stored as fp16 → VERBATIM; F32 → F32 copy; quantized →
  VERBATIM planes.
* engine scales are ALWAYS 4 bytes; pack scales are 2 bytes when `scales_fp16`. Hence
  dst_bytes = src_bytes + widening, widening = ne1·n_groups·2 exactly when scales are fp16.
* the planes must be contiguous in the order codes, scales, offsets and fill the span exactly.
* engine offsets: dst advances by `align_up(dst + dst_bytes)`, so every tensor starts aligned and
  tensors never overlap.
* act_kind (Q8_0 vs Q8_K activations) cannot be recovered from the S-form fields: IQ4_NL and
  IQ4_XS share every S-form field yet differ in act_kind.
* the written row has 19 columns (including act_kind); the module docstring lists only 18.

## Form template identifiers
kind selection → `loadKind`; plane sizes → `enginePlanes`/`packPlanes`; widening → `widening`;
align_up → `alignUp`; VDT → `actKind`; S-form fields → `SForm`/`sform`; header columns →
`headerColumns`/`docColumns`.
-/

namespace Strata.PackIndex

-- § load kind

/-- The four load kinds of the index. -/
inductive Kind | verbatim | bf16InF32 | f32Copy | f16InF32
deriving DecidableEq, Repr

/-- `kind` from `source_type` and `values_fp16`, in the script's branch order. -/
def loadKind (st : String) (valuesFp16 : Bool) : Kind :=
  if st = "BF16" then .bf16InF32
  else if st = "F16" ∧ !valuesFp16 then .f16InF32
  else if st = "F16" then .verbatim
  else if st = "F32" then .f32Copy
  else .verbatim

/-- Engine bytes for the float kinds. -/
def floatDst (k : Kind) (elems srcBytes : Nat) : Nat :=
  match k with
  | .bf16InF32 => elems * 2
  | .f16InF32 => elems * 2
  | .f32Copy => elems * 4
  | .verbatim => srcBytes

/-- The two 16-bit source types are re-rounded to 2 bytes per element whenever the pack promoted
them, and the promoted pack holds 4 bytes per element, so the saving is exactly half. -/
theorem rerounding_halves (elems : Nat) :
    floatDst (loadKind "BF16" false) elems (elems * 4) = elems * 2 ∧
      floatDst (loadKind "F16" false) elems (elems * 4) = elems * 2 ∧
      elems * 4 - elems * 2 = elems * 2 := by
  simp [loadKind, floatDst]; omega

/-- An F16 tensor already stored as fp16 is copied verbatim. -/
theorem f16_kept_verbatim (elems : Nat) :
    loadKind "F16" true = .verbatim ∧ floatDst (loadKind "F16" true) elems (elems * 2) = elems * 2 := by
  simp [loadKind, floatDst]

/-- Quantized types always load verbatim (their planes are re-sized, not re-encoded). -/
theorem quantized_verbatim : loadKind "Q2_0" false = .verbatim ∧ loadKind "Q4_K" true = .verbatim := by
  simp [loadKind]

-- § plane sizes

/-- Pack-form plane bytes: scales/offsets are 2 B if their fp16 flag is set, else 4 B. -/
def packPlanes (codes groups : Nat) (scFp16 hasOff offFp16 : Bool) : Nat :=
  codes + groups * (if scFp16 then 2 else 4) + (if hasOff then groups * (if offFp16 then 2 else 4) else 0)

/-- Engine-form plane bytes: scales and offsets are always 4 B. -/
def enginePlanes (codes groups : Nat) (hasOff : Bool) : Nat :=
  codes + groups * 4 + (if hasOff then groups * 4 else 0)

/-- Widening cost of a tensor: the extra 2 B per scale when the pack holds fp16 scales. -/
def widening (groups : Nat) (scFp16 : Bool) : Nat := if scFp16 then groups * 4 - groups * 2 else 0

/-- The check `dst_bytes == src_bytes + widen_here` holds whenever offsets are f32 (true for all
54 offset-carrying tensors of the current pack). -/
theorem engine_eq_pack_plus_widening (codes groups : Nat) (scFp16 hasOff : Bool) :
    enginePlanes codes groups hasOff =
      packPlanes codes groups scFp16 hasOff false + widening groups scFp16 := by
  unfold enginePlanes packPlanes widening
  cases scFp16 <;> cases hasOff <;> simp <;> omega

/-- If a future pack had fp16 OFFSETS, the same check would refuse it (widening counts scales
only), which is the conservative outcome. -/
theorem fp16_offsets_detected (codes groups : Nat) (hg : 0 < groups) :
    enginePlanes codes groups true ≠ packPlanes codes groups false true true + widening groups false := by
  unfold enginePlanes packPlanes widening; simp; omega

/-- Engine planes are never smaller than pack planes. -/
theorem engine_ge_pack (codes groups : Nat) (scFp16 hasOff offFp16 : Bool) :
    packPlanes codes groups scFp16 hasOff offFp16 ≤ enginePlanes codes groups hasOff := by
  unfold enginePlanes packPlanes
  cases scFp16 <;> cases hasOff <;> cases offFp16 <;> simp <;> omega

/-- Number of scale groups: `ne1 * (ne0 / group)`, and codes are `elems * bits / 8` bytes. -/
theorem groups_cover (ne0 ne1 g : Nat) (hd : ne0 % g = 0) : ne1 * (ne0 / g) * g = ne1 * ne0 := by
  rw [Nat.mul_assoc, Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)]

-- § contiguity

/-- Planes laid out contiguously in key order fill `[off, off + codes + scales + offsets)`. -/
theorem contiguous_span (off c s o : Nat) :
    let codesOff := off
    let scalesOff := codesOff + c
    let offsOff := scalesOff + s
    offsOff + o - off = c + s + o := by
  simp only; omega

-- § engine pool

/-- `align_up`. -/
def alignUp (n a : Nat) : Nat := (n + a - 1) / a * a

/-- The next destination offset is aligned and leaves room for the current tensor. -/
theorem next_dst (dst bytes : Nat) :
    dst + bytes ≤ alignUp (dst + bytes) 64 ∧ alignUp (dst + bytes) 64 % 64 = 0 := by
  unfold alignUp; omega

-- § activation kind

/-- The activation each weight type wants (VDT): 0 = Q8_0, 1 = Q8_K. -/
def actKind : String → Nat
  | "Q3_K" | "Q4_K" | "Q5_K" | "Q6_K" | "IQ4_XS" => 1
  | _ => 0

/-- The S-form fields the engine sees: (code bits, code bias, IQ4 codebook, has offset). -/
structure SForm where
  bits : Nat
  bias : Int
  iq4 : Bool
  offset : Bool
deriving DecidableEq

/-- S-form fields per type, from the MAPPINGS table. -/
def sform : String → SForm
  | "IQ4_NL" => ⟨4, 0, true, false⟩
  | "IQ4_XS" => ⟨4, 0, true, false⟩
  | "Q5_0" => ⟨8, -16, false, false⟩
  | "Q5_K" => ⟨8, 0, false, true⟩
  | _ => ⟨0, 0, false, false⟩

/-- IQ4_NL and IQ4_XS agree on every S-form field yet need different activations: act_kind must
be carried in the index. -/
theorem act_kind_not_derivable : sform "IQ4_NL" = sform "IQ4_XS" ∧ actKind "IQ4_NL" ≠ actKind "IQ4_XS" := by
  decide

/-- The source comment says Q5_0 and Q5_K differ only in `has_offset`; per MAPPINGS their code
bias differs too (-16 vs 0). Both differ in act_kind either way. -/
theorem q5_forms_differ_in_bias :
    (sform "Q5_0").bias ≠ (sform "Q5_K").bias ∧ (sform "Q5_0").offset ≠ (sform "Q5_K").offset ∧
      actKind "Q5_0" ≠ actKind "Q5_K" := by decide

-- § index format

/-- Columns written by the header line and the row format string. -/
def headerColumns : List String :=
  ["name", "file", "kind", "src_off", "src_bytes", "dst_off", "dst_bytes", "ne0", "ne1",
   "code_bits", "code_bias", "group_elems", "codebook", "has_offset", "codes_bytes",
   "scales_bytes", "offset_bytes", "scales_fp16", "act_kind"]

/-- Columns listed in the module docstring. -/
def docColumns : List String := headerColumns.take 18

/-- The row has 19 fields; the docstring documents 18 and omits act_kind. -/
theorem column_counts : headerColumns.length = 19 ∧ docColumns.length = 18 ∧
    headerColumns.getLast? = some "act_kind" := by decide

/-- The reported widening cost, 11,407,360 B, is 10.88 MiB. -/
theorem widening_mib : 1087 * 2 ^ 20 ≤ 100 * 11407360 ∧ 100 * 11407360 < 1088 * 2 ^ 20 := by decide

end Strata.PackIndex
