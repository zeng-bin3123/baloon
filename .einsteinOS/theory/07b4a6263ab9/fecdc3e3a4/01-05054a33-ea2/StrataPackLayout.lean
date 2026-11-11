/-!
# MODULE: strata_pack_layout
SOURCES: tools/strata_pack.py, tools/pack_layer.py, tools/pack_index.py, tools/canonical_xcheck.py
The Strata pack: the 1,382,400-byte expert blob with interleaved gate/up rows and separate code and
scale planes, the LSB-first code packing shared by every S-form, the canonical mapping registry, and
the engine-index rules (scale widening, BF16/F16 kinds, activation kind).

## Source semantics
* Expert blob planes in order: gate/up codes `2*640*640`, down codes `2560*160`, gate/up scales
  `2*640*40*2`, down scales `2560*10*2`; they sum to `3 * 460800 = 1382400` bytes.
* Gate and up rows interleave: row `r` of role `s` (gate 0, up 1) occupies slot `2r + s`.
* `pack_codes` stores `8 / bits` codes per byte, slot `k` at bit `k * bits`; `unpack_codes` inverts it.
* Each mapping has `group_elems` dividing `block_elems`; Q2_0 bias -1, Q3_K -4, Q4_0 -8, Q5_0 -16,
  Q6_K -32, Q8_0 -128.
* The engine widens fp16 scales to f32: `dst_bytes = src_bytes + ne1 * n_groups * 2` per fp16 plane.
* IQ4_NL and IQ4_XS have identical S-form fields yet different `vec_dot_type`, so the activation
  kind must be an explicit index column.

## Form template identifiers
`BLOB_BYTES` → `blobBytes`, `blob_offsets` → `offGuCodes`/`offDCodes`/`offGuScales`/`offDScales`,
gate/up slot → `guSlot`, `rebuild_role` code address → `guCodeAddr`, `pack_codes` → `packByte`,
`MAPPINGS` row → `SForm`, `VDT` → `actKind`, `pack_index` kind → `engineKind`.
-/

namespace Strata.Pack

-- § expert blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- Expert FFN width. -/
def FF : Nat := 640
/-- Experts per layer. -/
def NE : Nat := 512
/-- Code bytes of one gate/up row (2560 two-bit codes). -/
def rowBytesGU : Nat := H * 2 / 8
/-- Code bytes of one down row (640 two-bit codes). -/
def rowBytesD : Nat := FF * 2 / 8
/-- Q2_0 blocks per gate/up row. -/
def scalesGU : Nat := H / 64
/-- Q2_0 blocks per down row. -/
def scalesD : Nat := FF / 64
/-- Q2_0 bytes of one role of one expert. -/
def roleBytes : Nat := H * FF * 18 / 64
/-- One expert blob: three roles. -/
def blobBytes : Nat := 3 * roleBytes

/-- Row and block geometry: 640 B / 160 B of codes, 40 / 10 blocks per row. -/
theorem row_geometry : rowBytesGU = 640 ∧ rowBytesD = 160 ∧ scalesGU = 40 ∧ scalesD = 10 := by decide

/-- The architecture's blob size: 1,382,400 bytes. -/
theorem blobBytes_eq : blobBytes = 1382400 := by decide

/-- The four planes exactly tile the blob - the module-level assert of `pack_layer.py`. -/
theorem planes_sum :
    2 * FF * rowBytesGU + H * rowBytesD + 2 * FF * scalesGU * 2 + H * scalesD * 2 = blobBytes := by
  decide

/-- Gate/up codes start of plane. -/
def offGuCodes : Nat := 0
/-- Down codes start. -/
def offDCodes : Nat := offGuCodes + 2 * FF * rowBytesGU
/-- Gate/up scales start. -/
def offGuScales : Nat := offDCodes + H * rowBytesD
/-- Down scales start. -/
def offDScales : Nat := offGuScales + 2 * FF * scalesGU * 2

/-- `blob_offsets()`: 0, 819200, 1228800, 1331200, ending at the blob size. -/
theorem blob_offsets_values :
    offGuCodes = 0 ∧ offDCodes = 819200 ∧ offGuScales = 1228800 ∧ offDScales = 1331200 ∧
    offDScales + H * scalesD * 2 = blobBytes := by decide

/-- Interleaving: row `r` of role `s` (0 gate, 1 up) sits in slot `2r + s`. -/
def guSlot (r s : Nat) : Nat := 2 * r + s

/-- The interleave is a bijection onto `0..1279`: distinct (row, role) pairs get distinct slots. -/
theorem guSlot_injective {r1 s1 r2 s2 : Nat} (hs1 : s1 < 2) (hs2 : s2 < 2)
    (e : guSlot r1 s1 = guSlot r2 s2) : r1 = r2 ∧ s1 = s2 := by
  unfold guSlot at e; omega

/-- Every slot is in range for the 640 rows. -/
theorem guSlot_lt {r s : Nat} (hr : r < 640) (hs : s < 2) : guSlot r s < 1280 := by
  unfold guSlot; omega

/-- `rebuild_role`'s code address for gate/up block `b` of row `r`, role `s`. -/
def guCodeAddr (r s b : Nat) : Nat := offGuCodes + guSlot r s * 640 + b * 16
/-- `rebuild_role`'s scale address for gate/up block `b` of row `r`, role `s`. -/
def guScaleAddr (r s b : Nat) : Nat := offGuScales + (guSlot r s * 40 + b) * 2
/-- `rebuild_role`'s code address for down block `b` of row `r`. -/
def dCodeAddr (r b : Nat) : Nat := offDCodes + r * 160 + b * 16

/-- A gate/up block's 16 code bytes stay inside the gate/up code plane. -/
theorem guCodeAddr_in_plane {r s b : Nat} (hr : r < 640) (hs : s < 2) (hb : b < 40) :
    guCodeAddr r s b + 16 ≤ offDCodes := by
  have hv : offDCodes = 819200 := by decide
  unfold guCodeAddr guSlot offGuCodes; rw [hv]; omega

/-- A down block's 16 code bytes stay inside the down code plane. -/
theorem dCodeAddr_in_plane {r b : Nat} (hr : r < 2560) (hb : b < 10) :
    offDCodes ≤ dCodeAddr r b ∧ dCodeAddr r b + 16 ≤ offGuScales := by
  have h1 : offDCodes = 819200 := by decide
  have h2 : offGuScales = 1228800 := by decide
  unfold dCodeAddr; rw [h1, h2]; omega

/-- A gate/up block's 2-byte scale stays inside the gate/up scale plane. -/
theorem guScaleAddr_in_plane {r s b : Nat} (hr : r < 640) (hs : s < 2) (hb : b < 40) :
    offGuScales ≤ guScaleAddr r s b ∧ guScaleAddr r s b + 2 ≤ offDScales := by
  have h1 : offGuScales = 1228800 := by decide
  have h2 : offDScales = 1331200 := by decide
  unfold guScaleAddr guSlot; rw [h1, h2]; omega

/-- Distinct gate/up blocks have distinct code addresses: the relayout loses no block. -/
theorem guCodeAddr_injective {r1 s1 b1 r2 s2 b2 : Nat} (hs1 : s1 < 2) (hs2 : s2 < 2)
    (hb1 : b1 < 40) (hb2 : b2 < 40) (e : guCodeAddr r1 s1 b1 = guCodeAddr r2 s2 b2) :
    r1 = r2 ∧ s1 = s2 ∧ b1 = b2 := by
  unfold guCodeAddr guSlot offGuCodes at e; omega

/-- Expert `e`'s blob starts at `(layer * 512 + e) * blobBytes`; consecutive blobs abut. -/
theorem blob_base_step (layer e : Nat) :
    (layer * NE + (e + 1)) * blobBytes = (layer * NE + e) * blobBytes + blobBytes := by
  rw [Nat.add_succ, Nat.succ_mul]

/-- The 48-layer arena is 48 * 512 blobs. -/
theorem arena_bytes : 48 * NE * blobBytes = 33973862400 := by decide

-- § code packing

/-- `pack_codes` for one byte: slot `k` of `8 / bits` holds code `c k` at bit `k * bits`. -/
def packByte (bits : Nat) (c : Nat → Nat) : Nat :=
  (List.range (8 / bits)).foldl (fun acc k => acc ||| ((c k &&& (2 ^ bits - 1)) <<< (k * bits))) 0
/-- `unpack_codes` for one byte, slot `k`. -/
def unpackSlot (bits b k : Nat) : Nat := (b >>> (k * bits)) &&& (2 ^ bits - 1)

/-- 2-bit packing round-trips: every slot of every packed byte decodes to its code. -/
theorem pack2_roundtrip : ∀ c0, c0 < 4 → ∀ c1, c1 < 4 → ∀ c2, c2 < 4 → ∀ c3, c3 < 4 →
    let c := fun k => [c0, c1, c2, c3].getD k 0
    ∀ k, k < 4 → unpackSlot 2 (packByte 2 c) k = c k := by
  unfold packByte unpackSlot
  set_option maxRecDepth 20000 in decide

/-- 4-bit packing round-trips (low nibble first, as Q2_0's LSB-first convention extends). -/
theorem pack4_roundtrip : ∀ c0, c0 < 16 → ∀ c1, c1 < 16 →
    let c := fun k => [c0, c1].getD k 0
    ∀ k, k < 2 → unpackSlot 4 (packByte 4 c) k = c k := by
  unfold packByte unpackSlot
  set_option maxRecDepth 20000 in decide

/-- Packed code bytes: `elements * bits / 8`, so a Q2_0 tensor's code plane is a quarter of its
element count - the count the pack states on its own. -/
theorem code_plane_bytes (n : Nat) : n * 4 * 2 / 8 = n := by omega

-- § mapping registry

/-- The S-form fields of a `MAPPINGS` row. -/
structure SForm where
  form : String
  blockElems : Nat
  groupElems : Nat
  codebook : Option String
  codeBias : Int
  hasOffset : Bool
  deriving DecidableEq

/-- `MAPPINGS[t]` for the quantized source types. -/
def mapping : String → Option SForm
  | "Q2_0" => some ⟨"S2", 64, 64, none, -1, false⟩
  | "Q3_K" => some ⟨"S4", 256, 16, none, -4, false⟩
  | "Q4_0" => some ⟨"S4", 32, 32, none, -8, false⟩
  | "Q4_K" => some ⟨"S4", 256, 32, none, 0, true⟩
  | "Q5_0" => some ⟨"S8", 32, 32, none, -16, false⟩
  | "Q5_K" => some ⟨"S8", 256, 32, none, 0, true⟩
  | "Q6_K" => some ⟨"S8", 256, 16, none, -32, false⟩
  | "Q8_0" => some ⟨"S8", 32, 32, none, -128, false⟩
  | "IQ4_NL" => some ⟨"S4", 32, 32, some "IQ4NL", 0, false⟩
  | "IQ4_XS" => some ⟨"S4", 256, 32, some "IQ4NL", 0, false⟩
  | _ => none

/-- The quantized source types in the registry. -/
def quantTypes : List String :=
  ["Q2_0", "Q3_K", "Q4_0", "Q4_K", "Q5_0", "Q5_K", "Q6_K", "Q8_0", "IQ4_NL", "IQ4_XS"]

/-- In every mapping the scale group divides the source block: groups never straddle blocks. -/
theorem group_divides_block : ∀ t ∈ quantTypes, ∀ m, mapping t = some m → m.blockElems % m.groupElems = 0 := by
  decide

/-- Only the min-carrying K-quants (Q4_K, Q5_K) have an offset plane, and they have zero bias. -/
theorem offset_types : quantTypes.filter (fun t => (mapping t).any (·.hasOffset)) = ["Q4_K", "Q5_K"] := by
  decide

/-- `vec_dot_type` per source type: 0 = Q8_0 activations, 1 = Q8_K activations. -/
def actKind : String → Nat
  | "Q2_0" | "Q4_0" | "Q5_0" | "Q8_0" | "IQ4_NL" => 0
  | _ => 1

/-- IQ4_NL and IQ4_XS agree in every S-form field the engine carries except block size, yet need
different activation kinds - so `act_kind` cannot be re-derived from the S-form and must be its own
index column. -/
theorem iq4_needs_act_column :
    (mapping "IQ4_NL").map (fun m => (m.form, m.groupElems, m.codebook, m.codeBias, m.hasOffset)) =
    (mapping "IQ4_XS").map (fun m => (m.form, m.groupElems, m.codebook, m.codeBias, m.hasOffset)) ∧
    actKind "IQ4_NL" ≠ actKind "IQ4_XS" := by decide

-- § engine index

/-- Pack size of a quantized tensor's planes: codes, scales (2 or 4 B) and optional offsets. -/
def srcPlaneBytes (codes ne1 nGroups : Nat) (sc16 hasOff off16 : Bool) : Nat :=
  codes + ne1 * nGroups * (if sc16 then 2 else 4) +
    (if hasOff then ne1 * nGroups * (if off16 then 2 else 4) else 0)
/-- Engine size: scales and offsets always 4 B. -/
def dstPlaneBytes (codes ne1 nGroups : Nat) (hasOff : Bool) : Nat :=
  codes + ne1 * nGroups * 4 + (if hasOff then ne1 * nGroups * 4 else 0)
/-- The scale widening `pack_index` accounts for. -/
def widen (ne1 nGroups : Nat) (sc16 : Bool) : Nat := if sc16 then ne1 * nGroups * 2 else 0

/-- With f32 offsets (the case for all 54 offset-carrying tensors), the engine form is the pack
form plus exactly the scale widening - the check `pack_index.py` enforces. -/
theorem dst_eq_src_plus_widen (codes ne1 nGroups : Nat) (sc16 hasOff : Bool) :
    dstPlaneBytes codes ne1 nGroups hasOff =
      srcPlaneBytes codes ne1 nGroups sc16 hasOff false + widen ne1 nGroups sc16 := by
  unfold dstPlaneBytes srcPlaneBytes widen
  cases sc16 <;> cases hasOff <;> simp <;> omega

/-- Reading an fp16 scale plane as 4 B per scale doubles its size: the original loader bug, which
made the planes fail to add up and so was refused rather than silently mis-decoded. -/
theorem fp16_misread_doubles (ne1 nGroups : Nat) : ne1 * nGroups * 4 = 2 * (ne1 * nGroups * 2) := by
  omega

/-- Engine kinds. -/
inductive EngineKind
  | verbatim | bf16InF32 | f32Copy | f16InF32
  deriving DecidableEq

/-- `pack_index`'s kind selection from source type and `values_fp16`. -/
def engineKind (st : String) (valuesFp16 : Bool) : EngineKind :=
  if st = "BF16" then .bf16InF32
  else if st = "F16" ∧ ¬ valuesFp16 then .f16InF32
  else if st = "F32" then .f32Copy
  else .verbatim

/-- Engine bytes per element by kind (quantized tensors excluded): BF16/F16 promotions shrink back
to 2, F32 stays 4. -/
def engineElemBytes : EngineKind → Nat
  | .bf16InF32 => 2 | .f16InF32 => 2 | .f32Copy => 4 | .verbatim => 2

/-- A BF16 source is always re-rounded to 2 B, halving its pack (P32) footprint. -/
theorem bf16_halves (v : Bool) (elems : Nat) :
    engineKind "BF16" v = .bf16InF32 ∧ elems * 4 = 2 * (elems * engineElemBytes (engineKind "BF16" v)) := by
  constructor
  · rfl
  · simp [engineKind, engineElemBytes]; omega

/-- An F16 tensor already stored as 2 B is copied verbatim; one promoted to f32 needs conversion. -/
theorem f16_kinds : engineKind "F16" true = .verbatim ∧ engineKind "F16" false = .f16InF32 := by decide

/-- Engine-pool offsets are aligned to the manifest's 64-byte boundary. -/
theorem engine_align (n : Nat) : (n + 64 - 1) / 64 * 64 % 64 = 0 := Nat.mul_mod_left _ _

end Strata.Pack
