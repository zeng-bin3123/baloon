/-!
# MODULE: quantized_layer_composition
SOURCES: src/core/layer.cpp
The GDN / QSA / MoE layer composition: Q8_K/Q8_0 activation sizing, the three-plane layout check of a
quantized weight (`plane_ptrs`), the per-tensor activation dispatch (`gemv_quantized`), the 16-byte
aligned arena carving (`gdn_buffers_init`, `Cursor`), and the stage-selection logic of `block_layer_pre`.

## Source semantics
* A Q8_K image of `n` floats is `n/256` blocks of 292 bytes; a Q8_0 image is `n/32` blocks of 34 bytes.
* A quantized tensor is codes ++ scales ++ offsets; the three sizes recorded by the loader must add up to
  the tensor, codes and scales are non-empty, and `has_offset` iff the offset plane is non-empty.
* The activation a quantized weight wants is a property of the TENSOR: S2 (`code_bits = 2`) wants Q8_0;
  any other form wants Q8_K iff its manifest `source_type` says so (`wants_q8k`), otherwise Q8_0.
* Every scratch region is carved at a 16-byte aligned offset; the carve's total is the sum of the
  aligned parts and is never smaller than the raw sum.
* `block_layer_pre` runs stage k (0..4) when `stage_prefix > k` if a prefix is given, otherwise by
  `half` (0 = all; 1 = mixer stages 0..2; 2 = FFN-front stages 3..4).
* The MoE `k` must lie in 1..64.

## Form template identifiers
q8kBytes ↔ q8k_bytes · q8_0Bytes ↔ `(n/32)*34` · WeightRef/PlanesOk ↔ WeightRef + plane_ptrs ·
wantedAct ↔ gemv_quantized dispatch · align16/carve/carveTotal ↔ Cursor::take / gdn_buffers_init ·
runStage ↔ run0..run4 · kValid ↔ `k < 1 || k > 64` guard
-/

namespace QuantizedLayerComposition

-- § activation image sizes

/-- Bytes of a Q8_K activation image (`q8k_bytes`). -/
def q8kBytes (n : Nat) : Nat := (n / 256) * 292

/-- Bytes of a Q8_0 activation image. -/
def q8_0Bytes (n : Nat) : Nat := (n / 32) * 34

/-- A whole number of 256-blocks has exactly 292 bytes per block. -/
theorem q8kBytes_blocks (b : Nat) : q8kBytes (256 * b) = 292 * b := by
  unfold q8kBytes
  rw [Nat.mul_div_cancel_left b (by decide)]
  exact Nat.mul_comm b 292

/-- A whole number of 32-blocks has exactly 34 bytes per block. -/
theorem q8_0Bytes_blocks (b : Nat) : q8_0Bytes (32 * b) = 34 * b := by
  unfold q8_0Bytes
  rw [Nat.mul_div_cancel_left b (by decide)]
  exact Nat.mul_comm b 34

/-- The model width 2560 gives the concrete Q8_K/Q8_0 sizes used by every layer buffer. -/
theorem n_embd_images : q8kBytes 2560 = 2920 ∧ q8_0Bytes 2560 = 2720 := by
  decide

-- § the three planes of a quantized tensor

/-- The fields of a `WeightRef` the plane check reads. -/
structure WeightRef where
  codeBits : Nat
  codesBytes : Nat
  scalesBytes : Nat
  offsetBytes : Nat
  bytes : Nat
  hasOffset : Bool

/-- What `plane_ptrs` accepts: an S2/S4/S8 width, non-empty codes and scales, planes that add up to the
tensor, and `has_offset` agreeing with the offset plane. -/
structure PlanesOk (r : WeightRef) : Prop where
  width : r.codeBits = 2 ∨ r.codeBits = 4 ∨ r.codeBits = 8
  codes_pos : 0 < r.codesBytes
  scales_pos : 0 < r.scalesBytes
  sum : r.codesBytes + r.scalesBytes + r.offsetBytes = r.bytes
  offset_flag : (r.hasOffset = true) ↔ r.offsetBytes ≠ 0

/-- Byte offset of the scale plane. -/
def scalesStart (r : WeightRef) : Nat := r.codesBytes

/-- Byte offset of the offset plane. -/
def offsetStart (r : WeightRef) : Nat := r.codesBytes + r.scalesBytes

/-- The scale plane begins strictly inside the tensor and after the codes. -/
theorem scales_inside (r : WeightRef) (h : PlanesOk r) : scalesStart r < r.bytes := by
  have hs := h.sum
  have hp := h.scales_pos
  unfold scalesStart
  omega

/-- The offset plane ends exactly at the end of the tensor: nothing is read past it. -/
theorem offset_ends_at_tensor_end (r : WeightRef) (h : PlanesOk r) :
    offsetStart r + r.offsetBytes = r.bytes := by
  unfold offsetStart
  exact h.sum

/-- The three planes are disjoint and ordered: codes, then scales, then offsets. -/
theorem planes_ordered (r : WeightRef) (h : PlanesOk r) :
    r.codesBytes ≤ scalesStart r ∧ scalesStart r + r.scalesBytes = offsetStart r ∧
      offsetStart r ≤ r.bytes := by
  have hs := h.sum
  refine ⟨Nat.le_refl _, rfl, ?_⟩
  unfold offsetStart
  omega

/-- A tensor without the offset flag has an empty offset plane, so codes and scales fill it. -/
theorem no_offset_two_planes (r : WeightRef) (h : PlanesOk r) (hf : r.hasOffset = false) :
    r.codesBytes + r.scalesBytes = r.bytes := by
  have hz : r.offsetBytes = 0 := by
    cases hb : decide (r.offsetBytes = 0) with
    | true => exact of_decide_eq_true hb
    | false =>
      have hne : r.offsetBytes ≠ 0 := of_decide_eq_false hb
      have := h.offset_flag.mpr hne
      rw [hf] at this
      exact absurd this (by decide)
  have hs := h.sum
  omega

-- § activation dispatch

/-- The two canonical quantized activation formats. -/
inductive ActKind where
  | q8_0
  | q8_K
  deriving DecidableEq, Repr

/-- Which activation image a quantized weight consumes (`gemv_quantized`). -/
def wantedAct (codeBits : Nat) (wantsQ8K : Bool) : ActKind :=
  if codeBits = 2 then .q8_0 else if wantsQ8K then .q8_K else .q8_0

/-- An S2 tensor always takes Q8_0, whatever its manifest says. -/
theorem s2_takes_q8_0 (w : Bool) : wantedAct 2 w = .q8_0 := by
  simp [wantedAct]

/-- A non-S2 K-quant takes Q8_K. -/
theorem kquant_takes_q8K (c : Nat) (h : c ≠ 2) : wantedAct c true = .q8_K := by
  simp [wantedAct, h]

/-- A legacy form (Q4_0/Q5_0/Q8_0/IQ4_NL) takes Q8_0 regardless of its width: the width cannot decide. -/
theorem legacy_takes_q8_0 (c : Nat) : wantedAct c false = .q8_0 := by
  by_cases h : c = 2 <;> simp [wantedAct, h]

/-- Two tensors of the same width can still want different activations: `code_bits` alone is not the rule. -/
theorem width_does_not_decide : wantedAct 4 true ≠ wantedAct 4 false := by
  decide

-- § 16-byte aligned carving

/-- Round up to a multiple of 16 (`(v + 15) & ~15`). -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never shrinks a region. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- The rounded size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Rounding up wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- The start offsets the carve assigns to `parts`, beginning at `base`. -/
def carve : List Nat → Nat → List Nat
  | [], _ => []
  | p :: ps, base => base :: carve ps (base + align16 p)

/-- The bytes the carve consumes. -/
def carveTotal : List Nat → Nat
  | [] => 0
  | p :: ps => align16 p + carveTotal ps

/-- The raw byte sum of the parts. -/
def rawSum : List Nat → Nat
  | [] => 0
  | p :: ps => p + rawSum ps

/-- One offset per part. -/
theorem carve_length : ∀ (ps : List Nat) (a : Nat), (carve ps a).length = ps.length
  | [], _ => rfl
  | _ :: ps, a => by
    simp only [carve, List.length_cons]
    rw [carve_length ps]

/-- From an aligned base, every carved offset is 16-byte aligned. -/
theorem carve_aligned :
    ∀ (ps : List Nat) (a : Nat), a % 16 = 0 → ∀ x ∈ carve ps a, x % 16 = 0
  | [], _, _, x, hx => by simp [carve] at hx
  | p :: ps, a, ha, x, hx => by
    simp only [carve, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact ha
    · have hp := align16_mod p
      exact carve_aligned ps (a + align16 p) (by omega) x hx

/-- Every carved offset lies inside `[a, a + total]`: the arena of `carveTotal` bytes holds them all. -/
theorem carve_in_arena :
    ∀ (ps : List Nat) (a : Nat), ∀ x ∈ carve ps a, a ≤ x ∧ x ≤ a + carveTotal ps
  | [], _, x, hx => by simp [carve] at hx
  | p :: ps, a, x, hx => by
    simp only [carve, List.mem_cons] at hx
    simp only [carveTotal]
    rcases hx with rfl | hx
    · omega
    · have ih := carve_in_arena ps (a + align16 p) x hx
      omega

/-- The aligned total covers the raw sum of the parts (`gdn_buffers_bytes` is an upper bound). -/
theorem carveTotal_ge_rawSum : ∀ ps : List Nat, rawSum ps ≤ carveTotal ps
  | [] => Nat.le_refl 0
  | p :: ps => by
    simp only [rawSum, carveTotal]
    have h1 := align16_ge p
    have h2 := carveTotal_ge_rawSum ps
    omega

/-- The carve total is itself 16-byte aligned. -/
theorem carveTotal_aligned : ∀ ps : List Nat, carveTotal ps % 16 = 0
  | [] => rfl
  | p :: ps => by
    simp only [carveTotal]
    have h1 := align16_mod p
    have h2 := carveTotal_aligned ps
    omega

-- § stage selection in `block_layer_pre`

/-- Whether stage `k` (0..4) runs for the given `half` and `stage_prefix`. -/
def runStage (k half pfx : Nat) : Bool :=
  if pfx > 0 then decide (pfx ≥ k + 1)
  else if k ≤ 2 then (half == 0 || half == 1) else (half == 0 || half == 2)

/-- The default call (`half = 0`, no prefix) runs every stage. -/
theorem default_runs_all : ∀ k : Fin 5, runStage k.val 0 0 = true := by
  decide

/-- The two halves together cover every stage, and neither runs the other's. -/
theorem halves_partition :
    ∀ k : Fin 5, (runStage k.val 1 0 || runStage k.val 2 0) = true ∧
      (runStage k.val 1 0 && runStage k.val 2 0) = false := by
  decide

/-- A prefix `p` runs exactly the stages below `p`, regardless of `half`. -/
theorem prefix_runs_below (k p half : Nat) (hp : 0 < p) : runStage k half p = decide (k < p) := by
  unfold runStage
  simp only [hp, ite_true]
  exact decide_eq_decide.mpr (by omega)

/-- Prefix 5 runs the same stages as the full call, which is what the sweep relies on. -/
theorem prefix5_eq_full : ∀ k : Fin 5, runStage k.val 0 5 = runStage k.val 0 0 := by
  decide

-- § MoE k

/-- The `k` the MoE halves accept. -/
def kValid (k : Int) : Prop := 1 ≤ k ∧ k ≤ 64

/-- The artifact's top-10 routing is a valid `k`. -/
theorem k10_valid : kValid 10 := by
  unfold kValid; omega

/-- A valid `k` gives an id/weight buffer of at most 256 bytes each. -/
theorem k_buffer_bound (k : Int) (h : kValid k) : k * 4 ≤ 256 := by
  unfold kValid at h; omega

end QuantizedLayerComposition
