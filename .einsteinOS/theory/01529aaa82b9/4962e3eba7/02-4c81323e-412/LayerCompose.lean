/-!
# MODULE: layer_compose
Arithmetic and dispatch contract of `src/core/layer.cpp`: the GDN / QSA / MoE layer composition, its arena
carving (16-byte cursor), the plane-layout check of quantized tensors, the activation-format dispatch, the
stage-pfx selection of `block_layer_pre`, and the dump-slot layout.

## Source semantics
* Every buffer is carved from an arena with `used := align16 (used + count*size)`; `*_bytes` functions sum the
  same parts, so the init total equals the byte count and the 256-byte slack covers the extra rounding.
* A quantized tensor's three planes (codes | scales | offsets) must add up to the tensor and the offset plane
  exists iff `has_offset`.
* The activation a quantized weight reads is decided per TENSOR: native, S2/Q8_0, legacy Q8_0, or Q8_K.
* `stage_prefix = k > 0` runs stages `0..k-1`; otherwise `half` selects mixer (1), FFN-front+router (2) or all (0).
* RoPE rows read are `pos_base + pos`, which must be inside the `max_cells`-row table.

## Form template identifiers
* `alignUp16`  <- `align_up16`
* `q8kBytes`   <- `q8k_bytes`
* `PlaneSizes` <- `WeightRef{codes_bytes,scales_bytes,offset_bytes,bytes}` / `plane_ptrs`
* `dispatch`   <- `gemv_quantized`
* `runStage`   <- `block_layer_pre` run0..run4
* `takeAll16`  <- `Cursor::take` sequence
-/

namespace Strata.LayerCompose

-- § alignment

/-- 16-byte round-up used by `align_up16`. -/
def alignUp16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never loses bytes. -/
theorem alignUp16_ge (n : Nat) : n ≤ alignUp16 n := by
  unfold alignUp16; omega

/-- The rounded value is 16-aligned. -/
theorem alignUp16_mod (n : Nat) : alignUp16 n % 16 = 0 := by
  unfold alignUp16; omega

/-- Rounding adds at most 15 bytes. -/
theorem alignUp16_le (n : Nat) : alignUp16 n ≤ n + 15 := by
  unfold alignUp16; omega

/-- Rounding is idempotent. -/
theorem alignUp16_idem (n : Nat) : alignUp16 (alignUp16 n) = alignUp16 n := by
  unfold alignUp16; omega

-- § Q8_K sizes

/-- Bytes of a Q8_K buffer of `n` elements: one 292-byte block per 256 elements. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- `b` whole blocks occupy `292*b` bytes. -/
theorem q8k_blocks (b : Nat) : q8kBytes (256 * b) = 292 * b := by
  unfold q8kBytes; omega

/-- A Q8_K image is never smaller than the activation it encodes (for a multiple of 256). -/
theorem q8k_ge (n : Nat) (h : n % 256 = 0) : n ≤ q8kBytes n := by
  unfold q8kBytes; omega

-- § plane layout

/-- Recorded plane sizes of a quantized tensor. -/
structure PlaneSizes where
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat

/-- The check of `plane_ptrs`: both mandatory planes non-empty and the three add up to the tensor. -/
def PlaneSizes.ok (p : PlaneSizes) : Prop :=
  p.codes ≠ 0 ∧ p.scales ≠ 0 ∧ p.codes + p.scales + p.offset = p.bytes

/-- Valid planes tile the tensor exactly: the offset plane starts at `codes + scales` and ends at `bytes`. -/
theorem planes_partition (p : PlaneSizes) (h : p.ok) :
    p.codes + p.scales + p.offset = p.bytes := by
  unfold PlaneSizes.ok at h
  obtain ⟨_, _, h3⟩ := h
  exact h3

/-- Valid planes: the scale plane starts strictly inside the tensor. -/
theorem planes_scales_inside (p : PlaneSizes) (h : p.ok) : p.codes < p.bytes := by
  unfold PlaneSizes.ok at h
  obtain ⟨_, h2, h3⟩ := h
  omega

/-- Valid planes: the codes plane fits in the tensor. -/
theorem planes_codes_le (p : PlaneSizes) (h : p.ok) : p.codes + p.scales ≤ p.bytes := by
  unfold PlaneSizes.ok at h
  obtain ⟨_, _, h3⟩ := h
  omega

-- § activation dispatch

/-- Which path a projection takes. -/
inductive Dispatch
  | native
  | s2Q80
  | legacyQ80
  | q8kSplit
  deriving DecidableEq

/-- Decision of `gemv_quantized`. -/
def dispatch (native : Bool) (codeBits : Nat) (wantsQ8k : Bool) : Dispatch :=
  if native = true then .native
  else if codeBits = 2 then .s2Q80
  else if wantsQ8k = true then .q8kSplit
  else .legacyQ80

/-- A natively served tensor always takes the native path. -/
theorem dispatch_native (bits : Nat) (w : Bool) : dispatch true bits w = .native := by
  simp [dispatch]

/-- A 2-bit canonical tensor is always S2 with the Q8_0 activation, whatever `wants_q8k` says. -/
theorem dispatch_s2 (w : Bool) : dispatch false 2 w = .s2Q80 := by
  simp [dispatch]

/-- A K-quant (not 2-bit, wants Q8_K) takes the Q8_K split kernel. -/
theorem dispatch_q8k (bits : Nat) (h : bits ≠ 2) : dispatch false bits true = .q8kSplit := by
  simp [dispatch, h]

/-- A legacy 4/8-bit form takes Q8_0. -/
theorem dispatch_legacy (bits : Nat) (h : bits ≠ 2) : dispatch false bits false = .legacyQ80 := by
  simp [dispatch, h]

-- § stage pfx selection

/-- `run_i` of `block_layer_pre`: pfx mode when `pfx > 0`, else by `half`. -/
def runStage (i pfx half : Nat) : Bool :=
  if 0 < pfx then decide (i < pfx)
  else if i ≤ 2 then (half == 0 || half == 1)
  else (half == 0 || half == 2)

/-- In pfx mode stage `i` runs exactly when `i < pfx`. -/
theorem runStage_prefix (i p h : Nat) (hp : 0 < p) : runStage i p h = decide (i < p) := by
  simp [runStage, hp]

/-- Default call (`half = 0`, `stage_prefix = 0`) runs all five stages. -/
theorem runStage_all :
    runStage 0 0 0 = true ∧ runStage 1 0 0 = true ∧ runStage 2 0 0 = true ∧
    runStage 3 0 0 = true ∧ runStage 4 0 0 = true := by
  decide

/-- `half = 1` runs exactly the mixer stages 0..2. -/
theorem runStage_half1 :
    runStage 0 0 1 = true ∧ runStage 1 0 1 = true ∧ runStage 2 0 1 = true ∧
    runStage 3 0 1 = false ∧ runStage 4 0 1 = false := by
  decide

/-- `half = 2` runs exactly the FFN front and router (stages 3, 4). -/
theorem runStage_half2 :
    runStage 0 0 2 = false ∧ runStage 1 0 2 = false ∧ runStage 2 0 2 = false ∧
    runStage 3 0 2 = true ∧ runStage 4 0 2 = true := by
  decide

/-- Prefix sets are nested: whatever pfx `k` runs, pfx `k+1` runs too. -/
theorem runStage_prefix_mono (i k h : Nat) (hk : 0 < k) (hr : runStage i k h = true) :
    runStage i (k + 1) h = true := by
  rw [runStage_prefix i k h hk] at hr
  rw [runStage_prefix i (k + 1) h (by omega)]
  simp at hr ⊢
  omega

-- § RoPE table bound

/-- Row index read by RoPE is non-negative and below the table height once the guard of `qsa_layer` passed. -/
theorem rope_row_in_table (posBase pos maxCells : Int) (h0 : 0 ≤ posBase) (h1 : 0 ≤ pos)
    (h2 : posBase + pos < maxCells) : 0 ≤ posBase + pos ∧ posBase + pos < maxCells := by
  omega

-- § arena carving

/-- Sum of the 16-aligned sizes: what every `*_bytes` function returns. -/
def alignedTotal16 (l : List Nat) : Nat := (l.map alignUp16).sum

/-- The cursor sequence `used := align16 (used + b)`. -/
def takeAll16 : Nat → List Nat → Nat
  | used, [] => used
  | used, b :: t => takeAll16 (alignUp16 (used + b)) t

/-- Carving from an aligned start equals the sum of aligned parts: init total = `*_bytes`. -/
theorem takeAll16_eq (l : List Nat) :
    ∀ used, used % 16 = 0 → takeAll16 used l = used + alignedTotal16 l := by
  induction l with
  | nil => intro used _; simp [takeAll16, alignedTotal16]
  | cons b t ih =>
    intro used hu
    have h1 : alignUp16 (used + b) = used + alignUp16 b := by unfold alignUp16; omega
    have h2 : alignUp16 (used + b) % 16 = 0 := by unfold alignUp16; omega
    simp only [takeAll16, alignedTotal16, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih _ h2, h1]
    omega

/-- The init sequence from offset 0 consumes exactly `alignedTotal16`. -/
theorem init_total_eq_bytes (l : List Nat) : takeAll16 0 l = alignedTotal16 l := by
  have h := takeAll16_eq l 0 (by omega)
  omega

/-- Aligned total is at least the raw sum. -/
theorem alignedTotal16_ge (l : List Nat) : l.sum ≤ alignedTotal16 l := by
  induction l with
  | nil => simp [alignedTotal16]
  | cons a t ih =>
    have h := alignUp16_ge a
    simp only [alignedTotal16, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- Aligned total is 16-aligned. -/
theorem alignedTotal16_mod (l : List Nat) : alignedTotal16 l % 16 = 0 := by
  induction l with
  | nil => simp [alignedTotal16]
  | cons a t ih =>
    have h := alignUp16_mod a
    simp only [alignedTotal16, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- Per-part rounding costs at most 15 bytes each. -/
theorem alignedTotal16_le (l : List Nat) : alignedTotal16 l ≤ l.sum + 15 * l.length := by
  induction l with
  | nil => simp [alignedTotal16]
  | cons a t ih =>
    have h := alignUp16_le a
    simp only [alignedTotal16, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    omega

/-- The byte count `align16(sum) + 256` of `qsa_state_bytes` / `qsa_buffers_bytes` always covers the
per-take rounding of the matching `*_init` provided there are at most 17 takes. -/
theorem carve_fits_with_slack (l : List Nat) (hlen : l.length ≤ 17) :
    takeAll16 0 l ≤ alignUp16 l.sum + 256 := by
  rw [init_total_eq_bytes]
  have h1 := alignedTotal16_le l
  have h2 := alignUp16_ge l.sum
  omega

/-- A cursor `take` hands out the current offset and the next offset is past the requested bytes. -/
theorem cursor_take_disjoint (used count size : Nat) :
    used + count * size ≤ alignUp16 (used + count * size) := alignUp16_ge _

-- § dump slots

/-- Total floats per layer of the half-level dump (`dump_stride_floats`). -/
def dumpStride (n hc nhhd nvk : Nat) : Nat := 2 * n + 2 * hc + nhhd + 5 * nvk + 8

/-- The last write of `qsa_layer` (the post-RoPE `kcur`) ends exactly at the stride. -/
theorem dump_stride_covers (n hc nhhd nvk : Nat) :
    (2 * n + 2 * hc + nhhd + 2 * nvk + 8) + 2 * nvk + nvk = dumpStride n hc nhhd nvk := by
  unfold dumpStride; omega

/-- FINDING: with `nvk >= 2`, the `step[0..3]` write at `vs + nvk/2 + 4` (length 4) extends past `base = vs + 8`,
so it overlaps the `k_pool` pfx written at `base`; the two diagnostic copies are not disjoint. -/
theorem dump_step_overlaps_kpool (vs nvk : Nat) (h : 2 ≤ nvk) :
    ¬ (vs + nvk / 2 + 4 + 4 ≤ vs + 8) := by
  omega

end Strata.LayerCompose
