/-!
# MODULE: gdn_moe_qsa_layer
Composition of one decoder block: hyper-connection read, GDN or QSA mixer, router + shared expert + combine,
hyper-connection write. Models the arena arithmetic, plane layout checks, activation dispatch and stage gating.

## Source semantics
- A Q8_K buffer for n elements has (n/256) blocks of 292 bytes; every arena region is rounded up to 16 bytes.
- A quantized tensor's three planes (codes, scales, offsets) must add up to the tensor's byte size.
- code_bits = 2 or a non-Q8_K tensor takes the Q8_0 activation; otherwise Q8_K.
- `block_layer_pre` runs stages as selected by (stage_prefix, half); prefix k runs stages 0..k-1.
- The combine is sum_i w_i * part_i plus the shared expert output added plainly.
- RoPE reads row pos_base + pos of the table, so that must be < max_cells.

## Form template identifiers
q8k_bytes -> q8kBytes ; align_up16 -> align16 ; plane_ptrs checks -> planesOk ;
gemv_quantized dispatch -> actFor ; run0..run4 -> runStage ; moe_combine -> combine ;
dump_stride_floats -> dumpStride ; Cursor::take -> cursorTake
-/

namespace Strata.Layer

-- § arena arithmetic

/-- Bytes of a Q8_K buffer for `n` elements. -/
def q8kBytes (n : Nat) : Nat := (n / 256) * 292

/-- Round up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks a size. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- The result of alignment is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Alignment is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by unfold align16; omega

/-- Q8_K size for whole blocks. -/
theorem q8kBytes_blocks (k : Nat) : q8kBytes (256 * k) = 292 * k := by unfold q8kBytes; omega

/-- Total of a list of regions, each aligned. -/
def alignedTotal (l : List Nat) : Nat := (l.map align16).sum

/-- Aligned total is at least the raw total. -/
theorem sum_le_aligned (l : List Nat) : l.sum ≤ alignedTotal l := by
  unfold alignedTotal
  induction l with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.sum_cons]
    have := align16_ge a
    omega

/-- Aligned totals are additive over concatenation. -/
theorem alignedTotal_append (a b : List Nat) :
    alignedTotal (a ++ b) = alignedTotal a + alignedTotal b := by
  simp [alignedTotal, List.map_append, List.sum_append]

/-- Cursor take: the new used offset after reserving `bytes`. -/
def cursorTake (used bytes : Nat) : Nat := align16 (used + bytes)

/-- A take never moves the cursor backwards. -/
theorem cursorTake_mono (used bytes : Nat) : used ≤ cursorTake used bytes := by
  unfold cursorTake align16; omega

/-- After a take the cursor is aligned, so the next region starts aligned. -/
theorem cursorTake_aligned (used bytes : Nat) : cursorTake used bytes % 16 = 0 :=
  align16_mod _

-- § plane layout

/-- The loader-recorded planes describe the tensor. -/
def planesOk (codes scales offset bytes : Nat) (hasOffset : Bool) : Prop :=
  codes ≠ 0 ∧ scales ≠ 0 ∧ codes + scales + offset = bytes ∧ (hasOffset = true ↔ offset ≠ 0)

/-- The offset plane starts inside the tensor. -/
theorem planes_offset_start_le (c s o b : Nat) (h : Bool) (hp : planesOk c s o b h) :
    c + s ≤ b := by
  unfold planesOk at hp
  obtain ⟨_, _, h3, _⟩ := hp
  omega

/-- With no offset flag the planes are exactly codes plus scales. -/
theorem planes_no_offset (c s o b : Nat) (hp : planesOk c s o b false) : c + s = b := by
  unfold planesOk at hp
  obtain ⟨_, _, h3, h4⟩ := hp
  have : o = 0 := by
    by_cases ho : o = 0
    · exact ho
    · have := h4.mpr ho
      simp at this
  omega

-- § activation dispatch

/-- Which quantized image of the activation a weight reads. -/
inductive ActKind | q8_0 | q8k
  deriving DecidableEq

/-- The dispatch of `gemv_quantized`. -/
def actFor (codeBits : Nat) (wantsQ8k : Bool) : ActKind :=
  if codeBits = 2 ∨ wantsQ8k = false then .q8_0 else .q8k

/-- Two-bit forms always use Q8_0. -/
theorem actFor_two_bits (w : Bool) : actFor 2 w = .q8_0 := by
  simp [actFor]

/-- A tensor that does not want Q8_K uses Q8_0 whatever its width. -/
theorem actFor_legacy (b : Nat) : actFor b false = .q8_0 := by
  simp [actFor]

/-- Wide Q8_K tensors use Q8_K. -/
theorem actFor_q8k (b : Nat) (hb : b ≠ 2) : actFor b true = .q8k := by
  simp [actFor, hb]

-- § stage gating

/-- Whether stage `s` (0..4) runs for the given prefix/half. -/
def runStage (sp half s : Nat) : Bool :=
  if sp > 0 then decide (sp ≥ s + 1)
  else if s ≤ 2 then decide (half = 0 ∨ half = 1) else decide (half = 0 ∨ half = 2)

/-- Default arguments run every stage. -/
theorem runStage_default (s : Nat) : runStage 0 0 s = true := by
  unfold runStage
  by_cases h : s ≤ 2 <;> simp [h]

/-- A positive prefix k runs exactly stages below k. -/
theorem runStage_prefix (sp half s : Nat) (hsp : 0 < sp) :
    runStage sp half s = true ↔ s < sp := by
  unfold runStage
  simp [hsp] <;> omega

/-- Half 1 is the mixer: stages 0..2. -/
theorem runStage_half1 (s : Nat) : runStage 0 1 s = true ↔ s ≤ 2 := by
  unfold runStage
  by_cases h : s ≤ 2 <;> simp [h]

/-- Half 2 is the FFN front and router: stages 3..4. -/
theorem runStage_half2 (s : Nat) : runStage 0 2 s = true ↔ 3 ≤ s := by
  unfold runStage
  by_cases h : s ≤ 2 <;> simp [h] <;> omega

-- § MoE combine

/-- sum_i w_i * part_i + shared. -/
def combine (ws parts : List Int) (shared : Int) : Int :=
  (List.zipWith (· * ·) ws parts).sum + shared

/-- With no experts the combine is the shared output. -/
theorem combine_nil (s : Int) : combine [] [] s = s := by
  simp [combine]

/-- Combine unfolds one expert at a time. -/
theorem combine_cons (w p : Int) (ws ps : List Int) (s : Int) :
    combine (w :: ws) (p :: ps) s = w * p + combine ws ps s := by
  simp [combine, List.zipWith_cons_cons, List.sum_cons, Int.add_assoc]

-- § position and dump bounds

/-- RoPE row index is inside the table. -/
def ropeInRange (posBase pos maxCells : Int) : Prop := 0 ≤ posBase ∧ posBase + pos < maxCells

/-- Inside the table implies the plain position is too, for nonnegative positions. -/
theorem ropeInRange_pos_lt (pb p mc : Int) (hp : 0 ≤ p) (h : ropeInRange pb p mc) : p < mc := by
  unfold ropeInRange at h
  omega

/-- Floats per layer in the oracle dump. -/
def dumpStride (nEmbd hc nHead headDim nKv : Nat) : Nat :=
  2 * nEmbd + 2 * hc + nHead * headDim + 5 * (nKv * headDim) + 8

/-- The dump stride covers both halves' block outputs. -/
theorem dumpStride_ge (a b c d e : Nat) : 2 * a ≤ dumpStride a b c d e := by
  unfold dumpStride; omega

end Strata.Layer
