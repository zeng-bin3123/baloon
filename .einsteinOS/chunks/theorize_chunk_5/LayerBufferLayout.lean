/-!
# MODULE: layer_buffer_layout
Arena carving, quantized-plane consistency and stage selection of the composed GDN/QSA/MoE block.

## Source semantics
* Every carved region is 16-byte aligned; the cursor only moves forward and stays aligned.
* A quantized tensor's planes (codes, scales, offsets) must add up to the tensor's byte size, and
  the offset flag must match the offset plane; a plane width mistake (f32 vs fp16 scales) breaks the sum.
* `block_layer_pre` runs stages 0..k-1 when `stage_prefix = k > 0`, otherwise stages according to `half`
  (0 = all, 1 = mixer, 2 = FFN front + router).
* A Q8_K buffer for `n` elements takes `n/256` blocks of 292 bytes.

## Form template identifiers
* `align_up16`, `Cursor::take` → `align16`, `takeStep`
* `plane_ptrs` check → `planesOk`
* `run0..run4` → `runFlags`
* `q8k_bytes` → `q8kBytes`
-/

namespace Strata.LayerBuffers

-- § 16-byte alignment and the cursor

/-- Round up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Aligned values are multiples of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Alignment never shrinks. -/
theorem le_align16 (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- Alignment adds fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- Cursor advance after taking `bytes`. -/
def takeStep (used bytes : Nat) : Nat := align16 (used + bytes)

/-- The cursor never moves backwards. -/
theorem takeStep_mono (used bytes : Nat) : used ≤ takeStep used bytes := by
  unfold takeStep
  have := le_align16 (used + bytes)
  omega

/-- A taken region fits entirely below the new cursor. -/
theorem takeStep_covers (used bytes : Nat) : used + bytes ≤ takeStep used bytes :=
  le_align16 _

/-- An aligned cursor stays aligned, so every region start is aligned. -/
theorem takeStep_aligned (used bytes : Nat) : takeStep used bytes % 16 = 0 :=
  align16_mod _

-- § Q8_K sizing

/-- Bytes of a Q8_K buffer for `n` elements (a multiple of 256). -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- An empty activation needs no bytes. -/
theorem q8kBytes_zero : q8kBytes 0 = 0 := rfl

/-- Q8_K is monotone in the element count. -/
theorem q8kBytes_mono (a b : Nat) (h : a ≤ b) : q8kBytes a ≤ q8kBytes b := by
  unfold q8kBytes
  exact Nat.mul_le_mul_right _ (Nat.div_le_div_right h)

-- § quantized plane consistency

/-- The loader's recorded planes describe the tensor. -/
def planesOk (codes scales off bytes : Nat) (hasOffset : Bool) : Bool :=
  decide (codes ≠ 0) && decide (scales ≠ 0) && decide (codes + scales + off = bytes) &&
    (hasOffset == decide (off ≠ 0))

/-- Consistent planes pass. -/
theorem planesOk_intro (codes scales off bytes : Nat) (h0 : codes ≠ 0) (h1 : scales ≠ 0)
    (hs : codes + scales + off = bytes) :
    planesOk codes scales off bytes (decide (off ≠ 0)) = true := by
  unfold planesOk
  simp [h0, h1, hs]

/-- Doubling the scale plane (f32 width for fp16 data) is caught by the sum check. -/
theorem plane_width_mismatch (codes scales off : Nat) (h : 0 < scales) :
    codes + 2 * scales + off ≠ codes + scales + off := by
  omega

-- § stage selection

/-- (run0, run1, run2, run3, run4) for a `stage_prefix` and a `half`. -/
def runFlags (sp half : Nat) : Bool × Bool × Bool × Bool × Bool :=
  (if sp > 0 then decide (sp ≥ 1) else decide (half = 0 ∨ half = 1),
   if sp > 0 then decide (sp ≥ 2) else decide (half = 0 ∨ half = 1),
   if sp > 0 then decide (sp ≥ 3) else decide (half = 0 ∨ half = 1),
   if sp > 0 then decide (sp ≥ 4) else decide (half = 0 ∨ half = 2),
   if sp > 0 then decide (sp ≥ 5) else decide (half = 0 ∨ half = 2))

/-- The default call (no prefix, half 0) runs every stage. -/
theorem runFlags_default : runFlags 0 0 = (true, true, true, true, true) := by decide

/-- Prefix 5 is the whole pre-half, whatever `half` says. -/
theorem runFlags_prefix5 (h : Nat) : runFlags 5 h = (true, true, true, true, true) := by
  simp [runFlags]

/-- Half 1 is the mixer: stages 0..2 only. -/
theorem runFlags_half1 : runFlags 0 1 = (true, true, true, false, false) := by decide

/-- Half 2 skips the mixer and runs the FFN front and router. -/
theorem runFlags_half2 : runFlags 0 2 = (false, false, false, true, true) := by decide

/-- Prefix 3 is exactly the mixer. -/
theorem runFlags_prefix3 (h : Nat) : runFlags 3 h = (true, true, true, false, false) := by
  simp [runFlags]

end Strata.LayerBuffers
