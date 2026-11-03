/-!
# MODULE: layer_buffer_layout
Arena carving and structural checks of the composed layer (`layer.cpp`): 16-byte aligned buffer carving, Q8_K
sizing, the three-plane tensor consistency check, stage-selection flags of `block_layer_pre`, and the
GDN/QSA layer interleave.

## Source semantics
- Every buffer in a layer arena is rounded up to 16 bytes; a cursor never moves backwards.
- A Q8_K buffer holds `n/256` blocks of 292 bytes.
- A quantized tensor's codes + scales + offsets planes must add up to the tensor's byte count, and the offset
  flag must agree with the offset plane size.
- QSA layers are those with `layer % 4 == 3`: 12 of 48, leaving 36 GDN layers.
- `stage_prefix = k` runs stages `0..k-1`; `half = 0` runs everything, `1` the mixer, `2` the FFN front + router.

## Form template identifiers
- `align16`, `cursorTake` <- `(v + 15) & ~15`, `Cursor::take`
- `q8kBytes`              <- `q8k_bytes`
- `planesOk`              <- `plane_ptrs` consistency checks
- `runStage`, `runHalf`   <- `run0..run4` in `block_layer_pre`
- `isQsa`                 <- `is_qsa_layer`
-/

namespace Strata.Layer

/-- Round up to 16 (the `(v + 15) & ~15` of the carve loops). -/
def align16 (v : Nat) : Nat := (v + 15) / 16 * 16

/-- Cursor advance for `count` elements of `size` bytes. -/
def cursorTake (used count size : Nat) : Nat := align16 (used + count * size)

/-- Bytes of a Q8_K buffer of `n` elements. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- Plane consistency of a quantized tensor. -/
def planesOk (codes scales off total : Nat) (hasOffset : Bool) : Prop :=
  codes ≠ 0 ∧ scales ≠ 0 ∧ codes + scales + off = total ∧ (hasOffset = true ↔ off ≠ 0)

/-- Stage `i` runs under a stage prefix `k` (prefix mode). -/
def runStage (i k : Nat) : Bool := decide (i + 1 ≤ k)

/-- Stage `i` runs under `half` (0 = all, 1 = mixer, 2 = FFN front + router). -/
def runHalf (i half : Nat) : Bool :=
  if i < 3 then (half == 0 || half == 1) else (half == 0 || half == 2)

/-- QSA layer predicate. -/
def isQsa (l : Nat) : Bool := l % 4 == 3

/-- Aligned sizes are multiples of 16. -/
theorem align16_mod (v : Nat) : align16 v % 16 = 0 := by
  unfold align16; omega

/-- Alignment never shrinks a buffer. -/
theorem align16_ge (v : Nat) : v ≤ align16 v := by
  unfold align16; omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (v : Nat) : align16 v < v + 16 := by
  unfold align16; omega

/-- A cursor take never moves the cursor backwards and covers the requested bytes. -/
theorem cursorTake_covers (used count size : Nat) : used + count * size ≤ cursorTake used count size := by
  unfold cursorTake
  have := align16_ge (used + count * size)
  exact this

/-- A cursor that started 16-aligned stays 16-aligned. -/
theorem cursorTake_mod (used count size : Nat) : cursorTake used count size % 16 = 0 := by
  unfold cursorTake
  exact align16_mod _

/-- The activation row of the model's width needs 10 Q8_K blocks (2920 bytes). -/
theorem q8kBytes_2560 : q8kBytes 2560 = 2920 := by
  decide

/-- A Q8_K buffer for a multiple of 256 holds whole blocks. -/
theorem q8kBytes_blocks (n : Nat) (h : n % 256 = 0) : q8kBytes n = (n / 256) * 292 := by
  rfl

/-- Consistent planes lie wholly inside the tensor: the scales plane ends at or before the offset plane. -/
theorem planesOk_in_bounds (c s o t : Nat) (b : Bool) (h : planesOk c s o t b) : c + s ≤ t := by
  obtain ⟨_, _, h3, _⟩ := h
  omega

/-- A tensor with an offset plane declares `has_offset`, and conversely. -/
theorem planesOk_offset_flag (c s o t : Nat) (b : Bool) (h : planesOk c s o t b) (ho : o ≠ 0) :
    b = true := by
  obtain ⟨_, _, _, h4⟩ := h
  exact h4.mpr ho

/-- Running the first `k` stages means stage `i` runs exactly when `i < k`. -/
theorem runStage_iff (i k : Nat) : runStage i k = true ↔ i < k := by
  simp [runStage]

/-- A prefix of 5 runs every one of the five stages. -/
theorem runStage_full (i : Nat) (hi : i < 5) : runStage i 5 = true := by
  rw [runStage_iff]; exact hi

/-- Stage prefixes are monotone: a longer prefix runs everything a shorter one does. -/
theorem runStage_mono (i k k' : Nat) (h : k ≤ k') (hr : runStage i k = true) : runStage i k' = true := by
  rw [runStage_iff] at *
  omega

/-- `half = 0` runs every stage. -/
theorem runHalf_all : ∀ i : Fin 5, runHalf i.val 0 = true := by
  decide

/-- The two halves together cover every stage. -/
theorem runHalf_cover : ∀ i : Fin 5, (runHalf i.val 1 || runHalf i.val 2) = true := by
  decide

/-- The two halves never run the same stage twice. -/
theorem runHalf_disjoint : ∀ i : Fin 5, (runHalf i.val 1 && runHalf i.val 2) = false := by
  decide

/-- Exactly 12 of the 48 layers are QSA. -/
theorem qsa_count : ((List.range 48).filter (fun l => isQsa l)).length = 12 := by
  decide

/-- Exactly 36 of the 48 layers are GDN. -/
theorem gdn_count : ((List.range 48).filter (fun l => !isQsa l)).length = 36 := by
  decide

/-- The RoPE position guard: an accepted position lies below the table size. -/
theorem rope_pos_in_table (posBase pos maxCells : Nat) (h : posBase + pos < maxCells) : pos < maxCells := by
  omega

/-- The router `k` guard (`1..64`) leaves a non-empty, bounded expert set. -/
theorem k_guard (k : Nat) (h1 : 1 ≤ k) (h2 : k ≤ 64) : 0 < k ∧ k < 65 := by
  omega

end Strata.Layer
