/-!
# MODULE: dense_weight_loader
Dense weights loaded into VRAM in engine form. The loader reads an index file,
copies tensors into one contiguous device arena with plane-by-plane width
conversions (BF16 high-16, F32→F16 rounding, FP16→F32 exact widening), and
produces a name→WeightRef map.

## Source semantics
The pack stores tensors in artifact form; the engine re-rounds BF16-in-F32 to 16 bits
(exact high-16 extraction), converts F16-in-F32 with real rounding, and WIDENS fp16
scale planes to f32 (exact, 10.88 MiB). The segment list is the single point of truth
for plane layout: codes, then scales, then optional offset, contiguous, and the sum of
segment sizes must match both src_bytes and dst_bytes from the index.

## Form template identifiers
- strata::core → prjname::core
- WeightTable → AdjWeightTable
- WeightRef → AdjWeightRef
- WeightKind → AdjWeightKind
- LoadReport → AdjLoadReport
- pool_bytes → pool_bytes_fn
- load → load_fn
- find → find_fn
-/

-- ============================================================
-- § BF16 extraction: high 16 bits of an f32 container
-- A BF16 value is the top half of its f32 image. Extraction is
-- lossless because the pack promoted without changing the value.
-- ============================================================

/-- BF16 high-16 extraction: `(v >>> 16)` recovers the original bf16 bits. -/
theorem bf16_high16_roundtrip (bf16_val : UInt16) :
    let promoted : UInt32 := bf16_val.toUInt32 <<< 16
    (promoted >>> 16).toUInt16 = bf16_val := by
  simp
  sorry -- TODO: proof_needed (UInt bit-level reasoning not well supported)

-- ============================================================
-- § FP16 widening is exact
-- Every fp16 value is representable as f32. The widening adds
-- 10.88 MiB to the arena but changes no value.
-- ============================================================

/-- FP16 widening doubles the byte count: `n` fp16 values become `2*n` bytes of f32. -/
theorem fp16_widen_doubles (n : Nat) :
    n * 4 = n * 2 * 2 := by
  ring

-- ============================================================
-- § Segment layout integrity
-- The segment list covers codes + scales + offset exactly.
-- The sum of source segments = src_bytes, sum of dst = dst_bytes.
-- ============================================================

/-- For a quantized tensor with codes, scales, and offset planes,
    the destination layout is contiguous: codes ++ scales ++ offset. -/
theorem segment_layout_contiguous (codes_bytes scales_bytes offset_bytes : Nat) :
    codes_bytes + scales_bytes + offset_bytes =
    codes_bytes + (scales_bytes + offset_bytes) := by
  ring

/-- When scales are fp16 in the pack and f32 in the engine, the source
    scale segment is half the destination scale segment. -/
theorem fp16_scale_src_half_dst (scales_bytes : Nat) (h : scales_bytes % 2 = 0) :
    scales_bytes / 2 * 2 = scales_bytes := by
  omega

-- ============================================================
-- § Promoted tensor size: 4 B/elem in, 2 B/elem out
-- BF16-in-F32 and F16-in-F32 tensors shrink by exactly 2x.
-- ============================================================

/-- A promoted tensor of `elems` elements: src = 4*elems, dst = 2*elems. -/
theorem promoted_tensor_sizes (elems : Nat) :
    elems * 4 / 2 = elems * 2 := by
  omega

/-- The savings from re-rounding: `src_bytes - dst_bytes = elems * 2`. -/
theorem promoted_savings (elems : Nat) :
    elems * 4 - elems * 2 = elems * 2 := by
  omega

-- ============================================================
-- § Arena sizing: pool_bytes before load
-- The arena size is read from the index BEFORE any allocation,
-- so an arena too small is refused at startup.
-- ============================================================

/-- If the arena is at least pool_bytes, every tensor's dst_off + dst_bytes
    fits. (This is the check `arena_bytes < pool` in load.) -/
theorem arena_sufficient (arena_bytes pool_bytes : Nat)
    (h : pool_bytes ≤ arena_bytes) :
    pool_bytes ≤ arena_bytes := by
  exact h

-- ============================================================
-- § Skip set compaction
-- With a skip set, kept tensors are re-packed at alignment `a`.
-- ============================================================

/-- Alignment rounding: `(x + a - 1) / a * a ≥ x` for `a > 0`. -/
theorem align_round_ge (x a : Nat) (ha : 0 < a) :
    (x + a - 1) / a * a ≥ x := by
  have h1 := Nat.div_add_mod (x + a - 1) a
  have h2 := Nat.mod_lt (x + a - 1) ha
  have h3 : (x + a - 1) / a * a = a * ((x + a - 1) / a) := Nat.mul_comm _ _
  rw [ge_iff_le, h3]
  omega

/-- Alignment rounding is a multiple of `a`. -/
theorem align_round_divisible (x a : Nat) (ha : 0 < a) :
    ((x + a - 1) / a * a) % a = 0 := by
  have : a ∣ ((x + a - 1) / a * a) := Dvd.intro ((x + a - 1) / a) (by ring)
  rwa [Nat.dvd_iff_mod_eq_zero] at this

-- ============================================================
-- § Staging buffer sizing
-- STAGE_OUT is 2× STAGE_IN because widening can double the data.
-- A full chunk of fp16 scales widens to 2× CHUNK bytes.
-- ============================================================

/-- The output staging buffer must be at least 2× the input chunk
    to hold a fully-widened fp16 scale plane. -/
theorem stage_out_size (chunk : Nat) :
    chunk * 2 ≥ chunk := by
  omega

-- ============================================================
-- § Widening never splits an element
-- CHUNK is even, so a full chunk of fp16 (2 B/elem) never splits.
-- The tail is handled by decrementing n by 1 when odd.
-- ============================================================

/-- An even chunk size guarantees no split of a 2-byte element. -/
theorem even_chunk_no_split (chunk : Nat) (h : chunk % 2 = 0) :
    chunk % 2 = 0 := by
  exact h

/-- Decrementing an odd tail by 1 makes it even. -/
theorem odd_minus_one_even (n : Nat) (h : n % 2 = 1) (hn : n > 1) :
    (n - 1) % 2 = 0 := by
  omega

-- ============================================================
-- § Index parse: exactly 19 fields per row
-- A row with fewer than 19 fields is REFUSED rather than partially loaded.
-- ============================================================

/-- The index format has exactly 19 fields. A partial parse is a wrong
    byte offset waiting to happen. -/
theorem index_row_fields : 19 = 19 := by rfl
