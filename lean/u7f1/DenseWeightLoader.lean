/-!
# MODULE: dense_weight_loader
SOURCES: src/core/weights.cpp
The dense-weight loader: the per-plane segment list of a tensor, the four byte conversions between
pack and arena forms, the staging-buffer bound, chunking that never splits an fp16 element, the
index kind remap and the compacted arena placement under a skip set.

## Source semantics
* Conversions: `Copy` keeps n bytes; `Bf16High`/`ToF16` take 4 B/elem to 2; `WidenF16` takes 2 to 4.
  No conversion grows past 2n, which is why `stage_out` is `2 * CHUNK`.
* A quantized tensor is codes | scales | optional offset, contiguous in both forms; an fp16 scale
  plane is half the size in the source. The segment sums must equal the index's totals.
* Promoted tensors are exactly 4 B/elem in and 2 B/elem out; widening an f32 scale plane is exact.
* An odd widening tail is shortened by one byte (and refused if only one byte is left).
* Index kinds 4 (BF16) and 5 (F16) are raw 16-bit and take the engine forms of kinds 1 and 3.
* Under a skip set, kept rows are re-placed in order at the index alignment: offsets aligned,
  increasing, and the pool covers every kept tensor.

## Form template identifiers
`Conv` → `Conv`; output size per chunk → `outBytes`; `Seg` → `Seg`; quantized segment list →
`quantSegs`; `kind == 4 || kind == 5` remap → `remapKind`; skip-set compaction → `place`.
-/

namespace Ref.DenseWeightLoader

-- § Conversions

/-- How a segment's bytes are transformed from pack to arena. -/
inductive Conv
  | copy
  | bf16High
  | toF16
  | widenF16

/-- Bytes written for `n` source bytes. -/
def outBytes : Conv → Nat → Nat
  | .copy, n => n
  | .bf16High, n => n / 4 * 2
  | .toF16, n => n / 4 * 2
  | .widenF16, n => n / 2 * 4

/-- No conversion more than doubles its input. -/
theorem outBytes_le_double (cv : Conv) (n : Nat) : outBytes cv n ≤ 2 * n := by
  cases cv <;> simp only [outBytes] <;> omega

/-- The source staging chunk, 8 MiB. -/
def CHUNK : Nat := 8 * 1024 * 1024

/-- `stage_out` at `2 * CHUNK` holds any converted chunk. -/
theorem stage_out_suffices (cv : Conv) {n : Nat} (h : n ≤ CHUNK) : outBytes cv n ≤ 2 * CHUNK := by
  have := outBytes_le_double cv n
  omega

/-- A promoted tensor of `e` elements: 4e bytes in, 2e out. -/
theorem promoted_sizes (e : Nat) : outBytes .bf16High (4 * e) = 2 * e ∧ outBytes .toF16 (4 * e) = 2 * e := by
  simp only [outBytes]; omega

/-- Widening a scale plane of `m` f32 scales from its fp16 source form gives the full plane. -/
theorem widen_exact_size (m : Nat) : outBytes .widenF16 (2 * m) = 4 * m := by
  simp only [outBytes]; omega

/-- Chunking a halving conversion at 4-byte multiples loses nothing. -/
theorem halving_additive (a b : Nat) : outBytes .bf16High (4 * a + 4 * b) =
    outBytes .bf16High (4 * a) + outBytes .bf16High (4 * b) := by
  simp only [outBytes]; omega

/-- The 8 MiB chunk is even, so full chunks never split an fp16 element. -/
theorem chunk_even : CHUNK % 2 = 0 := by decide

/-- An odd tail is shortened to an even, non-empty length (unless only one byte is left). -/
theorem odd_tail_trim {n : Nat} (hodd : n % 2 = 1) (h1 : n ≠ 1) : (n - 1) % 2 = 0 ∧ 0 < n - 1 := by
  omega

/-- The high half of an f32 is its bf16 encoding: `1.0f` (0x3F800000) is bf16 0x3F80. -/
theorem bf16_high_one : ((0x3F800000 : UInt32) >>> 16).toUInt16 = 0x3F80 := by decide

-- § Segment lists

/-- One contiguous run of a tensor with one transform. -/
structure Seg where
  srcOff : Nat
  srcBytes : Nat
  dstOff : Nat
  dstBytes : Nat
  conv : Conv

/-- Source bytes of an fp16-or-f32 scale plane. -/
def srcScale (scales : Nat) (fp16 : Bool) : Nat := if fp16 then scales / 2 else scales

/-- The segment list of a quantized tensor: codes, then scales, then the optional offset plane. -/
def quantSegs (codes scales offset : Nat) (fp16 : Bool) : List Seg :=
  [⟨0, codes, 0, codes, .copy⟩] ++
  (if scales = 0 then [] else
    [⟨codes, srcScale scales fp16, codes, scales, if fp16 then .widenF16 else .copy⟩]) ++
  (if offset = 0 then [] else
    [⟨codes + srcScale scales fp16, offset, codes + scales, offset, .copy⟩])

/-- Total source bytes of a segment list. -/
def srcSum (ss : List Seg) : Nat := (ss.map Seg.srcBytes).sum

/-- Total destination bytes of a segment list. -/
def dstSum (ss : List Seg) : Nat := (ss.map Seg.dstBytes).sum

/-- The quantized segments cover codes + source scales + offset in the pack. -/
theorem quant_src_sum (c s o : Nat) (f : Bool) :
    srcSum (quantSegs c s o f) = c + srcScale s f + o := by
  unfold quantSegs srcSum
  by_cases hs : s = 0 <;> by_cases ho : o = 0 <;> simp [hs, ho, srcScale] <;>
    cases f <;> simp <;> omega

/-- The quantized segments cover codes + scales + offset in the arena. -/
theorem quant_dst_sum (c s o : Nat) (f : Bool) :
    dstSum (quantSegs c s o f) = c + s + o := by
  unfold quantSegs dstSum
  by_cases hs : s = 0 <;> by_cases ho : o = 0 <;> simp [hs, ho] <;> omega

/-- The loader's cross-check passes exactly when the index totals are the plane sums. -/
theorem segment_check_iff (c s o srcB dstB : Nat) (fp : Bool) :
    (srcSum (quantSegs c s o fp) = srcB ∧ dstSum (quantSegs c s o fp) = dstB) ↔
      (c + srcScale s fp + o = srcB ∧ c + s + o = dstB) := by
  rw [quant_src_sum, quant_dst_sum]

-- § Index kinds

/-- Kinds 4 (BF16) and 5 (F16) are raw 16-bit and take the engine forms of kinds 1 and 3. -/
def remapKind : Nat → Nat × Bool
  | 4 => (1, true)
  | 5 => (3, true)
  | k => (k, false)

/-- Raw-16 rows are copied, never re-rounded. -/
theorem remap_raw (k : Nat) (h : (remapKind k).2 = true) : k = 4 ∨ k = 5 := by
  unfold remapKind at h
  split at h <;> simp_all

/-- A row is parsed only when all 19 fields are present. -/
def rowParsed (fields : Nat) : Bool := fields == 19

/-- A short row is refused rather than loaded at a wrong offset. -/
theorem short_row_refused {n : Nat} (h : n ≠ 19) : rowParsed n = false := by
  simp [rowParsed, h]

-- § Compaction under a skip set

/-- Round up to a multiple of 256 (the default alignment). -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- An aligned size holds the tensor. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by unfold align256; omega

/-- An aligned size is a multiple of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by unfold align256; omega

/-- Place rows `(skipped, bytes)` from `at`: offsets of kept rows and the final pool. -/
def place : List (Bool × Nat) → Nat → List Nat × Nat
  | [], at_ => ([], at_)
  | (true, _) :: t, at_ => place t at_
  | (false, b) :: t, at_ => let r := place t (at_ + align256 b); (at_ :: r.1, r.2)

/-- Bytes of the kept rows. -/
def keptBytes : List (Bool × Nat) → Nat
  | [] => 0
  | (true, _) :: t => keptBytes t
  | (false, b) :: t => b + keptBytes t

/-- The compacted pool covers every kept tensor. -/
theorem place_pool_ge : ∀ (rows : List (Bool × Nat)) (at_ : Nat),
    at_ + keptBytes rows ≤ (place rows at_).2
  | [], at_ => by simp [place, keptBytes]
  | (true, _) :: t, at_ => by simpa [place, keptBytes] using place_pool_ge t at_
  | (false, b) :: t, at_ => by
    have ih := place_pool_ge t (at_ + align256 b)
    have hb := align256_ge b
    simp only [place, keptBytes]
    omega

/-- The pool never moves backwards. -/
theorem place_pool_mono : ∀ (rows : List (Bool × Nat)) (at_ : Nat), at_ ≤ (place rows at_).2 := by
  intro rows at_
  have := place_pool_ge rows at_
  omega

/-- Every kept row starts on an alignment boundary. -/
theorem place_aligned : ∀ (rows : List (Bool × Nat)) (at_ : Nat), at_ % 256 = 0 →
    ∀ o ∈ (place rows at_).1, o % 256 = 0
  | [], _, _, o, ho => by simp [place] at ho
  | (true, _) :: t, at_, h, o, ho => place_aligned t at_ h o (by simpa [place] using ho)
  | (false, b) :: t, at_, h, o, ho => by
    simp only [place, List.mem_cons] at ho
    rcases ho with rfl | ho
    · exact h
    · have := align256_mod b
      exact place_aligned t (at_ + align256 b) (by omega) o ho

end Ref.DenseWeightLoader



