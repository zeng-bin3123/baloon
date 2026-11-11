/-!
# MODULE: weight_pack_loader
SOURCES: src/core/weights.cpp
The dense-weight loader: each index row becomes a segment list (codes / scales / offsets for a
quantized tensor, one run otherwise), each segment carries a width transform (copy, bf16 high half,
f32→f16, fp16→f32 widening), and the segments are checked against the row's recorded totals before a
single byte moves.  A skip set compacts the arena by re-placing kept rows at the index alignment.

## Source semantics
* Quantized rows: codes (copy), scales (widened 2→4 B/elem iff `scales_fp16`), offsets (copy).  The
  destination planes are contiguous in the order codes, scales, offsets.
* A promoted row is exactly 4 B/elem in and 2 B/elem out; `Bf16High` keeps the top 16 bits of the
  f32 image, which is the bf16 encoding exactly, and loses only the low half.
* A widening chunk never splits an fp16 element: an odd tail is shortened by one.
* The staging output buffer is twice the input chunk because widening doubles the byte count.
* Compaction re-places kept rows at multiples of the alignment, so no two kept rows overlap.

## Form template identifiers
Seg/quantSegs ↔ Seg segs[3] · srcScaleBytes ↔ `scales_bytes / 2` · bf16High ↔ `v >> 16` ·
chunkLen ↔ the `n -= 1` tail rule · widenOut ↔ `elems * 4` · place ↔ the skip-set placement loop
-/

namespace WeightPackLoader

-- § the segment list of a quantized row

/-- One contiguous run of a row: source bytes in, destination bytes out, at a destination offset. -/
structure Seg where
  srcBytes : Nat
  dstOff : Nat
  dstBytes : Nat

/-- Source bytes of the scale plane: half the arena bytes when the pack holds fp16 scales. -/
def srcScaleBytes (scales : Nat) (fp16 : Bool) : Nat := if fp16 then scales / 2 else scales

/-- The three segments `load` builds for a quantized row. -/
def quantSegs (codes scales offs : Nat) (fp16 : Bool) : List Seg :=
  [⟨codes, 0, codes⟩, ⟨srcScaleBytes scales fp16, codes, scales⟩, ⟨offs, codes + scales, offs⟩]

/-- Sum of destination bytes. -/
def dstSum : List Seg → Nat
  | [] => 0
  | s :: ss => s.dstBytes + dstSum ss

/-- Sum of source bytes. -/
def srcSum : List Seg → Nat
  | [] => 0
  | s :: ss => s.srcBytes + srcSum ss

/-- The destination planes add up to codes + scales + offsets, the `dst_bytes` the index records. -/
theorem quant_dst_sum (c s o : Nat) (f : Bool) : dstSum (quantSegs c s o f) = c + s + o := by
  simp only [quantSegs, dstSum]
  omega

/-- Without fp16 scales the source span equals the destination span. -/
theorem quant_src_sum_f32 (c s o : Nat) : srcSum (quantSegs c s o false) = c + s + o := by
  simp [quantSegs, srcSum, srcScaleBytes]
  omega

/-- With fp16 scales of an even byte count the source span is smaller by exactly half the scale plane. -/
theorem quant_src_sum_f16 (c s o : Nat) (he : s % 2 = 0) :
    srcSum (quantSegs c s o true) + s / 2 = c + s + o := by
  simp only [quantSegs, srcSum, srcScaleBytes, ite_true]
  omega

/-- The destination planes are contiguous: each segment starts where the previous one ends. -/
theorem quant_dst_contiguous (c s o : Nat) (f : Bool) :
    match quantSegs c s o f with
    | [a, b, d] => a.dstOff + a.dstBytes = b.dstOff ∧ b.dstOff + b.dstBytes = d.dstOff
    | _ => False := by
  simp [quantSegs]

-- § width transforms

/-- The bf16 image of an f32 bit pattern: its high 16 bits. -/
def bf16High (v : Nat) : Nat := v / 65536

/-- The high half and the discarded low half reassemble the f32 bits exactly: nothing but the low
half is lost, so a pack-promoted bf16 (low half zero) round-trips. -/
theorem bf16High_recombine (v : Nat) : bf16High v * 65536 + v % 65536 = v := by
  unfold bf16High; omega

/-- A promoted bf16 value (low half zero) is recovered exactly. -/
theorem bf16High_exact (v : Nat) (h : v % 65536 = 0) : bf16High v * 65536 = v := by
  have hr := bf16High_recombine v
  rw [h, Nat.add_zero] at hr
  exact hr

/-- The high half of a 32-bit pattern fits in 16 bits. -/
theorem bf16High_fits (v : Nat) (h : v < 4294967296) : bf16High v < 65536 := by
  unfold bf16High; omega

/-- A promoted tensor of `e` elements: 4 bytes per element in, 2 out. -/
theorem promoted_halves (e : Nat) : (e * 4) / 4 * 2 = e * 2 := by
  omega

/-- Widening an even fp16 chunk of `n` bytes yields `2n` f32 bytes, which is why the output stage is
twice the input chunk. -/
theorem widen_doubles (n : Nat) (h : n % 2 = 0) : n / 2 * 4 = 2 * n := by
  omega

/-- Output never exceeds the `2 * CHUNK` staging buffer for any chunk of at most `CHUNK` bytes. -/
theorem widen_fits_stage (n chunk : Nat) (hn : n ≤ chunk) : n / 2 * 4 ≤ 2 * chunk := by
  omega

-- § chunking a widening plane

/-- The next chunk length: at most `chunk` and the remainder; shortened by one if odd. -/
def chunkLen (chunk rem : Nat) : Nat :=
  let n := min chunk rem
  if n % 2 = 1 then n - 1 else n

/-- A widening chunk is always a whole number of fp16 elements. -/
theorem chunkLen_even (chunk rem : Nat) : chunkLen chunk rem % 2 = 0 := by
  unfold chunkLen
  simp only
  split <;> omega

/-- A chunk never reads past the remaining plane. -/
theorem chunkLen_le_rem (chunk rem : Nat) : chunkLen chunk rem ≤ rem := by
  unfold chunkLen
  simp only
  split <;> omega

/-- With at least two bytes left and an even chunk size of at least 2, the loop makes progress. -/
theorem chunkLen_progress (chunk rem : Nat) (hc : 2 ≤ chunk) (hr : 2 ≤ rem) :
    0 < chunkLen chunk rem := by
  unfold chunkLen
  simp only
  split <;> omega

/-- The 8 MiB chunk is even, so a full chunk is never shortened. -/
theorem full_chunk_unshortened (rem : Nat) (h : 8388608 ≤ rem) : chunkLen 8388608 rem = 8388608 := by
  unfold chunkLen
  simp only
  have hm : min 8388608 rem = 8388608 := Nat.min_eq_left h
  rw [hm]
  rfl

-- § compaction under a skip set

/-- Round up to the index alignment of 256 bytes. -/
def alignUp (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Destination offsets of the kept rows, placed in index order from `base`. -/
def place : List Nat → Nat → List Nat
  | [], _ => []
  | b :: bs, base => base :: place bs (base + alignUp b)

/-- The pool the compacted arena needs. -/
def pool : List Nat → Nat
  | [] => 0
  | b :: bs => alignUp b + pool bs

/-- A row's bytes end before the next row's start: `alignUp` never shrinks. -/
theorem alignUp_ge (n : Nat) : n ≤ alignUp n := by
  unfold alignUp; omega

/-- Every placed row starts at an aligned offset when the base is aligned. -/
theorem place_aligned :
    ∀ (bs : List Nat) (a : Nat), a % 256 = 0 → ∀ x ∈ place bs a, x % 256 = 0
  | [], _, _, x, hx => by simp [place] at hx
  | b :: bs, a, ha, x, hx => by
    simp only [place, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact ha
    · have hb : alignUp b % 256 = 0 := by unfold alignUp; omega
      exact place_aligned bs (a + alignUp b) (by omega) x hx

/-- Every placed row lies inside the pool: `[a, a + pool]` holds all offsets. -/
theorem place_in_pool :
    ∀ (bs : List Nat) (a : Nat), ∀ x ∈ place bs a, a ≤ x ∧ x ≤ a + pool bs
  | [], _, x, hx => by simp [place] at hx
  | b :: bs, a, x, hx => by
    simp only [place, List.mem_cons] at hx
    simp only [pool]
    rcases hx with rfl | hx
    · omega
    · have ih := place_in_pool bs (a + alignUp b) x hx
      omega

/-- The load refuses unless the arena holds the pool. -/
def arenaFits (arena need : Nat) : Prop := need ≤ arena

/-- A fitting arena holds every placed row's start. -/
theorem fits_holds_rows (arena : Nat) (bs : List Nat) (h : arenaFits arena (pool bs)) :
    ∀ x ∈ place bs 0, x ≤ arena := by
  intro x hx
  have := place_in_pool bs 0 x hx
  unfold arenaFits at h
  omega

end WeightPackLoader
