/-!
# MODULE: weight_pack_loader
SOURCES: src/core/weights.cpp
The dense-weight loader: reads `index.txt` rows, converts each tensor plane by plane (copy, bf16 high half, f16 round,
fp16 widen to f32) through bounded staging buffers, and places the results in an aligned (optionally compacted) arena.

## Source semantics
* A quantized tensor is codes, then scales, then an optional offset plane; the segment sums must equal the recorded
  source and destination totals or the loader refuses.
* bf16 promotion takes the high 16 bits of the f32 container exactly; a promoted tensor is 4 B/elem in, 2 B/elem out.
* fp16 widening doubles the byte count, never splits an element, and fits the staging-out buffer (2 x CHUNK).
* A skip set compacts the arena: kept rows are re-placed at the index alignment, skipped rows take no bytes.
* The chunked copy loop covers each plane exactly once with chunks of at most CHUNK bytes.

## Form template identifiers
Conv -> Conv ; Seg -> segment sums ; CHUNK -> CHUNK ; stage_out -> stageOutBytes ; WidenF16 -> Conv.widenF16 ;
Bf16High -> Conv.bf16High ; pool -> compactTotal ; align -> a
-/

namespace Strata.WeightLoader

-- § staging buffers

/-- 8 MiB of source per staging round. -/
def CHUNK : Nat := 8 * 1048576

/-- CHUNK is even, so a full chunk cannot split an fp16 element. -/
theorem CHUNK_even : CHUNK % 2 = 0 := by decide

/-- How a segment's bytes are transformed. -/
inductive Conv where
  | copy
  | bf16High
  | toF16
  | widenF16

/-- Output bytes of a chunk of `n` source bytes. -/
def outBytes : Conv → Nat → Nat
  | .copy, n => n
  | .bf16High, n => n / 4 * 2
  | .toF16, n => n / 4 * 2
  | .widenF16, n => n / 2 * 4

/-- THE STAGING-OUT BUFFER (2 x CHUNK) holds any conversion of a chunk, including the widening that grows it. -/
theorem outBytes_le_stage (c : Conv) (n : Nat) (h : n ≤ CHUNK) : outBytes c n ≤ 2 * CHUNK := by
  cases c <;> simp only [outBytes] <;> omega

/-- Widening an even-length plane exactly doubles it. -/
theorem widen_doubles (n : Nat) (h : n % 2 = 0) : outBytes .widenF16 n = 2 * n := by
  simp only [outBytes]; omega

/-- A promoted tensor shrinks 4 B/elem to 2 B/elem: exactly half. -/
theorem promote_halves (elems : Nat) : outBytes .bf16High (elems * 4) = elems * 2 := by
  simp only [outBytes]; omega

/-- The promotion shape check: source and destination byte counts must agree with the element count. -/
def promotedOk (elems src dst : Nat) : Bool := decide (src = elems * 4) && decide (dst = elems * 2)

/-- If the check passes, the conversion's output size is the recorded destination size. -/
theorem promotedOk_sound (elems src dst : Nat) (h : promotedOk elems src dst = true) :
    outBytes .bf16High src = dst := by
  simp only [promotedOk, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨h1, h2⟩ := h
  subst h1
  simp only [outBytes]
  omega

/-- A chunk of a widening plane is trimmed to an even length (or refused if a single byte remains). -/
theorem chunk_even_of_even (rem chunk : Nat) (h1 : rem % 2 = 0) (h2 : chunk % 2 = 0) :
    (min rem chunk) % 2 = 0 := by
  omega

-- § bf16 and f16 forms

/-- The high 16 bits of a 32-bit container. -/
def bf16High (v : Nat) : Nat := v / 65536

/-- The high half fits in 16 bits. -/
theorem bf16High_lt (v : Nat) (h : v < 4294967296) : bf16High v < 65536 := by
  unfold bf16High; omega

/-- A container holding a bf16 value (low half zero) is recovered exactly: promotion is a copy with a stride. -/
theorem bf16High_exact (v : Nat) (h : v % 65536 = 0) : bf16High v * 65536 = v := by
  unfold bf16High; omega

-- § segment accounting

/-- Source bytes of a quantized tensor's segments. -/
def segSrcSum (codes scales off : Nat) (fp16 : Bool) : Nat :=
  codes + (if fp16 then scales / 2 else scales) + off

/-- Destination bytes of the segments. -/
def segDstSum (codes scales off : Nat) : Nat := codes + scales + off

/-- Without fp16 scales the planes are already in engine form: source and destination sums agree. -/
theorem seg_sums_agree_plain (codes scales off : Nat) :
    segSrcSum codes scales off false = segDstSum codes scales off := by
  simp [segSrcSum, segDstSum]

/-- With an fp16 scale plane the destination is larger by exactly the widened half. -/
theorem seg_sums_fp16 (codes scales off : Nat) (h : scales % 2 = 0) :
    segDstSum codes scales off = segSrcSum codes scales off true + scales / 2 := by
  simp only [segSrcSum, segDstSum, if_true]
  omega

/-- The loader's own consistency check of segment totals against the index row. -/
def segTotalsOk (sumSrc sumDst recSrc recDst : Nat) : Bool := decide (sumSrc = recSrc) && decide (sumDst = recDst)

/-- A mis-sized plane layout is caught. -/
theorem segTotalsOk_false_of_src (sumSrc sumDst recSrc recDst : Nat) (h : sumSrc ≠ recSrc) :
    segTotalsOk sumSrc sumDst recSrc recDst = false := by
  simp [segTotalsOk, h]

-- § alignment and compaction

/-- Round `n` up to a multiple of `a` (as the compaction does). -/
def roundUp (a n : Nat) : Nat := (n + a - 1) / a * a

/-- Rounding never shrinks a positive-alignment size. -/
theorem roundUp_ge (a n : Nat) (ha : 0 < a) : n ≤ roundUp a n := by
  unfold roundUp
  have h1 := Nat.div_add_mod (n + a - 1) a
  have h2 := Nat.mod_lt (n + a - 1) ha
  have h3 : (n + a - 1) / a * a = a * ((n + a - 1) / a) := Nat.mul_comm _ _
  omega

/-- Rounded sizes are multiples of the alignment. -/
theorem roundUp_dvd (a n : Nat) : a ∣ roundUp a n := by
  unfold roundUp
  exact Nat.dvd_mul_left a _

/-- Size of the compacted arena: aligned sizes of the kept rows, in index order. -/
def compactTotal (a : Nat) : List Nat → Nat
  | [] => 0
  | s :: ss => roundUp a s + compactTotal a ss

/-- The compacted arena is a multiple of the alignment. -/
theorem compactTotal_dvd (a : Nat) (ss : List Nat) : a ∣ compactTotal a ss := by
  induction ss with
  | nil => exact ⟨0, by simp [compactTotal]⟩
  | cons s ss ih =>
    simp only [compactTotal]
    exact Nat.dvd_add (roundUp_dvd a s) ih

/-- The compacted arena holds every kept row. -/
theorem compactTotal_ge (a : Nat) (ha : 0 < a) (ss : List Nat) : ss.sum ≤ compactTotal a ss := by
  induction ss with
  | nil => simp [compactTotal]
  | cons s ss ih =>
    simp only [compactTotal, List.sum_cons]
    have := roundUp_ge a s ha
    omega

/-- The arena size check: refuse when the arena is smaller than the pool. -/
def arenaOk (arena pool : Nat) : Bool := decide (pool ≤ arena)

/-- Refusal is exactly "too small". -/
theorem arenaOk_false_iff (arena pool : Nat) : arenaOk arena pool = false ↔ arena < pool := by
  unfold arenaOk; simp

-- § the chunked copy loop

/-- Chunk sizes produced by the `while (done < src_bytes)` loop (fuel-bounded form). -/
def chunkSizesF : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | f + 1, chunk, rem => if rem = 0 then [] else min rem chunk :: chunkSizesF f chunk (rem - min rem chunk)

/-- The chunks cover the plane exactly once. -/
theorem chunkSizesF_sum (chunk : Nat) (hc : 0 < chunk) :
    ∀ (f rem : Nat), rem ≤ f → (chunkSizesF f chunk rem).sum = rem := by
  intro f
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    simp [chunkSizesF]
  | succ f ih =>
    intro rem h
    by_cases h0 : rem = 0
    · subst h0; simp [chunkSizesF]
    · have := ih (rem - min rem chunk) (by omega)
      simp only [chunkSizesF, if_neg h0, List.sum_cons]
      omega

/-- No chunk exceeds the staging buffer. -/
theorem chunkSizesF_le (chunk : Nat) : ∀ (f rem : Nat), ∀ x ∈ chunkSizesF f chunk rem, x ≤ chunk := by
  intro f
  induction f with
  | zero => intro rem x hx; simp [chunkSizesF] at hx
  | succ f ih =>
    intro rem x hx
    by_cases h0 : rem = 0
    · subst h0; simp [chunkSizesF] at hx
    · simp only [chunkSizesF, if_neg h0, List.mem_cons] at hx
      rcases hx with hx | hx
      · omega
      · exact ih _ x hx

/-- Output offset of a widening chunk is `done * 2`, and of a halving chunk `done / 2`. -/
theorem widen_offset_consistent (done n : Nat) (h : n % 2 = 0) :
    (done + n) * 2 = done * 2 + outBytes .widenF16 n := by
  simp only [outBytes]; omega

end Strata.WeightLoader
