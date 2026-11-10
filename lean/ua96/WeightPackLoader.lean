/-!
# MODULE: weight_pack_loader
SOURCES: src/core/weights.cpp
The dense-weight loader: segment list per tensor, plane-by-plane width conversion (bf16 high-half, f16
re-round, exact fp16→f32 widening), chunked staging, and arena compaction under a skip set.

## Source semantics
* A quantized tensor is codes ++ scales ++ optional offsets; fp16 scales are half the size in the pack.
* Promoted (bf16/f16-in-f32) tensors are 4 B/elem in and 2 B/elem out.
* Chunks never split an element: widening needs even chunks; bf16/f16 conversion needs multiples of 4.
* With a skip set the kept rows are re-placed in index order at 256-byte alignment.

## Form template identifiers
srcSum/dstSum ↦ `sum_src`/`sum_dst`; chunkLen ↦ `min(CHUNK, remaining)`; carve256 ↦ compaction `at += align`.
-/
namespace Ref.WeightPackLoader

-- § segment accounting

/-- Source bytes of the three planes. -/
def srcSum (codes scales off : Nat) (fp16 : Bool) : Nat :=
  codes + (if fp16 then scales / 2 else scales) + off

/-- Destination bytes of the three planes. -/
def dstSum (codes scales off : Nat) : Nat := codes + scales + off

/-- Without fp16 scales source and destination agree. -/
theorem seg_sums_copy (c s o : Nat) : srcSum c s o false = dstSum c s o := by
  simp [srcSum, dstSum]

/-- With fp16 scales the destination grows by exactly the scale plane's source size. -/
theorem seg_sums_widen (c s o : Nat) (h : s % 2 = 0) :
    dstSum c s o = srcSum c s o true + s / 2 := by
  simp only [srcSum, dstSum, if_true]; omega

/-- An odd fp16 destination byte count is impossible after widening (f32 planes are multiples of 4). -/
theorem widen_dst_even (elems : Nat) : (elems * 4) % 2 = 0 := by omega

/-- Widening `n` source bytes yields `2n` destination bytes. -/
theorem widen_out (n : Nat) (h : n % 2 = 0) : n / 2 * 4 = 2 * n := by omega

/-- Promoted tensors: 2 B/elem out is half of 4 B/elem in. -/
theorem promoted_halves (elems : Nat) : elems * 2 * 2 = elems * 4 := by omega

-- § chunking

/-- The chunk the loop reads: `min CHUNK remaining`. -/
def chunkLen (chunk remaining : Nat) : Nat := min chunk remaining

/-- A chunk makes progress when anything remains. -/
theorem chunk_progress (chunk rem : Nat) (hc : 0 < chunk) (hr : 0 < rem) :
    0 < chunkLen chunk rem ∧ rem - chunkLen chunk rem < rem := by
  unfold chunkLen; omega

/-- With the real 8 MiB CHUNK and an even plane, every chunk is even (no half fp16). -/
theorem chunk_even (rem done : Nat) (hr : rem % 2 = 0) (hd : done % 2 = 0) :
    chunkLen 8388608 (rem - done) % 2 = 0 ∨ rem < done := by
  unfold chunkLen; omega

/-- bf16/f16 conversion with 4-multiple chunks keeps destination offsets contiguous (halved). -/
theorem bf16_dst_contiguous (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + n / 4 * 2 = (done + n) / 2 := by omega

/-- fp16 widening keeps destination offsets contiguous (doubled). -/
theorem widen_dst_contiguous (done n : Nat) (hd : done % 2 = 0) (hn : n % 2 = 0) :
    done * 2 + n / 2 * 4 = (done + n) * 2 := by omega

/-- The staging output buffer (2 * CHUNK) holds the widest conversion (widen doubles a chunk). -/
theorem stage_out_fits (n chunk : Nat) (h : n ≤ chunk) : n / 2 * 4 ≤ chunk * 2 := by omega

-- § arena compaction

/-- 256-byte alignment of a row's destination size. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment does not shrink. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- Alignment gives multiples of 256. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- Compacted placement total. -/
def carve : List Nat → Nat → Nat
  | [], s => s
  | p :: ps, s => carve ps (s + align256 p)

/-- Placement is monotone. -/
theorem carve_ge : ∀ (parts : List Nat) (s : Nat), s ≤ carve parts s
  | [], s => Nat.le_refl s
  | p :: ps, s => by
      rw [carve]
      have h := carve_ge ps (s + align256 p)
      omega

/-- Placement splits over concatenation. -/
theorem carve_append : ∀ (xs ys : List Nat) (s : Nat), carve (xs ++ ys) s = carve ys (carve xs s)
  | [], ys, s => by simp [carve]
  | x :: xs, ys, s => by
      simp only [List.cons_append, carve]
      exact carve_append xs ys _

/-- Offsets of compacted rows stay 256-aligned. -/
theorem carve_aligned : ∀ (parts : List Nat) (s : Nat), s % 256 = 0 → carve parts s % 256 = 0
  | [], s, h => by simpa [carve] using h
  | p :: ps, s, h => by
      rw [carve]
      apply carve_aligned ps
      have := align256_mod p
      omega

/-- A kept row's `[dst_off, dst_off+aligned)` lies inside the pool: rows never overlap or overrun. -/
theorem row_fits (pre post : List Nat) (x : Nat) :
    carve pre 0 + align256 x ≤ carve (pre ++ x :: post) 0 := by
  rw [carve_append]
  have h := carve_ge post (carve pre 0 + align256 x)
  simp only [carve]
  exact h

/-- The row's own bytes lie inside the pool. -/
theorem row_bytes_fit (pre post : List Nat) (x : Nat) :
    carve pre 0 + x ≤ carve (pre ++ x :: post) 0 := by
  have h1 := row_fits pre post x
  have h2 := align256_ge x
  omega

-- § index row

/-- A parsed row must have exactly 19 fields. -/
def rowParsed (n : Nat) : Bool := n == 19

/-- Only 19 fields parse. -/
theorem row_parsed_iff (n : Nat) : rowParsed n = true ↔ n = 19 := by simp [rowParsed]

end Ref.WeightPackLoader
