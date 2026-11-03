/-!
# MODULE: weight_loader_segments
The dense-weight loader (`weights.cpp`): chunked conversion progress for the widening and BF16 paths, staging-buffer
bounds, segment tiling and compacted arena offsets.

## Source semantics
* `CHUNK = 8 MiB`; widening fp16 -> f32 doubles a chunk, so `stage_out` is `2*CHUNK`.
* Widening advances the destination by `2*n` per `n` source bytes; BF16 promotion by `n/2` per `n` source bytes.
* A promoted tensor is `4*elems` in and `2*elems` out; the saving is `2*elems`.
* A skip set compacts the arena: kept rows are re-placed in index order at 256-byte alignment.

## Form template identifiers
`CHUNK` -> chunk, `Conv::WidenF16` -> widen, `Conv::Bf16High` -> bf16High, `dst_off` -> dstOff, `at` -> off.
-/

namespace Strata.Weights

/-- 8 MiB of source per staging round. -/
def chunk : Nat := 8388608

/-- CHUNK is a multiple of 4, so full chunks never split a 4-byte element. -/
theorem chunk_mod4 : chunk % 4 = 0 := by decide

-- § widening (fp16 -> f32)

/-- Widening `n` (even) source bytes writes `(n/2)*4` bytes. -/
def widenOut (n : Nat) : Nat := n / 2 * 4

/-- Consecutive widened chunks write consecutive destination ranges. -/
theorem widen_progress (dstOff done n : Nat) (hn : n % 2 = 0) :
    dstOff + (done + n) * 2 = (dstOff + done * 2) + widenOut n := by
  unfold widenOut; omega

/-- The widened output of a chunk fits `stage_out` (`2*CHUNK`). -/
theorem widen_fits (n : Nat) (h : n ≤ chunk) : widenOut n ≤ 2 * chunk := by
  unfold widenOut; omega

/-- Trim a possibly odd tail to whole elements. -/
def evenTrim (n : Nat) : Nat := if n % 2 = 1 then n - 1 else n

/-- A trimmed chunk is even, positive (when at least 2 bytes), and no larger. -/
theorem evenTrim_ok (n : Nat) (h : 2 ≤ n) :
    evenTrim n % 2 = 0 ∧ 0 < evenTrim n ∧ evenTrim n ≤ n := by
  unfold evenTrim; split <;> omega

-- § BF16 high-half promotion

/-- BF16 promotion keeps 2 of every 4 bytes. -/
def bf16Out (n : Nat) : Nat := n / 4 * 2

/-- Consecutive BF16 chunks (multiples of 4) write consecutive destination ranges. -/
theorem bf16_progress (dstOff done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    dstOff + (done + n) / 2 = (dstOff + done / 2) + bf16Out n := by
  unfold bf16Out; omega

/-- The BF16 output of a chunk fits `stage_out`. -/
theorem bf16_fits (n : Nat) (h : n ≤ chunk) : bf16Out n ≤ 2 * chunk := by
  unfold bf16Out; omega

/-- Taking the high 16 bits of a 32-bit value yields a 16-bit value. -/
theorem high16_lt (v : Nat) (h : v < 4294967296) : v / 65536 < 65536 := by omega

/-- A promoted tensor's saving is exactly its output size. -/
theorem promoted_saving (elems src dst : Nat) (hs : src = elems * 4) (hd : dst = elems * 2) :
    src - dst = dst := by omega

-- § segment tiling

/-- A quantized tensor's destination segments are contiguous: codes, scales, offsets. -/
theorem dst_segments_contiguous (codes scales offs dst : Nat) (h : dst = codes + scales + offs) :
    0 + codes = codes ∧ codes + scales = codes + scales ∧ codes + scales + offs = dst := by omega

/-- With fp16 scales (even byte count) the source scale plane is half the destination plane. -/
theorem fp16_source_plane (scales : Nat) (h : scales % 2 = 0) : (scales / 2) * 2 = scales := by omega

/-- The offset segment's source start follows the narrower source scale plane. -/
theorem src_offset_start (codes scales : Nat) (h : scales % 2 = 0) :
    codes + scales / 2 + scales / 2 = codes + scales := by omega

-- § compacted offsets

/-- Alignment used for kept rows. -/
def align256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- Aligned size covers the request and is a multiple of 256. -/
theorem align256_props (b : Nat) : b ≤ align256 b ∧ align256 b % 256 = 0 := by
  unfold align256; omega

/-- Offset of kept row `i`: the sum of the aligned sizes of the rows before it. -/
def off : List Nat → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | b :: bs, i + 1 => align256 b + off bs i

/-- Total arena size. -/
def pool : List Nat → Nat
  | [] => 0
  | b :: bs => align256 b + pool bs

/-- Every compacted offset is aligned. -/
theorem off_aligned : ∀ (l : List Nat) (i : Nat), off l i % 256 = 0
  | _, 0 => by simp [off]
  | [], _ + 1 => by simp [off]
  | b :: bs, i + 1 => by
      have h := off_aligned bs i
      have h2 := (align256_props b).2
      simp only [off]
      omega

/-- Every compacted offset lies within the pool. -/
theorem off_le_pool : ∀ (l : List Nat) (i : Nat), off l i ≤ pool l
  | _, 0 => by simp [off]
  | [], _ + 1 => by simp [off, pool]
  | b :: bs, i + 1 => by
      have h := off_le_pool bs i
      simp only [off, pool]
      omega

/-- A row's destination range ends within the pool (so `arena_bytes >= pool` suffices). -/
theorem row_end_in_pool : ∀ (l : List Nat) (i : Nat), i < l.length →
    off l i + align256 (l.getD i 0) ≤ pool l
  | [], i, h => by simp at h
  | b :: bs, 0, _ => by simp [off, pool, List.getD_cons_zero]
  | b :: bs, i + 1, h => by
      have h' : i < bs.length := by simpa using h
      have := row_end_in_pool bs i h'
      simp only [off, pool, List.getD_cons_succ]
      omega

end Strata.Weights
