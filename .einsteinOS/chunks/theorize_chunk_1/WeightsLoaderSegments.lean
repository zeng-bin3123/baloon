/-!
# MODULE: weights_loader_segments
Byte arithmetic of the dense-weight loader: plane segments, chunked conversion, compaction alignment.

## Source semantics
A quantized tensor is codes | scales | optional offsets. An fp16 scale plane is half its engine size in the pack and is
widened to f32 (exact). Promoted tensors are 4 B/elem in and 2 B/elem out. Chunks are at most 8 MiB of source; widening
never splits an element; stage-out is twice stage-in. Compacted placement rounds each tensor to the index alignment.

## Form template identifiers
- Seg list / sum_src / sum_dst -> srcTotal / dstTotal
- CHUNK                        -> CHUNK
- Conv::Bf16High / WidenF16    -> bf16_chunk_contiguous / widen_chunk_contiguous
- skip-set compaction          -> placeAt
-/
namespace Strata.WeightsLoaderSegments

def srcScale (scales : Nat) (fp16 : Bool) : Nat := if fp16 then scales / 2 else scales
def srcTotal (codes scales offs : Nat) (fp16 : Bool) : Nat := codes + srcScale scales fp16 + offs
def dstTotal (codes scales offs : Nat) : Nat := codes + scales + offs

/-- The widened plane is exactly twice its source (even byte count), so totals differ by the plane's source size. -/
theorem widen_total {codes scales offs : Nat} (he : scales % 2 = 0) :
    dstTotal codes scales offs = srcTotal codes scales offs true + scales / 2 := by
  unfold dstTotal srcTotal srcScale
  simp
  omega

/-- Without fp16 scales the two layouts have equal size. -/
theorem copy_total (codes scales offs : Nat) : dstTotal codes scales offs = srcTotal codes scales offs false := by
  simp [dstTotal, srcTotal, srcScale]

/-- The offset segment starts where the scale segment ends in the source and ends at the source total. -/
theorem offset_seg_end (codes scales offs : Nat) (fp16 : Bool) :
    (codes + srcScale scales fp16) + offs = srcTotal codes scales offs fp16 := by
  unfold srcTotal; omega

def CHUNK : Nat := 8388608

/-- CHUNK is even, so a full chunk never splits an fp16 element. -/
theorem CHUNK_even : CHUNK % 2 = 0 := by decide

def widenN (rem : Nat) : Nat :=
  if min 8388608 rem % 2 = 1 then min 8388608 rem - 1 else min 8388608 rem

/-- A widening chunk covers a whole number of fp16 elements. -/
theorem widenN_even (rem : Nat) : widenN rem % 2 = 0 := by
  unfold widenN
  split <;> omega

/-- A widening chunk never overreads the plane. -/
theorem widenN_le (rem : Nat) : widenN rem ≤ rem := by
  unfold widenN
  split <;> omega

/-- A widening chunk is non-empty while an even number of bytes remain. -/
theorem widenN_pos {rem : Nat} (h0 : 0 < rem) (he : rem % 2 = 0) : 0 < widenN rem := by
  unfold widenN
  split <;> omega

/-- A widened chunk (output = 2 × input) fits stage_out, which is twice CHUNK. -/
theorem widen_out_fits {n : Nat} (h : n ≤ 8388608) : n / 2 * 4 ≤ 2 * 8388608 := by omega

/-- bf16 promotion: with 4-byte-aligned progress, the next output offset equals the end of this chunk's output. -/
theorem bf16_chunk_contiguous {done n : Nat} (h1 : done % 4 = 0) (h2 : n % 4 = 0) :
    done / 2 + n / 4 * 2 = (done + n) / 2 := by omega

/-- fp16 widening: output offset `done*2` advances exactly by the chunk's output size. -/
theorem widen_chunk_contiguous {done n : Nat} (h2 : n % 2 = 0) : done * 2 + n / 2 * 4 = (done + n) * 2 := by omega

/-- Promoted tensor sizes: `4e` in, `2e` out. -/
theorem promoted_sizes (e : Nat) : 2 * e * 2 = 4 * e := by omega

def placeAt (at_ dst a : Nat) : Nat := at_ + (dst + a - 1) / a * a

/-- Compaction never overlaps tensors: each placed tensor still fits. -/
theorem placeAt_ge {at_ dst a : Nat} (ha : 0 < a) : at_ + dst ≤ placeAt at_ dst a := by
  unfold placeAt
  have h := Nat.div_add_mod (dst + a - 1) a
  have h2 := Nat.mod_lt (dst + a - 1) ha
  have h3 := Nat.mul_comm ((dst + a - 1) / a) a
  omega

/-- Compaction keeps every tensor start on the index alignment. -/
theorem placeAt_mod {at_ dst a : Nat} (h : at_ % a = 0) : placeAt at_ dst a % a = 0 := by
  unfold placeAt
  simp [Nat.add_mod, h]

/-- raw 16-bit index kinds map to the engine kinds 1 (BF16) and 3 (F16). -/
def engineKind (k : Nat) : Nat := if k = 4 then 1 else if k = 5 then 3 else k

/-- Kind 4 becomes BF16 (1). -/
theorem engineKind_bf16 : engineKind 4 = 1 := rfl
/-- Kind 5 becomes F16 (3). -/
theorem engineKind_f16 : engineKind 5 = 3 := rfl

end Strata.WeightsLoaderSegments
