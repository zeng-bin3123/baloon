/-!
# MODULE: qsa_decode_chunking
SOURCES: src/kernels/cuda/qsa_decode_attn.cu
Chunked split-KV decode attention: the selected cells are cut into chunks of 64, each chunk produces a
partial (acc, m, l), and a merge kernel combines the partials. This module covers the index arithmetic.

## Source semantics
* n_chunks = ceil(cap / 64); chunk k covers cells [64k, 64k + n_here) with n_here = min(64, n_ids - 64k).
* A chunk past n_ids is empty (n_here <= 0) and writes the sentinel (m = -FLT_MAX, l = 0).
* Scratch = n_chunks * n_head * (256 + 2) + 64 floats, laid out [acc][m][l].
* slot = kvh * n_chunks + chunk; query head h = kvh * 12 + hl.

## Form template identifiers
nChunks <- n_chunks, chunkSize <- n_here, scratchFloats <- qsa_decode_attn_scratch_floats,
slotOf <- slot.
-/

namespace Strata.QsaDecode

-- § chunk coverage

/-- Number of chunks for capacity `cap`. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Cells handled by chunk `k` (0 when the chunk is past the selection). -/
def chunkSize (n k : Nat) : Nat := min 64 (n - 64 * k)

/-- Total cells covered by the first `m` chunks. -/
def total (n : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => total n m + chunkSize n m

/-- A chunk past the selection is empty. -/
theorem chunkSize_zero {n k : Nat} (h : n ≤ 64 * k) : chunkSize n k = 0 := by
  unfold chunkSize; omega

/-- A fully inside chunk holds 64 cells. -/
theorem chunkSize_full {n k : Nat} (h : 64 * (k + 1) ≤ n) : chunkSize n k = 64 := by
  unfold chunkSize; omega

/-- Chunks never hold more than 64 cells. -/
theorem chunkSize_le (n k : Nat) : chunkSize n k ≤ 64 := by
  unfold chunkSize; omega

/-- The first `m` chunks cover exactly `min n (64 m)` cells. -/
theorem total_eq (n m : Nat) : total n m = min n (64 * m) := by
  induction m with
  | zero => simp [total]
  | succ m ih => simp only [total, chunkSize, ih]; omega

/-- With `n_ids <= cap`, all `nChunks cap` chunks cover every selected cell exactly once. -/
theorem total_cover {n cap : Nat} (h : n ≤ cap) : total n (nChunks cap) = n := by
  rw [total_eq]; unfold nChunks; omega

/-- The chunk count times 64 reaches the capacity. -/
theorem nChunks_mul_ge (cap : Nat) : cap ≤ 64 * nChunks cap := by
  unfold nChunks; omega

/-- A selected cell `i < n` lives in chunk `i / 64`, which is a valid chunk. -/
theorem cell_chunk_lt {i cap : Nat} (h : i < cap) : i / 64 < nChunks cap := by
  unfold nChunks; omega

/-- Its in-chunk offset is inside that chunk's `n_here`. -/
theorem cell_offset_lt {i n : Nat} (h : i < n) : i % 64 < chunkSize n (i / 64) := by
  unfold chunkSize; omega

-- § scratch layout

/-- Scratch floats for capacity `cap`. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * (256 + 2) + 64

/-- The acc region fits in the scratch. -/
theorem acc_region_fits (cap nHead : Nat) :
    nChunks cap * nHead * 256 ≤ scratchFloats cap nHead := by
  unfold scratchFloats; omega

/-- The m and l regions, placed after acc, end inside the scratch. -/
theorem ml_regions_fit (cap nHead : Nat) :
    nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead
      ≤ scratchFloats cap nHead := by
  unfold scratchFloats; omega

-- § slots and heads

/-- Partial slot index of (kv head, chunk). -/
def slotOf (kvh chunks c : Nat) : Nat := kvh * chunks + c

/-- A valid (kv head, chunk) pair has a slot inside `nKvH * chunks`. -/
theorem slotOf_lt {kvh c chunks nKvH : Nat} (hk : kvh < nKvH) (hc : c < chunks) :
    slotOf kvh chunks c < nKvH * chunks := by
  unfold slotOf
  have h1 : (kvh + 1) * chunks ≤ nKvH * chunks := Nat.mul_le_mul_right chunks hk
  have h2 : (kvh + 1) * chunks = kvh * chunks + chunks := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The merge kernel's head decomposition `h = kvh * 12 + hl` is exact. -/
theorem head_decomp (h : Nat) : h = (h / 12) * 12 + h % 12 := by omega

/-- The local head index is below 12. -/
theorem local_head_lt (h : Nat) : h % 12 < 12 := by omega

end Strata.QsaDecode
