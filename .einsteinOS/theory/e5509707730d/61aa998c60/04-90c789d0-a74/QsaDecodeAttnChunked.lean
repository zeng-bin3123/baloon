/-!
# MODULE: qsa_decode_attn_chunked
SOURCES: src/kernels/cuda/qsa_decode_attn.cu
Split-K decode attention over the selected cells: 64-cell chunks per (chunk, kv head, query) block with 12 query
heads per kv head, followed by a log-sum-exp merge. This module models chunk coverage, scratch layout and index bounds.

## Source semantics
* n_chunks = ceil(cap / 64); chunk c covers selected positions [64c, 64c+64); a chunk with no cells publishes m = -FLT_MAX, l = 0.
* Scratch per query holds acc (n_chunks * n_head * 256), m and l (n_chunks * n_head each); the stride is
  n_chunks * n_head * 258 + 64.
* Global query head h = kvh * 12 + hl; slot = kvh * n_chunks + chunk.

## Form template identifiers
- `CHUNK` = 64, `G` = 12, `HD` = 256
- `nChunks`, `nHere` : chunk count and cells in a chunk
- `slot`            : kvh * n_chunks + chunk
-/

namespace Strata.QsaDecode

-- § chunk coverage

/-- Number of 64-cell chunks for a capacity. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Cells handled by the chunk starting at `c0`. -/
def nHere (nIds c0 : Nat) : Nat := min 64 (nIds - c0)

/-- Every selected position falls in a chunk that exists. -/
theorem chunk_exists (i nIds cap : Nat) (hi : i < nIds) (hc : nIds ≤ cap) : i / 64 < nChunks cap := by
  unfold nChunks; omega

/-- A position lies inside its own chunk's window. -/
theorem cell_in_chunk (i : Nat) : i / 64 * 64 ≤ i ∧ i < i / 64 * 64 + 64 := by
  omega

/-- A chunk past the selection is empty and takes the -FLT_MAX path. -/
theorem nHere_empty (nIds c0 : Nat) (h : nIds ≤ c0) : nHere nIds c0 = 0 := by
  unfold nHere; omega

/-- A chunk that starts inside the selection has at least one cell. -/
theorem nHere_pos (nIds c0 : Nat) (h : c0 < nIds) : 0 < nHere nIds c0 := by
  unfold nHere; omega

/-- A chunk never holds more than 64 cells. -/
theorem nHere_le (nIds c0 : Nat) : nHere nIds c0 ≤ 64 := by
  unfold nHere; omega

/-- The last chunk of a full selection ends exactly at n_ids. -/
theorem nHere_tail (nIds c0 : Nat) (h : c0 < nIds) (h2 : nIds < c0 + 64) : c0 + nHere nIds c0 = nIds := by
  unfold nHere; omega

-- § scratch layout

/-- acc, m and l regions fit inside the declared per-query scratch size (a = n_chunks * n_head). -/
theorem scratch_fits (a : Nat) : a * 256 + a + a ≤ a * (256 + 2) + 64 := by
  omega

/-- Mixed-radix index bound. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m ha
  have h2 : (a + 1) * m = a * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The (kv head, chunk) slot is inside the partial buffers. -/
theorem slot_lt (kvh chunk nkv nch : Nat) (hk : kvh < nkv) (hc : chunk < nch) :
    kvh * nch + chunk < nkv * nch :=
  idx_lt kvh chunk nkv nch hk hc

-- § head and lane decomposition

/-- The global query head splits back into (kv head, local head) as written in the merge kernel. -/
theorem head_split (kvh hl : Nat) (h : hl < 12) :
    (kvh * 12 + hl) / 12 = kvh ∧ (kvh * 12 + hl) % 12 = hl := by
  omega

/-- Each lane's 8 dimensions stay inside the 256-wide head. -/
theorem lane_dims_lt (lane j : Nat) (hl : lane < 32) (hj : j < 8) : lane * 8 + j < 256 := by
  omega

/-- The second score half-slot of a lane stays in the 64-wide CHUNK row. -/
theorem lane_second_half (lane : Nat) (h : lane < 32) : lane + 32 < 64 := by
  omega

/-- Warp w handles heads w and w + 8, covering all 12 heads between warps 0..7. -/
theorem heads_covered (h : Nat) (hh : h < 12) : h < 8 ∨ (h - 8 < 8 ∧ h = (h - 8) + 8) := by
  omega

-- § merge weights

/-- An empty chunk (m = -FLT_MAX) is skipped, so the merge denominator only counts real chunks:
modelled on counts, populated chunks plus empty ones equal the chunk total. -/
theorem merge_chunks_partition (populated empty total : Nat) (h : populated + empty = total) :
    total - empty = populated := by
  omega

end Strata.QsaDecode
