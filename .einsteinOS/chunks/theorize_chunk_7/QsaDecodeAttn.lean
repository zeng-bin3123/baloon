/-!
# MODULE: qsa_decode_attn
Chunking, scratch layout and head mapping of the split-KV decode attention over the selected cells.

## Source semantics
Selected ids are split into chunks of 64; chunk c handles ids 64c .. 64c+63. Each of the 12 query heads per KV head stores a partial (acc, m, l) per chunk at slot kvh*n_chunks + chunk. A final merge combines the partials. Global head h maps to kv head h/12 and local head h%12. Scratch holds n_chunks*n_head*(256+2) floats plus 64.

## Form template identifiers
attn_chunk_kernel -> chunk_partial, attn_merge_kernel -> merge_partials, CHUNK -> chunk_size, G -> group_size
-/
namespace Strata.QsaDecodeAttn

/-- Number of chunks for a capacity. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

-- § chunking

/-- Chunks cover the capacity. -/
theorem chunks_cover (cap : Nat) : cap ≤ nChunks cap * 64 := by unfold nChunks; omega

/-- Every id below n_ids <= cap lands in a valid chunk. -/
theorem id_chunk_lt (j n cap : Nat) (hj : j < n) (hn : n ≤ cap) : j / 64 < nChunks cap := by unfold nChunks; omega

/-- Chunk and offset recompose the id. -/
theorem id_decomp (j : Nat) : (j / 64) * 64 + j % 64 = j := by omega

/-- Cells handled by a chunk. -/
def nHere (nIds c : Nat) : Nat := min 64 (nIds - c * 64)

/-- A chunk handles at most 64 cells. -/
theorem nHere_le (nIds c : Nat) : nHere nIds c ≤ 64 := by unfold nHere; omega

/-- Chunks past the end handle nothing. -/
theorem nHere_zero (nIds c : Nat) (h : nIds ≤ c * 64) : nHere nIds c = 0 := by unfold nHere; omega

/-- A chunk below the last full one handles a full 64. -/
theorem nHere_full (nIds c : Nat) (h : (c + 1) * 64 ≤ nIds) : nHere nIds c = 64 := by unfold nHere; omega

/-- Each lane covers two probability slots, lane and lane+32, giving all 64. -/
theorem prob_slots (c : Nat) (h : c < 64) : c < 32 ∨ (c - 32 < 32) := by omega

-- § head mapping

/-- Global head recomposes from kv head and local head. -/
theorem head_decomp (h : Nat) : (h / 12) * 12 + h % 12 = h := by omega

/-- A global head below 12*n_kv has a valid kv head. -/
theorem kvh_lt (h nk : Nat) (hh : h < 12 * nk) : h / 12 < nk := by omega

/-- The local head is below 12. -/
theorem hl_lt (h : Nat) : h % 12 < 12 := by omega

-- § slots and scratch

/-- A (kv head, chunk) slot is inside n_kv_heads * n_chunks. -/
theorem slot_lt (kvh chunk nk nc : Nat) (hk : kvh < nk) (hc : chunk < nc) : kvh * nc + chunk < nk * nc := by
  have h := Nat.mul_le_mul_right nc (show kvh + 1 ≤ nk by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Scratch floats. -/
def scratchFloats (nc nHead : Nat) : Nat := nc * nHead * (256 + 2) + 64

/-- The three scratch regions (acc, m, l) fit in the allocation. -/
theorem scratch_fits (nc nHead : Nat) : nc * nHead * 256 + nc * nHead + nc * nHead ≤ scratchFloats nc nHead := by unfold scratchFloats; omega

end Strata.QsaDecodeAttn
