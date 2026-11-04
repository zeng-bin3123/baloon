/-!
# MODULE: qsa_decode_attn_chunked
Chunked split-softmax decode attention over selected cells: each block handles 64 cells for one KV head
(12 query heads per KV head, head_dim 256), writes partial (acc, max, sum), and a merge kernel combines
chunks with rescaling by the global max. Supports a batched form (blockIdx.z = query) with per-query scratch.

## Source semantics
* `n_chunks = ceil(cap / 64)`; chunk `c` handles cells `[64c, min(64c+64, n_ids))`, empty chunks write `m = -FLT_MAX`, `l = 0`.
* Slot index `kvh*n_chunks + chunk`; partial accumulators indexed `(slot*12 + h)*256 + t`.
* Scratch floats = `n_chunks * n_head * (256 + 2) + 64`: acc region, then m region, then l region.
* Merge: `M = max_c m_c`; each non-empty chunk is weighted `exp(m_c - M) ≤ 1`; empty chunks are skipped.
* Global query head `h` maps to KV head `h / 12` and local head `h % 12`.
* Batched: query `z` owns scratch window `[z*stride, (z+1)*stride)`; windows are disjoint.

## Form template identifiers
* `HD`, `G`, `CHUNK`, `WARPS` -> 256, 12, 64, 8
* `nChunks`                   -> ceil division of capacity by chunk size
* `scratchFloats`             -> qsa_decode_attn_scratch_floats
-/

namespace Strata.QsaDecodeAttn

/-- Head dimension. -/
def HD : Nat := 256
/-- Query heads per KV head. -/
def G : Nat := 12
/-- Cells per block. -/
def CHUNK : Nat := 64
/-- Warps per block. -/
def WARPS : Nat := 8

/-- Row-major index lemma. -/
theorem idx_lt {a b i j : Nat} (hi : i < a) (hj : j < b) : i * b + j < a * b := by
  have h1 : (i + 1) * b ≤ a * b := Nat.mul_le_mul_right b (Nat.succ_le_of_lt hi)
  have h2 : (i + 1) * b = i * b + b := Nat.succ_mul i b
  omega

/-- Number of chunks for capacity `cap`. -/
def nChunks (cap : Nat) : Nat := (cap + CHUNK - 1) / CHUNK

/-- Chunks cover the capacity. -/
theorem nChunks_covers (cap : Nat) : cap ≤ nChunks cap * CHUNK := by
  unfold nChunks CHUNK
  omega

/-- No chunk beyond the last is allocated: for positive capacity, the last chunk starts below `cap`. -/
theorem nChunks_minimal (cap : Nat) (h : 0 < cap) : (nChunks cap - 1) * CHUNK < cap := by
  unfold nChunks CHUNK
  omega

/-- Cell count of chunk `c` for `n` selected cells (signed difference clipped at the chunk size). -/
def nHere (n c : Nat) : Nat := min CHUNK (n - c * CHUNK)

/-- Cell `j < n` lives in chunk `j/64` at local position `j%64`, and that local position is a valid cell of the chunk. -/
theorem cell_in_chunk (j n : Nat) (h : j < n) : j % CHUNK < nHere n (j / CHUNK) := by
  unfold nHere CHUNK
  omega

/-- A chunk handles at most 64 cells. -/
theorem nHere_le (n c : Nat) : nHere n c ≤ CHUNK := by
  unfold nHere
  omega

/-- Chunks at or past the end of the selection are empty. -/
theorem nHere_empty (n c : Nat) (h : n ≤ c * CHUNK) : nHere n c = 0 := by
  unfold nHere
  omega

/-- Partial-state slot is within `n_kv_heads * n_chunks`. -/
theorem slot_in_range {kvh chunk n_kv_heads n_chunks : Nat} (hk : kvh < n_kv_heads) (hc : chunk < n_chunks) :
    kvh * n_chunks + chunk < n_kv_heads * n_chunks := idx_lt hk hc

/-- Partial accumulator index `(slot*G + h)*HD + t` is within `slots*G*HD`. -/
theorem acc_index_in_range {slot h t slots : Nat} (hs : slot < slots) (hh : h < G) (ht : t < HD) :
    (slot * G + h) * HD + t < slots * G * HD := by
  have h1 : slot * G + h < slots * G := idx_lt hs hh
  exact idx_lt h1 ht

/-- Scratch float count. -/
def scratchFloats (cap n_head : Nat) : Nat := nChunks cap * n_head * (HD + 2) + 64

/-- The acc, m and l regions tile into the scratch size. -/
theorem scratch_regions_fit (cap n_head : Nat) :
    nChunks cap * n_head * HD + nChunks cap * n_head + nChunks cap * n_head ≤ scratchFloats cap n_head := by
  unfold scratchFloats HD
  omega

/-- Batched queries own disjoint scratch windows when each stays within one stride. -/
theorem batch_windows_disjoint {z1 z2 stride : Nat} (h : z1 < z2) :
    z1 * stride + stride ≤ z2 * stride := by
  have h1 : (z1 + 1) * stride ≤ z2 * stride := Nat.mul_le_mul_right stride (Nat.succ_le_of_lt h)
  have h2 : (z1 + 1) * stride = z1 * stride + stride := Nat.succ_mul z1 stride
  omega

-- § head mapping

/-- A global head splits into KV head and local head and reassembles. -/
theorem head_split (h : Nat) : G * (h / G) + h % G = h ∧ h % G < G := by
  unfold G
  omega

/-- The shared-memory query row of local head `hl` equals the global query row. -/
theorem query_row_agree (kvh hl d : Nat) (hl' : hl < G) (hd : d < HD) :
    (kvh * G) * HD + (hl * HD + d) = (kvh * G + hl) * HD + d := by
  unfold G HD at *
  omega

/-- The grid's KV-head dimension times 12 covers all query heads. -/
theorem heads_cover (n_kv_heads h : Nat) (hh : h < G * n_kv_heads) : h / G < n_kv_heads := by
  unfold G at *
  omega

-- § warp partitioning

/-- Cell `c < 64` is processed by warp `c % 8` (strided by 8). -/
theorem cell_warp (c : Nat) (hc : c < CHUNK) : c % WARPS < WARPS ∧ (c % WARPS) + WARPS * (c / WARPS) = c := by
  unfold CHUNK WARPS at *
  omega

/-- Each lane holds 8 consecutive dims: 32 lanes cover 256 = HD. -/
theorem lane_dims (lane j : Nat) (hl : lane < 32) (hj : j < 8) : lane * 8 + j < HD := by
  unfold HD
  omega

/-- Head `h < 12` is handled by warp `h % 8`, as either the first or second of its heads. -/
theorem head_warp (h : Nat) (hh : h < G) : h % WARPS < WARPS ∧ h / WARPS < 2 := by
  unfold G WARPS at *
  omega

-- § merge

/-- Every chunk max is bounded by the folded global max. -/
theorem le_foldr_max (b : Int) : ∀ (l : List Int) (m : Int), m ∈ l → m ≤ l.foldr max b
  | [], m, h => by simp at h
  | x :: l, m, h => by
    simp only [List.foldr_cons]
    rcases List.mem_cons.mp h with rfl | h'
    · omega
    · have := le_foldr_max b l m h'
      omega

/-- Hence the rescale exponent `m_c - M` is nonpositive, so the merge weight `exp(m_c - M) ≤ 1`. -/
theorem merge_exponent_nonpos (b : Int) (l : List Int) (m : Int) (h : m ∈ l) : m - l.foldr max b ≤ 0 := by
  have := le_foldr_max b l m h
  omega

end Strata.QsaDecodeAttn
