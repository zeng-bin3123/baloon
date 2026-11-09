/-!
# MODULE: qsa_decode_attention
SOURCES: src/kernels/cuda/qsa_decode_attn.cu
Split-KV decode attention over the QSA-selected cells: a chunk kernel (64 cells per block, one block per
(chunk, KV head [, query])) emitting per-chunk running max, exp-sum and weighted value accumulators for the 12
query heads sharing a KV head, then a merge kernel combining chunks by online-softmax rescaling. fp16 and int8
(grouped scale) pools are supported.

## Source semantics
* `HD = 256`, `G = 12` query heads per KV head, `CHUNK = 64`, 256 threads (8 warps).
* Scratch layout per query: `[acc: chunks*n_head*HD][m: chunks*n_head][l: chunks*n_head]` plus 64 slack floats.
* Chunk slot `kvh*n_chunks + chunk`; global query head `h = kvh*12 + hl`.
* Empty chunks write `m = -FLT_MAX`, `l = 0` and are skipped in the merge.

## Form template identifiers
* `HD` -> head_dim, `G` -> group_size, `CHUNK` -> cells_per_chunk, `scratchFloats` -> scratch_size.
-/

namespace Strata.QsaDecodeAttn

/-- Head dimension. -/
def HD : Nat := 256
/-- Query heads per KV head. -/
def G : Nat := 12
/-- Cells per chunk. -/
def CHUNK : Nat := 64
/-- Threads per block. -/
def THREADS : Nat := 256

/-- Scratch size (floats) from the host formula `chunks * n_head * (HD + 2) + 64`. -/
def scratchFloats (chunks n_head : Nat) : Nat := chunks * n_head * (HD + 2) + 64

/-- Eight warps per block. -/
theorem warps : THREADS / 32 = 8 := by decide

/-- Each lane holds 8 of the 256 dimensions: 32 lanes x 8 = HD. -/
theorem lane_dims : 32 * 8 = HD := by decide

/-- One thread per dimension in the value loop. -/
theorem thread_per_dim : THREADS = HD := by decide

/-- The 64 scores of a head are covered by two per lane (`lane`, `lane+32`). -/
theorem two_per_lane : 2 * 32 = CHUNK := by decide

/-- Query tile in shared memory is `12 * 256` floats = 12 KB. -/
theorem sq_bytes : G * HD * 4 = 12288 := by decide

/-- Heads `warp` and `warp + 8` cover all 12 heads. -/
theorem head_cover (h : Nat) (hh : h < 12) : ∃ w, w < 8 ∧ (h = w ∨ h = w + 8) := by
  by_cases h8 : h < 8
  · exact ⟨h, h8, Or.inl rfl⟩
  · exact ⟨h - 8, by omega, Or.inr (by omega)⟩

/-- Cells `warp, warp+8, ...` cover all 64 cells of a chunk. -/
theorem cell_cover (c : Nat) (h : c < 64) : ∃ w k, w < 8 ∧ k < 8 ∧ c = w + 8 * k := by
  exact ⟨c % 8, c / 8, by omega, by omega, by omega⟩

/-- Cell loop ownership is unique. -/
theorem cell_unique (w k w' k' : Nat) (hw : w < 8) (hw' : w' < 8)
    (h : w + 8 * k = w' + 8 * k') : w = w' ∧ k = k' := by omega

/-- The chunk count covers the capacity. -/
theorem chunks_cover (cap : Nat) : cap ≤ (cap + 63) / 64 * 64 := by omega

/-- Cell `c0 + t` of the last chunk is below `chunks*64`. -/
theorem chunk_cell_lt (cap chunk t : Nat) (hc : chunk < (cap + 63) / 64) (ht : t < 64) :
    chunk * 64 + t < (cap + 63) / 64 * 64 := by omega

/-- The three scratch regions fit in the scratch size (acc, then m, then l). -/
theorem scratch_regions_fit (chunks n_head : Nat) :
    chunks * n_head * 256 + chunks * n_head + chunks * n_head ≤ scratchFloats chunks n_head := by
  unfold scratchFloats HD; omega

/-- The `m` region starts exactly where `acc` ends and `l` where `m` ends. -/
theorem scratch_offsets (chunks n_head : Nat) :
    chunks * n_head * 256 + chunks * n_head = chunks * n_head * 257 := by omega

/-- A slot index `kvh*n_chunks + chunk` is below `n_kv_heads * n_chunks`. -/
theorem slot_lt (kvh chunk n_kv n_chunks : Nat) (hk : kvh < n_kv) (hc : chunk < n_chunks) :
    kvh * n_chunks + chunk < n_kv * n_chunks := by
  have h1 : (kvh + 1) * n_chunks ≤ n_kv * n_chunks := Nat.mul_le_mul_right _ (by omega)
  have h2 : (kvh + 1) * n_chunks = kvh * n_chunks + n_chunks := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Global head decomposition `h = kvh*12 + hl` round-trips. -/
theorem head_roundtrip (kvh hl : Nat) (hl' : hl < 12) :
    (kvh * 12 + hl) / 12 = kvh ∧ (kvh * 12 + hl) % 12 = hl := by omega

/-- Conversely every global head splits into a KV head and a local head. -/
theorem head_split (h : Nat) : h / 12 * 12 + h % 12 = h ∧ h % 12 < 12 := by omega

/-- The geometry check `n_head = 12 * n_kv_heads` makes every KV head have exactly 12 query heads. -/
theorem kv_head_lt (h n_kv : Nat) (hh : h < 12 * n_kv) : h / 12 < n_kv := by omega

/-- Batched grid: query row stride is `n_kv_heads * 12 * 256` floats; rows are disjoint. -/
theorem q_rows_disjoint (z z' n_kv : Nat) (h : z < z') :
    z * (n_kv * 12 * 256) + n_kv * 12 * 256 ≤ z' * (n_kv * 12 * 256) := by
  have := Nat.mul_le_mul_right (n_kv * 12 * 256) (show z + 1 ≤ z' by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- Online-softmax merge identity at scalar level: rescaling a partial sum from its own max `m` to a global
max `M` multiplies by `exp(m - M)`; here as the multiplicative law on exponent differences (naturals). -/
theorem exp_shift_compose (m1 m2 M : Nat) (h1 : m1 ≤ m2) (h2 : m2 ≤ M) :
    (M - m1) = (M - m2) + (m2 - m1) := by omega

/-- Merge guard: a chunk whose max equals the sentinel contributes nothing (abstractly `0 * x = 0`). -/
theorem empty_chunk_contributes_zero (x : Int) : 0 * x = 0 := by omega

/-- Chunks beyond the selection are empty: `c0 >= n_ids` gives `n_ids - c0 <= 0`. -/
theorem empty_chunk (c0 n_ids : Nat) (h : n_ids ≤ c0) : (n_ids : Int) - c0 ≤ 0 := by omega

end Strata.QsaDecodeAttn
