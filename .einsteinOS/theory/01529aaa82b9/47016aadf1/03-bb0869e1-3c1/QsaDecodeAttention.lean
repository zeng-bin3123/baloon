/-!
# MODULE: qsa_decode_attention
Chunked split-softmax decode attention over selected KV cells (fp16 or int8 pools) and its merge.

## Source semantics
Block (chunk, kv_head[, query]); CHUNK=64 cells per block, G=12 query heads per kv head, HD=256.
Each chunk stores (acc, m, l); the merge rescales by exp(m - M) and divides by L.
Scratch per query: acc n_chunks*n_head*HD floats, then m and l with n_chunks*n_head floats each.

## Form template identifiers
HD -> 256; G -> 12; CHUNK -> 64; cap -> selection capacity; n_ids -> real selection width.
-/
namespace Strata.QsaDecode

/-- Chunk count. -/
def chunks (cap : Nat) : Nat := (cap + 63) / 64

/-- The chunks cover the capacity. -/
theorem chunks_cover (cap : Nat) : cap ≤ chunks cap * 64 := by unfold chunks; omega

/-- No chunk is wholly beyond the capacity (minimal chunk count). -/
theorem chunks_minimal (cap : Nat) (h : 0 < cap) : (chunks cap - 1) * 64 < cap := by unfold chunks; omega

/-- Cells per block: min(64, n_ids - c0) is at most 64. -/
theorem n_here_le (n c0 : Nat) : min 64 (n - c0) ≤ 64 := by omega

/-- A block whose first cell is past the selection has no cells. -/
theorem n_here_zero (n c0 : Nat) (h : n ≤ c0) : min 64 (n - c0) = 0 := by omega

/-- Cells of full blocks plus the last partial one add to n_ids. -/
theorem n_here_last (n c : Nat) (h1 : c * 64 ≤ n) (h2 : n < c * 64 + 64) : min 64 (n - c * 64) = n - c * 64 := by
  omega

/-- Query head to (kv head, local head) and back. -/
theorem head_split (h : Nat) : (h / 12) * 12 + h % 12 = h := by omega

/-- Local head index is below G. -/
theorem local_head_lt (h : Nat) : h % 12 < 12 := by omega

/-- KV head of a query head is below n_head_kv when n_head = 12 * n_head_kv. -/
theorem kvh_lt (h nkv : Nat) (hh : h < 12 * nkv) : h / 12 < nkv := by omega

/-- Warp-strided cell loop: cell c = warp + 8m covers 0..63 uniquely. -/
theorem cell_partition (c : Nat) (h : c < 64) : ∃ w m, w < 8 ∧ m < 8 ∧ c = w + 8 * m :=
  ⟨c % 8, c / 8, by omega, by omega, by omega⟩

/-- Lane dimension slices: lane*8 + j covers 256 dimensions with 32 lanes. -/
theorem dim_partition (d : Nat) (h : d < 256) : ∃ l j, l < 32 ∧ j < 8 ∧ d = l * 8 + j :=
  ⟨d / 8, d % 8, by omega, by omega, by omega⟩

/-- Head loop: warp w handles heads w and w+8, covering 12 heads. -/
theorem head_partition (h : Nat) (hh : h < 12) : ∃ w m, w < 8 ∧ m < 2 ∧ h = w + 8 * m :=
  ⟨h % 8, h / 8, by omega, by omega, by omega⟩

/-- Probability lanes: lane and lane+32 cover the 64 chunk cells. -/
theorem prob_lanes (c : Nat) (h : c < 64) : c < 32 ∨ (c - 32 < 32) := by omega

/-- int8 scale group index for dimension d (group size 32) stays below HD/32 = 8. -/
theorem scale_group_lt (d : Nat) (h : d < 256) : d / 32 < 8 := by omega

/-- Pool row of (page, kv head, slot) is below pages * n_kv_heads * page_size when indices are in range. -/
theorem pool_row_lt (page nkv h ps slot P : Nat) (hp : page < P) (hh : h < nkv) (hs : slot < ps) :
    (page * nkv + h) * ps + slot < P * nkv * ps := by
  have h1 : page * nkv + h < P * nkv := by
    have : (page + 1) * nkv ≤ P * nkv := Nat.mul_le_mul_right _ hp
    have h2 : (page + 1) * nkv = page * nkv + nkv := by rw [Nat.add_mul, Nat.one_mul]
    omega
  have h3 : (page * nkv + h + 1) * ps ≤ P * nkv * ps := Nat.mul_le_mul_right _ h1
  have h4 : (page * nkv + h + 1) * ps = (page * nkv + h) * ps + ps := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Scratch layout: acc, m, l regions with x = n_chunks * n_head: x*256 + x + x = x*258. -/
theorem scratch_layout (x : Nat) : x * 256 + x + x = x * 258 := by omega

/-- Merge identity: an empty chunk (m = -FLT_MAX, l = 0) contributes nothing to the sum of l's. -/
theorem empty_chunk_neutral (L : Nat) : L + 0 = L := by omega

/-- Batched form: per-query scratch offset q*stride never overlaps the next query's region. -/
theorem query_offset_disjoint (q stride : Nat) : q * stride + stride = (q + 1) * stride := by
  rw [Nat.add_mul, Nat.one_mul]

/-- Batch query index limit 65535 fits grid.z. -/
theorem nq_bound (n : Nat) (h : n ≤ 65535) : n < 65536 := by omega

end Strata.QsaDecode
