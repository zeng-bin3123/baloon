/-!
# MODULE: qsa_decode_attention
Split-K decode attention over gathered cells: 64-cell chunks per (chunk, kv head), online-softmax partials
(max, sum, acc), then a merge kernel. FP16 or INT8+scale KV pools. 12 query heads per kv head, head_dim 256.

## Source semantics
Chunk c covers cells [64c, 64c+n_here). Empty chunks emit m = -FLT_MAX, l = 0 and are skipped by the merge.
Merge: M = max m; L = sum l*exp(m-M); out = acc/L or 0 if L = 0.

## Form template identifiers
chunk -> split_block; part_m/part_l/part_acc -> partial_max/partial_sum/partial_acc
-/
namespace Strata.QsaDecode

/-- Number of 64-cell chunks. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Chunks cover the capacity. -/
theorem chunks_cover (cap : Nat) : cap ≤ nChunks cap * 64 := by unfold nChunks; omega

/-- Cells handled by chunk `c` of `n` ids. -/
def nHere (n c : Nat) : Nat := min 64 (n - c * 64)

/-- A chunk handles at most 64 cells. -/
theorem nHere_le (n c : Nat) : nHere n c ≤ 64 := by unfold nHere; omega

/-- Chunk sizes sum correctly for the first chunk when the selection fits one chunk. -/
theorem nHere_single (n : Nat) (h : n ≤ 64) : nHere n 0 = n := by unfold nHere; omega

/-- Chunks past the end are empty. -/
theorem nHere_empty (n c : Nat) (h : n ≤ c * 64) : nHere n c = 0 := by unfold nHere; omega

/-- Scratch floats for a capacity, with 12 heads x kv heads = n_head. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * (256 + 2) + 64

/-- Scratch holds acc (HD per head) plus m and l (one each) per chunk-head. -/
theorem scratch_split (cap nHead : Nat) :
    scratchFloats cap nHead = nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead + 64 := by
  unfold scratchFloats
  generalize nChunks cap * nHead = m
  omega

/-- Query head index decomposes into kv head and local head. -/
theorem head_decomp (h : Nat) : (h / 12) * 12 + h % 12 = h := by omega

/-- Each of the 12 local heads is below 12. -/
theorem local_head_lt (h : Nat) : h % 12 < 12 := by omega

/-- KV head of a query head of 24 is below 2. -/
theorem kv_head_lt (h : Nat) (hh : h < 24) : h / 12 < 2 := by omega

/-- INT8 scale group index stays inside the row (256 dims, groups of KV_Q8_GROUP = 32 assumed). -/
theorem scale_group_lt (d : Nat) (hd : d < 256) : d / 32 < 8 := by omega

end Strata.QsaDecode
