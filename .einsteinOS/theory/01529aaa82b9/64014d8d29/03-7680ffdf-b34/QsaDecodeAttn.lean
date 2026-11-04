/-!
# MODULE: qsa_decode_attn
SOURCES: src/kernels/cuda/qsa_decode_attn.cu
Chunked split-K decode attention over the selected cells (GQA, 12 query heads per kv head, head_dim 256),
with per-chunk (max, exp-sum, accumulator) partials merged by a second kernel; supports a batched form
(blockIdx.z) and fp16 or int8+scale KV pools.

## Source semantics
* Cells are processed in chunks of 64; chunk c owns selection indices [64c, 64c+64) intersected with [0,n_ids).
* A chunk past n_ids writes the sentinel (m = -FLT_MAX, l = 0) and is skipped by the merge.
* Query head h reads kv head h / 12 and local head h % 12.
* Merge: M = max over chunk maxima, weights exp(m_c - M) <= 1, result acc / L.
* Scratch = chunks * n_head * (HD + 2) + 64 floats; batch elements are strided by that amount.

## Form template identifiers
CHUNK -> 64; G -> 12; HD -> 256; chunk max -> `maxList`.
-/

namespace QsaDecodeAttn

-- § helpers

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § chunking

/-- Number of chunks for a capacity. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Every selection index below the capacity lies in an existing chunk. -/
theorem index_in_chunk (i cap : Nat) (h : i < cap) : i / 64 < nChunks cap := by
  unfold nChunks; omega

/-- Cells handled by chunk `c`: min(64, n_ids - 64c) (may be non-positive for surplus chunks). -/
def nHere (n c : Int) : Int := min 64 (n - c * 64)

/-- A surplus chunk has nothing to do and takes the sentinel path. -/
theorem surplus_chunk_empty (n c : Int) (h : n ≤ c * 64) : nHere n c ≤ 0 := by
  unfold nHere; omega

/-- A live chunk handles between 1 and 64 cells. -/
theorem live_chunk_bounds (n c : Int) (h : c * 64 < n) : 0 < nHere n c ∧ nHere n c ≤ 64 := by
  unfold nHere; omega

/-- Every selection index `i < n` is handled by chunk `i/64` at lane `i%64 < nHere`. -/
theorem every_cell_handled (i n : Nat) (h : i < n) :
    (i : Int) % 64 < nHere (n : Int) ((i / 64 : Nat) : Int) ∧ ((i / 64 : Nat) : Int) * 64 + (i : Int) % 64 = i := by
  unfold nHere; omega

-- § GQA head mapping

/-- Query head -> kv head is integer division by 12, and stays below the kv head count. -/
theorem gqa_head_decode (h nKvHeads : Nat) (hh : h < 12 * nKvHeads) :
    h / 12 < nKvHeads ∧ (h / 12) * 12 + h % 12 = h := by omega

/-- In the real geometry the map is `[0 x12, 1 x12]`, which differs from `h % 2` (e.g. at h = 1). -/
theorem gqa_div_differs_from_mod : (1 : Nat) / 12 ≠ 1 % 2 := by decide

/-- Each lane holds 8 consecutive dims; 32 lanes cover the 256-wide head exactly once. -/
theorem lane_cover (d : Nat) (h : d < 256) : d = (d / 8) * 8 + d % 8 ∧ d / 8 < 32 := by omega

-- § scratch layout

/-- Scratch float count. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * (256 + 2) + 64

/-- The three partial arrays (acc, m, l) fit in the scratch. -/
theorem partials_fit (cap nHead : Nat) :
    nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead ≤ scratchFloats cap nHead := by
  unfold scratchFloats; omega

/-- Partial slot index is in range. -/
theorem slot_lt (kvh chunk nKvHeads nCh : Nat) (h1 : kvh < nKvHeads) (h2 : chunk < nCh) :
    kvh * nCh + chunk < nKvHeads * nCh := idx2_lt kvh chunk nKvHeads nCh h1 h2

/-- Accumulator index `(slot*12 + h)*256 + t` is in range. -/
theorem acc_index_lt (slot h t S : Nat) (hs : slot < S) (hh : h < 12) (ht : t < 256) :
    (slot * 12 + h) * 256 + t < S * 12 * 256 := by
  have h1 : slot * 12 + h < S * 12 := idx2_lt slot h S 12 hs hh
  exact idx2_lt (slot * 12 + h) t (S * 12) 256 h1 ht

-- § merge

/-- Running maximum over a list of chunk maxima. -/
def maxList : Int → List Int → Int
  | a, [] => a
  | a, x :: xs => maxList (max a x) xs

/-- The running maximum dominates its seed. -/
theorem maxList_ge_init : ∀ (l : List Int) (a : Int), a ≤ maxList a l
  | [], a => Int.le_refl a
  | x :: xs, a => Int.le_trans (by omega) (maxList_ge_init xs (max a x))

/-- The merged maximum dominates every chunk maximum, so every merge exponent `m_c - M` is `<= 0`. -/
theorem maxList_ge_mem : ∀ (l : List Int) (a : Int) (m : Int), m ∈ l → m ≤ maxList a l
  | [], a, m, hm => by cases hm
  | x :: xs, a, m, hm => by
    rcases List.mem_cons.1 hm with h | h
    · subst h
      exact Int.le_trans (by omega) (maxList_ge_init xs (max a m))
    · exact maxList_ge_mem xs (max a x) m h

/-- Merge exponents are non-positive. -/
theorem merge_exponent_nonpos (l : List Int) (a m : Int) (hm : m ∈ l) : m - maxList a l ≤ 0 := by
  have := maxList_ge_mem l a m hm
  omega

-- § batching

/-- Distinct batch elements occupy disjoint, ordered scratch ranges. -/
theorem batch_disjoint (z1 z2 stride : Nat) (h : z1 < z2) : z1 * stride + stride ≤ z2 * stride := by
  have h1 : (z1 + 1) * stride ≤ z2 * stride := Nat.mul_le_mul_right stride (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- The grid's merge kernel writes `attn[(z*n_head + h)*256 + d]`, in range for z < n_q. -/
theorem batch_attn_index_lt (z h d nQ nHead : Nat) (hz : z < nQ) (hh : h < nHead) (hd : d < 256) :
    (z * nHead + h) * 256 + d < nQ * nHead * 256 := by
  have h1 : z * nHead + h < nQ * nHead := idx2_lt z h nQ nHead hz hh
  exact idx2_lt (z * nHead + h) d (nQ * nHead) 256 h1 hd

end QsaDecodeAttn
