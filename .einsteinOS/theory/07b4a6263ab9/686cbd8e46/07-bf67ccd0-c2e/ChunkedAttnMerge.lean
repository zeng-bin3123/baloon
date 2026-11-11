/-!
# MODULE: chunked_attn_merge
SOURCES: src/kernels/cuda/qsa_decode_attn.cu, src/kernels/cuda/qsa.cu
Decode attention over the selected cells: `attn_chunk_kernel` splits the selection into 64-cell chunks
per KV head (12 query heads share one KV head), writes per-chunk `(acc, m, l)` partials, and
`attn_merge_kernel` combines them with the max-rescaled log-sum-exp merge. `qsa_attend_kernel` is the
one-block-per-head reference form with the same GQA mapping.

## Source semantics
* `n_chunks = ceil(cap / 64)`; selected index `c < n_ids ≤ cap` lies in chunk `c / 64 < n_chunks`.
* A chunk has `n_here = min(64, n_ids - c0)` live cells and is empty (sentinel `m = -FLT_MAX`,
  `l = 0`) iff `c0 ≥ n_ids`; empty chunks are skipped by the merge.
* GQA: query head `h` reads KV head `h / G` (`iv2 = iq2 / rv2`), never `h % n_head_kv`.
* Scratch per query: `acc` (`n_chunks*n_head*256`), then `m`, then `l` (`n_chunks*n_head` each), all
  inside `stride = n_chunks*n_head*258 + 64`; partial slots `(kvh*n_chunks + c)*G + hl` stay in range.
* Merge: with `M = max m_c` every weight `exp(m_c - M) ≤ 1`, so the rescale never overflows.

## Form template identifiers
`CHUNK` → `64`, `G` → `12`, `HD` → `256`, `n_chunks` → `nChunks`, `n_here` → `nHere`,
`slot` → `partSlot`, `qsa_decode_attn_scratch_floats` → `scratchFloats`, merge max `M` → `mergeMax`.
-/

namespace Strata.DecodeAttn

/-- `n_chunks = ceil(cap / 64)`. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Live cells of chunk `ch`. -/
def nHere (nIds ch : Nat) : Nat := min 64 (nIds - ch * 64)

/-- Every selected cell lies in a launched chunk. -/
theorem chunk_of_cell_lt (cap nIds c : Nat) (hc : c < nIds) (hn : nIds ≤ cap) : c / 64 < nChunks cap := by
  unfold nChunks
  omega

/-- The cell lies at offset `c % 64 < n_here` inside its chunk. -/
theorem cell_live_in_chunk (nIds c : Nat) (hc : c < nIds) : c % 64 < nHere nIds (c / 64) := by
  unfold nHere
  omega

/-- A chunk is empty (writes the sentinel) iff it starts at or past `n_ids`. -/
theorem chunk_empty_iff (nIds ch : Nat) : nHere nIds ch = 0 ↔ nIds ≤ ch * 64 := by
  unfold nHere
  omega

/-- A launched chunk never holds more than 64 cells (fits `sp[G][CHUNK]`). -/
theorem nHere_le (nIds ch : Nat) : nHere nIds ch ≤ 64 := Nat.min_le_left _ _

-- § GQA

/-- Query head `h < 12 * n_kv` reads KV head `h / 12 < n_kv` as its local head `h % 12`. -/
theorem gqa_decode (h nKv : Nat) (hh : h < 12 * nKv) : h / 12 < nKv ∧ h = h / 12 * 12 + h % 12 := by
  omega

/-- The general `qsa_attend` mapping `kv = h / (n_head / n_head_kv)` lands on a valid KV head. -/
theorem gqa_general (h g nKv : Nat) (hg : 0 < g) (hh : h < g * nKv) : h / g < nKv := by
  rw [Nat.mul_comm] at hh
  exact (Nat.div_lt_iff_lt_mul hg).mpr hh

/-- The modular mapping is NOT the reference: heads 0 and 1 share KV head 0 under `/`, not under `%`. -/
theorem gqa_mod_differs : (1 / 12 : Nat) ≠ 1 % 2 := by decide

-- § scratch layout

/-- `qsa_decode_attn_scratch_floats`. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * (256 + 2) + 64

/-- The three partial arrays end strictly inside the per-query stride. -/
theorem scratch_fits (cap nHead : Nat) :
    nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead < scratchFloats cap nHead := by
  unfold scratchFloats
  rw [Nat.mul_add]
  omega

/-- A partial slot `(kvh * n_chunks + c) * G + hl` is inside the `n_kv * n_chunks * G` partials. -/
theorem partSlot_lt (kvh nKv nC c hl : Nat) (hk : kvh < nKv) (hc : c < nC) (hl12 : hl < 12) :
    (kvh * nC + c) * 12 + hl < nKv * nC * 12 := by
  have h1 : (kvh + 1) * nC ≤ nKv * nC := Nat.mul_le_mul_right nC (Nat.succ_le_of_lt hk)
  rw [Nat.succ_mul] at h1
  have h2 : kvh * nC + c < nKv * nC := by omega
  have h3 : (kvh * nC + c + 1) * 12 ≤ nKv * nC * 12 := Nat.mul_le_mul_right 12 (Nat.succ_le_of_lt h2)
  rw [Nat.succ_mul] at h3
  omega

-- § merge

/-- The merge maximum over chunk maxima (`M`, starting from the sentinel `lo`). -/
def mergeMax (lo : Int) : List Int → Int
  | [] => lo
  | m :: ms => max m (mergeMax lo ms)

/-- Every chunk maximum is at most `M`, so every rescale exponent `m_c - M` is non-positive. -/
theorem mergeMax_ge (lo : Int) (ms : List Int) (m : Int) (h : m ∈ ms) : m - mergeMax lo ms ≤ 0 := by
  induction ms with
  | nil => simp at h
  | cons x xs ih =>
    simp only [mergeMax]
    rcases List.mem_cons.mp h with h | h
    · subst h; omega
    · have := ih h; omega

/-- Sentinel-only merges keep the sentinel, so an all-empty selection yields `L = 0` and output 0. -/
theorem mergeMax_all_sentinel (lo : Int) (ms : List Int) (h : ∀ m ∈ ms, m = lo) : mergeMax lo ms = lo := by
  induction ms with
  | nil => rfl
  | cons x xs ih =>
    simp only [mergeMax]
    have hx := h x (List.mem_cons_self ..)
    have hr := ih (fun m hm => h m (List.mem_cons_of_mem _ hm))
    omega

end Strata.DecodeAttn
