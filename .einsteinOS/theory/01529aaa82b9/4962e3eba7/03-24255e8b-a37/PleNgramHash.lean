/-!
# MODULE: ple_ngram_hash
PLE n-gram hashing into a 16-head embedding table, plus the IQ4_NL row layout used to dequantize rows.

## Source semantics
* NGRAM_SIZE = 3 and HEADS_PER_NGRAM = 8 are inferred from the 3 multipliers and 16 heads (assumption).
* Head `h = (n-2)*8 + g`; the table row for head `h` is `mixed % vocab[h] + offset[h]`.
* `offset[h+1] = offset[h] + vocab[h]`, so head ranges tile the table and the last end fits in 320001536 rows.
* Predecessor `s` positions back is `prev[n_prev - s]` (oldest first), always inside the window.
* An IQ4_NL row is 5 blocks of 18 bytes (2-byte scale + 16 nibble bytes), 90 B for 160 values; byte `j` holds
  elements `j` and `j+16` of its block.

## Form template identifiers
PLE_N_HEADS -> 16; vocab/offset arrays -> `vocab`/`offset`; head index -> `headIdx`; IQ4_NL table -> `iq4nl`;
row bytes -> `ROW_BYTES`.
-/

namespace Strata.Kernels.Ngram

-- § constants

/-- Per-head vocabulary sizes from the artifact metadata. -/
def vocab : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets from the artifact metadata. -/
def offset : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Both arrays have 16 entries. -/
theorem arrays_size : vocab.length = 16 ∧ offset.length = 16 := by decide

/-- The offsets are the running prefix sums of the vocabulary sizes (head ranges tile the table). -/
theorem offsets_are_prefix_sums :
    (List.range 15).all (fun i => decide (offset[i + 1]! = offset[i]! + vocab[i]!)) = true := by decide

/-- The last head's range ends below the table's 320001536 rows. -/
theorem last_end_fits : offset[15]! + vocab[15]! ≤ 320001536 := by decide

/-- The table occupies 28,800,138,240 bytes at 90 bytes per row. -/
theorem table_bytes : 320001536 * 90 = 28800138240 := by decide

/-- Row size: 5 blocks of 18 bytes. -/
theorem row_bytes : 160 / 32 * 18 = 90 := by decide

-- § head indexing

/-- Output head index for n-gram order `n` (2 or 3) and sub-head `g`. -/
def headIdx (n g : Nat) : Nat := (n - 2) * 8 + g

/-- Head indices stay below 16. -/
theorem headIdx_lt (n g : Nat) (hn : 2 ≤ n) (hn' : n ≤ 3) (hg : g < 8) : headIdx n g < 16 := by
  unfold headIdx
  omega

/-- Distinct (n, g) pairs give distinct heads: every head is written exactly once. -/
theorem headIdx_inj (n g n' g' : Nat) (hn : 2 ≤ n) (hn' : n ≤ 3) (hg : g < 8)
    (hm : 2 ≤ n') (hm' : n' ≤ 3) (hg' : g' < 8) (h : headIdx n g = headIdx n' g') : n = n' ∧ g = g' := by
  unfold headIdx at h
  omega

/-- Every head below 16 is produced by some (n, g). -/
theorem headIdx_surj (h : Nat) (hh : h < 16) : ∃ n g, 2 ≤ n ∧ n ≤ 3 ∧ g < 8 ∧ headIdx n g = h := by
  refine ⟨h / 8 + 2, h % 8, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · unfold headIdx
    omega

/-- A row index `mixed % v + off` lies in the head's range `[off, off + v)`. -/
theorem row_in_head_range (mixed v off : Nat) (hv : 0 < v) :
    off ≤ mixed % v + off ∧ mixed % v + off < off + v := by
  have := Nat.mod_lt mixed hv
  omega

/-- The flat output slot `i*16 + h` is inside a batch of `n` tokens. -/
theorem out_slot_lt (i n h : Nat) (hi : i < n) (hh : h < 16) : i * 16 + h < n * 16 := by
  omega

-- § context window

/-- Index into `prev` (oldest first) of the predecessor `s` positions back, window size `n_prev`. -/
def prevIdx (nPrev s : Nat) : Nat := nPrev - s

/-- For `1 <= s <= n_prev` the index is in range, and `s = 1` is the newest entry. -/
theorem prevIdx_bounds (nPrev s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ nPrev) :
    prevIdx nPrev s < nPrev ∧ (s = 1 → prevIdx nPrev s = nPrev - 1) := by
  unfold prevIdx
  omega

/-- Walking `s = 1 .. n_prev` visits each window entry once (the map `s -> n_prev - s` is injective). -/
theorem prevIdx_inj (nPrev s t : Nat) (hs : s ≤ nPrev) (ht : t ≤ nPrev) (h : prevIdx nPrev s = prevIdx nPrev t) :
    s = t := by
  unfold prevIdx at h
  omega

-- § IQ4_NL

/-- The 16-entry non-linear codebook. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries (one per 4-bit code). -/
theorem iq4nl_size : iq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_increasing :
    (List.range 15).all (fun i => decide (iq4nl[i]! < iq4nl[i + 1]!)) = true := by decide

/-- A 4-bit code (masked with 15) always indexes the codebook. -/
theorem code_index_lt (c : Nat) : c % 16 < 16 := by
  omega

/-- Split halves: byte `j` of a block holds elements `j` and `j + 16`; together they cover 0..31 once. -/
theorem split_halves_cover (k : Nat) (hk : k < 32) : ∃ j half, j < 16 ∧ half < 2 ∧ k = j + 16 * half := by
  refine ⟨k % 16, k / 16, ?_, ?_, ?_⟩ <;> omega

/-- Distinct (j, half) pairs write distinct elements. -/
theorem split_halves_inj (j half j' half' : Nat) (hj : j < 16) (hj' : j' < 16)
    (h : j + 16 * half = j' + 16 * half') : j = j' ∧ half = half' := by
  omega

/-- All block reads `b*18 + 2 + j` stay inside the 90-byte row. -/
theorem block_byte_in_row (b j : Nat) (hb : b < 5) (hj : j < 16) : b * 18 + 2 + j < 90 := by
  omega

/-- All outputs `b*32 + j + 16*half` stay inside the 160-value head. -/
theorem output_in_head (b j half : Nat) (hb : b < 5) (hj : j < 16) (hh : half < 2) :
    b * 32 + j + 16 * half < 160 := by
  omega

end Strata.Kernels.Ngram
