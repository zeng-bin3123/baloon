/-!
# MODULE: ple_ngram
SOURCES: include/strata/kernels/ngram.hpp, src/kernels/ngram.cpp
The PLE n-gram hash (16 row indices per token from the last three token ids) and the IQ4_NL row layout
of the per-layer token table.

## Source semantics
* Head `h = (n-2)*8 + g` for `n in {2,3}`, `g < 8`; its row is `mixed_n % vocab[h] + offset[h]`.
* `head_offsets` is the running sum of `head_vocab_sizes`, and the last head ends below the table's row count,
  so every hashed row is in range.
* The EOS cut propagates forward: once a predecessor is missing (`< 0`) or EOS, it and all older positions
  become EOS; the token's own id is never cut.
* `mixed` XORs products, so equal terms cancel (a sum would double them).
* A row is 5 IQ4_NL blocks of 18 bytes = 90 bytes = 160 values; sixteen rows flatten head-slowest to 2560.
* Nibble order is split-half: byte `j` holds elements `j` and `j+16`.

## Form template identifiers
ctxStep -> the per-predecessor `cut`/`ctx[s]` update in `ngram_rows`
headIndex -> `(n - 2) * HEADS_PER_NGRAM + g`
vocabs / offsets -> `ple_artifact_consts().vocab / .offset`
loByte / hiHalf -> `qs[j] & 0x0F` / `qs[j] >> 4` in `iq4nl_dequant_row`
-/

namespace Strata.Kernels

-- § geometry constants

/-- `PLE_HEAD_DIM`. -/
def pleHeadDim : Nat := 160
/-- `NGRAM_SIZE`. -/
def ngramSize : Nat := 3
/-- `HEADS_PER_NGRAM`. -/
def headsPerNgram : Nat := 8
/-- `PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM`. -/
def pleNHeads : Nat := (ngramSize - 1) * headsPerNgram
/-- `PLE_ROW_BYTES = (PLE_HEAD_DIM / 32) * 18`. -/
def pleRowBytes : Nat := (pleHeadDim / 32) * 18
/-- `PLE_TABLE_ROWS`. -/
def pleTableRows : Nat := 320001536

/-- Sixteen heads. -/
theorem n_heads_sixteen : pleNHeads = 16 := by decide

/-- A row is 90 bytes. -/
theorem row_bytes_ninety : pleRowBytes = 90 := by decide

/-- Sixteen gathered rows flatten to `n_embd = 2560` values. -/
theorem gather_is_n_embd : pleNHeads * pleHeadDim = 2560 := by decide

/-- The history window `(conv_kernel - 1) * ngram_size` is 9. -/
theorem hist_is_nine : (4 - 1) * ngramSize = 9 := by decide

-- § head indexing

/-- Head index of bigram/trigram `n` and sub-head `g`. -/
def headIndex (n g : Nat) : Nat := (n - 2) * 8 + g

/-- Heads of `n in {2,3}` cover `0..15` without leaving the 16-entry arrays. -/
theorem headIndex_lt (n g : Nat) (h1 : 2 ≤ n) (h2 : n ≤ 3) (hg : g < 8) : headIndex n g < 16 := by
  unfold headIndex
  omega

/-- Distinct `(n, g)` pairs give distinct heads. -/
theorem headIndex_inj (n m g k : Nat) (h1 : 2 ≤ n) (h2 : n ≤ 3) (h3 : 2 ≤ m) (h4 : m ≤ 3)
    (hg : g < 8) (hk : k < 8) (h : headIndex n g = headIndex m k) : n = m ∧ g = k := by
  unfold headIndex at h
  omega

-- § hash constants

/-- `head_vocab_sizes` of the artifact. -/
def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- `head_offsets` of the artifact. -/
def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Running sum of vocabulary sizes. -/
def prefixSums : Nat → List Nat → List Nat
  | _, [] => []
  | acc, v :: vs => acc :: prefixSums (acc + v) vs

/-- The offsets are exactly the running sum of the vocab sizes. -/
theorem offsets_are_prefix_sums : offsets = prefixSums 0 vocabs := by decide

/-- Both arrays have one entry per head. -/
theorem consts_length : vocabs.length = 16 ∧ offsets.length = 16 := by decide

/-- The vocabulary sizes together fit inside the table. -/
theorem vocab_sum_fits : vocabs.foldl (· + ·) 0 ≤ pleTableRows := by decide

/-- The last head ends at 320,001,446, below the table's 320,001,536 rows. -/
theorem last_head_end : 300001275 + 20000171 = 320001446 ∧ 320001446 ≤ pleTableRows := by decide

/-- A hashed row `mixed % vocab + offset` is strictly below `vocab + offset`. -/
theorem hashed_row_lt (mixed vocab offset : Nat) (h : 0 < vocab) :
    mixed % vocab + offset < vocab + offset := by
  have := Nat.mod_lt mixed h
  omega

/-- Hence every hashed row lies in the table, whenever `vocab + offset` is within it. -/
theorem hashed_row_in_table (mixed vocab offset : Nat) (h : 0 < vocab)
    (hend : vocab + offset ≤ pleTableRows) : mixed % vocab + offset < pleTableRows := by
  have := hashed_row_lt mixed vocab offset h
  omega

-- § XOR structure

/-- Mixed hash of two terms. -/
def mixed2 (c0 c1 m0 m1 : Nat) : Nat := (c0 * m0) ^^^ (c1 * m1)

/-- Equal context tokens with equal multipliers cancel to zero under XOR (a sum would double them). -/
theorem xor_equal_terms_cancel (c m : Nat) : mixed2 c c m m = 0 := by
  unfold mixed2
  exact Nat.xor_self _

/-- XOR combination is order-independent. -/
theorem mixed2_comm (c0 c1 m0 m1 : Nat) : mixed2 c0 c1 m0 m1 = mixed2 c1 c0 m1 m0 := by
  unfold mixed2
  exact Nat.xor_comm _ _

-- § EOS cut

/-- EOS token id. -/
def eosId : Int := 248044

/-- One predecessor step: new `cut` flag and the stored context value. -/
def ctxStep (cut : Bool) (t : Int) : Bool × Int :=
  let cut' := cut || decide (t < 0) || decide (t = eosId)
  (cut', if cut' then eosId else t)

/-- Once cut, every older position stays cut and becomes EOS. -/
theorem ctxStep_cut_propagates (t : Int) : ctxStep true t = (true, eosId) := by
  simp [ctxStep]

/-- A real, non-EOS predecessor with no earlier cut is kept verbatim (token id 0 is real). -/
theorem ctxStep_keeps_real (t : Int) (h0 : 0 ≤ t) (hne : t ≠ eosId) :
    ctxStep false t = (false, t) := by
  have h1 : ¬ t < 0 := by omega
  simp [ctxStep, h1, hne]

/-- A missing predecessor (`-1`) starts the cut. -/
theorem ctxStep_missing_cuts : ctxStep false (-1) = (true, eosId) := by
  decide

/-- An EOS predecessor is itself EOS and cuts everything older. -/
theorem ctxStep_eos_cuts : ctxStep false eosId = (true, eosId) := by
  decide

/-- Predecessor `s` back, oldest-first window of two, lives at index `2 - s`; `s = 1` is the newest. -/
theorem prev_index_in_window (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) : 2 - s < 2 ∧ (s = 1 → 2 - s = 1) := by
  omega

-- § IQ4_NL layout

/-- The IQ4_NL codebook. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries. -/
theorem iq4nl_length : iq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_sorted : List.Pairwise (· < ·) iq4nl := by decide

/-- Every codebook value fits an int8. -/
theorem iq4nl_int8 : ∀ v ∈ iq4nl, -128 ≤ v ∧ v ≤ 127 := by decide

/-- Byte index holding element `e` (0..31) of a block: split-half order. -/
def byteOf (e : Nat) : Nat := e % 16

/-- Whether element `e` is the high nibble of its byte. -/
def hiHalf (e : Nat) : Nat := e / 16

/-- Split-half packing: element `e` is `byte + 16 * half`, so byte `j` holds elements `j` and `j + 16`. -/
theorem split_half_roundtrip (e : Nat) : byteOf e + 16 * hiHalf e = e := by
  unfold byteOf hiHalf
  omega

/-- Every element of a 32-element block maps to a byte below 16 and a half below 2. -/
theorem split_half_in_range (e : Nat) (h : e < 32) : byteOf e < 16 ∧ hiHalf e < 2 := by
  unfold byteOf hiHalf
  omega

/-- Distinct elements map to distinct (byte, half) pairs. -/
theorem split_half_injective (a b : Nat) (ha : a < 32) (hb : b < 32)
    (h1 : byteOf a = byteOf b) (h2 : hiHalf a = hiHalf b) : a = b := by
  unfold byteOf hiHalf at *
  omega

end Strata.Kernels
