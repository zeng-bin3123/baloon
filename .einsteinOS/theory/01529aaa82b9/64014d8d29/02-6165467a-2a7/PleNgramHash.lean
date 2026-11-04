/-!
# MODULE: ple_ngram_hash
SOURCES: src/kernels/ngram.cpp, include/strata/kernels/ngram.hpp
The PLE n-gram hash (sixteen row indices per token from the last three token ids, XOR of
multiplier products reduced `mod` non-power-of-two head vocab sizes plus offsets), the IQ4_NL
row dequantization (split-half nibble order) and the `PleTable` issue/collect protocol.

## Source semantics
* Window is OLDEST FIRST; a missing predecessor is `-1`; the EOS cut propagates forward to older
  positions; the token's OWN eos does not cut its own context.
* Head `h = (n-2)*8 + g` for `n in {2,3}`, `g < 8` gives 16 distinct heads.
* Row index of head `h` is `mixed % vocab[h] + offset[h]`, which is below `offset[h] + vocab[h]`;
  `offset` is the running sum of `vocab`.
* A row is 160 values = 5 blocks of 32 at 18 bytes = 90 bytes; byte `j` carries elements `j`
  and `j + 16` of its block.
* The table accepts one in-flight `issue`; `collect` needs a pending issue.

## Form template identifiers
* `ngram_rows` context build   -> `ctx1`, `ctx2`, `isCut`
* `ple_artifact_consts`        -> `vocab`, `offs`
* `iq4nl_dequant_row`          -> `iq4`
* `PleTable::issue/collect`    -> `issue`, `collect`
-/

namespace Strata.PleNgram

-- § geometry constants

/-- `PLE_N_HEADS = (ngram_size - 1) * heads_per_ngram = 16`. -/
theorem n_heads : (3 - 1) * 8 = 16 := by decide

/-- `NG_HIST = (conv_kernel - 1) * ngram_size = 9`. -/
theorem hist_len : (4 - 1) * 3 = 9 := by decide

/-- `NG_HC_DIM = hc * n_embd = 10240`. -/
theorem hc_dim : 4 * 2560 = 10240 := by decide

/-- A row is 5 blocks x 18 bytes = 90 bytes. -/
theorem row_bytes : 160 / 32 * 18 = 90 := by decide

/-- Sixteen heads of 160 values flatten head-slowest to `n_embd = 2560`. -/
theorem heads_flatten : 16 * 160 = 2560 := by decide

/-- The table holds `320001536 * 90 = 28,800,138,240` bytes (the "28.8 GB"). -/
theorem table_bytes : 320001536 * 90 = 28800138240 := by decide

-- § head constants

/-- `head_vocab_sizes` of the artifact. -/
def vocab : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- `head_offsets` of the artifact. -/
def offs : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Both constant arrays have 16 entries. -/
theorem consts_len : vocab.length = 16 ∧ offs.length = 16 := by decide

/-- `head_offsets` is the running sum of `head_vocab_sizes`. -/
theorem offsets_running_sum : ∀ i : Fin 15,
    offs.getD (i.val + 1) 0 = offs.getD i.val 0 + vocab.getD i.val 0 := by
  decide

/-- Every head vocab size is positive, so `%` is well defined. -/
theorem vocab_pos : ∀ x ∈ vocab, 0 < x := by decide

/-- The last head's window ends below the table's row count (no head reaches past the table). -/
theorem heads_within_table : offs.getD 15 0 + vocab.getD 15 0 ≤ 320001536 := by decide

/-- A head's row index lies inside that head's window `[offset, offset + vocab)`. -/
theorem row_in_window (mixed v o : Nat) (hv : 0 < v) : mixed % v + o < o + v := by
  have := Nat.mod_lt mixed hv
  omega

-- § XOR structure

/-- Equal terms cancel under XOR (so XOR, not sum, is observable). -/
theorem xor_cancel (a : Nat) : a ^^^ a = 0 := Nat.xor_self a

-- § head indexing

/-- Head index `(n-2)*8 + g` for `n in {2,3}` and `g < 8` is below 16. -/
theorem head_idx_lt (n g : Nat) (hn : 2 ≤ n) (hn2 : n ≤ 3) (hg : g < 8) : (n - 2) * 8 + g < 16 := by
  omega

/-- Bigram heads occupy `0..7` and trigram heads `8..15`: the map `(n, g) -> head` is injective. -/
theorem head_idx_inj (n g n' g' : Nat) (hn : 2 ≤ n) (hn2 : n ≤ 3) (hg : g < 8)
    (hn' : 2 ≤ n') (hn2' : n' ≤ 3) (hg' : g' < 8) (h : (n - 2) * 8 + g = (n' - 2) * 8 + g') :
    n = n' ∧ g = g' := by
  omega

-- § EOS cut context

/-- EOS token id. -/
def EOS : Int := 248044

/-- A predecessor cuts the context when it is missing (`< 0`) or the EOS token. -/
def isCut (t : Int) : Bool := decide (t < 0) || decide (t = EOS)

/-- Context slot 1: the newest predecessor, or EOS if it cuts. -/
def ctx1 (p1 : Int) : Int := if isCut p1 then EOS else p1

/-- Context slot 2: the older predecessor, or EOS if it or the newer one cuts. -/
def ctx2 (p0 p1 : Int) : Int := if isCut p1 || isCut p0 then EOS else p0

/-- A cutting newest predecessor makes both context slots EOS (the cut propagates forward). -/
theorem cut_propagates (p0 p1 : Int) (h : isCut p1 = true) : ctx1 p1 = EOS ∧ ctx2 p0 p1 = EOS := by
  constructor
  · simp [ctx1, h]
  · simp [ctx2, h]

/-- A real newest predecessor is kept as is (the token's own EOS never enters here). -/
theorem real_newest_kept (p1 : Int) (h : isCut p1 = false) : ctx1 p1 = p1 := by
  simp [ctx1, h]

/-- A real newest predecessor with a cutting older one: only the older slot becomes EOS. -/
theorem older_cut_only (p0 p1 : Int) (h1 : isCut p1 = false) (h0 : isCut p0 = true) :
    ctx1 p1 = p1 ∧ ctx2 p0 p1 = EOS := by
  constructor
  · simp [ctx1, h1]
  · simp [ctx2, h1, h0]

/-- With no cut at all the full history is used. -/
theorem no_cut_full (p0 p1 : Int) (h1 : isCut p1 = false) (h0 : isCut p0 = false) :
    ctx1 p1 = p1 ∧ ctx2 p0 p1 = p0 := by
  constructor
  · simp [ctx1, h1]
  · simp [ctx2, h1, h0]

/-- Token id 0 is a real token: it is not a cut. -/
theorem token_zero_not_cut : isCut 0 = false := by decide

/-- A missing predecessor (`-1`) cuts. -/
theorem missing_cuts : isCut (-1) = true := by decide

-- § IQ4_NL dequantization

/-- The IQ4_NL codebook (`kvalues_iq4nl`), no `-8` shift. -/
def iq4 : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries. -/
theorem iq4_len : iq4.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4_sorted : List.Pairwise (· < ·) iq4 := by decide

/-- Split halves: byte `j < 16` writes elements `j` and `j + 16`, which are distinct and inside a block of 32. -/
theorem split_halves (j : Nat) (hj : j < 16) : j ≠ j + 16 ∧ j + 16 < 32 := by
  omega

/-- Output index `b*32 + j + 16` over 5 blocks stays inside the 160-value row. -/
theorem out_idx_lt (b j : Nat) (hb : b < 5) (hj : j < 16) : b * 32 + j + 16 < 160 := by
  omega

/-- Block `b` starts at byte `b * 18` and its 16 quant bytes end inside the 90-byte row. -/
theorem block_in_row (b : Nat) (hb : b < 5) : b * 18 + 2 + 16 ≤ 90 := by
  omega

-- § PleTable issue / collect protocol

/-- Table state: whether one token's reads are in flight. -/
structure TableState where
  pending : Bool

/-- `issue` fails when a token is already in flight. -/
def issue (s : TableState) : Option TableState := if s.pending then none else some ⟨true⟩

/-- `collect` needs a pending issue and clears it. -/
def collect (s : TableState) : Option TableState := if s.pending then some ⟨false⟩ else none

/-- Only one token may be in flight: a second `issue` is refused. -/
theorem issue_twice_fails : (issue ⟨true⟩) = none := by
  simp [issue]

/-- `collect` without a prior `issue` is refused. -/
theorem collect_without_issue : (collect ⟨false⟩) = none := by
  simp [collect]

/-- An `issue` from idle yields a pending table. -/
theorem issue_from_idle : (issue ⟨false⟩).map (·.pending) = some true := by
  simp [issue]

/-- After `issue` then `collect` the table is idle and can issue again. -/
theorem issue_collect_issue :
    ((issue ⟨false⟩).bind collect).bind issue = some ⟨true⟩ := by
  simp [issue, collect]

end Strata.PleNgram
