/-!
# MODULE: ple_ngram_hash
The PLE n-gram hash and the IQ4_NL row decode: each token plus its two predecessors is mixed with three odd
multipliers, reduced modulo 16 per-head prime vocabularies, and offset into one 320,001,536-row table.

## Source semantics
* Row index of head `h` is `mixed % vocab[h] + offset[h]`, so it lies in `[offset[h], offset[h] + vocab[h])`.
* `offset[h+1] = offset[h] + vocab[h]`: the per-head ranges tile the table without overlap.
* The last range ends before the table's 320,001,536 rows.
* Heads are `(n - 2) * 8 + g` for `n in {2, 3}` and `g < 8`: all 16 heads are covered exactly once.
* An EOS (or missing) predecessor cuts the context: every later position is EOS too.
* A row is five 18-byte IQ4_NL blocks; block `b` writes outputs `b*32 + j` and `b*32 + j + 16`, covering 160 values.

## Form template identifiers
* `PleConsts::vocab` -> `vocab`, `PleConsts::offset` -> `offset`, `ngram_rows` -> `headIndex`/`ctx2`
* `kIq4Nl` -> `iq4nl`, `iq4nl_dequant_row` -> `dequantSlot`
-/

namespace Strata.PleNgramHash

-- § head tables

/-- Per-head prime vocabulary sizes, verbatim from the artifact. -/
def vocab : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head table offsets, verbatim from the artifact. -/
def offset : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Rows in the PLE table. -/
def tableRows : Nat := 320001536

/-- Check that each offset is the previous offset plus the previous vocabulary size. -/
def tiles : List Nat → List Nat → Bool
  | o :: o2 :: os, v :: vs => decide (o + v = o2) && tiles (o2 :: os) vs
  | _, _ => true

/-- The head ranges tile the table: offsets are the prefix sums of the vocabularies. -/
theorem ranges_tile : tiles offset vocab = true := by
  decide

/-- Both tables have 16 entries (one per head). -/
theorem tables_len : vocab.length = 16 ∧ offset.length = 16 := by
  decide

/-- The last head's range ends before the end of the table. -/
theorem last_range_in_table : 300001275 + 20000171 ≤ tableRows := by
  unfold tableRows
  decide

/-- Every vocabulary size is positive, so the modulus is defined. -/
theorem vocab_pos : vocab.all (fun v => decide (0 < v)) = true := by
  decide

/-- A row index of one head lies inside that head's range. -/
theorem row_in_head_range (mixed v o : Nat) (hv : 0 < v) :
    o ≤ mixed % v + o ∧ mixed % v + o < o + v := by
  have := Nat.mod_lt mixed hv
  omega

-- § head numbering

/-- Head index for n-gram order `n` and sub-head `g`. -/
def headIndex (n g : Nat) : Nat := (n - 2) * 8 + g

/-- Orders 2 and 3 with eight sub-heads each give heads below 16. -/
theorem headIndex_lt (n g : Nat) (h1 : 2 ≤ n) (h2 : n ≤ 3) (hg : g < 8) : headIndex n g < 16 := by
  unfold headIndex
  omega

/-- Distinct (order, sub-head) pairs give distinct heads. -/
theorem headIndex_inj (n m g k : Nat) (h1 : 2 ≤ n) (h2 : n ≤ 3) (h3 : 2 ≤ m) (h4 : m ≤ 3)
    (hg : g < 8) (hk : k < 8) (h : headIndex n g = headIndex m k) : n = m ∧ g = k := by
  unfold headIndex at h
  omega

/-- Every head is reached: head `h < 16` decomposes into an order and a sub-head. -/
theorem headIndex_surj (h : Nat) (hh : h < 16) :
    ∃ n g, 2 ≤ n ∧ n ≤ 3 ∧ g < 8 ∧ headIndex n g = h := by
  refine ⟨h / 8 + 2, h % 8, ?_, ?_, ?_, ?_⟩ <;> (try unfold headIndex) <;> omega

/-- The predecessor read `prev[n_prev - s]` with `n_prev = 2` and `s in {1, 2}` is a valid index. -/
theorem prev_index_valid (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) : 2 - s < 2 := by
  omega

-- § EOS cut

/-- Context of order three: the token, then two predecessors (newest first), with the cut rule.
`eos` is the EOS id; a negative predecessor is a missing one. -/
def ctx2 (eos : Int) (tok newest oldest : Int) : Int × Int × Int :=
  let cut1 := decide (newest < 0 ∨ newest = eos)
  let c1 := if cut1 then eos else newest
  let t2 := if cut1 then (-1 : Int) else oldest
  let cut2 := cut1 || decide (t2 < 0 ∨ t2 = eos)
  let c2 := if cut2 then eos else t2
  (tok, c1, c2)

/-- Once the newest predecessor cuts the context, the older position is EOS too. -/
theorem eos_cut_propagates (eos tok newest oldest : Int) (h : newest < 0 ∨ newest = eos) :
    (ctx2 eos tok newest oldest).2.2 = eos := by
  simp [ctx2, h]

/-- The token itself is passed through unchanged. -/
theorem ctx_keeps_token (eos tok newest oldest : Int) : (ctx2 eos tok newest oldest).1 = tok := by
  simp [ctx2]

/-- Without any cut the history is the real tokens. -/
theorem no_cut_keeps_history (eos tok newest oldest : Int) (h1 : 0 ≤ newest) (h2 : newest ≠ eos)
    (h3 : 0 ≤ oldest) (h4 : oldest ≠ eos) :
    (ctx2 eos tok newest oldest).2 = (newest, oldest) := by
  have a : ¬ (newest < 0 ∨ newest = eos) := by omega
  have b : ¬ (oldest < 0 ∨ oldest = eos) := by omega
  simp [ctx2, a, b]

-- § IQ4_NL row decode

/-- The IQ4_NL codebook. -/
def iq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Strictly increasing check. -/
def strictInc : List Int → Bool
  | a :: b :: rest => decide (a < b) && strictInc (b :: rest)
  | _ => true

/-- The codebook has 16 entries (one per nibble). -/
theorem iq4nl_length : iq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_strictly_increasing : strictInc iq4nl = true := by decide

/-- The codebook fits in a signed byte. -/
theorem iq4nl_in_int8 : iq4nl.all (fun v => decide (-128 ≤ v ∧ v ≤ 127)) = true := by decide

/-- Output slot decode: nibble pair `j < 16` of block `b < 5` writes slots `b*32 + j` and `b*32 + j + 16`. -/
theorem dequant_covers (k : Nat) (hk : k < 160) :
    ∃ b j, b < 5 ∧ j < 16 ∧ (k = b * 32 + j ∨ k = b * 32 + j + 16) := by
  by_cases h : k % 32 < 16
  · exact ⟨k / 32, k % 32, by omega, h, Or.inl (by omega)⟩
  · exact ⟨k / 32, k % 32 - 16, by omega, by omega, Or.inr (by omega)⟩

/-- No slot is written twice: the two halves of one block are disjoint ranges. -/
theorem dequant_disjoint (b j b2 j2 : Nat) (hj : j < 16) (hj2 : j2 < 16) :
    b * 32 + j ≠ b2 * 32 + j2 + 16 := by
  omega

/-- A row of five 18-byte blocks is 90 bytes and holds 160 values. -/
theorem row_geometry : 5 * 18 = 90 ∧ 5 * 32 = 160 := by decide

end Strata.PleNgramHash
