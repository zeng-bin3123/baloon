/-!
# MODULE: ple_ngram_hash
The PLE n-gram hash (16 heads, 8 per n-gram order, trigram window) and the IQ4_NL row dequantizer.

## Source semantics
Each head `h` maps a token context to row `mixed % vocab[h] + offset[h]`; offsets are prefix sums of the
vocab sizes, so heads index disjoint ranges of the table. Context positions after an EOS (or a negative
token) are EOS. IQ4_NL decodes 4-bit codes through a 16-entry increasing table with split-halves layout; a
row is 5 blocks of 18 bytes = 90 bytes for 160 values; head `h` owns outputs `[h*160, (h+1)*160)`.

## Form template identifiers
- `vocabs`, `offsets` <- PleConsts::vocab / offset
- `mixed2`, `mixed3` <- ngram_mixed for n = 2, 3 (xor of wrapped products)
- `ctx3` <- context construction with the EOS cut in ngram_rows
- `iq4Table` <- kIq4Nl
-/

namespace Strata.PleNgramHash

-- § constants

def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- There are 16 heads. -/
theorem vocabs_length : vocabs.length = 16 := by decide

/-- Every head vocabulary is positive (the modulus is never 0). -/
theorem vocabs_pos : ∀ v ∈ vocabs, 0 < v := by decide

/-- Head offsets are exactly the prefix sums of the vocab sizes, so head ranges are disjoint. -/
theorem offsets_are_prefix_sums : (List.scanl (· + ·) 0 vocabs).take 16 = offsets := by decide

/-- The last head's range ends inside the 320,001,536-row table. -/
theorem last_head_end_in_table : 300001275 + 20000171 ≤ 320001536 := by decide

/-- Every head's range `[offset, offset+vocab)` fits in the table. -/
theorem heads_fit_table :
    ∀ i, i < 16 → offsets.getD i 0 + vocabs.getD i 0 ≤ 320001536 := by decide

/-- A head's row index lies inside that head's range. -/
theorem head_index_in_range (mixed v off : Nat) (hv : 0 < v) :
    off ≤ mixed % v + off ∧ mixed % v + off < off + v := by
  have := Nat.mod_lt mixed hv
  omega

-- § mixing

def M64 : Nat := 2 ^ 64

def mult0 : Nat := 23703573157769
def mult1 : Nat := 20109073645365
def mult2 : Nat := 8052911324071

/-- Bigram mix: `ctx0*m0 ^ ctx1*m1` with each product wrapped mod 2^64. -/
def mixed2 (c0 c1 : Nat) : Nat := ((c0 * mult0) % M64) ^^^ ((c1 * mult1) % M64)

/-- Trigram mix. -/
def mixed3 (c0 c1 c2 : Nat) : Nat := mixed2 c0 c1 ^^^ ((c2 * mult2) % M64)

/-- A bigram mix fits in 64 bits. -/
theorem mixed2_lt (c0 c1 : Nat) : mixed2 c0 c1 < M64 := by
  unfold mixed2 M64
  exact Nat.xor_lt_two_pow (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))

/-- A trigram mix fits in 64 bits. -/
theorem mixed3_lt (c0 c1 c2 : Nat) : mixed3 c0 c1 c2 < M64 := by
  unfold mixed3 M64
  have h := mixed2_lt c0 c1
  unfold M64 at h
  exact Nat.xor_lt_two_pow h (Nat.mod_lt _ (by decide))

-- § context cut

/-- Context for the trigram window (current token, newest predecessor, older predecessor); once a
predecessor is negative or EOS, it and everything older is EOS. -/
def ctx3 (eos : Int) (tok pNew pOld : Int) : Int × Int × Int :=
  let cut1 := pNew < 0 ∨ pNew = eos
  let c1 := if cut1 then eos else pNew
  let cut2 := cut1 ∨ pOld < 0 ∨ pOld = eos
  let c2 := if cut2 then eos else pOld
  (tok, c1, c2)

/-- If the newest predecessor is EOS, the older position is EOS too: no history leaks past an EOS. -/
theorem ctx3_cut_propagates (eos tok pNew pOld : Int) (h : pNew = eos) :
    (ctx3 eos tok pNew pOld).2.1 = eos ∧ (ctx3 eos tok pNew pOld).2.2 = eos := by
  unfold ctx3
  subst h
  simp

/-- With valid predecessors both are kept unchanged. -/
theorem ctx3_no_cut (eos tok pNew pOld : Int) (h1 : 0 ≤ pNew) (h2 : pNew ≠ eos)
    (h3 : 0 ≤ pOld) (h4 : pOld ≠ eos) :
    ctx3 eos tok pNew pOld = (tok, pNew, pOld) := by
  unfold ctx3
  have a1 : ¬ pNew < 0 := by omega
  have c3 : ¬ pOld < 0 := by omega
  simp [a1, h2, c3, h4]

-- § IQ4_NL

def iq4Table : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The code table has 16 entries. -/
theorem iq4_length : iq4Table.length = 16 := by decide

/-- The table is strictly increasing. -/
theorem iq4_strictly_increasing : ∀ i, i < 15 → iq4Table.getD i 0 < iq4Table.getD (i + 1) 0 := by decide

/-- All table values lie in `[-127, 113]`. -/
theorem iq4_range : ∀ v ∈ iq4Table, -127 ≤ v ∧ v ≤ 113 := by decide

/-- A masked 4-bit code is a valid table index. -/
theorem code_index_lt (c : Nat) : (c &&& 15) < 16 := by
  have : (c &&& 15) ≤ 15 := Nat.and_le_right
  omega

/-- The high nibble of a byte is a valid table index. -/
theorem high_nibble_lt (b : Nat) (hb : b < 256) : (b >>> 4) < 16 := by
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow]
  omega

/-- Split halves: each of the 32 positions of a block is produced by exactly one `j` (as `j` or `j+16`). -/
theorem split_halves_cover (i : Nat) (hi : i < 32) : ∃ j, j < 16 ∧ (i = j ∨ i = j + 16) :=
  ⟨i % 16, by omega, by omega⟩

/-- The low and high halves never write the same position. -/
theorem split_halves_disjoint (j k : Nat) (hj : j < 16) (hk : k < 16) : j ≠ k + 16 := by omega

/-- A row is 5 blocks of 18 bytes = 90 bytes for 160 values. -/
theorem row_bytes_value : 160 / 32 * 18 = 90 := by decide

-- § layout

/-- Head `h` writes outputs `[h*160, (h+1)*160)` inside the 2560-wide PLE embedding. -/
theorem head_output_in_range (h d : Nat) (hh : h < 16) (hd : d < 160) : h * 160 + d < 2560 := by omega

/-- Different heads write disjoint output ranges. -/
theorem head_outputs_disjoint (h1 h2 d1 d2 : Nat) (hd1 : d1 < 160) (hd2 : d2 < 160) (hne : h1 < h2) :
    h1 * 160 + d1 < h2 * 160 + d2 := by omega

/-- The batched gather reads `n_tokens * 16` rows and writes `160` floats per row. -/
theorem gather_batch_out_lt (i nTokens : Nat) (hi : i < nTokens * 16) (d : Nat) (hd : d < 160) :
    i * 160 + d < nTokens * 16 * 160 := by
  have h1 : (i + 1) * 160 ≤ nTokens * 16 * 160 := Nat.mul_le_mul_right 160 hi
  have h2 : (i + 1) * 160 = i * 160 + 160 := by rw [Nat.add_mul, Nat.one_mul]
  omega

end Strata.PleNgramHash
