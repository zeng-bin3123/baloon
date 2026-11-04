/-!
# MODULE: ngram_hash
The PLE n-gram hash and the IQ4_NL table row read (`src/kernels/ngram.cpp`): the 16 head row indices
are `mixed % vocab_h + offset_h`, the context is cut at the first EOS/negative predecessor, and rows are
dequantized from 5 blocks of 18 bytes with split-halves nibble layout.

## Source semantics
* Head offsets are the running sums of the head vocabularies, so head ranges tile `[0, 320001446)`
  and the table (320,001,536 rows) contains every produced index.
* There are `(NGRAM_SIZE - 1) * HEADS_PER_NGRAM = 2 * 8 = 16` heads; head `(n-2)*8 + g` stays below 16.
* Once the context has been cut, every later entry is the EOS token and the cut stays set.
* A row is 5 blocks * 18 bytes = 90 bytes decoding to 160 values; block `b` writes `b*32 + j` and `b*32 + j + 16`.
* The gather output is head-slowest: head `h`, element `d` lands at `h*160 + d < 2560`.
* The IQ4_NL codebook is strictly increasing and indexed by a nibble in `0..15`.

## Form template identifiers
* `vocabs`, `offsets` <- `vocab[PLE_N_HEADS]`, `offset[PLE_N_HEADS]`
* `ctxStep`           <- one iteration of the predecessor loop in `ngram_rows`
* `headIndex`         <- `mixed % vocab[h] + offset[h]`
* `rowBytes`          <- `PLE_ROW_BYTES`
-/

namespace Strata.Ngram

-- § head tables

/-- Per-head hash moduli, verbatim from the artifact. -/
def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets, verbatim from the artifact. -/
def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Rows in the PLE table (`ne1`). -/
def tableRows : Nat := 320001536

/-- Both head tables have 16 entries. -/
theorem tables_length : vocabs.length = 16 ∧ offsets.length = 16 := by decide

/-- Offsets are the running sums of the vocabularies: head `h+1` starts where head `h` ends. -/
theorem offsets_cumulative :
    ∀ h, h < 15 → offsets.getD (h + 1) 0 = offsets.getD h 0 + vocabs.getD h 0 := by decide

/-- The last head ends inside the table. -/
theorem last_head_in_table : offsets.getD 15 0 + vocabs.getD 15 0 ≤ tableRows := by decide

/-- Every head vocabulary is positive, so the modulo is defined. -/
theorem vocabs_pos : ∀ h, h < 16 → 0 < vocabs.getD h 0 := by decide

/-- Head index is inside its own range `[offset, offset + vocab)`. -/
theorem head_index_in_range (mixed vocab off : Nat) (hv : 0 < vocab) :
    off ≤ mixed % vocab + off ∧ mixed % vocab + off < off + vocab := by
  have := Nat.mod_lt mixed hv
  omega

/-- Every head range ends inside the table. -/
theorem ends_in_table : ∀ h, h < 16 → offsets.getD h 0 + vocabs.getD h 0 ≤ tableRows := by decide

/-- Every head index lies inside the table. -/
theorem head_index_in_table (h : Nat) (hh : h < 16) (mixed : Nat) :
    mixed % vocabs.getD h 0 + offsets.getD h 0 < tableRows := by
  have hv : 0 < vocabs.getD h 0 := vocabs_pos h hh
  have hm := Nat.mod_lt mixed hv
  have hend := ends_in_table h hh
  omega

/-- Head layout: head `(n-2)*8 + g` with `n ∈ {2,3}`, `g < 8` is below 16. -/
theorem head_slot_lt (n g : Nat) (hn1 : 2 ≤ n) (hn2 : n ≤ 3) (hg : g < 8) : (n - 2) * 8 + g < 16 := by
  omega

/-- Head slots of the two n-gram orders are disjoint. -/
theorem head_slots_disjoint (g g' : Nat) (hg : g < 8) (hg' : g' < 8) : 0 * 8 + g ≠ 1 * 8 + g' := by
  omega

-- § context cut

-- The EOS token id used by the PLE (abstract).
variable (eos : Int)

/-- One predecessor step: cut when already cut, negative, or EOS; a cut position stores EOS. -/
def ctxStep (cut : Bool) (t : Int) : Bool × Int :=
  let t' := if cut then -1 else t
  let cut' := cut || decide (t' < 0) || decide (t' = eos)
  (cut', if cut' then eos else t')

/-- Once cut, the cut persists and the stored token is EOS. -/
theorem ctxStep_cut_persists (t : Int) : ctxStep eos true t = (true, eos) := by
  simp [ctxStep]

/-- A negative predecessor triggers the cut and stores EOS. -/
theorem ctxStep_negative (t : Int) (h : t < 0) : ctxStep eos false t = (true, eos) := by
  simp [ctxStep, h]

/-- An EOS predecessor triggers the cut and stores EOS. -/
theorem ctxStep_eos : ctxStep eos false eos = (true, eos) := by
  simp [ctxStep]

/-- A valid, non-EOS predecessor keeps the history and the cut stays unset. -/
theorem ctxStep_valid (t : Int) (h0 : 0 ≤ t) (hne : t ≠ eos) : ctxStep eos false t = (false, t) := by
  have h1 : ¬ t < 0 := by omega
  simp [ctxStep, h1, hne]

-- § IQ4_NL row decode

/-- Bytes per PLE row: 5 blocks of 18 bytes. -/
def rowBytes : Nat := 5 * 18

/-- A row is 90 bytes. -/
theorem rowBytes_value : rowBytes = 90 := by decide

/-- The table must exactly fill the data section: `rows * 90` equals the bytes available. -/
theorem table_size_check (rows have_ : Nat) (h : rows * rowBytes = have_) : have_ = rows * 90 := by
  unfold rowBytes at h
  omega

/-- Block `b < 5` writes `b*32 + j` and `b*32 + j + 16` for `j < 16`, all below 160. -/
theorem block_outputs_in_row (b j : Nat) (hb : b < 5) (hj : j < 16) :
    b * 32 + j < 160 ∧ b * 32 + j + 16 < 160 := by omega

/-- The two halves of a block are disjoint. -/
theorem block_halves_disjoint (b j j' : Nat) (hj : j < 16) : b * 32 + j < b * 32 + j' + 16 := by omega

/-- Block source offset `b*18` plus two bytes of scale and 16 of nibbles ends inside the 90-byte row. -/
theorem block_source_in_row (b : Nat) (hb : b < 5) : b * 18 + 18 ≤ 90 := by omega

/-- Head-slowest gather: head `h`, element `d` lands below 2560. -/
theorem gather_index_lt (h d : Nat) (hh : h < 16) (hd : d < 160) : h * 160 + d < 2560 := by omega

/-- Gather indices are injective: equal outputs imply equal head and element. -/
theorem gather_injective (h d h' d' : Nat) (hd : d < 160) (hd' : d' < 160)
    (e : h * 160 + d = h' * 160 + d') : h = h' ∧ d = d' := by omega

-- § IQ4_NL codebook

/-- The IQ4_NL codebook. -/
def codebook : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, one per nibble. -/
theorem codebook_length : codebook.length = 16 := by decide

/-- The codebook is strictly increasing, so dequantization is monotone in the nibble. -/
theorem codebook_increasing : codebook.Pairwise (· < ·) := by decide

/-- A nibble taken modulo 16 always indexes the codebook. -/
theorem nibble_index_lt (code : Nat) : code % 16 < codebook.length := by
  rw [codebook_length]
  omega

end Strata.Ngram
