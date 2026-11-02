/-!
# MODULE: ngram_hash_iq4nl_table
The PLE n-gram hash (token context -> 16 table-row indices) and the IQ4_NL table read
(row addressing, split-halves dequantisation layout, head-slowest gather layout).

## Source semantics
* `ngram_mixed(ctx, mult, n)` assigns `ctx[0]*mult[0]` and XORs `ctx[j]*mult[j]` for `1 ≤ j < n` (mod 2^64).
* `ngram_rows`: predecessor `s` positions back is `prev[n_prev - s]` (prev is oldest first); an EOS or a
  negative predecessor cuts the history and every later position (and the cut position itself) is EOS.
* Row index of head `h` is `mixed % vocab[h] + offset[h]`; heads of n-gram size `n` are `(n-2)*8 .. (n-2)*8+7`.
* A row is 90 bytes = 5 blocks of 18 bytes (fp16 scale + 16 nibble bytes); byte `j` of a block holds element
  `j` (low nibble) and `j+16` (high nibble).
* `gather` writes head `h` at `[h*160, (h+1)*160)`; out-of-range rows read as zeros.

## Form template identifiers
ngram_mixed -> mixed_hash ; ngram_rows -> context_to_rows ; PleConsts.vocab -> head_vocab ;
PleConsts.offset -> head_offset ; kIq4Nl -> nl_table ; iq4nl_dequant_row -> row_dequant ;
PleTable::gather -> head_slowest_gather ; PleTable::read_row -> bounded_row_read
-/

namespace Strata.Kernels.Ngram

/-- N-gram size (context length). -/
def NGRAM_SIZE : Nat := 3
/-- Heads per n-gram size. -/
def HEADS_PER_NGRAM : Nat := 8
/-- Total heads. -/
def PLE_N_HEADS : Nat := 16
/-- Values per head row. -/
def PLE_HEAD_DIM : Nat := 160
/-- Bytes per IQ4_NL row. -/
def PLE_ROW_BYTES : Nat := 90
/-- Flattened embedding width. -/
def NG_N_EMBD : Nat := 2560
/-- "No predecessor" marker. -/
def TOKEN_NULL : Int := -1

/-- Per-head vocabulary sizes of the artifact. -/
def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets of the artifact. -/
def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

-- § constants

/-- Heads are exactly (NGRAM_SIZE-1) groups of HEADS_PER_NGRAM. -/
theorem heads_eq : (NGRAM_SIZE - 1) * HEADS_PER_NGRAM = PLE_N_HEADS := by decide

/-- Sixteen heads of 160 values make the 2560-wide embedding. -/
theorem embd_eq : PLE_N_HEADS * PLE_HEAD_DIM = NG_N_EMBD := by decide

/-- A row is five 18-byte blocks. -/
theorem row_bytes_eq : (PLE_HEAD_DIM / 32) * 18 = PLE_ROW_BYTES := by decide

/-- Both constant tables have one entry per head. -/
theorem tables_length : vocabs.length = 16 ∧ offsets.length = 16 := by decide

-- § head tables

/-- The offsets are the prefix sums of the vocabulary sizes (the table is a concatenation). -/
theorem offsets_prefix_sums :
    ∀ i : Fin 15, offsets[i.val + 1]! = offsets[i.val]! + vocabs[i.val]! := by decide

/-- Every head's index range ends inside the table (320,001,536 rows). -/
theorem head_ranges_in_table :
    ∀ i : Fin 16, offsets[i.val]! + vocabs[i.val]! ≤ 320001536 := by decide

/-- The row index always fits a uint32 (the C++ cast is lossless). -/
theorem head_ranges_fit_u32 :
    ∀ i : Fin 16, offsets[i.val]! + vocabs[i.val]! < 4294967296 := by decide

/-- Every vocabulary is positive, so the modulo is defined. -/
theorem vocabs_pos : ∀ i : Fin 16, 0 < vocabs[i.val]! := by decide

/-- The row index of a head lies in that head's own range `[off, off+v)`. -/
theorem head_index_bounds (m v off : Nat) (hv : 0 < v) :
    off ≤ m % v + off ∧ m % v + off < off + v := by
  have := Nat.mod_lt m hv
  omega

/-- For the artifact constants the row index is below 2^32 for every hash value. -/
theorem head_index_u32 (m : Nat) (h : Fin 16) :
    m % vocabs[h.val]! + offsets[h.val]! < 4294967296 := by
  have h1 := head_index_bounds m vocabs[h.val]! offsets[h.val]! (vocabs_pos h)
  have h2 := head_ranges_fit_u32 h
  omega

/-- Distinct heads of the same hash value address disjoint ranges (offsets are prefix sums). -/
theorem head_ranges_disjoint :
    ∀ i : Fin 15, offsets[i.val]! + vocabs[i.val]! ≤ offsets[i.val + 1]! := by
  intro i
  have := offsets_prefix_sums i
  omega

-- § ngram_mixed

/-- Model of `ngram_mixed`: first product assigned, the rest XORed in. -/
def ngramMixed (ctx mult : Nat → UInt64) : Nat → UInt64
  | 0 => 0
  | 1 => ctx 0 * mult 0
  | n + 2 => ngramMixed ctx mult (n + 1) ^^^ (ctx (n + 1) * mult (n + 1))

/-- Bigram hash unfolds to the two-term XOR. -/
theorem ngramMixed_two (ctx mult : Nat → UInt64) :
    ngramMixed ctx mult 2 = (ctx 0 * mult 0) ^^^ (ctx 1 * mult 1) := rfl

/-- Trigram hash unfolds to the three-term XOR. -/
theorem ngramMixed_three (ctx mult : Nat → UInt64) :
    ngramMixed ctx mult 3 = ((ctx 0 * mult 0) ^^^ (ctx 1 * mult 1)) ^^^ (ctx 2 * mult 2) := rfl

/-- The hash of the first `n` positions depends only on those positions. -/
theorem ngramMixed_congr (ctx ctx' mult : Nat → UInt64) :
    ∀ n, (∀ j, j < n → ctx j = ctx' j) → ngramMixed ctx mult n = ngramMixed ctx' mult n
  | 0, _ => rfl
  | 1, h => by simp [ngramMixed, h 0 (by omega)]
  | n + 2, h => by
      simp only [ngramMixed]
      rw [ngramMixed_congr ctx ctx' mult (n + 1) (fun j hj => h j (by omega)), h (n + 1) (by omega)]

-- § context cut

/-- One step of the history walk: returns the new cut flag and the stored context value. -/
def ctxStep (eos : Int) (cut : Bool) (raw : Int) : Bool × Int :=
  let t := if cut then TOKEN_NULL else raw
  let cut' := cut || decide (t < 0) || decide (t = eos)
  (cut', if cut' then eos else t)

/-- Once cut, the flag stays set and the stored value is EOS. -/
theorem ctxStep_sticky (eos raw : Int) :
    (ctxStep eos true raw).1 = true ∧ (ctxStep eos true raw).2 = eos := by
  simp [ctxStep, TOKEN_NULL]

/-- A real, non-EOS predecessor passes through unchanged and does not cut. -/
theorem ctxStep_passthrough (eos raw : Int) (h0 : 0 ≤ raw) (hne : raw ≠ eos) :
    ctxStep eos false raw = (false, raw) := by
  have h1 : ¬ raw < 0 := by omega
  simp [ctxStep, h1, hne]

/-- An EOS predecessor is itself stored as EOS and cuts everything after. -/
theorem ctxStep_eos_cuts (eos : Int) :
    (ctxStep eos false eos).1 = true ∧ (ctxStep eos false eos).2 = eos := by
  simp [ctxStep]

/-- A missing (negative) predecessor cuts. -/
theorem ctxStep_negative_cuts (eos raw : Int) (h : raw < 0) :
    (ctxStep eos false raw).1 = true := by
  simp [ctxStep, h]

/-- The whole history walk over the raw predecessors. -/
def ctxChain (eos : Int) : Bool → List Int → List Int
  | _, [] => []
  | cut, r :: rs => (ctxStep eos cut r).2 :: ctxChain eos (ctxStep eos cut r).1 rs

/-- The walk preserves length. -/
theorem ctxChain_length (eos : Int) (rs : List Int) :
    ∀ cut, (ctxChain eos cut rs).length = rs.length := by
  induction rs with
  | nil => intro cut; rfl
  | cons r rs ih => intro cut; simp [ctxChain, ih]

/-- After a cut every remaining position is EOS. -/
theorem ctxChain_after_cut (eos : Int) :
    ∀ rs : List Int, ctxChain eos true rs = List.replicate rs.length eos
  | [] => rfl
  | r :: rs => by
      have h : ctxStep eos true r = (true, eos) := by simp [ctxStep, TOKEN_NULL]
      simp [ctxChain, h, List.replicate_succ, ctxChain_after_cut eos rs]

/-- The position whose predecessor is EOS is EOS, and so is all later history. -/
theorem ctxChain_eos_then_eos (eos : Int) (rs : List Int) :
    ctxChain eos false (eos :: rs) = eos :: List.replicate rs.length eos := by
  have h : ctxStep eos false eos = (true, eos) := by simp [ctxStep]
  simp [ctxChain, h, ctxChain_after_cut]

-- § index arithmetic of ngram_rows

/-- `prev` is oldest first: predecessor `s` back is entry `2 - s`, in range. -/
theorem prev_index_lt (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) : 2 - s < 2 := by omega

/-- Predecessor 1 is the NEWEST entry (index 1), predecessor 2 the oldest (index 0). -/
theorem prev_newest_oldest : (2 - 1 = 1) ∧ (2 - 2 = 0) := by decide

/-- The prev row of token `i` stays inside its own two-entry window. -/
theorem prev_offset_lt (i s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) :
    i * 2 + (2 - s) < (i + 1) * 2 := by omega

/-- Output index of head `h` for token `i` is inside token `i`'s 16-wide block. -/
theorem out_index_lt (i h : Nat) (hh : h < 16) : i * 16 + h < (i + 1) * 16 := by omega

/-- Head slot of (n-gram size, group) is below 16. -/
theorem head_slot_lt (n g : Nat) (hn : 2 ≤ n) (hn3 : n ≤ 3) (hg : g < 8) :
    (n - 2) * 8 + g < 16 := by omega

/-- (n, g) -> head slot is injective, so every head is written exactly once. -/
theorem head_slot_inj (n g n' g' : Nat) (hg : g < 8) (hg' : g' < 8) :
    (n - 2) * 8 + g = (n' - 2) * 8 + g' → n - 2 = n' - 2 ∧ g = g' := by omega

-- § IQ4_NL table and dequantisation layout

/-- The non-linear 4-bit codebook. -/
def iq4nlTable : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook lookup with the 4-bit mask (`code & 15`, written as `% 16`). -/
def iq4nl (code : Nat) : Int := iq4nlTable[code % 16]!

/-- The table has 16 entries. -/
theorem iq4nl_length : iq4nlTable.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_strictly_increasing :
    ∀ i : Fin 15, iq4nlTable[i.val]! < iq4nlTable[i.val + 1]! := by decide

/-- Every codebook value fits int8 and avoids -128. -/
theorem iq4nl_range : ∀ i : Fin 16, -127 ≤ iq4nlTable[i.val]! ∧ iq4nlTable[i.val]! ≤ 127 := by
  decide

/-- Masking is idempotent. -/
theorem iq4nl_mask (code : Nat) : iq4nl code = iq4nl (code % 16) := by
  simp [iq4nl, Nat.mod_mod]

/-- A byte splits into low and high nibbles, each a valid code. -/
theorem nibble_split (b : Nat) (hb : b < 256) : b % 16 + 16 * (b / 16) = b ∧ b / 16 < 16 := by
  omega

/-- Split halves: every element of a 32-block is reached as `j` or `j+16` with `j < 16`. -/
theorem dequant_cover (e : Nat) (he : e < 160) :
    ∃ b j, b < 5 ∧ j < 16 ∧ (e = b * 32 + j ∨ e = b * 32 + j + 16) := by
  by_cases h : e % 32 < 16
  · exact ⟨e / 32, e % 32, by omega, h, Or.inl (by omega)⟩
  · exact ⟨e / 32, e % 32 - 16, by omega, by omega, Or.inr (by omega)⟩

/-- Low-nibble destinations are injective. -/
theorem dequant_lo_inj (b j b' j' : Nat) (hj : j < 16) (hj' : j' < 16)
    (h : b * 32 + j = b' * 32 + j') : b = b' ∧ j = j' := by omega

/-- High-nibble destinations are injective. -/
theorem dequant_hi_inj (b j b' j' : Nat) (hj : j < 16) (hj' : j' < 16)
    (h : b * 32 + j + 16 = b' * 32 + j' + 16) : b = b' ∧ j = j' := by omega

/-- Low and high destinations never collide, so no element is written twice. -/
theorem dequant_lo_hi_disjoint (b j b' j' : Nat) (hj : j < 16) (hj' : j' < 16)
    (h : b * 32 + j = b' * 32 + j' + 16) : False := by omega

/-- Each 18-byte block stays inside the 90-byte row. -/
theorem block_fits (b : Nat) (hb : b < 5) : b * 18 + 18 ≤ 90 := by omega

/-- Head-slowest gather: head `h`, element `d` lands inside the 2560 buffer. -/
theorem gather_index_lt (h d : Nat) (hh : h < 16) (hd : d < 160) : h * 160 + d < 2560 := by
  omega

/-- The gather layout is injective in (h, d). -/
theorem gather_index_inj (h d h' d' : Nat) (hd : d < 160) (hd' : d' < 160)
    (e : h * 160 + d = h' * 160 + d') : h = h' ∧ d = d' := by omega

/-- A row inside the table has all 90 bytes inside the data region. -/
theorem row_in_table (row n : Nat) (hr : row < n) : (row + 1) * 90 ≤ n * 90 := by omega

/-- The prefetch list never has more ranges than rows (out-of-range rows are skipped). -/
theorem prefetch_ranges_le (rows : List Nat) (n : Nat) :
    (rows.filter (· < n)).length ≤ rows.length :=
  List.length_filter_le _ _

end Strata.Kernels.Ngram
