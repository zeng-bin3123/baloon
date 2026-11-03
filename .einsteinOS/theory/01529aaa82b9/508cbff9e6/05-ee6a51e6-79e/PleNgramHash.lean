/-!
# MODULE: ple_ngram_hash
The per-layer-embedding (PLE) n-gram hash and the IQ4_NL table row read.

## Source semantics
For each token, the context is `[token, prev newest, ..., prev oldest]`; `prev` is stored oldest first, so
predecessor `s` positions back is entry `n_prev - s`. The context is cut at the first negative token or EOS: the
cut position and everything after it hold EOS. For n = 2..NGRAM_SIZE the mixed value
`ctx[0]*m[0] ^ ctx[1]*m[1] ^ ...` (products wrap mod 2^64) selects, for each head h, the row
`mixed % vocab[h] + offset[h]`. Head offsets are the prefix sums of the head vocab sizes, so heads address
disjoint row ranges. A row is 5 IQ4_NL blocks of 18 bytes = 90 bytes for 160 values; a token reads 16 rows.
(NGRAM_SIZE = 3 and 8 heads per n-gram are inferred from the three multipliers and sixteen heads.)

## Form template identifiers
- `ple_artifact_consts` -> vocab / offset, `ngram_mixed` -> mixed, `ngram_rows` -> context cut
- `PLE_ROW_BYTES` -> 90, `PLE_HEAD_DIM` -> 160, `PLE_N_HEADS` -> 16
-/

namespace Strata.PleNgram

-- § artifact constants

/-- Per-head vocabulary sizes. -/
def vocab : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets. -/
def offset : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- There are 16 heads. -/
theorem heads_count : vocab.length = 16 ∧ offset.length = 16 := by decide

/-- Offsets are the prefix sums of the vocab sizes: each head's range ends where the next begins. -/
theorem offsets_prefix_sum :
    List.zipWith (· + ·) offset vocab = offset.drop 1 ++ [320001446] := by decide

/-- The sizes fit in the 320001536-row table. -/
theorem total_fits : 320001446 ≤ 320001536 := by decide

/-- Table size in bytes (about 26.8 GiB). -/
theorem table_bytes : 320001536 * 90 = 28800138240 := by decide

/-- All vocab sizes are positive, so the modulus is safe. -/
theorem vocab_pos : vocab.all (fun v => decide (0 < v)) = true := by decide

-- § row selection

/-- A head's selected row lies in `[offset, offset + vocab)`. -/
theorem head_slot_bounds (m v o : Nat) (hv : 0 < v) : o ≤ m % v + o ∧ m % v + o < o + v := by
  have := Nat.mod_lt m hv
  omega

/-- Two heads whose ranges are consecutive never select the same row. -/
theorem consecutive_heads_disjoint (m m' v o : Nat) (hv : 0 < v) (v' : Nat) (hv' : 0 < v') :
    m % v + o ≠ m' % v' + (o + v) := by
  have := Nat.mod_lt m hv
  omega

-- § mixing

/-- XOR-fold of products over the remaining context, starting from an accumulator. -/
def mixedFrom (acc : Nat) : List Nat → List Nat → Nat
  | c :: cs, m :: ms => mixedFrom ((acc ^^^ (c * m % 2 ^ 64))) cs ms
  | _, _ => acc

/-- `ngram_mixed`: first product assigned, the rest XORed in. -/
def mixed (ctx mult : List Nat) : Nat :=
  match ctx, mult with
  | c :: cs, m :: ms => mixedFrom (c * m % 2 ^ 64) cs ms
  | _, _ => 0

/-- A one-token context hashes to the wrapped product. -/
theorem mixed_singleton (c m : Nat) : mixed [c] [m] = c * m % 2 ^ 64 := by
  simp [mixed, mixedFrom]

/-- Two-token context XORs the second product into the first. -/
theorem mixed_pair (c0 c1 m0 m1 : Nat) :
    mixed [c0, c1] [m0, m1] = (c0 * m0 % 2 ^ 64) ^^^ (c1 * m1 % 2 ^ 64) := by
  simp [mixed, mixedFrom]

-- § context cut

/-- Context tail: once cut, every later position is EOS; a position whose value is negative or EOS cuts. -/
def ctxTail (eos : Int) : Bool → List Int → List Int
  | _, [] => []
  | cut, t :: ts =>
      let cut' := cut || decide (t < 0) || decide (t = eos)
      (if cut' then eos else t) :: ctxTail eos cut' ts

/-- After a cut everything is EOS. -/
theorem ctxTail_cut (eos : Int) (ts : List Int) : ctxTail eos true ts = ts.map (fun _ => eos) := by
  induction ts with
  | nil => rfl
  | cons t ts ih => simp [ctxTail, ih]

/-- The context length is preserved. -/
theorem ctxTail_length (eos : Int) (cut : Bool) (ts : List Int) : (ctxTail eos cut ts).length = ts.length := by
  induction ts generalizing cut with
  | nil => rfl
  | cons t ts ih => simp [ctxTail, ih]

/-- Predecessor `s` back (1 ≤ s ≤ n_prev) reads entry `n_prev - s`, which is in range. -/
theorem prev_index_in_range (nPrev s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ nPrev) : nPrev - s < nPrev := by
  omega

/-- The newest predecessor (s = 1) is the last stored entry. -/
theorem newest_is_last (nPrev : Nat) (h : 1 ≤ nPrev) : nPrev - 1 + 1 = nPrev := by omega

-- § row layout and IQ4_NL decoding

/-- A row is 5 blocks of 18 bytes. -/
theorem row_bytes : 160 / 32 * 18 = 90 := by decide

/-- A token's sixteen heads fill the 2560-wide embedding and read 1440 bytes. -/
theorem token_geometry : 16 * 160 = 2560 ∧ 16 * 90 = 1440 := by decide

/-- A packed byte holds two nibbles that reassemble it (split halves: elements j and j+16). -/
theorem nibble_split (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 ∧ b % 16 + 16 * (b / 16) = b := by
  omega

/-- Output positions `j` and `j + 16` of a block never collide for `j < 16`. -/
theorem split_halves_disjoint (j j' : Nat) (hj : j < 16) (hj' : j' < 16) : j ≠ j' + 16 := by omega

end Strata.PleNgram
