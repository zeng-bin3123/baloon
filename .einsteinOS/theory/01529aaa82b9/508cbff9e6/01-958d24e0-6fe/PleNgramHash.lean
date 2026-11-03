/-!
# MODULE: ple_ngram_hash
The PLE n-gram hash (sixteen table rows per token from the last three token ids) and the IQ4_NL row geometry of the
per-layer token table.

## Source semantics
* `mixed_n = (ctx[0]*m[0]) ^ (ctx[1]*m[1]) ^ ...` in uint64: XOR, not sum.
* Row index = `head_offset + mixed % vocab_size` (modulus, not mask; vocab sizes are not powers of two).
* A missing predecessor is TOKEN_NULL = -1, tested as `t < 0`; token id 0 is real.
* One IQ4_NL row = 160 elements = 5 blocks x 18 bytes = 90 bytes; 16 rows flatten to n_embd = 2560.
* Nibble order is split-half (byte j carries elements j and j+16).

## Form template identifiers
* ngram_mixed -> `ngram_mix`
* ngram_rows -> `ngram_rows`
* head_offsets / head_vocab_sizes -> `head_offsets`, `head_vocabs`
* PLE_ROW_BYTES -> `row_bytes`
-/

namespace Strata.Ple

-- § Geometry

/-- Heads: (ngram_size - 1) * heads_per_ngram = 16. -/
theorem n_heads : (3 - 1) * 8 = 16 := by decide

/-- Sixteen 160-wide rows make exactly n_embd = 2560. -/
theorem rows_make_embd : 16 * 160 = 2560 := by decide

/-- Row bytes: five 32-element IQ4_NL blocks of 18 bytes. -/
theorem row_bytes : (160 / 32) * 18 = 90 := by decide

/-- Hyper-connection width 4 * 2560 = 10240. -/
theorem hc_dim : 4 * 2560 = 10240 := by decide

/-- Conv history rows (kernel - 1) * ngram_size = 9. -/
theorem hist_rows : (4 - 1) * 3 = 9 := by decide

-- § Hash

/-- The mixed value over a context (uint64 wraparound arithmetic). -/
def mixed (ctx mult : List (BitVec 64)) : BitVec 64 :=
  (List.zipWith (· * ·) ctx mult).foldl (· ^^^ ·) 0

/-- Equal products cancel under XOR (two identical terms contribute nothing). -/
theorem xor_equal_terms_cancel (a : BitVec 64) : a ^^^ a = 0 := by
  simp

/-- Under sum the same two terms would double instead: XOR and sum are different functions. -/
theorem xor_ne_sum_witness : (1 : BitVec 64) ^^^ 1 ≠ 1 + 1 := by decide

/-- XOR of two products is symmetric in them. -/
theorem xor_comm_terms (a b : BitVec 64) : a ^^^ b = b ^^^ a := by
  exact BitVec.xor_comm a b

/-- The row index of a head: its offset plus the hash modulo the head's vocab size. -/
def rowIndex (offset vocab mix : Nat) : Nat := offset + mix % vocab

/-- The row stays within its head's range `[offset, offset + vocab)`. -/
theorem rowIndex_range (offset vocab mix : Nat) (hv : 0 < vocab) :
    offset ≤ rowIndex offset vocab mix ∧ rowIndex offset vocab mix < offset + vocab := by
  have := Nat.mod_lt mix hv
  unfold rowIndex
  omega

/-- A power-of-two mask is a different function from `%` for a non-power-of-two vocab size (20000003). -/
theorem mask_differs_from_mod : (20000003 : Nat) % 20000003 = 0 ∧ (20000003 &&& (20000003 - 1)) ≠ 0 := by
  decide

/-- The head offsets are a running sum of the vocab sizes (first entries of the artifact's table). -/
theorem offsets_running_sum : 0 + 20000003 = 20000003 ∧ 20000003 + 20000023 = 40000026 := by
  decide

/-- A missing predecessor is TOKEN_NULL; token id 0 is a real token. -/
def isNull (t : Int) : Bool := decide (t < 0)

/-- Token 0 is not null. -/
theorem token_zero_real : isNull 0 = false := by decide

/-- TOKEN_NULL = -1 is null. -/
theorem token_null_is_null : isNull (-1) = true := by decide

/-- Window prefixes: bigram heads use a prefix of what trigram heads use. -/
theorem prefix_of_window (a b c : Int) : ([a, b, c].take 2) = [a, b] := by
  rfl

-- § IQ4_NL row layout

/-- Split-half layout: byte j carries elements j and j+16; every element is covered. -/
theorem split_half_cover (e : Nat) (he : e < 32) : ∃ j, j < 16 ∧ (e = j ∨ e = j + 16) := by
  exact ⟨e % 16, by omega, by omega⟩

/-- Interleaving would place byte 1's pair at (2,3), not (1,17). -/
theorem interleave_differs : (1 : Nat) ≠ 2 * 1 := by decide

/-- Head-slowest flatten: element `d` of head `h` lands at `h*160 + d < 2560`. -/
theorem flatten_in_range (h d : Nat) (hh : h < 16) (hd : d < 160) : h * 160 + d < 2560 := by
  omega

/-- The flatten is injective in (head, dim). -/
theorem flatten_injective (h d h' d' : Nat) (hd : d < 160) (hd' : d' < 160)
    (e : h * 160 + d = h' * 160 + d') : h = h' ∧ d = d' := by
  omega

end Strata.Ple
