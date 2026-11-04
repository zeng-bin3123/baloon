/-!
# MODULE: ple_block_kernels
Scratch carving, history ring, dilated conv rows, bf16 rounding and overlap checks of the PLE block
(`ple_block`, `history_advance_kernel`, `conv_kernel`, `to_bf16_kernel`, `ple_block_scratch_bytes`).

## Source semantics
The caller owns one scratch block: five `hc_dim` float buffers, then `n_embd` and `hc` floats (one float
region), then a Q8 activation image of `(n_embd/32)*34` bytes, then a BF16 copy of the embedding; each
region 16-byte aligned.  The float region is carved as key, query, norm, gated, conv (each `D`), value (`N`),
gate (`H`).  The conv history is row-fastest: `hist[row + 9*c]`.  With kernel 4, dilation 3, history 9 the
taps read rows 0, 3, 6, 9 and row 9 is the NEW row.  `history_advance` shifts each column up by one and
appends `normalized`.  `bf16_bits` rounds to nearest even using `(i + ((i>>16)&1) + 0x7FFF) & 0xFFFF0000`.
`overlap` returns false for null or empty spans and otherwise is interval intersection.

## Form template identifiers
- `scratch`          -> offsets `offKey .. offGate`
- `hist`             -> `histIdx`
- `bf16_bits`        -> `bf16Bits`
- `overlap`          -> `overlapP`
-/

namespace Strata.Kernels.PleBlock

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § scratch layout

/-- Embedding width. -/
def NG_N_EMBD : Nat := 2560
/-- Streams. -/
def NG_HC : Nat := 4
/-- Flattened width. -/
def NG_HC_DIM : Nat := NG_N_EMBD * NG_HC
/-- History rows. -/
def NG_HIST : Nat := 9

/-- Round up to 16. -/
def alignUp (x : Nat) : Nat := ((x + 15) / 16) * 16

/-- Aligned size covers the raw size. -/
theorem alignUp_ge (x : Nat) : x ≤ alignUp x := by
  unfold alignUp
  omega

/-- Aligned size is a multiple of 16. -/
theorem alignUp_mod (x : Nat) : alignUp x % 16 = 0 := by
  unfold alignUp
  omega

/-- Total scratch bytes as in `ple_block_scratch_bytes`. -/
def scratchBytes (f q e : Nat) : Nat := alignUp f + alignUp q + e + 256

/-- At the real geometry the scratch is 223,152 bytes. -/
theorem scratch_real :
    scratchBytes ((5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * 4) ((NG_N_EMBD / 32) * 34) (NG_N_EMBD * 2)
      = 223152 := by
  decide

/-- The three carved regions are in order and fit: floats end before the Q8 image, which ends before
the BF16 copy, which ends before the 256-byte tail. -/
theorem regions_fit (f q e : Nat) :
    f ≤ alignUp f ∧ alignUp f + q ≤ alignUp f + alignUp q ∧
      alignUp f + alignUp q + e ≤ scratchBytes f q e := by
  have h1 := alignUp_ge f
  have h2 := alignUp_ge q
  unfold scratchBytes
  omega

/-- The float region is carved into seven consecutive pieces ending at `5*D + N + H` floats. -/
theorem float_region_total : 5 * NG_HC_DIM + NG_N_EMBD + NG_HC = 53764 := by
  decide

/-- Offsets (in floats) of key, query, norm, gated, conv, value, gate. -/
def floatOffsets : List Nat :=
  [0, NG_HC_DIM, 2 * NG_HC_DIM, 3 * NG_HC_DIM, 4 * NG_HC_DIM, 5 * NG_HC_DIM,
    5 * NG_HC_DIM + NG_N_EMBD]

/-- Piece sizes in floats, in the same order. -/
def floatSizes : List Nat := [NG_HC_DIM, NG_HC_DIM, NG_HC_DIM, NG_HC_DIM, NG_HC_DIM, NG_N_EMBD, NG_HC]

/-- Each piece starts exactly where the previous one ends, so no two pieces overlap. -/
theorem float_pieces_adjacent :
    (floatOffsets.zip floatSizes).map (fun p => p.1 + p.2) = floatOffsets.drop 1 ++ [53764] := by
  decide

-- § history and conv rows

/-- Row-fastest history address `row + hist*c`. -/
def histIdx (nhist c row : Nat) : Nat := row + nhist * c

/-- A history address is inside `nhist * hc_dim`. -/
theorem histIdx_lt {nhist c row hc_dim : Nat} (hc : c < hc_dim) (hr : row < nhist) :
    histIdx nhist c row < nhist * hc_dim := by
  unfold histIdx
  have h := mul_add_lt_aux hc hr
  have e1 : nhist * c = c * nhist := Nat.mul_comm _ _
  have e2 : nhist * hc_dim = hc_dim * nhist := Nat.mul_comm _ _
  omega

/-- History addresses are unique per `(channel, row)`. -/
theorem histIdx_inj {nhist c c' row row' : Nat} (hr : row < nhist) (hr' : row' < nhist)
    (h : histIdx nhist c row = histIdx nhist c' row') : c = c' ∧ row = row' := by
  unfold histIdx at h
  have e1 : nhist * c = c * nhist := Nat.mul_comm _ _
  have e2 : nhist * c' = c' * nhist := Nat.mul_comm _ _
  have h' : c * nhist + row = c' * nhist + row' := by omega
  obtain ⟨a, b⟩ := split_unique_aux hr hr' h'
  exact ⟨a, b⟩
where
  /-- Row-major addresses are unique. -/
  split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
      (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
    have hn : 0 < n := by omega
    have d1 : (a * n + c) / n = a := by
      have h1 := Nat.add_mul_div_right c a hn
      have h0 : c / n = 0 := Nat.div_eq_of_lt hc
      have e : a * n + c = c + a * n := Nat.add_comm _ _
      rw [e]
      omega
    have d2 : (a' * n + c') / n = a' := by
      have h1 := Nat.add_mul_div_right c' a' hn
      have h0 : c' / n = 0 := Nat.div_eq_of_lt hc'
      have e : a' * n + c' = c' + a' * n := Nat.add_comm _ _
      rw [e]
      omega
    have ha : a = a' := by rw [← d1, ← d2, h]
    subst ha
    exact ⟨rfl, by omega⟩

/-- The tap `k` reads row `nhist - (kern-1-k)*dil`; at the real geometry rows are `0, 3, 6, 9`. -/
theorem conv_tap_rows : ∀ k : Nat, k < 4 → 9 - (4 - 1 - k) * 3 = 3 * k := by
  intro k hk
  omega

/-- Only the last tap reads the new row (`row = nhist`); the three others are inside the history. -/
theorem only_last_tap_is_new : ∀ k : Nat, k < 4 → (9 - (4 - 1 - k) * 3 = 9 ↔ k = 3) := by
  intro k hk
  omega

/-- The history shift: drop the oldest row, append the new one. -/
def advance (col : List Int) (x : Int) : List Int := col.tail ++ [x]

/-- The shifted column keeps its length when non-empty. -/
theorem advance_length (col : List Int) (x : Int) (h : col ≠ []) :
    (advance col x).length = col.length := by
  cases col with
  | nil => exact absurd rfl h
  | cons a as => simp [advance]

/-- The newest row of the shifted column is the appended value. -/
theorem advance_last (col : List Int) (x : Int) : (advance col x).getLast? = some x := by
  simp [advance]

-- § bf16 rounding

/-- `(i + ((i>>16)&1) + 0x7FFF) & 0xFFFF0000 >> 16` on a 32-bit word. -/
def bf16Bits (i : Nat) : Nat := ((i + (i / 65536) % 2 + 32767) % 4294967296) / 65536

/-- Aligned-word fact: `(a*65536 + t) mod 2^32` keeps `a mod 65536` in its high half. -/
theorem core_div2 (a t : Nat) (ht : t < 65536) :
    ((a * 65536 + t) % 4294967296) / 65536 = a % 65536 := by
  omega

/-- Closed form of the rounding: the high half plus the carry out of the low half, modulo 2^16. -/
theorem bf16Bits_eq (i : Nat) :
    bf16Bits i = (i / 65536 + (i % 65536 + (i / 65536) % 2 + 32767) / 65536) % 65536 := by
  unfold bf16Bits
  have e : i + (i / 65536) % 2 + 32767 =
      (i / 65536 + (i % 65536 + (i / 65536) % 2 + 32767) / 65536) * 65536 +
        (i % 65536 + (i / 65536) % 2 + 32767) % 65536 := by
    omega
  rw [e]
  exact core_div2 _ _ (Nat.mod_lt _ (by decide))

/-- A bf16 bit pattern has 16 bits. -/
theorem bf16Bits_lt (i : Nat) : bf16Bits i < 65536 := by
  unfold bf16Bits
  omega

/-- Below the halfway point the high half is kept. -/
theorem bf16_round_down {i : Nat} (hi : i < 4294967296) (h : i % 65536 < 32768) :
    bf16Bits i = i / 65536 := by
  rw [bf16Bits_eq]
  have hc : (i % 65536 + (i / 65536) % 2 + 32767) / 65536 = 0 := by omega
  rw [hc]
  omega

/-- Above the halfway point the high half is incremented. -/
theorem bf16_round_up {i : Nat} (hi : i < 4294967296) (h : 32768 < i % 65536) :
    bf16Bits i = (i / 65536 + 1) % 65536 := by
  rw [bf16Bits_eq]
  have hc : (i % 65536 + (i / 65536) % 2 + 32767) / 65536 = 1 := by omega
  rw [hc]

/-- At exactly the halfway point the result is the even neighbour. -/
theorem bf16_tie_even {i : Nat} (hi : i < 4294967296) (h : i % 65536 = 32768) :
    bf16Bits i = (i / 65536 + (i / 65536) % 2) % 65536 := by
  rw [bf16Bits_eq]
  have hc : (i % 65536 + (i / 65536) % 2 + 32767) / 65536 = (i / 65536) % 2 := by omega
  rw [hc]

/-- Expanding a bf16 pattern is a left shift by 16, and rounding it again is the identity. -/
theorem bf16_roundtrip {h : Nat} (hh : h < 65536) : bf16Bits (h * 65536) = h := by
  have h1 : (h * 65536) % 65536 < 32768 := by omega
  rw [bf16_round_down (by omega) h1]
  omega

-- § overlap rule

/-- Interval intersection for non-null, non-empty spans, as `overlap` computes it. -/
def overlapP (a an b bn : Nat) : Prop :=
  (a < b ∧ b - a < an) ∨ (¬ a < b ∧ a - b < bn)

/-- For non-empty spans the source's overlap test is exactly interval intersection. -/
theorem overlapP_iff {a an b bn : Nat} (ha : 0 < an) (hb : 0 < bn) :
    overlapP a an b bn ↔ (a < b + bn ∧ b < a + an) := by
  unfold overlapP
  omega

/-- The test is symmetric on non-empty spans. -/
theorem overlapP_symm {a an b bn : Nat} (ha : 0 < an) (hb : 0 < bn) :
    overlapP a an b bn ↔ overlapP b bn a an := by
  rw [overlapP_iff ha hb, overlapP_iff hb ha]
  omega

end Strata.Kernels.PleBlock
