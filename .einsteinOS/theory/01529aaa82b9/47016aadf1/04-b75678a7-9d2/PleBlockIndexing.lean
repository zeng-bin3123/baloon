/-!
# MODULE: ple_block_indexing
History shift, conv tap rows, stream layout and scratch sizing of the PLE block in `ple.cu`.

## Source semantics
* `history_advance` shifts each channel column up by one row and appends the new normalised value last.
* The depthwise conv for kernel 4, dilation 3 and history 9 reads rows `hist - (3-k)*3 = 0, 3, 6, 9`; only
  tap 3 reads the new row (row 9 = `hist`); the history layout is row-fastest: `row + hist*channel`.
* A stream `c` of the grouped norm occupies `[c*n_embd, (c+1)*n_embd)`.
* The gate is `sigmoid(sgn(s)*sqrt(max(|s|, 1e-6)))`; `sgn` is odd.
* Scratch regions are rounded up to 16 bytes.

## Form template identifiers
NG_HIST -> history_rows; NG_HC_DIM -> stream_dim; NG_HC -> stream_count; NG_N_EMBD -> embedding_dim;
kern -> conv_kernel; dil -> conv_dilation.
-/

namespace PleBlockIndexing

-- § history advance

/-- Shift one channel column: drop the oldest entry, append the new one. -/
def advance (col : List Int) (x : Int) : List Int := col.tail ++ [x]

/-- The column length is preserved for a non-empty column. -/
theorem advance_length (col : List Int) (x : Int) (h : col ≠ []) : (advance col x).length = col.length := by
  cases col with
  | nil => exact absurd rfl h
  | cons a as => simp [advance]

/-- The new value is the last entry. -/
theorem advance_last (col : List Int) (x : Int) : (advance col x).getLast? = some x := by
  simp [advance]

/-- The kept entries are the previous ones shifted by one. -/
theorem advance_shift (a : Int) (as : List Int) (x : Int) : advance (a :: as) x = as ++ [x] := by
  simp [advance]

-- § conv taps

/-- Row read by tap `k`. -/
def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- The four taps of the real geometry read rows 0, 3, 6, 9. -/
theorem tap_rows : [0, 1, 2, 3].map (fun k => tapRow 9 4 3 k) = [0, 3, 6, 9] := by decide

/-- Only the last tap reads the new row. -/
theorem only_last_tap_new (k : Nat) (hk : k < 4) : tapRow 9 4 3 k = 9 ↔ k = 3 := by
  unfold tapRow; omega

/-- Tap rows never exceed the history length. -/
theorem tap_row_le (k : Nat) : tapRow 9 4 3 k ≤ 9 := by unfold tapRow; omega

/-- A history row below `nhist` of channel `c` is inside the flat history buffer. -/
theorem hist_index_lt (row nhist c hc : Nat) (hr : row < nhist) (hch : c < hc) :
    row + nhist * c < nhist * hc := by
  have h1 : nhist * (c + 1) ≤ nhist * hc := Nat.mul_le_mul_left _ hch
  rw [Nat.mul_add, Nat.mul_one] at h1
  omega

-- § stream layout

/-- Stream `c` element `d` stays inside the `hc * n_embd` vector. -/
theorem stream_index_lt (c d n hc : Nat) (hc' : c < hc) (hd : d < n) : c * n + d < hc * n := by
  have h1 : (c + 1) * n ≤ hc * n := Nat.mul_le_mul_right _ hc'
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- The broadcast gate index of a flat element is a valid stream. -/
theorem gate_index_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- The broadcast value index is inside the embedding. -/
theorem value_index_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- Real geometry: four streams of 2560 make 10240. -/
theorem geometry : 4 * 2560 = 10240 ∧ 10240 / 2560 = 4 := by decide

-- § gate sign

/-- Sign function. -/
def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- The sign is odd, which makes the gate symmetric about 1/2. -/
theorem sgn_neg (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn; split_ifs <;> omega

/-- The sign times the value is non-negative (the argument of the root is a magnitude). -/
theorem sgn_mul_nonneg (s : Int) : 0 ≤ sgn s * s := by
  unfold sgn
  split_ifs with h1 h2
  · omega
  · omega
  · omega

-- § scratch sizing

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Rounding yields a multiple of 16 (the float and byte regions stay aligned). -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

/-- Scratch layout: float region, Q8 activation image, BF16 embedding copy, then 256 bytes of slack. -/
theorem scratch_ordering (f q e : Nat) : align16 f + align16 q + e + 256 ≥ f + q + e := by
  have h1 := align16_ge f
  have h2 := align16_ge q
  omega

end PleBlockIndexing
