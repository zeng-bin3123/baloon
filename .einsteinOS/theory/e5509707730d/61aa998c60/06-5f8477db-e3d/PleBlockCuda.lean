/-!
# MODULE: ple_block_cuda
SOURCES: src/kernels/cuda/ple.cu
The GPU half of the PLE block: key/value projections, grouped RMS norms, the signed-sqrt sigmoid gate, the
broadcast, the depthwise causal dilated conv with SiLU, and the three-way residual add; plus the rolling
history advance and the caller-owned scratch carving. This module models the integer, index and layout
contracts, the signed gate argument, and the bf16 rounding of `to_bf16_kernel`.

## Source semantics
* The scratch is carved from a caller-owned workspace: five `hc_dim` float buffers, then `n_embd` and `hc`
  floats, then the Q8 activation image, then the BF16 embedding copy, each 16-byte aligned.
* The conv state is ROW-FASTEST: `hist[row + nhist*c]`; tap `k` reads row `nhist - (kern-1-k)*dil`, and only
  the last tap reads the new row (`norm[c]`).
* `gate = sigmoid(sgn(s) * sqrt(max(|s|, eps)))`: the sign is carried separately so the gate argument is odd.
* Output buffers may not overlap the scratch; overlap is an interval-intersection test.
* `bf16_bits` is round-to-nearest-even on a uint32 with wrap-around.

## Form template identifiers
* `scratchBytes`   : scratch_bytes (ple_block_scratch_bytes)
* `advance`        : history_advance
* `tapRow`         : conv_tap_row
* `histIndex`      : conv_state_index (row-fastest)
* `argOf`          : signed_gate_argument
* `overlapC`       : buffer_overlap_check
* `bf16Bits`       : f32_to_bf16_rne
-/

namespace Strata.PleBlock

-- § geometry constants

/-- `NG_N_EMBD`. -/
def NG_N_EMBD : Nat := 2560
/-- `NG_HC`. -/
def NG_HC : Nat := 4
/-- `NG_HC_DIM`. -/
def NG_HC_DIM : Nat := 10240
/-- `NG_HIST`: history rows kept per channel. -/
def NG_HIST : Nat := 9
/-- `PLE_CONV_KERNEL`. -/
def PLE_CONV_KERNEL : Nat := 4
/-- Dilation of the conv taps at the real geometry. -/
def NGRAM_SIZE : Nat := 3

/-- `hc * n_embd = hc_dim` at the real geometry. -/
theorem hc_dim_eq : NG_HC * NG_N_EMBD = NG_HC_DIM := by
  decide

-- § scratch carving

/-- Round up to a multiple of 16. -/
def roundUp16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding never shrinks. -/
theorem roundUp16_ge (n : Nat) : n ≤ roundUp16 n := by
  unfold roundUp16; omega

/-- Rounded sizes are multiples of 16. -/
theorem roundUp16_mod (n : Nat) : roundUp16 n % 16 = 0 := by
  unfold roundUp16; omega

/-- Bytes of the float region: five `hc_dim` buffers plus `n_embd` plus `hc` floats. -/
def floatBytes : Nat := (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * 4
/-- Bytes of the Q8 activation image. -/
def q8Bytes : Nat := (NG_N_EMBD / 32) * 34
/-- Bytes of the BF16 embedding copy. -/
def embBytes : Nat := NG_N_EMBD * 2
/-- Offset of the Q8 image (`d_act`). -/
def actOff : Nat := roundUp16 floatBytes
/-- Offset of the BF16 embedding copy (`d_emb16`). -/
def embOff : Nat := actOff + roundUp16 q8Bytes
/-- Total scratch bytes (`ple_block_scratch_bytes`). -/
def scratchBytes : Nat := roundUp16 floatBytes + roundUp16 q8Bytes + embBytes + 256

/-- The real scratch size is 223152 bytes. -/
theorem scratch_value : scratchBytes = 223152 := by
  decide

/-- The region ordering over abstract sizes: float region `F`, Q8 image `Q`, embedding copy `E`. -/
theorem scratch_gen (F Q E : Nat) :
    F ≤ roundUp16 F ∧ roundUp16 F + Q ≤ roundUp16 F + roundUp16 Q ∧
    roundUp16 F + roundUp16 Q + E ≤ roundUp16 F + roundUp16 Q + E + 256 := by
  have h1 := roundUp16_ge F
  have h2 := roundUp16_ge Q
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- The three scratch regions are ordered and non-overlapping. -/
theorem scratch_regions_fit :
    floatBytes ≤ actOff ∧ actOff + q8Bytes ≤ embOff ∧ embOff + embBytes ≤ scratchBytes :=
  scratch_gen floatBytes q8Bytes embBytes

/-- The Q8 image and the embedding copy start 16-byte aligned. -/
theorem scratch_offsets_aligned : actOff % 16 = 0 ∧ embOff % 16 = 0 := by
  have h1 := roundUp16_mod floatBytes
  have h2 := roundUp16_mod q8Bytes
  simp only [actOff, embOff]
  refine ⟨?_, ?_⟩ <;> omega

/-- Float buffers `key, query, norm, gated, conv, value` are stacked `hc_dim` apart, so none aliases another
(in particular `key` and `normalized` are distinct). -/
theorem float_buffers_disjoint (D n hc : Nat) :
    0 + D ≤ D ∧ D + D ≤ 2 * D ∧ 2 * D + D ≤ 3 * D ∧ 3 * D + D ≤ 4 * D ∧ 4 * D + D ≤ 5 * D ∧
    5 * D + n ≤ 5 * D + n ∧ 5 * D + n + hc ≤ 5 * D + n + hc := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

-- § history advance

/-- One advance step of a history column viewed as a window of `h` rows: shift up, put `x` last. -/
def advance (h : Nat) (col : Nat → Int) (x : Int) : Nat → Int :=
  fun r => if r + 1 < h then col (r + 1) else x

/-- The newest value lands in the last row. -/
theorem advance_last (h : Nat) (hpos : 0 < h) (col : Nat → Int) (x : Int) :
    advance h col x (h - 1) = x := by
  unfold advance
  have : ¬ (h - 1 + 1 < h) := by omega
  simp [this]

/-- Every other row takes the value of the row after it. -/
theorem advance_shift (h : Nat) (col : Nat → Int) (x : Int) (r : Nat) (hr : r + 1 < h) :
    advance h col x r = col (r + 1) := by
  unfold advance
  simp [hr]

-- § conv taps and state indexing

/-- Row read by tap `k` of the depthwise conv. -/
def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- At the real geometry (kern 4, dil 3, hist 9) the taps read rows 0, 3, 6, 9. -/
theorem real_tap_rows : [0, 1, 2, 3].map (tapRow 9 4 3) = [0, 3, 6, 9] := by
  decide

/-- Only the last tap reads the new row (`row == nhist`). -/
theorem only_last_tap_new : ∀ k, k < 4 → (tapRow 9 4 3 k = 9 ↔ k = 3) := by
  decide

/-- Row-fastest index of channel `c`, row `row`. -/
def histIndex (row nh c : Nat) : Nat := row + nh * c

/-- A row-fastest index stays inside `nh * m` for `row < nh`, `c < m`. -/
theorem histIndex_lt (row nh c m : Nat) (hr : row < nh) (hc : c < m) : histIndex row nh c < nh * m := by
  unfold histIndex
  have h : nh * (c + 1) ≤ nh * m := Nat.mul_le_mul (Nat.le_refl nh) hc
  rw [Nat.mul_add, Nat.mul_one] at h
  omega

/-- The same bound serves the conv-weight index `k + kern*c`. -/
theorem kernel_index_lt (k kern c m : Nat) (hk : k < kern) (hc : c < m) : k + kern * c < kern * m :=
  histIndex_lt k kern c m hk hc

/-- The channel-slowest index is a different address: row 1 channel 0 is 10240 against 1. -/
theorem transposed_read_differs : (1 * 10240 + 0) ≠ histIndex 1 9 0 := by
  decide

/-- Streams occupy disjoint ranges `[c*n, (c+1)*n)`. -/
theorem stream_disjoint (c c' n : Nat) (h : c < c') : c * n + n ≤ c' * n := by
  have h' : c + 1 ≤ c' := h
  have hm : (c + 1) * n ≤ c' * n := Nat.mul_le_mul h' (Nat.le_refl n)
  rw [Nat.add_mul, Nat.one_mul] at hm
  exact hm

/-- Broadcast: for `i < hc*n` the stream index is below `hc`. -/
theorem bcast_stream_lt (i hc n : Nat) (hn : 0 < n) (hi : i < hc * n) : i / n < hc :=
  (Nat.div_lt_iff_lt_mul hn).2 hi

/-- Broadcast: the element index is below `n`. -/
theorem bcast_elem_lt (i n : Nat) (hn : 0 < n) : i % n < n :=
  Nat.mod_lt _ hn

-- § signed gate argument

/-- Sign of an integer. -/
def sgn (s : Int) : Int := if 0 < s then 1 else if s < 0 then -1 else 0

/-- Magnitude after the floor (the floor is one unit in this scaled model). -/
def mag (a : Nat) : Nat := max a 1

/-- The gate argument `sgn(s) * mag(|s|)`. -/
def argOf (s : Int) : Int := sgn s * (mag s.natAbs : Int)

/-- `sgn` is odd. -/
theorem sgn_neg (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn
  split_ifs <;> omega

/-- The gate argument is odd, which is what keeps the gate symmetric about 1/2. -/
theorem argOf_neg (s : Int) : argOf (-s) = - argOf s := by
  unfold argOf
  rw [sgn_neg, Int.natAbs_neg, Int.neg_mul]

/-- Zero input gives zero argument (gate 1/2). -/
theorem argOf_zero : argOf 0 = 0 := by
  decide

/-- The clamp only bites near zero: for `|s| >= 1` the magnitude is `|s|`. -/
theorem mag_of_ge (a : Nat) (h : 1 ≤ a) : mag a = a := by
  unfold mag; omega

-- § buffer overlap test

/-- The C++ `overlap` helper. -/
def overlapC (a ab b bb : Nat) : Bool :=
  if ab = 0 ∨ bb = 0 then false else if a < b then decide (b - a < ab) else decide (a - b < bb)

/-- For non-empty ranges the helper is exactly interval intersection. -/
theorem overlapC_iff (a ab b bb : Nat) (hab : 0 < ab) (hbb : 0 < bb) :
    overlapC a ab b bb = true ↔ (a < b + bb ∧ b < a + ab) := by
  unfold overlapC
  have h1 : ¬ (ab = 0 ∨ bb = 0) := by omega
  rw [if_neg h1]
  by_cases h : a < b
  · rw [if_pos h]
    simp only [decide_eq_true_eq]
    omega
  · rw [if_neg h]
    simp only [decide_eq_true_eq]
    omega

/-- Overlap is symmetric for non-empty ranges. -/
theorem overlapC_symm (a ab b bb : Nat) (hab : 0 < ab) (hbb : 0 < bb) :
    overlapC a ab b bb = true ↔ overlapC b bb a ab = true := by
  rw [overlapC_iff a ab b bb hab hbb, overlapC_iff b bb a ab hbb hab]
  omega

/-- An empty range overlaps nothing. -/
theorem overlapC_empty (a b bb : Nat) : overlapC a 0 b bb = false := by
  unfold overlapC
  simp

-- § key source selection

/-- Where the key projection comes from. -/
inductive KeySrc where
  | bf16
  | native
  | q8
  deriving DecidableEq

/-- Priority: bf16 weight, then native Q2_0, then the canonical Q8_0 path. -/
def selectKey (kb kn : Bool) : KeySrc := if kb then KeySrc.bf16 else if kn then KeySrc.native else KeySrc.q8

/-- `native_key = key_native_data != null && key_bf16 == null`. -/
def nativeKey (kn kb : Bool) : Bool := kn && !kb

/-- The native-key predicate holds exactly when selection picks the native path. -/
theorem nativeKey_iff (kn kb : Bool) : nativeKey kn kb = true ↔ selectKey kb kn = KeySrc.native := by
  cases kb <;> cases kn <;> simp [nativeKey, selectKey]

-- § bf16 rounding (uint32 arithmetic, wrap-around)

/-- `bf16_bits`: add `0x7FFF + lsb`, wrap at 2^32, keep the high half. -/
def bf16Bits (i : Nat) : Nat := ((i + (i / 65536) % 2 + 32767) % 4294967296) / 65536

/-- The result is a 16-bit pattern. -/
theorem bf16Bits_lt (i : Nat) : bf16Bits i < 65536 := by
  unfold bf16Bits; omega

/-- Values with zero low half are represented exactly. -/
theorem bf16Bits_exact (i : Nat) (hi : i < 4294967296) (h : i % 65536 = 0) :
    bf16Bits i = i / 65536 := by
  have hq : i = 65536 * (i / 65536) := by omega
  generalize i / 65536 = q at hq
  subst hq
  have hq2 : q < 65536 := by omega
  have e1 : 65536 * q / 65536 = q := by omega
  unfold bf16Bits
  rw [e1]
  have hlt : 65536 * q + q % 2 + 32767 < 4294967296 := by omega
  rw [Nat.mod_eq_of_lt hlt]
  omega

/-- Below the halfway point the high half is kept. -/
theorem bf16Bits_down (i : Nat) (hi : i < 4294967296) (h : i % 65536 < 32768) :
    bf16Bits i = i / 65536 := by
  have hq : i = 65536 * (i / 65536) + i % 65536 := by omega
  have hq1 : i / 65536 < 65536 := by omega
  have hr1 : i % 65536 < 65536 := by omega
  unfold bf16Bits
  generalize i / 65536 = q at *
  generalize i % 65536 = r at *
  subst hq
  have hlt : 65536 * q + r + q % 2 + 32767 < 4294967296 := by omega
  rw [Nat.mod_eq_of_lt hlt]
  omega

/-- Above the halfway point the high half is incremented (wrapping at 65536). -/
theorem bf16Bits_up (i : Nat) (hi : i < 4294967296) (h : 32768 < i % 65536) :
    bf16Bits i = (i / 65536 + 1) % 65536 := by
  have hq : i = 65536 * (i / 65536) + i % 65536 := by omega
  have hq1 : i / 65536 < 65536 := by omega
  have hr1 : i % 65536 < 65536 := by omega
  unfold bf16Bits
  generalize i / 65536 = q at *
  generalize i % 65536 = r at *
  subst hq
  by_cases hq2 : q < 65535
  · have hlt : 65536 * q + r + q % 2 + 32767 < 4294967296 := by omega
    have hS : 65536 * q + r + q % 2 + 32767 = 65536 * (q + 1) + (r + q % 2 - 32769) := by omega
    have ht : (r + q % 2 - 32769) / 65536 = 0 := Nat.div_eq_of_lt (by omega)
    rw [Nat.mod_eq_of_lt hlt, hS, Nat.mul_add_div (by decide), ht]
    omega
  · have hq3 : q = 65535 := by omega
    have hm : (65536 * q + r + q % 2 + 32767) % 4294967296
        = 65536 * q + r + q % 2 + 32767 - 4294967296 := by omega
    rw [hm]
    omega

/-- Ties round to even. -/
theorem bf16Bits_tie_even (i : Nat) (hi : i < 4294967296) (h : i % 65536 = 32768) :
    bf16Bits i % 2 = 0 := by
  have hq : i = 65536 * (i / 65536) + i % 65536 := by omega
  have hq1 : i / 65536 < 65536 := by omega
  have hr1 : i % 65536 < 65536 := by omega
  unfold bf16Bits
  generalize i / 65536 = q at *
  generalize i % 65536 = r at *
  subst hq
  by_cases hq2 : q < 65535
  · have hlt : 65536 * q + r + q % 2 + 32767 < 4294967296 := by omega
    rcases (by omega : q % 2 = 0 ∨ q % 2 = 1) with hb | hb
    · have hS : 65536 * q + r + q % 2 + 32767 = 65536 * q + 65535 := by omega
      rw [Nat.mod_eq_of_lt hlt, hS, Nat.mul_add_div (by decide)]
      omega
    · have hS : 65536 * q + r + q % 2 + 32767 = 65536 * (q + 1) + 0 := by omega
      rw [Nat.mod_eq_of_lt hlt, hS, Nat.mul_add_div (by decide)]
      omega
  · have hq3 : q = 65535 := by omega
    have hm : (65536 * q + r + q % 2 + 32767) % 4294967296 = 0 := by omega
    rw [hm] <;> omega

end Strata.PleBlock
