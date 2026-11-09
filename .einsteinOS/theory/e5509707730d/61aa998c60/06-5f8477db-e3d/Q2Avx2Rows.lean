/-!
# MODULE: q2_avx2_rows
SOURCES: src/kernels/cpu/q2_avx2.cpp
The Q2_0 expert rows and the Q8_1 activation quantizer for CPUs without AVX-512 (AVX2 only): the 64-code
unpack order, the 32-value chunk/scale indexing, the saturation-free `maddubs` bounds, the horizontal
integer sum, and the token chunking loop.

## Source semantics
* A 18-byte block is a 2-byte scale followed by 16 code bytes; value `i` of the block is in byte `i/4` at
  bit offset `2*(i%4)`; the unpack puts it at output position `4*byte + j`.
* `lo` holds values 0..31 (bytes 0..7) and `hi` values 32..63 (bytes 8..15); each is paired with the
  activation chunk `2b` or `2b+1`.
* Codes are at most 3 and activations are within ±127, so no int16 lane saturates and the int32 total of a
  block is tiny.
* The integer horizontal sum of 8 lanes equals the plain sum of the lanes.
* The quantizer writes `a.q[k*32 + j]` for `k < n/32`, `j < 32`: a bijection onto `[0, n)` when `32 | n`.

## Form template identifiers
* `valuePos`       : unpack_position
* `halfOf`         : activation_chunk_of_value
* `hsum8`          : integer_horizontal_sum
* `chunks4`        : four_token_chunking
* `quantIndex`     : quantizer_output_index
-/

namespace Strata.Q2Avx2

-- § unpack order

/-- Output position of code `j` (0..3) of byte `k` (0..15) after the interleaving unpacks. -/
def valuePos (k j : Nat) : Nat := 4 * k + j

/-- The position map is a bijection of `[0,16) x [0,4)` onto `[0,64)`. -/
theorem valuePos_roundtrip (k j : Nat) (hk : k < 16) (hj : j < 4) :
    valuePos k j < 64 ∧ valuePos k j / 4 = k ∧ valuePos k j % 4 = j := by
  unfold valuePos
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Distinct (byte, code) pairs land on distinct positions. -/
theorem valuePos_inj (k j k' j' : Nat) (hj : j < 4) (hj' : j' < 4) (h : valuePos k j = valuePos k' j') :
    k = k' ∧ j = j' := by
  unfold valuePos at h
  refine ⟨?_, ?_⟩ <;> omega

/-- Values `0..31` come from bytes `0..7` (the `lo` vector). -/
theorem lo_bytes (v : Nat) (hv : v < 32) : v / 4 < 8 := by
  omega

/-- Values `32..63` come from bytes `8..15` (the `hi` vector). -/
theorem hi_bytes (v : Nat) (h1 : 32 ≤ v) (h2 : v < 64) : 8 ≤ v / 4 ∧ v / 4 < 16 := by
  refine ⟨?_, ?_⟩ <;> omega

/-- Activation chunk of value `v` of block `b`: `lo` pairs with chunk `2b`, `hi` with `2b+1`. -/
def halfOf (b v : Nat) : Nat := (b * 64 + v) / 32

/-- The chunk index is `2*b + v/32`. -/
theorem halfOf_eq (b v : Nat) (hv : v < 64) : halfOf b v = 2 * b + v / 32 := by
  unfold halfOf
  omega

/-- A 2-bit code `(x >> 2j) & 3` is below 4. -/
theorem code_lt (x : Nat) : x % 4 < 4 :=
  Nat.mod_lt _ (by decide)

-- § saturation-free arithmetic

/-- A code (0..3) times an int8 activation is within ±381. -/
theorem code_mul_bound (c q : Int) (h0 : 0 ≤ c) (h3 : c ≤ 3) (hq1 : -127 ≤ q) (hq2 : q ≤ 127) :
    -381 ≤ c * q ∧ c * q ≤ 381 := by
  interval_cases c <;> omega

/-- One `maddubs` int16 lane (two products) fits easily in int16. -/
theorem maddubs_lane_fits (c1 c2 q1 q2 : Int) (h1 : 0 ≤ c1) (h2 : c1 ≤ 3) (h3 : 0 ≤ c2) (h4 : c2 ≤ 3)
    (hq1 : -127 ≤ q1) (hq2 : q1 ≤ 127) (hq3 : -127 ≤ q2) (hq4 : q2 ≤ 127) :
    -762 ≤ c1 * q1 + c2 * q2 ∧ c1 * q1 + c2 * q2 ≤ 762 := by
  have a := code_mul_bound c1 q1 h1 h2 hq1 hq2
  have b := code_mul_bound c2 q2 h3 h4 hq3 hq4
  exact ⟨by omega, by omega⟩

/-- A `madd` against ones adds two int16 lanes: still within ±1524. -/
theorem madd_pair_fits (x y : Int) (hx1 : -762 ≤ x) (hx2 : x ≤ 762) (hy1 : -762 ≤ y) (hy2 : y ≤ 762) :
    -1524 ≤ x + y ∧ x + y ≤ 1524 := by
  exact ⟨by omega, by omega⟩

/-- The 64 products of a block are bounded by 64*381, inside int32. -/
theorem block_total_fits (n : Nat) (hn : n ≤ 64) (s : Int) (h1 : -(381 * (n : Int)) ≤ s)
    (h2 : s ≤ 381 * (n : Int)) : -24384 ≤ s ∧ s ≤ 24384 := by
  have : (n : Int) ≤ 64 := by exact_mod_cast hn
  exact ⟨by omega, by omega⟩

-- § horizontal sums

/-- Horizontal sum of 8 integer lanes by the 256 -> 128 -> 64 -> 32 reduction used by the source. -/
def hsum8 (l0 l1 l2 l3 l4 l5 l6 l7 : Int) : Int :=
  ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))

/-- The staged reduction equals the plain sum of the lanes. -/
theorem hsum8_eq (l0 l1 l2 l3 l4 l5 l6 l7 : Int) :
    hsum8 l0 l1 l2 l3 l4 l5 l6 l7 = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 := by
  unfold hsum8
  omega

-- § quantizer indexing

/-- Output index of value `j` of chunk `k`. -/
def quantIndex (k j : Nat) : Nat := k * 32 + j

/-- With `32 | n` the chunk count times 32 is `n`. -/
theorem nchunks_exact (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by
  omega

/-- Each (chunk, value) pair is an in-range index. -/
theorem quantIndex_lt (n k j : Nat) (h : n % 32 = 0) (hk : k < n / 32) (hj : j < 32) :
    quantIndex k j < n := by
  unfold quantIndex
  omega

/-- The index map is injective: no two writes collide. -/
theorem quantIndex_inj (k j k' j' : Nat) (hj : j < 32) (hj' : j' < 32)
    (h : quantIndex k j = quantIndex k' j') : k = k' ∧ j = j' := by
  unfold quantIndex at h
  refine ⟨?_, ?_⟩ <;> omega

/-- The `qi` staging array holds 4 vectors of 8 lanes, exactly one 32-value chunk. -/
theorem staging_size : 4 * 8 = 32 := by
  decide

/-- Vector `i < 4`, lane `l < 8` of the staging array is index `8i + l < 32`. -/
theorem staging_index_lt (i l : Nat) (hi : i < 4) (hl : l < 8) : 8 * i + l < 32 := by
  omega

-- § four-token chunking

/-- Sizes of the chunks of at most 4 tokens a loop over `rem` tokens produces, with fuel `f`. -/
def chunks4 : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, rem => if rem = 0 then [] else min 4 rem :: chunks4 f (rem - min 4 rem)

/-- The chunks cover all tokens when the fuel suffices. -/
theorem chunks4_sum (f : Nat) : ∀ rem, rem ≤ 4 * f → (chunks4 f rem).sum = rem := by
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    rfl
  | succ f ih =>
    intro rem h
    by_cases hr : rem = 0
    · subst hr
      simp [chunks4]
    · have e : chunks4 (f + 1) rem = min 4 rem :: chunks4 f (rem - min 4 rem) := by
        simp [chunks4, hr]
      rw [e, List.sum_cons, ih (rem - min 4 rem) (by omega)]
      omega

/-- Every chunk size is in `1..4`. -/
theorem chunks4_range (f : Nat) : ∀ rem, ∀ c ∈ chunks4 f rem, 1 ≤ c ∧ c ≤ 4 := by
  induction f with
  | zero => intro rem c hc; simp [chunks4] at hc
  | succ f ih =>
    intro rem c hc
    by_cases hr : rem = 0
    · subst hr
      simp [chunks4] at hc
    · have e : chunks4 (f + 1) rem = min 4 rem :: chunks4 f (rem - min 4 rem) := by
        simp [chunks4, hr]
      rw [e] at hc
      rcases List.mem_cons.mp hc with h | h
      · rw [h]; omega
      · exact ih _ c h

end Strata.Q2Avx2
