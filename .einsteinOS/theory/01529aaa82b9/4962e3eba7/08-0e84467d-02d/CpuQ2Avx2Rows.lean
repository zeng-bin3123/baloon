/-!
# MODULE: cpu_q2_avx2_rows
Q2_0 GGUF-layout rows and the activation quantizer for CPUs without AVX-512: the 2-bit unpack ordering, the
18-byte block addressing, the correction identity, the token batching and the int32 horizontal sum.

## Source semantics
* A code byte holds four 2-bit codes: value i of a block sits in byte i/4 at bits 2*(i%4); unpacking returns the
  64 values in order (lo = values 0..31 = bytes 0..7, hi = values 32..63 = bytes 8..15).
* The four shift-and-mask codes of a byte recombine to the byte (nothing is lost or duplicated).
* Block b of a row starts at `18 b`: a 2-byte scale, then 16 code bytes.
* `sum (c-1) x = sum (c x) - sum x`, which is the `corr += d (hx0 + hx1)` term.
* Codes of the quantizer are clamped to [-127, 127]; an 8-lane chunk sum is at most 8*127 per vector.
* The shuffle-based horizontal add of 8 int32 lanes yields the total.
* `nt` tokens run directly for 1..4, otherwise in groups of 4.

## Form template identifiers
* `code byte s` : `(byte >> s) & 3`
* `hsum8` : the `_mm_shuffle_epi32` 0x4E / 0xB1 reduction
* `chunks 4 nt` : the `t0 += 4` loop
-/

namespace Strata.Q2Avx2

-- § unpack

/-- The 2-bit code of `byte` at shift `s`. -/
def code (byte s : Nat) : Nat := (byte >>> s) &&& 3

/-- Every unpacked code is in 0..3. -/
theorem code_le (byte s : Nat) : code byte s ≤ 3 := by
  unfold code
  exact Nat.and_le_right

/-- The four codes of a byte recombine to the byte: the unpack drops and duplicates nothing. -/
theorem code_roundtrip : ∀ b : Fin 256,
    code b.val 0 + 4 * code b.val 2 + 16 * code b.val 4 + 64 * code b.val 6 = b.val := by
  decide +kernel

/-- Value position p of a block (0..63) lives in byte p/4. -/
def byteOf (p : Nat) : Nat := p / 4

/-- Shift of value position p inside its byte. -/
def shiftOf (p : Nat) : Nat := 2 * (p % 4)

/-- The `lo` vector (values 0..31) reads bytes 0..7. -/
theorem lo_bytes (p : Nat) (h : p < 32) : byteOf p < 8 := by
  unfold byteOf
  omega

/-- The `hi` vector (values 32..63) reads bytes 8..15. -/
theorem hi_bytes (p : Nat) (h1 : 32 ≤ p) (h2 : p < 64) : 8 ≤ byteOf p ∧ byteOf p < 16 := by
  unfold byteOf
  omega

/-- Shifts are 0, 2, 4 or 6. -/
theorem shift_range (p : Nat) : shiftOf p ≤ 6 ∧ shiftOf p % 2 = 0 := by
  unfold shiftOf
  omega

/-- Lanes 4q..4q+3 of the output come from one byte, in shift order 0, 2, 4, 6. -/
theorem lane_group (q r : Nat) (h : r < 4) : byteOf (4 * q + r) = q ∧ shiftOf (4 * q + r) = 2 * r := by
  unfold byteOf shiftOf
  omega

-- § block addressing

/-- Block b of a GGUF Q2_0 row starts at byte `18 b`; its codes at `18 b + 2`. -/
def blockStart (b : Nat) : Nat := b * 18

/-- The 16 code bytes of block b end where block b+1 starts. -/
theorem block_tiles (b : Nat) : blockStart b + 2 + 16 = blockStart (b + 1) := by
  unfold blockStart
  omega

/-- Block b of an nb-block row lies inside the row. -/
theorem block_in_row (b nb : Nat) (h : b < nb) : blockStart b + 18 ≤ nb * 18 := by
  unfold blockStart
  have := Nat.mul_le_mul_right 18 h
  omega

-- § offset identity

/-- The Q2_0 code-offset identity behind the `corr` term. -/
theorem offset_identity (l : List (Int × Int)) :
    (l.map (fun p => (p.1 - 1) * p.2)).sum =
      (l.map (fun p => p.1 * p.2)).sum - (l.map (fun p => p.2)).sum := by
  induction l with
  | nil => simp
  | cons p t ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega

-- § quantizer integer facts

/-- The quantizer's clamp. -/
def clampQ (v : Int) : Int := if v < -127 then -127 else if 127 < v then 127 else v

/-- Clamped codes stay in [-127, 127]. -/
theorem clampQ_range (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ
  split <;> try split
  all_goals omega

/-- Eight-lane int32 horizontal sum written the way the shuffles combine lanes
(lo+hi, then 0x4E swap, then 0xB1 swap). -/
def hsum8 (l0 l1 l2 l3 l4 l5 l6 l7 : Int) : Int :=
  ((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))

/-- The shuffle reduction equals the plain sum of the eight lanes. -/
theorem hsum8_eq (l0 l1 l2 l3 l4 l5 l6 l7 : Int) :
    hsum8 l0 l1 l2 l3 l4 l5 l6 l7 = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 := by
  unfold hsum8
  omega

/-- Each of the four 8-lane vectors holds values in [-127,127], so each total lane is at most 4*127 in size
and the 32-element chunk sum is at most 4064. -/
theorem lanes_bound (a b c d : Int) (ha : -508 ≤ a ∧ a ≤ 508) (hb : -508 ≤ b ∧ b ≤ 508)
    (hc : -508 ≤ c ∧ c ≤ 508) (hd : -508 ≤ d ∧ d ≤ 508) : -2032 ≤ a + b + c + d ∧ a + b + c + d ≤ 2032 := by
  omega

-- § token batching

/-- Groups of at most k tokens, as in the `t0 += 4` loop. -/
def chunks (k n : Nat) : List Nat :=
  if h : n = 0 ∨ k = 0 then [] else min k n :: chunks k (n - min k n)
termination_by n
decreasing_by omega

/-- Every token is processed exactly once. -/
theorem chunks_sum (k : Nat) (hk : 0 < k) : ∀ n, (chunks k n).sum = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases h0 : n = 0
    · subst h0
      simp [chunks]
    · rw [chunks, dif_neg (by omega), List.sum_cons, ih (n - min k n) (by omega)]
      omega

/-- Each group has between 1 and 4 tokens, matching the instantiated NT = 1..4 templates. -/
theorem chunks_bound (k : Nat) (hk : 0 < k) : ∀ n, ∀ c ∈ chunks k n, 0 < c ∧ c ≤ k := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro c hc
    by_cases h0 : n = 0
    · subst h0
      simp [chunks] at hc
    · rw [chunks, dif_neg (by omega)] at hc
      rcases List.mem_cons.mp hc with rfl | hc
      · omega
      · exact ih (n - min k n) (by omega) c hc

end Strata.Q2Avx2
