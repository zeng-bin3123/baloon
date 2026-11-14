/-!
# PerLayerEmbeddingBlock — PLE GPU block geometry
Formalises the row-fastest history layout, dilated conv tap addresses,
per-stream gate sign/magnitude, value broadcast, scratch-buffer carving,
bf16 round-to-nearest-even, and the overlap predicate.

Sources: src/kernels/cuda/ple.cu, include/baloon/kern_ple.hpp
-/

namespace Baloon.PleBlock

-- ── history buffer (row-fastest) ─────────────────────────────────────────────

/-- History depth: 9 rows per channel. -/
def histRows : Nat := 9

/-- Linear index into the row-fastest `histRows × hcDim` history buffer. -/
def histIdx (row chan : Nat) : Nat := row + histRows * chan

/-- Every (row, channel) pair is in-bounds. -/
theorem histIdx_inBuf {row chan hcDim : Nat} (hr : row < histRows) (hc : chan < hcDim) :
    histIdx row chan < histRows * hcDim := by
  unfold histIdx histRows at *; omega

/-- `histIdx` is injective in `(row, channel)`. -/
theorem histIdx_inj {r c r' c' : Nat} (hr : r < histRows) (hr' : r' < histRows)
    (h : histIdx r c = histIdx r' c') : r = r' ∧ c = c' := by
  unfold histIdx histRows at *; omega

/-- Advance a channel column: drop oldest row, append new value. -/
def histAdvance (col : List Int) (x : Int) : List Int := col.tail ++ [x]

/-- After an advance the column still has `histRows` entries, with `x` last. -/
theorem histAdvance_spec (col : List Int) (x : Int) (hl : col.length = 9) :
    (histAdvance col x).length = 9 ∧ (histAdvance col x).getLast? = some x := by
  cases col with
  | nil => simp at hl
  | cons a t => simp_all [histAdvance]

-- ── dilated conv tap rows ─────────────────────────────────────────────────────

/-- Row index read by tap `k` of a causal dilated conv. -/
def tapIdx (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- At the real geometry (nhist=9, kern=4, dil=3) the taps read rows 0,3,6,9. -/
theorem tapIdx_real : (List.range 4).map (tapIdx 9 4 3) = [0, 3, 6, 9] := by decide

/-- Only the last tap (k=3) reads the new normalised row (row 9 = nhist). -/
theorem tapIdx_lastOnly : ∀ k < 4, (tapIdx 9 4 3 k = 9 ↔ k = 3) := by decide

-- ── per-stream gate ───────────────────────────────────────────────────────────

/-- Sign function: −1 for negatives, 0 for zero, 1 for positives. -/
def signOf (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- `signOf` is odd: `signOf (-s) = -signOf s`. -/
theorem signOf_neg (s : Int) : signOf (-s) = -signOf s := by
  unfold signOf
  by_cases h1 : s > 0
  · simp [show ¬ (-s > 0) from by omega, show -s < 0 from by omega, h1]
  · by_cases h3 : s < 0
    · simp [show -s > 0 from by omega, h1, h3, Int.neg_neg]
    · have : s = 0 := by omega; subst this; decide

/-- The magnitude clamp `max(|s|, floor)` is never below the floor. -/
theorem magnitudeClamp (s : Int) (floor : Nat) : floor ≤ max s.natAbs floor := by omega

/-- Gate's signed argument is odd in `s` (sign × magnitude). -/
theorem gateArg_odd (mag : Int → Int) (s : Int) :
    signOf (-s) * mag ((-s).natAbs) = -(signOf s * mag s.natAbs) := by
  rw [signOf_neg, Int.natAbs_neg, Int.neg_mul]

-- ── broadcast ────────────────────────────────────────────────────────────────

/-- Broadcast: element `i` = `value[i % ne] * gate[i / ne]`. -/
def broadcastElem (value gate : Nat → Int) (ne i : Nat) : Int :=
  value (i % ne) * gate (i / ne)

/-- Element `(c, d)` with `d < ne` equals `value d * gate c`. -/
theorem broadcastElem_at (value gate : Nat → Int) {ne c d : Nat} (hd : d < ne) :
    broadcastElem value gate ne (c * ne + d) = value d * gate c := by
  unfold broadcastElem
  have hne : 0 < ne := by omega
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd,
      Nat.add_mul_div_right _ _ hne, Nat.div_eq_of_lt hd, Nat.zero_add]

/-- Three-way add is associative: `h + (g + c) = h + g + c`. -/
theorem add3_assoc (h g c : Int) : h + (g + c) = h + g + c := by omega

-- ── scratch-buffer layout ─────────────────────────────────────────────────────

/-- Align `x` up to the next multiple of 16. -/
def alignUp16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Float workspace: five `hc_dim` buffers + `n_embd` + `hc` floats → bytes. -/
def floatWsBytes : Nat := (5 * 10240 + 2560 + 4) * 4

/-- Q8_0 activation image: `n_embd / 32 * 34` bytes. -/
def q8ImgBytes : Nat := 2560 / 32 * 34

/-- BF16 embedding copy: `n_embd * 2` bytes. -/
def bf16CopyBytes : Nat := 2560 * 2

/-- Offset of the Q8 image region. -/
def q8Off : Nat := alignUp16 floatWsBytes

/-- Offset of the BF16 copy region. -/
def bf16Off : Nat := q8Off + alignUp16 q8ImgBytes

/-- Total scratch size reported by `ple_block_scratch_bytes`. -/
def scratchTotal : Nat := alignUp16 floatWsBytes + alignUp16 q8ImgBytes + bf16CopyBytes + 256

/-- The three regions fit, are ordered, and the two dynamic regions are 16-aligned. -/
theorem scratchLayout :
    floatWsBytes ≤ q8Off ∧ q8Off + q8ImgBytes ≤ bf16Off ∧
    bf16Off + bf16CopyBytes ≤ scratchTotal ∧ q8Off % 16 = 0 ∧ bf16Off % 16 = 0 := by decide

/-- The five float sub-buffers plus value and gate are back-to-back and non-overlapping. -/
theorem floatSubbufs :
    let off := [0, 10240, 20480, 30720, 40960, 51200, 53760]
    let len := [10240, 10240, 10240, 10240, 10240, 2560, 4]
    (List.zipWith (· + ·) off len).dropLast = off.tail ∧
      off.getLast! + len.getLast! = 5 * 10240 + 2560 + 4 := by decide

-- ── bf16 round-to-nearest-even ────────────────────────────────────────────────

/-- Upper 16 bits after bf16 round-to-nearest-even (safe below NaN range). -/
def bf16RoundHi (i : Nat) : Nat := (i + (i / 2 ^ 16) % 2 + 0x7FFF) / 2 ^ 16

/-- Values already bf16-aligned round to themselves. -/
theorem bf16RoundHi_exact {i : Nat} (h : i % 2 ^ 16 = 0) : bf16RoundHi i * 2 ^ 16 = i := by
  unfold bf16RoundHi; have : (i / 2 ^ 16) % 2 ≤ 1 := by omega; omega

/-- Rounding error is bounded by half a bf16 ulp. -/
theorem bf16RoundHi_bound (i : Nat) :
    bf16RoundHi i * 2 ^ 16 ≤ i + 2 ^ 15 ∧ i ≤ bf16RoundHi i * 2 ^ 16 + 2 ^ 15 := by
  unfold bf16RoundHi; have : (i / 2 ^ 16) % 2 ≤ 1 := by omega; omega

-- ── overlap predicate ────────────────────────────────────────────────────────

/-- `spanOverlap a an b bn`: two spans overlap (null or empty spans never do). -/
def spanOverlap (a an b bn : Nat) : Bool :=
  if a = 0 || b = 0 || an = 0 || bn = 0 then false
  else if a < b then decide (b - a < an) else decide (a - b < bn)

/-- An empty span overlaps nothing. -/
theorem spanOverlap_empty (a b bn : Nat) : spanOverlap a 0 b bn = false := by
  simp [spanOverlap]

/-- Adjacent spans do not overlap. -/
theorem spanOverlap_adjacent {a an bn : Nat} : spanOverlap a an (a + an) bn = false := by
  unfold spanOverlap
  by_cases han : an = 0
  · simp [han]
  · have hlt : a < a + an := by omega
    split
    · rfl
    · simp

end Baloon.PleBlock
