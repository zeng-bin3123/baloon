/-!
# MODULE: q2_avx2_fallback
AVX2-only Q2_0 row dot (GGUF 18-byte blocks) and int8 activation quantiser for CPUs without AVX-512.

## Source semantics
* GGUF block = fp16 scale (2 bytes) then 16 code bytes: 18 bytes per 64 weights; row stride is `18 * nblocks`.
* Value `v` of a block lives in code byte `v / 4` at bit position `2 * (v % 4)`; the unpack yields values
  0..31 in the low vector and 32..63 in the high vector.
* The dot is `d * sum(c * x)` per 32-chunk times chunk scale, minus `d * hx` (the -1 code offset).
* The quantiser takes `amax = max |x|`, uses scale `amax/127` (0 when `amax = 0`), clamps to `[-127,127]`;
  `sum`/`hx` are per-chunk sums of the quantised values.
* More than four tokens are handled in groups of at most four.

## Form template identifiers
* `blockOff` ~ `row + b*18`, `codeByte` / `codeShift` ~ unpack addressing
* `amaxL` ~ the vector max of `|x|`, `clamp127` ~ clamp, `groupSize` ~ `k = min(4, nt - t0)`
-/

namespace Strata.Q2Avx2

-- § GGUF block addressing

/-- Byte offset of block `b` within a row. -/
def blockOff (b : Nat) : Nat := b * 18

/-- Offset of the code bytes of block `b` (after the 2-byte scale). -/
def codeOff (b : Nat) : Nat := blockOff b + 2

/-- Consecutive blocks tile the row: block `b+1` starts where block `b` ends. -/
theorem block_tiling (b : Nat) : blockOff (b + 1) = blockOff b + 18 := by
  unfold blockOff
  omega

/-- All 16 code bytes of block `b` lie inside the block. -/
theorem code_bytes_in_block (b k : Nat) (h : k < 16) : codeOff b + k < blockOff (b + 1) := by
  unfold codeOff blockOff
  omega

/-- The scale header of block `b` is disjoint from its code bytes. -/
theorem scale_before_codes (b : Nat) : blockOff b + 2 = codeOff b := by
  rfl

-- § 2-bit unpack addressing

/-- Code byte holding value `v`. -/
def codeByte (v : Nat) : Nat := v / 4

/-- Bit shift of value `v` inside its byte. -/
def codeShift (v : Nat) : Nat := 2 * (v % 4)

/-- Every value of a 64-value block lives in one of the 16 code bytes. -/
theorem codeByte_lt (v : Nat) (h : v < 64) : codeByte v < 16 := by
  unfold codeByte
  omega

/-- Shifts are one of 0, 2, 4, 6. -/
theorem codeShift_cases (v : Nat) :
    codeShift v = 0 ∨ codeShift v = 2 ∨ codeShift v = 4 ∨ codeShift v = 6 := by
  unfold codeShift
  omega

/-- The addressing is injective: distinct values never share a (byte, shift) slot. -/
theorem code_slot_injective (v w : Nat) (hb : codeByte v = codeByte w) (hs : codeShift v = codeShift w) :
    v = w := by
  unfold codeByte codeShift at *
  omega

/-- The low vector holds values 0..31 and the high vector 32..63: the split point is byte 8. -/
theorem lo_hi_split (v : Nat) (h : v < 64) : v < 32 ↔ codeByte v < 8 := by
  unfold codeByte
  omega

-- § offset-corrected dot over a block

/-- `sum c * x` over paired lists. -/
def dotList : List Nat → List Int → Int
  | c :: cs, x :: xs => (c : Int) * x + dotList cs xs
  | _, _ => 0

/-- `sum x`. -/
def sumList : List Int → Int
  | [] => 0
  | x :: xs => x + sumList xs

/-- `sum (c-1) * x`. -/
def offDot : List Nat → List Int → Int
  | c :: cs, x :: xs => ((c : Int) - 1) * x + offDot cs xs
  | _, _ => 0

/-- The row kernel's `acc - corr` equals the mathematical offset dot. -/
theorem offDot_eq (cs : List Nat) (xs : List Int) (h : cs.length = xs.length) :
    offDot cs xs = dotList cs xs - sumList xs := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [offDot, dotList, sumList]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have ih' := ih xs h'
      simp only [offDot, dotList, sumList]
      rw [Int.sub_mul, Int.one_mul]
      omega

-- § activation quantiser

/-- Max absolute value of a chunk. -/
def amaxL : List Int → Nat
  | [] => 0
  | x :: xs => max x.natAbs (amaxL xs)

/-- Every element is bounded by the chunk maximum. -/
theorem le_amaxL (x : Int) (xs : List Int) (h : x ∈ xs) : x.natAbs ≤ amaxL xs := by
  induction xs with
  | nil => simp at h
  | cons y ys ih =>
    simp only [List.mem_cons] at h
    show x.natAbs ≤ max y.natAbs (amaxL ys)
    rcases h with h | h
    · subst h
      exact Nat.le_max_left _ _
    · exact Nat.le_trans (ih h) (Nat.le_max_right _ _)

/-- A zero maximum means the chunk is all zero (the quantiser then uses scale 0). -/
theorem amaxL_zero (x : Int) (xs : List Int) (h : x ∈ xs) (h0 : amaxL xs = 0) : x = 0 := by
  have := le_amaxL x xs h
  omega

/-- Clamp to the symmetric int8 range. -/
def clamp127 (v : Int) : Int := min 127 (max (-127) v)

/-- The clamp lands in `[-127,127]`. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- Chunk count of `n` values. -/
def nChunks (n : Nat) : Nat := n / 32

/-- `n = 32k` values give exactly `k` chunks, each addressing `[32k', 32k'+32)` inside the buffer. -/
theorem chunk_in_range (k n i j : Nat) (hn : n = 32 * k) (hi : i < nChunks n) (hj : j < 32) :
    32 * i + j < n := by
  unfold nChunks at hi
  omega

-- § token grouping

/-- Group size taken at offset `t0` of `nt` tokens. -/
def groupSize (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- Groups are non-empty while tokens remain. -/
theorem groupSize_pos (nt t0 : Nat) (h : t0 < nt) : 1 ≤ groupSize nt t0 := by
  unfold groupSize
  split <;> omega

/-- Groups are at most four tokens (the largest instantiated kernel). -/
theorem groupSize_le (nt t0 : Nat) : groupSize nt t0 ≤ 4 := by
  unfold groupSize
  split <;> omega

/-- Groups never pass the last token. -/
theorem groupSize_fits (nt t0 : Nat) (h : t0 < nt) : t0 + groupSize nt t0 ≤ nt := by
  unfold groupSize
  split <;> omega

end Strata.Q2Avx2
