/-!
# MODULE: gguf_q2_0_writer
Arithmetic of `tools/gguf_writer.py`: the minimal GGUF v3 writer with the Q2_0 quantizer, its
alignment rules and its self-check (requantization fixed point).

## Source semantics
* Q2_0 block = 64 elements, fp16 scale `d = amax`, 16 bytes of 2-bit codes, LSB first.
* `code = clamp(round(w/d) + 1, 0, 3)`, symbol = `code - 1` in {-1,0,1,2}.  Since `|round(w/d)| <= 1`,
  the +2 level is never produced by this quantizer.
* Tensor data are padded to the alignment (default 32); offsets in the directory are aligned.
* Requantizing dequantized values is exact (fixed point) at the code level.

## Form template identifiers
* q2Code       <- `np.clip(np.rint(b / d) + 1, 0, 3)`
* alignUp32    <- `(offset + alignment - 1) // alignment * alignment`
* pad32        <- `(-len(body)) % alignment`
-/

namespace Strata.GgufWriter

-- § Alignment

/-- Round up to the default 32-byte alignment. -/
def alignUp32 (n : Nat) : Nat := (n + 31) / 32 * 32

/-- Padding needed after `n` bytes, i.e. `(-n) % 32`. -/
def pad32 (n : Nat) : Nat := (32 - n % 32) % 32

/-- Alignment never shrinks. -/
theorem alignUp32_ge (n : Nat) : n ≤ alignUp32 n := by
  unfold alignUp32
  omega

/-- Result is a multiple of 32. -/
theorem alignUp32_mod (n : Nat) : alignUp32 n % 32 = 0 := by
  unfold alignUp32
  omega

/-- Alignment adds less than one alignment unit. -/
theorem alignUp32_lt (n : Nat) : alignUp32 n < n + 32 := by
  unfold alignUp32
  omega

/-- Aligning equals adding the padding written to the file. -/
theorem alignUp32_eq_add_pad (n : Nat) : alignUp32 n = n + pad32 n := by
  unfold alignUp32 pad32
  omega

/-- After padding, the position is a multiple of 32. -/
theorem pad32_aligns (n : Nat) : (n + pad32 n) % 32 = 0 := by
  unfold pad32
  omega

/-- If a tensor starts on an aligned offset, the next directory offset is aligned too. -/
theorem next_offset_aligned (off len : Nat) (_h : off % 32 = 0) :
    alignUp32 (off + len) % 32 = 0 :=
  alignUp32_mod _

/-- Fixed GGUF header: magic(4) + version(4) + tensor count(8) + kv count(8). -/
theorem header_fixed_bytes : 4 + 4 + 8 + 8 = 24 := by decide

-- § Q2_0 quantizer

/-- Quantizer code for an already rounded value `r = rint(w/d)`. -/
def q2Code (r : Int) : Nat := (max 0 (min 3 (r + 1))).toNat

/-- Codes are 2-bit. -/
theorem q2Code_lt (r : Int) : q2Code r < 4 := by
  unfold q2Code
  omega

/-- The decoded symbol is `r` clamped to [-1, 2]. -/
theorem q2Code_symbol (r : Int) : (q2Code r : Int) - 1 = max (-1) (min 2 r) := by
  unfold q2Code
  omega

/-- Inside the grid the symbol is exactly `r`. -/
theorem q2Code_fixed (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 2) : (q2Code r : Int) - 1 = r := by
  unfold q2Code
  omega

/-- `d = amax` means `|r| <= 1`, so the code 3 (symbol +2) is never emitted. -/
theorem q2Code_no_plus2 (r : Int) (h : r ≤ 1) : q2Code r ≤ 2 := by
  unfold q2Code
  omega

/-- Requantizing a dequantized symbol returns the same code (code-level fixed point). -/
theorem q2Code_roundtrip (c : Nat) (h : c < 4) : q2Code ((c : Int) - 1) = c := by
  unfold q2Code
  omega

/-- Packing layout: element j sits in byte j/4 at shift 2*(j%4), 4 codes per byte. -/
theorem pack_layout (j : Nat) (h : j < 64) :
    j / 4 < 16 ∧ (j % 4) * 2 < 8 ∧ (j % 4) * 2 % 2 = 0 := by
  omega

/-- Block bytes: scale plus 64/4 code bytes. -/
theorem block_bytes : 2 + 64 / 4 = 18 := by decide

/-- The self-check tensor (16 rows of 64) occupies 16 blocks = 288 bytes. -/
theorem selftest_bytes : 16 * 64 / 64 * 18 = 288 := by decide

end Strata.GgufWriter
