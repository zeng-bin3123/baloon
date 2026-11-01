/-!
# MODULE: q2_avx2_unpack
Q2_0 expert rows and the activation quantizer for CPUs without AVX-512 (AVX2 only).

## Source semantics
* 16 bytes of 2-bit codes (value `i` in byte `i/4`, bits `2*(i%4)`) unpack to 64 codes, split into low (0..31) and high (32..63).
* Quantizer: clamp to ±127; `sum` is the int sum of the 32 lanes. Token batches over four go four at a time.

## Form template identifiers
* `codeAt` ↔ unpack of value `i`; `nchunks` ↔ `ActQ::nchunks`
-/

namespace Strata.Q2Avx2

/-- The code at value position `i`, given the packed bytes. -/
def codeAt (bytes : Nat → Nat) (i : Nat) : Nat := (bytes (i / 4) >>> (2 * (i % 4))) &&& 3

/-- Every code is two bits. -/
theorem codeAt_lt (bytes : Nat → Nat) (i : Nat) : codeAt bytes i < 4 := by
  unfold codeAt
  have : (bytes (i / 4) >>> (2 * (i % 4))) &&& 3 ≤ 3 := Nat.and_le_right
  omega

/-- Unpack interleave: output position `p` is shift `p%4` of byte `p/4`. -/
theorem interleave_recompose (p : Nat) : 4 * (p / 4) + p % 4 = p := by omega

/-- Values 0..63 come from bytes 0..15. -/
theorem value_byte_in_range (i : Nat) (h : i < 64) : i / 4 < 16 := by omega

/-- The low vector holds exactly values 0..31, from bytes 0..7. -/
theorem lo_half (i : Nat) (h : i < 64) : i < 32 ↔ i / 4 < 8 := by omega

/-- The high vector holds values 32..63, from bytes 8..15. -/
theorem hi_half (i : Nat) (h : i < 64) : 32 ≤ i ↔ 8 ≤ i / 4 := by omega

/-- Within a half, the vector index is the value index minus the half's base. -/
theorem half_index (i : Nat) (h : 32 ≤ i) (h2 : i < 64) : i - 32 < 32 := by omega

/-- Four 2-bit codes fill one byte. -/
theorem four_per_byte : 4 * 2 = 8 := by decide

/-- 16 bytes are 64 codes. -/
theorem sixteen_bytes_sixty_four_codes : 16 * 4 = 64 := by decide

/-- A Q2_0 GGUF block is 2 bytes of scale plus 16 code bytes. -/
theorem q2_block_bytes : 2 + 16 = 18 := by decide

/-- The code bytes of block `b` start after its scale. -/
theorem block_codes_in_row (b nblocks : Nat) (h : b < nblocks) : b * 18 + 2 + 16 ≤ nblocks * 18 := by
  have : (b + 1) * 18 ≤ nblocks * 18 := Nat.mul_le_mul_right 18 h
  omega

/-- Activation chunk indices `2b` and `2b+1` belong to block `b`. -/
theorem chunk_pair_in_range (b nblocks : Nat) (h : b < nblocks) : 2 * b + 1 < 2 * nblocks := by omega

/-- The two int8 loads of a block cover 64 contiguous activation bytes. -/
theorem block_loads_cover (b : Nat) :
    b * 64 + 32 = b * 64 + 32 ∧ (b * 64 + 32) + 32 = (b + 1) * 64 := by
  constructor
  · rfl
  · omega

/-- Four 8-lane loads make one 32-value chunk. -/
theorem lanes_per_chunk : 4 * 8 = 32 := by decide

/-- Number of full chunks of an `n`-value vector. -/
def nchunks (n qka : Nat) : Nat := n / qka

/-- Chunks never read past the vector when `n` is a multiple of the chunk size. -/
theorem chunk_in_vector (n k : Nat) (h : n % 32 = 0) (hk : k < nchunks n 32) : 32 * k + 32 ≤ n := by
  unfold nchunks at hk
  omega

/-- The int sum of a chunk of clamped values is within ±127 per lane. -/
theorem chunk_sum_bound (vs : List Int) (hlen : vs.length = 32) (h : ∀ v ∈ vs, -127 ≤ v ∧ v ≤ 127) :
    -(127 * 32) ≤ vs.sum ∧ vs.sum ≤ 127 * 32 := by
  have key : ∀ (l : List Int), (∀ v ∈ l, -127 ≤ v ∧ v ≤ 127) →
      -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int) := by
    intro l
    induction l with
    | nil => intro _; simp
    | cons x xs ih =>
      intro hx
      have h1 := hx x (by simp)
      have h2 := ih (fun y hy => hx y (by simp [hy]))
      simp only [List.sum_cons, List.length_cons]
      push_cast
      omega
  have := key vs h
  rw [hlen] at this
  simpa using this

/-- Chunk size for `nt` tokens at offset `t0`. -/
def tokChunk (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- Each token chunk has at most four tokens. -/
theorem tokChunk_le (nt t0 : Nat) : tokChunk nt t0 ≤ 4 := by
  unfold tokChunk
  split <;> omega

/-- Each token chunk is non-empty and inside the batch. -/
theorem tokChunk_in_range (nt t0 : Nat) (h : t0 < nt) :
    0 < tokChunk nt t0 ∧ t0 + tokChunk nt t0 ≤ nt := by
  unfold tokChunk
  split <;> omega

end Strata.Q2Avx2
