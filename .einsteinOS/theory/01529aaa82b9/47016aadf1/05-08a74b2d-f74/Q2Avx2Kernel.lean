/-!
# MODULE: q2_avx2_kernel
The AVX2-only Q2_0 row dot over GGUF-layout blocks (18 bytes: fp16 scale + 16 code bytes) and the scalar-rule
activation quantiser, for CPUs without AVX-512.

## Source semantics
* Value `4k+t` of an unpacked block is code `t` of byte `k`; the low half is values 0..31, the high half 32..63.
* `maddubs` of codes (≤3) with int8 activations never saturates an int16 lane, so the integer sums are exact.
* The quantiser clamps to `[-127, 127]`; a chunk sum is bounded by `127*32`.
* Token batches are processed four at a time.

## Form template identifiers
unpack64 -> unpack_block, row_multi -> row_dot_multi, rows -> rows, act_quant_q8_1_avx2 -> act_quant.
-/

namespace Strata.Q2Avx2

-- § block layout

/-- Bytes per GGUF Q2_0 block. -/
def BLOCK_BYTES : Nat := 18

/-- Scale (2 bytes) plus 16 code bytes is 18 bytes. -/
theorem block_bytes_eq : 2 + 16 = BLOCK_BYTES := by decide

/-- Block `b` has its scale at `18b` and codes at `18b+2`, ending exactly where block `b+1` starts. -/
theorem block_layout (b : Nat) : 18 * b + 2 + 16 = 18 * (b + 1) := by omega

/-- Activation chunk indices of block `b`: low half at `64b`, high half at `64b+32 = 32*(2b+1)`. -/
theorem act_halves (b : Nat) : 64 * b + 32 = 32 * (2 * b + 1) ∧ 64 * b = 32 * (2 * b) := by omega

-- § unpack order

/-- Output value index `4k+t` comes from byte `k`, code slot `t` (the unpack interleave order). -/
theorem unpack_index (k t : Nat) (ht : t < 4) : (4 * k + t) / 4 = k ∧ (4 * k + t) % 4 = t := by
  omega

/-- The low vector holds values 0..31 (bytes 0..7) and the high vector 32..63 (bytes 8..15). -/
theorem unpack_halves (i : Nat) (hi : i < 64) :
    (i < 32 → i / 4 < 8) ∧ (32 ≤ i → 8 ≤ i / 4 ∧ i / 4 < 16) := by omega

/-- Extracting code slot `t` of a byte with shift-and-mask equals the div/mod formula. -/
def codeOf (byte t : Nat) : Nat := byte / 4 ^ t % 4

/-- Packing four codes then unpacking returns them. -/
theorem unpack_pack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    codeOf (c0 + 4 * c1 + 16 * c2 + 64 * c3) 0 = c0 ∧ codeOf (c0 + 4 * c1 + 16 * c2 + 64 * c3) 1 = c1 ∧
    codeOf (c0 + 4 * c1 + 16 * c2 + 64 * c3) 2 = c2 ∧ codeOf (c0 + 4 * c1 + 16 * c2 + 64 * c3) 3 = c3 := by
  simp only [codeOf]
  omega

-- § exact integer sums

/-- `maddubs` of two code*int8 products fits int16: `2 * 3 * 127 ≤ 32767`. -/
theorem maddubs_exact : 2 * 3 * 127 ≤ 32767 := by decide

/-- After `madd` with ones, a 32-value half sum is at most `3*127*32`, far below int32. -/
theorem half_sum_fits : 3 * 127 * 32 < 2 ^ 31 := by decide

/-- Codes decode to `c - 1 ∈ [-1, 2]`. -/
theorem decoded_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by omega

-- § quantiser

/-- Clamp to the int8 symmetric range. -/
def clamp127 (v : Int) : Int :=
  if v < -127 then -127 else if v > 127 then 127 else v

/-- The clamp stays within `[-127, 127]`. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  split <;> (try split) <;> omega

/-- A list of in-range values has sum bounded by `127 * length`. -/
theorem sum_bound : ∀ (l : List Int), (∀ v ∈ l, -127 ≤ v ∧ v ≤ 127) →
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | a :: t, h => by
    have ha := h a (by simp)
    have ht := sum_bound t (fun v hv => h v (by simp [hv]))
    simp only [List.sum_cons, List.length_cons]
    omega

/-- Chunks produced for `n` activations. -/
def nChunks (n : Nat) : Nat := n / 32

/-- Four 8-lane vectors cover a 32-value chunk. -/
theorem vectors_per_chunk : 4 * 8 = 32 := by decide

/-- Chunk `k` reads values `32k .. 32k+31`, inside the input when `k < n/32`. -/
theorem chunk_in_range (n k : Nat) (hk : k < nChunks n) : 32 * k + 32 ≤ n := by
  unfold nChunks at hk
  omega

-- § token batching

/-- Tokens in the batch starting at `t0`: at most four. -/
def batchSize (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- Batches are non-empty, bounded by four, and stay within the token count. -/
theorem batch_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ batchSize nt t0 ∧ batchSize nt t0 ≤ 4 ∧ t0 + batchSize nt t0 ≤ nt := by
  unfold batchSize
  split <;> omega

end Strata.Q2Avx2
