/-!
# MODULE: q2_avx2_fallback
Q2_0 expert rows and the Q8_1 activation quantiser for CPUs with AVX2 but no AVX-512. Same arithmetic as the
AVX-512 kernels: codes against int8 activations per 32-value chunk, minus the offset correction.

## Source semantics
* GGUF Q2_0 block: fp16 scale (2 bytes) then 16 code bytes, stride 18, 64 weights per block.
* 16 code bytes are unpacked by four shift-and-mask planes plus byte and word interleaves: output position p
  takes plane p mod 4 of byte p / 4; positions 0..31 come from bytes 0..7, 32..63 from bytes 8..15.
* Offset identity: sum (c-1) x = sum c x - sum x.
* Quantiser: values clamped to [-127,127], stored as int8 (identity on that range), chunk sums bounded by 127*32.
* Token tiling is groups of 4.

## Form template identifiers
unpack64 -> code_unpack ; row_multi -> row_kernel ; act_quant_q8_1_avx2 -> act_quantizer ;
q2_0_gguf_rows_multi_avx2 -> multi_token_rows
-/

namespace Strata.Cpu.Q2Avx2

-- § block geometry

/-- Bytes per GGUF Q2_0 block. -/
def blockStride : Nat := 18

/-- Scale bytes plus code bytes make up a block. -/
theorem stride_split : blockStride = 2 + 16 := by decide

/-- Sixty-four 2-bit codes occupy sixteen bytes. -/
theorem codes_bytes : 64 / 4 = 16 := by decide

/-- Block `b`'s scale and codes are inside its stride. -/
theorem block_offsets (b nblocks : Nat) (h : b < nblocks) :
    b * blockStride + 2 + 16 ≤ nblocks * blockStride := by
  unfold blockStride
  have : (b + 1) * 18 ≤ nblocks * 18 := Nat.mul_le_mul_right 18 h
  omega

-- § unpack index map

/-- Output position `p` takes byte `p/4`, plane `p%4`. -/
theorem unpack_position (p : Nat) : 4 * (p / 4) + p % 4 = p := by omega

/-- Low half (positions 0..31) reads bytes 0..7. -/
theorem lo_half_bytes (p : Nat) (h : p < 32) : p / 4 < 8 := by omega

/-- High half (positions 32..63) reads bytes 8..15. -/
theorem hi_half_bytes (p : Nat) (h1 : 32 ≤ p) (h2 : p < 64) : 8 ≤ p / 4 ∧ p / 4 < 16 := by omega

/-- Plane index is below 4. -/
theorem plane_lt (p : Nat) : p % 4 < 4 := by omega

/-- A 2-bit code extracted from a byte is at most 3. -/
theorem plane_code_le (b k : Nat) : ((b >>> (2 * k)) &&& 3) ≤ 3 := Nat.and_le_right

-- § offset algebra

/-- Integer dot product. -/
def dotI : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | c :: cs, x :: xs => c * x + dotI cs xs

/-- Integer sum. -/
def sumI : List Int → Int
  | [] => 0
  | x :: xs => x + sumI xs

/-- Offset identity used by the `hx` correction. -/
theorem dot_offset : ∀ (cs xs : List Int), cs.length = xs.length →
    dotI (cs.map (· - 1)) xs = dotI cs xs - sumI xs
  | [], [], _ => by simp [dotI, sumI]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | c :: cs, x :: xs, h => by
      have ih := dot_offset cs xs (by simpa using h)
      have hm : (c - 1) * x = c * x - x := by rw [Int.sub_mul, Int.one_mul]
      simp only [List.map_cons, dotI, sumI]
      rw [ih, hm]
      omega

-- § quantiser

/-- Clamp to [-127, 127]. -/
def clamp127 (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Clamped values lie in [-127, 127]. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  by_cases h1 : v < -127 <;> by_cases h2 : v > 127 <;> simp [h1, h2] <;> omega

/-- Two's-complement wrap into int8. -/
def wrap8 (v : Int) : Int := (v + 128) % 256 - 128

/-- Storing an in-range value as int8 is lossless. -/
theorem wrap8_id (v : Int) (h1 : -128 ≤ v) (h2 : v ≤ 127) : wrap8 v = v := by
  unfold wrap8
  omega

/-- Clamped values survive the int8 store. -/
theorem clamp_store_lossless (v : Int) : wrap8 (clamp127 v) = clamp127 v := by
  have h := clamp127_bounds v
  exact wrap8_id _ (by omega) h.2

/-- Sum of in-range quantised values is bounded by 127 per element. -/
theorem sum_bound : ∀ (l : List Int), (∀ v, v ∈ l → -127 ≤ v ∧ v ≤ 127) →
    -(127 * (l.length : Int)) ≤ sumI l ∧ sumI l ≤ 127 * (l.length : Int)
  | [], _ => by simp [sumI]
  | x :: xs, h => by
      have hx := h x (List.mem_cons_self ..)
      have ih := sum_bound xs (fun v hv => h v (List.mem_cons_of_mem _ hv))
      simp only [sumI, List.length_cons]
      omega

/-- A 32-value chunk sum fits in int32 with a large margin. -/
theorem chunk_sum_fits (s : Int) (h : -(127 * 32) ≤ s ∧ s ≤ 127 * 32) :
    -2147483648 ≤ s ∧ s ≤ 2147483647 := by omega

/-- Chunk `k`, lane `j` indexes inside the row when `n = 32 * nchunks`. -/
theorem chunk_index_lt (n k j : Nat) (hn : n % 32 = 0) (hk : k < n / 32) (hj : j < 32) :
    k * 32 + j < n := by omega

-- § token tiling

/-- Group size of the tiling loop. -/
def tileSize (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Groups are non-empty, at most 4, and inside the token range. -/
theorem tile_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ tileSize nt t0 ∧ tileSize nt t0 ≤ 4 ∧ t0 + tileSize nt t0 ≤ nt := by
  unfold tileSize
  omega

/-- The tiling loop advances. -/
theorem tile_advance (nt t0 : Nat) (h : t0 < nt) : t0 < t0 + tileSize nt t0 := by
  unfold tileSize
  omega

end Strata.Cpu.Q2Avx2
