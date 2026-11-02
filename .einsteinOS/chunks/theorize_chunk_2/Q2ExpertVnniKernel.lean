/-!
# MODULE: q2_expert_vnni_kernel
Q2_0 expert kernel (gate/up -> SwiGLU -> Q8_1 requantise -> down) for AVX-512 VNNI CPUs. Weights are 2-bit
codes 0..3 (value = code - 1) with an fp16 scale per 64-weight block; activations are int8 per 32-value chunk.

## Source semantics
* Offset identity: sum_j (c_j - 1) * xhat_j = sum(c * xhat) - sum(xhat), so codes may be fed unsigned to VNNI
  and the `hx = scale * sum` correction subtracted afterwards.
* A 64-weight block is two activation chunks of 32; the weight scale is shared, the activation scale is per chunk.
* Tail handling of the 512-bit row dot: groups of up to 8 blocks, masks cover exactly `2*nb` lanes.
* Activation quantiser: round half away from zero, clamp to [-127,127], store as int8.
* Token tile dispatch: 1..8 tokens select the template, larger counts fall to 8, oracle / out-of-range uses
  the single-token path. Gate row is 2r, up row is 2r+1.

## Form template identifiers
row_dot_z / row_dot_multi_z -> row_kernel ; act_quant_q8_1 -> act_quantizer ; QK, QKA -> block_size, chunk_size ;
s2_expert_vnni_multi -> multi_token_dispatch ; unpack_q2_0 -> code_unpack
-/

namespace Strata.Cpu.ExpertKernel

-- § constants and block geometry

/-- Weights per Q2_0 block. -/
def QK : Nat := 64

/-- Values per activation chunk. -/
def QKA : Nat := 32

/-- A weight block spans exactly two activation chunks. -/
theorem qk_eq_two_chunks : QK = 2 * QKA := by decide

/-- Chunk count of `k` full chunks is `k`. -/
theorem nchunks_of_mul (k : Nat) : (k * QKA) / QKA = k := by
  unfold QKA; omega

/-- Full blocks cover a row whose length is a multiple of the block size. -/
theorem blocks_cover (n : Nat) (h : n % QK = 0) : (n / QK) * QK = n := by
  unfold QK at *; omega

-- § code unpacking (vpmultishiftqb + and 3)

/-- The 2-bit code number `j` of a 16-bit lane. -/
def code2 (w j : Nat) : Nat := (w >>> (2 * j)) &&& 3

/-- Every unpacked code is at most 3. -/
theorem code2_le (w j : Nat) : code2 w j ≤ 3 := Nat.and_le_right

/-- Multishift selects byte `(w >> 2i) & 0xFF`; AND 3 leaves exactly the i-th code. -/
theorem multishift_and3 (w i : Nat) : ((w >>> (2 * i)) &&& 0xFF) &&& 3 = code2 w i := by
  have h : (0xFF : Nat) &&& 3 = 3 := by decide
  unfold code2
  rw [Nat.and_assoc, h]

/-- Scalar reference: code of weight `j` in a block lives in byte `j/4` at shift `2*(j%4)`. -/
def codeAt (bytes : Nat → Nat) (j : Nat) : Nat := (bytes (j / 4) >>> (2 * (j % 4))) &&& 3

/-- Scalar-reference codes are at most 3. -/
theorem codeAt_le (bytes : Nat → Nat) (j : Nat) : codeAt bytes j ≤ 3 := Nat.and_le_right

/-- Lane `l` (a 16-bit pair of bytes) holds weights `8l .. 8l+7`; weight `8l+i` sits in byte `2l + i/4`. -/
theorem lane_byte_index (l i : Nat) (hi : i < 8) : (8 * l + i) / 4 = 2 * l + i / 4 := by
  omega

/-- The shift of weight `8l+i` inside its byte equals the in-lane shift index modulo 4. -/
theorem lane_shift_index (l i : Nat) : (8 * l + i) % 4 = i % 4 := by
  omega

-- § offset algebra on integer lists

/-- Integer dot product of two lists (truncating at the shorter). -/
def dotI : List Int → List Int → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | c :: cs, x :: xs => c * x + dotI cs xs

/-- Integer sum of a list. -/
def sumI : List Int → Int
  | [] => 0
  | x :: xs => x + sumI xs

/-- The code offset identity: sum (c-1) x = sum c x - sum x. -/
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

/-- A dot product splits across a concatenation (per-chunk splitting of a block). -/
theorem dot_append : ∀ (a b c d : List Int), a.length = c.length →
    dotI (a ++ b) (c ++ d) = dotI a c + dotI b d
  | [], b, [], d, _ => by simp [dotI]
  | [], _, _ :: _, _, h => by simp at h
  | _ :: _, _, [], _, h => by simp at h
  | x :: a, b, y :: c, d, h => by
      have ih := dot_append a b c d (by simpa using h)
      simp only [List.cons_append, dotI]
      rw [ih]
      omega

/-- A zero code contributes nothing, so zero lanes left by the narrow load are harmless. -/
theorem zero_code_contrib (x : Int) : (0 : Int) * x = 0 := Int.zero_mul x

-- § tail masks and 8-block batches

/-- Lane mask for a batch of `nb` blocks (two lanes per block, 16 lanes max). -/
def tailMask (nb : Nat) : Nat := if nb ≥ 8 then 0xFFFF else (1 <<< (2 * nb)) - 1

/-- Masks never exceed 16 lanes. -/
theorem tailMask_le (nb : Nat) : tailMask nb ≤ 0xFFFF := by
  have small : ∀ k, k < 8 → (1 <<< (2 * k)) - 1 ≤ 0xFFFF := by decide
  unfold tailMask
  split
  · exact Nat.le_refl _
  · rename_i h
    exact small nb (by omega)

/-- A full batch uses the full mask. -/
theorem tailMask_full (nb : Nat) (h : nb ≥ 8) : tailMask nb = 0xFFFF := by
  unfold tailMask
  rw [if_pos h]

/-- Blocks in the batch starting at `b0`. -/
def batchSize (nblocks b0 : Nat) : Nat := min 8 (nblocks - b0)

/-- A batch is non-empty, at most 8, and stays inside the row. -/
theorem batch_bounds (nblocks b0 : Nat) (h : b0 < nblocks) :
    1 ≤ batchSize nblocks b0 ∧ batchSize nblocks b0 ≤ 8 ∧ b0 + batchSize nblocks b0 ≤ nblocks := by
  unfold batchSize
  omega

/-- The next batch start strictly advances, so the loop terminates. -/
theorem batch_advance (nblocks b0 : Nat) (h : b0 < nblocks) : b0 < b0 + 8 := by omega

-- § activation quantiser rule

/-- Clamp to the symmetric int8 range [-127, 127]. -/
def clamp127 (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Clamp output lies in [-127, 127]. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  by_cases h1 : v < -127 <;> by_cases h2 : v > 127 <;> simp [h1, h2] <;> omega

/-- Clamp is the identity on in-range values. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  have a : ¬ v < -127 := by omega
  have b : ¬ v > 127 := by omega
  simp [a, b]

/-- Clamp is idempotent. -/
theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v := by
  have h := clamp127_bounds v
  exact clamp127_id _ h.1 h.2

/-- Round half away from zero on a value scaled by two. -/
def roundAway (u : Int) : Int := if 0 ≤ u then (u + 1) / 2 else -((-u + 1) / 2)

/-- Round-half-away-from-zero is odd-symmetric (unlike round-half-even). -/
theorem roundAway_neg (u : Int) : roundAway (-u) = -roundAway u := by
  unfold roundAway
  by_cases h1 : 0 ≤ u <;> by_cases h2 : 0 ≤ -u <;> simp [h1, h2] <;> omega

/-- Quantised magnitude never exceeds 127 once clamped. -/
theorem quant_in_range (u : Int) : -127 ≤ clamp127 (roundAway u) ∧ clamp127 (roundAway u) ≤ 127 :=
  clamp127_bounds _

-- § row layout and token-tile dispatch

/-- Gate and up rows of the interleaved projection never coincide. -/
theorem gu_rows_distinct (r r' : Nat) : 2 * r ≠ 2 * r' + 1 := by omega

/-- The up row of an in-range index is in range. -/
theorem gu_row_lt (ff r : Nat) (h : r < ff) : 2 * r + 1 < 2 * ff := by omega

/-- Template selection: 8 or more tokens use the 8-token tile. -/
def ntSel (n : Nat) : Nat := if n ≥ 8 then 8 else n

/-- The selected tile never exceeds 8. -/
theorem ntSel_le (n : Nat) : ntSel n ≤ 8 := by
  unfold ntSel
  split <;> omega

/-- For 1..8 tokens the tile equals the token count. -/
theorem ntSel_id (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : ntSel n = n := by
  unfold ntSel
  split <;> omega

/-- Multi-token path is taken only without the oracle and with a valid token count. -/
def useMulti (oracle : Bool) (n maxt : Nat) : Bool := !oracle && decide (1 ≤ n) && decide (n ≤ maxt)

/-- The oracle always forces the single-token path. -/
theorem oracle_forces_single (n maxt : Nat) : useMulti true n maxt = false := by
  simp [useMulti]

/-- A valid count without the oracle takes the multi path. -/
theorem valid_uses_multi (n maxt : Nat) (h1 : 1 ≤ n) (h2 : n ≤ maxt) : useMulti false n maxt = true := by
  simp [useMulti, h1, h2]

end Strata.Cpu.ExpertKernel
