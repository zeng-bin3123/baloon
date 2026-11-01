/-!
# MODULE: activation_quant_layout
Block layouts and clamp contracts of `quantize_act.cu` (Q8_0 and Q8_K).

## Source semantics
`block_q8_0` = fp16 d + 32 int8 = 34 bytes; `block_q8_K` = f32 d + 256 int8 + 16 int16 bsums = 292 bytes.
The scaled Q8_0 quantizer clamps to [-127,127]; the Q8_K quantizer only clamps above (min 127).
Block counts are `n / 32` and `n / 256` and require divisibility.

## Form template identifiers
q8_0Bytes / q8KBytes -> block sizes; clamp127 -> `v < -127 ? -127 : v > 127 ? 127 : v`.
-/

namespace Strata.ActivationQuantLayout

def q8_0Bytes : Nat := 2 + 32
def q8KBytes : Nat := 4 + 256 + 16 * 2

theorem q8_0_is_34 : q8_0Bytes = 34 := by decide
theorem q8K_is_292 : q8KBytes = 292 := by decide

/-- The 34-byte stride makes the int8 payload start at an even, non-4-aligned offset. -/
theorem q8_0_payload_not_word_aligned : (2 : Nat) % 4 ≠ 0 ∧ q8_0Bytes % 2 = 0 := by
  decide

-- § clamps

def clamp127 (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

theorem clamp127_range (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  split <;> (try split) <;> omega

theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  rw [if_neg (by omega), if_neg (by omega)]

theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v := by
  exact clamp127_id _ (clamp127_range v).1 (clamp127_range v).2

/-- Q8_K clamps only above. -/
def minOnly (v : Int) : Int := if v < 127 then v else 127

theorem minOnly_le (v : Int) : minOnly v ≤ 127 := by
  unfold minOnly
  split <;> omega

-- § block counts

theorem q8_0_blocks_cover (n : Nat) (h : n % 32 = 0) : n / 32 * 32 = n := by omega
theorem q8K_blocks_cover (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by omega

/-- Q8_K blocks hold sixteen groups of sixteen for `bsums`. -/
theorem bsums_groups : 256 / 16 = 16 := by decide

/-- A zero block (amax = 0) has a zero sum, so the stored correction is zero. -/
theorem zero_block_sum : (List.replicate 32 (0 : Int)).sum = 0 := by decide

end Strata.ActivationQuantLayout
