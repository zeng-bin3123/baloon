/-!
# MODULE: mtp_draft_quantizers
`tools/mtp_pack.py`: packs the MTP draft block into a GGUF with Q2_0 / Q4_0 / Q8_0 experts.

## Source semantics
* Q2_0 scale search tries 17 factors f in [1/2, 1] times amax and picks the least squared error.
* Q4_0 reference: code = min 15 (trunc (x/d + 8.5)); two nibbles per byte, low half first.
* bf16 -> f32 widening shifts left by 16 and is exact.
* Expert tensors are [512, 1280, 2560] and [512, 2560, 640] quantized along the last axis;
  the stated sizes (0.71 / 1.42 / 2.67 GB) follow from block geometry.

## Form template identifiers
q2_0 / q4_0 / q8_0   -> quantizer kernels
block, block_bytes   -> QUANT table fields
-/

namespace Strata.MtpDraftQuantizers

-- § Q2_0 scale search grid

/-- Search factor `i` of 17 is `(16 + i) / 32`, in numerator units over 32. -/
def factorNum (i : Nat) : Nat := 16 + i

/-- All 17 factors lie in [1/2, 1]. -/
theorem factor_range (i : Nat) (h : i ≤ 16) : 16 ≤ factorNum i ∧ factorNum i ≤ 32 := by
  unfold factorNum; omega

/-- Neighbouring factors are one step of 1/32 apart. -/
theorem factor_step (i : Nat) : factorNum (i + 1) = factorNum i + 1 := by
  unfold factorNum; omega

/-- Endpoints: factor 0 is 1/2 and factor 16 is 1 (the ggml reference scale). -/
theorem factor_endpoints : factorNum 0 = 16 ∧ factorNum 16 = 32 := by decide

-- § Q4_0 reference rounding

/-- Q4_0 code from the truncated integer `t = trunc(x/d + 8.5)`. -/
def q4code (t : Nat) : Nat := min 15 t

/-- Q4_0 codes are nibbles. -/
theorem q4code_lt (t : Nat) : q4code t < 16 := by unfold q4code; omega

/-- Two nibbles form one byte (low half first). -/
theorem q4_pack_lt (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) : lo + 16 * hi < 256 := by omega

/-- Unpacking returns both nibbles. -/
theorem q4_unpack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi := by omega

-- § bf16 widening

/-- A bf16 pattern shifted into the high half of an f32 word. -/
def widen (x : Nat) : Nat := x * 65536

/-- Widening leaves the low 16 bits zero. -/
theorem widen_low_zero (x : Nat) : widen x % 65536 = 0 := by unfold widen; omega

/-- Widening is invertible by the high half: it is exact. -/
theorem widen_high (x : Nat) : widen x / 65536 = x := by unfold widen; omega

/-- A 16-bit pattern widens to a 32-bit pattern. -/
theorem widen_lt (x : Nat) (h : x < 65536) : widen x < 4294967296 := by unfold widen; omega

-- § tensor byte accounting

/-- Bytes per row for a type: `n / block * block_bytes`. -/
def rowBytes (n block blockBytes : Nat) : Nat := n / block * blockBytes

/-- The expert inner dimensions give 720 and 180 Q2_0 bytes per row (matching the runtime pack). -/
theorem q2_0_row_bytes : rowBytes 2560 64 18 = 720 ∧ rowBytes 640 64 18 = 180 := by decide

/-- Total bytes of the two fused expert tensors for a format. -/
def expertBytes (block blockBytes : Nat) : Nat :=
  512 * 1280 * rowBytes 2560 block blockBytes + 512 * 2560 * rowBytes 640 block blockBytes

/-- Q2_0 experts: 707,788,800 B (~0.71 GB). -/
theorem q2_0_total : expertBytes 64 18 = 707788800 := by decide

/-- Q4_0 experts: 1,415,577,600 B (~1.42 GB). -/
theorem q4_0_total : expertBytes 32 18 = 1415577600 := by decide

/-- Q8_0 experts: 2,673,868,800 B (~2.67 GB). -/
theorem q8_0_total : expertBytes 32 34 = 2673868800 := by decide

/-- Q2_0 is smaller than Q4_0, which is smaller than Q8_0. -/
theorem format_order : expertBytes 64 18 < expertBytes 32 18 ∧ expertBytes 32 18 < expertBytes 32 34 := by
  decide

/-- Inner dimensions are whole blocks for every format. -/
theorem inner_dims_divisible : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

end Strata.MtpDraftQuantizers
