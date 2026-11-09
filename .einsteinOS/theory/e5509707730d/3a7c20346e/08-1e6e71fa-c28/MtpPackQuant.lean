/-!
# MODULE: mtp_pack_quant
SOURCES: tools/mtp_pack.py
Packs the MTP draft block as a GGUF: dense tensors stay BF16/F32, fused routed experts
(`gate_up_proj [512,1280,2560]`, `down_proj [512,2560,640]`) are quantised along the last axis to Q2_0, Q4_0 or
Q8_0 with round-to-nearest.

## Source semantics
* Q2_0 searches 17 candidate scales `amax * f` with `f` from 1/2 to 1 and clamps symbols to [-1, 2]; the
  stored code is `symbol + 1` in 0..3.
* Q4_0 packs `q[:16] | q[16:] << 4` with every code at most 15.
* Q8_0 codes are `rint(x / d)` with `d = amax/127`, so they fit an int8.
* The inner dimension must be a multiple of the block size, and the byte counts are
  `elements / block * block_bytes` (about 0.71, 1.42 and 2.67 GB for the three formats).

## Form template identifiers
- np.linspace(0.5, 1.0, 17)  → `scaleNumer` (numerator over 32)
- np.clip(rint(x*inv), -1, 2)+1 → `codeOfSymbol`
- q[:16] | (q[16:] << 4)     → `nibblePack`
- blob size                  → `expertBytes`
-/

namespace Strata.MtpPack

-- § Q2_0 scale search

/-- Candidate scale factor `f_i = (16 + i) / 32`, `i = 0..16`, as a numerator over 32. -/
def scaleNumer (i : Nat) : Nat := 16 + i

/-- The 17 candidates span `amax/2` to `amax` inclusive. -/
theorem scale_grid_span (i : Nat) (h : i ≤ 16) : 16 ≤ scaleNumer i ∧ scaleNumer i ≤ 32 := by
  unfold scaleNumer
  omega

/-- There are exactly 17 candidates. -/
theorem scale_grid_count : (List.range 17).length = 17 := by
  decide

/-- Symbol in [-1, 2] maps to a 2-bit code `symbol + 1` in [0, 3]. -/
def codeOfSymbol (s : Int) : Int := s + 1

/-- Codes of clamped symbols are valid 2-bit codes. -/
theorem code_of_symbol_range (s : Int) (h1 : -1 ≤ s) (h2 : s ≤ 2) :
    0 ≤ codeOfSymbol s ∧ codeOfSymbol s ≤ 3 := by
  unfold codeOfSymbol
  omega

-- § Q4_0 / Q8_0

/-- `q[:16] | (q[16:] << 4)` for one nibble pair. -/
def nibblePack (lo hi : Nat) : Nat := lo + 16 * hi

/-- Packing two codes of at most 15 fits a byte and unpacks to the same pair. -/
theorem nibble_pack_roundtrip (lo hi : Nat) (hl : lo ≤ 15) (hh : hi ≤ 15) :
    nibblePack lo hi < 256 ∧ nibblePack lo hi % 16 = lo ∧ nibblePack lo hi / 16 = hi := by
  unfold nibblePack
  omega

/-- `min 15 t` is a valid Q4_0 code. -/
theorem q4_0_code_clamped (t : Nat) : min 15 t ≤ 15 := by
  omega

/-- A Q8_0 code with magnitude at most 127 fits an int8. -/
theorem q8_0_code_int8 (r : Int) (h1 : -127 ≤ r) (h2 : r ≤ 127) : -128 ≤ r ∧ r ≤ 127 := by
  omega

-- § expert sizes

/-- Bytes of a quantised expert tensor: `elements / block * block_bytes`. -/
def expertBytes (elems block blockBytes : Nat) : Nat := elems / block * blockBytes

/-- Inner dimensions 2560 and 640 are multiples of both block sizes. -/
theorem inner_dims_divisible :
    2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by
  decide

/-- Q2_0 bytes of the two fused expert tensors and their sum (about 0.71 GB). -/
theorem q2_0_expert_bytes :
    expertBytes (512 * 1280 * 2560) 64 18 = 471859200 ∧
    expertBytes (512 * 2560 * 640) 64 18 = 235929600 ∧
    expertBytes (512 * 1280 * 2560) 64 18 + expertBytes (512 * 2560 * 640) 64 18 = 707788800 := by
  decide

/-- Q4_0 bytes of the two fused expert tensors and their sum (about 1.42 GB). -/
theorem q4_0_expert_bytes :
    expertBytes (512 * 1280 * 2560) 32 18 + expertBytes (512 * 2560 * 640) 32 18 = 1415577600 := by
  decide

/-- Q8_0 bytes of the two fused expert tensors and their sum (about 2.67 GB). -/
theorem q8_0_expert_bytes :
    expertBytes (512 * 1280 * 2560) 32 34 + expertBytes (512 * 2560 * 640) 32 34 = 2673868800 := by
  decide

/-- Q2_0 is exactly half of Q4_0 on the experts (2 bits vs 4, same 2-byte scale per 2x elements). -/
theorem q2_0_half_of_q4_0 :
    2 * (expertBytes (512 * 1280 * 2560) 64 18 + expertBytes (512 * 2560 * 640) 64 18) =
      expertBytes (512 * 1280 * 2560) 32 18 + expertBytes (512 * 2560 * 640) 32 18 := by
  decide

end Strata.MtpPack
