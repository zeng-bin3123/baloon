/-!
# MODULE: iq_quant_kernels
I-quant (IQ2/IQ3/IQ1/IQ4, Q2_0, Q3_K) dot products, dequantizers and grouped native-expert layout.

## Source semantics
Each format has `qk` values per block, `ipb` dot calls per block and an `iqs` stride `step`; a warp processes
call `k` as block `k / ipb`, part `k % ipb`. Row bytes are `n / qk * sizeof(block)`. A native expert blob is
`[gate rows | up rows | down rows]` with `up_off = n_ff * gu_row` and `down_off = 2 * up_off`.
Gate/up dequantization interleaves rows as `2r + parity`.

## Form template identifiers
- `Fmt<TY>` -> fmtQk / fmtIpb / fmtStep, `iq_row_bytes` -> rowBytes
- `NativeExpertLayout` -> ExpertLayout, `native_expert_layout` -> mkLayout
-/

namespace Strata.IqKernels

-- § format table

/-- Values per block. -/
def fmtQk : Nat → Nat
  | 16 => 256 | 17 => 256 | 18 => 256 | 20 => 32 | 21 => 256 | 22 => 256
  | 29 => 256 | 42 => 64 | 11 => 256 | 23 => 256 | _ => 0

/-- Dot calls per block. -/
def fmtIpb : Nat → Nat
  | 16 => 8 | 17 => 8 | 18 => 8 | 20 => 2 | 21 => 8 | 22 => 8 | 29 => 8 | 42 => 2 | _ => 0

/-- Stride of the `iqs` argument between calls. -/
def fmtStep : Nat → Nat
  | 16 => 2 | 17 => 2 | 18 => 2 | 20 => 2 | 21 => 2 | 22 => 2 | 29 => 1 | 42 => 1 | _ => 0

/-- Block sizes in bytes (ggml-common.h): IQ2_XXS 66, IQ2_XS 74, IQ3_XXS 98, IQ4_NL 18, IQ3_S 110,
IQ2_S 82, IQ1_M 56, Q2_0 18, Q3_K 110, IQ4_XS 136. -/
def blockBytes : Nat → Nat
  | 16 => 66 | 17 => 74 | 18 => 98 | 20 => 18 | 21 => 110 | 22 => 82
  | 29 => 56 | 42 => 18 | 11 => 110 | 23 => 136 | _ => 0

/-- Bytes of one weight row of `n` elements. -/
def rowBytes (t n : Nat) : Nat := n / fmtQk t * blockBytes t

/-- Example: IQ2_XXS row of 2560. -/
theorem rowBytes_iq2xxs : rowBytes 16 2560 = 10 * 66 := by decide

/-- Example: Q3_K row of 2560. -/
theorem rowBytes_q3k : rowBytes 11 2560 = 10 * 110 := by decide

/-- Example: IQ4_NL row of 640. -/
theorem rowBytes_iq4nl : rowBytes 20 640 = 20 * 18 := by decide

/-- Unknown types have zero row bytes. -/
theorem rowBytes_unknown (n : Nat) : rowBytes 5 n = 0 := by
  simp [rowBytes, blockBytes]

-- § call decomposition

/-- Call `k` decomposes into block `k / ipb` and part `k % ipb`. -/
theorem call_decomposition (k ipb : Nat) : ipb * (k / ipb) + k % ipb = k := Nat.div_add_mod k ipb

/-- The part index is below `ipb`. -/
theorem part_lt (k ipb : Nat) (h : 0 < ipb) : k % ipb < ipb := Nat.mod_lt _ h

/-- For 256-value formats with step 2 the `iqs` argument stays below 16 (= qi). -/
theorem iqs_range_step2 (part : Nat) (h : part < 8) : 2 * part < 16 := by omega

/-- Warp coverage of an `ipb`-per-block row: 32 lanes cover every call. -/
theorem lane_cover (nb ipb : Nat) : nb * ipb ≤ ((nb * ipb + 31) / 32) * 32 := by omega

-- § native expert layout

/-- Byte layout of one expert blob. -/
structure ExpertLayout where
  guRow : Nat
  dRow : Nat
  upOff : Nat
  downOff : Nat
  bytes : Nat

/-- `native_expert_layout`. -/
def mkLayout (guType dType nEmbd nFf : Nat) : ExpertLayout :=
  let guRow := rowBytes guType nEmbd
  let dRow := rowBytes dType nFf
  let upOff := nFf * guRow
  let downOff := 2 * upOff
  { guRow := guRow, dRow := dRow, upOff := upOff, downOff := downOff,
    bytes := downOff + nEmbd * dRow }

/-- The blob is gate, up and down matrices back to back. -/
theorem layout_bytes (g d e f : Nat) :
    (mkLayout g d e f).bytes = 2 * (f * rowBytes g e) + e * rowBytes d f := rfl

/-- The up matrix starts right after gate. -/
theorem layout_up_off (g d e f : Nat) : (mkLayout g d e f).upOff = f * rowBytes g e := rfl

/-- The down matrix starts right after up. -/
theorem layout_down_off (g d e f : Nat) :
    (mkLayout g d e f).downOff = 2 * (mkLayout g d e f).upOff := rfl

/-- Offsets are ordered. -/
theorem layout_monotone (g d e f : Nat) :
    (mkLayout g d e f).upOff ≤ (mkLayout g d e f).downOff ∧
    (mkLayout g d e f).downOff ≤ (mkLayout g d e f).bytes := by
  simp only [mkLayout]
  omega

-- § alignment

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by unfold align256; omega

/-- Alignment yields a multiple of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by unfold align256; omega

/-- Alignment is tight. -/
theorem align256_lt (x : Nat) : align256 x < x + 256 := by unfold align256; omega

-- § dequant indexing

/-- Gate/up interleave: row slot `2r + parity` determines `(r, parity)` uniquely. -/
theorem interleave_injective (r p r' p' : Nat) (hp : p < 2) (hp' : p' < 2)
    (h : 2 * r + p = 2 * r' + p') : r = r' ∧ p = p' := by omega

/-- Q2_0 codes decode to {-1,0,1,2}. -/
theorem q2_0_code_range (code : Nat) (h : code < 4) :
    -1 ≤ (code : Int) - 1 ∧ (code : Int) - 1 ≤ 2 := by omega

/-- Superblock-to-element mapping for the flat dequantizer. -/
theorem superblock_offset (i j : Nat) (hj : j < 256) : (i * 256 + j) / 256 = i := by omega

/-- Row-major grid for MMVQ covers all rows with 4 rows per block. -/
theorem mmvq_grid_cover (nOut : Nat) : nOut ≤ ((nOut + 3) / 4) * 4 := by omega

end Strata.IqKernels
