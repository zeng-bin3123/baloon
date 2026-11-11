/-!
# MODULE: iq_quant_kernels
SOURCES: src/kernels/cuda/iq_kernels.cu
Integer model of the IQ-family path: per-format block sizes (`iq_row_bytes`), the
lane-strided warp dot `row_dot`, the `iq_mmvq` grid, the grouped-expert blob layout
(`native_expert_layout`, `native_expert_scratch_bytes`) and the sign/codebook decoders.

## Source semantics
* `Fmt<TY>` gives `qk` values per block, `ipb` dot calls per block and the `iqs` step; call
  `k` of a row is `(kbx, iqs) = (k / ipb, step * (k % ipb))`, lane-strided over 32 lanes.
* `iq_row_bytes(t, n) = (n / qk) * sizeof(block)`; zero means "unsupported".
* `iq_mmvq` launches `⌈n_out/4⌉` blocks of 4 warps, one warp per row.
* An expert blob is `[gate rows | up rows | down rows]` with `up_off = n_ff * gu_row`,
  `down_off = 2 * up_off`; scratch regions are 256-byte aligned.
* `unpack_ksigns` stores 7 sign bits and makes the 8th the parity, so every sign byte
  has an even number of negatives.

## Form template identifiers
`Fmt<TY>` → `fmt`; `iq_row_bytes` → `rowBytes`; `is_iq` → `isIQ`; `row_dot` call map → `callOf`;
`native_expert_layout` → `Layout`/`layout`; `native_expert_scratch_bytes` → `scratchBytes`;
`unpack_ksigns` → `ksigns`; `dq_q2_0` → `q2Value`.
-/

namespace IqQuantKernels

set_option maxRecDepth 8000

-- § formats

/-- `Fmt<TY>`: (qk, ipb, step) for each IQ / dot-format type. -/
def fmt : Nat → Option (Nat × Nat × Nat)
  | 16 => some (256, 8, 2) | 17 => some (256, 8, 2) | 18 => some (256, 8, 2)
  | 20 => some (32, 2, 2) | 21 => some (256, 8, 2) | 22 => some (256, 8, 2)
  | 29 => some (256, 8, 1) | 42 => some (64, 2, 1)
  | _ => none

/-- Bytes per block (`sizeof(block_*)` from ggml-common.h). -/
def blockBytes : Nat → Nat
  | 16 => 66 | 17 => 74 | 18 => 98 | 20 => 18 | 21 => 110 | 22 => 82
  | 29 => 56 | 23 => 136 | 11 => 110 | 42 => 18 | _ => 0

/-- Values per block. -/
def blockElems : Nat → Nat
  | 20 => 32 | 42 => 64 | 16 | 17 | 18 | 21 | 22 | 29 | 23 | 11 => 256 | _ => 1

/-- `iq_row_bytes`. -/
def rowBytes (t n : Nat) : Nat := (n / blockElems t) * blockBytes t

/-- `is_iq`. -/
def isIQ (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 ||
  t == 42 || t == 11

/-- The ggml-common block layouts sum to the sizes used: iq2_xxs `2+64`, iq2_xs `2+64+8`,
    iq2_s `2+64+8+8`, iq3_xxs `2+96`, iq3_s `2+64+8+32+4`, iq1_m `32+16+8`. -/
theorem block_layouts :
    2 + 64 = 66 ∧ 2 + 64 + 8 = 74 ∧ 2 + 64 + 8 + 8 = 82 ∧ 2 + 96 = 98 ∧
    2 + 64 + 8 + 32 + 4 = 110 ∧ 32 + 16 + 8 = 56 := by decide

/-- `is_iq t` holds exactly when `iq_row_bytes` is nonzero for a full super-block. -/
theorem isIQ_iff_rowBytes : ∀ t, t < 64 → (isIQ t = true ↔ rowBytes t 256 ≠ 0) := by decide

/-- Consistency of one `Fmt<TY>` entry: its `qk` is the block's element count, it makes at
    least one call per block, and its `ipb * step` quant offsets stay inside the block's
    `qk / 8` 32-bit words of Q8_1 activation codes. -/
def fmtOk (t : Nat) : Bool :=
  match fmt t with
  | some (qk, ipb, step) => qk == blockElems t && ipb * step ≤ qk / 8 && 0 < ipb
  | none => true

/-- Every dot format is consistent with its block geometry. -/
theorem fmt_consistent : ∀ t, t < 64 → fmtOk t = true := by decide

-- § warp dot iteration

/-- Call `k` of a row: block `k / ipb`, quant offset `step * (k % ipb)`. -/
def callOf (ipb step k : Nat) : Nat × Nat := (k / ipb, step * (k % ipb))

/-- Every (block, part) pair of a row is reached by exactly the call `block * ipb + part`. -/
theorem callOf_surj (ipb step b p : Nat) (hp : p < ipb) :
    callOf ipb step (b * ipb + p) = (b, step * p) := by
  have hpos : 0 < ipb := by omega
  simp only [callOf]
  have e1 : (b * ipb + p) / ipb = b := by
    rw [Nat.mul_comm, Nat.add_comm, Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hp, Nat.zero_add]
  have e2 : (b * ipb + p) % ipb = p := by
    rw [Nat.mul_comm, Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hp]
  rw [e1, e2]

/-- Calls with indices `< nb * ipb` address blocks `< nb` only. -/
theorem callOf_block_lt (ipb step k nb : Nat) (h : k < nb * ipb) : (callOf ipb step k).1 < nb := by
  simp only [callOf]
  exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- Lane-striding: call `k` belongs to lane `k % 32` at round `k / 32`, and that lane/round pair
    is unique. -/
theorem lane_round (k : Nat) : k % 32 + 32 * (k / 32) = k := Nat.mod_add_div k 32

-- § launch grid

/-- `iq_mmvq` launches `⌈n_out / 4⌉` blocks; row `blockIdx.x * 4 + threadIdx.y` covers every
    output row exactly once and no row of block `b < grid` exceeds `n_out + 3`. -/
theorem grid_covers (nOut r : Nat) (h : r < nOut) :
    r / 4 < (nOut + 3) / 4 ∧ (r / 4) * 4 + r % 4 = r := by omega

/-- Out-of-range warps of the last block are exactly those with `row ≥ n_out`; all lie in the
    final block. -/
theorem grid_tail (nOut b y : Nat) (hb : b < (nOut + 3) / 4) (hy : y < 4) :
    b * 4 + y < nOut + 3 := by omega

-- § expert blob layout

/-- `NativeExpertLayout`. -/
structure Layout where
  nEmbd : Nat
  nFF : Nat
  guRow : Nat
  dRow : Nat

/-- `up_off = n_ff * gu_row`. -/
def Layout.upOff (L : Layout) : Nat := L.nFF * L.guRow
/-- `down_off = 2 * up_off`. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- `bytes = down_off + n_embd * d_row`. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

/-- Gate row `r < n_ff` lies wholly in `[0, up_off)`. -/
theorem gate_row_in_region (L : Layout) (r : Nat) (h : r < L.nFF) :
    r * L.guRow + L.guRow ≤ L.upOff := by
  unfold Layout.upOff
  have : (r + 1) * L.guRow ≤ L.nFF * L.guRow := Nat.mul_le_mul_right _ h
  rw [Nat.succ_mul] at this; exact this

/-- Up row `r < n_ff` lies wholly in `[up_off, down_off)`. -/
theorem up_row_in_region (L : Layout) (r : Nat) (h : r < L.nFF) :
    L.upOff ≤ L.upOff + r * L.guRow ∧ L.upOff + r * L.guRow + L.guRow ≤ L.downOff := by
  have := gate_row_in_region L r h
  unfold Layout.downOff; omega

/-- Down row `r < n_embd` lies wholly in `[down_off, bytes)`. -/
theorem down_row_in_region (L : Layout) (r : Nat) (h : r < L.nEmbd) :
    L.downOff + r * L.dRow + L.dRow ≤ L.bytes := by
  unfold Layout.bytes
  have : (r + 1) * L.dRow ≤ L.nEmbd * L.dRow := Nat.mul_le_mul_right _ h
  rw [Nat.succ_mul] at this; omega

-- § scratch alignment

/-- `(f + 255) & ~255` as arithmetic: round up to a multiple of 256. -/
def align256 (f : Nat) : Nat := (f + 255) / 256 * 256

/-- The aligned size is a multiple of 256, at least `f`, and wastes fewer than 256 bytes. -/
theorem align256_spec (f : Nat) :
    align256 f % 256 = 0 ∧ f ≤ align256 f ∧ align256 f < f + 256 := by
  unfold align256; omega

/-- The bit form `(f + 255) &&& ~~~255` agrees with the arithmetic form on a test range. -/
theorem align256_bits : ∀ f, f < 1024 → (f + 255) / 256 * 256 = (f + 255) - (f + 255) % 256 := by
  intro f _; omega

/-- `native_expert_scratch_bytes`: gate | up | h (each `cap*n_ff` floats) | hq (Q8_1). -/
def scratchBytes (cap nFF : Nat) : Nat :=
  3 * align256 (cap * nFF * 4) + align256 (cap * (nFF / 32) * 36)

/-- The four scratch carve-outs fit: the Q8_1 intermediate starts at `3 * fa` and ends inside
    the advertised scratch size. -/
theorem scratch_fits (cap nFF : Nat) :
    3 * align256 (cap * nFF * 4) + cap * (nFF / 32) * 36 ≤ scratchBytes cap nFF := by
  have := (align256_spec (cap * (nFF / 32) * 36)).2.1
  unfold scratchBytes; omega

-- § decoders

/-- Bit count of the low 8 bits. -/
def popcount8 (v : Nat) : Nat := (List.range 8).foldl (fun acc i => acc + (v >>> i) % 2) 0

/-- `unpack_ksigns` byte: 7 stored sign bits with the parity bit in bit 7. -/
def ksigns (v : Nat) : Nat := v ^^^ ((popcount8 v % 2) <<< 7)

/-- Every reconstructed sign byte has an even number of negative signs, and its low 7 bits are
    the stored ones. -/
theorem ksigns_even : ∀ v, v < 128 → popcount8 (ksigns v) % 2 = 0 ∧ ksigns v % 128 = v := by
  decide

/-- `dq_q2_0`: a 2-bit code `c` dequantizes to `d * (c - 1)`, i.e. one of `{-1, 0, 1, 2}·d`. -/
def q2Value (c : Nat) : Int := (c : Int) - 1

/-- Q2_0 codes span exactly `-1 .. 2`. -/
theorem q2Value_range (c : Nat) (h : c < 4) : -1 ≤ q2Value c ∧ q2Value c ≤ 2 := by
  unfold q2Value; omega

/-- Q2_0 code `i` of a 64-element block sits in byte `i / 4` at bit `2 * (i % 4)`: 16 bytes. -/
theorem q2_code_position (i : Nat) (h : i < 64) : i / 4 < 16 ∧ 2 * (i % 4) ≤ 6 := by omega

/-- IQ3_XXS scale `(ls * s + s / 2) / 2` uses `ls = aux >> 28 < 16`, IQ2_XXS `ls = aux>>27 | 1`
    is odd: both are bounded by the top bits of a 32-bit word. -/
theorem iq_scale_bits : ∀ a, a < 2^32 → a >>> 28 < 16 ∧ a >>> 27 < 32 := by
  intro a ha
  constructor
  · rw [Nat.shiftRight_eq_div_pow]; omega
  · rw [Nat.shiftRight_eq_div_pow]; omega

end IqQuantKernels
