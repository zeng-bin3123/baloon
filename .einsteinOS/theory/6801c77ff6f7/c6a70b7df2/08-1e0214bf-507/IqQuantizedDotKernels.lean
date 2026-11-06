/-!
# MODULE: iq_quantized_dot_kernels
SOURCES: src/kernels/cuda/iq_kernels.cu
Integer dot products, dequantizers and the grouped native-expert pipeline for the IQ/Q3_K/Q2_0 block formats
(types 16,17,18,20,21,22,23,29,42,11), transcribed from llama.cpp's vecdotq/dequantize/quantize.

## Source semantics
* `iq_row_bytes` is (values / values-per-block) * sizeof(block); unsupported types give 0.
* `unpack_ksigns` restores the 8th sign bit of a 7-bit sign index as the parity of the 7 bits.
* The warp loop `row_dot` strides call index k = (block, part) over lanes; kbx = k / ipb, iqs = step * (k % ipb).
* `native_expert_layout` stores [gate rows | up rows | down rows] contiguously.
* Dequantizer thread maps write disjoint 8-value (or 4+4 value) slices of a 256-value superblock.

## Form template identifiers
iq_row_bytes -> row_bytes_fn ; unpack_ksigns -> sign_expand ; NativeExpertLayout -> expert_blob_layout ;
Fmt<TY> -> format_traits ; native_expert_scratch_bytes -> scratch_size_fn ; dq_dispatch -> dequant_dispatch
-/

namespace Strata.IqKernels

/-- Is this ggml type id one of the IQ-family types the file handles? -/
def iqSupported (t : Nat) : Bool :=
  decide (t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 20 ∨ t = 21 ∨ t = 22 ∨ t = 23 ∨ t = 29 ∨ t = 42 ∨ t = 11)

/-- Bytes of one weight row of `n` values (sizes of the ggml block structs). -/
def iqRowBytes (t n : Nat) : Nat :=
  match t with
  | 16 => (n / 256) * 66
  | 17 => (n / 256) * 74
  | 18 => (n / 256) * 98
  | 20 => (n / 32) * 18
  | 21 => (n / 256) * 110
  | 22 => (n / 256) * 82
  | 29 => (n / 256) * 56
  | 23 => (n / 256) * 136
  | 11 => (n / 256) * 110
  | 42 => (n / 64) * 18
  | _ => 0

-- § row bytes

/-- Every supported type has a positive row size for a 256-value row. -/
theorem iqRowBytes_pos_of_supported :
    ∀ t ∈ [16, 17, 18, 20, 21, 22, 23, 29, 42, 11], 0 < iqRowBytes t 256 := by
  decide

/-- Unsupported types report zero bytes, which is how the host detects them. -/
theorem iqRowBytes_unsupported (t n : Nat) (h : iqSupported t = false) : iqRowBytes t n = 0 := by
  unfold iqRowBytes
  split <;> simp_all [iqSupported]

/-- A superblock count times 256 recovers the row length when the row is 256-aligned. -/
theorem superblocks_cover (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by
  omega

-- § sign unpacking

/-- Population count of the low 8 bits. -/
def popc8 (v : Nat) : Nat := (List.range 8).foldl (fun a i => a + (v >>> i) % 2) 0

/-- Model of `unpack_ksigns` before the 0x01010101 broadcast. -/
def ksigns (v : Nat) : Nat := v ^^^ ((popc8 v % 2) <<< 7)

/-- For a 7-bit sign index the restored top bit is the parity of the index. -/
theorem ksigns_bit7 : ∀ v < 128, ksigns v / 128 % 2 = popc8 v % 2 := by
  decide +kernel

/-- The low seven bits of the sign index are preserved. -/
theorem ksigns_low : ∀ v < 128, ksigns v % 128 = v := by
  decide +kernel

/-- Multiplying a byte by 0x01010101 replicates it into all four bytes. -/
theorem broadcast_bytes : ∀ s < 256, ∀ k < 4, (s * 16843009 / 256 ^ k) % 256 = s := by
  decide +kernel

-- § warp call mapping

/-- Lane-strided call index covers the block range: the block index is in range. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < nb * ipb) : k / ipb < nb := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  exact h

/-- The (block, part) decomposition of a call index is exact. -/
theorem call_decomp (k ipb : Nat) : (k / ipb) * ipb + k % ipb = k := by
  have := Nat.div_add_mod k ipb
  rw [Nat.mul_comm] at this
  omega

/-- The iqs offset stays under step * ipb. -/
theorem iqs_lt (k ipb step : Nat) (hp : 0 < ipb) : step * (k % ipb) < step * ipb ∨ step = 0 := by
  by_cases hs : step = 0
  · exact Or.inr hs
  · left
    exact Nat.mul_lt_mul_of_pos_left (Nat.mod_lt _ hp) (Nat.pos_of_ne_zero hs)

/-- Lane decomposition of a call index: lane = k % 32 and round = k / 32. -/
theorem lane_round (k : Nat) : k % 32 + 32 * (k / 32) = k := by
  omega

-- § format traits

/-- Format traits: values per block, calls per block, iqs stride. -/
def fmtOf (t : Nat) : Option (Nat × Nat × Nat) :=
  match t with
  | 16 => some (256, 8, 2)
  | 17 => some (256, 8, 2)
  | 18 => some (256, 8, 2)
  | 20 => some (32, 2, 2)
  | 21 => some (256, 8, 2)
  | 22 => some (256, 8, 2)
  | 29 => some (256, 8, 1)
  | 42 => some (64, 2, 1)
  | _ => none

/-- Sanity predicate for a trait row. -/
def fmtOk (t : Nat) : Bool :=
  match fmtOf t with
  | some (qk, ipb, _) => decide (0 < ipb ∧ qk % 32 = 0)
  | none => false

/-- All eight kernel formats have well-formed traits. -/
theorem fmtOk_all : ∀ t ∈ [16, 17, 18, 20, 21, 22, 29, 42], fmtOk t = true := by
  decide

-- § native expert layout

/-- Layout of one expert blob. -/
structure NativeExpertLayout where
  guRow : Nat
  dRow : Nat
  upOff : Nat
  downOff : Nat
  bytes : Nat

/-- `native_expert_layout`: gate rows, then up rows, then down rows. -/
def nativeExpertLayout (guType dType nEmbd nFf : Nat) : NativeExpertLayout :=
  { guRow := iqRowBytes guType nEmbd
    dRow := iqRowBytes dType nFf
    upOff := nFf * iqRowBytes guType nEmbd
    downOff := 2 * (nFf * iqRowBytes guType nEmbd)
    bytes := 2 * (nFf * iqRowBytes guType nEmbd) + nEmbd * iqRowBytes dType nFf }

/-- The product bound used for every row-in-region argument. -/
theorem row_end_le (r n w : Nat) (h : r < n) : r * w + w ≤ n * w := by
  have := Nat.mul_le_mul_right w (show r + 1 ≤ n by omega)
  simp only [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The up region starts exactly after the gate region. -/
theorem up_after_gate (gu d e f : Nat) :
    (nativeExpertLayout gu d e f).downOff =
      (nativeExpertLayout gu d e f).upOff + f * (nativeExpertLayout gu d e f).guRow := by
  simp only [nativeExpertLayout]
  omega

/-- A gate row lies inside the gate region. -/
theorem gate_row_in_bounds (gu d e f r : Nat) (h : r < f) :
    r * (nativeExpertLayout gu d e f).guRow + (nativeExpertLayout gu d e f).guRow ≤
      (nativeExpertLayout gu d e f).upOff := by
  simp only [nativeExpertLayout]
  exact row_end_le r f _ h

/-- An up row lies inside the up region. -/
theorem up_row_in_bounds (gu d e f r : Nat) (h : r < f) :
    (nativeExpertLayout gu d e f).upOff + r * (nativeExpertLayout gu d e f).guRow +
      (nativeExpertLayout gu d e f).guRow ≤ (nativeExpertLayout gu d e f).downOff := by
  have := gate_row_in_bounds gu d e f r h
  have h2 := up_after_gate gu d e f
  simp only [nativeExpertLayout] at *
  omega

/-- A down row lies inside the blob. -/
theorem down_row_in_bounds (gu d e f r : Nat) (h : r < e) :
    (nativeExpertLayout gu d e f).downOff + r * (nativeExpertLayout gu d e f).dRow +
      (nativeExpertLayout gu d e f).dRow ≤ (nativeExpertLayout gu d e f).bytes := by
  have := row_end_le r e (iqRowBytes d f) h
  simp only [nativeExpertLayout]
  omega

/-- Gate/up grid: the row blocks cover 2 * n_ff rows. -/
theorem gu_grid_covers (nFf : Nat) : 2 * nFf ≤ 8 * ((2 * nFf + 7) / 8) := by
  omega

/-- Row slot decomposes into (is_up, r) with r in range. -/
theorem gu_row_split (row nFf : Nat) (h : row < 2 * nFf) (hu : nFf ≤ row) : row - nFf < nFf := by
  omega

-- § scratch sizing

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by
  unfold align256
  omega

/-- Alignment yields a multiple of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256
  omega

/-- Scratch: three float planes plus the q8_1 image of the intermediate. -/
def nativeExpertScratchBytes (cap nFf : Nat) : Nat :=
  3 * align256 (cap * nFf * 4) + align256 (cap * (nFf / 32) * 36)

/-- The scratch holds three full float planes. -/
theorem scratch_holds_planes (cap nFf : Nat) : 3 * (cap * nFf * 4) ≤ nativeExpertScratchBytes cap nFf := by
  have := align256_ge (cap * nFf * 4)
  unfold nativeExpertScratchBytes
  omega

-- § dequantizer thread maps

/-- iq2_xxs/xs/s dequant: thread tid writes offset 32*(tid%8)+8*(tid/8); the 8-wide slice fits in 256. -/
theorem dq8_slice_fits (tid : Nat) (h : tid < 32) : 32 * (tid % 8) + 8 * (tid / 8) + 8 ≤ 256 := by
  omega

/-- The 8-slice offsets of distinct threads differ. -/
theorem dq8_injective (a b : Nat) (ha : a < 32) (hb : b < 32)
    (h : 32 * (a % 8) + 8 * (a / 8) = 32 * (b % 8) + 8 * (b / 8)) : a = b := by
  omega

/-- q2_0 dequant: thread tid writes values 8*tid .. 8*tid+7 of the superblock. -/
theorem dq_q2_0_offset (tid : Nat) : (tid / 8) * 64 + (tid % 8) * 8 = tid * 8 := by
  omega

/-- iq4_nl dequant writes offsets 4*il+j and 4*il+16+j (j<4) inside 32 values. -/
theorem dq_iq4_nl_in_range (tid j : Nat) (h : tid < 32) (hj : j < 4) :
    4 * (tid / 8) + 16 + j < 32 := by
  omega

/-- The gate/up interleaved row slot 2r+parity is injective. -/
theorem interleave_injective (r p r' p' : Nat) (hp : p < 2) (hp' : p' < 2)
    (h : 2 * r + p = 2 * r' + p') : r = r' ∧ p = p' := by
  omega

end Strata.IqKernels
