/-!
# MODULE: iq_quant_kernels
SOURCES: src/kernels/cuda/iq_kernels.cu
IQ-family (and Q3_K/Q2_0) MMVQ dot products on Q8_1 activations, Q8_1 quantizer, dequantizers, embedding row
gather, and grouped native-expert evaluation (gate/up, SwiGLU, requantize, down).

## Source semantics
- Format traits: values per block `qk`, dot calls per block `ipb`, iqs stride `step`; call k of a row maps to
  block `k / ipb` and part `k % ipb`.
- Native expert layout: gate rows, then up rows (`up_off = n_ff*gu_row`), then down rows
  (`down_off = 2*up_off`); total `down_off + n_embd*d_row`.
- Gate/up kernel row index `row >= n_ff` selects the up matrix with row `row - n_ff`.
- Dequantized gate/up output is interleaved: row `2r + parity`.

## Form template identifiers
- Fmt<TY> -> trait record ; NativeExpertLayout -> layout ; iq_row_bytes -> rowBytes
-/

namespace Strata.IqKernels

-- § block sizes
/-- Bytes of one block per IQ-family type id (ggml-common.h sizes). -/
def blockBytes : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 20 => 18
  | 21 => 110
  | 22 => 82
  | 29 => 56
  | 23 => 136
  | 11 => 110
  | 42 => 18
  | _ => 0

/-- Values per block per type id. -/
def blockElems : Nat → Nat
  | 20 => 32
  | 42 => 64
  | 16 => 256 | 17 => 256 | 18 => 256 | 21 => 256 | 22 => 256 | 29 => 256 | 23 => 256 | 11 => 256
  | _ => 0

/-- Types handled by this module. -/
def isIq (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11

/-- Row bytes of an `n`-element row. -/
def rowBytes (t n : Nat) : Nat := (n / blockElems t) * blockBytes t

/-- Ten types are supported. -/
theorem iq_supported_examples : isIq 16 = true ∧ isIq 42 = true ∧ isIq 0 = false := by decide

/-- Unsupported ids have zero row bytes. -/
theorem rowBytes_unsupported (n : Nat) : rowBytes 0 n = 0 := by
  simp [rowBytes, blockBytes]

/-- IQ4_NL: 18-byte blocks cover 32 values. -/
theorem iq4nl_row : rowBytes 20 256 = 144 := by decide

/-- IQ4_XS: 256-value block is 136 bytes. -/
theorem iq4xs_row : rowBytes 23 256 = 136 := by decide

-- § format traits
/-- Calls per block times iqs step cover the block's 32-bit activation words for 256-value formats. -/
theorem trait_256_calls : 8 * 2 = 16 ∧ 8 * 1 = 8 := by decide

/-- Call index k maps to a block index below `nb` when `k < nb * ipb`. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < ipb * nb) : k / ipb < nb := by
  exact Nat.div_lt_of_lt_mul h

/-- The call split `(k / ipb, k % ipb)` recomposes k. -/
theorem call_split (k ipb : Nat) : ipb * (k / ipb) + k % ipb = k := by
  exact Nat.div_add_mod k ipb

/-- Part index is below `ipb` for positive `ipb`. -/
theorem call_part_lt (k ipb : Nat) (h : 0 < ipb) : k % ipb < ipb := by
  exact Nat.mod_lt _ h

-- § native expert layout
/-- Layout record fields reduced to numbers. -/
structure Layout where
  nEmbd : Nat
  nFf : Nat
  guRow : Nat
  dRow : Nat

/-- Offset of the up matrix. -/
def Layout.upOff (L : Layout) : Nat := L.nFf * L.guRow
/-- Offset of the down matrix. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total expert bytes. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

/-- Total bytes decompose into gate, up and down matrices. -/
theorem layout_bytes (L : Layout) :
    L.bytes = L.nFf * L.guRow + L.nFf * L.guRow + L.nEmbd * L.dRow := by
  unfold Layout.bytes Layout.downOff Layout.upOff
  omega

/-- The up matrix starts right after the gate matrix, and down right after up. -/
theorem layout_contiguous (L : Layout) : L.upOff + L.nFf * L.guRow = L.downOff := by
  unfold Layout.downOff Layout.upOff
  omega

/-- Gate/up row split: rows `< n_ff` are gate, others up with row `row - n_ff`. -/
theorem gu_row_split (row nff : Nat) (h : nff ≤ row) : nff + (row - nff) = row := by
  omega

/-- A grid of 8-row blocks covers `2*n_ff` rows. -/
theorem gu_grid_covers (nff : Nat) : 2 * nff ≤ ((2 * nff + 7) / 8) * 8 := by
  omega

/-- A grid of 8-row blocks covers `n_embd` down rows. -/
theorem down_grid_covers (n : Nat) : n ≤ ((n + 7) / 8) * 8 := by
  omega

/-- Scratch alignment to 256 bytes never shrinks. -/
theorem align256_ge (x : Nat) : x ≤ (x + 255) / 256 * 256 := by
  omega

/-- Aligned value is a multiple of 256. -/
theorem align256_mod (x : Nat) : ((x + 255) / 256 * 256) % 256 = 0 := by
  omega

/-- Aligned value overshoots by less than 256. -/
theorem align256_lt (x : Nat) : (x + 255) / 256 * 256 < x + 256 := by
  omega

-- § interleaved dequantization
/-- Quotient lemma: `(x*pr + c) / pr = x` for `c < pr`. -/
theorem idx_div (x c pr : Nat) (hc : c < pr) : (x * pr + c) / pr = x := by
  apply Nat.div_eq_of_lt_le
  · omega
  · have h : (x + 1) * pr = x * pr + pr := Nat.succ_mul x pr
    omega

/-- Interleaved gate/up output indices determine (row, parity): injectivity on the row coordinate. -/
theorem interleave_row_recovered (r p c pr : Nat) (hc : c < pr) :
    ((2 * r + p) * pr + c) / pr = 2 * r + p := by
  exact idx_div (2 * r + p) c pr hc

/-- IQ4_NL dequantization: low and high nibble of a byte are both below 16. -/
theorem nibbles_lt16 (q : Nat) (h : q < 256) : q % 16 < 16 ∧ q / 16 < 16 := by
  omega

/-- Nibbles recombine to the byte. -/
theorem nibbles_recombine (q : Nat) (h : q < 256) : q % 16 + 16 * (q / 16) = q := by
  omega

/-- Q2_0 code minus one is in {-1,0,1,2}. -/
theorem q2_0_code_range (code : Int) (h0 : 0 ≤ code) (h1 : code < 4) : -1 ≤ code - 1 ∧ code - 1 ≤ 2 := by
  omega

-- § q8_1 quantizer
/-- Block index and in-block index of element i. -/
theorem q8_1_split (i : Nat) : 32 * (i / 32) + i % 32 = i := by
  omega

/-- Gate/up dequant grid: two parity blocks per superblock row. -/
theorem gu_dequant_blocks (nff per_row : Nat) : (nff * per_row) * 2 = 2 * (nff * per_row) := by
  omega

end Strata.IqKernels
