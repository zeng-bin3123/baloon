/-!
# MODULE: iq_codebook_kernels
SOURCES: src/kernels/cuda/iq_kernels.cu
Warp-per-row MMVQ, grouped native MoE experts, q8_1 quantizer and dequantizers for the i-quant family
(IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_M, IQ4_NL, IQ4_XS, Q3_K, Q2_0), transcribed from llama.cpp.

## Source semantics
Every format has a block geometry (elements per block, bytes per block). A row of n values occupies
(n / qk) * sizeof(block) bytes. A native expert blob is [gate rows | up rows | down rows]; gate/up use the
gate/up type and the down rows use the down type. Scratch regions are 256-byte aligned. dp4a is a 4-lane
int8 dot-product-accumulate; adding a constant to codes is corrected with a dp4a against all-ones.

## Form template identifiers
blockBytes/blockElems -> abstract "format geometry"; mkLayout -> "expert blob layout";
align256 -> "scratch alignment"; dp4/dp4a -> "packed int8 dot".
-/
namespace Strata.IqKernels

-- § block geometry
/-- sizeof of the ggml block structs (QK_K = 256), keyed by ggml type id; 0 when unsupported. -/
def blockBytes (t : Nat) : Nat :=
  if t = 16 then 66 else if t = 17 then 74 else if t = 18 then 98 else if t = 20 then 18
  else if t = 21 then 110 else if t = 22 then 82 else if t = 23 then 136 else if t = 29 then 56
  else if t = 11 then 110 else if t = 42 then 18 else 0

/-- Values per block. -/
def blockElems (t : Nat) : Nat :=
  if t = 20 then 32 else if t = 42 then 64 else
  if t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 23 ∨ t = 29 ∨ t = 11 then 256 else 0

/-- `is_iq`: the type ids this file handles. -/
def isIq (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11

/-- `iq_row_bytes`. -/
def rowBytes (t n : Nat) : Nat := n / blockElems t * blockBytes t

/-- Extending a row by one block adds exactly one block of bytes. -/
theorem rowBytes_add_block (t n : Nat) (h : 0 < blockElems t) :
    rowBytes t (n + blockElems t) = rowBytes t n + blockBytes t := by
  unfold rowBytes
  rw [Nat.add_div_right _ h, Nat.add_mul, Nat.one_mul]

/-- IQ2_XXS: 256 values occupy one 66-byte block. -/
theorem rowBytes_iq2_xxs_256 : rowBytes 16 256 = 66 := by
  decide

/-- Unsupported ids (below 64) have zero row bytes, mirroring the `default: return 0` arm. -/
theorem rowBytes_unsupported (t n : Nat) (ht : t < 64) (h : isIq t = false) : rowBytes t n = 0 := by
  have hb : blockBytes t = 0 :=
    (by decide : ∀ t, t < 64 → isIq t = false → blockBytes t = 0) t ht h
  simp [rowBytes, hb]

-- § per-format dot-call traits
/-- Calls-per-block for a format (`Fmt<TY>::ipb`). -/
def fmtIpb (t : Nat) : Nat := if t = 20 ∨ t = 42 then 2 else 8

/-- The `iqs` stride between calls (`Fmt<TY>::step`). -/
def fmtStep (t : Nat) : Nat := if t = 29 ∨ t = 42 then 1 else 2

/-- The `Fmt` block size agrees with the geometry table for every kernel-dispatched format. -/
theorem fmt_qk_matches (t : Nat) (ht : t < 64) (h : isIq t = true) (h1 : t ≠ 11) (h2 : t ≠ 23) :
    blockElems t = (if t = 20 then 32 else if t = 42 then 64 else 256) := by
  revert t
  decide

/-- For the 256-wide formats stepping by 2, every call's q8_1 block index `iqs/2` is inside the block. -/
theorem fmt_q8_index_in_range (t : Nat) (ht : t < 64) (h : t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22) :
    ∀ k, k < fmtIpb t → (fmtStep t * k) / 2 < blockElems t / 32 := by
  revert t
  decide

/-- IQ1_M and Q2_0 index q8_1 blocks directly by `iqs`, which stays inside the block. -/
theorem fmt_q8_direct_index (t : Nat) (ht : t < 64) (h : t = 29 ∨ t = 42) :
    ∀ k, k < fmtIpb t → fmtStep t * k < blockElems t / 32 := by
  revert t
  decide

-- § expert blob layout
/-- `NativeExpertLayout`. -/
structure Layout where
  guRow : Nat
  dRow : Nat
  upOff : Nat
  downOff : Nat
  bytes : Nat

/-- `native_expert_layout`. -/
def mkLayout (gu d ne nf : Nat) : Layout :=
  { guRow := rowBytes gu ne, dRow := rowBytes d nf,
    upOff := nf * rowBytes gu ne, downOff := 2 * (nf * rowBytes gu ne),
    bytes := 2 * (nf * rowBytes gu ne) + ne * rowBytes d nf }

/-- The down block starts after both gate and up regions. -/
theorem layout_down_off (gu d ne nf : Nat) :
    (mkLayout gu d ne nf).downOff = 2 * (mkLayout gu d ne nf).upOff := by
  rfl

/-- Total blob size is the down offset plus the down rows. -/
theorem layout_bytes (gu d ne nf : Nat) :
    (mkLayout gu d ne nf).bytes = (mkLayout gu d ne nf).downOff + ne * (mkLayout gu d ne nf).dRow := by
  rfl

/-- Every up row (index r < n_ff) ends before the down region begins. -/
theorem layout_up_row_fits (gu d ne nf r : Nat) (h : r < nf) :
    (mkLayout gu d ne nf).upOff + r * (mkLayout gu d ne nf).guRow + (mkLayout gu d ne nf).guRow
      ≤ (mkLayout gu d ne nf).downOff := by
  have h1 := Nat.mul_le_mul_right (rowBytes gu ne) (show r + 1 ≤ nf by omega)
  have h2 : r * rowBytes gu ne + rowBytes gu ne ≤ nf * rowBytes gu ne := by
    simpa [Nat.add_mul] using h1
  show nf * rowBytes gu ne + r * rowBytes gu ne + rowBytes gu ne ≤ 2 * (nf * rowBytes gu ne)
  omega

-- § scratch sizing
/-- Round up to a multiple of 256. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by
  unfold align256
  omega

/-- Alignment yields a multiple of 256. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by
  unfold align256
  omega

/-- Alignment wastes fewer than 256 bytes. -/
theorem align256_lt (n : Nat) : align256 n < n + 256 := by
  unfold align256
  omega

/-- `native_expert_scratch_bytes` (sizeof block_q8_1 = 36). -/
def scratchBytes (cap nf : Nat) : Nat :=
  3 * align256 (cap * nf * 4) + align256 (cap * (nf / 32) * 36)

/-- The scratch size is itself a multiple of 256, so the carved regions stay aligned. -/
theorem scratch_aligned (cap nf : Nat) : scratchBytes cap nf % 256 = 0 := by
  unfold scratchBytes align256
  omega

-- § packed int8 dot
/-- `__dp4a` without accumulator. -/
def dp4 (a b : Fin 4 → Int) : Int := a 0 * b 0 + a 1 * b 1 + a 2 * b 2 + a 3 * b 3

/-- `__dp4a(a, b, c)`. -/
def dp4a (a b : Fin 4 → Int) (c : Int) : Int := c + dp4 a b

/-- The accumulator only shifts the result. -/
theorem dp4a_acc (a b : Fin 4 → Int) (c : Int) : dp4a a b c = c + dp4a a b 0 := by
  unfold dp4a
  omega

/-- Dotting against 0x01010101 sums the lanes (the `sumy` trick of IQ1_M). -/
theorem dp4_ones (b : Fin 4 → Int) : dp4 (fun _ => 1) b = b 0 + b 1 + b 2 + b 3 := by
  unfold dp4
  simp

/-- A constant bias on the codes is corrected by a multiple of the lane sum. -/
theorem dp4_sub_const (a b : Fin 4 → Int) (k : Int) :
    dp4 (fun i => a i - k) b = dp4 a b - k * (b 0 + b 1 + b 2 + b 3) := by
  unfold dp4
  simp only [Int.sub_mul, Int.mul_add]
  omega

end Strata.IqKernels
