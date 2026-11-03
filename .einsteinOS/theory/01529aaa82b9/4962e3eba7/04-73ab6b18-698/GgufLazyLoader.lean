/-!
# MODULE: gguf_lazy_loader
Lazy demand-paged GGUF loading (ref/load.py): per-expert slicing of a fused `[in, out, experts]`
tensor, per-row gathering, bf16 widening. Byte-offset arithmetic only.

## Source semantics
Expert e of a fused tensor is the contiguous run of `in*out` elements at element offset `e*in*out`;
its byte start is `e * (per_expert / el) * by`. Row r of a 2-D tensor starts at `r * row_bytes`.
Neither touches any other expert or row. bf16 widens by a 16-bit left shift.

## Form template identifiers
expertLen -> per_expert_bytes ; expertStart -> start ; rowStart -> row * rb ; bf16Bits -> bf16_to_f32
-/

set_option maxRecDepth 100000

namespace Strata.Loader

/-- Bytes per expert: `per_expert // el * by`. -/
def expertLen (per el b : Nat) : Nat := per / el * b

/-- Byte start of expert e, written as in the Python: `e * (per // el) * by`. -/
def expertStartPy (e per el b : Nat) : Nat := e * (per / el) * b

/-- Byte start as `e * len`. -/
def expertStart (e per el b : Nat) : Nat := e * expertLen per el b

/-- The Python expression equals `e * len`. -/
theorem start_eq (e per el b : Nat) : expertStartPy e per el b = expertStart e per el b := by
  unfold expertStartPy expertStart expertLen
  exact Nat.mul_assoc _ _ _

/-- Expert e ends exactly where expert e+1 starts: runs are adjacent. -/
theorem adjacent (e len : Nat) : e * len + len = (e + 1) * len := by
  rw [Nat.add_mul, Nat.one_mul]

/-- Expert e fits inside the tensor of n experts. -/
theorem expert_in_bounds (e n len : Nat) (h : e < n) : e * len + len ≤ n * len := by
  have h1 : (e + 1) * len ≤ n * len := Nat.mul_le_mul_right len h
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- Two different experts never overlap: the earlier one ends before the later one starts. -/
theorem experts_disjoint (e1 e2 len : Nat) (h : e1 < e2) : e1 * len + len ≤ e2 * len := by
  have h1 : (e1 + 1) * len ≤ e2 * len := Nat.mul_le_mul_right len h
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- When the element count per expert is a multiple of the block size, the whole tensor is
n times one expert. -/
theorem total_bytes (n per el b : Nat) (h : el ∣ per) :
    n * per / el * b = n * expertLen per el b := by
  unfold expertLen
  rw [Nat.mul_div_assoc n h]
  exact Nat.mul_assoc _ _ _

/-- Real routed expert: 2560 x 640 Q2_0 (64 per 18 B) is 460,800 bytes. -/
theorem q2_expert_len : expertLen (2560 * 640) 64 18 = 460800 := by decide

/-- Per-expert element count is a multiple of the block size. -/
theorem q2_expert_aligned : (2560 * 640) % 64 = 0 := by decide

/-- 512 fused experts: 235,929,600 bytes. -/
theorem q2_fused_total : 512 * 460800 = 235929600 := by decide

-- § Rows

/-- Byte start of row r. -/
def rowStart (r rb : Nat) : Nat := r * rb

/-- A gathered row lies inside the tensor. -/
theorem row_in_bounds (r ncols rb : Nat) (h : r < ncols) : rowStart r rb + rb ≤ ncols * rb := by
  unfold rowStart
  exact expert_in_bounds r ncols rb h

/-- PLE table: width 160 IQ4_NL gives 90-byte rows. -/
theorem ple_row : 160 / 32 * 18 = 90 := by decide

/-- PLE table size in bytes: 28.8e9, i.e. 26.82 GiB (between 26 and 27 GiB). -/
theorem ple_gib : 26 * 1073741824 < 320001536 * 90 ∧ 320001536 * 90 < 27 * 1073741824 := by decide

/-- Decoding the whole table to fp32 would take about 205 GB. -/
theorem ple_full_fp32 : 320001536 * 160 * 4 = 204800983040 := by decide

/-- A token reads 16 rows of 160 = 2560 floats. -/
theorem ple_token_floats : 16 * 160 = 2560 := rfl

-- § bf16

/-- bf16 to f32 bits: shift left by 16. -/
def bf16Bits (h : Nat) : Nat := h * 65536

/-- The result fits in 32 bits. -/
theorem bf16_fits (h : Nat) (hh : h < 65536) : bf16Bits h < 4294967296 := by
  unfold bf16Bits; omega

/-- The low 16 bits are zero. -/
theorem bf16_low_zero (h : Nat) : bf16Bits h % 65536 = 0 := by
  unfold bf16Bits; omega

/-- The shift is injective: the high half recovers the input. -/
theorem bf16_recover (h : Nat) : bf16Bits h / 65536 = h := by
  unfold bf16Bits; omega

/-- Index range check mirrors `rows()`: valid exactly when the index lies in [0, n). -/
theorem idx_valid (i : Int) (n : Int) : (0 ≤ i ∧ i < n) ↔ ¬ (i < 0 ∨ n ≤ i) := by omega

end Strata.Loader
