/-!
# MODULE: lazy_expert_loader
SOURCES: ref/load.py
GGUF -> FP32 on demand with lazy per-expert and per-row dequantization, plus shard discovery arithmetic.

## Source semantics
* row_bytes = (ne0 / elems) * bytes. Expert e of a fused [in,out,experts] tensor is the contiguous run of
  per_expert/el*by bytes starting at e * (per_expert/el) * by; touching it never reads other experts.
* rows(): row r is rb bytes at r*rb. The PLE table is [160, 320001536] IQ4_NL = 26.82 GiB.
* Shard discovery requires exactly `total` siblings.

## Form template identifiers
Gguf.expert -> expert_slice_offset ; Gguf.rows -> row_slice_offset ; Gguf.row_bytes -> row_bytes ;
_bf16_to_f32 -> bf16_widen ; _discover_shards -> shard_set_check
-/
set_option maxRecDepth 100000

namespace Strata.LazyLoad

def rowBytes (ne0 el by_ : Nat) : Nat := ne0 / el * by_
def expertStart (e blocks by_ : Nat) : Nat := e * blocks * by_
def expertLen (blocks by_ : Nat) : Nat := blocks * by_

/-- Expert e's byte range ends no later than the whole tensor. -/
theorem expert_in_bounds (e n blocks by_ : Nat) (h : e < n) :
    expertStart e blocks by_ + expertLen blocks by_ ≤ n * blocks * by_ := by
  unfold expertStart expertLen
  have h1 : (e + 1) * blocks * by_ ≤ n * blocks * by_ :=
    Nat.mul_le_mul (Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl blocks)) (Nat.le_refl by_)
  have h2 : (e + 1) * blocks * by_ = e * blocks * by_ + blocks * by_ := by
    rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  omega

/-- Distinct experts read disjoint ranges: e1's range ends before e2's starts. -/
theorem experts_disjoint (e1 e2 blocks by_ : Nat) (h : e1 < e2) :
    expertStart e1 blocks by_ + expertLen blocks by_ ≤ expertStart e2 blocks by_ := by
  unfold expertStart expertLen
  have h1 : (e1 + 1) * blocks * by_ ≤ e2 * blocks * by_ :=
    Nat.mul_le_mul (Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl blocks)) (Nat.le_refl by_)
  have h2 : (e1 + 1) * blocks * by_ = e1 * blocks * by_ + blocks * by_ := by
    rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  omega

/-- Expert 0 starts at the beginning of the tensor. -/
theorem expert_zero_start (blocks by_ : Nat) : expertStart 0 blocks by_ = 0 := by
  simp [expertStart]

/-- Row r of a 2-D tensor lies inside the tensor. -/
theorem row_in_bounds (r n rb : Nat) (h : r < n) : r * rb + rb ≤ n * rb := by
  have h1 : (r + 1) * rb ≤ n * rb := Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl rb)
  have h2 : (r + 1) * rb = r * rb + rb := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Per-expert element count for the real experts is block-aligned: 2560*640 is a multiple of 64. -/
theorem expert_block_aligned : 2560 * 640 % 64 = 0 := by decide

/-- The real per-expert byte size is 460,800 (Q2_0). -/
theorem expert_bytes_q2_0 : 2560 * 640 / 64 * 18 = 460800 := by decide

/-- The PLE row of 160 IQ4_NL elements is 5 blocks of 18 bytes = 90 bytes. -/
theorem ple_row_bytes : rowBytes 160 32 18 = 90 := by decide

/-- The PLE table is within 26.82 GiB (rounded) of IQ4_NL data. -/
theorem ple_table_gib :
    2682 * 2 ^ 30 ≤ 320001536 * rowBytes 160 32 18 * 100 ∧
    320001536 * rowBytes 160 32 18 * 100 < 2683 * 2 ^ 30 := by decide

/-- 160 is a multiple of the IQ4_NL block size 32. -/
theorem ple_width_aligned : 160 % 32 = 0 := by decide

/-- bf16 widening (shift left 16) is exact. -/
theorem bf16_widen_roundtrip (h : Nat) : h * 65536 / 65536 = h := by omega

/-- bf16 widening keeps a 16-bit value inside 32 bits. -/
theorem bf16_widen_bound (h : Nat) (hh : h < 65536) : h * 65536 < 4294967296 := by omega

/-- A shard set is accepted only when the sibling count equals the declared total. -/
def shardSetOk (siblings total : Nat) : Bool := siblings == total

/-- Accepting a shard set means the counts agree. -/
theorem shardSetOk_iff (s t : Nat) : shardSetOk s t = true ↔ s = t := by
  unfold shardSetOk
  simp

/-- The suffix "-NNNNN-of-MMMMM" is 15 characters before the extension. -/
theorem shard_suffix_len : 1 + 5 + 3 + 1 + 5 = 15 := by decide

end Strata.LazyLoad
