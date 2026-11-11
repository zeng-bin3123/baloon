/-!
# MODULE: s2_gemv_q8_activation
SOURCES: src/kernels/cuda/s2_gemv_q8.cu
The S2 GEMV that consumes ggml Q8_0 activation blocks (the `vec_dot_type` of Q2_0), so GPU and CPU
expert paths compute the same activation contract: one block per row, threads over quads, each quad's
four int8 activations read from one 34-byte block with one fp16 block scale.

## Source semantics
* Quad `q`: weight byte `c[q]`, group `q >> 4`; activation block `ablk = 4q / 32` at byte `34·ablk`,
  scale bytes `blk[0] | blk[1] << 8`, int8 values `blk[2 + off + k]` with `off = 4q % 32`.
* All four elements of a quad lie in the same 32-element activation block, so the block scale is loaded
  once per quad (exact, not an approximation).
* Element value `((code_k − 1)·d) · (xq·dx)`: the activation is dequantised, then multiplied — the same
  float expression as `s2_gemv_quads`.
* Host: `n_in, n_out > 0` else no-op; `n_in % 32 = 0 ∧ n_in % 64 = 0` (the first implied by the second);
  `threads_per_row` is not validated (see `S2GemvQuads.non_pow2_tpr_drops`).

## Form template identifiers
* `QK_S2 = 64`, `QK8_0 = 32`, block = 34 bytes
* `actBlock`, `actOff`, `actByte` ↔ `ablk`, `off`, `blk + 2 + off + k`
-/

namespace Strata.S2Q8

-- § quad ↔ activation block

/-- Activation block of quad `q`. -/
def actBlock (q : Nat) : Nat := 4 * q / 32

/-- Offset of quad `q` inside its activation block. -/
def actOff (q : Nat) : Nat := 4 * q % 32

/-- Byte of element `4q + k` in the activation buffer. -/
def actByte (q k : Nat) : Nat := 34 * actBlock q + 2 + actOff q + k

/-- **A quad never straddles an activation block**: elements `4q … 4q+3` share block `ablk` and sit at
offsets `off … off+3 < 32` — so one scale load per quad is exact. -/
theorem quad_in_one_block (q k : Nat) (hk : k < 4) :
    (4 * q + k) / 32 = actBlock q ∧ (4 * q + k) % 32 = actOff q + k ∧ actOff q + k < 32 := by
  unfold actBlock actOff; omega

/-- The int8 read stays inside the block's `qs[32]`, after its 2-byte scale. -/
theorem act_read_in_block (q k : Nat) (hk : k < 4) :
    34 * actBlock q + 2 ≤ actByte q k ∧ actByte q k < 34 * actBlock q + 34 := by
  have := quad_in_one_block q k hk
  unfold actByte; omega

/-- Every element of a row maps to a byte inside the `n_in/32` blocks. -/
theorem act_read_in_buffer (n_in q k : Nat) (hn : n_in % 32 = 0) (hk : k < 4) (hq : 4 * q + k < n_in) :
    actByte q k < 34 * (n_in / 32) := by
  unfold actByte actBlock actOff; omega

/-- The element addressed is the plane-order element `4q + k`: block-major, offset-minor. -/
theorem act_matches_element (q k : Nat) (hk : k < 4) :
    32 * actBlock q + actOff q + k = 4 * q + k := by
  unfold actBlock actOff; omega

-- § scale bytes and weight group

/-- `blk[0] | (blk[1] << 8)` is the little-endian 16-bit scale: OR is addition for a byte low part. -/
theorem scale_bytes (lo hi : Nat) (hlo : lo < 256) : lo ||| (hi <<< 8) = lo + 256 * hi := by
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt (by simpa using hlo), Nat.shiftLeft_eq]
  simp; omega

/-- The weight group of quad `q` is `q >> 4 = 4q / 64`: one S2 scale serves four activation blocks'
worth of a quad stream, and a quad never straddles an S2 group either. -/
theorem weight_group (q k : Nat) (hk : k < 4) : (4 * q + k) / 64 = q >>> 4 := by
  rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- The S2 group (64) is exactly two activation blocks (32): group boundaries are block boundaries. -/
theorem group_block_nesting (q : Nat) : actBlock q / 2 = q >>> 4 := by
  unfold actBlock; rw [Nat.shiftRight_eq_div_pow]; simp; omega

-- § exact-arithmetic contract

/-- In exact arithmetic the kernel's per-element expression equals the product of dequantised weight
and dequantised activation, whichever way it is associated; only FP32 rounding distinguishes them. -/
theorem element_value (c d xq dx : Int) : ((c - 1) * d) * (xq * dx) = (c - 1) * xq * (d * dx) := by
  simp only [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]

-- § host guard

/-- The `% 32` half of the guard is implied by the `% 64` half. -/
theorem guard_redundant (n : Nat) (h : n % 64 = 0) : n % 32 = 0 := by omega

/-- Accepted shapes have whole quads, whole activation blocks and whole S2 groups. -/
theorem accepted_shape (n : Nat) (h : n % 64 = 0) :
    4 * (n / 4) = n ∧ 32 * (n / 32) = n ∧ 64 * (n / 64) = n := by omega

end Strata.S2Q8
