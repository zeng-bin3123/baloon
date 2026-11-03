/-!
# MODULE: s2_gemv_q8_activation
The S2 GEMV that consumes Q8_0 activations (`s2_gemv_q8.cu`), the correctness version of the ggml contract
(`vec_dot_type(Q2_0) = Q8_0`).  One block per output row, `threads_per_row` threads splitting the quads.

## Source semantics
* Activations are `n_in/32` blocks of 34 bytes: `fp16 d` then `int8 qs[32]`.
* A quad of four elements lies inside one 32-element block, so the block scale is loaded once.
* The weight `(code - 1) * d_w` is formed first and multiplied by the dequantised activation `xq * d_x`.
* One S2 weight scale covers 64 elements = 16 quads: index `q >> 4`.

## Form template identifiers
- `QK_S2` → `qk`; `QK8_0` → `qk8`; block bytes 34 → `blk`
- `threads_per_row` → `tpr`; quad index → `q`
-/

namespace Strata.S2GemvQ8

/-- Activation block of quad `q`. -/
def actBlock (q : Nat) : Nat := (q * 4) / 32

/-- Element offset inside the activation block. -/
def actOff (q : Nat) : Nat := (q * 4) % 32

/-- The quad's four elements stay inside the block: `off + 3 < 32`. -/
theorem quad_in_block (q : Nat) : actOff q + 3 < 32 := by
  unfold actOff; omega

/-- Block and offset recompose the element index. -/
theorem block_off_recompose (q : Nat) : actBlock q * 32 + actOff q = q * 4 := by
  unfold actBlock actOff; omega

/-- Byte address of the block: stride 34. -/
def blockByte (q : Nat) : Nat := actBlock q * 34

/-- The scale `d` (2 bytes at the block start) and the data (from byte 2) are contiguous in a 34-byte block. -/
theorem block_layout : 2 + 32 = 34 := by decide

/-- Quad data start at byte `blockByte + 2 + off`; the last read byte is inside the block. -/
theorem quad_bytes_in_block (q : Nat) : 2 + actOff q + 3 < 34 := by
  unfold actOff; omega

/-- The shared 16-bit scale read `blk[0] | blk[1] << 8` at an even stride is 2-byte aligned. -/
theorem scale_aligned (q : Nat) : blockByte q % 2 = 0 := by
  unfold blockByte; omega

-- § Group index and shapes

/-- S2 group index as a shift: `(q*4) >> 6 = q >> 4`. -/
theorem group_of_quad (q : Nat) : (q * 4) / 64 = q / 16 := by omega

/-- The two 32-blocks of one 64-group share the weight scale: both quads' groups agree. -/
theorem two_blocks_one_group (q : Nat) : ((q * 4) / 32) / 2 = (q * 4) / 64 := by omega

/-- Host guard: `n_in` multiple of 32 and 64 reduces to multiple of 64. -/
theorem guard_reduces (n : Nat) : (n % 32 = 0 ∧ n % 64 = 0) ↔ n % 64 = 0 := by omega

/-- Whole quads, 16 quads per group. -/
theorem quads_per_group (n : Nat) (h : n % 64 = 0) : n / 4 = 16 * (n / 64) := by omega

/-- Number of activation blocks for `n_in`. -/
theorem act_blocks (n : Nat) (h : n % 64 = 0) : n / 32 = 2 * (n / 64) := by omega

/-- Strided ownership of quads. -/
theorem quad_owner (q tpr : Nat) (h : 0 < tpr) : q = q % tpr + tpr * (q / tpr) := by
  have := Nat.mod_add_div q tpr
  omega

/-- Every quad index below `n_in/4` lies in a valid activation block. -/
theorem block_in_range (q n : Nat) (hq : q < n / 4) (h : n % 64 = 0) : actBlock q < n / 32 := by
  unfold actBlock; omega

-- § The arithmetic: weight times dequantised activation, in the kernel's association

/-- Integer version of the per-element product `(c - 1) * x`. -/
def term (c x : Int) : Int := (c - 1) * x

/-- A quad's four terms equal the code-dot minus the plain sum of activations (the bias split used by the
grouped expert kernel). -/
theorem quad_bias_split (c0 c1 c2 c3 x0 x1 x2 x3 : Int) :
    term c0 x0 + term c1 x1 + term c2 x2 + term c3 x3 =
      (c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3) - (x0 + x1 + x2 + x3) := by
  unfold term
  simp only [Int.sub_mul, Int.one_mul]
  omega

/-- Biased code range: code 0..3 gives -1..2, so a term is bounded by `2*|x|`; with `|x| ≤ 127` it fits in 10 bits. -/
theorem term_bound (c x : Int) (hc0 : 0 ≤ c) (hc3 : c ≤ 3) (hx0 : -127 ≤ x) (hx1 : x ≤ 127) :
    -254 ≤ term c x ∧ term c x ≤ 254 := by
  unfold term
  rcases (by omega : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3) with h | h | h | h <;> subst h <;> omega

/-- The 32 products of one block sum to at most `32*254`, far below int32. -/
theorem block_sum_bound : 32 * 254 < 2 ^ 31 := by decide

/-- Launch: one block per row, `n_out` blocks, partials of `tpr` floats. -/
theorem smem_fits (tpr : Nat) (h : tpr ≤ 1024) : tpr * 4 ≤ 49152 := by omega

end Strata.S2GemvQ8
