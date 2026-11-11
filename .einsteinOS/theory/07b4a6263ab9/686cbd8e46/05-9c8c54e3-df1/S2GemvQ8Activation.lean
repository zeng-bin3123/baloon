/-!
# MODULE: s2_gemv_q8_activation
SOURCES: src/kernels/cuda/s2_gemv_q8.cu
The S2 GEMV that consumes Q8_0-quantized activations, as `ggml_mul_mat` does for Q2_0 (`vec_dot_type = Q8_0`).
One block per row, one quad per thread-iteration; each activation is dequantized `xq · dx` with `dx` the
block's fp16 scale.  This module proves the activation quad lies in one Q8_0 block (so one scale load is
right), the fp16 scale bytes are assembled little-endian, the weight decode matches the plane, the exact
per-element term, and the host gates (including the unchecked `threads_per_row`).

## Source semantics
* Quad `q`: weight byte `c[q]`, group `q >> 4`; activation block `ablk = 4q / 32`, byte offset `34·ablk`,
  `dbits = blk[0] | blk[1] << 8`, quants at `blk + 2 + off`, `off = 4q % 32`.
* Term: `a_k += w_k · (xq_k · dx)` with `w_k = (code_k - 1) · d`.
* Host: `n_in % 32 ≠ 0 ∨ n_in % 64 ≠ 0` → exit; `threads_per_row` is NOT validated (same halving tree as
  `s2_gemv_quads`); synchronous only when `stream == nullptr`.

## Form template identifiers
`s2_gemv_q8_kernel` → quad model; `ablk` → `actBlock`; `off` → `actOff`; `dbits` → `le16`;
`QK8_0` → 32; `QK_S2` → 64; host gate → `hostOk`; tree → `halvingTree`.
-/

namespace Strata.S2GemvQ8Activation

-- § activation addressing

/-- `ablk = (4q) / 32`. -/
def actBlock (q : Nat) : Nat := 4 * q / 32

/-- `off = (4q) % 32`. -/
def actOff (q : Nat) : Nat := 4 * q % 32

/-- **One block per quad**: all four activations of quad `q` are elements `off … off+3` of block `ablk`. -/
theorem quad_in_one_block (q k : Nat) (hk : k < 4) :
    (4 * q + k) / 32 = actBlock q ∧ (4 * q + k) % 32 = actOff q + k ∧ actOff q + 3 < 32 := by
  unfold actBlock actOff; omega

/-- A quad's activation block is `q / 8` and its offset `4·(q % 8)`. -/
theorem actBlock_simple (q : Nat) : actBlock q = q / 8 ∧ actOff q = 4 * (q % 8) := by
  unfold actBlock actOff; omega

/-- Block byte addresses: `d` at `34·ablk`, quant `k` at `34·ablk + 2 + off + k`, inside the 34-byte block. -/
theorem quant_byte_in_block (q k : Nat) (hk : k < 4) :
    34 * actBlock q + 2 + actOff q + k < 34 * (actBlock q + 1) := by
  have := (quad_in_one_block q k hk).2.2
  omega

/-- The little-endian fp16 assembly `b0 | b1 << 8`. -/
def le16 (b0 b1 : Nat) : Nat := b0 ||| (b1 <<< 8)

/-- **`dbits` is the little-endian value** `b0 + 256·b1` for bytes `b0 < 256`. -/
theorem le16_eq (b0 b1 : Nat) (h0 : b0 < 256) : le16 b0 b1 = b0 + 256 * b1 := by
  unfold le16
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt (by simpa using h0), Nat.shiftLeft_eq]
  omega

-- § weights

/-- Plane code of element `e`. -/
def planeCode (byte : Nat → Nat) (e : Nat) : Nat := (byte (e / 4) >>> ((e % 4) * 2)) &&& 3

/-- The quad's weight codes are plane elements `4q … 4q+3`. -/
theorem weight_code (byte : Nat → Nat) (q k : Nat) (hk : k < 4) :
    (byte q >>> (k * 2)) &&& 3 = planeCode byte (4 * q + k) := by
  unfold planeCode
  rw [show (4 * q + k) / 4 = q by omega, show (4 * q + k) % 4 = k by omega]

/-- The weight group and the activation block are both constant across the quad, and a weight group
(64 elements) spans exactly two activation blocks (32 elements). -/
theorem group_spans_two_blocks (q : Nat) : (q >>> 4) = actBlock q / 2 := by
  unfold actBlock; rw [Nat.shiftRight_eq_div_pow]; simp; omega

-- § the term

/-- `w · (xq · dx)` with `w = (code - 1)·d` (exact model). -/
def term (code : Nat) (d : Int) (xq : Int) (dx : Int) : Int := (((code : Int) - 1) * d) * (xq * dx)

/-- Exactly, the term is the product of the dequantized weight and the dequantized activation, so the kernel
agrees with "dequantize both, then multiply"; only FP rounding of the grouping differs. -/
theorem term_eq (code : Nat) (d xq dx : Int) :
    term code d xq dx = ((code : Int) - 1) * d * xq * dx := by
  unfold term; rw [Int.mul_assoc (((code : Int) - 1) * d) xq dx]

/-- Code 1 contributes nothing regardless of the activation. -/
theorem term_code_one (d xq dx : Int) : term 1 d xq dx = 0 := by simp [term]

-- § host gates

/-- `s2_gemv_q8`'s `n_in` gate. -/
def hostOk (nIn : Nat) : Bool := 0 < nIn && nIn % 32 == 0 && nIn % 64 == 0

/-- The two checks collapse to `% 64` (the error message prints 64 for both). -/
theorem gate_is_mod64 (nIn : Nat) : hostOk nIn = (0 < nIn && nIn % 64 == 0) := by
  unfold hostOk
  by_cases h : nIn % 64 = 0
  · have : nIn % 32 = 0 := by omega
    simp [h, this]
  · have hb : (nIn % 64 == 0) = false := by simpa using h
    rw [hb]; simp

/-- Under the gate every quad's activation block exists: `ablk < n_in / 32`. -/
theorem blocks_in_range (nIn q : Nat) (h : hostOk nIn = true) (hq : q < nIn / 4) : actBlock q < nIn / 32 := by
  simp [hostOk] at h; unfold actBlock; omega

/-- One tree level. -/
def treeStep (a : Nat → Int) (off : Nat) : Nat → Int :=
  fun t => if t < off then a t + a (t + off) else a t

/-- The halving loop over `tpr` partials. -/
def halvingGo (a : Nat → Int) : Nat → Nat → Int
  | 0, _ => a 0
  | fuel + 1, step => if step = 0 then a 0 else halvingGo (treeStep a step) fuel (step / 2)

/-- The kernel's reduction. -/
def halvingTree (a : Nat → Int) (tpr : Nat) : Int := halvingGo a 64 (tpr / 2)

/-- **Unchecked `threads_per_row`**: a non-power-of-two silently drops partial sums (here 48 threads keep 32). -/
theorem tpr_unchecked_drops : halvingTree (fun _ => 1) 48 = 32 ∧ halvingTree (fun _ => 1) 64 = 64 := by decide

end Strata.S2GemvQ8Activation
