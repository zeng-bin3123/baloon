/-!
# MODULE: s2_expert_grouped
Grouped GPU MoE expert evaluation over Q2_0 blobs: gate/up projection, SwiGLU, requantization of the
intermediate, down projection, plus device-side hit selection and grouping.

## Source semantics
A blob is `[gate/up codes | down codes | gate/up scales | down scales]`. Row slot `i` of the gate/up rows
is gate row `i/2` when even and up row `i/2` when odd, written GATE-MAJOR: every hit's gate rows first, then
every hit's up rows. A weight code `c` encodes `c - 1`; with Q8_0 activations
`sum (c-1)*x = sum c*x - sum x`, both computed by `dp4a`. One weight scale covers 64 elements (two
activation chunks). The CPU-order variant reduces eight lanes as `((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7))`.

## Form template identifiers
- `H, FF, QK, ROW_GU, ROW_D, SC_GU, SC_D` -> geometry constants
- `O_D_CODES, O_GU_SCALES, O_D_SCALES` -> blob section offsets
- `gu_kernel/down_kernel` -> gate/up and down projection, `hit_select_*` -> hit compaction
-/

namespace Strata.S2ExpertGrouped

-- § blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- Expert FFN width. -/
def FF : Nat := 640
/-- Q2_0 group size. -/
def QK : Nat := 64
/-- Code bytes per gate/up row. -/
def ROW_GU : Nat := H / 4
/-- Code bytes per down row. -/
def ROW_D : Nat := FF / 4
/-- Scales per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Scales per down row. -/
def SC_D : Nat := FF / QK
/-- Offset of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Offset of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
/-- Total blob size. -/
def BLOB_BYTES : Nat := O_D_SCALES + H * SC_D * 2

/-- Row geometry. -/
theorem row_geometry : ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 := by decide

/-- Section offsets. -/
theorem section_offsets : O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 := by
  decide

/-- Blob size. -/
theorem blob_size : BLOB_BYTES = 1382400 := by decide

/-- Sections are ordered. -/
theorem sections_ordered : O_D_CODES < O_GU_SCALES ∧ O_GU_SCALES < O_D_SCALES ∧ O_D_SCALES < BLOB_BYTES := by
  decide

/-- Code bytes equal 2 bits per weight over all three matrices: 3*FF*H/4. -/
theorem code_bytes_total : O_GU_SCALES = 3 * FF * H / 4 := by decide

/-- Scale bytes equal fp16 per 64 weights over all three matrices. -/
theorem scale_bytes_total : BLOB_BYTES - O_GU_SCALES = 3 * FF * H / QK * 2 := by decide

-- § chunk arithmetic

/-- Activation chunks per gate/up row, per down row. -/
theorem chunk_counts : H / 32 = 80 ∧ FF / 32 = 20 := by decide

/-- A weight scale covers two chunks: the scale index `c/2` is in range for gate/up rows. -/
theorem scale_index_gu (c : Nat) (h : c < 80) : c / 2 < 40 := by omega

/-- Same for down rows. -/
theorem scale_index_down (c : Nat) (h : c < 20) : c / 2 < 10 := by omega

/-- Chunks per lane in the grouped gate/up kernel: three suffice. -/
theorem gu_chunks_per_lane : (80 + 31) / 32 = 3 ∧ 80 ≤ 3 * 32 := by decide

/-- A down row (20 chunks) fits in one warp. -/
theorem down_chunks_fit : 20 ≤ 32 := by decide

/-- Block tiling of the grouped kernels is exact. -/
theorem grouped_tiling : 2 * FF / 32 * 32 = 2 * FF ∧ H / 64 * 64 = H ∧ 32 / 8 = 4 ∧ 64 / 8 = 8 := by decide

/-- Shared memory of the grouped gate/up kernel (8 entries) fits in 48 KiB. -/
theorem gu_shared_fits : 8 * (H / 4) * 4 + 8 * (H / 32) * 4 ≤ 49152 := by decide

/-- Shared memory of the grouped down kernel fits in 48 KiB. -/
theorem down_shared_fits : 8 * (FF / 4) * 4 + 8 * (FF / 32) * 4 ≤ 49152 := by decide

-- § integer identity behind the dp4a pair

/-- `(c-1) * x = c*x - x`: the weight-independent term is the activation sum. -/
theorem bias_split (c x : Int) : (c - 1) * x = c * x - x := by
  rw [Int.sub_mul, Int.one_mul]

-- § gate-major output layout

/-- Output slot of row slot `i` of hit `h` among `n` hits: gate rows first, then up rows. -/
def outIdx (n h i : Nat) : Nat :=
  if i % 2 = 1 then n * 640 + h * 640 + i / 2 else h * 640 + i / 2

/-- Every output slot is inside the `2 * n * FF` buffer. -/
theorem outIdx_lt (n h i : Nat) (hh : h < n) (hi : i < 1280) : outIdx n h i < 2 * n * 640 := by
  unfold outIdx
  split <;> omega

/-- The layout is injective: no two (hit, row slot) pairs collide. -/
theorem outIdx_injective (n h i h' i' : Nat) (hh : h < n) (hh' : h' < n) (hi : i < 1280)
    (hi' : i' < 1280) (e : outIdx n h i = outIdx n h' i') : h = h' ∧ i = i' := by
  unfold outIdx at e
  split at e <;> split at e <;> omega

/-- Gate rows of hit `h` are contiguous from `h * FF`. -/
theorem gate_contiguous (n h r : Nat) : outIdx n h (2 * r) = h * 640 + r := by
  unfold outIdx
  rw [if_neg (by omega)]
  omega

/-- Up rows of hit `h` are contiguous from `n * FF + h * FF`. -/
theorem up_contiguous (n h r : Nat) : outIdx n h (2 * r + 1) = n * 640 + h * 640 + r := by
  unfold outIdx
  rw [if_pos (by omega)]
  omega

-- § CPU-order reduction

/-- Lane-0 result of the shuffle sequence (4, then 1, then 2). -/
def lane0CpuOrder (a0 a1 a2 a3 a4 a5 a6 a7 : Int) : Int :=
  ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7))

/-- In exact arithmetic the CPU-order reduction is the plain sum. -/
theorem lane0CpuOrder_sum (a0 a1 a2 a3 a4 a5 a6 a7 : Int) :
    lane0CpuOrder a0 a1 a2 a3 a4 a5 a6 a7 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 := by
  unfold lane0CpuOrder
  omega

-- § scratch sizing

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Alignment yields multiples of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

/-- Hit-add grid width never exceeds 8. -/
theorem hit_add_grid_le (n : Nat) : min ((n + 255) / 256) 8 ≤ 8 := by omega

end Strata.S2ExpertGrouped
