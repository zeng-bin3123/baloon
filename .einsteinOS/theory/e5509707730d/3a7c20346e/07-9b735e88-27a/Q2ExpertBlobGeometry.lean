/-!
# MODULE: q2_expert_blob_geometry
SOURCES: include/strata/kernels/cpu/expert.hpp, src/kernels/cpu/expert.cpp
Fixed geometry of one Q2_0 routed expert (H=2560, FF=640, QK=64, QKA=32) and the internal layout of its
1,382,400-byte blob: gate/up codes, down codes, gate/up scales, down scales.

## Source semantics
* One weight block is QK=64 weights with one fp16 scale (18 bytes); one activation chunk is QKA=32 elements, so
  a weight block spans exactly two activation chunks.
* Gate and up rows are interleaved (row 2r = gate_r, row 2r+1 = up_r); regions are contiguous and tile the blob.

## Form template identifiers
H -> hidden_width, FF -> expert_inner_width, QK -> weight_block, QKA -> activation_chunk, BB -> block_bytes,
BLOB -> expert_blob_bytes, O_* -> region_offset, ROW_* -> code_bytes_per_row, SC_* -> scale_blocks_per_row
-/

namespace Strata.Q2ExpertGeometry

-- § constants
def H : Nat := 2560
def FF : Nat := 640
def NE : Nat := 512
def QK : Nat := 64
def QKA : Nat := 32
def BB : Nat := 18
def MAXC : Nat := H / QKA
def ROW_GU : Nat := H * 2 / 8
def ROW_D : Nat := FF * 2 / 8
def SC_GU : Nat := H / QK
def SC_D : Nat := FF / QK
def BLOB : Nat := 3 * (H * FF * BB / QK)
def O_GU_CODES : Nat := 0
def O_D_CODES : Nat := 2 * FF * ROW_GU
def O_GU_SCALES : Nat := O_D_CODES + 1 * H * ROW_D
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2

-- § numeric values
/-- `H` is 2560. -/
theorem H_eq : H = 2560 := rfl
/-- `FF` is 640. -/
theorem FF_eq : FF = 640 := rfl
/-- `MAXC` is 80 activation chunks. -/
theorem MAXC_eq : MAXC = 80 := by decide
/-- Gate/up code bytes per row. -/
theorem ROW_GU_eq : ROW_GU = 640 := by decide
/-- Down code bytes per row. -/
theorem ROW_D_eq : ROW_D = 160 := by decide
/-- Weight blocks per gate/up row. -/
theorem SC_GU_eq : SC_GU = 40 := by decide
/-- Weight blocks per down row. -/
theorem SC_D_eq : SC_D = 10 := by decide
/-- The blob is 1,382,400 bytes. -/
theorem BLOB_eq : BLOB = 1382400 := by decide
/-- Offset of the down codes. -/
theorem O_D_CODES_eq : O_D_CODES = 819200 := by decide
/-- Offset of the gate/up scales. -/
theorem O_GU_SCALES_eq : O_GU_SCALES = 1228800 := by decide
/-- Offset of the down scales. -/
theorem O_D_SCALES_eq : O_D_SCALES = 1331200 := by decide

-- § block / chunk contract
/-- A Q2_0 block is 2 bytes of fp16 scale plus 16 code bytes (64 codes at 2 bits). -/
theorem block_bytes_decomposition : BB = 2 + QK * 2 / 8 := by decide
/-- A weight block spans exactly two activation chunks. -/
theorem weight_block_is_two_chunks : QK = 2 * QKA := by decide
/-- The hidden width is a whole number of activation chunks, and `MAXC` counts them. -/
theorem hidden_is_chunks : H = MAXC * QKA := by decide
/-- Chunk count of a row equals twice its weight-block count (gate/up). -/
theorem gu_chunks_twice_blocks : H / QKA = 2 * SC_GU := by decide
/-- Chunk count of the intermediate equals twice the down weight-block count. -/
theorem d_chunks_twice_blocks : FF / QKA = 2 * SC_D := by decide
/-- The intermediate activation fits in the `MAXC`-chunk scratch. -/
theorem intermediate_fits_scratch : FF / QKA ≤ MAXC := by decide
/-- Code bytes per row are 16 per weight block. -/
theorem code_bytes_per_block : ROW_GU = SC_GU * 16 ∧ ROW_D = SC_D * 16 := by decide

-- § region layout
/-- Regions are ordered and the last one ends exactly at `BLOB`. -/
theorem regions_ordered_and_tile :
    O_GU_CODES ≤ O_D_CODES ∧ O_D_CODES ≤ O_GU_SCALES ∧ O_GU_SCALES ≤ O_D_SCALES ∧
      O_D_SCALES + 1 * H * SC_D * 2 = BLOB := by decide

/-- The code regions hold 3 projections of 2-bit codes, and the scales hold one fp16 per 64 weights. -/
theorem code_and_scale_totals :
    O_GU_SCALES = 3 * (H * FF * 2 / 8) ∧ BLOB - O_GU_SCALES = 3 * (H * FF / QK * 2) := by decide

/-- Every gate and up row lies inside the gate/up code region. -/
theorem gu_rows_in_codes (r : Nat) (h : r < FF) :
    O_GU_CODES + (2 * r + 1) * ROW_GU + ROW_GU ≤ O_D_CODES := by
  simp only [ROW_GU_eq, O_D_CODES_eq, FF_eq] at h ⊢
  have : O_GU_CODES = 0 := rfl
  omega

/-- Every down row lies inside the down code region. -/
theorem down_rows_in_codes (r : Nat) (h : r < H) :
    O_D_CODES + (r + 1) * ROW_D ≤ O_GU_SCALES := by
  simp only [ROW_D_eq, O_D_CODES_eq, O_GU_SCALES_eq, H_eq] at h ⊢
  omega

/-- Every gate and up scale row lies inside the gate/up scale region. -/
theorem gu_scales_in_region (r : Nat) (h : r < FF) :
    O_GU_SCALES + ((2 * r + 1) * SC_GU * 2 + SC_GU * 2) ≤ O_D_SCALES := by
  simp only [SC_GU_eq, O_GU_SCALES_eq, O_D_SCALES_eq, FF_eq] at h ⊢
  omega

/-- Every down scale row lies inside the down scale region, which ends at the blob end. -/
theorem down_scales_in_region (r : Nat) (h : r < H) :
    O_D_SCALES + (r + 1) * SC_D * 2 ≤ BLOB := by
  simp only [SC_D_eq, O_D_SCALES_eq, BLOB_eq, H_eq] at h ⊢
  omega

/-- Activation chunk indices touched by weight block `b` of a gate/up row stay below `MAXC`. -/
theorem block_chunks_in_range (b : Nat) (h : b < SC_GU) : 2 * b + 1 < MAXC := by
  simp only [SC_GU_eq, MAXC_eq] at h ⊢
  omega

end Strata.Q2ExpertGeometry
