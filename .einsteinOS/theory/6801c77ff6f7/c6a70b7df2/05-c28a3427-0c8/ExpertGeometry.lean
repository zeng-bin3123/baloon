/-!
# MODULE: expert_geometry
SOURCES: include/strata/kernels/cpu/expert.hpp
The fixed geometry of one Q2_0 routed expert (one 1,382,400-byte blob), the offsets of its four regions,
the weight-block / activation-chunk relationship, and the AVX-512 feature predicate that gates the kernel.

## Source semantics
* One expert blob = gate/up codes, down codes, gate/up scales, down scales, in that order, tiling the blob exactly.
* Gate and up rows are interleaved: gate row r is matrix row 2r, up row r is matrix row 2r+1.
* One 64-weight Q2_0 block spans TWO 32-element activation chunks (each with its own scale).
* `CpuFeatures.usable` is the conjunction of the five required AVX-512 flags; `reason` is "ok" iff usable.

## Form template identifiers
H → hidden_width, FF → expert_ff_width, NE → experts_per_layer, QK → weights_per_scale,
QKA → act_elems_per_scale, BB → block_bytes, ROW_GU/ROW_D → code_row_bytes_{gate_up,down},
SC_GU/SC_D → blocks_per_row_{gate_up,down}, BLOB → expert_blob_bytes, O_* → region_offset_*,
CpuFeatures.usable → isa_gate.
-/

namespace Strata.Kernels.Cpu.ExpertGeometry

-- § constants

/-- n_embd. -/
def H : Nat := 2560
/-- Expert intermediate width. -/
def FF : Nat := 640
/-- Routed experts per layer. -/
def NE : Nat := 512
/-- Weights per fp16 scale (QK2_0). -/
def QK : Nat := 64
/-- Activation elements per scale (QK8_1). -/
def QKA : Nat := 32
/-- Bytes per 64-weight Q2_0 block: 2-byte scale plus 16 code bytes. -/
def BB : Nat := 18
/-- Activation chunks across the widest reduction. -/
def MAXC : Nat := H / QKA
/-- Code bytes per gate/up row. -/
def ROW_GU : Nat := H * 2 / 8
/-- Code bytes per down row. -/
def ROW_D : Nat := FF * 2 / 8
/-- Weight blocks per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Weight blocks per down row. -/
def SC_D : Nat := FF / QK
/-- Bytes of one expert blob. -/
def BLOB : Nat := 3 * (H * FF * BB / QK)
/-- Offset of the gate/up code region. -/
def O_GU_CODES : Nat := 0
/-- Offset of the down code region. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scale region. -/
def O_GU_SCALES : Nat := O_D_CODES + 1 * H * ROW_D
/-- Offset of the down scale region. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
/-- Maximum number of tokens routed to one expert in a verify window. -/
def MAXT : Nat := 8

-- § numeric facts

/-- The derived row/chunk counts have the values the comments in the header state. -/
theorem derived_counts :
    MAXC = 80 ∧ ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 := by
  decide

/-- The blob is 1,382,400 bytes. -/
theorem blob_size : BLOB = 1382400 := by
  decide

/-- The four region offsets. -/
theorem region_offsets :
    O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 := by
  decide

/-- The four regions tile the blob exactly: the last region ends at `BLOB`. -/
theorem layout_tiles_blob : O_D_SCALES + H * SC_D * 2 = BLOB := by
  decide

/-- The regions are strictly ordered. -/
theorem region_order :
    O_GU_CODES < O_D_CODES ∧ O_D_CODES < O_GU_SCALES ∧ O_GU_SCALES < O_D_SCALES ∧ O_D_SCALES < BLOB := by
  decide

/-- A block is its 2-byte scale plus 64 two-bit codes packed into 16 bytes. -/
theorem block_bytes : BB = 2 + QK * 2 / 8 := by
  decide

/-- One weight block spans two activation chunks. -/
theorem weight_block_two_chunks : QK = 2 * QKA := by
  decide

/-- Chunk counts agree: gate/up rows have `MAXC` chunks, down rows cover `FF` elements. -/
theorem chunk_counts : SC_GU * 2 = MAXC ∧ SC_D * 2 * QKA = FF := by
  decide

-- § row addressing stays inside its region

/-- Every gate/up code row (matrix rows 0..2*FF-1) lies inside the gate/up code region. -/
theorem gu_rows_within_codes (r : Nat) (hr : r < 2 * FF) :
    r * ROW_GU + ROW_GU ≤ O_D_CODES := by
  have e1 : ROW_GU = 640 := by decide
  have e2 : O_D_CODES = 819200 := by decide
  have e3 : FF = 640 := by decide
  rw [e1, e2]
  rw [e3] at hr
  omega

/-- Every down code row lies inside the down code region. -/
theorem down_rows_within_codes (r : Nat) (hr : r < H) :
    O_D_CODES + r * ROW_D + ROW_D ≤ O_GU_SCALES := by
  have e1 : ROW_D = 160 := by decide
  have e2 : O_D_CODES = 819200 := by decide
  have e3 : O_GU_SCALES = 1228800 := by decide
  have e4 : H = 2560 := by decide
  rw [e1, e2, e3]
  rw [e4] at hr
  omega

/-- Every gate/up scale row lies inside the gate/up scale region. -/
theorem gu_scales_within_region (r : Nat) (hr : r < 2 * FF) :
    O_GU_SCALES + r * SC_GU * 2 + SC_GU * 2 ≤ O_D_SCALES := by
  have e1 : SC_GU = 40 := by decide
  have e2 : O_GU_SCALES = 1228800 := by decide
  have e3 : O_D_SCALES = 1331200 := by decide
  have e4 : FF = 640 := by decide
  rw [e1, e2, e3]
  rw [e4] at hr
  omega

/-- Every down scale row lies inside the down scale region, which ends at the blob end. -/
theorem down_scales_within_blob (r : Nat) (hr : r < H) :
    O_D_SCALES + r * SC_D * 2 + SC_D * 2 ≤ BLOB := by
  have e1 : SC_D = 10 := by decide
  have e2 : O_D_SCALES = 1331200 := by decide
  have e3 : BLOB = 1382400 := by decide
  have e4 : H = 2560 := by decide
  rw [e1, e2, e3]
  rw [e4] at hr
  omega

/-- Gate row r and up row r are distinct, adjacent matrix rows, both inside the gate/up matrix. -/
theorem gate_up_interleave (r : Nat) (hr : r < FF) :
    2 * r < 2 * FF ∧ 2 * r + 1 < 2 * FF ∧ 2 * r ≠ 2 * r + 1 := by
  omega

/-- Gate/up is a 2*FF-row matrix and down is an H-row matrix: row counts times row bytes fill the code regions. -/
theorem code_regions_sizes :
    2 * FF * ROW_GU = O_D_CODES ∧ H * ROW_D = O_GU_SCALES - O_D_CODES := by
  decide

-- § the ISA gate

/-- The CPU feature flags the expert kernel needs (all default to false). -/
structure CpuFeatures where
  avx512f : Bool := false
  avx512bw : Bool := false
  avx512vl : Bool := false
  avx512_vnni : Bool := false
  avx512_vbmi : Bool := false

/-- The kernel is usable iff all five flags are present. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- The ids (1..5) of the missing features, in the order `reason` names them. -/
def CpuFeatures.missing (f : CpuFeatures) : List Nat :=
  (if f.avx512f then [] else [1]) ++ (if f.avx512bw then [] else [2]) ++
  (if f.avx512vl then [] else [3]) ++ (if f.avx512_vnni then [] else [4]) ++
  (if f.avx512_vbmi then [] else [5])

/-- `usable` is exactly the conjunction of the five flags. -/
theorem usable_iff (f : CpuFeatures) :
    f.usable = true ↔
      (f.avx512f = true ∧ f.avx512bw = true ∧ f.avx512vl = true ∧
        f.avx512_vnni = true ∧ f.avx512_vbmi = true) := by
  simp [CpuFeatures.usable, and_assoc]

/-- A machine reporting nothing is refused. -/
theorem default_not_usable : ({} : CpuFeatures).usable = false := by
  rfl

/-- A machine without VNNI is refused even if everything else is present. -/
theorem missing_vnni_refused (f : CpuFeatures) (h : f.avx512_vnni = false) : f.usable = false := by
  simp [CpuFeatures.usable, h]

/-- A machine without VBMI is refused even if everything else is present. -/
theorem missing_vbmi_refused (f : CpuFeatures) (h : f.avx512_vbmi = false) : f.usable = false := by
  simp [CpuFeatures.usable, h]

/-- `reason` reports "ok" (nothing missing) exactly when the machine is usable. -/
theorem reason_ok_iff_usable (f : CpuFeatures) : f.missing = [] ↔ f.usable = true := by
  rcases f with ⟨a, b, c, d, e⟩
  cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> decide

end Strata.Kernels.Cpu.ExpertGeometry
