/-!
# MODULE: expert_geometry
Fixed geometry of the CPU Q2_0 expert blob and the activation chunking (`expert.hpp`).

## Source semantics
One expert blob is gate/up codes, down codes, gate/up scales (fp16), down scales (fp16), totalling
1,382,400 bytes. A 64-weight block spans two 32-element activation chunks.

## Form template identifiers
* `H` → `H`, `FF` → `FF`, `QK` → `QK`, `QKA` → `QKA`, `BB` → `BB`, `MAXC` → `MAXC`
* `O_D_CODES`, `O_GU_SCALES`, `O_D_SCALES`, `BLOB` → same names
-/

namespace StrataExpertGeometry

/-- n_embd. -/
def H : Nat := 2560
/-- Expert intermediate width. -/
def FF : Nat := 640
/-- Weights per fp16 scale. -/
def QK : Nat := 64
/-- Activation elements per scale. -/
def QKA : Nat := 32
/-- Bytes per Q2_0 block. -/
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
/-- Blob size of one expert. -/
def BLOB : Nat := 3 * (H * FF * BB / QK)
/-- Offset of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Offset of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2

-- § sizes

/-- The blob is 1,382,400 bytes. -/
theorem blob_size : BLOB = 1382400 := by decide

/-- Down scales end exactly at the end of the blob. -/
theorem blob_end : O_D_SCALES + H * SC_D * 2 = BLOB := by decide

/-- Regions are ordered: codes, then scales. -/
theorem regions_ordered : O_D_CODES < O_GU_SCALES ∧ O_GU_SCALES < O_D_SCALES ∧ O_D_SCALES < BLOB := by
  decide

/-- Row geometry. -/
theorem row_sizes : ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 := by decide

/-- Gate/up blocks tile the hidden width. -/
theorem gu_blocks_tile : SC_GU * QK = H := by decide

/-- Down blocks tile the intermediate width. -/
theorem down_blocks_tile : SC_D * QK = FF := by decide

/-- A weight block spans exactly two activation chunks. -/
theorem block_spans_two_chunks : QK = 2 * QKA := by decide

/-- A Q2_0 block is a 2-byte scale plus 16 code bytes. -/
theorem block_bytes : BB = 2 + QK * 2 / 8 := by decide

/-- Chunks tile the hidden width and match two per gate/up block. -/
theorem chunk_counts : MAXC * QKA = H ∧ SC_GU * 2 = MAXC := by decide

/-- Gate and up rows (`2r`, `2r+1`) of a row `r < FF` stay inside the gate/up code region. -/
theorem gu_rows_in_region (r : Nat) (h : r < 640) : (2 * r + 1) * 640 + 640 ≤ 819200 := by omega

/-- Each gate/up row's scales (`SC_GU*2` bytes) stay inside the gate/up scale region. -/
theorem gu_scale_rows_in_region (r : Nat) (h : r < 640) :
    (2 * r + 1) * 40 * 2 + 40 * 2 ≤ 2 * 640 * 40 * 2 := by omega

/-- A quantized activation of width `n` (multiple of 32) has `n/32` chunks covering it. -/
theorem nchunks_cover (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by omega

-- § CPU feature gate

/-- Required AVX-512 feature flags. -/
structure CpuFeatures where
  avx512f : Bool
  avx512bw : Bool
  avx512vl : Bool
  avx512_vnni : Bool
  avx512_vbmi : Bool

/-- The kernel is usable iff all five features are present. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- Missing VNNI makes the CPU unusable. -/
theorem unusable_without_vnni (f : CpuFeatures) (h : f.avx512_vnni = false) : f.usable = false := by
  simp [CpuFeatures.usable, h]

/-- Missing VBMI makes the CPU unusable. -/
theorem unusable_without_vbmi (f : CpuFeatures) (h : f.avx512_vbmi = false) : f.usable = false := by
  simp [CpuFeatures.usable, h]

end StrataExpertGeometry
