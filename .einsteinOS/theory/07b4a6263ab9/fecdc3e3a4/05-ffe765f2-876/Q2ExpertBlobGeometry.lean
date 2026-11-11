/-!
# MODULE: q2_expert_blob_geometry
SOURCES: include/strata/kernels/cpu/expert.hpp, src/kernels/cpu/expert.cpp, src/kernels/cpu/expert_layout.cpp
Fixes the geometry of one Q2_0 routed expert (H=2560, FF=640, QK=64, QKA=32) and the byte layout of its
1,382,400-byte blob, and proves every row/scale address the kernels form stays inside its region.  Also
models the CPUID feature gate (`CpuFeatures::usable`, `cpu_avx512_ok`).

## Source semantics
* A Q2_0 weight block is QK=64 two-bit codes (16 bytes) plus one fp16 scale (2 bytes) = BB=18 bytes.
* An activation chunk is QKA=32 int8 values with its own scale, so ONE weight block spans TWO chunks.
* Blob layout: gate/up codes [2*FF rows x 640 B] | down codes [H rows x 160 B] |
  gate/up scales [2*FF x 40 fp16] | down scales [H x 10 fp16]; the four regions tile the blob exactly.
* Gate row r is interleaved row 2r, up row r is row 2r+1.
* The expert kernel is usable iff AVX512F, BW, VL, VNNI and VBMI are all present (CPUID leaf 7:
  EBX bits 16,30,31; ECX bits 11,1); `cpu_avx512_ok` additionally requires OSXSAVE and XCR0 & 0xE6 = 0xE6.

## Form template identifiers
H → embedWidth, FF → ffWidth, QK → weightBlock, QKA → actChunk, BB → blockBytes, BLOB → blobBytes,
O_GU_CODES/O_D_CODES/O_GU_SCALES/O_D_SCALES → regionOffset_{guCodes,dCodes,guScales,dScales},
CpuFeatures.usable → featureGate, cpu_avx512_ok → osFeatureGate
-/

namespace Strata.Kernels.Cpu.Geometry

-- § constants (expert.hpp)

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
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2

-- § evaluated constants

/-- 80 activation chunks across the widest reduction. -/
theorem maxc_val : MAXC = 80 := by decide

/-- 640 code bytes per gate/up row. -/
theorem row_gu_val : ROW_GU = 640 := by decide

/-- 160 code bytes per down row. -/
theorem row_d_val : ROW_D = 160 := by decide

/-- 40 weight blocks per gate/up row. -/
theorem sc_gu_val : SC_GU = 40 := by decide

/-- 10 weight blocks per down row. -/
theorem sc_d_val : SC_D = 10 := by decide

/-- The header's stated blob size. -/
theorem blob_val : BLOB = 1382400 := by decide

/-- Region offsets, evaluated. -/
theorem offsets_val :
    O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 := by decide

-- § block contract

/-- THE contract: a 64-weight block covers exactly two 32-element activation chunks. -/
theorem weight_block_spans_two_chunks : QK = 2 * QKA := by decide

/-- 18 bytes per block = 16 code bytes (2 bits per weight) + one fp16 scale. -/
theorem block_bytes : QK * 2 / 8 + 2 = BB := by decide

/-- A row's code bytes are 16 per weight block. -/
theorem row_codes_per_block : ROW_GU = SC_GU * 16 ∧ ROW_D = SC_D * 16 := by decide

/-- The four regions tile the blob exactly: the last region ends at `BLOB`. -/
theorem layout_tiles_blob : O_D_SCALES + H * SC_D * 2 = BLOB := by decide

/-- The blob equals three matrices of H*FF Q2_0 weights at 18 bytes per 64. -/
theorem blob_three_matrices : BLOB = (2 * FF * H + H * FF) * BB / QK := by decide

-- § addresses stay in their regions

/-- Gate row 2r and up row 2r+1 codes lie inside the gate/up code region. -/
theorem gu_codes_in_region (r : Nat) (hr : r < FF) :
    O_GU_CODES + (2 * r + 1) * ROW_GU + ROW_GU ≤ O_D_CODES := by
  have h1 : ROW_GU = 640 := row_gu_val
  have h2 : O_D_CODES = 819200 := offsets_val.1
  have h3 : FF = 640 := rfl
  rw [h1, h2]
  simp only [O_GU_CODES]
  omega

/-- Down row r's codes lie inside the down code region. -/
theorem d_codes_in_region (r : Nat) (hr : r < H) :
    O_D_CODES + r * ROW_D + ROW_D ≤ O_GU_SCALES := by
  have h1 : ROW_D = 160 := row_d_val
  have h2 := offsets_val
  have h3 : H = 2560 := rfl
  rw [h1, h2.1, h2.2.1]
  omega

/-- Gate/up scale rows lie inside the gate/up scale region. -/
theorem gu_scales_in_region (r : Nat) (hr : r < FF) :
    O_GU_SCALES + (2 * r + 1) * SC_GU * 2 + SC_GU * 2 ≤ O_D_SCALES := by
  have h1 : SC_GU = 40 := sc_gu_val
  have h2 := offsets_val
  have h3 : FF = 640 := rfl
  rw [h1, h2.2.1, h2.2.2]
  omega

/-- Down scale rows lie inside the last region, hence inside the blob. -/
theorem d_scales_in_blob (r : Nat) (hr : r < H) :
    O_D_SCALES + r * SC_D * 2 + SC_D * 2 ≤ BLOB := by
  have h1 : SC_D = 10 := sc_d_val
  have h2 := offsets_val
  have h3 : H = 2560 := rfl
  have h4 : BLOB = 1382400 := blob_val
  rw [h1, h2.2.2, h4]
  omega

/-- Weight j of block b in a gate/up row reads code byte `b*16 + j/4`, which is inside the row. -/
theorem code_byte_in_row (b j : Nat) (hb : b < SC_GU) (hj : j < QK) :
    b * 16 + j / 4 < ROW_GU := by
  have h1 : SC_GU = 40 := sc_gu_val
  have h2 : ROW_GU = 640 := row_gu_val
  have h3 : QK = 64 := rfl
  rw [h1] at hb; rw [h3] at hj; rw [h2]
  omega

/-- Block b uses activation chunks 2b and 2b+1, both inside the 80-chunk scale arrays. -/
theorem block_chunks_in_range (b : Nat) (hb : b < SC_GU) : 2 * b + 1 < MAXC := by
  have h1 : SC_GU = 40 := sc_gu_val
  have h2 : MAXC = 80 := maxc_val
  rw [h1] at hb; rw [h2]
  omega

/-- The chunk of element i is `i / QKA`, and for i in the block it is one of the block's two chunks. -/
theorem element_chunk_of_block (b j : Nat) (hj : j < QK) :
    (b * QK + j) / QKA = 2 * b ∨ (b * QK + j) / QKA = 2 * b + 1 := by
  have h3 : QK = 64 := rfl
  have h4 : QKA = 32 := rfl
  rw [h3] at hj ⊢; rw [h4]
  omega

-- § CPU feature gate

/-- The five feature flags the kernel needs. -/
structure CpuFeatures where
  avx512f : Bool
  avx512bw : Bool
  avx512vl : Bool
  avx512_vnni : Bool
  avx512_vbmi : Bool

/-- `CpuFeatures::usable`. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- The CPUID leaf-7 bit extraction of `cpu_features()`. -/
def featuresOf (ebx ecx : Nat) : CpuFeatures :=
  { avx512f := ebx.testBit 16, avx512bw := ebx.testBit 30, avx512vl := ebx.testBit 31,
    avx512_vnni := ecx.testBit 11, avx512_vbmi := ecx.testBit 1 }

/-- Usable exactly when every one of the five flags is set (missing ANY one refuses). -/
theorem usable_iff (f : CpuFeatures) :
    f.usable = true ↔ f.avx512f = true ∧ f.avx512bw = true ∧ f.avx512vl = true ∧
      f.avx512_vnni = true ∧ f.avx512_vbmi = true := by
  cases f with
  | mk a b c d e => cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> decide

/-- A machine with AVX-512F/BW/VL but no VNNI is refused. -/
theorem refused_without_vnni (f : CpuFeatures) (h : f.avx512_vnni = false) : f.usable = false := by
  cases f with
  | mk a b c d e =>
    simp only [CpuFeatures.usable] at h ⊢
    rw [h]
    cases a <;> cases b <;> cases c <;> rfl

/-- `cpu_avx512_ok`: OSXSAVE, the OS-saved XCR0 state mask, and the same five CPUID bits. -/
def osFeatureGate (ecx1 xcr0 ebx7 ecx7 : Nat) : Bool :=
  ecx1.testBit 27 && (xcr0 &&& 0xE6 == 0xE6) && (featuresOf ebx7 ecx7).usable

/-- The XCR0 mask 0xE6 is SSE, AVX, opmask, ZMM_Hi256 and Hi16_ZMM state (bits 1,2,5,6,7). -/
theorem xcr0_mask_bits : (0xE6 : Nat) = 2 ^ 1 + 2 ^ 2 + 2 ^ 5 + 2 ^ 6 + 2 ^ 7 := by decide

/-- The OS gate implies the CPU-only gate: it can only refuse more machines, never fewer. -/
theorem osGate_implies_usable (e1 x b c : Nat) (h : osFeatureGate e1 x b c = true) :
    (featuresOf b c).usable = true := by
  simp only [osFeatureGate, Bool.and_eq_true] at h
  exact h.2

/-- Example: a fully featured EBX/ECX pair is usable. -/
theorem full_features_usable :
    (featuresOf (2 ^ 16 + 2 ^ 30 + 2 ^ 31) (2 ^ 11 + 2 ^ 1)).usable = true := by decide

end Strata.Kernels.Cpu.Geometry
