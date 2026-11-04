/-!
# MODULE: cpu_expert_geometry
Fixed geometry of the Q2_0 routed-expert blob (hidden width, FFN width, block sizes, byte offsets) and the
runtime CPU-feature gate that decides whether the AVX-512 expert kernel may run.

## Source semantics
* One Q2_0 weight block is `QK = 64` weights = fp16 scale (2 B) + 16 code bytes = 18 B; one activation chunk is
  `QKA = 32` elements, so a weight block spans exactly two activation chunks.
* One expert blob = gate/up codes, down codes, gate/up scales, down scales; its size is 1,382,400 bytes.
* `CpuFeatures::usable()` is the conjunction of five CPUID bits; `reason()` names exactly the missing ones.
* An activation with `n ≤ H` elements and `n % QKA = 0` uses at most `MAXC = 80` chunks.

## Form template identifiers
H → hidden_width, FF → ffn_width, NE → experts_per_layer, QK → weight_block_len, QKA → act_chunk_len,
BB → weight_block_bytes, MAXC → max_act_chunks, ROW_GU/ROW_D → code_bytes_per_row,
SC_GU/SC_D → blocks_per_row, BLOB → expert_blob_bytes, O_* → blob_section_offset,
CpuFeatures → cpu_feature_set, usable → feature_gate, reason → missing_feature_list.
-/

namespace Strata.Cpu.Geometry

/-- n_embd. -/
def H : Nat := 2560
/-- Expert intermediate width. -/
def FF : Nat := 640
/-- Routed experts per layer. -/
def NE : Nat := 512
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
/-- Bytes of one expert blob. -/
def BLOB : Nat := 3 * (H * FF * BB / QK)

/-- Offset of the gate/up codes. -/
def O_GU_CODES : Nat := 0
/-- Offset of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Offset of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2

-- § derived constants

/-- The derived numeric constants take the values recorded in the header comments. -/
theorem constants_values :
    MAXC = 80 ∧ ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 ∧ BLOB = 1382400 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> decide

/-- A weight block spans exactly two activation chunks. -/
theorem block_spans_two_chunks : QK = 2 * QKA := by decide

/-- A block is its 2-byte scale plus `QK / 4` code bytes. -/
theorem block_bytes_decomposition : BB = 2 + QK / 4 := by decide

/-- Row code bytes equal 16 per weight block, for both gate/up and down rows. -/
theorem row_bytes_per_block : ROW_GU = SC_GU * 16 ∧ ROW_D = SC_D * 16 := by
  constructor <;> decide

/-- Row widths are whole numbers of weight blocks. -/
theorem rows_are_block_multiples : H = SC_GU * QK ∧ FF = SC_D * QK := by
  constructor <;> decide

/-- The activation chunk count of the hidden vector is twice the gate/up block count. -/
theorem chunks_twice_blocks : MAXC = 2 * SC_GU := by decide

-- § blob layout

/-- The four sections tile the blob exactly: the last section ends at `BLOB`. -/
theorem blob_layout_tiles :
    O_GU_CODES = 0 ∧ O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 ∧
      O_D_SCALES + H * SC_D * 2 = BLOB := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> decide

/-- Sections are ordered, so no two overlap. -/
theorem blob_sections_ordered :
    O_GU_CODES < O_D_CODES ∧ O_D_CODES < O_GU_SCALES ∧ O_GU_SCALES < O_D_SCALES ∧
      O_D_SCALES < BLOB := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide

/-- Last gate/up row (index `2*FF-1`) of codes stays inside the gate/up code section. -/
theorem gu_codes_row_in_bounds (r : Nat) (hr : r < FF) :
    (2 * r + 1 + 1) * ROW_GU ≤ O_D_CODES := by
  have h : 2 * r + 2 ≤ 2 * FF := by unfold FF at *; omega
  have h2 : (2 * r + 2) * ROW_GU ≤ (2 * FF) * ROW_GU := Nat.mul_le_mul_right _ h
  have h3 : (2 * r + 1 + 1) = 2 * r + 2 := by omega
  rw [h3]
  unfold O_D_CODES
  exact h2

/-- Gate/up scales of row `2r+1` stay inside the scale section. -/
theorem gu_scales_row_in_bounds (r : Nat) (hr : r < FF) :
    (2 * r + 1) * SC_GU * 2 + SC_GU * 2 ≤ 2 * FF * SC_GU * 2 := by
  unfold FF SC_GU H QK at *
  omega

-- § activation chunk counts

/-- An activation of `n ≤ H` elements uses at most `MAXC` chunks. -/
theorem nchunks_le_maxc (n : Nat) (h : n ≤ H) : n / QKA ≤ MAXC := by
  unfold MAXC H QKA at *
  omega

/-- When `n` is a multiple of the chunk length, the chunks cover `n` exactly. -/
theorem chunks_cover_exactly (n : Nat) (h : n % QKA = 0) : (n / QKA) * QKA = n := by
  unfold QKA at *
  omega

/-- Chunk `2b` and `2b+1` are the two halves of weight block `b`. -/
theorem chunk_index_of_block (b h : Nat) (hh : h < 2) : (2 * b + h) / 2 = b ∧ (2 * b + h) % 2 = h := by
  constructor <;> omega

-- § CPU feature gate

/-- The five CPUID feature bits the kernel requires. -/
structure CpuFeatures where
  avx512f : Bool
  avx512bw : Bool
  avx512vl : Bool
  avx512_vnni : Bool
  avx512_vbmi : Bool

/-- `usable()` is the conjunction of all five bits. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- The names `reason()` prints, in source order. -/
def CpuFeatures.missing (f : CpuFeatures) : List String :=
  (if f.avx512f then [] else ["AVX512F"]) ++ (if f.avx512bw then [] else ["AVX512BW"]) ++
    (if f.avx512vl then [] else ["AVX512VL"]) ++ (if f.avx512_vnni then [] else ["AVX512-VNNI"]) ++
    (if f.avx512_vbmi then [] else ["AVX512-VBMI"])

/-- `reason()` reports "ok" exactly when the CPU is usable. -/
theorem usable_iff_nothing_missing (f : CpuFeatures) : f.usable = true ↔ f.missing = [] := by
  obtain ⟨a, b, c, d, e⟩ := f
  cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> simp [CpuFeatures.usable, CpuFeatures.missing]

/-- A CPU with every bit set is usable. -/
theorem all_features_usable : (CpuFeatures.mk true true true true true).usable = true := by decide

/-- A CPU lacking only VNNI is rejected, and the reason names only VNNI. -/
theorem only_vnni_missing :
    (CpuFeatures.mk true true true false true).usable = false ∧
      (CpuFeatures.mk true true true false true).missing = ["AVX512-VNNI"] := by
  constructor <;> decide

/-- Bit `k` of a CPUID register. -/
def cpuidBit (x k : Nat) : Bool := (x / 2 ^ k) % 2 == 1

/-- The leaf-7 bit positions used by `cpu_features()` are pairwise distinct per register. -/
theorem cpuid_positions_distinct :
    (16 ≠ 30 ∧ 16 ≠ 31 ∧ 30 ≠ 31) ∧ (11 ≠ 1) := by decide

/-- Setting only bit 16 of EBX sets only AVX512F among the EBX-derived flags. -/
theorem ebx_bit16_only_f :
    cpuidBit 65536 16 = true ∧ cpuidBit 65536 30 = false ∧ cpuidBit 65536 31 = false := by decide

end Strata.Cpu.Geometry
