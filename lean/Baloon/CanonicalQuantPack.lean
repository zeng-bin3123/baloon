/-!
# MODULE: canonical_quant_pack
SOURCES: tools/canonical_xcheck.py, tools/ref_pack.py, tools/mtp_pack.py
Canonical code-plane forms of the GGUF quantization types (S2/S4/S8), bit packing of code planes,
64-byte plane alignment, block-geometry consistency, and the MTP expert quantizer.
## Source semantics
* Each type decodes as `(code + bias) * scale` (+ offset), the bias being applied to the INTEGER code
  before the single float multiply: Q2_0 -1, Q4_0 -8, Q5_0 -16, Q3_K -4, Q6_K -32, Q8_0 -128.
* Codes are packed LSB-first, `8 / bits` per byte; unpacking recovers every code.
* Nibble splits (Q4_0, IQ4_NL, Q4_K ...) put low nibbles first, high nibbles second.
* Planes start on 64-byte boundaries.
* The Q2_0 MTP quantizer maps `q` clipped to {-1..2} to code `q+1` in 0..3 and searches 17 scales
  `amax * (16+k)/32`.

## Form template identifiers
* MAPPINGS -> type_mapping_table, code_bias -> integer_code_bias, ALIGN -> plane_alignment
* pack_codes / unpack_codes -> code_plane_pack / code_plane_unpack
-/

namespace CanonPack

-- § code-plane byte sizing (pytools/planes.py: `out |= c[:, k] << (bits * k)`)

/-- Bytes needed to store `n` LSB-first codes at `bits` bits/code, when `8/bits | n`. -/
def cpbytes (n bits : Nat) : Nat := n / (8 / bits)

theorem cpbytes_q2  : cpbytes 64 2 = 16 := by decide
theorem cpbytes_q4  : cpbytes 64 4 = 32 := by decide
theorem cpbytes_q8  : cpbytes 64 8 = 64 := by decide

/-- No bits are wasted: `cpbytes n bits * 8 = n * bits` for 2- and 4-bit widths. -/
theorem cpbytes_no_waste_2 (n : Nat) (hn : n % 4 = 0) :
    cpbytes n 2 * 8 = n * 2 := by unfold cpbytes; omega

theorem cpbytes_no_waste_4 (n : Nat) (hn : n % 2 = 0) :
    cpbytes n 4 * 8 = n * 4 := by unfold cpbytes; omega

-- § Q2_0 block geometry: 2 scale bytes + 16 code bytes = 18 bytes, 64 codes per block

/-- Number of codes in one Q2_0 block. -/
def q2CodesPerBlock : Nat := 64

/-- Byte count of one Q2_0 block (fp16 scale + 64 2-bit codes). -/
def q2BlockSize : Nat := 2 + cpbytes q2CodesPerBlock 2

theorem q2BlockSize_val : q2BlockSize = 18 := by decide

/-- How many Q2_0 blocks cover `n` codes (must be a multiple of 64). -/
def q2BlockCnt (n : Nat) : Nat := n / q2CodesPerBlock

theorem q2BlockCnt_times_size (nBlocks : Nat) :
    q2BlockCnt (nBlocks * q2CodesPerBlock) = nBlocks := by
  unfold q2BlockCnt q2CodesPerBlock; simp [Nat.mul_div_cancel_left]

-- § decoded integer value range  (pytools/canon_xcheck.py: c.codes.astype(np.int32) - s.bias)

/-- Decoded integer value: unsigned `code` shifted by `lo` (the minimum). -/
def codeToInt (code : Nat) (lo : Int) : Int := lo + (code : Int)

/-- For Q2_0 (bias = -1, 2 bits): decoded values span exactly {-1, 0, 1, 2}. -/
theorem q2_int_range (c : Nat) (h : c < 4) :
    -1 ≤ codeToInt c (-1) ∧ codeToInt c (-1) ≤ 2 := by
  unfold codeToInt; constructor <;> omega

/-- For Q4_0 (bias = -8, 4 bits): decoded values span exactly {-8, …, 7}. -/
theorem q4_int_range (c : Nat) (h : c < 16) :
    -8 ≤ codeToInt c (-8) ∧ codeToInt c (-8) ≤ 7 := by
  unfold codeToInt; constructor <;> omega

/-- For Q8_0 (bias = -128, 8 bits): every int8 is representable. -/
theorem q8_encodable (v : Int) (hlo : -128 ≤ v) (hhi : v ≤ 127) :
    ∃ c : Nat, c < 256 ∧ codeToInt c (-128) = v := by
  exact ⟨(v + 128).toNat, by omega, by unfold codeToInt; omega⟩

-- § MTP Q2_0 expert quantizer  (pytools/mtp_quant.py)

/-- Clip a signed weight to the Q2_0 representable grid {-1, 0, 1, 2}. -/
def mtpClip (q : Int) : Int := max (-1) (min q 2)

theorem mtpClip_range (q : Int) : -1 ≤ mtpClip q ∧ mtpClip q ≤ 2 := by
  unfold mtpClip; constructor <;> simp [max, min] <;> omega

/-- The stored code is `mtpClip(q) + 1`, always a 2-bit value. -/
theorem mtpCode_in_range (q : Int) : 0 ≤ mtpClip q + 1 ∧ mtpClip q + 1 ≤ 3 := by
  obtain ⟨h1, h2⟩ := mtpClip_range q; omega

/-- 17 scale candidates: k = 0 .. 16, factor = (16 + k) / 32.
    The first candidate (k=0) has factor 1/2, the last (k=16) has factor 1. -/
def mtpScaleCand : Finset Nat := Finset.Icc 0 16

theorem mtpScaleCand_card : mtpScaleCand.card = 17 := by decide

theorem mtpScaleCand_extremes :
    0 ∈ mtpScaleCand ∧ 16 ∈ mtpScaleCand := by decide

-- § 64-byte plane alignment  (pytools/planes.py: `(ALIGN - n % ALIGN) % ALIGN`)

/-- Gap bytes appended so the next offset is 64-aligned. -/
def planeGap (off : Nat) : Nat := (64 - off % 64) % 64

/-- After adding the gap the offset is a multiple of 64. -/
theorem planeGap_aligns (off : Nat) : (off + planeGap off) % 64 = 0 := by
  unfold planeGap; omega

/-- The gap is strictly less than the alignment unit. -/
theorem planeGap_lt64 (off : Nat) : planeGap off < 64 := by
  unfold planeGap; omega

/-- An already-aligned offset needs no gap. -/
theorem planeGap_zero_iff (off : Nat) (h : off % 64 = 0) : planeGap off = 0 := by
  unfold planeGap; omega

-- § Q2_0 expert blob dimensions  (pytools/ref_packer.py, include/baloon/ref_pack_builder.hpp)
-- H=2560, FF=640, QK=64, NE=512: gate+up codes + scales + down codes + scales = 1382400 B.

/-- Gate/Up combined code plane: 2*FF rows, SC_GU blocks/row, 16 code bytes each. -/
def q2GuCodes : Nat := 2 * 640 * (2560 / 64 * 16)

/-- Gate/Up scale plane: 2*FF rows, SC_GU blocks/row, 2 scale bytes each. -/
def q2GuScales : Nat := 2 * 640 * (2560 / 64 * 2)

/-- Down code plane: H rows, SC_D blocks/row, 16 code bytes each. -/
def q2DCodes : Nat := 2560 * (640 / 64 * 16)

/-- Down scale plane: H rows, SC_D blocks/row, 2 scale bytes each. -/
def q2DScales : Nat := 2560 * (640 / 64 * 2)

/-- The four planes sum to 1382400 bytes, matching kBlobBytes in ref_pack_builder.hpp. -/
theorem q2ExpertBlobBytes :
    q2GuCodes + q2GuScales + q2DCodes + q2DScales = 1382400 := by
  decide

/-- Each plane boundary inside the blob falls on a 64-byte boundary. -/
theorem q2BlobPlanesAligned :
    q2GuCodes % 64 = 0 ∧ (q2GuCodes + q2GuScales) % 64 = 0 ∧
    (q2GuCodes + q2GuScales + q2DCodes) % 64 = 0 := by
  decide

end CanonPack
