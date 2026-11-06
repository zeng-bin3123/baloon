/-!
# MODULE: q2_0_gguf_writer
SOURCES: tools/gguf_writer.py
Minimal GGUF v3 writer with the Q2_0 quantizer: code = clamp(round(w/d)+1, 0, 3), four codes per byte
LSB-first, 32-byte aligned tensor data.

## Source semantics
* d = amax, so w/d lies in [-1,1] and the +2 symbol (code 3) is never produced by this quantizer.
* The quantizer is idempotent: requantizing dequantized symbols returns the same codes.
* Tensor directory offsets and the data section are padded to the alignment (default 32).
* Metadata type ids 0..12 match the reader's MetaType enum.

## Form template identifiers
quantize_q2_0 -> q2_0_quantize_block ; dequantize_q2_0 -> q2_0_dequant_block ;
GGUFWriter.write -> gguf_v3_serialise ; _META_IDS -> meta_type_ids
-/
set_option maxRecDepth 100000

namespace Strata.Writer

-- § quantizer
def clampCode (r : Int) : Nat := Int.toNat (max 0 (min 3 (r + 1)))
def symbolOf (c : Nat) : Int := (c : Int) - 1
def packByte (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- The code is always a valid 2-bit value. -/
theorem clampCode_lt (r : Int) : clampCode r < 4 := by
  unfold clampCode
  omega

/-- Rounded ratios in {-1,0,1,2} are represented exactly. -/
theorem clampCode_exact (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 2) : symbolOf (clampCode r) = r := by
  unfold clampCode symbolOf
  omega

/-- With d = amax the rounded ratio is in [-1,1], so code 3 (+2) is never emitted. -/
theorem clampCode_amax_never_three (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 1) : clampCode r ≤ 2 := by
  unfold clampCode
  omega

/-- Out-of-range ratios saturate at the ends of the grid. -/
theorem clampCode_saturates (r : Int) :
    (r ≤ -1 → clampCode r = 0) ∧ (2 ≤ r → clampCode r = 3) := by
  unfold clampCode
  constructor <;> intro h <;> omega

/-- Requantization is a fixed point: quantizing the symbol of a code gives that code back. -/
theorem requantize_fixed_point (c : Nat) (hc : c < 4) : clampCode (symbolOf c) = c := by
  unfold clampCode symbolOf
  omega

/-- A packed byte fits in 8 bits. -/
theorem packByte_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 < 256 := by
  unfold packByte
  omega

/-- Packing places code j%4 at bit (j%4)*2; unpacking recovers all four codes. -/
theorem packByte_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 % 4 = c0 ∧ packByte c0 c1 c2 c3 / 4 % 4 = c1 ∧
    packByte c0 c1 c2 c3 / 16 % 4 = c2 ∧ packByte c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold packByte
  omega

/-- Element j is stored in byte j/4 at bit offset (j%4)*2, which is below 8. -/
theorem element_slot (j : Nat) (hj : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 < 8 := by
  omega

/-- A Q2_0 block is 2 + 16 = 18 bytes holding 64 codes. -/
theorem q2_0_block_bytes : 2 + 64 / 4 = 18 := by decide

/-- Quantized size is 18 bytes per 64 elements. -/
theorem q2_0_size (n : Nat) : n * 64 / 64 * 18 = n * 18 := by omega

-- § alignment and layout
def padTo32 (n : Nat) : Nat := (32 - n % 32) % 32

/-- Padding is below 32 and makes the length a multiple of 32. -/
theorem padTo32_aligns (n : Nat) : padTo32 n < 32 ∧ (n + padTo32 n) % 32 = 0 := by
  unfold padTo32
  omega

/-- Writer padding agrees with the reader's round-up formula (c + 31)/32*32. -/
theorem pad_matches_reader (n : Nat) : n + padTo32 n = (n + 32 - 1) / 32 * 32 := by
  unfold padTo32
  omega

/-- Each tensor's next directory offset is aligned and not before the end of its data. -/
theorem next_offset_aligned (o len : Nat) :
    (o + len + 31) / 32 * 32 % 32 = 0 ∧ o + len ≤ (o + len + 31) / 32 * 32 := by
  omega

/-- Header prefix is magic 4 + version 4 + tensor count 8 + kv count 8 = 24 bytes. -/
theorem header_prefix_bytes : 4 + 4 + 8 + 8 = 24 := by decide

/-- A tensor directory entry occupies 8 + name + 4 + 8*ndims + 4 + 8 bytes. -/
def tensorInfoBytes (nameLen nd : Nat) : Nat := 8 + nameLen + 4 + 8 * nd + 4 + 8

/-- A directory entry for a 1-D tensor is name + 32 bytes. -/
theorem tensorInfo_1d (n : Nat) : tensorInfoBytes n 1 = n + 32 := by
  unfold tensorInfoBytes
  omega

/-- Directory entries grow by 8 bytes per extra dimension. -/
theorem tensorInfo_dim_step (n nd : Nat) : tensorInfoBytes n (nd + 1) = tensorInfoBytes n nd + 8 := by
  unfold tensorInfoBytes
  omega

/-- The selftest tensor [64,16] Q2_0 holds 16 blocks = 288 bytes. -/
theorem selftest_tensor_bytes : 64 * 16 / 64 * 18 = 288 := by decide

-- § metadata ids
def metaIds : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

/-- Metadata type ids are 13 distinct values 0..12 as in the reader's enum. -/
theorem meta_ids_distinct : metaIds.length = 13 ∧ metaIds.Nodup := by decide

/-- Supported tensor type ids for writing are F32=0, F16=1, Q2_0=42. -/
theorem writer_type_ids : [0, 1, 42].Nodup ∧ [0, 1, 42].length = 3 := by decide

end Strata.Writer
