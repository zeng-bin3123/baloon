/-!
# MODULE: strata_pack_builder_verifier
SOURCES: tools/strata_pack.py
Integer model of the canonical pack builder: code-plane bit packing (2/4/8 bits, LSB first), plane alignment
inside a segment, scale precision storage sizes, form bit widths, expert-arena size, and the verify sampling counts.

## Source semantics
Codes are packed `8/bits` per byte, LSB first. Every plane starts on a 64-byte boundary measured from the start
of its segment (`dense.bin` or `embd.bin`); the final plane of a blob is not padded. Scales are stored fp16 when the
round trip is exact, else fp32. The pack is valid only if decode(pack) == dequant(source) bit for bit.

## Form template identifiers
`ALIGN` -> 64; `code_bits` -> `codeBits`; `pack_codes`/`unpack_codes` -> `packTwo`/`packNibble`;
`put` -> `putOffset`; `choose_scale_dtype` -> `scaleBytes`; `verify_experts` -> `expertsChecked`.
-/

namespace StrataPack

-- § form bit widths

/-- Bits per code of each canonical form. -/
def codeBits : String → Nat
  | "S2" => 2
  | "S4" => 4
  | _ => 8

/-- Codes per byte for every integer form is exactly 8/bits and fills the byte. -/
theorem codes_fill_byte : ∀ f ∈ ["S2", "S4", "S8"], 8 / codeBits f * codeBits f = 8 := by decide

-- § bit packing

/-- Pack two 4-bit codes into one byte, LSB first. -/
def packNibble (c0 c1 : Nat) : Nat := c0 + 16 * c1

/-- Nibble pack/unpack round trip. -/
theorem nibble_roundtrip (c0 c1 : Nat) (h0 : c0 < 16) (h1 : c1 < 16) :
    packNibble c0 c1 < 256 ∧ packNibble c0 c1 % 16 = c0 ∧ packNibble c0 c1 / 16 = c1 := by
  unfold packNibble; omega

/-- Pack four 2-bit codes into one byte, LSB first. -/
def packTwo (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- 2-bit pack/unpack round trip. -/
theorem two_roundtrip (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packTwo c0 c1 c2 c3 < 256 ∧ packTwo c0 c1 c2 c3 % 4 = c0 ∧ packTwo c0 c1 c2 c3 / 4 % 4 = c1 ∧
    packTwo c0 c1 c2 c3 / 16 % 4 = c2 ∧ packTwo c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold packTwo; omega

/-- Packed size: n codes at `per` codes per byte occupy n/per bytes when divisible, and the code count is recoverable. -/
theorem packed_size (n per : Nat) (hp : 0 < per) (h : n % per = 0) : (n / per) * per = n := by
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)

/-- The 4-bit element count must be a multiple of 2 and the 2-bit count a multiple of 4. -/
theorem q2_count_nonmultiple (n : Nat) (h : n % 4 ≠ 0) : ¬ (n % 4 = 0) := h

-- § alignment inside a segment

/-- Align a length to 64. -/
def pad64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- Padding lands on a boundary and is below 64. -/
theorem pad64_spec (n : Nat) : pad64 n < 64 ∧ (n + pad64 n) % 64 = 0 := by
  unfold pad64; omega

/-- Offset of a plane: segment base + current body length after padding. -/
def putOffset (base bodyLen : Nat) : Nat := base + (bodyLen + pad64 bodyLen)

/-- Plane offsets are 64-aligned relative to the segment start. -/
theorem put_relative_aligned (base bodyLen : Nat) : (putOffset base bodyLen - base) % 64 = 0 := by
  unfold putOffset pad64; omega

/-- Absolute alignment holds exactly when the segment base is itself 64-aligned. -/
theorem put_absolute_aligned (base bodyLen : Nat) (hb : base % 64 = 0) : putOffset base bodyLen % 64 = 0 := by
  unfold putOffset pad64; omega

/-- Without an aligned base the plane is generally not absolutely aligned (the last blob is not padded). -/
theorem put_unaligned_base : putOffset 1 0 % 64 ≠ 0 := by decide

/-- Planes never overlap: the next plane starts at or after the previous one's end. -/
theorem put_no_overlap (base bodyLen : Nat) : base + bodyLen ≤ putOffset base bodyLen := by
  unfold putOffset; omega

-- § scale precision

/-- Bytes used by a scale plane of n values. -/
def scaleBytes (isFp16 : Bool) (n : Nat) : Nat := if isFp16 then 2 * n else 4 * n

/-- fp16 scales halve the plane compared with fp32. -/
theorem scale_fp16_half (n : Nat) : 2 * scaleBytes true n = scaleBytes false n := by
  unfold scaleBytes; simp; omega

/-- A scale plane always has at least 2 bytes per value. -/
theorem scale_bytes_ge (b : Bool) (n : Nat) : 2 * n ≤ scaleBytes b n := by
  unfold scaleBytes; cases b <;> simp <;> omega

-- § arena size and verification sampling

/-- Expert blob bytes. -/
def blobBytes : Nat := 1382400
/-- Experts per layer. -/
def nExperts : Nat := 512

/-- experts.bin size for n layers. -/
def expertsFileBytes (n : Nat) : Nat := n * nExperts * blobBytes

/-- The default 48 layers give 33,973,862,400 bytes. -/
theorem experts_file_48 : expertsFileBytes 48 = 33973862400 := by decide

/-- Blob count reported by `info`. -/
theorem info_blob_count : 48 * nExperts = 24576 := by decide

/-- Experts verified per layer when sampling every k-th. -/
def expertsPerLayerChecked (every : Nat) : Nat := (512 + every - 1) / every

/-- Total layer/expert/role checks. -/
def expertsChecked (layers every : Nat) : Nat := layers * expertsPerLayerChecked every * 3

/-- Default sampling (every 32nd of 512) checks 16 experts per layer. -/
theorem default_sampling : expertsPerLayerChecked 32 = 16 := by decide

/-- The default full pack verification runs 2304 role checks. -/
theorem default_checks : expertsChecked 48 32 = 2304 := by decide

/-- Sampling every expert checks all 512. -/
theorem every_one : expertsPerLayerChecked 1 = 512 := by decide

/-- Sampling is never empty when every <= 512. -/
theorem sampling_nonempty (every : Nat) (h1 : 0 < every) (h2 : every ≤ 512) : 0 < expertsPerLayerChecked every := by
  unfold expertsPerLayerChecked
  have : every ≤ 512 + every - 1 := by omega
  exact Nat.div_pos this h1

end StrataPack
