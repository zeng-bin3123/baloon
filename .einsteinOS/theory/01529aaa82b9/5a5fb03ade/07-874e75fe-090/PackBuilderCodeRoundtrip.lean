/-!
# MODULE: pack_builder_code_roundtrip
`tools/strata_pack.py`: pack builder and verifier.  Code planes are packed LSB-first with `8/bits` codes per
byte; every plane starts on a 64-byte boundary; scales are stored fp16 only when exact; the verifier inverts
the expert blob layout.

## Source semantics
* pack_codes / unpack_codes are inverse for bits in {2,4,8}.
* `put` pads the body to 64 before each plane, so every plane offset is 64-aligned when the base is.
* `rebuild_role` addresses the same bytes `pack_layer` wrote, so it must stay in bounds.
* the verifier samples every Nth expert; `every = 32` still covers all layers.

## Form template identifiers
pack2 / pack4 -> pack_codes for bits 2 and 4
pad64         -> `(-len(body)) % ALIGN`
slot          -> interleaved gate/up row
-/

namespace Strata.PackBuilderCodeRoundtrip

-- § code packing

/-- Two 4-bit codes per byte. -/
def pack4 (a b : Nat) : Nat := a + 16 * b

/-- Unpack one nibble. -/
def unpackNib (byte k : Nat) : Nat := (byte / 16 ^ k) % 16

/-- 4-bit packing round-trips. -/
theorem pack4_roundtrip (a b : Nat) (ha : a < 16) (hb : b < 16) :
    unpackNib (pack4 a b) 0 = a ∧ unpackNib (pack4 a b) 1 = b := by
  unfold unpackNib pack4
  constructor <;> simp <;> omega

/-- 4-bit packing yields a byte. -/
theorem pack4_lt (a b : Nat) (ha : a < 16) (hb : b < 16) : pack4 a b < 256 := by
  unfold pack4; omega

/-- Four 2-bit codes per byte. -/
def pack2 (a b c d : Nat) : Nat := a + 4 * b + 16 * c + 64 * d

/-- Unpack one 2-bit code. -/
def unpackCrumb (byte k : Nat) : Nat := (byte / 4 ^ k) % 4

/-- 2-bit packing round-trips. -/
theorem pack2_roundtrip (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    unpackCrumb (pack2 a b c d) 0 = a ∧ unpackCrumb (pack2 a b c d) 1 = b ∧
      unpackCrumb (pack2 a b c d) 2 = c ∧ unpackCrumb (pack2 a b c d) 3 = d := by
  unfold unpackCrumb pack2
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp <;> omega

/-- 2-bit packing yields a byte. -/
theorem pack2_lt (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    pack2 a b c d < 256 := by
  unfold pack2; omega

/-- Codes per byte for the three forms: S2 -> 4, S4 -> 2, S8 -> 1. -/
theorem codes_per_byte : 8 / 2 = 4 ∧ 8 / 4 = 2 ∧ 8 / 8 = 1 := by decide

/-- Packed size of `n` codes at `bits` when `n` is a multiple of `8/bits`. -/
theorem packed_size_s4 (k : Nat) : (2 * k) / 2 = k := by omega

theorem packed_size_s2 (k : Nat) : (4 * k) / 4 = k := by omega

-- § plane alignment

/-- Padding before a plane. -/
def pad64 (len : Nat) : Nat := (64 - len % 64) % 64

/-- After the pad the plane begins on a 64-byte boundary. -/
theorem pad64_aligned (len : Nat) : (len + pad64 len) % 64 = 0 := by unfold pad64; omega

/-- Padding is under 64. -/
theorem pad64_lt (len : Nat) : pad64 len < 64 := by unfold pad64; omega

/-- A plane at an aligned body offset has an aligned absolute offset when the segment base is aligned. -/
theorem plane_abs_aligned (base len : Nat) (hb : base % 64 = 0) : (base + (len + pad64 len)) % 64 = 0 := by
  unfold pad64; omega

-- § scale precision

/-- FP16-or-FP32 choice records the width in bytes. -/
def scaleWidth (exact16 : Bool) : Nat := if exact16 then 2 else 4

/-- The recorded width is one of the two supported widths. -/
theorem scaleWidth_cases (b : Bool) : scaleWidth b = 2 ∨ scaleWidth b = 4 := by
  unfold scaleWidth; cases b <;> simp

/-- Scale plane bytes are `groups * width`; the byte count identifies the width when `groups > 0`. -/
theorem width_recoverable (g : Nat) (hg : 0 < g) (a b : Bool) (h : g * scaleWidth a = g * scaleWidth b) :
    a = b := by
  unfold scaleWidth at h
  cases a <;> cases b <;> simp at h ⊢ <;> omega

-- § arena inverse

/-- Slot of a gate or up row in the interleaved blob: gate 2r, up 2r+1. -/
def slot (isUp : Bool) (r : Nat) : Nat := 2 * r + (if isUp then 1 else 0)

/-- Slots of gate and up never coincide. -/
theorem slot_ne (r s : Nat) : slot false r ≠ slot true s := by unfold slot; simp; omega

/-- Gate/up code and scale addressing stay in the blob's planes. -/
theorem rebuild_gu_bounds (isUp : Bool) (row b : Nat) (hr : row < 640) (hb : b < 40) :
    slot isUp row * 640 + b * 16 + 16 ≤ 819200 ∧ (slot isUp row * 40 + b) * 2 + 2 ≤ 102400 := by
  unfold slot
  cases isUp <;> simp <;> omega

/-- The block index of the verifier splits as row and in-row block exactly. -/
theorem blk_split (blk : Nat) (h : blk < 25600) : blk / 40 < 640 ∧ blk % 40 < 40 := by omega

/-- Down role: 2560 rows of 10 blocks. -/
theorem blk_split_down (blk : Nat) (h : blk < 25600) : blk / 10 < 2560 ∧ blk % 10 < 10 := by omega

/-- Row and in-row block recombine to the block index. -/
theorem blk_recombine (blk : Nat) : (blk / 40) * 40 + blk % 40 = blk := by omega

-- § verifier coverage

/-- Sampling every 32nd expert checks 16 experts per layer. -/
theorem experts_checked_per_layer : (512 + 31) / 32 = 16 := by decide

/-- Experts 0, 32, ..., 480 are the sampled ones, all in range. -/
theorem sampled_in_range (i : Nat) (h : i < 16) : i * 32 < 512 := by omega

end Strata.PackBuilderCodeRoundtrip
