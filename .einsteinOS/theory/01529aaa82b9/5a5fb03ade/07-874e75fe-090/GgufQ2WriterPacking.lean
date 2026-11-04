/-!
# MODULE: gguf_q2_writer_packing
The minimal GGUF v3 writer: Q2_0 quantizer packing (4 two-bit codes per byte, LSB first), clamp rule, and the
32-byte alignment of the tensor directory and data section.

## Source semantics
* Block = 64 elements = fp16 scale (2 B) + 16 bytes of codes; element j is in byte j/4 at bit (j%4)*2.
* code = clamp(round(w/d) + 1, 0, 3); symbol = code - 1 in {-1,0,1,2}; requantizing a decoded block is a fixed point.
* Offsets are padded up to `alignment` (32) after every tensor and after the header.

## Form template identifiers
pack4        -> packed byte of four codes
codeAt       -> element j's code from a byte array
clamp03      -> code clamp
align32      -> padding to DEFAULT_ALIGNMENT
-/

namespace Strata.GgufQ2WriterPacking

-- § 2-bit packing

/-- Four 2-bit codes in one byte, least significant first. -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed byte is a byte. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4; omega

/-- Code of element `j` read from a byte array: byte `j/4`, bit offset `(j%4)*2`. -/
def codeAt (qs : Nat → Nat) (j : Nat) : Nat := (qs (j / 4) / 4 ^ (j % 4)) % 4

/-- Slot 0 of a packed byte. -/
theorem codeAt_slot0 (c0 c1 c2 c3 j : Nat) (h0 : c0 < 4) (hj : j % 4 = 0) :
    codeAt (fun _ => pack4 c0 c1 c2 c3) j = c0 := by
  simp only [codeAt, pack4, hj]; omega

/-- Slot 1 of a packed byte. -/
theorem codeAt_slot1 (c0 c1 c2 c3 j : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (hj : j % 4 = 1) :
    codeAt (fun _ => pack4 c0 c1 c2 c3) j = c1 := by
  simp only [codeAt, pack4, hj]; omega

/-- Slot 2 of a packed byte. -/
theorem codeAt_slot2 (c0 c1 c2 c3 j : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (hj : j % 4 = 2) :
    codeAt (fun _ => pack4 c0 c1 c2 c3) j = c2 := by
  simp only [codeAt, pack4, hj]; omega

/-- Slot 3 of a packed byte. -/
theorem codeAt_slot3 (c0 c1 c2 c3 j : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4)
    (hj : j % 4 = 3) : codeAt (fun _ => pack4 c0 c1 c2 c3) j = c3 := by
  simp only [codeAt, pack4, hj]; omega

/-- Packing is injective on valid codes. -/
theorem pack4_injective (a0 a1 a2 a3 b0 b1 b2 b3 : Nat)
    (ha0 : a0 < 4) (ha1 : a1 < 4) (ha2 : a2 < 4) (ha3 : a3 < 4)
    (hb0 : b0 < 4) (hb1 : b1 < 4) (hb2 : b2 < 4) (hb3 : b3 < 4)
    (h : pack4 a0 a1 a2 a3 = pack4 b0 b1 b2 b3) :
    a0 = b0 ∧ a1 = b1 ∧ a2 = b2 ∧ a3 = b3 := by
  unfold pack4 at h; omega

-- § block geometry

/-- 16 code bytes + 2 scale bytes = 18 bytes per 64 elements. -/
theorem block_bytes : 64 / 4 + 2 = 18 := by decide

/-- `n` blocks of 64 elements occupy `18 n` bytes. -/
theorem q2_0_bytes (n : Nat) : (64 * n) / 64 * 18 = 18 * n := by omega

-- § quantizer clamp

/-- `clamp(x + 1, 0, 3)` as a natural number code. -/
def clamp03 (x : Int) : Nat := (max 0 (min 3 x)).toNat

/-- The clamped code is a valid 2-bit code. -/
theorem clamp03_lt (x : Int) : clamp03 x < 4 := by
  unfold clamp03; omega

/-- A symbol in {-1,0,1,2} is reproduced exactly: the requantization fixed point. -/
theorem clamp03_fixed (c : Nat) (h : c < 4) : clamp03 (((c : Int) - 1) + 1) = c := by
  unfold clamp03; omega

/-- Symbols outside the grid saturate to the end codes. -/
theorem clamp03_saturate (x : Int) (h : 2 ≤ x) : clamp03 (x + 1) = 3 := by
  unfold clamp03; omega

-- § alignment

/-- Padding up to a multiple of 32. -/
def align32 (n : Nat) : Nat := (n + 31) / 32 * 32

/-- Alignment never shrinks. -/
theorem align32_ge (n : Nat) : n ≤ align32 n := by unfold align32; omega

/-- Alignment gives a multiple of 32. -/
theorem align32_mod (n : Nat) : align32 n % 32 = 0 := by unfold align32; omega

/-- Alignment adds fewer than 32 bytes. -/
theorem align32_lt (n : Nat) : align32 n < n + 32 := by unfold align32; omega

/-- The next tensor offset is aligned and does not overlap the previous tensor. -/
theorem next_offset_ok (offset len : Nat) :
    offset + len ≤ align32 (offset + len) ∧ align32 (offset + len) % 32 = 0 :=
  ⟨align32_ge _, align32_mod _⟩

/-- Alignment is idempotent. -/
theorem align32_idem (n : Nat) : align32 (align32 n) = align32 n := by
  unfold align32; omega

-- § container constants

/-- Bytes of the ASCII magic `GGUF`, read as a little-endian u32. -/
theorem magic_le : 0x47 + 0x47 * 256 + 0x55 * 65536 + 0x46 * 16777216 = 0x46554747 := by decide

/-- The metadata value-type ids are pairwise distinct. -/
theorem meta_ids_nodup : ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] : List Nat).Nodup := by decide

end Strata.GgufQ2WriterPacking
