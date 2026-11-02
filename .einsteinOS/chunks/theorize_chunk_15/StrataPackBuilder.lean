/-!
# MODULE: strata_pack_builder
Layout arithmetic and invariants of the `strata-pack` builder/verifier: LSB-first code packing, 64-byte plane
alignment, FP16-or-FP32 scale selection, and the expert-arena row/slot indexing and size accounting.

## Source semantics
* `pack_codes`/`unpack_codes` store `8/bits` codes per byte, LSB first, and are inverse on valid codes.
* `put` pads the segment body to a multiple of 64 before each plane; offsets are `base + len(body)`.
  Alignment is therefore relative to the segment; it is absolute only if `base % 64 = 0`
  (note: the builder advances `seg_base` by the raw blob length, which need not be a multiple of 64).
* `choose_scale_dtype` returns FP16 only if the round trip is exact, else the original FP32 values.
* The expert arena interleaves gate/up rows (slot `2*row + role`) and splits each block into a 2-byte scale
  plane and 16-byte code plane; `rebuild_role` inverts that indexing.
* One expert blob is 1,382,400 bytes; 48 layers × 512 experts is 31.64 GiB.

## Form template identifiers
* `packByte4`/`unpackLo`/`unpackHi` ← `pack_codes`/`unpack_codes` with bits = 4
* `padTo`                ← `pad = (-len(body)) % ALIGN`
* `chooseScale`          ← `choose_scale_dtype`
* `rowOf`/`bInOf`        ← `row, b_in = blk // S, blk % S`
* `gateUpSlot`           ← `slot = 2*row + (0 if gate else 1)`
-/

namespace StrataPack

-- § code packing

/-- Pack two 4-bit codes into a byte, LSB first. -/
def packByte4 (a b : Nat) : Nat := a % 16 + 16 * (b % 16)

/-- Low code of a 4-bit-packed byte. -/
def unpackLo (x : Nat) : Nat := x % 16

/-- High code of a 4-bit-packed byte. -/
def unpackHi (x : Nat) : Nat := x / 16

/-- A 4-bit-packed byte fits in a byte. -/
theorem packByte4_lt (a b : Nat) : packByte4 a b < 256 := by unfold packByte4; omega

/-- 4-bit packing round-trips on valid codes. -/
theorem pack4_roundtrip (a b : Nat) (ha : a < 16) (hb : b < 16) :
    unpackLo (packByte4 a b) = a ∧ unpackHi (packByte4 a b) = b := by
  unfold unpackLo unpackHi packByte4; omega

/-- Pack four 2-bit codes into a byte, LSB first. -/
def packByte2 (a b c d : Nat) : Nat := a % 4 + 4 * (b % 4) + 16 * (c % 4) + 64 * (d % 4)

/-- A 2-bit-packed byte fits in a byte. -/
theorem packByte2_lt (a b c d : Nat) : packByte2 a b c d < 256 := by unfold packByte2; omega

/-- 2-bit packing is injective on valid codes. -/
theorem packByte2_injective (a b c d a' b' c' d' : Nat)
    (h1 : a < 4) (h2 : b < 4) (h3 : c < 4) (h4 : d < 4)
    (g1 : a' < 4) (g2 : b' < 4) (g3 : c' < 4) (g4 : d' < 4)
    (h : packByte2 a b c d = packByte2 a' b' c' d') : a = a' ∧ b = b' ∧ c = c' ∧ d = d' := by
  unfold packByte2 at h; omega

/-- Codes per byte times bits per code is 8 for every supported width. -/
theorem codes_per_byte : [2, 4, 8].all (fun b => (8 / b) * b == 8) = true := by decide

/-- A Q2_0 block's 64 codes occupy exactly 16 packed bytes. -/
theorem q2_codes_bytes_per_block : 64 / (8 / 2) = 16 := by decide

-- § alignment

/-- Padding to the next multiple of 64. -/
def padTo (len : Nat) : Nat := (64 - len % 64) % 64

/-- After padding, the length is a multiple of 64. -/
theorem padTo_aligned (len : Nat) : (len + padTo len) % 64 = 0 := by unfold padTo; omega

/-- Padding is strictly less than the alignment. -/
theorem padTo_lt (len : Nat) : padTo len < 64 := by unfold padTo; omega

/-- Already aligned lengths get no padding. -/
theorem padTo_zero_of_aligned (len : Nat) (h : len % 64 = 0) : padTo len = 0 := by
  unfold padTo; omega

/-- A plane's absolute offset is congruent to the segment base mod 64: alignment is relative to `base`. -/
theorem plane_offset_mod (base len : Nat) : (base + (len + padTo len)) % 64 = base % 64 := by
  unfold padTo; omega

/-- Plane offsets are 64-aligned in absolute terms exactly when the segment base is. -/
theorem plane_offset_aligned_iff (base len : Nat) :
    (base + (len + padTo len)) % 64 = 0 ↔ base % 64 = 0 := by
  unfold padTo; omega

/-- The builder's `seg_base` advance (raw blob length) preserves alignment only for aligned blobs. -/
theorem seg_base_advance_aligned (base blob : Nat) (hb : base % 64 = 0) (hl : blob % 64 = 0) :
    (base + blob) % 64 = 0 := by omega

-- § scale dtype choice

/-- `choose_scale_dtype` on an abstract round-trip function `r` (FP32→FP16→FP32). -/
def chooseScale {α : Type} [DecidableEq α] (r : α → α) (s : α) : α := if r s = s then r s else s

/-- Whichever branch is taken, the decoded scales equal the original: the choice is lossless. -/
theorem chooseScale_lossless {α : Type} [DecidableEq α] (r : α → α) (s : α) : chooseScale r s = s := by
  unfold chooseScale
  split
  · assumption
  · rfl

-- § expert arena indexing

/-- Row of a flat block index given `S` blocks per row. -/
def rowOf (blk S : Nat) : Nat := blk / S

/-- In-row block of a flat block index. -/
def bInOf (blk S : Nat) : Nat := blk % S

/-- `(row, b_in)` reconstructs the flat block index. -/
theorem row_bin_recompose (blk S : Nat) : S * rowOf blk S + bInOf blk S = blk :=
  Nat.div_add_mod blk S

/-- In-row block index is within the row. -/
theorem bIn_lt (blk S : Nat) (hS : 0 < S) : bInOf blk S < S := Nat.mod_lt _ hS

/-- Gate/up row interleave slot: gate = 0, up = 1. -/
def gateUpSlot (row role : Nat) : Nat := 2 * row + role

/-- Distinct (row, role) pairs get distinct slots, so gate and up never collide. -/
theorem gateUpSlot_injective (r1 r2 a b : Nat) (ha : a < 2) (hb : b < 2)
    (h : gateUpSlot r1 a = gateUpSlot r2 b) : r1 = r2 ∧ a = b := by
  unfold gateUpSlot at h; omega

/-- Gate and up of the same row occupy adjacent slots. -/
theorem gate_up_adjacent (row : Nat) : gateUpSlot row 1 = gateUpSlot row 0 + 1 := by
  unfold gateUpSlot; omega

-- § expert arena sizes

/-- Per-role bytes for one expert: `2560 * 640 / 64` blocks of 18 bytes. -/
theorem role_bytes : 2560 * 640 / 64 * 18 = 460800 := by decide

/-- One expert blob (gate + up + down) is exactly 1,382,400 bytes. -/
theorem blob_bytes : 3 * (2560 * 640 / 64 * 18) = 1382400 := by decide

/-- A block splits into a 2-byte scale and 16 code bytes: 18 bytes. -/
theorem block_split : 2 + 16 = 18 := by decide

/-- 48 layers of 512 experts is 33,973,862,400 bytes (31.64 GiB). -/
theorem arena_total : 48 * 512 * 1382400 = 33973862400 := by decide

/-- The arena size in whole GiB-hundredths is 3164 (31.64 GiB). -/
theorem arena_gib_hundredths : 33973862400 * 100 / 2 ^ 30 = 3164 := by decide

/-- Blocks per expert tensor times bytes agree with the arena's per-role accounting. -/
theorem blocks_per_role : 2560 * 640 / 64 = 25600 := by decide

end StrataPack
