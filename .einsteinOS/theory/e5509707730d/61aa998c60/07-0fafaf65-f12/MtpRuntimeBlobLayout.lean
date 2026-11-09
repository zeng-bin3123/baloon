/-!
# MODULE: mtp_runtime_blob_layout
SOURCES: tools/mtp_rt.py
Integer model of the MTP runtime files: the expert blob (gate/up row interleave, code plane then scale plane),
its byte budget, dense.bin padding to 256 and the Gemma "+1" norm weight.

## Source semantics
BLOB = 3*(H*FF*18/64). Blob = interleaved gate/up codes, down codes, interleaved gate/up scales, down scales.
Interleave: slot 2r = gate row r, slot 2r+1 = up row r. dense.bin tensors are padded to 256-byte offsets.

## Form template identifiers
`H, FF, NE` -> `h, ff, ne`; `BLOB` -> `blob`; `blob_of` -> plane sizes; `pad = (-off) % 256` -> `pad256`.
-/

namespace StrataMtpRt

/-- Hidden size. -/
def h : Nat := 2560
/-- FFN width. -/
def ff : Nat := 640
/-- Experts. -/
def ne : Nat := 512

/-- Blob size: three roles of H*FF*18/64 bytes. -/
def blob : Nat := 3 * (h * ff * 18 / 64)

/-- The blob is 1,382,400 bytes, matching the main pack. -/
theorem blob_value : blob = 1382400 := by decide

/-- Gate/up code plane: 1280 rows x 40 blocks x 16 code bytes. -/
theorem gu_codes : 2 * ff * (h / 64) * 16 = 819200 := by decide
/-- Down code plane: 2560 rows x 10 blocks x 16 code bytes. -/
theorem down_codes : h * (ff / 64) * 16 = 409600 := by decide
/-- Gate/up scale plane: fp16 per block. -/
theorem gu_scales : 2 * ff * (h / 64) * 2 = 102400 := by decide
/-- Down scale plane. -/
theorem down_scales : h * (ff / 64) * 2 = 51200 := by decide

/-- The four planes add up to the blob exactly. -/
theorem planes_sum : 819200 + 409600 + 102400 + 51200 = blob := by decide

/-- The expert file is 512 blobs. -/
theorem experts_file : ne * blob = 707788800 := by decide

/-- Q2_0 row widths in bytes: gate/up row 720, down row 180. -/
theorem row_bytes : h / 64 * 18 = 720 ∧ ff / 64 * 18 = 180 := by decide

-- § gate/up interleave

/-- Slot of gate row r. -/
def gateSlot (r : Nat) : Nat := 2 * r
/-- Slot of up row r. -/
def upSlot (r : Nat) : Nat := 2 * r + 1

/-- Gate slots are even and recover the row. -/
theorem gate_slot_inv (r : Nat) : gateSlot r % 2 = 0 ∧ gateSlot r / 2 = r := by
  unfold gateSlot; omega

/-- Up slots are odd and recover the row. -/
theorem up_slot_inv (r : Nat) : upSlot r % 2 = 1 ∧ upSlot r / 2 = r := by
  unfold upSlot; omega

/-- Every slot is exactly one gate or one up row (the interleave is a bijection). -/
theorem slot_decomp (s : Nat) : s = 2 * (s / 2) + s % 2 := by omega

/-- Gate and up never collide. -/
theorem slots_disjoint (r r' : Nat) : gateSlot r ≠ upSlot r' := by
  unfold gateSlot upSlot; omega

/-- For rows below 640 the slots stay below 1280. -/
theorem slots_bound (r : Nat) (hr : r < 640) : gateSlot r < 1280 ∧ upSlot r < 1280 := by
  unfold gateSlot upSlot; omega

/-- A slot's code bytes end inside the gate/up code plane. -/
theorem code_slot_in_plane (s b : Nat) (hs : s < 1280) (hb : b < 40) : s * 640 + b * 16 + 16 ≤ 819200 := by
  omega

/-- A slot's scale bytes end inside the gate/up scale plane. -/
theorem scale_slot_in_plane (s b : Nat) (hs : s < 1280) (hb : b < 40) : (s * 40 + b) * 2 + 2 ≤ 102400 := by
  omega

-- § dense.bin

/-- Padding to a 256-byte offset. -/
def pad256 (n : Nat) : Nat := (256 - n % 256) % 256

/-- Padding is below 256 and aligns the offset. -/
theorem pad256_spec (n : Nat) : pad256 n < 256 ∧ (n + pad256 n) % 256 = 0 := by
  unfold pad256; omega

/-- Gemma RMSNorm folds 1+w; with integer-valued w the stored value is w+1 (shift is invertible). -/
theorem gemma_plus_one (w : Int) : (w + 1) - 1 = w := by omega

/-- Quantising a row to Q8_0 requires a multiple of 32 columns; all large projections have 2560 columns. -/
theorem q8_cols : 2560 % 32 = 0 := by decide

end StrataMtpRt
