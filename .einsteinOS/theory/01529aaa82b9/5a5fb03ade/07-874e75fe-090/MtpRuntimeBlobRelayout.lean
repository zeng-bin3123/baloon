/-!
# MODULE: mtp_runtime_blob_relayout
`tools/mtp_rt.py`: lossless relayout of the MTP Q2_0 experts into the engine blob (gate/up rows interleaved,
code plane then scale plane), and the 256-byte padded dense file.

## Source semantics
* BLOB = 3 * (H*FF*18/64) = 1,382,400 B with H=2560, FF=640.
* interleaved row 2r is gate row r, row 2r+1 is up row r.
* each dense tensor starts at a multiple of 256.

## Form template identifiers
H, FF, NE   -> hidden, ffn width, expert count
slot        -> interleaved row index
-/

namespace Strata.MtpRuntimeBlobRelayout

-- § blob size

/-- Blob size from geometry. -/
theorem blob_bytes : 3 * (2560 * 640 * 18 / 64) = 1382400 := by decide

/-- The four planes add up to the blob exactly. -/
theorem planes_sum :
    1280 * (2560 / 64 * 16) + 2560 * (640 / 64 * 16) + 1280 * (2560 / 64 * 2) + 2560 * (640 / 64 * 2) = 1382400 := by
  decide

/-- Code planes and scale planes separately: 18 = 16 + 2 per block. -/
theorem block_split : 16 + 2 = 18 := by decide

-- § row interleave

/-- Interleaved slot of a gate row. -/
def gateSlot (r : Nat) : Nat := 2 * r

/-- Interleaved slot of an up row. -/
def upSlot (r : Nat) : Nat := 2 * r + 1

/-- Gate slots are even. -/
theorem gate_even (r : Nat) : gateSlot r % 2 = 0 := by unfold gateSlot; omega

/-- Up slots are odd. -/
theorem up_odd (r : Nat) : upSlot r % 2 = 1 := by unfold upSlot; omega

/-- Gate and up slots never collide. -/
theorem slots_disjoint (r s : Nat) : gateSlot r ≠ upSlot s := by unfold gateSlot upSlot; omega

/-- Both slot maps stay inside the 1280 interleaved rows. -/
theorem slots_in_range (r : Nat) (h : r < 640) : gateSlot r < 1280 ∧ upSlot r < 1280 := by
  unfold gateSlot upSlot; omega

/-- The interleave is invertible: row = slot / 2, role = slot % 2. -/
theorem slot_inverse (r : Nat) : gateSlot r / 2 = r ∧ upSlot r / 2 = r := by
  unfold gateSlot upSlot; omega

/-- Every interleaved slot is a gate or an up row. -/
theorem slot_surjective (s : Nat) (h : s < 1280) :
    (∃ r, r < 640 ∧ s = gateSlot r) ∨ (∃ r, r < 640 ∧ s = upSlot r) := by
  by_cases hs : s % 2 = 0
  · exact Or.inl ⟨s / 2, by omega, by unfold gateSlot; omega⟩
  · exact Or.inr ⟨s / 2, by omega, by unfold upSlot; omega⟩

-- § dense padding

/-- Padding before a dense tensor so that it starts at a multiple of 256. -/
def pad256 (off : Nat) : Nat := (256 - off % 256) % 256

/-- After padding the offset is a multiple of 256. -/
theorem pad256_aligned (off : Nat) : (off + pad256 off) % 256 = 0 := by unfold pad256; omega

/-- Padding is smaller than the alignment. -/
theorem pad256_lt (off : Nat) : pad256 off < 256 := by unfold pad256; omega

/-- Per-expert blob offsets are `e * BLOB`, non-overlapping. -/
theorem expert_blobs_disjoint (e e' : Nat) (h : e < e') : e * 1382400 + 1382400 ≤ e' * 1382400 := by
  omega

/-- Total size of 512 experts. -/
theorem experts_total : 512 * 1382400 = 707788800 := by decide

end Strata.MtpRuntimeBlobRelayout
