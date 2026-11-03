/-!
# MODULE: mtp_runtime_relayout
Arithmetic of `tools/mtp_rt.py`: relayout of the MTP expert tensors into the engine's blob format
(`experts.bin`) and the 256-byte aligned `dense.bin`.

## Source semantics
* H = 2560, FF = 640, NE = 512 experts.  One blob = gate/up rows interleaved (slot 2r = gate r,
  slot 2r+1 = up r), then down rows; code planes first, fp16 scale planes after.
* `BLOB = 3 * (H*FF*18/64) = 1,382,400` bytes; the four planes sum to exactly BLOB.
* The interleave is a bijection on 2*FF rows.
* Dense tensors are padded so each starts on a 256-byte boundary.

## Form template identifiers
* srcRow / dstSlot   <- `inter[0::2] = gub[:FF]; inter[1::2] = gub[FF:]`
* BLOB               <- `BLOB = 3 * (H * FF * 18 // 64)`
* padTo256           <- `pad = (-off) % 256`
-/

namespace Strata.MtpRuntime

-- § Blob geometry

/-- Blob size from the source formula. -/
def blob : Nat := 3 * (2560 * 640 * 18 / 64)

/-- It equals the architecture constant 1,382,400. -/
theorem blob_value : blob = 1382400 := by decide

/-- Plane sizes: gate_up codes, down codes, gate_up scales, down scales. -/
theorem plane_sizes :
    1280 * (2560 / 64) * 16 = 819200 ∧ 2560 * (640 / 64) * 16 = 409600 ∧
    1280 * (2560 / 64) * 2 = 102400 ∧ 2560 * (640 / 64) * 2 = 51200 := by
  decide

/-- The four planes fill the blob exactly (the `assert len(out) == BLOB`). -/
theorem planes_sum_to_blob : 819200 + 409600 + 102400 + 51200 = blob := by decide

/-- GGUF raw row widths: 720 and 180 Q2_0 bytes per row. -/
theorem gguf_row_bytes : 2560 / 64 * 18 = 720 ∧ 640 / 64 * 18 = 180 := by decide

/-- The experts file is NE blobs. -/
theorem experts_file_bytes : 512 * blob = 707788800 := by decide

-- § Gate/up interleave

/-- Source row (in the gate-then-up layout) feeding interleaved slot `s`. -/
def srcRow (s : Nat) : Nat := if s % 2 = 0 then s / 2 else 640 + s / 2

/-- Interleaved slot receiving source row `r`. -/
def dstSlot (r : Nat) : Nat := if r < 640 then 2 * r else 2 * (r - 640) + 1

/-- Every slot reads a valid source row. -/
theorem srcRow_lt (s : Nat) (h : s < 1280) : srcRow s < 1280 := by
  unfold srcRow
  split <;> omega

/-- Every source row lands in a valid slot. -/
theorem dstSlot_lt (r : Nat) (h : r < 1280) : dstSlot r < 1280 := by
  unfold dstSlot
  split <;> omega

/-- The interleave is invertible: slot -> row -> slot is the identity. -/
theorem dstSlot_srcRow (s : Nat) (h : s < 1280) : dstSlot (srcRow s) = s := by
  unfold srcRow dstSlot
  split <;> split <;> omega

/-- And row -> slot -> row is the identity, so no row is lost or duplicated. -/
theorem srcRow_dstSlot (r : Nat) (h : r < 1280) : srcRow (dstSlot r) = r := by
  unfold srcRow dstSlot
  split <;> split <;> omega

/-- Gate rows go to even slots, up rows to odd slots. -/
theorem gate_even_up_odd (r : Nat) (h : r < 640) :
    dstSlot r % 2 = 0 ∧ dstSlot (640 + r) % 2 = 1 := by
  unfold dstSlot
  split <;> split <;> omega

-- § Dense file alignment

/-- Padding to the next 256-byte boundary, `(-off) % 256`. -/
def padTo256 (off : Nat) : Nat := (256 - off % 256) % 256

/-- A padded offset is 256-aligned. -/
theorem padTo256_aligns (off : Nat) : (off + padTo256 off) % 256 = 0 := by
  unfold padTo256
  omega

/-- Padding is smaller than the alignment. -/
theorem padTo256_lt (off : Nat) : padTo256 off < 256 := by
  unfold padTo256
  omega

/-- Q8_0 conversion of the large projections needs 32 | cols; the hidden size qualifies. -/
theorem hidden_is_q8_block_multiple : 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

end Strata.MtpRuntime
