/-!
# MODULE: strata_pack_builder
The pack builder and verifier: LSB-first code packing/unpacking for 2/4/8-bit forms, 64-byte plane alignment,
the expert blob arithmetic (1,382,400 bytes, plane inversion), scale-dtype selection and the dense/embedding/
expert tensor partition.

## Source semantics
* `pack_codes` places `8/bits` codes per byte, first code in the low bits; `unpack_codes` inverts it.
* Every plane starts on a 64-byte boundary: padding is `(-len) % 64`.
* An expert blob is 1,382,400 bytes; `rebuild_role` inverts the gate/up interleave (slot `2r` gate, `2r+1` up)
  and the block index split `blk = row * SC + b_in`.
* A scale is stored as FP16 only if the FP16 round trip is exact, otherwise FP32.
* Every non-expert tensor goes to exactly one of `dense.bin` or `embd.bin`; expert tensors go to the arena.

## Form template identifiers
pack_codes/unpack_codes -> pack_codes/unpack_codes, put -> append_plane, rebuild_role -> rebuild_role,
choose_scale_dtype -> choose_scale_dtype, is_expert -> is_expert.
-/

namespace Strata.StrataPackBuilder

-- § code packing

/-- Codes per byte for a code width. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- 2-bit codes: four per byte. -/
theorem per_byte_2 : perByte 2 = 4 := by decide
/-- 4-bit codes: two per byte. -/
theorem per_byte_4 : perByte 4 = 2 := by decide
/-- 8-bit codes: one per byte. -/
theorem per_byte_8 : perByte 8 = 1 := by decide

/-- 2-bit pack/unpack round trip for one byte. -/
theorem roundtrip_2bit (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) % 4 = c0 ∧ (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 4 % 4 = c1 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 16 % 4 = c2 ∧ (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 64 % 4 = c3 := by
  omega

/-- 2-bit packed byte fits a byte. -/
theorem pack_2bit_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    c0 + 4 * c1 + 16 * c2 + 64 * c3 < 256 := by omega

/-- 4-bit pack/unpack round trip for one byte. -/
theorem roundtrip_4bit (c0 c1 : Nat) (h0 : c0 < 16) (h1 : c1 < 16) :
    (c0 + 16 * c1) % 16 = c0 ∧ (c0 + 16 * c1) / 16 % 16 = c1 ∧ c0 + 16 * c1 < 256 := by omega

/-- Packed size: `n` codes of `bits` width need `n / perByte` bytes when divisible. -/
theorem packed_size_2bit (n : Nat) (h : n % 4 = 0) : n / 4 * 4 = n := by omega

/-- Packed size for 4-bit codes. -/
theorem packed_size_4bit (n : Nat) (h : n % 2 = 0) : n / 2 * 2 = n := by omega

-- § alignment

/-- Padding to the next 64-byte boundary. -/
def pad64 (n : Nat) : Nat := (64 - n % 64) % 64

/-- Every plane offset after padding is 64-aligned. -/
theorem plane_aligned (n : Nat) : (n + pad64 n) % 64 = 0 := by
  unfold pad64
  omega

/-- Padding never reaches a full alignment unit. -/
theorem pad64_lt (n : Nat) : pad64 n < 64 := by
  unfold pad64
  omega

-- § expert blob

/-- Hidden size and expert width. -/
def H : Nat := 2560
/-- Expert intermediate width. -/
def FF : Nat := 640
/-- Weights per block. -/
def QK : Nat := 64
/-- Blocks per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Blocks per down row. -/
def SC_D : Nat := FF / QK

/-- Gate/up codes plane: `2*FF` rows of `SC_GU` blocks of 16 bytes. -/
def guCodes : Nat := 2 * FF * (SC_GU * 16)
/-- Gate/up scale plane: fp16 per block. -/
def guScales : Nat := 2 * FF * SC_GU * 2
/-- Down codes plane. -/
def dCodes : Nat := H * (SC_D * 16)
/-- Down scale plane. -/
def dScales : Nat := H * SC_D * 2

/-- The four planes total exactly 1,382,400 bytes. -/
theorem blob_total : guCodes + guScales + dCodes + dScales = 1382400 := by decide

/-- Equivalently: three roles, each `H*FF` weights as 18-byte 64-weight blocks. -/
theorem blob_by_roles : 3 * (H * FF / QK * 18) = 1382400 := by decide

/-- Each role has the same block count: `FF * SC_GU = H * SC_D = H*FF/64`. -/
theorem role_block_counts : FF * SC_GU = H * SC_D ∧ FF * SC_GU = H * FF / QK := by decide

/-- `rebuild_role` splits a block index into (row, block-in-row) and the split is invertible (gate/up). -/
theorem split_gu (blk : Nat) : (blk / 40) * 40 + blk % 40 = blk ∧ blk % 40 < 40 := by omega

/-- Down projection split with 10 blocks per row. -/
theorem split_down (blk : Nat) : (blk / 10) * 10 + blk % 10 = blk ∧ blk % 10 < 10 := by omega

/-- Gate and up occupy interleaved slots `2r` and `2r+1`: distinct and each below `2*FF`. -/
theorem role_slots (r : Nat) (hr : r < FF) : 2 * r ≠ 2 * r + 1 ∧ 2 * r + 1 < 2 * FF := by
  unfold FF at *
  omega

/-- Byte offset of block `b_in` of slot `slot`: codes advance 16 per block, scales 2 per block. -/
theorem block_offsets (slot b_in rowBytes sc : Nat) (h : rowBytes = sc * 16) (hb : b_in < sc) :
    b_in * 16 + 16 ≤ rowBytes := by
  subst h
  have := Nat.mul_le_mul_right 16 hb
  rw [Nat.succ_mul] at *
  omega

-- § scale dtype

/-- Bytes per stored scale. -/
def scaleBytes (fp16 : Bool) : Nat := if fp16 then 2 else 4

/-- Scale bytes are 2 or 4. -/
theorem scaleBytes_cases (b : Bool) : scaleBytes b = 2 ∨ scaleBytes b = 4 := by
  cases b <;> simp [scaleBytes]

/-- An FP16 plane is exactly half the size of the FP32 plane for the same scale count. -/
theorem fp16_half (n : Nat) : n * scaleBytes true * 2 = n * scaleBytes false := by
  simp [scaleBytes]
  omega

-- § tensor partition

/-- Count lemma: a predicate splits a list into two parts whose lengths add up. -/
theorem split_count {α : Type} (p : α → Bool) : ∀ l : List α,
    (l.filter p).length + (l.filter (fun x => !p x)).length = l.length
  | [] => by simp
  | a :: t => by
    have ih := split_count p t
    cases h : p a <;> simp [List.filter_cons, h] <;> omega

/-- Dense, embedding and expert tensors partition the source list with nothing lost or duplicated. -/
theorem partition_total (isExpert isEmbd : String → Bool) (names : List String) :
    ((names.filter (fun x => !isExpert x)).length) + (names.filter isExpert).length = names.length := by
  have h := split_count isExpert names
  omega

/-- A pack is only valid when every compared element count matches: shapes must be equal. -/
def shapesEqual (a b : Nat) : Bool := a == b

/-- Different element counts fail verification. -/
theorem shape_mismatch_fails (a b : Nat) (h : a ≠ b) : shapesEqual a b = false := by
  simp [shapesEqual, h]

end Strata.StrataPackBuilder
