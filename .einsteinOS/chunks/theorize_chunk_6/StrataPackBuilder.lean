/-!
# MODULE: strata_pack_builder
The pack builder / verifier: bit-packing of canonical codes, 64-byte plane alignment, FP16-vs-FP32 scale
selection, the expert blob inverse used by `verify_experts`, and the manifest-driven decode rule.

## Source semantics
* `pack_codes` stores `8 // bits` codes per byte, LSB first (`byte |= code << (k*bits)`); `unpack_codes` inverts it.
* `put` pads the body to a multiple of `ALIGN = 64` before each plane, so each plane starts on a 64-byte
  boundary RELATIVE to its blob; the absolute file offset is `seg_base + len(body)`.
* A canonical scale is stored FP16 only when the FP16 round trip is exact, otherwise FP32; `verify` re-checks
  bit-exactness, so a wrong choice is a failed build, never a silently lossy pack.
* `rebuild_role` inverts the arena layout: gate/up rows are interleaved (`slot = 2*row + (0|1)`), codes and
  scales live in separate planes, each block is `fp16 d` then 16 code bytes.
* `decode_entry` uses only the manifest: `(code + code_bias) * scale [+ offset]`, or the IQ4 table lookup.

## Form template identifiers
align := ALIGN ; plane_pad := (-len(body)) % ALIGN ; pack_codes := pack_codes ; unpack_codes := unpack_codes ;
blob_bytes := BLOB_BYTES ; gate_up_slot := 2*row + role ; scales_per_row_gu := SCALES_GU ;
scales_per_row_down := SCALES_D ; code_bits := code_bits(form)
-/

namespace Strata.PackBuilder

-- § code bit widths

/-- The canonical integer forms. -/
inductive Form where
  | S2
  | S4
  | S8
  deriving DecidableEq

/-- Bits per code of each canonical integer form. -/
def codeBits : Form → Nat
  | Form.S2 => 2
  | Form.S4 => 4
  | Form.S8 => 8

/-- The three integer forms store 2, 4 and 8 bits per code. -/
theorem code_bits_values : codeBits Form.S2 = 2 ∧ codeBits Form.S4 = 4 ∧ codeBits Form.S8 = 8 := by
  decide

/-- Every integer form's width divides 8, so `8 // bits` codes fill a byte exactly. -/
theorem code_bits_divide_byte : 8 % 2 = 0 ∧ 8 % 4 = 0 ∧ 8 % 8 = 0 := by
  decide

-- § bit packing

/-- Two 4-bit codes pack into one byte (LSB first) and unpack to themselves. -/
theorem pack4_roundtrip (c0 c1 : Nat) (h0 : c0 < 16) (h1 : c1 < 16) :
    (c0 + 16 * c1) < 256 ∧ (c0 + 16 * c1) % 16 = c0 ∧ ((c0 + 16 * c1) / 16) % 16 = c1 := by
  omega

/-- Four 2-bit codes pack into one byte (LSB first) and unpack to themselves. -/
theorem pack2_roundtrip (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) < 256 ∧ (c0 + 4 * c1 + 16 * c2 + 64 * c3) % 4 = c0 ∧
      ((c0 + 4 * c1 + 16 * c2 + 64 * c3) / 4) % 4 = c1 ∧ ((c0 + 4 * c1 + 16 * c2 + 64 * c3) / 16) % 4 = c2 ∧
      ((c0 + 4 * c1 + 16 * c2 + 64 * c3) / 64) % 4 = c3 := by
  omega

/-- Unpacking then repacking a byte restores it: the byte format is lossless for 4-bit codes. -/
theorem unpack4_pack (b : Nat) (h : b < 256) : b % 16 + 16 * ((b / 16) % 16) = b := by
  omega

/-- Unpacking then repacking a byte restores it for 2-bit codes. -/
theorem unpack2_pack (b : Nat) (h : b < 256) :
    b % 4 + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) = b := by
  omega

/-- Bytes occupied by `n` codes of `bits` bits when `n` is a multiple of the codes per byte. -/
def codeBytes (n bits : Nat) : Nat := n * bits / 8

/-- A 64-element Q2_0 block's 2-bit codes occupy exactly its 16 source code bytes. -/
theorem q2_block_code_bytes : codeBytes 64 2 = 16 := by
  decide

/-- A 256-element 4-bit block occupies 128 bytes; an 8-bit block occupies 256. -/
theorem s4_s8_block_code_bytes : codeBytes 256 4 = 128 ∧ codeBytes 256 8 = 256 := by
  decide

-- § 64-byte plane alignment

/-- Padding inserted before a plane: `(-len) % 64`. -/
def planePad (len : Nat) : Nat := (64 - len % 64) % 64

/-- The pad is less than one alignment unit. -/
theorem planePad_lt (len : Nat) : planePad len < 64 := by
  unfold planePad
  omega

/-- After padding, the body length is a multiple of 64. -/
theorem planePad_aligned (len : Nat) : (len + planePad len) % 64 = 0 := by
  unfold planePad
  omega

/-- An already aligned body gets no padding. -/
theorem planePad_zero (len : Nat) (h : len % 64 = 0) : planePad len = 0 := by
  unfold planePad
  omega

/-- A plane's absolute offset is `seg_base + len(body)`; when `seg_base` is 64-aligned the plane is
aligned in the file too. -/
theorem plane_abs_aligned (base len : Nat) (hb : base % 64 = 0) : (base + (len + planePad len)) % 64 = 0 := by
  have h := planePad_aligned len
  omega

/-- Observation: `seg_base` advances by the blob length, which is NOT padded at its end, so absolute
alignment of later tensors holds only when each blob length is a multiple of 64. -/
theorem next_base_aligned (base blobLen : Nat) (hb : base % 64 = 0) (hl : blobLen % 64 = 0) :
    (base + blobLen) % 64 = 0 := by
  omega

-- § scale precision

/-- Bytes per stored scale/offset entry. -/
def scaleBytes (isFp16 : Bool) : Nat := if isFp16 then 2 else 4

/-- Scale planes are either 2 or 4 bytes per entry. -/
theorem scaleBytes_values : scaleBytes true = 2 ∧ scaleBytes false = 4 := by
  decide

/-- Expert-arena scales are FP16 by construction: 40 scales per 2560-wide gate/up row. -/
theorem arena_scales_per_gu_row : 2560 / 64 = 40 ∧ 640 / 64 = 10 := by
  decide

-- § expert blob (inverse layout used by `rebuild_role`)

def ROW_BYTES_GU : Nat := 640
def ROW_BYTES_D : Nat := 160
def SCALES_GU : Nat := 40
def SCALES_D : Nat := 10
def O_D_CODES : Nat := 1280 * ROW_BYTES_GU
def O_GU_SC : Nat := O_D_CODES + 2560 * ROW_BYTES_D
def O_D_SC : Nat := O_GU_SC + 1280 * SCALES_GU * 2
def BLOB_BYTES : Nat := O_D_SC + 2560 * SCALES_D * 2

/-- The blob is exactly the 1,382,400 bytes architecture §3.2 fixes. -/
theorem blob_bytes_fixed : BLOB_BYTES = 1382400 := by
  decide

/-- Each role (gate, up, down) has `2560 * 640 / 64 = 25600` Q2_0 blocks. -/
theorem blocks_per_role : 2560 * 640 / 64 = 25600 := by
  decide

/-- Gate/up: the block's row is `blk / 40`, below the 640 rows of one role. -/
theorem gu_row_in_range (blk : Nat) (h : blk < 25600) : blk / SCALES_GU < 640 := by
  unfold SCALES_GU
  omega

/-- Gate/up: interleaved slot `2*row + role` stays inside the 1280 slots. -/
theorem gu_slot_in_range (row role : Nat) (hr : row < 640) (hrole : role < 2) : 2 * row + role < 1280 := by
  omega

/-- Gate and up of the same row occupy adjacent, distinct slots. -/
theorem gu_slots_distinct (row : Nat) : 2 * row + 0 ≠ 2 * row + 1 := by
  omega

/-- Gate/up code byte offset of block `(slot, b_in)` stays inside the gate/up code plane. -/
theorem gu_code_offset_lt (slot b_in : Nat) (hs : slot < 1280) (hb : b_in < SCALES_GU) :
    slot * ROW_BYTES_GU + b_in * 16 + 16 ≤ O_D_CODES := by
  unfold O_D_CODES ROW_BYTES_GU SCALES_GU at *
  omega

/-- Gate/up scale byte offset (two bytes each) stays inside the gate/up scale plane. -/
theorem gu_scale_offset_lt (slot b_in : Nat) (hs : slot < 1280) (hb : b_in < SCALES_GU) :
    (slot * SCALES_GU + b_in) * 2 + 2 ≤ 1280 * SCALES_GU * 2 := by
  unfold SCALES_GU at *
  omega

/-- Down: the block's row is `blk / 10`, below the 2560 rows. -/
theorem down_row_in_range (blk : Nat) (h : blk < 25600) : blk / SCALES_D < 2560 := by
  unfold SCALES_D
  omega

/-- Down code byte offset stays inside the down code plane. -/
theorem down_code_offset_lt (row b_in : Nat) (hr : row < 2560) (hb : b_in < SCALES_D) :
    row * ROW_BYTES_D + b_in * 16 + 16 ≤ 2560 * ROW_BYTES_D := by
  unfold SCALES_D ROW_BYTES_D at *
  omega

/-- Down scale byte offset stays inside the down scale plane. -/
theorem down_scale_offset_lt (row b_in : Nat) (hr : row < 2560) (hb : b_in < SCALES_D) :
    (row * SCALES_D + b_in) * 2 + 2 ≤ 2560 * SCALES_D * 2 := by
  unfold SCALES_D at *
  omega

/-- The block index is recovered from `(row, b_in)` as `row * per_row + b_in`: the inverse is a bijection. -/
theorem block_index_bijection (per_row row b_in : Nat) (hp : 0 < per_row) (hb : b_in < per_row) :
    (row * per_row + b_in) / per_row = row ∧ (row * per_row + b_in) % per_row = b_in := by
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hb]
    simp
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt hb

/-- Each rebuilt Q2_0 block is `2` scale bytes then `16` code bytes: `18` bytes. -/
theorem rebuilt_block_bytes : 2 + 16 = 18 := by
  decide

/-- A rebuilt role has `25600 * 18` bytes, the size of a Q2_0 tensor slice of that many blocks. -/
theorem rebuilt_role_bytes : 25600 * 18 = 460800 := by
  decide

-- § verification sampling

/-- `verify_experts` checks experts `0, every, 2*every, ...`; for `every = 32` that is 16 of 512 per layer. -/
theorem experts_checked_per_layer : (512 + 32 - 1) / 32 = 16 := by
  decide

/-- The sampled expert ids are all valid expert indices. -/
theorem sampled_expert_valid (k every : Nat) (hk : k < 16) (he : every = 32) : k * every < 512 := by
  subst he
  omega

end Strata.PackBuilder
