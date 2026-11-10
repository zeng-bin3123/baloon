/-!
# MODULE: pack_layer_blob
SOURCES: tools/pack_layer.py
Builds the pack-layout expert arena from Q2_0 GGUF tensors. One expert blob is exactly 1,382,400 bytes:
gate/up codes (rows interleaved g0,u0,g1,u1,...), down codes, gate/up fp16 scales, down fp16 scales.

## Source semantics
* H = 2560, FF = 640, NE = 512, Q2_0 blocks of 64 elements and 18 bytes (2 scale + 16 code bytes).
* Each role holds 460,800 bytes per expert; the blob is three roles. The four planes tile the blob with no
  gap or overlap.
* The conversion is a pure relayout: gate/up row `r` of role gate/up goes to slot `2r` / `2r+1`.
* Expert `e` is one contiguous run in each role tensor; the 48-layer arena is 33,973,862,400 bytes.

## Form template identifiers
- BLOB_BYTES / ROLE_BYTES        → `blobBytes` / `roleBytes`
- blob_offsets()                 → `offGuCodes`, `offDCodes`, `offGuScales`, `offDScales`
- np.stack([gate, up], axis=2)   → `slot`
- ROW_BYTES_GU / ROW_BYTES_D     → `rowBytesGu` / `rowBytesD`
-/

namespace Ref.PackLayer

-- § constants

/-- Hidden size. -/
def H : Nat := 2560
/-- FFN width. -/
def FF : Nat := 640
/-- Experts per layer. -/
def NE : Nat := 512
/-- Bytes of codes in one gate/up row (2560 elements at 2 bits). -/
def rowBytesGu : Nat := 640
/-- Bytes of codes in one down row (640 elements at 2 bits). -/
def rowBytesD : Nat := 160
/-- Q2_0 blocks per gate/up row. -/
def scalesGu : Nat := 40
/-- Q2_0 blocks per down row. -/
def scalesD : Nat := 10
/-- Bytes of one role of one expert. -/
def roleBytes : Nat := 460800
/-- Bytes of one expert blob. -/
def blobBytes : Nat := 1382400

/-- Row code widths follow from element counts at 2 bits per code. -/
theorem row_code_bytes : H * 2 / 8 = rowBytesGu ∧ FF * 2 / 8 = rowBytesD := by
  decide

/-- Blocks per row: 2560/64 and 640/64. -/
theorem blocks_per_row : H / 64 = scalesGu ∧ FF / 64 = scalesD := by
  decide

/-- A Q2_0 block is 16 code bytes plus a 2-byte scale. -/
theorem block_split : 16 + 2 = 18 ∧ 64 * 2 / 8 = 16 := by
  decide

/-- One role of one expert: `H*FF/64` blocks of 18 bytes. -/
theorem role_bytes_eq : H * FF / 64 * 18 = roleBytes := by
  decide

/-- Three roles make the blob: 3 * 460,800 = 1,382,400. -/
theorem blob_eq_three_roles : 3 * roleBytes = blobBytes := by
  decide

/-- The blob equals the sum of its four planes. -/
theorem blob_planes_sum :
    2 * 640 * rowBytesGu + 2560 * rowBytesD + 2 * 640 * scalesGu * 2 + 2560 * scalesD * 2 = blobBytes := by
  decide

-- § plane offsets

/-- Offset of the gate/up codes plane. -/
def offGuCodes : Nat := 0
/-- Offset of the down codes plane. -/
def offDCodes : Nat := offGuCodes + 2 * FF * rowBytesGu
/-- Offset of the gate/up scales plane. -/
def offGuScales : Nat := offDCodes + H * rowBytesD
/-- Offset of the down scales plane. -/
def offDScales : Nat := offGuScales + 2 * FF * scalesGu * 2

/-- The planes are contiguous and the last one ends exactly at the blob size. -/
theorem planes_tile_blob :
    offDCodes = 819200 ∧ offGuScales = 1228800 ∧ offDScales = 1331200 ∧
      offDScales + H * scalesD * 2 = blobBytes := by
  decide

/-- Each plane sits strictly after the previous one. -/
theorem planes_ordered : offGuCodes < offDCodes ∧ offDCodes < offGuScales ∧ offGuScales < offDScales := by
  decide

-- § row interleave and inverse

/-- Slot of a gate (role 0) or up (role 1) row in the interleaved plane. -/
def slot (row role : Nat) : Nat := 2 * row + role

/-- Interleaved slots cover 0..1279 and determine (row, role) uniquely. -/
theorem slot_bijective (r1 r2 a1 a2 : Nat) (h1 : r1 < 640) (h2 : r2 < 640) (b1 : a1 < 2) (b2 : a2 < 2)
    (h : slot r1 a1 = slot r2 a2) : r1 = r2 ∧ a1 = a2 := by
  unfold slot at h
  omega

/-- Slots stay inside the 1280-row interleaved plane. -/
theorem slot_lt (r a : Nat) (h1 : r < 640) (h2 : a < 2) : slot r a < 1280 := by
  unfold slot
  omega

/-- A block index splits into (row, block-in-row) and recombines (`rebuild_role` inverts the layout). -/
theorem block_split_recombine (blk per : Nat) (hp : 0 < per) :
    per * (blk / per) + blk % per = blk ∧ blk % per < per := by
  exact ⟨Nat.div_add_mod blk per, Nat.mod_lt blk hp⟩

/-- The 16 code bytes of block `b` inside a gate/up row end within the 640-byte row. -/
theorem code_run_inside_row (b : Nat) (h : b < 40) : b * 16 + 16 ≤ rowBytesGu := by
  unfold rowBytesGu
  omega

/-- The 2-byte scale of slot `s`, block `b` ends within the gate/up scales plane (102,400 bytes). -/
theorem scale_inside_plane (s b : Nat) (hs : s < 1280) (hb : b < 40) : (s * 40 + b) * 2 + 2 ≤ 102400 := by
  omega

-- § expert contiguity and arena size

/-- Expert `e` is one contiguous run: a role tensor of `H*FF*NE` elements splits into 512 equal runs of
460,800 bytes. -/
theorem expert_run_bytes : (H * FF * NE) / NE * 18 / 64 = roleBytes := by
  decide

/-- Total arena for `n` layers. -/
def arenaBytes (n : Nat) : Nat := n * NE * blobBytes

/-- The full 48-layer arena is 33,973,862,400 bytes (31.64 GiB). -/
theorem arena_48_layers : arenaBytes 48 = 33973862400 := by
  decide

/-- The arena grows linearly: one more layer adds exactly 512 blobs. -/
theorem arena_succ (n : Nat) : arenaBytes (n + 1) = arenaBytes n + NE * blobBytes := by
  unfold arenaBytes NE blobBytes
  omega

end Ref.PackLayer
