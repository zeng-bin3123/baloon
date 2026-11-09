/-!
# MODULE: expert_arena_blob_layout
SOURCES: tools/pack_layer.py
Integer model of the pack-layout expert blob: gate/up row interleave, code plane then scale plane, exact byte
budget of 1,382,400, per-expert contiguity inside the role tensors, and the total arena size.

## Source semantics
Blob = gate_up.codes (2*640 rows x 640 B) | down.codes (2560 rows x 160 B) | gate_up.scales (2*640*40 fp16)
| down.scales (2560*10 fp16) = 1,382,400 B. Expert e is a contiguous 460,800-byte run in each role tensor.
Conversion is a pure relayout, so bytes are only moved, never changed.

## Form template identifiers
`H, FF, NE, QK, BLOCK_BYTES` -> `hid, ff, ne, qk, blockBytes`; `ROW_BYTES_GU/D`, `SCALES_GU/D`, `ROLE_BYTES`,
`BLOB_BYTES`, `blob_offsets` -> same-named definitions.
-/

namespace StrataPackLayer

-- § constants

/-- Hidden size. -/
def hid : Nat := 2560
/-- FFN width. -/
def ff : Nat := 640
/-- Experts per layer. -/
def ne : Nat := 512
/-- Q2_0 block elements. -/
def qk : Nat := 64
/-- Q2_0 block bytes. -/
def blockBytes : Nat := 18
/-- Code bytes of one gate/up row (2560 x 2 bits). -/
def rowBytesGu : Nat := hid * 2 / 8
/-- Code bytes of one down row (640 x 2 bits). -/
def rowBytesD : Nat := ff * 2 / 8
/-- Scale blocks per gate/up row. -/
def scalesGu : Nat := hid / qk
/-- Scale blocks per down row. -/
def scalesD : Nat := ff / qk
/-- Bytes per role per expert. -/
def roleBytes : Nat := hid * ff * blockBytes / qk
/-- Bytes per expert blob. -/
def blobBytes : Nat := 3 * roleBytes

/-- Gate/up row code bytes. -/
theorem row_bytes_gu : rowBytesGu = 640 := by decide
/-- Down row code bytes. -/
theorem row_bytes_d : rowBytesD = 160 := by decide
/-- Scales per gate/up row. -/
theorem scales_gu : scalesGu = 40 := by decide
/-- Scales per down row. -/
theorem scales_d : scalesD = 10 := by decide
/-- Role bytes per expert. -/
theorem role_bytes : roleBytes = 460800 := by decide
/-- The blob is exactly 1,382,400 bytes. -/
theorem blob_bytes : blobBytes = 1382400 := by decide

-- § plane offsets

/-- Offset of gate_up.codes. -/
def offGuCodes : Nat := 0
/-- Offset of down.codes. -/
def offDCodes : Nat := offGuCodes + 2 * ff * rowBytesGu
/-- Offset of gate_up.scales. -/
def offGuScales : Nat := offDCodes + hid * rowBytesD
/-- Offset of down.scales. -/
def offDScales : Nat := offGuScales + 2 * ff * scalesGu * 2
/-- End of the blob. -/
def offEnd : Nat := offDScales + hid * scalesD * 2

/-- The four planes tile the blob exactly with no gap or overlap. -/
theorem planes_tile : offEnd = blobBytes := by decide

/-- The plane offsets are 0, 819200, 1228800, 1331200 and the blob ends at 1382400. -/
theorem plane_offsets : offGuCodes = 0 ∧ offDCodes = 819200 ∧ offGuScales = 1228800 ∧ offDScales = 1331200 := by
  decide

/-- The plane offsets are strictly increasing. -/
theorem planes_ordered : offGuCodes < offDCodes ∧ offDCodes < offGuScales ∧ offGuScales < offDScales ∧
    offDScales < offEnd := by decide

/-- The explicit sum from the source's assertion. -/
theorem blob_sum : 2 * 640 * 640 + 2560 * 160 + 2 * 640 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

-- § per-expert contiguity

/-- Expert e occupies bytes [e*460800, (e+1)*460800) of a role tensor. -/
def expertStart (e : Nat) : Nat := e * 460800

/-- Experts are consecutive and non-overlapping. -/
theorem expert_runs_adjacent (e : Nat) : expertStart e + 460800 = expertStart (e + 1) := by
  unfold expertStart; omega

/-- All 512 expert runs fit inside a role tensor of 512*460800 bytes. -/
theorem expert_in_tensor (e : Nat) (he : e < 512) : expertStart e + 460800 ≤ 512 * 460800 := by
  unfold expertStart; omega

/-- A role tensor holds elements/512 per expert and that equals 460800 bytes of Q2_0. -/
theorem per_expert_elems : (2560 * 640 * 512) / 512 * 18 / 64 = 460800 := by decide

-- § interleave

/-- Row index of gate row r (0) or up row r (1) in the interleaved stack. -/
def stackRow (r k : Nat) : Nat := 2 * r + k

/-- Stacking gate and up along a new axis then flattening gives rows g0,u0,g1,u1,...: a bijection onto 0..1279. -/
theorem stack_row_inv (r k : Nat) (hk : k < 2) : stackRow r k / 2 = r ∧ stackRow r k % 2 = k := by
  unfold stackRow; omega

/-- Interleaved rows stay in range. -/
theorem stack_row_bound (r k : Nat) (hr : r < 640) (hk : k < 2) : stackRow r k < 1280 := by
  unfold stackRow; omega

/-- Inverse used by the verifier: slot -> (row, role), and recombination. -/
theorem slot_roundtrip (s : Nat) : stackRow (s / 2) (s % 2) = s := by
  unfold stackRow; omega

/-- The verifier's code offset for a (slot, block) ends inside the gate/up code plane. -/
theorem verifier_code_in_plane (slot b : Nat) (hs : slot < 1280) (hb : b < 40) :
    offGuCodes + slot * rowBytesGu + b * 16 + 16 ≤ offDCodes := by
  unfold offGuCodes offDCodes rowBytesGu ff hid; omega

/-- The verifier's scale offset for a (slot, block) ends inside the gate/up scale plane. -/
theorem verifier_scale_in_plane (slot b : Nat) (hs : slot < 1280) (hb : b < 40) :
    offGuScales + (slot * scalesGu + b) * 2 + 2 ≤ offDScales := by
  simp only [offGuCodes, offDCodes, offGuScales, offDScales, scalesGu, rowBytesGu, rowBytesD, ff, hid, qk]; omega

/-- Down code offsets end inside the down code plane. -/
theorem verifier_down_code_in_plane (row b : Nat) (hr : row < 2560) (hb : b < 10) :
    offDCodes + row * rowBytesD + b * 16 + 16 ≤ offGuScales := by
  simp only [offGuCodes, offDCodes, offGuScales, rowBytesGu, rowBytesD, ff, hid]; omega

-- § arena size

/-- Total arena bytes for n layers. -/
def arenaBytes (n : Nat) : Nat := n * ne * blobBytes

/-- The full 48-layer arena is 33,973,862,400 bytes (31.64 GiB). -/
theorem arena_48 : arenaBytes 48 = 33973862400 := by decide

/-- One layer is 512 blobs of 1,382,400 bytes. -/
theorem arena_1 : arenaBytes 1 = 707788800 := by decide

/-- The arena grows linearly: each layer adds the same bytes. -/
theorem arena_succ (n : Nat) : arenaBytes (n + 1) = arenaBytes n + 707788800 := by
  unfold arenaBytes ne blobBytes roleBytes hid ff blockBytes qk; omega

end StrataPackLayer
