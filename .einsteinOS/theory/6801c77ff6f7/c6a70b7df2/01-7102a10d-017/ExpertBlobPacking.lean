/-!
# MODULE: expert_blob_packing
SOURCES: tools/pack_layer.py
Builds the pack-layout expert arena from the GGUF: one expert blob is 1,382,400 bytes, a pure relayout of
Q2_0 codes and fp16 scales with gate/up rows interleaved.

## Source semantics
* blob = gate_up.codes (2 x 640 rows x 640 B) ++ down.codes (2560 x 160 B) ++ gate_up.scales (2x640x40xFP16)
  ++ down.scales (2560 x 10 x FP16) = 819,200 + 409,600 + 102,400 + 51,200.
* Expert e of a role tensor is a contiguous 460,800-byte run. Layer l, expert e is at (l*512+e)*blob.
* gate row r -> interleaved row 2r, up row r -> 2r+1.

## Form template identifiers
BLOB_BYTES -> expert_blob_size ; blob_offsets -> blob_region_offsets ; gu interleave -> gate_up_row_interleave ;
ROLE_BYTES -> role_bytes_per_expert
-/
set_option maxRecDepth 100000

namespace Strata.Pack

def H : Nat := 2560
def FF : Nat := 640
def NE : Nat := 512
def QK : Nat := 64
def BB : Nat := 18
def rowBytesGU : Nat := H * 2 / 8
def rowBytesD : Nat := FF * 2 / 8
def scalesGU : Nat := H / QK
def scalesD : Nat := FF / QK
def roleBytes : Nat := H * FF * BB / QK
def blobBytes : Nat := 3 * roleBytes
def offGuCodes : Nat := 0
def offDCodes : Nat := offGuCodes + 2 * FF * rowBytesGU
def offGuScales : Nat := offDCodes + H * rowBytesD
def offDScales : Nat := offGuScales + 2 * FF * scalesGU * 2
def offEnd : Nat := offDScales + H * scalesD * 2

/-- Row geometry: 640 B of codes per 2560-wide row, 160 B per 640-wide row, 40 and 10 scale blocks. -/
theorem row_geometry : rowBytesGU = 640 ∧ rowBytesD = 160 ∧ scalesGU = 40 ∧ scalesD = 10 := by decide

/-- Each role is 460,800 bytes per expert. -/
theorem role_bytes_value : roleBytes = 460800 := by decide

/-- The blob is exactly 1,382,400 bytes. -/
theorem blob_bytes_value : blobBytes = 1382400 := by decide

/-- The four regions tile the blob exactly: start offsets and end. -/
theorem blob_offsets_values :
    offGuCodes = 0 ∧ offDCodes = 819200 ∧ offGuScales = 1228800 ∧ offDScales = 1331200 ∧
    offEnd = blobBytes := by decide

/-- Region sizes sum: 819,200 + 409,600 + 102,400 + 51,200 = 1,382,400. -/
theorem region_sizes_sum : 819200 + 409600 + 102400 + 51200 = 1382400 := by decide

/-- Regions are consecutive and strictly increasing. -/
theorem regions_ordered : offGuCodes < offDCodes ∧ offDCodes < offGuScales ∧
    offGuScales < offDScales ∧ offDScales < offEnd := by decide

/-- A block-multiple of elements costs codes + scales: blocks*18 = blocks*16 + blocks*2. -/
theorem block_split (blocks : Nat) : blocks * 18 = blocks * 16 + blocks * 2 := by omega

/-- One role per expert splits into 409,600 B of codes and 51,200 B of scales. -/
theorem role_split : H * FF / 4 = 409600 ∧ H * FF / QK * 2 = 51200 ∧
    H * FF / 4 + H * FF / QK * 2 = roleBytes := by decide

/-- A fused role tensor of [2560,640,512] has per-expert elements giving exactly one role of bytes. -/
theorem per_expert_matches_role : H * FF * NE / NE * BB / QK = roleBytes := by decide

/-- Interleaving: gate row r goes to 2r and up row r to 2r+1, covering [0,1280) without overlap. -/
theorem interleave_bijective (r s : Nat) (hr : r < 640) (hs : s < 2) :
    2 * r + s < 1280 ∧ (2 * r + s) / 2 = r ∧ (2 * r + s) % 2 = s := by
  omega

/-- Every interleaved row index below 1280 comes from exactly one (row, role) pair. -/
theorem interleave_surjective (i : Nat) (hi : i < 1280) :
    2 * (i / 2) + i % 2 = i ∧ i / 2 < 640 := by
  omega

/-- Flattened (layer, expert) index is injective for expert < 512. -/
theorem blob_index_injective (l e l' e' : Nat) (he : e < 512) (he' : e' < 512)
    (h : l * 512 + e = l' * 512 + e') : l = l' ∧ e = e' := by
  omega

/-- A smaller flat index gives an earlier, non-overlapping blob. -/
theorem blobs_disjoint (i j : Nat) (h : i < j) : i * blobBytes + blobBytes ≤ j * blobBytes := by
  have h1 : (i + 1) * blobBytes ≤ j * blobBytes := Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl _)
  have h2 : (i + 1) * blobBytes = i * blobBytes + blobBytes := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Layer blocks of 512 blobs do not overlap across layers. -/
theorem layers_disjoint (l l' : Nat) (h : l < l') :
    (l * NE) * blobBytes + NE * blobBytes ≤ (l' * NE) * blobBytes := by
  have h1 : (l + 1) * NE * blobBytes ≤ l' * NE * blobBytes :=
    Nat.mul_le_mul (Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl _)) (Nat.le_refl _)
  have h2 : (l + 1) * NE * blobBytes = l * NE * blobBytes + NE * blobBytes := by
    rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  omega

/-- One layer of 512 experts is 707,788,800 bytes. -/
theorem layer_bytes : NE * blobBytes = 707788800 := by decide

/-- The full 48-layer arena is 33,973,862,400 bytes (the 34 GB arena). -/
theorem arena_bytes : 48 * NE * blobBytes = 33973862400 := by decide

/-- The 48-layer arena is 31.64 GiB (rounded). -/
theorem arena_gib : 3164 * 2 ^ 30 ≤ 48 * NE * blobBytes * 100 ∧
    48 * NE * blobBytes * 100 < 3165 * 2 ^ 30 := by decide

/-- Three roles per layer over 48 layers: 144 role tensors of 512 experts tile the arena. -/
theorem role_tensors_per_arena : 48 * 3 * NE * roleBytes = 48 * NE * blobBytes := by decide

end Strata.Pack
