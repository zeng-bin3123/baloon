/-!
# MODULE: expert_blob_layout
SOURCES: tools/pack_layer.py, tools/strata_pack.py
The expert arena (`experts.bin`): one 1,382,400-byte blob per (layer, expert), holding the Q2_0
codes and fp16 scales of gate, up and down in four planes, with gate and up ROWS interleaved
(g0,u0,g1,u1,…). `rebuild_role` is the inverse relayout that the verifier uses.

## Source semantics
* H = 2560, FF = 640, NE = 512, Q2_0 blocks of 64 elements / 18 bytes.
* planes: gate_up.codes (2·640 rows × 640 B) at 0, down.codes (2560 rows × 160 B) at 819,200,
  gate_up.scales (2·640·40 fp16) at 1,228,800, down.scales (2560·10 fp16) at 1,331,200, ending at
  exactly 1,382,400 = 3 × 460,800, the source's per-expert Q2_0 bytes.
* block `blk` of a gate/up role sits in row `blk / 40`, column `blk % 40`, and interleaved slot
  `2·row + role`; down blocks sit in row `blk / 10`, column `blk % 10`.
* every (role, block) gets its own 16-byte code slot and 2-byte scale slot inside its plane; the
  relayout is a pure permutation, which is why dequant(pack) = dequant(ggml) bit for bit.
* expert e of a role tensor is the contiguous run [e·460800, (e+1)·460800); blob (layer, e) starts
  at (layer·512 + e)·1,382,400; 48 layers total 31.64 GiB.

## Form template identifiers
pack_layer constants → `hDim`,`ffDim`,`nExperts`,`blobBytes`; blob_offsets → `offGuCodes`,
`offDCodes`, `offGuScales`, `offDScales`; rebuild_role → `guCodeOff`, `guScaleOff`, `dCodeOff`,
`dScaleOff`; np.stack row interleave → `guSlot`; experts.bin addressing → `blobStart`.
-/

namespace Strata.ExpertBlob

-- § constants

/-- Hidden size. -/
def hDim : Nat := 2560
/-- Expert FFN size. -/
def ffDim : Nat := 640
/-- Experts per layer. -/
def nExperts : Nat := 512
/-- Q2_0 elements and bytes per block. -/
def qk : Nat := 64
def blockBytes : Nat := 18
/-- Bytes of one role (gate, up or down) of one expert in the source. -/
def roleBytes : Nat := hDim * ffDim * blockBytes / qk
/-- Bytes of one blob. -/
def blobBytes : Nat := 3 * roleBytes

/-- The blob is exactly 1,382,400 bytes, the module's first assertion. -/
theorem blob_size : roleBytes = 460800 ∧ blobBytes = 1382400 := by decide

/-- The plane sizes add up to the blob, the module's second assertion. -/
theorem planes_sum :
    2 * 640 * (hDim * 2 / 8) + 2560 * (ffDim * 2 / 8) + 2 * 640 * (hDim / qk) * 2 +
      2560 * (ffDim / qk) * 2 = blobBytes := by decide

-- § plane offsets

def offGuCodes : Nat := 0
def offDCodes : Nat := offGuCodes + 2 * ffDim * 640
def offGuScales : Nat := offDCodes + hDim * 160
def offDScales : Nat := offGuScales + 2 * ffDim * 40 * 2

/-- The offsets of `blob_offsets()`, with the last plane ending exactly at the blob size. -/
theorem offsets_values :
    offDCodes = 819200 ∧ offGuScales = 1228800 ∧ offDScales = 1331200 ∧
      offDScales + hDim * 10 * 2 = blobBytes := by decide

/-- Numeric values of the plane offsets and the blob size, for rewriting. -/
theorem offGuCodes_val : offGuCodes = 0 := rfl
theorem offDCodes_val : offDCodes = 819200 := by decide
theorem offGuScales_val : offGuScales = 1228800 := by decide
theorem offDScales_val : offDScales = 1331200 := by decide
theorem blobBytes_val : blobBytes = 1382400 := by decide

-- § rebuild_role addressing

/-- Interleaved gate/up row slot: role 0 = gate, 1 = up. -/
def guSlot (row role : Nat) : Nat := 2 * row + role

/-- Code offset of block `blk` of a gate/up role. -/
def guCodeOff (role blk : Nat) : Nat := offGuCodes + guSlot (blk / 40) role * 640 + blk % 40 * 16
/-- Scale offset of block `blk` of a gate/up role. -/
def guScaleOff (role blk : Nat) : Nat := offGuScales + (guSlot (blk / 40) role * 40 + blk % 40) * 2
/-- Code offset of block `blk` of the down role. -/
def dCodeOff (blk : Nat) : Nat := offDCodes + blk / 10 * 160 + blk % 10 * 16
/-- Scale offset of block `blk` of the down role. -/
def dScaleOff (blk : Nat) : Nat := offDScales + (blk / 10 * 10 + blk % 10) * 2

/-- Blocks per role per expert. -/
theorem blocks_per_role : hDim * ffDim / qk = 25600 ∧ 25600 * blockBytes = roleBytes := by decide

/-- The numpy `stack([gate, up], axis=2)` places role `r` of row `i` at row `2i + r`, the same
slot `rebuild_role` reads. -/
theorem stack_row_index (i r : Nat) (hr : r < 2) : (2 * i + r) / 2 = i ∧ (2 * i + r) % 2 = r := by
  omega

/-- Gate/up code slots lie inside the gate_up.codes plane, 16 bytes each. -/
theorem gu_code_in_plane (role blk : Nat) (hr : role < 2) (hb : blk < 25600) :
    guCodeOff role blk + 16 ≤ offDCodes := by
  simp only [guCodeOff, guSlot, offGuCodes_val, offDCodes_val]; omega

/-- Distinct (role, block) pairs get distinct gate/up code slots. -/
theorem gu_code_injective (r1 b1 r2 b2 : Nat) (h1 : r1 < 2) (h2 : r2 < 2)
    (h : guCodeOff r1 b1 = guCodeOff r2 b2) : r1 = r2 ∧ b1 = b2 := by
  unfold guCodeOff guSlot offGuCodes at h; omega

/-- Code slots are 16-byte aligned, so distinct slots never overlap. -/
theorem gu_code_aligned (role blk : Nat) : guCodeOff role blk % 16 = 0 := by
  unfold guCodeOff guSlot offGuCodes; omega

/-- Gate/up scale slots lie inside gate_up.scales and are injective. -/
theorem gu_scale_in_plane_injective (r1 b1 r2 b2 : Nat) (h1 : r1 < 2) (h2 : r2 < 2)
    (hb1 : b1 < 25600) :
    (offGuScales ≤ guScaleOff r1 b1 ∧ guScaleOff r1 b1 + 2 ≤ offDScales) ∧
      (guScaleOff r1 b1 = guScaleOff r2 b2 → r1 = r2 ∧ b1 = b2) := by
  simp only [guScaleOff, guSlot, offGuScales_val, offDScales_val]
  constructor
  · omega
  · intro h; omega

/-- Down code slots lie inside down.codes and are injective. -/
theorem d_code_in_plane_injective (b1 b2 : Nat) (hb1 : b1 < 25600) :
    (offDCodes ≤ dCodeOff b1 ∧ dCodeOff b1 + 16 ≤ offGuScales) ∧
      (dCodeOff b1 = dCodeOff b2 → b1 = b2) := by
  simp only [dCodeOff, offDCodes_val, offGuScales_val]
  constructor
  · omega
  · intro h; omega

/-- Down scale slots lie inside down.scales and are injective. -/
theorem d_scale_in_plane_injective (b1 b2 : Nat) (hb1 : b1 < 25600) :
    (offDScales ≤ dScaleOff b1 ∧ dScaleOff b1 + 2 ≤ blobBytes) ∧
      (dScaleOff b1 = dScaleOff b2 → b1 = b2) := by
  simp only [dScaleOff, offDScales_val, blobBytes_val]
  constructor
  · omega
  · intro h; omega

/-- Slot counts fill the planes exactly: an injective map with these counts is a permutation. -/
theorem slot_counts :
    2 * 25600 * 16 = offDCodes - offGuCodes ∧ 25600 * 16 = offGuScales - offDCodes ∧
      2 * 25600 * 2 = offDScales - offGuScales ∧ 25600 * 2 = blobBytes - offDScales := by decide

-- § arena addressing

/-- Start of blob (layer, expert) in `experts.bin`. -/
def blobStart (layer e : Nat) : Nat := (layer * nExperts + e) * blobBytes

/-- Blobs of distinct (layer, expert) pairs do not overlap. -/
theorem blobs_disjoint (l1 e1 l2 e2 : Nat) (he1 : e1 < 512) (he2 : e2 < 512)
    (h : (l1, e1) ≠ (l2, e2)) :
    blobStart l1 e1 + blobBytes ≤ blobStart l2 e2 ∨ blobStart l2 e2 + blobBytes ≤ blobStart l1 e1 := by
  unfold blobStart nExperts blobBytes roleBytes hDim ffDim blockBytes qk
  simp only [ne_eq, Prod.mk.injEq] at h
  have : l1 * 512 + e1 ≠ l2 * 512 + e2 := by omega
  rcases Nat.lt_or_gt_of_ne this with hlt | hgt
  · left
    have := Nat.mul_le_mul_right (2560 * 640 * 18 / 64 * 3) (show l1 * 512 + e1 + 1 ≤ l2 * 512 + e2 by omega)
    rw [Nat.add_mul, Nat.one_mul] at this; exact this
  · right
    have := Nat.mul_le_mul_right (2560 * 640 * 18 / 64 * 3) (show l2 * 512 + e2 + 1 ≤ l1 * 512 + e1 by omega)
    rw [Nat.add_mul, Nat.one_mul] at this; exact this

/-- Expert `e` of a source role tensor is the contiguous run starting at `e * roleBytes`, and the
512 runs tile the tensor. -/
theorem expert_runs_tile : nExperts * roleBytes = hDim * ffDim * nExperts * blockBytes / qk := by
  decide

/-- The full 48-layer arena is 33,973,862,400 bytes, i.e. 31.64 GiB. -/
theorem arena_size :
    48 * nExperts * blobBytes = 33973862400 ∧
      3164 * 2 ^ 30 ≤ 100 * 33973862400 ∧ 100 * 33973862400 < 3165 * 2 ^ 30 := by decide

end Strata.ExpertBlob
