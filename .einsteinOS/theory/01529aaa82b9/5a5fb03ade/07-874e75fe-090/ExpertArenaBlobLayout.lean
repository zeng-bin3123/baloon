/-!
# MODULE: expert_arena_blob_layout
`tools/pack_layer.py`: builds the pack-layout expert arena.  One expert blob is exactly 1,382,400 bytes:
gate/up codes, down codes, gate/up scales, down scales.  The conversion is a pure relayout.

## Source semantics
* gate_up codes = 2*640 rows * 640 B; down codes = 2560 rows * 160 B;
  gate_up scales = 2*640 * 40 * 2 B; down scales = 2560 * 10 * 2 B.
* the four planes are contiguous and end exactly at BLOB_BYTES.
* expert `e` is one contiguous 460,800-byte run inside each role tensor.
* 48 layers * 512 experts * blob = 31.64 GiB.

## Form template identifiers
OFF            -> blob_offsets()
ROLE_BYTES     -> bytes per role per expert
BLOB_BYTES     -> bytes per expert blob
-/

namespace Strata.ExpertArenaBlobLayout

-- § plane sizes and offsets

/-- Row byte widths: 2560 two-bit codes = 640 B; 640 codes = 160 B. -/
theorem row_bytes : 2560 * 2 / 8 = 640 ∧ 640 * 2 / 8 = 160 := by decide

/-- Per-role bytes per expert. -/
theorem role_bytes : 2560 * 640 * 18 / 64 = 460800 := by decide

/-- Three roles make the blob. -/
theorem blob_bytes : 3 * 460800 = 1382400 := by decide

/-- The four plane sizes. -/
theorem plane_sizes :
    2 * 640 * 640 = 819200 ∧ 2560 * 160 = 409600 ∧ 2 * 640 * 40 * 2 = 102400 ∧ 2560 * 10 * 2 = 51200 := by
  decide

/-- The planes sum to the blob exactly. -/
theorem planes_sum : 819200 + 409600 + 102400 + 51200 = 1382400 := by decide

/-- Plane offsets, in order. -/
def guCodes : Nat := 0
def dCodes : Nat := guCodes + 819200
def guScales : Nat := dCodes + 409600
def dScales : Nat := guScales + 102400
def blobEnd : Nat := dScales + 51200

/-- The final offset is the blob size. -/
theorem blob_end : blobEnd = 1382400 := by decide

/-- Planes are ordered without overlap. -/
theorem planes_ordered :
    guCodes + 819200 ≤ dCodes ∧ dCodes + 409600 ≤ guScales ∧ guScales + 102400 ≤ dScales ∧
      dScales + 51200 ≤ blobEnd := by decide

/-- Concrete offsets. -/
theorem offsets_values : dCodes = 819200 ∧ guScales = 1228800 ∧ dScales = 1331200 := by decide

-- § interleave and in-plane addressing

/-- gate/up code byte offset of slot `s`, block `b`: `slot*640 + b*16` stays inside the plane. -/
theorem gu_code_inbounds (s b : Nat) (hs : s < 1280) (hb : b < 40) :
    s * 640 + b * 16 + 16 ≤ 819200 := by omega

/-- down code byte offset of row `r`, block `b`. -/
theorem d_code_inbounds (r b : Nat) (hr : r < 2560) (hb : b < 10) :
    r * 160 + b * 16 + 16 ≤ 409600 := by omega

/-- gate/up scale offset of slot `s`, block `b`. -/
theorem gu_scale_inbounds (s b : Nat) (hs : s < 1280) (hb : b < 40) :
    (s * 40 + b) * 2 + 2 ≤ 102400 := by omega

/-- down scale offset of row `r`, block `b`. -/
theorem d_scale_inbounds (r b : Nat) (hr : r < 2560) (hb : b < 10) :
    (r * 10 + b) * 2 + 2 ≤ 51200 := by omega

/-- Distinct (row, block) pairs of a gate/up row map to distinct code offsets. -/
theorem gu_code_injective (s b s' b' : Nat) (hb : b < 40) (hb' : b' < 40)
    (h : s * 640 + b * 16 = s' * 640 + b' * 16) : s = s' ∧ b = b' := by
  omega

-- § expert runs inside role tensors

/-- Start of expert `e` inside a role tensor. -/
def expertStart (e : Nat) : Nat := e * 460800

/-- Experts are disjoint, contiguous runs. -/
theorem experts_disjoint (e e' : Nat) (h : e < e') : expertStart e + 460800 ≤ expertStart e' := by
  unfold expertStart; omega

/-- Expert 511 ends at the end of a 512-expert role tensor. -/
theorem last_expert_end : expertStart 511 + 460800 = 512 * 460800 := by decide

-- § arena size

/-- Blob address of (layer, expert) in the arena. -/
def blobAddr (layer e : Nat) : Nat := (layer * 512 + e) * 1382400

/-- Addresses are strictly increasing in (layer, expert) order. -/
theorem blobAddr_next (layer e : Nat) (he : e < 511) :
    blobAddr layer (e + 1) = blobAddr layer e + 1382400 := by
  unfold blobAddr; omega

/-- The last expert of a layer is followed by the first of the next. -/
theorem blobAddr_layer_step (layer : Nat) : blobAddr (layer + 1) 0 = blobAddr layer 511 + 1382400 := by
  unfold blobAddr; omega

/-- A full 48-layer arena is 33,973,862,400 bytes. -/
theorem full_arena : 48 * 512 * 1382400 = 33973862400 := by decide

/-- Any in-range blob ends inside the full arena. -/
theorem blob_in_arena (layer e : Nat) (hl : layer < 48) (he : e < 512) :
    blobAddr layer e + 1382400 ≤ 33973862400 := by
  unfold blobAddr; omega

end Strata.ExpertArenaBlobLayout
