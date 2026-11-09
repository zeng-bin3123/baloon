/-!
# MODULE: weight_loader_segments
SOURCES: src/core/weights.cpp
Dense-weight loader: per-plane segment accounting (codes, fp16-or-f32 scales, offsets), chunked staging that
never splits an fp16 element, promoted-type 4→2 byte offsets, and compact arena placement.

## Source semantics
* An fp16 scale plane is half as many source bytes as destination bytes and must have an even destination count.
* Segment sums are checked against the index's recorded totals.
* A widening chunk is even-length; chunks make progress while two or more bytes remain.
* Promoted tensors read 4 B/elem and write 2 B/elem; destination offset is `done / 2`.
* Skipped rows get no bytes; kept rows are re-placed at 256-byte alignment.

## Form template identifiers
scales_fp16 → srcScaleBytes ; CHUNK → chunkBase ; Seg sums → segSrcTotal/segDstTotal ; compaction → compactStep
-/

namespace Strata.Weights

-- § plane sizes

/-- Source byte count of a scale plane. -/
def srcScaleBytes : Bool → Nat → Nat
  | true, s => s / 2
  | false, s => s

/-- Destination byte count of the scale plane is the index's `scales_bytes`. -/
def segSrcTotal (codes scales offset : Nat) (fp16 : Bool) : Nat :=
  codes + srcScaleBytes fp16 scales + offset

/-- Destination total of a quantized tensor. -/
def segDstTotal (codes scales offset : Nat) : Nat := codes + scales + offset

/-- An even fp16 scale plane doubles exactly when widened. -/
theorem srcScale_double (s : Nat) (h : s % 2 = 0) : srcScaleBytes true s * 2 = s := by
  show s / 2 * 2 = s
  omega

/-- Without fp16 scales the planes are copied verbatim, so source and destination agree. -/
theorem seg_total_copy (codes scales offset : Nat) :
    segSrcTotal codes scales offset false = segDstTotal codes scales offset := by
  rfl

/-- With fp16 scales the destination exceeds the source by exactly the widened half. -/
theorem seg_total_fp16 (codes scales offset : Nat) (h : scales % 2 = 0) :
    segDstTotal codes scales offset = segSrcTotal codes scales offset true + scales / 2 := by
  unfold segDstTotal segSrcTotal
  show codes + scales + offset = codes + scales / 2 + offset + scales / 2
  omega

-- § chunking

/-- Base chunk: at most 8 MiB. -/
def chunkBase (r : Nat) : Nat := if r < 8388608 then r else 8388608

/-- Chunk for a widening plane: drop one byte if the chunk is odd. -/
def chunkWiden (r : Nat) : Nat := if chunkBase r % 2 = 1 then chunkBase r - 1 else chunkBase r

/-- The base chunk never exceeds what remains. -/
theorem chunkBase_le (r : Nat) : chunkBase r ≤ r := by
  unfold chunkBase; split <;> omega

/-- The base chunk is even whenever the remainder is even. -/
theorem chunkBase_even (r : Nat) (h : r % 2 = 0) : chunkBase r % 2 = 0 := by
  unfold chunkBase; split <;> omega

/-- The base chunk is at least 2 when at least 2 bytes remain. -/
theorem chunkBase_ge (r : Nat) (h : 2 ≤ r) : 2 ≤ chunkBase r := by
  unfold chunkBase; split <;> omega

/-- A widening chunk never overruns the plane. -/
theorem chunkWiden_le (r : Nat) : chunkWiden r ≤ r := by
  have := chunkBase_le r
  unfold chunkWiden; split <;> omega

/-- A widening chunk is always even: it never splits an fp16 element. -/
theorem chunkWiden_even (r : Nat) : chunkWiden r % 2 = 0 := by
  unfold chunkWiden; split <;> omega

/-- A widening loop makes progress while two or more bytes remain, so it terminates. -/
theorem chunkWiden_pos (r : Nat) (h : 2 ≤ r) : 0 < chunkWiden r := by
  have := chunkBase_ge r h
  unfold chunkWiden; split <;> omega

/-- On an even plane a widening chunk equals the base chunk (no byte is left stranded). -/
theorem chunkWiden_eq_of_even (r : Nat) (h : r % 2 = 0) : chunkWiden r = chunkBase r := by
  have := chunkBase_even r h
  unfold chunkWiden; split <;> omega

-- § promoted conversions

/-- Promoted tensors: the destination offset `done / 2` composes across 4-byte-aligned chunks. -/
theorem promoted_offsets (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + n / 2 = (done + n) / 2 := by
  omega

/-- Widening offsets compose too: `done * 2 + n * 2 = (done + n) * 2`. -/
theorem widen_offsets (done n : Nat) : done * 2 + n * 2 = (done + n) * 2 := by
  omega

/-- A promoted tensor's expected byte counts: 4 per element in, 2 per element out. -/
theorem promoted_ratio (elems : Nat) : elems * 4 = 2 * (elems * 2) := by
  omega

-- § compact arena placement

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- One placement step of a kept row: cursor advances by the aligned destination size. -/
def compactStep (at_ dst : Nat) : Nat := at_ + align256 dst

/-- Placement keeps every cursor 256-aligned. -/
theorem compactStep_mod (at_ dst : Nat) (h : at_ % 256 = 0) : compactStep at_ dst % 256 = 0 := by
  unfold compactStep align256; omega

/-- Placed rows never overlap: the cursor moves past the row's bytes. -/
theorem compactStep_ge (at_ dst : Nat) : at_ + dst ≤ compactStep at_ dst := by
  unfold compactStep align256; omega

end Strata.Weights
