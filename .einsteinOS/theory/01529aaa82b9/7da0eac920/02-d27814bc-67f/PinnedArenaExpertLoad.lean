/-!
# MODULE: pinned_arena_expert_load
The pinned host arena (sliced CUDA host registration, locked remainder) and the parallel, checksummed expert
load.

## Source semantics
* Uniform registration slices are `[i*s, (i+1)*s)` for `i < bytes / s`; they end at or before `bytes`, and the
  tail past the last full slice (fewer than `s` bytes) is left to the working-set lock.
* The per-layer checksum is FNV-1a folded chunk by chunk; folding chunks equals folding the concatenation, so
  the checksum does not depend on the chunking.
* Layer `L` of a uniform expert file starts at `L * blobs_per_layer * blob_bytes`; layers tile the file.
* In the chunk loop `pos + remaining` is invariant, so writes stay inside the layer's range.
* The bandwidth probe copies only whole chunks; when the chunk does not divide the buffer, the reported byte
  count overstates what was copied.

## Form template identifiers
* `uniform_bounds` -> `boundAt`, `sliceCount`
* `fnv1a64` -> `fnvStep`, `fnvFold`
* `load_experts` layer offsets -> `layerOff`
* `load_experts_ranges` chunk loop -> `chunkStep`
* `stream_bandwidth` -> `copiedPerPass`
-/

namespace Strata.Pinned

-- § sliced registration

/-- Number of full slices in a buffer. -/
def sliceCount (bytes slice : Nat) : Nat := bytes / slice

/-- Start of slice `i`. -/
def boundAt (slice i : Nat) : Nat := i * slice

/-- The last boundary does not exceed the buffer. -/
theorem lastBound_le (bytes slice : Nat) (hs : 0 < slice) :
    boundAt slice (sliceCount bytes slice) ≤ bytes := by
  unfold boundAt sliceCount
  have h := Nat.div_add_mod bytes slice
  have hc : bytes / slice * slice = slice * (bytes / slice) := Nat.mul_comm _ _
  omega

/-- The unregistered tail is shorter than one slice. -/
theorem tail_lt_slice (bytes slice : Nat) (hs : 0 < slice) :
    bytes - boundAt slice (sliceCount bytes slice) < slice := by
  unfold boundAt sliceCount
  have h := Nat.div_add_mod bytes slice
  have h2 := Nat.mod_lt bytes hs
  have hc : bytes / slice * slice = slice * (bytes / slice) := Nat.mul_comm _ _
  omega

/-- Consecutive boundaries differ by exactly one slice. -/
theorem bound_step (slice i : Nat) : boundAt slice (i + 1) = boundAt slice i + slice := by
  unfold boundAt; exact Nat.succ_mul i slice

/-- Registered plus locked bytes account for the whole buffer. -/
theorem registered_plus_tail (bytes slice : Nat) (hs : 0 < slice) :
    boundAt slice (sliceCount bytes slice) + (bytes - boundAt slice (sliceCount bytes slice)) = bytes := by
  have := lastBound_le bytes slice hs
  omega

/-- Registering a prefix of `j ≤ count` slices stays inside the buffer. -/
theorem prefix_registered_le (bytes slice j : Nat) (hs : 0 < slice) (hj : j ≤ sliceCount bytes slice) :
    boundAt slice j ≤ bytes := by
  have h1 : j * slice ≤ sliceCount bytes slice * slice := Nat.mul_le_mul_right _ hj
  have h2 := lastBound_le bytes slice hs
  unfold boundAt at *
  omega

-- § checksum

/-- One FNV-1a step. -/
def fnvStep (h b : UInt64) : UInt64 := (h ^^^ b) * 1099511628211

/-- FNV-1a fold over a byte list from a seed. -/
def fnvFold (seed : UInt64) (bs : List UInt64) : UInt64 := bs.foldl fnvStep seed

/-- The empty chunk leaves the hash unchanged. -/
theorem fnvFold_nil (seed : UInt64) : fnvFold seed [] = seed := rfl

/-- Chunked hashing equals hashing the concatenation, whatever the chunking. -/
theorem fnvFold_append (seed : UInt64) (a b : List UInt64) :
    fnvFold seed (a ++ b) = fnvFold (fnvFold seed a) b := by
  unfold fnvFold
  exact List.foldl_append

/-- Three chunks fold the same as two regrouped ones. -/
theorem fnvFold_assoc (seed : UInt64) (a b c : List UInt64) :
    fnvFold (fnvFold (fnvFold seed a) b) c = fnvFold seed (a ++ b ++ c) := by
  rw [fnvFold_append, fnvFold_append]

-- § layer layout

/-- Start of layer `L` in a uniform expert file. -/
def layerOff (bpl blob L : Nat) : Nat := L * bpl * blob

/-- Layers tile the file: layer `L+1` starts where layer `L` ends. -/
theorem layerOff_succ (bpl blob L : Nat) : layerOff bpl blob (L + 1) = layerOff bpl blob L + bpl * blob := by
  unfold layerOff
  rw [Nat.succ_mul, Nat.add_mul]

/-- Layer 0 starts at 0. -/
theorem layerOff_zero (bpl blob : Nat) : layerOff bpl blob 0 = 0 := by
  simp [layerOff]

/-- The last layer ends at `layers * bpl * blob`. -/
theorem layerOff_end (bpl blob layers : Nat) :
    layerOff bpl blob layers = layers * bpl * blob := rfl

-- § chunk loop

/-- One chunk iteration: chunk size, new position, new remaining. -/
def chunkStep (chunk pos remaining : Nat) : Nat × Nat × Nat :=
  let n := if remaining < chunk then remaining else chunk
  (n, pos + n, remaining - n)

/-- `pos + remaining` is invariant. -/
theorem chunk_invariant (chunk pos remaining : Nat) :
    (chunkStep chunk pos remaining).2.1 + (chunkStep chunk pos remaining).2.2 = pos + remaining := by
  unfold chunkStep
  by_cases h : remaining < chunk <;> simp [h] <;> omega

/-- A chunk never exceeds the chunk size. -/
theorem chunk_le (chunk pos remaining : Nat) : (chunkStep chunk pos remaining).1 ≤ chunk ∨ remaining < chunk := by
  unfold chunkStep
  by_cases h : remaining < chunk
  · exact Or.inr h
  · simp [h]

/-- A chunk never exceeds what remains. -/
theorem chunk_le_remaining (chunk pos remaining : Nat) :
    (chunkStep chunk pos remaining).1 ≤ remaining := by
  unfold chunkStep
  by_cases h : remaining < chunk <;> simp [h] <;> omega

/-- Writes stay inside the layer: `pos + n ≤ layerBytes` whenever `pos + remaining = layerBytes`. -/
theorem write_in_range (chunk pos remaining total : Nat) (h : pos + remaining = total) :
    pos + (chunkStep chunk pos remaining).1 ≤ total := by
  have := chunk_le_remaining chunk pos remaining
  omega

/-- A positive chunk makes progress while data remains. -/
theorem chunk_progress (chunk pos remaining : Nat) (hc : 0 < chunk) (hr : 0 < remaining) :
    0 < (chunkStep chunk pos remaining).1 := by
  unfold chunkStep
  by_cases h : remaining < chunk <;> simp [h] <;> omega

-- § bandwidth probe

/-- Bytes the probe actually copies per pass: whole chunks only. -/
def copiedPerPass (bytes chunk : Nat) : Nat := bytes / chunk * chunk

/-- The probe never copies more than the buffer. -/
theorem copied_le (bytes chunk : Nat) (hc : 0 < chunk) : copiedPerPass bytes chunk ≤ bytes := by
  unfold copiedPerPass
  have h := Nat.div_add_mod bytes chunk
  have hc2 : bytes / chunk * chunk = chunk * (bytes / chunk) := Nat.mul_comm _ _
  omega

/-- If the chunk divides the buffer, the reported byte count is exact. -/
theorem copied_exact (bytes chunk : Nat) (h : bytes % chunk = 0) : copiedPerPass bytes chunk = bytes := by
  unfold copiedPerPass
  have h1 := Nat.div_add_mod bytes chunk
  have hc2 : bytes / chunk * chunk = chunk * (bytes / chunk) := Nat.mul_comm _ _
  omega

/-- Otherwise the reported `bytes * iters` overstates the copied amount. -/
theorem copied_overreport (bytes chunk : Nat) (hc : 0 < chunk) (h : bytes % chunk ≠ 0) :
    copiedPerPass bytes chunk < bytes := by
  have h1 := copied_le bytes chunk hc
  have h2 : copiedPerPass bytes chunk ≠ bytes := by
    intro heq
    unfold copiedPerPass at heq
    have h3 := Nat.div_add_mod bytes chunk
    have hc2 : bytes / chunk * chunk = chunk * (bytes / chunk) := Nat.mul_comm _ _
    omega
  omega

end Strata.Pinned
