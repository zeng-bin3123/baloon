/-!
# MODULE: pinned_arena_load
The pinned host arena's registration slices and the parallel expert load: per-layer byte ranges, chunked reads
with a short final chunk, the FNV-1a step, and the way registered bytes are accounted for sliced registration.

## Source semantics
* `uniform_bounds(bytes, slice)`: offsets `0, slice, 2*slice, ...` while `off + slice <= bytes`, then one
  closing bound; slices are contiguous and equal.
* `load_experts`: layer `L` starts at `L * blobs_per_layer * blob_bytes` and has `blobs_per_layer * blob_bytes`
  bytes; the ranges tile `layers * blobs_per_layer * blob_bytes`.
* Reading a layer in chunks of `chunk` bytes: every chunk is `min(chunk, remaining)`; the chunk lengths sum to
  the layer size and the copy offsets are contiguous.
* FNV-1a: `h = (h xor byte) * prime`, over 64-bit words.
* With sliced registration the registered prefix is `off + n` of the last slice that pinned.

## Form template identifiers
* `layerStart`, `layerLen`   : `load_experts` offsets
* `chunkLen`                 : `remaining < chunk ? remaining : chunk`
* `fnvStep`                  : one FNV-1a byte step
-/

namespace Strata.Pinned

-- § uniform slice bounds

/-- Offsets of the uniform slices: `k * slice` for `k < count`, closing with `count * slice`. -/
def boundAt (slice k : Nat) : Nat := k * slice

/-- Number of whole slices in `bytes`. -/
def sliceCount (bytes slice : Nat) : Nat := bytes / slice

/-- Whole slices fit inside the arena. -/
theorem slices_fit (bytes slice : Nat) (h : 0 < slice) : sliceCount bytes slice * slice ≤ bytes := by
  unfold sliceCount
  exact Nat.div_mul_le_self bytes slice

/-- Consecutive bounds differ by exactly one slice (contiguous, equal slices). -/
theorem bound_step (slice k : Nat) : boundAt slice (k + 1) - boundAt slice k = slice := by
  unfold boundAt
  rw [Nat.succ_mul]
  omega

/-- The last whole slice ends at or before `bytes`. -/
theorem last_bound_le (bytes slice : Nat) (h : 0 < slice) :
    boundAt slice (sliceCount bytes slice) ≤ bytes := by
  unfold boundAt sliceCount
  exact Nat.div_mul_le_self bytes slice

/-- Slices cover all but fewer than `slice` trailing bytes. -/
theorem tail_lt_slice (bytes slice : Nat) (h : 0 < slice) :
    bytes - boundAt slice (sliceCount bytes slice) < slice := by
  unfold boundAt sliceCount
  have h1 := Nat.div_add_mod bytes slice
  have h2 := Nat.mod_lt bytes h
  rw [Nat.mul_comm (bytes / slice) slice]
  omega

-- § per-layer ranges

/-- Start of layer `L`. -/
def layerStart (perLayer blob L : Nat) : Nat := L * perLayer * blob

/-- Length of a layer. -/
def layerLen (perLayer blob : Nat) : Nat := perLayer * blob

/-- Layer ranges abut: layer `L` ends exactly where layer `L+1` begins. -/
theorem layer_abut (perLayer blob L : Nat) :
    layerStart perLayer blob L + layerLen perLayer blob = layerStart perLayer blob (L + 1) := by
  unfold layerStart layerLen
  simp [Nat.add_mul]

/-- The layers tile the whole expert file. -/
theorem layers_tile (perLayer blob layers : Nat) :
    layerStart perLayer blob layers = layers * perLayer * blob := rfl

/-- The last layer ends at the file size. -/
theorem last_layer_end (perLayer blob layers : Nat) (h : 0 < layers) :
    layerStart perLayer blob (layers - 1) + layerLen perLayer blob = layers * perLayer * blob := by
  rw [layer_abut]
  have : layers - 1 + 1 = layers := by omega
  rw [this]
  rfl

-- § chunked reads

/-- Length of the next chunk. -/
def chunkLen (chunk remaining : Nat) : Nat := if remaining < chunk then remaining else chunk

/-- A chunk never exceeds what remains. -/
theorem chunkLen_le_remaining (chunk remaining : Nat) : chunkLen chunk remaining ≤ remaining := by
  unfold chunkLen; split <;> omega

/-- A chunk never exceeds the chunk size. -/
theorem chunkLen_le_chunk (chunk remaining : Nat) : chunkLen chunk remaining ≤ chunk := by
  unfold chunkLen; split <;> omega

/-- A positive remainder with a positive chunk makes progress (the loop terminates). -/
theorem chunkLen_pos (chunk remaining : Nat) (hc : 0 < chunk) (hr : 0 < remaining) : 0 < chunkLen chunk remaining := by
  unfold chunkLen; split <;> omega

/-- Total bytes copied by the chunk loop, with fuel. -/
def copied : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | fuel + 1, chunk, remaining =>
      if remaining = 0 then 0
      else chunkLen chunk remaining + copied fuel chunk (remaining - chunkLen chunk remaining)

/-- With enough fuel the loop copies exactly the layer's bytes: no byte skipped, none read twice. -/
theorem copied_eq (chunk : Nat) (hc : 0 < chunk) : ∀ (fuel remaining : Nat), remaining ≤ fuel →
    copied fuel chunk remaining = remaining := by
  intro fuel
  induction fuel with
  | zero =>
    intro r hr
    have h0 : r = 0 := by omega
    subst h0
    simp [copied]
  | succ f ih =>
    intro r hr
    simp only [copied]
    by_cases h0 : r = 0
    · simp [h0]
    · have hp := chunkLen_pos chunk r hc (by omega)
      have hle := chunkLen_le_remaining chunk r
      have := ih (r - chunkLen chunk r) (by omega)
      simp only [h0, if_false]
      omega

-- § FNV-1a

/-- FNV-1a 64-bit prime. -/
def fnvPrime : Nat := 1099511628211

/-- One byte step on 64-bit words (modelled with `% 2^64`). -/
def fnvStep (h b : Nat) : Nat := ((h ^^^ b) * fnvPrime) % 2 ^ 64

/-- The step stays inside 64 bits. -/
theorem fnvStep_lt (h b : Nat) : fnvStep h b < 2 ^ 64 := by
  unfold fnvStep
  exact Nat.mod_lt _ (by decide)

-- § registered-prefix accounting

/-- Registered bytes after pinning slices up to and including the one at `off` of length `n`. -/
def registeredAfter (off n : Nat) : Nat := off + n

/-- If the final slice pinned, the registered prefix reaches that slice's end; it is within the arena. -/
theorem registered_le (off n bytes : Nat) (h : off + n ≤ bytes) : registeredAfter off n ≤ bytes := h

/-- The remainder to lock resident is whatever the registered prefix did not cover. -/
theorem lock_remainder (registered bytes : Nat) (h : registered ≤ bytes) :
    registered + (bytes - registered) = bytes := by omega

end Strata.Pinned
