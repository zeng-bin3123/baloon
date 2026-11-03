/-!
# MODULE: pinned_arena
Contracts of `src/core/pinned.cu`: sliced `cudaHostRegister` bookkeeping, the FNV-1a chunked checksum, the
per-layer expert load ranges and the bandwidth loop.

## Source semantics
* `uniform_bounds(bytes, slice)` has `bytes/slice` slices and covers `(bytes/slice)*slice <= bytes` bytes.
* Slice-registered prefix plus the locked remainder equal the arena size.
* FNV-1a is a left fold, so hashing a file in chunks equals hashing it whole (checksums are chunk-size independent).
* Layer `L`'s range is `[L*X, (L+1)*X)`; consecutive ranges are adjacent and disjoint.
* `stream_bandwidth` moves `(bytes/chunk)*chunk` per pass, equal to `bytes` only when `chunk | bytes`.

## Form template identifiers
* `fnv` <- `fnv1a64`
* `sliceCount`, `sliceCovered` <- `uniform_bounds`
* `layerOff` <- `L * blobs_per_layer * blob_bytes`
-/

namespace Strata.PinnedArena

-- § slice registration

/-- Number of whole slices. -/
def sliceCount (bytes slice : Nat) : Nat := bytes / slice

/-- Bytes covered by the registered slices. -/
def sliceCovered (bytes slice : Nat) : Nat := bytes / slice * slice

/-- Registered slices never exceed the arena. -/
theorem slice_covered_le (bytes slice : Nat) : sliceCovered bytes slice ≤ bytes := by
  unfold sliceCovered
  exact Nat.div_mul_le_self bytes slice

/-- The uncovered tail is shorter than a slice. -/
theorem slice_tail_lt (bytes slice : Nat) (h : 0 < slice) :
    bytes - sliceCovered bytes slice < slice := by
  unfold sliceCovered
  have h1 := Nat.div_add_mod bytes slice
  have h2 := Nat.mod_lt bytes h
  rw [Nat.mul_comm (bytes / slice) slice]
  omega

/-- Registered prefix plus locked remainder equal the arena. -/
theorem registered_plus_locked (bytes registered : Nat) (h : registered ≤ bytes) :
    registered + (bytes - registered) = bytes := by
  omega

-- § FNV-1a

/-- FNV-1a over a byte list, starting from `seed`. -/
def fnv (seed : UInt64) (l : List UInt8) : UInt64 :=
  l.foldl (fun h b => (h ^^^ b.toUInt64) * 1099511628211) seed

/-- Chunked hashing equals whole hashing: the per-layer checksum does not depend on the chunk size. -/
theorem fnv_append (seed : UInt64) (a b : List UInt8) :
    fnv seed (a ++ b) = fnv (fnv seed a) b := by
  simp [fnv, List.foldl_append]

/-- Hashing nothing keeps the seed. -/
theorem fnv_nil (seed : UInt64) : fnv seed [] = seed := by
  simp [fnv]

-- § per-layer ranges

/-- Offset of layer `L` for `X = blobs_per_layer * blob_bytes`. -/
def layerOff (X L : Nat) : Nat := L * X

/-- Layer `L` ends where layer `L+1` begins: ranges are adjacent and disjoint. -/
theorem layer_adjacent (X L : Nat) : layerOff X L + X = layerOff X (L + 1) := by
  unfold layerOff
  rw [Nat.add_mul, Nat.one_mul]

/-- The last layer ends at `layers * X`, the total bytes read. -/
theorem layer_total (X n : Nat) : layerOff X n = n * X := rfl

/-- One read step conserves `pos + remaining = layer bytes`. -/
theorem read_step_conserve (pos rem chunk total : Nat) (h : pos + rem = total) :
    (pos + min rem chunk) + (rem - min rem chunk) = total := by
  omega

/-- A read step makes progress when the chunk is positive. -/
theorem read_step_progress (rem chunk : Nat) (hr : 0 < rem) (hc : 0 < chunk) :
    0 < min rem chunk ∧ min rem chunk ≤ rem := by
  omega

-- § bandwidth loop

/-- Bytes actually moved per pass of the bandwidth loop. -/
def movedPerPass (bytes chunk : Nat) : Nat := bytes / chunk * chunk

/-- Moved bytes never exceed the claimed `bytes`. -/
theorem moved_le_claimed (bytes chunk : Nat) : movedPerPass bytes chunk ≤ bytes := by
  unfold movedPerPass
  exact Nat.div_mul_le_self bytes chunk

/-- Claimed and moved bytes agree exactly when `chunk` divides `bytes`. -/
theorem moved_eq_claimed (bytes chunk : Nat) (h : bytes % chunk = 0) :
    movedPerPass bytes chunk = bytes := by
  unfold movedPerPass
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)

end Strata.PinnedArena
