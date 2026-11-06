/-!
# MODULE: pinned_arena_loader
SOURCES: src/core/pinned.cu
The pinned host arena (large pages, `cudaHostRegister`, sliced registration fallback, resident lock for the tail) and the
parallel per-layer expert load with an FNV-1a checksum per layer.

## Source semantics
* Layers are claimed from an atomic counter, so each layer is loaded by exactly one worker.
* Layer `L` occupies `[L*bpl*blob, (L+1)*bpl*blob)`; layer ranges abut and tile the whole file.
* The per-layer hash is accumulated chunk by chunk with the running value as the next seed, which equals hashing the
  concatenation; every hash value is reduced mod 2^64.
* Sliced registration pins `bytes / slice` whole slices; the unpinned tail (< one slice) is locked resident instead, and
  the locked range plus the registered prefix cover the arena exactly.

## Form template identifiers
fnv1a64 -> fnv ; layer_off -> layerOff ; uniform_bounds -> lastBound ; lock_resident range -> lockedRange ;
next_layer.fetch_add -> claim ; registered_bytes -> registered
-/

namespace Strata.PinnedArena

-- § FNV-1a

/-- One FNV-1a step in 64-bit arithmetic: xor then multiply by the FNV prime, mod 2^64. -/
def fnvStep (h b : Nat) : Nat := ((h ^^^ b) * 1099511628211) % 18446744073709551616

/-- FNV-1a over a byte list from a seed. -/
def fnv (seed : Nat) (bs : List Nat) : Nat := bs.foldl fnvStep seed

/-- The empty input leaves the seed unchanged. -/
theorem fnv_nil (seed : Nat) : fnv seed [] = seed := rfl

/-- CHUNKING IS TRANSPARENT: hashing chunk by chunk, seeding each with the previous value, equals hashing the whole. -/
theorem fnv_append (seed : Nat) (xs ys : List Nat) : fnv seed (xs ++ ys) = fnv (fnv seed xs) ys := by
  unfold fnv
  rw [List.foldl_append]

/-- A single step stays below 2^64. -/
theorem fnvStep_lt (h b : Nat) : fnvStep h b < 18446744073709551616 :=
  Nat.mod_lt _ (by decide)

/-- Any non-empty-or-empty hash from a 64-bit seed stays 64-bit. -/
theorem fnv_lt (bs : List Nat) :
    ∀ seed, seed < 18446744073709551616 → fnv seed bs < 18446744073709551616 := by
  induction bs with
  | nil => intro seed h; exact h
  | cons b bs ih => intro seed _; exact ih _ (fnvStep_lt seed b)

/-- Observed quirk: the source's seed 1469598103934665603 is the standard FNV-1a offset basis with its last digit
dropped. Both loaders use the same constant, so checksums agree with each other; they do not match standard FNV-1a. -/
theorem seed_differs_from_standard : (1469598103934665603 : Nat) ≠ 14695981039346656037 := by
  decide

-- § layer tiling

/-- Start offset of layer `L`: `L * (bpl * blob)`. -/
def layerOff (bpl blob L : Nat) : Nat := L * (bpl * blob)

/-- Consecutive layers abut: no gap, no overlap. -/
theorem layerOff_succ (bpl blob L : Nat) :
    layerOff bpl blob (L + 1) = layerOff bpl blob L + bpl * blob := by
  unfold layerOff
  rw [Nat.add_mul, Nat.one_mul]

/-- Layer 0 starts at 0. -/
theorem layerOff_zero (bpl blob : Nat) : layerOff bpl blob 0 = 0 := by
  simp [layerOff]

/-- The 48 layers of 512 blobs tile exactly the 34 GB expert file. -/
theorem layers_tile_file : layerOff 512 1382400 48 = 33973862400 := by decide

-- § work claiming

/-- `next_layer.fetch_add(1)`: returns the old value and bumps the counter. -/
def claim (c : Nat) : Nat × Nat := (c, c + 1)

/-- Two successive claims never return the same layer. -/
theorem claim_fresh (c : Nat) : (claim c).1 < (claim (claim c).2).1 := by
  simp [claim]

/-- A claim at or past the layer count stops the worker, and the counter keeps growing monotonically. -/
theorem claim_monotone (c : Nat) : c < (claim c).2 := by
  simp [claim]

-- § chunked read accounting

/-- One iteration of the read loop: consume `n = min remaining chunk` bytes. -/
def stepRem (remaining chunk : Nat) : Nat := remaining - min remaining chunk

/-- Bytes consumed so far plus bytes remaining is invariant: the loop neither loses nor repeats bytes. -/
theorem step_invariant (pos remaining chunk : Nat) :
    (pos + min remaining chunk) + stepRem remaining chunk = pos + remaining := by
  unfold stepRem; omega

/-- A positive chunk makes progress while bytes remain. -/
theorem step_progress (remaining chunk : Nat) (h0 : 0 < remaining) (hc : 0 < chunk) :
    stepRem remaining chunk < remaining := by
  unfold stepRem; omega

-- § sliced registration

/-- End of the last whole slice. -/
def lastBound (bytes slice : Nat) : Nat := bytes / slice * slice

/-- Whole slices fit in the arena. -/
theorem lastBound_le (bytes slice : Nat) : lastBound bytes slice ≤ bytes := by
  unfold lastBound
  exact Nat.div_mul_le_self bytes slice

/-- The unregistered tail is shorter than one slice. -/
theorem tail_lt_slice (bytes slice : Nat) (h : 0 < slice) : bytes - lastBound bytes slice < slice := by
  unfold lastBound
  have h1 := Nat.div_add_mod bytes slice
  have h2 := Nat.mod_lt bytes h
  have h3 : bytes / slice * slice = slice * (bytes / slice) := Nat.mul_comm _ _
  omega

/-- The locked range `(start, length)` for the tail. -/
def lockedRange (bytes registered : Nat) : Nat × Nat := (registered, bytes - registered)

/-- Registered prefix plus locked tail cover the arena exactly. -/
theorem registered_plus_locked (bytes registered : Nat) (h : registered ≤ bytes) :
    (lockedRange bytes registered).1 + (lockedRange bytes registered).2 = bytes := by
  show registered + (bytes - registered) = bytes
  omega

end Strata.PinnedArena
