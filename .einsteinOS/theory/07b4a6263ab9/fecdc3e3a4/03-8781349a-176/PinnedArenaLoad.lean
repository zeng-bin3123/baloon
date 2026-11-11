/-!
# MODULE: pinned_arena_load
SOURCES: src/core/pinned.cu
The pinned host arena (large pages, then `cudaHostRegister` of the whole range, else slice by
slice, else working-set locking) and the parallel chunked expert load with per-layer FNV-1a hashes.

## Source semantics
* `uniform_bounds(bytes, slice)` lists `0, slice, 2*slice, ...` while a whole slice fits, then the
  end of the last whole slice; with `slice = 0` it is empty.
* Sliced registration pins slices in order and STOPS at the first failure, so the registered
  region is always a prefix `[0, registered_bytes)`; the remainder `[registered_bytes, bytes)` is
  what the working-set lock covers.
* `load_experts_ranges` reads each layer in chunks of at most `chunk` bytes; the hash is carried
  across chunks, so chunking does not change it (FNV-1a is a left fold).
* `load_experts` lays layers out contiguously: layer `L` at `L * blobs_per_layer * blob_bytes`.

## Form template identifiers
`uniform_bounds` → `uniformBounds`; registration loop → `regPrefix`; `fnv1a64` → `fnv`;
chunk loop → `chunks`; `load_experts` offsets → `layerOff`.
-/

namespace Strata.PinnedArenaLoad

-- § uniform slice bounds

/-- `uniform_bounds`: slice starts, then the end of the last whole slice. -/
def uniformBounds (bytes slice : Nat) : List Nat :=
  if slice = 0 then []
  else
    let n := bytes / slice
    if n = 0 then [] else (List.range n).map (· * slice) ++ [n * slice]

/-- A zero slice size produces no bounds. -/
theorem uniformBounds_zero (b : Nat) : uniformBounds b 0 = [] := by simp [uniformBounds]

/-- Every bound lies inside the arena. -/
theorem uniformBounds_le (b s : Nat) : ∀ x ∈ uniformBounds b s, x ≤ b := by
  intro x hx
  unfold uniformBounds at hx
  have hd := Nat.div_mul_le_self b s
  split at hx
  · contradiction
  · simp only at hx
    split at hx
    · contradiction
    · simp only [List.mem_append, List.mem_map, List.mem_range, List.mem_singleton] at hx
      rcases hx with ⟨i, hi, rfl⟩ | rfl
      · have : i * s ≤ b / s * s := Nat.mul_le_mul_right _ (Nat.le_of_lt hi)
        omega
      · exact hd

/-- The bound list has `bytes / slice + 1` entries when at least one slice fits. -/
theorem uniformBounds_length (b s : Nat) (hs : s ≠ 0) (hn : b / s ≠ 0) :
    (uniformBounds b s).length = b / s + 1 := by
  simp [uniformBounds, hs, hn]

-- § sliced registration is a prefix

/-- Register slices `[bᵢ, bᵢ₊₁)` in order; stop at the first failure.  Returns registered bytes. -/
def regPrefix (ok : Nat → Bool) (acc : Nat) : List Nat → Nat
  | a :: b :: rest => if ok a then regPrefix ok b (b :: rest) else acc
  | _ => acc

/-- Registered bytes never exceed the last bound when bounds are increasing from `acc`. -/
theorem regPrefix_le (ok : Nat → Bool) : ∀ (bs : List Nat) (acc : Nat),
    (∀ x ∈ bs, x ≤ B) → acc ≤ B → regPrefix ok acc bs ≤ B := by
  intro bs
  induction bs with
  | nil => intro acc _ h; simpa [regPrefix] using h
  | cons a rest ih =>
    intro acc hb hacc
    cases rest with
    | nil => simpa [regPrefix] using hacc
    | cons b rest' =>
      simp only [regPrefix]
      split
      · exact ih b (fun x hx => hb x (List.mem_cons_of_mem _ hx)) (hb b (by simp))
      · exact hacc

/-- If every slice registers, the whole range up to the last bound is pinned. -/
theorem regPrefix_all (bs : List Nat) : ∀ acc last, (a :: bs).getLast? = some last →
    regPrefix (fun _ => true) acc (a :: bs) = (if bs = [] then acc else last) := by
  induction bs generalizing a with
  | nil => intro acc last _; simp [regPrefix]
  | cons b rest ih =>
    intro acc last hl
    simp only [regPrefix, if_true]
    have hl' : (b :: rest).getLast? = some last := by
      simpa [List.getLast?_cons] using hl
    rw [ih b last hl']
    cases rest <;> simp_all [List.getLast?]

/-- The locked remainder plus the registered prefix exactly cover the arena. -/
theorem lock_covers_rest (reg bytes : Nat) (h : reg ≤ bytes) : reg + (bytes - reg) = bytes := by omega

-- § FNV-1a and chunk invariance

/-- The 64-bit modulus. -/
def W : Nat := 2 ^ 64

/-- One FNV-1a step: xor the byte, multiply by the 64-bit prime, wrap. -/
def fnvStep (h b : Nat) : Nat := ((h ^^^ b) * 1099511628211) % W

/-- `fnv1a64(p, n, seed)` as a left fold over the bytes. -/
def fnv (seed : Nat) (bytes : List Nat) : Nat := bytes.foldl fnvStep seed

/-- The FNV-1a 64-bit offset basis used as every layer's seed. -/
def basis : Nat := 1469598103934665603

/-- CHUNK INVARIANCE: hashing two chunks with the hash carried equals hashing their concatenation. -/
theorem fnv_append (seed : Nat) (a b : List Nat) : fnv (fnv seed a) b = fnv seed (a ++ b) := by
  simp [fnv, List.foldl_append]

/-- Hashing any chunk sequence (carrying the state) equals hashing the joined layer. -/
theorem fnv_chunks (seed : Nat) (cs : List (List Nat)) :
    cs.foldl fnv seed = fnv seed cs.flatten := by
  induction cs generalizing seed with
  | nil => rfl
  | cons c cs ih => simp only [List.foldl, List.flatten, ih, fnv_append]

/-- The hash of an empty layer is its seed. -/
theorem fnv_nil (seed : Nat) : fnv seed [] = seed := rfl

/-- Every hash value fits in 64 bits after at least one byte. -/
theorem fnv_lt (seed : Nat) (b : Nat) (bs : List Nat) : fnv seed (bs ++ [b]) < W := by
  rw [← fnv_append]
  simp only [fnv, List.foldl, fnvStep]
  exact Nat.mod_lt _ (by decide)

-- § chunked reads

/-- Read sizes for `remaining` bytes in chunks of `c`: full chunks, then the tail. -/
def chunks (rem c : Nat) : List Nat :=
  List.replicate (rem / c) c ++ (if rem % c = 0 then [] else [rem % c])

/-- Sum of a replicated list. -/
theorem sum_replicate (n c : Nat) : (List.replicate n c).foldl (· + ·) 0 = n * c := by
  have : ∀ acc, (List.replicate n c).foldl (· + ·) acc = acc + n * c := by
    induction n with
    | zero => intro; simp
    | succ n ih => intro acc; simp only [List.replicate_succ, List.foldl]; rw [ih]; rw [Nat.succ_mul]; omega
  simpa using this 0

/-- The chunk loop reads exactly the layer's bytes. -/
theorem chunks_sum (rem c : Nat) : (chunks rem c).foldl (· + ·) 0 = rem := by
  unfold chunks
  rw [List.foldl_append, sum_replicate]
  have hd := Nat.div_add_mod rem c
  split
  · simp; rw [Nat.mul_comm]; omega
  · simp; rw [Nat.mul_comm]; omega

/-- No single read exceeds the chunk size. -/
theorem chunks_le (rem c : Nat) (hc : 0 < c) : ∀ x ∈ chunks rem c, x ≤ c := by
  intro x hx
  simp only [chunks, List.mem_append, List.mem_replicate] at hx
  rcases hx with ⟨_, rfl⟩ | hx
  · exact Nat.le_refl _
  · split at hx
    · contradiction
    · simp at hx; subst hx; exact Nat.le_of_lt (Nat.mod_lt _ hc)

-- § layer layout of `load_experts`

/-- `load_experts`'s layer offset. -/
def layerOff (blobBytes perLayer L : Nat) : Nat := L * perLayer * blobBytes

/-- Layers are contiguous: layer `L` ends exactly where layer `L+1` begins. -/
theorem layerOff_succ (bb pl L : Nat) : layerOff bb pl L + pl * bb = layerOff bb pl (L + 1) := by
  unfold layerOff; rw [Nat.succ_mul, Nat.add_mul]

end Strata.PinnedArenaLoad
