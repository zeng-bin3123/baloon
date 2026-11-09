/-!
# MODULE: vision_encoder_protocol
SOURCES: tools/vision/strata_vision.cpp
The resident image encoder: line protocol `ENC <image> <out>` -> `OK n nx ny ms` / `ERR msg`, and the SVE1
embeddings file (5 x int32 header then `n x n_embd` float32 rows in row-major grid order).

## Source semantics
* `parse_enc` splits the argument at the LAST space: image path may contain spaces, the output path may not;
  both parts must be non-empty.
* The header magic `0x31455653` is "SVE1" in little-endian bytes.
* The grid comes from the last token's decoder position: `nx = x+1`, `ny = y+1`, and the encoder insists
  `nx * ny = n` (rectangular). Row `i` sits at `(x, y) = (i % nx, i / nx)`.
* File size is `20 + 4 * n * n_embd` bytes.

## Form template identifiers
- parse_enc (rfind ' ')          → `lastSplit` / `parseEnc`
- hdr[5] = {magic,n,nx,ny,n_embd} → `headerBytes`
- row i at (i % nx, i / nx)      → `gridPos`
- last decoder pos + 1           → `last_token_grid`
-/

namespace Strata.VisionEncoder

-- § line parsing

/-- Split a list at its LAST space (code 32): (before, after). -/
def lastSplit : List Nat → Option (List Nat × List Nat)
  | [] => none
  | x :: xs =>
    match lastSplit xs with
    | some (b, a) => some (x :: b, a)
    | none => if x = 32 then some ([], xs) else none

/-- A list without a space has no split point. -/
theorem lastSplit_none : ∀ l : List Nat, (∀ y ∈ l, y ≠ 32) → lastSplit l = none := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons x xs ih =>
    intro h
    have hx : x ≠ 32 := h x (by simp)
    have hr : ∀ y ∈ xs, y ≠ 32 := fun y hy => h y (by simp [hy])
    simp [lastSplit, ih hr, hx]

/-- When the output path has no space, the split recovers (image, output) exactly. -/
theorem lastSplit_append (out : List Nat) (h : ∀ y ∈ out, y ≠ 32) :
    ∀ img : List Nat, lastSplit (img ++ 32 :: out) = some (img, out) := by
  intro img
  induction img with
  | nil => simp [lastSplit, lastSplit_none out h]
  | cons x xs ih => simp [lastSplit, ih]

/-- The `parse_enc` result: both parts must be non-empty. -/
def parseEnc (l : List Nat) : Option (List Nat × List Nat) :=
  match lastSplit l with
  | none => none
  | some (b, a) => if b.isEmpty || a.isEmpty then none else some (b, a)

/-- Well-formed `image <space> output` lines parse back to their two parts. -/
theorem parseEnc_roundtrip (x : Nat) (xs : List Nat) (y : Nat) (ys : List Nat)
    (h : ∀ z ∈ y :: ys, z ≠ 32) :
    parseEnc ((x :: xs) ++ 32 :: (y :: ys)) = some (x :: xs, y :: ys) := by
  have hs : lastSplit (x :: (xs ++ 32 :: y :: ys)) = some (x :: xs, y :: ys) :=
    lastSplit_append (y :: ys) h (x :: xs)
  simp [parseEnc, hs]

/-- A line starting with the separator (empty image path, `sp == 0`) is rejected. -/
theorem parseEnc_empty_image (out : List Nat) (h : ∀ y ∈ out, y ≠ 32) : parseEnc (32 :: out) = none := by
  simp [parseEnc, lastSplit, lastSplit_none out h]

-- § SVE1 file format

/-- The header magic. -/
def sveMagic : Nat := 0x31455653

/-- Little-endian bytes of the magic spell S, V, E, 1. -/
theorem sve_magic_bytes :
    sveMagic % 256 = 83 ∧ sveMagic / 256 % 256 = 86 ∧ sveMagic / 65536 % 256 = 69 ∧
      sveMagic / 16777216 = 49 := by
  decide

/-- Header size: five int32 values. -/
def headerBytes : Nat := 5 * 4

/-- Total file size for `n` tokens of width `nEmbd`. -/
def fileBytes (n nEmbd : Nat) : Nat := headerBytes + 4 * n * nEmbd

/-- The header is 20 bytes. -/
theorem header_is_20 : headerBytes = 20 := by
  decide

/-- File size grows by exactly one row of `4 * n_embd` bytes per extra token. -/
theorem fileBytes_succ (n nEmbd : Nat) : fileBytes (n + 1) nEmbd = fileBytes n nEmbd + 4 * nEmbd := by
  unfold fileBytes headerBytes
  rw [Nat.mul_add 4 n 1, Nat.add_mul]
  omega

-- § grid

/-- Position of token `i` in a grid of width `nx`: (x, y). -/
def gridPos (nx i : Nat) : Nat × Nat := (i % nx, i / nx)

/-- Every token of an `nx x ny` grid sits inside it, and (x, y) recombines to the index. -/
theorem gridPos_in_bounds (nx ny i : Nat) (hnx : 0 < nx) (hi : i < nx * ny) :
    (gridPos nx i).1 < nx ∧ (gridPos nx i).2 < ny ∧ (gridPos nx i).2 * nx + (gridPos nx i).1 = i := by
  unfold gridPos
  refine ⟨Nat.mod_lt _ hnx, Nat.div_lt_of_lt_mul hi, ?_⟩
  have := Nat.div_add_mod i nx
  rw [Nat.mul_comm] at this
  exact this

/-- The last token of a `(j+1) x (k+1)` grid is at (j, k), so `nx = x+1`, `ny = y+1` recovers the grid. -/
theorem last_token_grid (j k : Nat) :
    ((j + 1) * (k + 1) - 1) % (j + 1) = j ∧ ((j + 1) * (k + 1) - 1) / (j + 1) = k := by
  have h : (j + 1) * (k + 1) - 1 = j + (j + 1) * k := by
    have := Nat.mul_add (j + 1) k 1
    omega
  have hp : 0 < j + 1 := Nat.succ_pos j
  rw [h]
  constructor
  · rw [Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt (Nat.lt_succ_self j)
  · rw [Nat.add_mul_div_left _ _ hp, Nat.div_eq_of_lt (Nat.lt_succ_self j)]
    omega

end Strata.VisionEncoder
