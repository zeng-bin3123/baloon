/-!
# MODULE: vision_encoder_protocol
`tools/vision/strata_vision.cpp`: resident image encoder.  Reads `ENC <image> <out>` lines, writes a binary
embedding file with a 5-int32 header followed by `n` rows of `n_embd` float32, rows laid out on an `nx` by `ny`
grid.

## Source semantics
* `parse_enc` splits at the LAST space: the image path may contain spaces, the output path may not.
* the grid is derived from the last token's decoder position: nx = x+1, ny = y+1; require nx*ny = n.
* row i sits at (x = i % nx, y = i / nx).
* file size = 20 + 4 * n * n_embd bytes.

## Form template identifiers
splitLast    -> `rest.rfind(' ')` split
header       -> hdr[5] = {magic, n, nx, ny, n_embd}
gridPos      -> (i % nx, i / nx)
-/

namespace Strata.VisionEncoderProtocol

-- § parsing the ENC line

/-- Split at the last space (code 32): `(before, after)`; none if there is no space. -/
def splitLast : List Nat → Option (List Nat × List Nat)
  | [] => none
  | x :: xs =>
    match splitLast xs with
    | some (a, b) => some (x :: a, b)
    | none => if x = 32 then some ([], xs) else none

/-- A space-free list has no split point. -/
theorem splitLast_none (out : List Nat) (h : ∀ c ∈ out, c ≠ 32) : splitLast out = none := by
  induction out with
  | nil => rfl
  | cons x xs ih =>
    have hx : x ≠ 32 := h x (by simp)
    have hxs : ∀ c ∈ xs, c ≠ 32 := fun c hc => h c (by simp [hc])
    simp [splitLast, ih hxs, hx]

/-- Image path (spaces allowed) and a space-free output path are recovered exactly. -/
theorem splitLast_correct (img out : List Nat) (h : ∀ c ∈ out, c ≠ 32) :
    splitLast (img ++ 32 :: out) = some (img, out) := by
  induction img with
  | nil => simp [splitLast, splitLast_none out h]
  | cons x xs ih => simp [splitLast, ih]

/-- A line with no space yields no split. -/
theorem no_space_no_split (l : List Nat) (h : ∀ c ∈ l, c ≠ 32) : splitLast l = none :=
  splitLast_none l h

/-- Parsing succeeds with a nonempty image and output only when both are nonempty. -/
def validEnc (img out : List Nat) : Bool := !img.isEmpty && !out.isEmpty

theorem validEnc_iff (img out : List Nat) :
    validEnc img out = true ↔ img ≠ [] ∧ out ≠ [] := by
  unfold validEnc
  cases img <;> cases out <;> simp

-- § grid layout

/-- Grid coordinates of row `i`. -/
def gridPos (nx i : Nat) : Nat × Nat := (i % nx, i / nx)

/-- Every row of an `nx * ny` grid lands inside it. -/
theorem gridPos_in_range (nx ny i : Nat) (hnx : 0 < nx) (hi : i < nx * ny) :
    (gridPos nx i).1 < nx ∧ (gridPos nx i).2 < ny := by
  unfold gridPos
  exact ⟨Nat.mod_lt _ hnx, (Nat.div_lt_iff_lt_mul hnx).2 (by rw [Nat.mul_comm]; exact hi)⟩

/-- Row-major layout round-trips: row `y*nx + x` is at `(x, y)`. -/
theorem gridPos_roundtrip (nx x y : Nat) (hx : x < nx) : gridPos nx (y * nx + x) = (x, y) := by
  unfold gridPos
  have hpos : 0 < nx := by omega
  have h := (Nat.div_mod_unique hpos).2 (⟨by rw [Nat.mul_comm nx y, Nat.add_comm], hx⟩ :
    x + nx * y = y * nx + x ∧ x < nx)
  simp [h.1, h.2]

/-- The last token of a full grid is at `(nx-1, ny-1)`, which is how `nx`, `ny` are derived. -/
theorem last_token_pos (nx ny : Nat) (hnx : 0 < nx) (hny : 0 < ny) :
    gridPos nx ((ny - 1) * nx + (nx - 1)) = (nx - 1, ny - 1) :=
  gridPos_roundtrip nx (nx - 1) (ny - 1) (by omega)

/-- Deriving `nx*ny` from the last position reproduces the token count of a rectangular grid. -/
theorem grid_count (nx ny : Nat) (hnx : 0 < nx) (hny : 0 < ny) :
    (nx - 1 + 1) * (ny - 1 + 1) = nx * ny := by
  have h1 : nx - 1 + 1 = nx := by omega
  have h2 : ny - 1 + 1 = ny := by omega
  rw [h1, h2]

-- § output file size

/-- Header bytes: five int32. -/
theorem header_bytes : 5 * 4 = 20 := by decide

/-- File size for `n` rows of `n_embd` floats. -/
def fileBytes (n nEmbd : Nat) : Nat := 20 + 4 * (n * nEmbd)

/-- Size grows by exactly one row of floats per token. -/
theorem fileBytes_step (n nEmbd : Nat) : fileBytes (n + 1) nEmbd = fileBytes n nEmbd + 4 * nEmbd := by
  unfold fileBytes
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- An empty encoding is just the header. -/
theorem fileBytes_zero (nEmbd : Nat) : fileBytes 0 nEmbd = 20 := by
  unfold fileBytes; simp

/-- The magic `SVE1` bytes 'S','V','E','1' read as little-endian int32. -/
theorem magic : 0x53 + 0x56 * 256 + 0x45 * 65536 + 0x31 * 16777216 = 0x31455653 := by decide

/-- The magic fits in a positive int32. -/
theorem magic_int32 : 0x31455653 < 2147483648 := by decide

end Strata.VisionEncoderProtocol
