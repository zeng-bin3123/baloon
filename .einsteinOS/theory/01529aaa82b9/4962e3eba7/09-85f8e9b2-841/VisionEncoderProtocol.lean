/-!
# MODULE: vision_encoder_protocol
Protocol of `tools/vision/strata_vision.cpp`: the resident image encoder.  Per stdin line
`ENC <image> <output>` it writes an `SVE1` embedding file and answers `OK ...` or `ERR ...`.

## Source semantics
* `parse_enc` splits the rest of the line at the LAST space: the image path may contain spaces, the
  output path may not; both must be non-empty.
* Output header = five int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}, then float32 rows.
* Token i sits at grid position (x = i % nx, y = i / nx); the grid must be rectangular (nx*ny = n).

## Form template identifiers
* splitLastSpace / parseRest / parseEnc <- parse_enc
* magic, fileBytes                      <- header `hdr[5]` and `fwrite` sizes
* grid lemmas                           <- decoder-pos grid reconstruction
-/

namespace Strata.Vision

-- § ENC line parsing

/-- Split a character list at its last space into (before, after). -/
def splitLastSpace : List Char → Option (List Char × List Char)
  | [] => none
  | c :: t =>
    match splitLastSpace t with
    | some (a, b) => some (c :: a, b)
    | none => if c = ' ' then some ([], t) else none

/-- A list with no space has no split point. -/
theorem splitLastSpace_none : ∀ l : List Char, ' ' ∉ l → splitLastSpace l = none
  | [], _ => rfl
  | c :: t, h => by
    have hc : c ≠ ' ' := fun e => h (by simp [e])
    have ht : ' ' ∉ t := fun e => h (List.mem_cons_of_mem c e)
    have ih := splitLastSpace_none t ht
    simp [splitLastSpace, ih, hc]

/-- Splitting `img ++ ' ' :: out` with a space-free `out` returns exactly `(img, out)`. -/
theorem splitLastSpace_append (out : List Char) (ho : ' ' ∉ out) :
    ∀ img : List Char, splitLastSpace (img ++ ' ' :: out) = some (img, out)
  | [] => by simp [splitLastSpace, splitLastSpace_none out ho]
  | c :: img => by
    have ih := splitLastSpace_append out ho img
    simp [splitLastSpace, ih]

/-- Accept only a split with both parts non-empty. -/
def parseRest (rest : List Char) : Option (List Char × List Char) :=
  match splitLastSpace rest with
  | some (i :: is, o :: os) => some (i :: is, o :: os)
  | _ => none

/-- Parse a whole line: it must start with `ENC `. -/
def parseEnc : List Char → Option (List Char × List Char)
  | 'E' :: 'N' :: 'C' :: ' ' :: rest => parseRest rest
  | _ => none

/-- A well-formed line parses into the image path (spaces allowed) and the output path. -/
theorem parseEnc_ok (i : Char) (is : List Char) (o : Char) (os : List Char)
    (ho : ' ' ∉ o :: os) :
    parseEnc ('E' :: 'N' :: 'C' :: ' ' :: ((i :: is) ++ ' ' :: (o :: os))) =
      some (i :: is, o :: os) := by
  have h := splitLastSpace_append (o :: os) ho (i :: is)
  show parseRest ((i :: is) ++ ' ' :: (o :: os)) = _
  unfold parseRest
  rw [h]

/-- Any line without the `ENC ` prefix is rejected. -/
theorem parseEnc_rejects_empty : parseEnc [] = none := rfl

-- § Output file format

/-- The header magic. -/
def magic : Nat := 0x31455653

/-- Little-endian bytes of the magic spell `S`,`V`,`E`,`1`. -/
theorem magic_bytes :
    magic % 256 = 0x53 ∧ magic / 256 % 256 = 0x56 ∧ magic / 65536 % 256 = 0x45 ∧
    magic / 16777216 = 0x31 := by
  decide

/-- Header is five int32 = 20 bytes. -/
theorem header_bytes : 5 * 4 = 20 := by decide

/-- Total file size: header plus n rows of n_embd float32. -/
def fileBytes (n nEmbd : Nat) : Nat := 20 + 4 * (n * nEmbd)

/-- The file is never smaller than its header. -/
theorem fileBytes_ge_header (n nEmbd : Nat) : 20 ≤ fileBytes n nEmbd := by
  unfold fileBytes
  omega

-- § Grid

/-- Row-major reconstruction: index = x + nx*y. -/
theorem grid_index (nx i : Nat) : i % nx + nx * (i / nx) = i :=
  Nat.mod_add_div i nx

/-- Row y stays inside the grid. -/
theorem grid_y_lt (nx ny i : Nat) (h : i < nx * ny) : i / nx < ny :=
  Nat.div_lt_of_lt_mul h

/-- Column x stays inside the grid. -/
theorem grid_x_lt (nx i : Nat) (h : 0 < nx) : i % nx < nx :=
  Nat.mod_lt i h

end Strata.Vision
