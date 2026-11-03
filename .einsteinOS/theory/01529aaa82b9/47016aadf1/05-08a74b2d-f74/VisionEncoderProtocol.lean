/-!
# MODULE: vision_encoder_protocol
The resident `strata-vision` image encoder's wire format and command handling: the `ENC <image> <output>`
split rule, the 20-byte output header with its `SVE1` magic, the token grid geometry and the embedding file size.

## Source semantics
* `ENC <image> <out>`: the image path ends at the LAST space of the remainder; both parts must be non-empty.
* Output file: five int32 `{magic, n_tokens, nx, ny, n_embd}` then `n_tokens * n_embd` float32 values.
* The grid is recovered from the last token's decoder position: `nx = x+1`, `ny = y+1`, and `nx*ny` must equal
  `n_tokens`; row `i` sits at `(x = i % nx, y = i / nx)`.
* Optional numeric arguments only override defaults when positive.

## Form template identifiers
parse_enc -> parse_command, hdr -> header, n_embd -> embed_width, nx/ny -> grid_dims.
-/

namespace Strata.VisionEncoder

-- § command parsing

/-- A command remainder of length `len` with its last space at `sp` yields an image part of length `sp`
and an output part of length `len - sp - 1`. -/
theorem split_lengths (sp len : Nat) (h : sp + 1 ≤ len) : sp + 1 + (len - sp - 1) = len := by omega

/-- The command is accepted only when both parts are non-empty. -/
def accepts (sp len : Nat) : Bool := decide (0 < sp ∧ sp + 1 < len)

/-- An accepted split has a non-empty image and a non-empty output path. -/
theorem accepts_nonempty (sp len : Nat) (h : accepts sp len = true) :
    0 < sp ∧ 0 < len - sp - 1 := by
  simp only [accepts, decide_eq_true_eq] at h
  omega

/-- A space in the first position (empty image part) is rejected. -/
theorem empty_image_rejected (len : Nat) : accepts 0 len = false := by
  simp [accepts]

/-- A trailing space (empty output part) is rejected. -/
theorem empty_output_rejected (sp len : Nat) (h : len ≤ sp + 1) : accepts sp len = false := by
  simp only [accepts, decide_eq_false_iff_not]
  omega

-- § header and file size

/-- The magic number 0x31455653 spells `SVE1` in little-endian byte order. -/
def magic : Nat := 0x31455653

/-- Bytes in order: `S`=0x53, `V`=0x56, `E`=0x45, `1`=0x31. -/
theorem magic_bytes :
    magic % 256 = 0x53 ∧ magic / 256 % 256 = 0x56 ∧ magic / 65536 % 256 = 0x45 ∧
    magic / 16777216 % 256 = 0x31 := by
  decide

/-- The header is five int32 values: 20 bytes. -/
theorem header_bytes : 5 * 4 = 20 := by decide

/-- Total file size: header plus `n` rows of `e` float32 values. -/
def fileBytes (n e : Nat) : Nat := 20 + n * (4 * e)

/-- The file size equals header plus four bytes per embedding value. -/
theorem fileBytes_eq (n e : Nat) : fileBytes n e = 20 + 4 * (n * e) := by
  unfold fileBytes
  rw [Nat.mul_left_comm]

-- § grid geometry

/-- Row `i` of the embedding matrix sits at grid position `(i % nx, i / nx)` and the position recovers `i`. -/
theorem row_position (i nx : Nat) : nx * (i / nx) + i % nx = i := Nat.div_add_mod i nx

/-- The in-row coordinate is a valid column. -/
theorem col_lt (i nx : Nat) (h : 0 < nx) : i % nx < nx := Nat.mod_lt i h

/-- For a rectangular grid of `nx` columns and `k+1` rows, the last token is at `(nx-1, k)`, which is how
`nx = x+1` and `ny = y+1` are recovered. -/
theorem last_token_position (nx k : Nat) (hnx : 0 < nx) :
    (nx * (k + 1) - 1) % nx = nx - 1 ∧ (nx * (k + 1) - 1) / nx = k := by
  have h : nx * (k + 1) - 1 = (nx - 1) + nx * k := by
    rw [Nat.mul_succ]
    omega
  constructor
  · rw [h, Nat.add_mul_mod_self_left]
    exact Nat.mod_eq_of_lt (by omega)
  · rw [h, Nat.add_mul_div_left _ _ hnx, Nat.div_eq_of_lt (by omega)]
    simp

/-- The grid check: `nx * ny` must equal the token count, else the image is rejected. -/
def gridOk (nx ny n : Nat) : Bool := nx * ny == n

/-- A mismatching grid is rejected. -/
theorem grid_mismatch (nx ny n : Nat) (h : nx * ny ≠ n) : gridOk nx ny n = false := by
  simp [gridOk, h]

-- § arguments

/-- A numeric option overrides the default only when positive. -/
def effective (req dflt : Nat) : Nat := if req > 0 then req else dflt

/-- A non-positive request keeps the default. -/
theorem effective_default (d : Nat) : effective 0 d = d := by
  simp [effective]

/-- A positive request wins. -/
theorem effective_req (g d : Nat) (h : 0 < g) : effective g d = g := by
  simp [effective, h]

/-- The embedding width must be positive or the encoder refuses to start. -/
def widthOk (nEmbd : Nat) : Bool := decide (0 < nEmbd)

/-- A zero projection width is rejected. -/
theorem zero_width_rejected : widthOk 0 = false := by decide

end Strata.VisionEncoder
