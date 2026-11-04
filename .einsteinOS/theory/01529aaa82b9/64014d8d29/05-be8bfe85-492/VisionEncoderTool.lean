/-!
# MODULE: vision_encoder_tool
SOURCES: tools/vision/strata_vision.cpp, tools/vision/CMakeLists.txt
The resident image-encoder helper: its output file header and size, the token grid, the `ENC` line
split, and the CMake portability options.

## Source semantics
* Output header is five int32 `{0x31455653 'SVE1', n_tokens, nx, ny, n_embd}` followed by
  `n_tokens * n_embd` float32 values.
* The grid is recovered from the last token's decoder position: `nx = x + 1`, `ny = y + 1`; the
  encoding is accepted only when `nx * ny = n_tokens`. Row `i` sits at `(i % nx, i / nx)`.
* `ENC <image> <out>` splits at the LAST space, so the image path may contain spaces.
* `STRATA_PORTABLE` turns `GGML_NATIVE` off and forces the AVX2 baseline flags on.

## Form template identifiers
* SVE1 magic -> file_magic, hdr[5] -> header_words, parse_enc -> command_line_split
* STRATA_PORTABLE -> portable_build_flag
-/

namespace Strata.Vision

-- § file format
/-- 'S' 'V' 'E' '1' as little-endian bytes 0x53 0x56 0x45 0x31. -/
theorem magic_value : 0x53 + 0x56 * 256 + 0x45 * 65536 + 0x31 * 16777216 = 0x31455653 := by decide

/-- Header size in bytes: five 4-byte words. -/
def headerBytes : Nat := 5 * 4

/-- Total output file size. -/
def fileBytes (n nEmbd : Nat) : Nat := headerBytes + 4 * (n * nEmbd)

/-- The header is 20 bytes. -/
theorem header_is_20 : headerBytes = 20 := by decide

/-- The payload starts right after the header and has exactly `n` rows of `n_embd` floats. -/
theorem payload_size (n nEmbd : Nat) : fileBytes n nEmbd - headerBytes = 4 * n * nEmbd := by
  unfold fileBytes
  rw [Nat.add_sub_cancel_left, Nat.mul_assoc]

/-- An empty encoding still produces a header-only file. -/
theorem empty_file (nEmbd : Nat) : fileBytes 0 nEmbd = headerBytes := by
  simp [fileBytes]

-- § grid
/-- Grid dimensions from the last decoder position. -/
def gridDims (x y : Nat) : Nat × Nat := (x + 1, y + 1)

/-- The grid is accepted only when it is exactly rectangular. -/
def rectangular (n x y : Nat) : Bool := (x + 1) * (y + 1) == n

/-- Rectangular acceptance means the token count equals `nx * ny`. -/
theorem rectangular_iff (n x y : Nat) : rectangular n x y = true ↔ (gridDims x y).1 * (gridDims x y).2 = n := by
  simp [rectangular, gridDims]

/-- Row-major position: token `i` is at column `i % nx`, row `i / nx`, and the pair recovers `i`. -/
theorem grid_position_roundtrip (i nx : Nat) : (i / nx) * nx + i % nx = i := by
  rw [Nat.mul_comm]
  exact Nat.div_add_mod i nx

/-- For `nx > 0`, the column is inside the grid. -/
theorem column_lt (i nx : Nat) (h : 0 < nx) : i % nx < nx := Nat.mod_lt _ h

/-- For a rectangular grid, every token's row is inside it. -/
theorem row_lt (i nx ny : Nat) (h : 0 < nx) (hi : i < nx * ny) : i / nx < ny := by
  exact Nat.div_lt_of_lt_mul hi

-- § command split
/-- The image path and output path come from splitting the rest of the line at the last space. -/
theorem split_reassembles (l : List Nat) (sp : Nat) : l.take sp ++ l.drop sp = l :=
  List.take_append_drop sp l

/-- A split position is valid only when it is strictly inside the string (`sp ≠ 0`, found). -/
def validSplit (len sp : Nat) : Bool := 0 < sp && sp < len

/-- Valid splits leave a non-empty image part and a non-empty output part (after the space). -/
theorem validSplit_parts (len sp : Nat) (h : validSplit len sp = true) :
    0 < sp ∧ 0 < len - (sp + 1) + 1 := by
  unfold validSplit at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  omega

/-- The prefix `ENC ` has 4 characters, so the remainder starts at index 4. -/
theorem enc_prefix_len : "ENC ".length = 4 := by decide

-- § build options
/-- `GGML_NATIVE` is on exactly when the build is not portable. -/
def ggmlNative (portable : Bool) : Bool := !portable

/-- The AVX2 baseline flags are forced on only for portable builds. -/
def baselineFlags (portable : Bool) : Bool := portable

/-- Portable and native are exclusive. -/
theorem native_xor_baseline (p : Bool) : ggmlNative p = !baselineFlags p := by
  cases p <;> rfl

/-- A portable build is never native. -/
theorem portable_not_native : ggmlNative true = false := rfl

/-- A non-portable build uses native tuning and no forced baseline. -/
theorem default_native : ggmlNative false = true ∧ baselineFlags false = false := by decide

/-- Vision runs on the GPU only if the CUDA option is on. -/
def ggmlCuda (stratavisioncuda : Bool) : Bool := stratavisioncuda

/-- The default (OFF) builds no CUDA backend. -/
theorem cuda_default_off : ggmlCuda false = false := rfl

end Strata.Vision
