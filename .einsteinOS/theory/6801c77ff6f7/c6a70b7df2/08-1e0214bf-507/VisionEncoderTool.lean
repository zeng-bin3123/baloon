/-!
# MODULE: vision_encoder_tool
SOURCES: tools/vision/CMakeLists.txt, tools/vision/strata_vision.cpp
The resident image-encoder helper `strata-vision`: reads `ENC <image> <output>` lines, runs llama.cpp's mtmd
encoder, writes an embeddings file (header SVE1, n_tokens, nx, ny, n_embd; then float32 rows) and answers
`OK n nx ny ms` or `ERR msg`. CMake builds it against mtmd with optional CUDA and a portable (AVX2) mode.

## Source semantics
* parse_enc: line must start with "ENC "; the output path is the text after the LAST space, the image path is
  everything before it (so image paths may contain spaces); both must be non-empty.
* Output header: five int32 = 20 bytes, magic 0x31455653 = "SVE1" little-endian; payload 4*n*n_embd bytes.
* Token i sits at grid position (x = i % nx, y = i / nx); the grid must be rectangular: nx*ny = n.
* CMake: GGML_CUDA follows STRATA_VISION_CUDA; STRATA_PORTABLE turns GGML_NATIVE and GGML_OPENMP off and forces
  AVX/AVX2/FMA/F16C/BMI2 on; otherwise GGML_NATIVE is on.

## Form template identifiers
parse_enc -> command_parser ; hdr[5] -> embedding_header ; STRATA_PORTABLE -> portable_flag ;
STRATA_VISION_CUDA -> gpu_flag ; mtmd_encode_chunk -> encode_step
-/

namespace Strata.VisionTool

-- § command parsing

/-- Longest prefix without a space, and the rest (starting at the first space if any). -/
def spanNS : List Char → List Char × List Char
  | [] => ([], [])
  | c :: cs => if c = ' ' then ([], c :: cs) else ((c :: (spanNS cs).1), (spanNS cs).2)

/-- spanNS on (xs ++ ys) with xs space-free and ys empty or starting with a space returns (xs, ys). -/
theorem spanNS_append (xs ys : List Char) (h : ∀ c ∈ xs, c ≠ ' ')
    (hy : ys = [] ∨ ∃ t, ys = ' ' :: t) : spanNS (xs ++ ys) = (xs, ys) := by
  induction xs with
  | nil =>
    rcases hy with rfl | ⟨t, rfl⟩ <;> simp [spanNS]
  | cons c cs ih =>
    have hc : c ≠ ' ' := h c (by simp)
    have := ih (fun d hd => h d (by simp [hd]))
    simp [spanNS, hc, this]

/-- Split at the last space: (image, output); mirrors `rest.rfind(' ')` with the emptiness checks. -/
def splitLastSpace (l : List Char) : Option (List Char × List Char) :=
  match spanNS l.reverse with
  | (o, ' ' :: i) => if i = [] ∨ o = [] then none else some (i.reverse, o.reverse)
  | _ => none

/-- Correctness: a space-free output path after the last space is recovered exactly, so image paths with
spaces survive. -/
theorem splitLastSpace_correct (img out : List Char) (hno : ∀ c ∈ out, c ≠ ' ')
    (himg : img ≠ []) (hout : out ≠ []) :
    splitLastSpace (img ++ ' ' :: out) = some (img, out) := by
  have hrev : (img ++ ' ' :: out).reverse = out.reverse ++ ' ' :: img.reverse := by
    simp [List.reverse_append]
  have hs := spanNS_append out.reverse (' ' :: img.reverse)
    (fun c hc => hno c (List.mem_reverse.mp hc)) (Or.inr ⟨img.reverse, rfl⟩)
  unfold splitLastSpace
  rw [hrev, hs]
  simp [himg, hout]

/-- A line without any space has no separator, so it is rejected. -/
theorem splitLastSpace_no_space (l : List Char) (h : ∀ c ∈ l, c ≠ ' ') : splitLastSpace l = none := by
  have hr : ∀ c ∈ l.reverse, c ≠ ' ' := fun c hc => h c (List.mem_reverse.mp hc)
  have := spanNS_append l.reverse [] hr (Or.inl rfl)
  unfold splitLastSpace
  rw [List.append_nil] at this
  simp [this]

-- § output file format

/-- Header size: five int32. -/
theorem header_bytes : 5 * 4 = 20 := by
  decide

/-- The magic is "SVE1" in little-endian byte order. -/
theorem magic_bytes : 0x53 + 0x56 * 256 + 0x45 * 65536 + 0x31 * 16777216 = 0x31455653 := by
  decide

/-- File size = header + float32 rows. -/
def fileBytes (n nEmbd : Nat) : Nat := 20 + 4 * (n * nEmbd)

/-- Payload bytes grow linearly in the token count. -/
theorem fileBytes_succ (n nEmbd : Nat) : fileBytes (n + 1) nEmbd = fileBytes n nEmbd + 4 * nEmbd := by
  unfold fileBytes
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- An empty image writes only the header. -/
theorem fileBytes_zero (nEmbd : Nat) : fileBytes 0 nEmbd = 20 := by
  simp [fileBytes]

-- § grid positions

/-- Token i is at x = i % nx, y = i / nx; it lies inside the nx*ny grid. -/
theorem grid_position_in_range (i nx ny : Nat) (hnx : 0 < nx) (h : i < nx * ny) :
    i % nx < nx ∧ i / nx < ny := by
  exact ⟨Nat.mod_lt _ hnx, Nat.div_lt_of_lt_mul h⟩

/-- The position determines the token index. -/
theorem grid_position_roundtrip (i nx : Nat) : nx * (i / nx) + i % nx = i := by
  exact Nat.div_add_mod i nx

/-- A rectangular grid read from the last token's position (x, y) has nx = x+1, ny = y+1 and
nx*ny = n exactly when the last index fits. -/
theorem last_token_grid (x y : Nat) (n : Nat) (h : n = (x + 1) * (y + 1)) :
    n - 1 = (y) * (x + 1) + x := by
  subst h
  rw [Nat.mul_add, Nat.mul_one]
  have : (x + 1) * y = y * (x + 1) := Nat.mul_comm _ _
  omega

-- § build configuration

/-- Build switches relevant to the vision helper. -/
structure Config where
  portable : Bool
  cuda : Bool

/-- GGML_NATIVE. -/
def ggmlNative (c : Config) : Bool := !c.portable

/-- GGML_OPENMP is forced off only in portable mode (no runtime library needed on the user's PC). -/
def ggmlOpenmpForcedOff (c : Config) : Bool := c.portable

/-- GGML_CUDA follows the option. -/
def ggmlCuda (c : Config) : Bool := c.cuda

/-- Portable builds never use -march=native. -/
theorem portable_not_native (c : Config) (h : c.portable = true) : ggmlNative c = false := by
  simp [ggmlNative, h]

/-- Non-portable builds use native tuning. -/
theorem nonportable_native (c : Config) (h : c.portable = false) : ggmlNative c = true := by
  simp [ggmlNative, h]

/-- The CUDA switch is independent of portability. -/
theorem cuda_independent (c : Config) : ggmlCuda c = c.cuda := by
  rfl

end Strata.VisionTool
