/-!
# MODULE: vision_encoder_service
SOURCES: tools/vision/strata_vision.cpp, tools/vision/CMakeLists.txt
The resident image encoder `strata-vision`: the `ENC <image> <out>` line protocol, the SVE1 embeddings file
layout and grid, and the CMake option logic (portable vs native, CUDA passthrough).

## Source semantics
* The image path ends at the LAST space; the output path contains no space; both must be non-empty.
* SVE1 header = 5 int32 (magic 0x31455653, n_tokens, nx, ny, n_embd) followed by `n * n_embd` float32 rows.
* The grid is derived from the last token's position: nx = x+1, ny = y+1, and must satisfy `nx * ny = n`.
* Row i sits at (x = i mod nx, y = i / nx).
* Portable builds turn native tuning off and force the AVX2-class baseline; native builds do the opposite.

## Form template identifiers
- `parse_enc` -> `enc_line_parser`; header writer -> `sve1_writer`; `mtmd_image_tokens_get_decoder_pos` -> `grid_from_last_token`;
  `STRATA_PORTABLE` -> `portable_build_switch`.
-/

namespace Strata.VisionEncoderService

-- § ENC line parsing

/-- Split at the last space (32): (before, after). `none` when there is no space. -/
def lastSplit : List Nat → Option (List Nat × List Nat)
  | [] => none
  | c :: r =>
    match lastSplit r with
    | some (a, b) => some (c :: a, b)
    | none => if c = 32 then some ([], r) else none

/-- If no split exists the list contains no space. -/
theorem lastSplit_none (l : List Nat) (h : lastSplit l = none) : ∀ c ∈ l, c ≠ 32 := by
  induction l with
  | nil => intro c hc; simp at hc
  | cons c r ih =>
    unfold lastSplit at h
    split at h
    · cases h
    · rename_i hr
      split at h
      · cases h
      · rename_i hc
        intro x hx
        rcases List.mem_cons.mp hx with e | e
        · subst e; exact hc
        · exact ih hr x e

/-- A split reproduces the input around one space, and the tail contains no space. -/
theorem lastSplit_spec (l a b : List Nat) (h : lastSplit l = some (a, b)) :
    l = a ++ 32 :: b ∧ ∀ c ∈ b, c ≠ 32 := by
  induction l generalizing a b with
  | nil => simp [lastSplit] at h
  | cons c r ih =>
    unfold lastSplit at h
    split at h
    · rename_i a' b' hr
      injection h with h1
      have ⟨e1, e2⟩ := Prod.mk.inj h1
      have ⟨ih1, ih2⟩ := ih a' b' hr
      subst e1
      subst e2
      exact ⟨by simp [ih1], ih2⟩
    · rename_i hr
      split at h
      · rename_i hc
        injection h with h1
        have ⟨e1, e2⟩ := Prod.mk.inj h1
        subst e1
        subst e2
        refine ⟨by simp [hc], ?_⟩
        exact lastSplit_none r hr
      · cases h

/-- `parse_enc` on the text after `ENC `: image and output must both be non-empty. -/
def parseEnc (rest : List Nat) : Option (List Nat × List Nat) :=
  match lastSplit rest with
  | some (a, b) => if a = [] ∨ b = [] then none else some (a, b)
  | none => none

/-- A parsed line has a non-empty image path and a non-empty output path. -/
theorem parseEnc_nonempty (rest a b : List Nat) (h : parseEnc rest = some (a, b)) : a ≠ [] ∧ b ≠ [] := by
  unfold parseEnc at h
  split at h
  · rename_i x y hs
    split at h
    · cases h
    · rename_i hn
      injection h with h1
      have ⟨e1, e2⟩ := Prod.mk.inj h1
      subst e1
      subst e2
      simp at hn
      exact hn
  · cases h

/-- A space-free output path parses back exactly (a path with a space in the image part is fine). -/
theorem parseEnc_example : parseEnc [1, 2, 32, 3] = some ([1, 2], [3]) := by
  decide

/-- No space at all: the line is rejected. -/
theorem parseEnc_no_space : parseEnc [1, 2, 3] = none := by
  decide

/-- A leading space leaves an empty image path, which is rejected. -/
theorem parseEnc_empty_image : parseEnc [32, 3] = none := by
  decide

-- § SVE1 file layout

/-- Header bytes: five int32 values. -/
def headerBytes : Nat := 5 * 4

/-- Total size of an SVE1 file for `n` tokens of width `e`. -/
def sveFileSize (n e : Nat) : Nat := headerBytes + n * e * 4

/-- The header is 20 bytes. -/
theorem header_is_20 : headerBytes = 20 := by
  decide

/-- The magic 0x31455653 spells `SVE1` in little-endian bytes S V E 1 = 0x53 0x56 0x45 0x31. -/
theorem sve1_magic : 0x31455653 = 0x53 + 256 * (0x56 + 256 * (0x45 + 256 * 0x31)) := by
  decide

/-- The file grows linearly: one more token adds exactly one row of `4 * e` bytes. -/
theorem sveFileSize_succ (n e : Nat) : sveFileSize (n + 1) e = sveFileSize n e + 4 * e := by
  unfold sveFileSize
  rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  omega

/-- An empty image yields a header-only file. -/
theorem sveFileSize_zero (e : Nat) : sveFileSize 0 e = 20 := by
  simp [sveFileSize, headerBytes]

-- § grid

/-- Grid width and height from the last token's decoder position. -/
def gridOf (x y : Nat) : Nat × Nat := (x + 1, y + 1)

/-- The grid is rectangular iff its area equals the token count (the source rejects otherwise). -/
def rectangular (n x y : Nat) : Bool := (x + 1) * (y + 1) == n

/-- Both grid dimensions are positive. -/
theorem grid_positive (x y : Nat) : 0 < (gridOf x y).1 ∧ 0 < (gridOf x y).2 := by
  simp [gridOf]

/-- A rectangular grid has exactly `n` cells. -/
theorem rectangular_iff (n x y : Nat) : rectangular n x y = true ↔ (x + 1) * (y + 1) = n := by
  simp [rectangular]

/-- Row `i` position: (i mod nx, i / nx) recomposes to `i`. -/
theorem position_recompose (i nx : Nat) : (i / nx) * nx + i % nx = i := by
  have := Nat.div_add_mod i nx
  rw [Nat.mul_comm] at this
  exact this

/-- Every token of a rectangular grid lands inside it: x < nx and y < ny. -/
theorem position_in_grid (i nx ny : Nat) (hnx : 0 < nx) (h : i < nx * ny) :
    i % nx < nx ∧ i / nx < ny := by
  refine ⟨Nat.mod_lt _ hnx, ?_⟩
  exact Nat.div_lt_of_lt_mul h

-- § CMake option logic

/-- GGML_NATIVE is on exactly when the build is not portable. -/
def ggmlNative (portable : Bool) : Bool := !portable

/-- GGML_OPENMP is forced off in portable builds (no vcomp/libgomp on the user's PC). -/
def ggmlOpenmpForcedOff (portable : Bool) : Bool := portable

/-- Static MSVC runtime only for portable MSVC builds. -/
def staticRuntime (portable msvc : Bool) : Bool := portable && msvc

/-- Native and portable are mutually exclusive. -/
theorem native_xor_portable (p : Bool) : ggmlNative p = !p := by
  rfl

/-- A non-portable build never uses the static MSVC runtime. -/
theorem no_static_when_native (msvc : Bool) : staticRuntime false msvc = false := by
  simp [staticRuntime]

/-- The portable ISA baseline switches five features on. -/
def portableIsa : List String := ["GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C", "GGML_BMI2"]

/-- The baseline has five distinct feature flags. -/
theorem portable_isa_distinct : portableIsa.length = 5 ∧ portableIsa.Nodup := by
  decide

/-- The CUDA backend flag simply mirrors the `STRATA_VISION_CUDA` option. -/
def ggmlCuda (opt : Bool) : Bool := opt

/-- CUDA is on iff the option is on. -/
theorem cuda_mirrors_option (opt : Bool) : ggmlCuda opt = opt := by
  rfl

end Strata.VisionEncoderService
