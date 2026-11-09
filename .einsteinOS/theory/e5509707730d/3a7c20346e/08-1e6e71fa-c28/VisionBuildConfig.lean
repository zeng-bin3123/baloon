/-!
# MODULE: vision_build_config
SOURCES: tools/vision/CMakeLists.txt
Build configuration of `strata-vision`: llama.cpp's mtmd library at the engine's pinned commit, with two
user options (`STRATA_VISION_CUDA`, `STRATA_PORTABLE`) that determine the ggml flags.

## Source semantics
* Always off (forced): tools, tests, examples, server, app, common, OpenSSL, curl, shared libs, backend-DL.
  Always on: `LLAMA_BUILD_MTMD`.
* `STRATA_PORTABLE` turns native tuning off, OpenMP off, and the AVX/AVX2/FMA/F16C/BMI2 baseline on;
  otherwise `GGML_NATIVE` is on. The static MSVC runtime applies only to portable MSVC builds.
* `GGML_CUDA` follows `STRATA_VISION_CUDA` exactly.
* Minimum CMake 3.21 and C++17.

## Form template identifiers
- STRATA_PORTABLE / STRATA_VISION_CUDA → `Config.portable` / `Config.cuda`
- GGML_NATIVE / GGML_OPENMP / GGML_CUDA → `native` / `openmp` / `cudaOn`
- CMAKE_MSVC_RUNTIME_LIBRARY            → `staticRuntime`
-/

namespace Strata.VisionBuild

-- § options

/-- The user-visible options. -/
structure Config where
  portable : Bool
  cuda : Bool
  msvc : Bool

/-- `GGML_NATIVE`. -/
def native (c : Config) : Bool := !c.portable

/-- `GGML_OPENMP` is forced off only in portable builds (left at the default otherwise). -/
def openmpForcedOff (c : Config) : Bool := c.portable

/-- The AVX2-family baseline flags are forced on only in portable builds. -/
def baselineIsa (c : Config) : Bool := c.portable

/-- `GGML_CUDA`. -/
def cudaOn (c : Config) : Bool := c.cuda

/-- Static MSVC runtime: portable and MSVC. -/
def staticRuntime (c : Config) : Bool := c.portable && c.msvc

/-- `GGML_BACKEND_DL` is always off. -/
def backendDl (_ : Config) : Bool := false

-- § consistency

/-- Native tuning and the portable baseline are mutually exclusive. -/
theorem native_xor_portable (c : Config) : ¬ (native c = true ∧ c.portable = true) := by
  unfold native
  intro h
  obtain ⟨h1, h2⟩ := h
  simp [h2] at h1

/-- Exactly one of {native tuning, portable baseline ISA} is selected. -/
theorem exactly_one_isa_mode (c : Config) : native c = !baselineIsa c := by
  rfl

/-- The CUDA flag follows the option exactly. -/
theorem cuda_follows_option (c : Config) : cudaOn c = c.cuda := by
  rfl

/-- The static runtime applies only to portable builds. -/
theorem static_runtime_needs_portable (c : Config) (h : staticRuntime c = true) : c.portable = true := by
  unfold staticRuntime at h
  cases hp : c.portable <;> simp_all

/-- A non-portable build never forces OpenMP off nor the static runtime. -/
theorem non_portable_defaults (c : Config) (h : c.portable = false) :
    openmpForcedOff c = false ∧ staticRuntime c = false := by
  unfold openmpForcedOff staticRuntime
  simp [h]

/-- Backend DL is off for every configuration. -/
theorem backend_dl_never (c : Config) : backendDl c = false := by
  rfl

-- § toolchain floor

/-- CMake 3.21 and C++17 minimums are ordered as stated. -/
theorem toolchain_floor : 3 * 100 + 21 = 321 ∧ 17 ≥ 17 := by
  decide

end Strata.VisionBuild
