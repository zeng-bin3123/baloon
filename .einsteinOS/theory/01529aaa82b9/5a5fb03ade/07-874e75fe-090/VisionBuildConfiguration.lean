/-!
# MODULE: vision_build_configuration
`tools/vision/CMakeLists.txt`: the optional `strata-vision` build.  Two user options (`STRATA_VISION_CUDA`,
`STRATA_PORTABLE`) determine the forced ggml cache variables.

## Source semantics
* CUDA is on exactly when `STRATA_VISION_CUDA` is on.
* Portable builds turn off native tuning and OpenMP and force the AVX2 baseline flags on.
* Non-portable builds use native tuning.
* Only the MTMD target is built from llama.cpp; tools, tests, examples, server, app, curl, openssl are off.
* The static MSVC runtime applies only to portable MSVC builds.

## Form template identifiers
Opts          -> the two CMake options
ggml          -> forced GGML_* variables
-/

namespace Strata.VisionBuildConfiguration

/-- The user-facing options. -/
structure Opts where
  cuda : Bool
  portable : Bool
  msvc : Bool

/-- Forced ggml variables. -/
structure Ggml where
  native : Bool
  openmp : Bool
  avx : Bool
  avx2 : Bool
  fma : Bool
  f16c : Bool
  bmi2 : Bool
  cuda : Bool
  backendDl : Bool

/-- The configuration the CMake file computes (OpenMP is left at its default, on, when not portable). -/
def ggml (o : Opts) : Ggml :=
  if o.portable then
    { native := false, openmp := false, avx := true, avx2 := true, fma := true, f16c := true,
      bmi2 := true, cuda := o.cuda, backendDl := false }
  else
    { native := true, openmp := true, avx := false, avx2 := false, fma := false, f16c := false,
      bmi2 := false, cuda := o.cuda, backendDl := false }

/-- CUDA mirrors the option, in both modes. -/
theorem cuda_mirrors (o : Opts) : (ggml o).cuda = o.cuda := by
  unfold ggml; split <;> rfl

/-- Portable builds disable native tuning and OpenMP. -/
theorem portable_no_native (o : Opts) (h : o.portable = true) :
    (ggml o).native = false ∧ (ggml o).openmp = false := by
  unfold ggml; simp [h]

/-- Portable builds force the AVX2 baseline. -/
theorem portable_avx2_baseline (o : Opts) (h : o.portable = true) :
    (ggml o).avx = true ∧ (ggml o).avx2 = true ∧ (ggml o).fma = true ∧ (ggml o).f16c = true ∧
      (ggml o).bmi2 = true := by
  unfold ggml; simp [h]

/-- Non-portable builds use native tuning. -/
theorem nonportable_native (o : Opts) (h : o.portable = false) : (ggml o).native = true := by
  unfold ggml; simp [h]

/-- Native and portable are mutually exclusive. -/
theorem native_xor_portable (o : Opts) : (ggml o).native = !o.portable := by
  unfold ggml; cases h : o.portable <;> simp

/-- Backends are statically linked in both modes. -/
theorem backend_static (o : Opts) : (ggml o).backendDl = false := by
  unfold ggml; split <;> rfl

/-- The CPU flags of a portable build do not depend on whether CUDA is requested. -/
theorem portable_cpu_flags_independent_of_cuda (o : Opts) (c : Bool) (h : o.portable = true) :
    (ggml o).avx2 = (ggml { o with cuda := c }).avx2 := by
  unfold ggml; simp [h]

/-- Static MSVC runtime is selected only for portable MSVC builds. -/
def staticRuntime (o : Opts) : Bool := o.portable && o.msvc

/-- Non-portable builds never force the static runtime. -/
theorem no_static_when_nonportable (o : Opts) (h : o.portable = false) : staticRuntime o = false := by
  unfold staticRuntime; simp [h]

/-- Non-MSVC builds never force the static runtime. -/
theorem no_static_when_not_msvc (o : Opts) (h : o.msvc = false) : staticRuntime o = false := by
  unfold staticRuntime; simp [h]

-- § forced llama.cpp build switches

/-- Names of the `LLAMA_BUILD_*` switches and their forced values. -/
def llamaSwitches : List (String × Bool) :=
  [("COMMON", false), ("TOOLS", false), ("TESTS", false), ("EXAMPLES", false), ("SERVER", false),
   ("APP", false), ("MTMD", true)]

/-- Exactly one switch (MTMD) is on. -/
theorem only_mtmd_on : (llamaSwitches.filter (·.2)).map (·.1) = ["MTMD"] := by decide

/-- Seven build switches are forced. -/
theorem switch_count : llamaSwitches.length = 7 := by decide

end Strata.VisionBuildConfiguration
