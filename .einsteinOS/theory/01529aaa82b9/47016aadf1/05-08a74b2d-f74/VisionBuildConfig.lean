/-!
# MODULE: vision_build_config
The configuration logic of the strata-vision CMake project: how the `STRATA_PORTABLE` and `STRATA_VISION_CUDA`
options determine the forced ggml / llama.cpp cache settings.

## Source semantics
* A portable build disables native CPU tuning and OpenMP and forces the AVX2 baseline flags on.
* A non-portable build enables native CPU tuning.
* `GGML_CUDA` mirrors `STRATA_VISION_CUDA`.
* Only the mtmd library is built from llama.cpp; tools, tests, examples, server and app are off.
* The static MSVC runtime is selected only for portable MSVC builds.

## Form template identifiers
STRATA_PORTABLE -> portable, STRATA_VISION_CUDA -> cuda, GGML_NATIVE -> native_tuning,
GGML_OPENMP -> openmp, LLAMA_BUILD_MTMD -> build_mtmd.
-/

namespace Strata.VisionBuildConfig

/-- The user-facing options. -/
structure Options where
  portable : Bool := false
  cuda : Bool := false
  msvc : Bool := false

/-- The forced cache settings that depend on the options. -/
structure Settings where
  native : Bool
  openmp : Option Bool
  avx : Bool
  avx2 : Bool
  fma : Bool
  f16c : Bool
  bmi2 : Bool
  ggmlCuda : Bool
  staticRuntime : Bool

/-- The settings produced by the CMake logic. -/
def configure (o : Options) : Settings :=
  if o.portable then
    { native := false, openmp := some false, avx := true, avx2 := true, fma := true, f16c := true,
      bmi2 := true, ggmlCuda := o.cuda, staticRuntime := o.msvc }
  else
    { native := true, openmp := none, avx := false, avx2 := false, fma := false, f16c := false,
      bmi2 := false, ggmlCuda := o.cuda, staticRuntime := false }

/-- A portable build never tunes for the build machine and never needs an OpenMP runtime. -/
theorem portable_not_native (o : Options) (h : o.portable = true) :
    (configure o).native = false ∧ (configure o).openmp = some false := by
  simp [configure, h]

/-- A portable build forces the AVX2 baseline on. -/
theorem portable_baseline (o : Options) (h : o.portable = true) :
    (configure o).avx = true ∧ (configure o).avx2 = true ∧ (configure o).fma = true ∧
    (configure o).f16c = true ∧ (configure o).bmi2 = true := by
  simp [configure, h]

/-- A non-portable build tunes for the native CPU. -/
theorem nonportable_native (o : Options) (h : o.portable = false) : (configure o).native = true := by
  simp [configure, h]

/-- The CUDA backend mirrors the option in both modes. -/
theorem cuda_mirrors (o : Options) : (configure o).ggmlCuda = o.cuda := by
  unfold configure
  split <;> rfl

/-- The static MSVC runtime is chosen only for portable MSVC builds. -/
theorem static_runtime_iff (o : Options) :
    (configure o).staticRuntime = true ↔ (o.portable = true ∧ o.msvc = true) := by
  unfold configure
  cases hp : o.portable <;> cases hm : o.msvc <;> simp

/-- Native tuning and the portable baseline are mutually exclusive. -/
theorem native_xor_baseline (o : Options) : (configure o).native = !(configure o).avx2 := by
  unfold configure
  split <;> rfl

/-- Flags forced off in every configuration of llama.cpp: tools, tests, examples, server, app, curl, openssl. -/
def alwaysOff : List String :=
  ["LLAMA_BUILD_COMMON", "LLAMA_BUILD_TOOLS", "LLAMA_BUILD_TESTS", "LLAMA_BUILD_EXAMPLES",
   "LLAMA_BUILD_SERVER", "LLAMA_BUILD_APP", "LLAMA_OPENSSL", "LLAMA_CURL"]

/-- The only llama.cpp component switched on is mtmd. -/
def alwaysOn : List String := ["LLAMA_BUILD_MTMD"]

/-- The on and off sets are disjoint. -/
theorem on_off_disjoint : ∀ s ∈ alwaysOn, s ∉ alwaysOff := by decide

/-- Eight options are forced off and one forced on. -/
theorem forced_counts : alwaysOff.length = 8 ∧ alwaysOn.length = 1 := by decide

end Strata.VisionBuildConfig
