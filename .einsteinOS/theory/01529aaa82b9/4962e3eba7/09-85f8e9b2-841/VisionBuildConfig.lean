/-!
# MODULE: vision_build_config
Configuration resolution of `tools/vision/CMakeLists.txt`: how the options `STRATA_PORTABLE`,
`STRATA_VISION_CUDA` and the MSVC toolchain determine the forced llama.cpp/ggml cache variables.

## Source semantics
* Always: static libraries, only the `mtmd` part of llama.cpp (common/tools/tests/examples/server/app
  off), no OpenSSL/CURL, ggml backends not dynamic.
* Portable: `GGML_NATIVE` off, OpenMP off, AVX/AVX2/FMA/F16C/BMI2 forced on; static MSVC runtime only
  when also building with MSVC.
* Non-portable: `GGML_NATIVE` on.  `GGML_CUDA` mirrors `STRATA_VISION_CUDA`.

## Form template identifiers
* resolve     <- the `if(STRATA_PORTABLE) ... else() ... endif()` block
* msvcStatic  <- `if(STRATA_PORTABLE AND MSVC)`
-/

namespace Strata.VisionBuild

/-- The cache variables the file forces; `none` means left at the llama.cpp default. -/
structure Cfg where
  native : Bool
  openmp : Option Bool
  avx : Option Bool
  avx2 : Option Bool
  fma : Option Bool
  f16c : Option Bool
  bmi2 : Option Bool
  cuda : Bool
  sharedLibs : Bool
  backendDl : Bool
  mtmd : Bool
  common : Bool
  tools : Bool
  tests : Bool
  examples : Bool
  server : Bool
  app : Bool
  curl : Bool
  openssl : Bool

/-- Resolve the configuration from the two user options. -/
def resolve (portable cuda : Bool) : Cfg :=
  { native := !portable
    openmp := if portable then some false else none
    avx := if portable then some true else none
    avx2 := if portable then some true else none
    fma := if portable then some true else none
    f16c := if portable then some true else none
    bmi2 := if portable then some true else none
    cuda := cuda
    sharedLibs := false, backendDl := false, mtmd := true, common := false, tools := false,
    tests := false, examples := false, server := false, app := false, curl := false,
    openssl := false }

/-- Portable builds never use `-march=native`. -/
theorem portable_not_native (c : Bool) : (resolve true c).native = false := rfl

/-- Non-portable builds use native tuning. -/
theorem default_native (c : Bool) : (resolve false c).native = true := rfl

/-- Portable builds need no OpenMP runtime and force the AVX2 baseline. -/
theorem portable_baseline (c : Bool) :
    (resolve true c).openmp = some false ∧ (resolve true c).avx = some true ∧
    (resolve true c).avx2 = some true ∧ (resolve true c).fma = some true ∧
    (resolve true c).f16c = some true ∧ (resolve true c).bmi2 = some true :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Non-portable builds leave the instruction flags at their defaults (native decides). -/
theorem default_flags_untouched (c : Bool) :
    (resolve false c).openmp = none ∧ (resolve false c).avx2 = none := ⟨rfl, rfl⟩

/-- GPU encoding is exactly the user's CUDA option, whichever portability is chosen. -/
theorem cuda_passthrough (p c : Bool) : (resolve p c).cuda = c := rfl

/-- Only the multimodal library is built from llama.cpp, statically, with no network deps. -/
theorem fixed_options (p c : Bool) :
    (resolve p c).mtmd = true ∧ (resolve p c).common = false ∧ (resolve p c).tools = false ∧
    (resolve p c).server = false ∧ (resolve p c).sharedLibs = false ∧
    (resolve p c).backendDl = false ∧ (resolve p c).curl = false ∧
    (resolve p c).openssl = false :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Static MSVC runtime is selected only for portable builds under MSVC. -/
def msvcStatic (portable msvc : Bool) : Bool := portable && msvc

/-- `msvcStatic` is the conjunction of the two conditions. -/
theorem msvcStatic_iff (p m : Bool) : msvcStatic p m = true ↔ p = true ∧ m = true := by
  simp [msvcStatic]

/-- A non-portable build never forces the static runtime. -/
theorem no_static_when_not_portable (m : Bool) : msvcStatic false m = false := rfl

end Strata.VisionBuild
