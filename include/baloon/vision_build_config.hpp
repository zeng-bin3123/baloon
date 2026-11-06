// vision_build_config.hpp - model of the ref-vision CMake configuration: how REF_PORTABLE and
// REF_VISION_CUDA determine the forced ggml / llama.cpp cache settings.
#pragma once
#include <array>
#include <cstring>
#include <string>

namespace baloon {
namespace vision_cfg {

struct Options { bool portable = false, cuda = false, msvc = false; };

struct Settings {
    bool native;         // GGML_NATIVE
    bool openmp_forced;  // GGML_OPENMP is forced (only to OFF) when true
    bool openmp;         // value when forced
    bool avx, avx2, fma, f16c, bmi2;
    bool ggml_cuda;      // GGML_CUDA
    bool static_runtime;
    bool shared_libs;    // BUILD_SHARED_LIBS: always OFF (static libraries)
    bool backend_dl;     // GGML_BACKEND_DL: always OFF (backends are not dynamically loaded)
};

inline Settings configure(const Options& o) {
    if (o.portable) return {false, true, false, true, true, true, true, true, o.cuda, o.msvc, false, false};
    return {true, false, false, false, false, false, false, false, o.cuda, false, false, false};
}

constexpr std::array<const char*, 10> ALWAYS_OFF = {"LLAMA_BUILD_COMMON", "LLAMA_BUILD_TOOLS", "LLAMA_BUILD_TESTS",
    "LLAMA_BUILD_EXAMPLES", "LLAMA_BUILD_SERVER", "LLAMA_BUILD_APP", "LLAMA_OPENSSL", "LLAMA_CURL",
    "BUILD_SHARED_LIBS", "GGML_BACKEND_DL"};
constexpr std::array<const char*, 1> ALWAYS_ON = {"LLAMA_BUILD_MTMD"};

}  // namespace vision_cfg
}  // namespace baloon
