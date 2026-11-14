// cmake_flags(config): verify every config produces required CMake -D flags.
// Ref: setup.py cmake_build() / vision_build_config.hpp
#include <algorithm>
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
static bool has(const std::vector<std::string>& v, const std::string& s) {
    for (auto& x : v) if (x == s) return true; return false;
}
int main() {
    for (const char* cfg : {"portable", "avx2", "avx512", "cuda"}) {
        auto f = baloon::setup::cmake_flags(cfg);
        // Always OFF
        assert(has(f, "-DBUILD_SHARED_LIBS=OFF"));
        assert(has(f, "-DGGML_BACKEND_DL=OFF"));
        assert(has(f, "-DLLAMA_BUILD_TOOLS=OFF"));
        assert(has(f, "-DLLAMA_BUILD_TESTS=OFF"));
        // Always ON
        assert(has(f, "-DLLAMA_BUILD_MTMD=ON"));
    }
    // Config-specific
    assert(has(baloon::setup::cmake_flags("cuda"), "-DGGML_CUDA=ON"));
    assert(has(baloon::setup::cmake_flags("portable"), "-DGGML_NATIVE=OFF"));
    assert(has(baloon::setup::cmake_flags("avx512"), "-DGGML_NATIVE=ON"));
    for (const char* c : {"portable", "avx2", "avx512"})
        assert(has(baloon::setup::cmake_flags(c), "-DGGML_CUDA=OFF"));
    return 0;
}
