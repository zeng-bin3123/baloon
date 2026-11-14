// cmake_build(runner, src, bld, config): cmake configure + build.
// Config flags from cmake_flags(): CUDA on only for "cuda", NATIVE off for "portable".
// Ref: setup.py cmake_build(runner, src, bld, config)
#include <cassert>
#include <algorithm>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
static bool has(const std::vector<std::string>& v, const std::string& s) {
    for (auto& x : v) if (x == s) return true; return false;
}
int main() {
    // cmake_flags sanity
    auto fc = baloon::setup::cmake_flags("cuda");
    assert(has(fc, "-DGGML_CUDA=ON"));
    assert(has(fc, "-DBUILD_SHARED_LIBS=OFF"));
    assert(has(fc, "-DGGML_BACKEND_DL=OFF"));
    assert(has(fc, "-DLLAMA_BUILD_TOOLS=OFF"));
    assert(has(fc, "-DLLAMA_BUILD_MTMD=ON"));
    for (const char* c : {"portable","avx2","avx512"})
        assert(has(baloon::setup::cmake_flags(c), "-DGGML_CUDA=OFF"));
    assert(has(baloon::setup::cmake_flags("portable"), "-DGGML_NATIVE=OFF"));
    assert(has(baloon::setup::cmake_flags("avx512"), "-DGGML_NATIVE=ON"));
    // cmake_build records exactly 2 commands
    baloon::setup::Runner r;
    baloon::setup::cmake_build(r, "/s", "/b", "cuda");
    assert(r.log.size() == 2);
    assert(r.log[0].find("cmake -S '/s' -B '/b'") == 0);
    assert(r.log[1].find("cmake --build") == 0);
    return 0;
}
