// build_engine(runner, root, config): clone llama.cpp + cmake_build; 3 total commands.
// Ref: setup.py build_engine(runner, root, config)
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    baloon::setup::Runner r;
    baloon::setup::build_engine(r, "/r", "cuda");
    assert(r.log.size() == 3);
    assert(r.log[0].find("git clone") == 0);              // get_llama_cpp
    assert(r.log[1].find("cmake -S '/r/llama.cpp'") == 0); // configure
    assert(r.log[1].find("'/r/build-cuda'") != std::string::npos);
    assert(r.log[1].find("-DGGML_CUDA=ON") != std::string::npos);
    assert(r.log[2].find("cmake --build") == 0);          // build
    return 0;
}
