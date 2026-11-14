// get_llama_cpp(runner, dir, tag): git clone --depth 1 --branch <tag> llama.cpp into dir.
// Ref: setup.py get_llama_cpp(runner, dir, tag)
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    baloon::setup::Runner r;
    baloon::setup::get_llama_cpp(r, "/r/llama.cpp", "master");
    assert(r.log.size() == 1);
    assert(r.log[0].find("git clone") == 0);
    assert(r.log[0].find("--depth 1") != std::string::npos);
    assert(r.log[0].find("'master'") != std::string::npos);
    assert(r.log[0].find("llama.cpp") != std::string::npos);
    assert(r.log[0].find("'/r/llama.cpp'") != std::string::npos);
    return 0;
}
