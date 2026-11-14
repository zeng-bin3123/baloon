// install_build_tools(runner): apt-get update + apt-get install build-essential cmake git.
// Stops early if first command fails.
// Ref: setup.py install_build_tools(runner)
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    baloon::setup::Runner r;
    baloon::setup::install_build_tools(r);
    assert(r.log.size() == 2);
    assert(r.log[0].find("apt-get update") != std::string::npos);
    assert(r.log[1].find("cmake") != std::string::npos);
    assert(r.log[1].find("build-essential") != std::string::npos);
    return 0;
}
