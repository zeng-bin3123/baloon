// find_tool(name, path, exists): first dir/name in PATH-like list that exists.
// Empty dirs skipped; returns "" when not found.
// Ref: setup.py find_tool(name, path, exists)
#include <cassert>
#include <set>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    std::set<std::string> fs = {"/usr/bin/cmake", "/opt/b/nvcc"};
    baloon::setup::Exists ex = [&](const std::string& p){ return fs.count(p) != 0; };
    assert(baloon::setup::find_tool("cmake", "/bin::/usr/bin:/opt", ex) == "/usr/bin/cmake");
    assert(baloon::setup::find_tool("nvcc", "/opt/b:/usr/local", ex) == "/opt/b/nvcc");
    assert(baloon::setup::find_tool("gcc", "/bin:/usr/bin", ex).empty());
    assert(baloon::setup::find_tool("cmake", "", ex).empty());
    return 0;
}
