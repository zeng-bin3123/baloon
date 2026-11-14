// find_nvcc(root, path, exists): corex root/bin/nvcc first; falls back to PATH.
// Ref: setup.py find_nvcc(root, path, exists)
#include <cassert>
#include <set>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    std::set<std::string> fs = {"/usr/local/corex/bin/nvcc", "/opt/b/nvcc"};
    baloon::setup::Exists ex = [&](const std::string& p){ return fs.count(p) != 0; };
    // corex root wins over PATH
    assert(baloon::setup::find_nvcc("/usr/local/corex", "/opt/b", ex) == "/usr/local/corex/bin/nvcc");
    // non-existent root falls back to PATH
    assert(baloon::setup::find_nvcc("/nonexistent", "/opt/b", ex) == "/opt/b/nvcc");
    // empty root uses PATH
    assert(baloon::setup::find_nvcc("", "/opt/b", ex) == "/opt/b/nvcc");
    // nothing found
    assert(baloon::setup::find_nvcc("", "", ex).empty());
    return 0;
}
