// find_vcvars(roots, exists): vcvars64.bat under VS roots, edition precedence order.
// Order: Enterprise > Professional > Community > BuildTools.
// Ref: setup.py find_vcvars(vs_roots, exists)
#include <cassert>
#include <set>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    std::set<std::string> fs = {
        "/vs/Community/VC/Auxiliary/Build/vcvars64.bat",
        "/vs/BuildTools/VC/Auxiliary/Build/vcvars64.bat"};
    baloon::setup::Exists ex = [&](const std::string& p){ return fs.count(p) != 0; };
    // Community preferred over BuildTools
    assert(baloon::setup::find_vcvars({"/vs"}, ex) == "/vs/Community/VC/Auxiliary/Build/vcvars64.bat");
    assert(baloon::setup::find_vcvars({"/none"}, ex).empty());
    assert(baloon::setup::find_vcvars({}, ex).empty());
    return 0;
}
