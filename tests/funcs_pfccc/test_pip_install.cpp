// pip_install(runner, pkgs): python3 -m pip install --break-system-packages <pkgs>.
// Empty list: no command recorded. Shell-quotes each package name.
// Ref: setup.py pip_install(runner, pkgs)
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    baloon::setup::Runner r;
    baloon::setup::pip_install(r, {});
    assert(r.log.empty());    // nothing to install → no command
    baloon::setup::pip_install(r, {"numpy", "x y"});
    assert(r.log.size() == 1);
    assert(r.log[0].find("pip install") != std::string::npos);
    assert(r.log[0].find("'x y'") != std::string::npos);  // space-containing pkg quoted
    assert(r.log[0].find("--break-system-packages") != std::string::npos);
    return 0;
}
