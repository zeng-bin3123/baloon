// fail(): "[FAIL] " prefix for fatal errors.
// Ref: setup.py fail(msg) -> "[FAIL] " + msg
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    assert(baloon::setup::fail("no cuda") == "[FAIL] no cuda");
    assert(baloon::setup::fail("") == "[FAIL] ");
    // fail is distinct from warn and ok
    assert(baloon::setup::fail("x") != baloon::setup::warn("x"));
    assert(baloon::setup::fail("x") != baloon::setup::ok("x"));
    return 0;
}
