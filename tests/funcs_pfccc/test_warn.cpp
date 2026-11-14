// warn(): "[warn] " prefix for non-fatal warnings.
// Ref: setup.py warn(msg) -> "[warn] " + msg
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    assert(baloon::setup::warn("low disk") == "[warn] low disk");
    assert(baloon::setup::warn("") == "[warn] ");
    // warn is distinct from ok and fail
    assert(baloon::setup::warn("x") != baloon::setup::ok("x"));
    assert(baloon::setup::warn("x") != baloon::setup::fail("x"));
    return 0;
}
