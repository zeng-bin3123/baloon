// mark(good): "OK" when good, "--" when not.
// Ref: setup.py mark(good) -> "OK" if good else "--"
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    assert(baloon::setup::mark(true) == "OK");
    assert(baloon::setup::mark(false) == "--");
    assert(baloon::setup::mark(true) != baloon::setup::mark(false));
    return 0;
}
