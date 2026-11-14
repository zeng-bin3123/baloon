// out(s): captured command output with trailing whitespace/newlines stripped.
// Ref: setup.py out(s) -> s.rstrip()
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    assert(baloon::setup::out("abc \n\r\n") == "abc");
    assert(baloon::setup::out("  a b") == "  a b");     // leading space preserved
    assert(baloon::setup::out("\n") == "");
    assert(baloon::setup::out("") == "");
    assert(baloon::setup::out("x\t \r\n") == "x");
    return 0;
}
