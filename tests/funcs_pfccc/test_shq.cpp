// shq(s): single-quote the string for shell commands; embedded ' escaped as '\''.
// Ref: setup.py _shq() / shq()
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    assert(baloon::setup::shq("simple") == "'simple'");
    assert(baloon::setup::shq("with space") == "'with space'");
    assert(baloon::setup::shq("it's") == "'it'\\''s'");
    assert(baloon::setup::shq("") == "''");
    assert(baloon::setup::shq("a'b'c") == "'a'\\''b'\\''c'");
    return 0;
}
