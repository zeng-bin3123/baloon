// say(): prepend "  " prefix to a user-facing message.
// Ref: setup.py say(msg) -> "  " + msg
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    assert(baloon::setup::say("hello") == "  hello");
    assert(baloon::setup::say("") == "  ");
    assert(baloon::setup::say("x\ny") == "  x\ny");
    return 0;
}
