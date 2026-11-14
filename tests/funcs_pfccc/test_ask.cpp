// ask(answer, dflt): parse a yes/no answer; empty input returns default.
// Ref: setup.py ask(answer, default) → bool
// y/yes → true; n/no → false; empty or unrecognised → default (case-insensitive).
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    using baloon::setup::ask;
    assert(ask("", true) == true && ask("", false) == false);
    assert(ask("y", false) && ask("Y", false) && ask("yes", false) && ask(" yes \n", false));
    assert(!ask("n", true) && !ask("N", true) && !ask("no", true) && !ask("No", true));
    assert(ask("maybe", true) && !ask("maybe", false));
    assert(ask("YES", false) && !ask("NO", true));
    return 0;
}
