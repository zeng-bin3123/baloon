// ok(): "[ ok ] " prefix for success messages.
// Ref: setup.py ok(msg) -> "[ ok ] " + msg
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    assert(baloon::setup::ok("done") == "[ ok ] done");
    assert(baloon::setup::ok("") == "[ ok ] ");
    return 0;
}
