// run(runner, cmd): record cmd in runner.log; execute iff !runner.dry.
// Dry mode always returns 0; live mode returns system() result.
// Ref: setup.py run(runner, cmd)
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    // dry mode: records but never executes
    baloon::setup::Runner dry;
    assert(baloon::setup::run(dry, "false") == 0);  // would fail if executed
    assert(dry.log.size() == 1 && dry.log[0] == "false");
    assert(baloon::setup::run(dry, "exit 1") == 0);
    assert(dry.log.size() == 2);
    // live mode: actually runs
    baloon::setup::Runner live; live.dry = false;
    assert(baloon::setup::run(live, "true") == 0);
    assert(baloon::setup::run(live, "exit 3") != 0);
    assert(live.log.size() == 2);
    return 0;
}
