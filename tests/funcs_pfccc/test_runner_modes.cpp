// Runner dry vs live: dry records but doesn't execute; live executes.
// Early-termination: if a step fails (live mode), chain stops.
// Ref: setup.py Runner + run()
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    using namespace baloon::setup;
    // Dry mode: even a "false" command returns 0
    Runner dry;
    assert(run(dry, "false") == 0);
    assert(run(dry, "exit 42") == 0);
    assert(dry.log.size() == 2);
    // Live mode: true succeeds, false fails
    Runner live; live.dry = false;
    assert(run(live, "true") == 0);
    assert(run(live, "exit 7") != 0);
    // Chain stops on failure in live mode (cmake_build)
    Runner chain; chain.dry = false;
    int rc = cmake_build(chain, "/nonexistent_src", "/nonexistent_bld", "portable");
    // First cmake should fail (no source dir), second should NOT be logged
    assert(chain.log.size() == 1);  // only the failing step recorded
    assert(rc != 0);
    return 0;
}
