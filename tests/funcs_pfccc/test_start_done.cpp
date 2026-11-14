// start(name): "== name setup ==" header.
// done(run_cmd): "Setup complete. Start with: run_cmd" footer.
// Ref: setup.py start() / done()
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    assert(baloon::setup::start("baloon") == "== baloon setup ==");
    assert(baloon::setup::start("x") == "== x setup ==");
    assert(baloon::setup::done("/r/run.sh") == "Setup complete. Start with: /r/run.sh");
    assert(baloon::setup::done("") == "Setup complete. Start with: ");
    return 0;
}
