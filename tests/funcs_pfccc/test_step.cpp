// step(n, total, title): "[n/total] title" progress display.
// Ref: setup.py step(n, total, title) -> "[n/total] title"
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    assert(baloon::setup::step(1, 5, "init") == "[1/5] init");
    assert(baloon::setup::step(2, 5, "build") == "[2/5] build");
    assert(baloon::setup::step(10, 10, "done") == "[10/10] done");
    assert(baloon::setup::step(0, 0, "") == "[0/0] ");
    return 0;
}
