// driver_major(version_str): major version number of "M.m.p" string; -1 when absent.
// Ref: setup.py driver_major(v)
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    using baloon::setup::driver_major;
    assert(driver_major("3.2.1") == 3);
    assert(driver_major(" 535.104.05 ") == 535);
    assert(driver_major("12") == 12);
    assert(driver_major("") == -1);
    assert(driver_major("x.1") == -1);
    assert(driver_major(".5") == -1);
    assert(driver_major("0.9") == 0);
    return 0;
}
