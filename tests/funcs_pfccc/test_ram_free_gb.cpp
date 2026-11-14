// ram_gb(bytes): total RAM in GiB (bytes / 2^30).
// free_gb(bytes): available RAM in GiB (bytes / 2^30).
// Ref: setup.py ram_gb / free_gb → float
#include <cassert>
#include <cmath>
#include <cstdint>
#include "baloon/setup_tool.hpp"
int main() {
    using baloon::setup::ram_gb; using baloon::setup::free_gb;
    assert(ram_gb(1ull << 30) == 1.0);
    assert(ram_gb(3ull << 29) == 1.5);
    assert(ram_gb(0) == 0.0);
    assert(free_gb(1ull << 30) == 1.0);
    assert(free_gb(0) == 0.0);
    // 503 GiB (from actual machine)
    double r = ram_gb(527921112ull * 1024);
    assert(r > 500.0 && r < 505.0);
    return 0;
}
