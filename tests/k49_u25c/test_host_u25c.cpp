// Host-only run of the u25c78c49e6 invariants (links the production core sources).
#include "invariants.hpp"
int main() {
    u25c::run_all();
    if (g_fail) { std::printf("U25C HOST INVARIANTS FAILED (%d)\n", g_fail); return 1; }
    std::printf("U25C HOST INVARIANTS PASSED\n");
    return 0;
}
