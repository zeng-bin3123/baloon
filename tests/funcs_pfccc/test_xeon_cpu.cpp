// Intel Xeon Gold 6530 specific: 128 logical CPUs, full AVX-512 suite.
// Verifies the actual hardware from the real machine is parsed correctly.
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    // Abbreviated /proc/cpuinfo from the real machine (2 sockets × 32C × 2T = 128 logical)
    std::string cpuinfo;
    for (int i = 0; i < 128; ++i) {
        cpuinfo += "processor\t: " + std::to_string(i) + "\n";
        if (i == 0) {
            cpuinfo += "model name\t: INTEL(R) XEON(R) GOLD 6530\n";
            cpuinfo += "flags\t\t: fpu avx avx2 fma f16c avx512f avx512bw avx512dq "
                       "avx512vl avx512_vbmi avx512_vnni\n";
        }
        cpuinfo += "\n";
    }
    auto c = baloon::setup::cpu_info(cpuinfo);
    assert(c.model == "INTEL(R) XEON(R) GOLD 6530");
    assert(c.logical == 128);
    assert(c.avx2 && c.fma && c.f16c && c.avx512_full);
    return 0;
}
