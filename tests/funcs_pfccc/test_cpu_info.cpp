// cpu_info(cpuinfo_text): parse /proc/cpuinfo text.
// Returns CpuInfo{model, logical, avx2, fma, f16c, avx512_full}.
// Rules: first "model name", every "processor" counted, flags of FIRST processor only.
// Ref: setup.py cpu_info(text)
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
static std::string mkcpu(const std::string& flags) {
    return "processor\t: 0\nmodel name\t: INTEL(R) XEON(R) GOLD 6530\nflags\t\t: " + flags
         + "\n\nprocessor\t: 1\nmodel name\t: other\nflags\t\t: sse\n";
}
int main() {
    using baloon::setup::cpu_info;
    const std::string avx512 = "avx2 fma f16c avx512f avx512bw avx512dq avx512vl avx512_vbmi avx512_vnni";
    auto c = cpu_info(mkcpu(avx512));
    assert(c.model == "INTEL(R) XEON(R) GOLD 6530");
    assert(c.logical == 2);         // counts all processors
    assert(c.avx2 && c.fma && c.f16c && c.avx512_full);
    // flags of second processor must NOT count
    auto c2 = cpu_info("processor\t: 0\nflags\t: avx2\n\nprocessor\t: 1\nflags\t: " + avx512 + "\n");
    assert(c2.avx2 && !c2.fma && !c2.avx512_full);
    // empty input
    auto empty = cpu_info("");
    assert(empty.model.empty() && empty.logical == 0 && !empty.avx2);
    return 0;
}
