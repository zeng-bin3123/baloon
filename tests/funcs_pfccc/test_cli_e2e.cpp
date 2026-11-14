// End-to-end integration test: probe inputs → hardware detection → plan generation.
// Matches the exact scenario from the real Iluvatar machine used in test.sh CLI test.
// Ref: setup.py main() / test.sh == CLI end to end ==
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    using namespace baloon::setup;
    // Captured from the real Iluvatar machine (Xeon Gold 6530 + 2× BI-V100)
    const std::string cpuinfo =
        "processor\t: 0\n"
        "model name\t: INTEL(R) XEON(R) GOLD 6530\n"
        "flags\t\t: avx2 fma f16c avx512f avx512bw avx512dq avx512vl avx512_vbmi avx512_vnni\n";
    const std::string meminfo =
        "MemTotal:       527000000 kB\nMemAvailable:   369000000 kB\n";
    const std::string smi =
        "name, memory.total [MiB], driver_version\n"
        "Iluvatar BI-V100, 32768 MiB, 3.2.1\n"
        "Iluvatar BI-V100, 32768 MiB, 3.2.1\n";
    auto c = cpu_info(cpuinfo);
    auto g = gpu_info(smi);
    // config: cuda (GPU present)
    assert(choose_config(c, g) == "cuda");
    // RAM: ~502.6 GiB
    double ram = ram_gb(meminfo_bytes(meminfo, "MemTotal"));
    assert(ram > 502.0 && ram < 503.0);
    // Plan: install(2) + git(1) + cmake configure(1) + cmake build(1) = 5 commands
    Runner r;
    install_build_tools(r);
    build_engine(r, "/r", "cuda");
    assert(r.log.size() == 5);
    // Run script has LD_LIBRARY_PATH
    auto rs = write_run_script("/r", "cuda", "m.gguf");
    assert(rs.find("LD_LIBRARY_PATH=/usr/local/corex/lib64") != std::string::npos);
    return 0;
}
