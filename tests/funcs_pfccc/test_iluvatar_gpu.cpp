// Iluvatar BI-V100 specific: ixsmi CSV format for 4× GPU setup.
// Verifies the actual hardware from the real machine is parsed correctly.
// GPU: 4× Iluvatar BI-V100, 32768 MiB, driver 3.2.1, driver_major=3.
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    // Actual ixsmi --query-gpu=name,memory.total,driver_version --format=csv output
    const std::string csv =
        "name, memory.total [MiB], driver_version\n"
        "Iluvatar BI-V100, 32768 MiB, 3.2.1\n"
        "Iluvatar BI-V100, 32768 MiB, 3.2.1\n"
        "Iluvatar BI-V100, 32768 MiB, 3.2.1\n"
        "Iluvatar BI-V100, 32768 MiB, 3.2.1\n";
    auto g = baloon::setup::gpu_info(csv);
    assert(g.size() == 4);
    for (auto& x : g) {
        assert(x.name == "Iluvatar BI-V100");
        assert(x.mib == 32768);
        assert(baloon::setup::driver_major(x.driver) == 3);
    }
    // choose_config: 4 GPUs → cuda
    baloon::setup::CpuInfo c;
    assert(baloon::setup::choose_config(c, g) == "cuda");
    return 0;
}
