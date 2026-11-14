// gpu_info(csv): parse ixsmi --query-gpu CSV output.
// Skips header and malformed rows; cell format "32768 MiB".
// Ref: setup.py gpu_info(csv)
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    using baloon::setup::gpu_info;
    const std::string csv =
        "name, memory.total [MiB], driver_version\n"
        "Iluvatar BI-V100, 32768 MiB, 3.2.1\n"
        "Iluvatar BI-V100, 32768 MiB, 3.2.1\n"
        "bad row\n"
        "foo, 12 GB, 1.0\n"    // wrong unit
        ", 5 MiB, 1\n";        // empty name
    auto g = gpu_info(csv);
    assert(g.size() == 2);
    assert(g[0].name == "Iluvatar BI-V100" && g[0].mib == 32768);
    assert(g[1].driver == "3.2.1");
    assert(gpu_info("").empty());
    return 0;
}
