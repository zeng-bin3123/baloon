// installed_configs(dirs): engine-* directory names → known tag subset in canonical order.
// Order: portable < avx2 < avx512 < cuda.  Duplicates silently deduplicated.
// Ref: setup.py installed_configs(dirs)
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    using baloon::setup::installed_configs;
    auto r = installed_configs({"engine-cuda", "x", "engine-avx2", "engine-bogus", "engine-avx2"});
    assert((r == std::vector<std::string>{"avx2", "cuda"}));
    assert(installed_configs({}).empty());
    auto all = installed_configs({"engine-portable","engine-avx2","engine-avx512","engine-cuda"});
    assert((all == std::vector<std::string>{"portable","avx2","avx512","cuda"}));
    return 0;
}
