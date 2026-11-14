// cuda_lib_dirs(root): [root/lib64, root/lib] — standard CUDA/corex lib paths.
// Ref: setup.py cuda_lib_dirs(root)
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    auto d = baloon::setup::cuda_lib_dirs("/usr/local/corex");
    assert(d.size() == 2);
    assert(d[0] == "/usr/local/corex/lib64");
    assert(d[1] == "/usr/local/corex/lib");
    auto d2 = baloon::setup::cuda_lib_dirs("/r");
    assert(d2[0] == "/r/lib64" && d2[1] == "/r/lib");
    return 0;
}
