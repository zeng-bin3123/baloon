// corex-specific library paths for the Iluvatar BI-V100 build environment.
// write_run_script() prepends /usr/local/corex/lib64:/usr/local/corex/lib to LD_LIBRARY_PATH.
// Ref: ENV.md / setup.py cuda_lib_dirs + write_run_script
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    // cuda_lib_dirs with the actual corex path from the real machine
    auto dirs = baloon::setup::cuda_lib_dirs("/usr/local/corex");
    assert(dirs.size() == 2);
    assert(dirs[0] == "/usr/local/corex/lib64");
    assert(dirs[1] == "/usr/local/corex/lib");
    // write_run_script produces the correct LD_LIBRARY_PATH prefix
    auto s = baloon::setup::write_run_script("/opt/baloon", "cuda", "model.gguf");
    assert(s.find("LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:") != std::string::npos);
    // CPU config has no LD_LIBRARY_PATH
    auto s2 = baloon::setup::write_run_script("/opt/baloon", "avx512", "model.gguf");
    assert(s2.find("LD_LIBRARY_PATH") == std::string::npos);
    return 0;
}
