// write_run_script(root, config, model): generate #!/bin/sh launcher script.
// CUDA config prepends LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:
// Ref: setup.py write_run_script(root, config, model)
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    auto cuda = baloon::setup::write_run_script("/r", "cuda", "m.gguf");
    assert(cuda.find("#!/bin/sh") == 0);
    assert(cuda.find("LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:") != std::string::npos);
    assert(cuda.find("build-cuda/bin/engine") != std::string::npos);
    assert(cuda.find("'m.gguf'") != std::string::npos);
    assert(cuda.find("\"$@\"") != std::string::npos);
    auto cpu = baloon::setup::write_run_script("/r", "avx2", "m.gguf");
    assert(cpu.find("LD_LIBRARY_PATH") == std::string::npos);
    assert(cpu.find("build-avx2/bin/engine") != std::string::npos);
    return 0;
}
