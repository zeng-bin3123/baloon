// main() in src/tools/setup_cli.cpp:
//   --cpuinfo F --meminfo F --smi F [--root DIR] [--model PATH]
//   reads probe files, prints hardware summary and dry-run build plan.
// Ref: setup.py main()
// This file documents the pipeline: cpu_info→gpu_info→choose_config→build_engine→write_run_script.
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    using namespace baloon::setup;
    CpuInfo none, avx2c; avx2c.avx2 = avx2c.fma = avx2c.f16c = true;
    CpuInfo full = avx2c; full.avx512_full = true;
    std::vector<GpuInfo> gpu(1), nogpu;
    // choose_config: GPU → cuda; full AVX-512 → avx512; avx2+fma+f16c → avx2; else → portable
    assert(choose_config(full, gpu) == "cuda");
    assert(choose_config(full, nogpu) == "avx512");
    assert(choose_config(avx2c, nogpu) == "avx2");
    assert(choose_config(none, nogpu) == "portable");
    // Plan for cuda config: install + git + cmake configure + cmake build + (run script output)
    Runner r;
    install_build_tools(r);
    build_engine(r, "/r", "cuda");
    assert(r.log.size() == 5);  // update + install + git + cmake -S + cmake --build
    return 0;
}
