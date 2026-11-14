// choose_config(cpu, gpus): pick best engine config for the hardware.
// gpu → cuda; avx512_full → avx512; avx2+fma+f16c → avx2; else → portable.
// Ref: setup.py choose_config()
#include <cassert>
#include <string>
#include <vector>
#include "baloon/setup_tool.hpp"
int main() {
    using namespace baloon::setup;
    CpuInfo none, avx2c, avx512c;
    avx2c.avx2 = avx2c.fma = avx2c.f16c = true;
    avx512c = avx2c; avx512c.avx512_full = true;
    std::vector<GpuInfo> gpu(1), nogpu;
    // GPU always wins
    assert(choose_config(none, gpu) == "cuda");
    assert(choose_config(avx512c, gpu) == "cuda");
    // Without GPU: best ISA
    assert(choose_config(avx512c, nogpu) == "avx512");
    assert(choose_config(avx2c, nogpu) == "avx2");
    assert(choose_config(none, nogpu) == "portable");
    // avx2 without fma/f16c → portable (not enough)
    CpuInfo avx2only; avx2only.avx2 = true;
    assert(choose_config(avx2only, nogpu) == "portable");
    return 0;
}
