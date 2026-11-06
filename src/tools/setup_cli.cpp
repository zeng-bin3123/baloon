// setup_cli: `--cpuinfo F --meminfo F --smi F [--root DIR] [--model PATH]` reads probe outputs captured from the machine and
// prints the detected hardware, the chosen configuration and the dry-run command plan (nothing is executed).
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include "baloon/setup_tool.hpp"

static std::string slurp(const std::string& p) { std::ifstream f(p); return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); }

int main(int argc, char** argv) {
    using namespace baloon::setup;
    std::string cpu, mem, smi, root = "/opt/baloon", model = "model.gguf";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cpuinfo" && i + 1 < argc) cpu = slurp(argv[++i]);
        else if (a == "--meminfo" && i + 1 < argc) mem = slurp(argv[++i]);
        else if (a == "--smi" && i + 1 < argc) smi = slurp(argv[++i]);
        else if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--model" && i + 1 < argc) model = argv[++i];
        else { std::fprintf(stderr, "usage: setup_cli --cpuinfo F --meminfo F --smi F [--root DIR] [--model PATH]\n"); return a == "--help" ? 0 : 2; }
    }
    const CpuInfo c = cpu_info(cpu);
    const std::vector<GpuInfo> g = gpu_info(smi);
    std::puts(start("baloon").c_str());
    std::printf("cpu: %s (%d logical) avx2=%s avx512=%s\n", c.model.c_str(), c.logical, mark(c.avx2).c_str(), mark(c.avx512_full).c_str());
    std::printf("ram_gb: %.1f\n", ram_gb(meminfo_bytes(mem, "MemTotal")));
    std::printf("free_gb: %.1f\n", free_gb(meminfo_bytes(mem, "MemAvailable")));
    for (const GpuInfo& x : g) std::printf("gpu: %s %llu MiB driver_major=%d\n", x.name.c_str(), (unsigned long long)x.mib, driver_major(x.driver));
    const std::string cfg = choose_config(c, g);
    std::printf("config: %s\n", cfg.c_str());
    Runner r; r.dry = true;
    install_build_tools(r);
    build_engine(r, root, cfg);
    for (const std::string& l : r.log) std::printf("plan: %s\n", l.c_str());
    std::fputs(write_run_script(root, cfg, model).c_str(), stdout);
    std::puts(done(root + "/run.sh").c_str());
    return 0;
}
