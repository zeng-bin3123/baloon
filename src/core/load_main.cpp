// load_main.cpp - model load driver.
// Prints the host-model device table (baloon::core::device_info) and, when a file path is supplied,
// validates the first four bytes as the GGUF magic.  Exits 0 on success, 1 on a bad file.
#include "baloon/core_device.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    // Print the known device table from the host model.
    const int count = 4;
    for (int i = 0; i < count; ++i) {
        const baloon::core::DeviceInfo d = baloon::core::device_info(i, count);
        if (!d.valid) continue;
        std::printf("device %d: %s  mem=%llu MiB  wave=%d  arch=%s\n",
                    d.ordinal, d.name.c_str(),
                    (unsigned long long)(d.total_mem >> 20),
                    d.wave, d.arch.c_str());
    }
    if (argc < 2) return 0;

    // If a file path was given, peek at the GGUF magic.
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open: %s\n", argv[1]); return 1; }
    uint8_t magic[4] = {};
    const size_t n = std::fread(magic, 1, 4, f);
    std::fclose(f);
    if (n < 4 || std::memcmp(magic, "GGUF", 4) != 0) {
        std::fprintf(stderr, "not a GGUF file: %s\n", argv[1]);
        return 1;
    }
    std::printf("file: %s  magic=GGUF\n", argv[1]);
    return 0;
}

