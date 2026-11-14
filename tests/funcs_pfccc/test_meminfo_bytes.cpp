// meminfo_bytes(text, key): value of a /proc/meminfo key in bytes.
// kB cells multiplied by 1024; whole-word key match; 0 when missing/malformed.
// Ref: setup.py meminfo_bytes()
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    const std::string mi =
        "MemTotal:       528000000 kB\nMemFree: 1 kB\nMemAvailable: 370000000 kB\n";
    using baloon::setup::meminfo_bytes;
    assert(meminfo_bytes(mi, "MemTotal") == 528000000ull * 1024);
    assert(meminfo_bytes(mi, "MemAvailable") == 370000000ull * 1024);
    assert(meminfo_bytes(mi, "MemFree") == 1024ull);
    // Prefix must match whole key (not "Mem")
    assert(meminfo_bytes(mi, "Mem") == 0);
    // Missing key
    assert(meminfo_bytes(mi, "Nope") == 0);
    // Malformed value
    assert(meminfo_bytes("MemTotal: abc kB", "MemTotal") == 0);
    return 0;
}
