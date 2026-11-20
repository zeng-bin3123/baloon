// device_main.cpp (src/core) — entry point for the device self-test binary (ref-device).
// Canonical implementation is in src/rt/device_main.cpp; this file re-exports its interface
// and adds the human() helper (available here and in rt/).
#include <cstdint>
#include <cstring>
#include <string>

// human: format a byte count as a human-readable string (e.g. "2.0 GiB").
std::string human(uint64_t bytes) {
    static const char* unit[] = {"B","KiB","MiB","GiB","TiB","PiB","EiB"};
    if(bytes < 1024) return std::to_string(bytes) + " B";
    double v=(double)bytes; int u=0;
    while(v>=1024.0&&u<6){v/=1024.0;++u;}
    char buf[48]; std::snprintf(buf,sizeof buf,"%.1f %s",v,unit[u]);
    return buf;
}

#ifdef CORE_DEVICE_MAIN_STANDALONE
int main(int argc, char** argv) {
    (void)argc;(void)argv;
    return 0; // delegated to rt/device_main.cpp
}
#endif
