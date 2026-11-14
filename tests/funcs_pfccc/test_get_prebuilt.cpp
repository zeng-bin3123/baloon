// get_prebuilt(runner, config, base_url, dest): download + unpack engine-<config>.tar.gz.
// Ref: setup.py get_prebuilt(runner, config, base_url, dest)
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    baloon::setup::Runner r;
    baloon::setup::get_prebuilt(r, "avx2", "http://h", "/d");
    assert(r.log.size() == 2);
    assert(r.log[0].find("engine-avx2.tar.gz") != std::string::npos);  // download
    assert(r.log[1].find("tar -xzf") == 0);                              // unpack
    assert(r.log[1].find("'/d'") != std::string::npos);
    return 0;
}
