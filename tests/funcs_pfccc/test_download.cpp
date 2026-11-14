// download(runner, url, dest): curl -fL --retry 3 -o dest url.
// Ref: setup.py download(runner, url, dest)
#include <cassert>
#include <string>
#include "baloon/setup_tool.hpp"
int main() {
    baloon::setup::Runner r;
    baloon::setup::download(r, "http://h/f", "/t/f");
    assert(r.log.size() == 1);
    assert(r.log[0].find("'http://h/f'") != std::string::npos);
    assert(r.log[0].find("-o '/t/f'") != std::string::npos);
    assert(r.log[0].find("--retry 3") != std::string::npos);
    assert(r.log[0].find("curl") == 0 || r.log[0].find("curl") != std::string::npos);
    return 0;
}
