// main() in src/tools/chat_cli.cpp:
//   --request MSG [--model M] [--max-tokens N]  → print JSON request body
//   --parse                                       → read SSE from stdin, print assistant text
//   exactly one of --request / --parse required; exit 2 on usage error.
// Ref: chat.py main()
// This file documents the CLI contract; actual end-to-end is tested by test.sh.
#include <cassert>
#include <string>
#include "baloon/chat_cli.hpp"
int main() {
    // build_request produces valid JSON for the CLI's --request path
    auto b = baloon::chat::build_request("baloon", {{"user", "hi"}}, 256);
    assert(b.find("\"model\":\"baloon\"") != std::string::npos);
    assert(b.find("\"stream\":true") != std::string::npos);
    assert(b.find("\"max_tokens\":256") != std::string::npos);
    // --model override
    auto b2 = baloon::chat::build_request("my-model", {}, 1, false);
    assert(b2.find("\"model\":\"my-model\"") != std::string::npos);
    assert(b2.find("\"stream\":false") != std::string::npos);
    return 0;
}
