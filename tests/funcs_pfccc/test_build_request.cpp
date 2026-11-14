// build_request(model, msgs, max_tokens, stream_on):
//   produce valid JSON for the OpenAI chat-completions endpoint.
// Ref: chat.py build_request()
#include <cassert>
#include <string>
#include <vector>
#include "baloon/chat_cli.hpp"
int main() {
    using namespace baloon::chat;
    // Basic request structure
    auto b = build_request("m", {{"user","hi"},{"assistant","ok"}}, 64);
    assert(b.find("\"model\":\"m\"") != std::string::npos);
    assert(b.find("\"stream\":true") != std::string::npos);
    assert(b.find("\"max_tokens\":64") != std::string::npos);
    // stream_on=false
    auto b2 = build_request("m", {}, 1, false);
    assert(b2.find("\"stream\":false") != std::string::npos);
    assert(b2.find("\"messages\":[]") != std::string::npos);
    // Model with special chars escaped
    auto b3 = build_request("m\"x", {}, 1);
    assert(b3.find("\"model\":\"m\\\"x\"") != std::string::npos);
    return 0;
}
