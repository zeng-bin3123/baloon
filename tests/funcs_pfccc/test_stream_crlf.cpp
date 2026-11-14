// stream() must strip \r before \n from SSE lines (HTTP uses CRLF).
// Ref: chat.py stream() CRLF handling
#include <cassert>
#include <string>
#include "baloon/chat_cli.hpp"
int main() {
    // CRLF line endings
    std::string sse =
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n"
        "\r\n"
        "data: [DONE]\r\n";
    assert(baloon::chat::stream(sse) == "hi");
    // Mixed CRLF and LF
    std::string sse2 =
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\r\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n"
        "data: [DONE]\n";
    assert(baloon::chat::stream(sse2) == "ab");
    return 0;
}
