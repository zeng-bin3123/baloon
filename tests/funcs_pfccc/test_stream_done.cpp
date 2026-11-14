// stream() stops at the first [DONE] line; subsequent data: lines are ignored.
// Ref: chat.py stream() [DONE] handling
#include <cassert>
#include <string>
#include "baloon/chat_cli.hpp"
static std::string ev(const std::string& c) {
    return "data: {\"choices\":[{\"delta\":{\"content\":\"" + c + "\"}}]}\n\n";
}
int main() {
    // [DONE] stops parsing
    assert(baloon::chat::stream(ev("x") + "data: [DONE]\n" + ev("y")) == "x");
    // Bare [DONE] with no preceding data
    assert(baloon::chat::stream("data: [DONE]\n").empty());
    // [DONE] within longer stream
    assert(baloon::chat::stream(ev("a") + ev("b") + "data: [DONE]\n" + ev("c")) == "ab");
    return 0;
}
