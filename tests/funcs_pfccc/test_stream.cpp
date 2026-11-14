// stream(sse, sink): parse OpenAI-style SSE; emit delta.content of choices[0]; stop at [DONE].
// Never emits a partial (unterminated) line. Skips comments, non-data lines, empty content.
// Ref: chat.py stream(sse, sink)
#include <cassert>
#include <string>
#include <vector>
#include "baloon/chat_cli.hpp"
static std::string ev(const std::string& c) {
    return "data: {\"choices\":[{\"delta\":{\"content\":\"" + c + "\"}}]}\n\n";
}
int main() {
    std::vector<std::string> got;
    std::string all = baloon::chat::stream(
        ev("Hel") + ": keep-alive\n" + "event: x\n" + ev("lo") +
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n" +
        ev("") + "data: [DONE]\n" + ev("never"),
        [&](const std::string& d){ got.push_back(d); });
    assert(all == "Hello");
    assert((got == std::vector<std::string>{"Hel", "lo"}));
    // Unterminated last line must not be emitted
    assert(baloon::chat::stream(ev("ab") + "data: {\"choices\":[{\"delta\":{\"content\":\"cd\"}}]}") == "ab");
    // [DONE] stops parsing; subsequent data: lines ignored
    assert(baloon::chat::stream("data: [DONE]\n" + ev("x")).empty());
    assert(baloon::chat::stream("").empty());
    return 0;
}
