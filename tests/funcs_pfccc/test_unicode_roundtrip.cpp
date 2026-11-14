// stream() correctly decodes \uXXXX and surrogate pairs back to UTF-8.
// Ref: chat.py stream() with JSON unicode sequences
#include <cassert>
#include <string>
#include "baloon/chat_cli.hpp"
static std::string ev(const std::string& c) {
    return "data: {\"choices\":[{\"delta\":{\"content\":\"" + c + "\"}}]}\n\n";
}
int main() {
    // BMP character: é (U+00E9) -> \u00e9 -> UTF-8 \xC3\xA9
    assert(baloon::chat::stream(ev("\\u00e9")) == "\xC3\xA9");
    // CJK: 中 (U+4E2D) -> \u4e2d -> UTF-8 \xE4\xB8\xAD
    assert(baloon::chat::stream(ev("\\u4e2d")) == "\xE4\xB8\xAD");
    // Emoji via surrogate pair: 😀 (U+1F600) -> \uD83D\uDE00 -> UTF-8 \xF0\x9F\x98\x80
    assert(baloon::chat::stream(ev("\\ud83d\\ude00")) == "\xF0\x9F\x98\x80");
    // Lone surrogate -> U+FFFD replacement
    assert(baloon::chat::stream(ev("\\ud83dX")) == "\xEF\xBF\xBDX");
    return 0;
}
