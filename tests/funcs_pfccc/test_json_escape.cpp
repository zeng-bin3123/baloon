// json_escape(s): JSON-safe escaping of arbitrary Unicode text.
// Rules: control chars -> \uXXXX, special chars -> short escapes (\n \r \t \\ \").
// Ref: chat.py json_escape (via build_request)
#include <cassert>
#include <string>
#include "baloon/chat_cli.hpp"
int main() {
    using baloon::chat::json_escape;
    assert(json_escape("plain") == "plain");
    assert(json_escape("a\"b") == "a\\\"b");
    assert(json_escape("a\\b") == "a\\\\b");
    assert(json_escape("a\nb") == "a\\nb");
    assert(json_escape("a\rb") == "a\\rb");
    assert(json_escape("a\tb") == "a\\tb");
    assert(json_escape(std::string("\x01", 1)) == "\\u0001");
    assert(json_escape(std::string("\x1f", 1)) == "\\u001f");
    // UTF-8 passthrough unchanged
    assert(json_escape("\xC3\xA9") == "\xC3\xA9");
    return 0;
}
