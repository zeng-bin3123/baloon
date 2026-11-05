// chat_cli.hpp - terminal chat client logic: request body construction and the streaming (server-sent events) reader.
// stream() releases the text of every complete `data:` line's choices[0].delta.content in order, stops at `[DONE]`,
// ignores comments / other fields / malformed payloads, and never releases a partial (unterminated) line.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace baloon {
namespace chat {

inline std::string json_escape(const std::string& s) {
    static const char* hex = "0123456789abcdef";
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { o += "\\u00"; o += hex[c >> 4]; o += hex[c & 15]; } else o += (char)c;
        }
    }
    return o;
}

inline void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
    else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
}

// Reads the JSON string literal starting at s[i] == '"'; on success i points past the closing quote.
inline bool read_json_string(const std::string& s, size_t& i, std::string& out) {
    if (i >= s.size() || s[i] != '"') return false;
    ++i; out.clear();
    auto hex4 = [&](size_t p, uint32_t& v) {
        if (p + 4 > s.size()) return false;
        v = 0;
        for (size_t k = 0; k < 4; ++k) {
            const char c = s[p + k]; v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else return false;
        }
        return true;
    };
    while (i < s.size()) {
        const char c = s[i++];
        if (c == '"') return true;
        if (c != '\\') { out += c; continue; }
        if (i >= s.size()) return false;
        const char e = s[i++];
        switch (e) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case '/': case '"': case '\\': out += e; break;
            case 'u': {
                uint32_t cp;
                if (!hex4(i, cp)) return false;
                i += 4;
                if (cp >= 0xD800 && cp < 0xDC00) {  // high surrogate: needs \uDC00..DFFF next
                    uint32_t lo;
                    if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u' && hex4(i + 2, lo) && lo >= 0xDC00 && lo < 0xE000) {
                        i += 6; cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else cp = 0xFFFD;
                } else if (cp >= 0xDC00 && cp < 0xE000) cp = 0xFFFD;
                put_utf8(out, cp);
                break;
            }
            default: return false;
        }
    }
    return false;
}

// content of choices[0].delta in one SSE payload; false when absent / malformed.
inline bool delta_content(const std::string& payload, std::string& out) {
    const std::string dkey = "\"delta\"";
    size_t p = payload.find(dkey);
    if (p == std::string::npos) return false;
    p = payload.find("\"content\"", p + dkey.size());
    if (p == std::string::npos) return false;
    p += 9;
    while (p < payload.size() && (payload[p] == ' ' || payload[p] == ':')) ++p;
    return read_json_string(payload, p, out);
}

// Feed the whole received text; sink gets each delta in order. Returns the concatenation of all deltas.
inline std::string stream(const std::string& sse, const std::function<void(const std::string&)>& sink = nullptr) {
    std::string all;
    size_t pos = 0;
    for (;;) {
        const size_t nl = sse.find('\n', pos);
        if (nl == std::string::npos) break;  // partial last line: never released
        std::string line = sse.substr(pos, nl - pos);
        pos = nl + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.compare(0, 5, "data:") != 0) continue;
        size_t b = 5;
        while (b < line.size() && line[b] == ' ') ++b;
        const std::string payload = line.substr(b);
        if (payload == "[DONE]") break;
        std::string d;
        if (delta_content(payload, d) && !d.empty()) { if (sink) sink(d); all += d; }
    }
    return all;
}

struct Message { std::string role, content; };

// Chat-completions request body for the conversation (roles/contents escaped), streaming on.
inline std::string build_request(const std::string& model, const std::vector<Message>& msgs, int max_tokens, bool stream_on = true) {
    std::string b = "{\"model\":\"" + json_escape(model) + "\",\"messages\":[";
    for (size_t i = 0; i < msgs.size(); ++i) {
        if (i) b += ",";
        b += "{\"role\":\"" + json_escape(msgs[i].role) + "\",\"content\":\"" + json_escape(msgs[i].content) + "\"}";
    }
    b += "],\"max_tokens\":" + std::to_string(max_tokens) + ",\"stream\":" + (stream_on ? "true" : "false") + "}";
    return b;
}

}  // namespace chat
}  // namespace baloon
