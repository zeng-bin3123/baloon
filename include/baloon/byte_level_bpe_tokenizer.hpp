// byte_level_bpe_tokenizer.hpp - GPT-2 style byte-level BPE: byte<->unicode bijection, single-step merge
// length law, special-token matching classes and vocabulary / merge bookkeeping.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace baloon {
namespace bpe {

constexpr uint32_t kVocabSize = 248320;   // qwen35 vocabulary size
constexpr uint32_t kNumMerges = 247587;   // qwen35 merge count
static_assert(kNumMerges < kVocabSize, "fewer merges than tokens");
static_assert(256 + kNumMerges <= kVocabSize, "byte tokens plus merges fit in the vocabulary");

// Token types.
constexpr int kTypeNormal = 1;
constexpr int kTypeControl = 3;
constexpr int kTypeUserDefined = 4;

// ---- byte <-> unicode ---------------------------------------------------------------------------
// Bytes 33..126, 161..172 and 174..255 map to themselves; the other 68 map to 256 + n in byte order.
constexpr bool printable(unsigned b) {
    return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
}

inline const std::array<uint32_t, 256>& byte_map() {
    static const std::array<uint32_t, 256> t = [] {
        std::array<uint32_t, 256> a{};
        uint32_t n = 0;
        for (unsigned b = 0; b < 256; ++b) a[b] = printable(b) ? b : 256 + n++;
        return a;
    }();
    return t;
}

inline uint32_t byte_to_codepoint(uint8_t b) { return byte_map()[b]; }

// -1 when the code point is not in the image of the map.
inline int codepoint_to_byte(uint32_t cp) {
    static const std::unordered_map<uint32_t, int> inv = [] {
        std::unordered_map<uint32_t, int> m;
        for (int b = 0; b < 256; ++b) m[byte_map()[b]] = b;
        return m;
    }();
    auto it = inv.find(cp);
    return it == inv.end() ? -1 : it->second;
}

inline void utf8_append(uint32_t cp, std::string& out) {
    if (cp < 0x80) out.push_back((char)cp);
    else if (cp < 0x800) { out.push_back((char)(0xC0 | (cp >> 6))); out.push_back((char)(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18))); out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

// Strict UTF-8 decoder (rejects truncated / overlong / surrogate / > U+10FFFF sequences).
inline bool utf8_decode(const std::string& s, std::vector<uint32_t>& cps) {
    cps.clear();
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t c = (uint8_t)s[i];
        uint32_t cp; size_t n; uint32_t min;
        if (c < 0x80) { cp = c; n = 1; min = 0; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 2; min = 0x80; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 3; min = 0x800; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; n = 4; min = 0x10000; }
        else return false;
        if (i + n > s.size()) return false;
        for (size_t k = 1; k < n; ++k) {
            const uint8_t cc = (uint8_t)s[i + k];
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        cps.push_back(cp);
        i += n;
    }
    return true;
}

// Raw bytes -> visible-glyph string (one code point per byte).
inline std::string bytes_to_mapped(const std::string& bytes) {
    std::string out;
    for (unsigned char b : bytes) utf8_append(byte_to_codepoint(b), out);
    return out;
}

// Visible-glyph string -> raw bytes; ok=false if any code point is outside the byte map.
inline std::string mapped_to_bytes(const std::string& mapped, bool* ok) {
    std::vector<uint32_t> cps;
    std::string out;
    bool good = utf8_decode(mapped, cps);
    if (good) {
        for (uint32_t cp : cps) {
            const int b = codepoint_to_byte(cp);
            if (b < 0) { good = false; break; }
            out.push_back((char)b);
        }
    }
    if (ok) *ok = good;
    return good ? out : std::string();
}

// ---- BPE step -----------------------------------------------------------------------------------
// Replace the adjacent pair at i by the single part m; shortens the word by exactly one.
template <typename T>
std::vector<T> merge_at(const std::vector<T>& l, size_t i, const T& m) {
    std::vector<T> out(l.begin(), l.begin() + (std::ptrdiff_t)i);
    out.push_back(m);
    out.insert(out.end(), l.begin() + (std::ptrdiff_t)(i + 2), l.end());
    return out;
}

// ---- special tokens and bookkeeping --------------------------------------------------------------
// USER_DEFINED is always matched; CONTROL only with parse_special; everything else never.
constexpr bool matched(int type, bool parse_special) {
    return type == kTypeUserDefined || (type == kTypeControl && parse_special);
}

// A merge line must split into exactly two parts.
constexpr bool valid_merge(size_t n_parts) { return n_parts == 2; }

// Split a merge line "A B" at single spaces. Returns false unless exactly two non-empty parts.
inline bool parse_merge_line(const std::string& line, std::string* a, std::string* b) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ' ') { parts.push_back(line.substr(start, i - start)); start = i + 1; }
    }
    if (!valid_merge(parts.size()) || parts[0].empty() || parts[1].empty()) return false;
    if (a) *a = parts[0];
    if (b) *b = parts[1];
    return true;
}

// Token ids must be inside the vocabulary.
inline bool id_in_range(int64_t id, uint64_t vocab) { return id >= 0 && (uint64_t)id < vocab; }

// ---- tokenizer ------------------------------------------------------------------------------------
class ByteLevelBpe {
public:
    // tokens[id] = (mapped token string, type); merges = lines "A B" in priority order.
    // Returns an empty string on success, else an error description.
    std::string load(std::vector<std::pair<std::string, int>> tokens, const std::vector<std::string>& merges) {
        tokens_ = std::move(tokens);
        str_to_id_.clear();
        rank_.clear();
        for (size_t i = 0; i < tokens_.size(); ++i) str_to_id_.emplace(tokens_[i].first, (int32_t)i);
        for (size_t r = 0; r < merges.size(); ++r) {
            std::string a, b;
            if (!parse_merge_line(merges[r], &a, &b)) return "bad merge line " + std::to_string(r);
            if (!str_to_id_.count(a) || !str_to_id_.count(b)) return "merge names unknown token at " + std::to_string(r);
            rank_.emplace(a + " " + b, (uint32_t)r);
        }
        if (merges.size() > tokens_.size()) return "more merges than tokens";
        return std::string();
    }

    size_t vocab_size() const { return tokens_.size(); }
    size_t merge_count() const { return rank_.size(); }

    // BPE-encode one pre-tokenized word given as raw bytes. Fails (false) when a resulting part is not
    // a vocabulary token.
    bool encode_word(const std::string& bytes, std::vector<int32_t>* ids) const {
        std::vector<std::string> parts;
        for (unsigned char b : bytes) { std::string s; utf8_append(byte_to_codepoint(b), s); parts.push_back(s); }
        while (parts.size() >= 2) {
            size_t best = parts.size();
            uint32_t best_rank = 0;
            for (size_t i = 0; i + 1 < parts.size(); ++i) {
                auto it = rank_.find(parts[i] + " " + parts[i + 1]);
                if (it != rank_.end() && (best == parts.size() || it->second < best_rank)) { best = i; best_rank = it->second; }
            }
            if (best == parts.size()) break;
            parts = merge_at<std::string>(parts, best, parts[best] + parts[best + 1]);
        }
        std::vector<int32_t> out;
        for (const auto& p : parts) {
            auto it = str_to_id_.find(p);
            if (it == str_to_id_.end()) return false;
            out.push_back(it->second);
        }
        if (ids) *ids = std::move(out);
        return true;
    }

    // Decode ids to raw bytes. Rejects ids outside the vocabulary. Normal tokens are inverted through the
    // byte map; control / user-defined tokens contribute their literal text.
    bool decode(const std::vector<int64_t>& ids, std::string* bytes) const {
        std::string out;
        for (int64_t id : ids) {
            if (!id_in_range(id, tokens_.size())) return false;
            const auto& t = tokens_[(size_t)id];
            if (t.second == kTypeNormal) {
                bool ok = false;
                out += mapped_to_bytes(t.first, &ok);
                if (!ok) return false;
            } else {
                out += t.first;
            }
        }
        if (bytes) *bytes = std::move(out);
        return true;
    }

    // Literal strings matched during special-token splitting.
    std::vector<std::pair<std::string, int32_t>> special_literals(bool parse_special) const {
        std::vector<std::pair<std::string, int32_t>> v;
        for (size_t i = 0; i < tokens_.size(); ++i)
            if (matched(tokens_[i].second, parse_special)) v.emplace_back(tokens_[i].first, (int32_t)i);
        return v;
    }

private:
    std::vector<std::pair<std::string, int>> tokens_;
    std::unordered_map<std::string, int32_t> str_to_id_;
    std::unordered_map<std::string, uint32_t> rank_;
};

}  // namespace bpe
}  // namespace baloon
