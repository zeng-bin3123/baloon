// serve_chat.hpp - text/image request normalisation and the streaming reply parser of the serving layer.
// OpenAI / Anthropic style messages become one neutral Message list; reasoning effort and thinking budgets map to
// template levels (monotone: <2048 low, <8192 medium, else xhigh); the reply parser splits model output into reasoning,
// content and tool calls and never emits a partial tag (the held suffix is a proper prefix of the awaited tag).
// Tool-call parameter values lose at most one leading and one trailing newline.
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "baloon/serve_frontend.hpp"

namespace baloon {
namespace serve_api {

// ---- messages ---------------------------------------------------------------------------------------------------------
struct ToolInvocation {
    std::string name;
    std::vector<std::pair<std::string, std::string>> args;  // in order of appearance
    bool operator==(const ToolInvocation& o) const { return name == o.name && args == o.args; }
};

struct Part {
    bool image = false;
    std::string text;    // text part
    std::string source;  // image part: data: URL or http(s) URL
};
struct Message {
    std::string role;
    std::vector<Part> parts;
    std::vector<ToolInvocation> calls;  // assistant tool calls
};

struct ImageSource {
    bool ok = false;
    bool inline_data = false;  // true: base64 payload carried in the request itself
    std::string payload;       // base64 text or URL
};

// Plain string content -> one text part (empty string -> no parts).
inline std::vector<Part> parts_of(const std::string& content) {
    std::vector<Part> v;
    if (!content.empty()) { Part p; p.text = content; v.push_back(p); }
    return v;
}
inline std::string text_of(const Message& m) {
    std::string s;
    for (const Part& p : m.parts) if (!p.image) s += p.text;
    return s;
}
inline bool has_image(const Message& m) {
    for (const Part& p : m.parts) if (p.image) return true;
    return false;
}
inline ImageSource image_source(const std::string& url) {
    ImageSource r;
    const std::string b64 = ";base64,";
    if (url.rfind("data:", 0) == 0) {
        const size_t k = url.find(b64);
        if (k == std::string::npos || k + b64.size() >= url.size()) return r;
        r.ok = true; r.inline_data = true; r.payload = url.substr(k + b64.size());
    } else if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
        r.ok = true; r.payload = url;
    }
    return r;
}
// Image sources of the whole conversation, in order of appearance.
inline std::vector<std::string> collect_images(const std::vector<Message>& ms) {
    std::vector<std::string> v;
    for (const Message& m : ms) for (const Part& p : m.parts) if (p.image) v.push_back(p.source);
    return v;
}

// OpenAI dialect: developer -> system, function -> tool; messages with neither content nor calls are dropped.
// Unknown roles reject the request (nullopt).
inline std::optional<std::vector<Message>> from_openai(const std::vector<Message>& in) {
    std::vector<Message> out;
    for (Message m : in) {
        if (m.role == "developer") m.role = "system";
        else if (m.role == "function") m.role = "tool";
        if (m.role != "system" && m.role != "user" && m.role != "assistant" && m.role != "tool") return std::nullopt;
        if (m.parts.empty() && m.calls.empty()) continue;
        out.push_back(std::move(m));
    }
    return out;
}
// Anthropic dialect: the system prompt is a separate field and goes first; only user/assistant turns are allowed.
inline std::optional<std::vector<Message>> from_anthropic(const std::string& system, const std::vector<Message>& in) {
    std::vector<Message> out;
    if (!system.empty()) { Message s; s.role = "system"; s.parts = parts_of(system); out.push_back(std::move(s)); }
    for (const Message& m : in) {
        if (m.role != "user" && m.role != "assistant") return std::nullopt;
        out.push_back(m);
    }
    return out;
}

// ---- reasoning effort ---------------------------------------------------------------------------------------------------
inline const char* level_name(serve_frontend::Level l) {
    return l == serve_frontend::Level::Low ? "low" : (l == serve_frontend::Level::Medium ? "medium" : "xhigh");
}
inline const char* budget_effort_name(uint64_t thinking_budget) {
    return level_name(serve_frontend::budget_level(thinking_budget));
}
struct TemplateKwargs {
    bool ok = false;           // false: unknown effort spelling
    bool enable_thinking = true;
    std::string effort;        // empty when thinking is disabled
};
inline TemplateKwargs template_kwargs(const std::string& effort_spelling) {
    TemplateKwargs k;
    const serve_frontend::Effort e = serve_frontend::effort_of(effort_spelling);
    if (!e.known) return k;
    k.ok = true;
    if (e.disabled) { k.enable_thinking = false; return k; }
    k.effort = level_name(e.level);
    return k;
}

// ---- chat template --------------------------------------------------------------------------------------------------------
struct ChatTemplate {
    std::string im_start = "<|im_start|>", im_end = "<|im_end|>";
    std::string image_pad = "<|image_pad|>", vision_start = "<|vision_start|>", vision_end = "<|vision_end|>";

    static std::string render_call(const ToolInvocation& c) {
        std::string s = "<tool_call>\n<function=" + c.name + ">\n";
        for (const auto& kv : c.args) s += "<parameter=" + kv.first + ">\n" + kv.second + "\n</parameter>\n";
        return s + "</function>\n</tool_call>";
    }
    // One <|image_pad|> per image here; the service expands each to one per image token.
    std::string render(const std::vector<Message>& ms, const TemplateKwargs& kw) const {
        std::string s;
        if (kw.enable_thinking && !kw.effort.empty())
            s += im_start + "system\nReasoning effort: " + kw.effort + im_end + "\n";
        for (const Message& m : ms) {
            s += im_start + m.role + "\n";
            bool any = false;
            for (const Part& p : m.parts) { s += p.image ? vision_start + image_pad + vision_end : p.text; any = true; }
            for (const ToolInvocation& c : m.calls) { if (any) s += "\n"; s += render_call(c); any = true; }
            s += im_end + "\n";
        }
        s += im_start + "assistant\n";
        s += kw.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
        return s;
    }
};

// ---- tool call body -------------------------------------------------------------------------------------------------------
// Body of one <tool_call>: <function=NAME><parameter=K>V</parameter>...</function>. Values lose one newline per end.
inline std::string strip_value(std::string v) {
    if (!v.empty() && v.front() == '\n') v.erase(0, 1);
    if (!v.empty() && v.back() == '\n') v.pop_back();
    return v;
}
inline std::optional<ToolInvocation> decode_tool_call(const std::string& body) {
    const std::string fo = "<function=", po = "<parameter=", pc = "</parameter>";
    size_t f = body.find(fo);
    if (f == std::string::npos) return std::nullopt;
    f += fo.size();
    const size_t fe = body.find('>', f);
    if (fe == std::string::npos) return std::nullopt;
    ToolInvocation c;
    c.name = body.substr(f, fe - f);
    if (c.name.empty()) return std::nullopt;
    size_t pos = fe + 1;
    for (;;) {
        const size_t p = body.find(po, pos);
        if (p == std::string::npos) break;
        const size_t ke = body.find('>', p + po.size());
        if (ke == std::string::npos) return std::nullopt;
        const size_t ve = body.find(pc, ke + 1);
        if (ve == std::string::npos) return std::nullopt;
        c.args.emplace_back(body.substr(p + po.size(), ke - p - po.size()), strip_value(body.substr(ke + 1, ve - ke - 1)));
        pos = ve + pc.size();
    }
    return c;
}

// ---- streaming reply parser --------------------------------------------------------------------------------------------
struct ChatEvent {
    enum Kind { Reasoning, Content, Call } kind = Content;
    std::string text;      // Reasoning / Content text
    ToolInvocation call;   // Call
};

class ReplyParser {
public:
    explicit ReplyParser(bool start_in_reasoning = true)
        : st_(start_in_reasoning ? serve_frontend::PState::Reasoning : serve_frontend::PState::Content) {}
    serve_frontend::PState state() const { return st_; }
    uint32_t calls() const { return calls_; }
    bool bad_call() const { return bad_; }         // a completed <tool_call> body that did not decode
    bool truncated_call() const { return trunc_; } // stream ended inside a <tool_call>
    size_t held() const { return buf_.size(); }

    std::vector<ChatEvent> feed(const std::string& chunk) { buf_ += chunk; return drain(false); }
    std::vector<ChatEvent> finish() { return drain(true); }

private:
    using PState = serve_frontend::PState;
    PState st_;
    std::string buf_, call_body_;
    uint32_t calls_ = 0;
    bool bad_ = false, trunc_ = false;
    const std::string te_ = "</think>", cs_ = "<tool_call>", ce_ = "</tool_call>";

    void emit(std::vector<ChatEvent>& out, const std::string& s) const {
        if (s.empty()) return;
        const ChatEvent::Kind k = st_ == PState::Reasoning ? ChatEvent::Reasoning : ChatEvent::Content;
        if (!out.empty() && out.back().kind == k) { out.back().text += s; return; }
        ChatEvent e; e.kind = k; e.text = s; out.push_back(std::move(e));
    }
    std::vector<ChatEvent> drain(bool fin) {
        std::vector<ChatEvent> out;
        for (;;) {
            const std::string& tag = st_ == PState::Reasoning ? te_ : st_ == PState::Content ? cs_ : ce_;
            const size_t p = buf_.find(tag);
            if (st_ == PState::Call) {
                if (p == std::string::npos) {
                    // the body is only released whole: keep everything, but do not let the buffer grow without bound
                    call_body_ += buf_.substr(0, buf_.size() - serve_frontend::hold(buf_, tag));
                    buf_.erase(0, buf_.size() - serve_frontend::hold(buf_, tag));
                    if (fin) { trunc_ = true; call_body_.clear(); buf_.clear(); }
                    return out;
                }
                call_body_ += buf_.substr(0, p);
                buf_.erase(0, p + tag.size());
                ChatEvent e; e.kind = ChatEvent::Call;
                if (auto c = decode_tool_call(call_body_)) { e.call = *c; out.push_back(std::move(e)); ++calls_; }
                else bad_ = true;
                call_body_.clear();
                st_ = PState::Content;
                continue;
            }
            if (p != std::string::npos) {
                emit(out, buf_.substr(0, p));
                buf_.erase(0, p + tag.size());
                st_ = serve_frontend::next(st_, st_ == PState::Reasoning ? serve_frontend::Ev::ThinkEnd : serve_frontend::Ev::CallStart);
                continue;
            }
            const size_t h = fin ? 0 : serve_frontend::hold(buf_, tag);
            emit(out, buf_.substr(0, buf_.size() - h));
            buf_.erase(0, buf_.size() - h);
            return out;
        }
    }
};

}  // namespace serve_api
}  // namespace baloon

