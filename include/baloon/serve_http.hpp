// serve_http.hpp - one resident engine behind OpenAI / Anthropic style endpoints.
// ChatService serialises requests FIFO and rejects (never truncates) a request whose prompt plus max tokens exceed the
// context. LineEngine speaks the GEN / T / DONE / ERR protocol and never lets more than max_new tokens through. The
// image-embedding cache holds at most 64 entries (oldest evicted first). With no API key every request is authorised.
// "tool_calls" / "tool_use" iff a call was made and generation stopped naturally. The transport is abstracted: HttpServer
// maps a parsed ApiRequest to an ApiResponse, and run_requests / serve_main drive it without sockets.
#pragma once
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "baloon/serve_chat.hpp"
#include "baloon/serve_engine.hpp"
#include "baloon/serve_server.hpp"

namespace baloon {
namespace serve_api {

// ---- byte tokenizer / streaming detokenizer ---------------------------------------------------------------------------
constexpr uint32_t kEos = 256;       // end of turn
constexpr uint32_t kImagePad = 257;  // <|image_pad|>

struct ByteCodec {
    static std::vector<uint32_t> encode(const std::string& s) {
        std::vector<uint32_t> v;
        v.reserve(s.size());
        for (unsigned char c : s) v.push_back(c);
        return v;
    }
    static std::string decode(const std::vector<uint32_t>& ids) {
        std::string s;
        for (uint32_t t : ids) if (t < 256) s.push_back((char)t);
        return s;
    }
    // Text with every literal pad string mapped to the pad id and everything else to bytes.
    static std::vector<uint32_t> encode_with_pad(const std::string& s, const std::string& pad) {
        std::vector<uint32_t> v;
        size_t pos = 0;
        while (pad.size() && pos <= s.size()) {
            const size_t p = s.find(pad, pos);
            if (p == std::string::npos) break;
            for (size_t i = pos; i < p; ++i) v.push_back((unsigned char)s[i]);
            v.push_back(kImagePad);
            pos = p + pad.size();
        }
        for (size_t i = pos; i < s.size(); ++i) v.push_back((unsigned char)s[i]);
        return v;
    }
};

// Length of the longest prefix of b that does not end inside a UTF-8 sequence.
inline size_t utf8_complete_prefix(const std::string& b) {
    const size_t n = b.size();
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        const unsigned char c = (unsigned char)b[n - back];
        if ((c & 0xC0) == 0x80) continue;
        const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? n - back : n;
    }
    return n;
}

// push() returns only the text that became complete; sent never moves backwards, and the deltas concatenate to the text.
class StreamDecoder {
public:
    std::string push(uint32_t tok) {
        if (tok < 256) bytes_.push_back((char)tok);
        return take(utf8_complete_prefix(bytes_));
    }
    std::string flush() { return take(bytes_.size()); }
    size_t sent() const { return sent_; }
    size_t decoded() const { return bytes_.size(); }
private:
    std::string bytes_;
    size_t sent_ = 0;
    std::string take(size_t cut) {
        if (cut <= sent_) return std::string();
        std::string d = bytes_.substr(sent_, cut - sent_);
        sent_ = cut;
        return d;
    }
};

// Expand each pad id to counts[i] pads, in order; pads beyond the counts stay single; no counts -> ids unchanged.
inline std::vector<uint32_t> expand_pads(const std::vector<uint32_t>& ids, uint32_t pad, const std::vector<uint32_t>& counts) {
    std::vector<uint32_t> out;
    size_t k = 0;
    for (uint32_t t : ids) {
        if (t == pad && k < counts.size()) { for (uint32_t i = 0; i < counts[k]; ++i) out.push_back(pad); ++k; }
        else out.push_back(t);
    }
    return out;
}

// ---- engines ------------------------------------------------------------------------------------------------------------
struct GenResult {
    bool ok = false;
    std::string error;
    uint64_t n = 0;        // tokens delivered (the stop token included)
    bool stopped = false;  // natural stop (stop token) rather than the max_new limit
};
using TokenSink = std::function<void(uint32_t)>;

struct GenEngine {
    virtual ~GenEngine() = default;
    virtual GenResult generate(const std::vector<uint32_t>& prompt, uint64_t max_new, const TokenSink& sink) = 0;
};

// Deterministic engine: replays a script, then the stop token.
class ScriptedEngine : public GenEngine {
public:
    explicit ScriptedEngine(std::vector<uint32_t> script) : script_(std::move(script)) {}
    GenResult generate(const std::vector<uint32_t>& prompt, uint64_t max_new, const TokenSink& sink) override {
        GenResult r;
        last_prompt = prompt;
        ++calls;
        if (prompt.empty() || max_new == 0) { r.error = "empty prompt or max_new == 0"; return r; }
        r.ok = true;
        for (uint64_t i = 0; i < max_new; ++i) {
            const uint32_t t = i < script_.size() ? script_[i] : kEos;
            sink(t);
            ++r.n;
            if (t == kEos) { r.stopped = true; break; }
        }
        return r;
    }
    std::vector<uint32_t> last_prompt;
    uint64_t calls = 0;
private:
    std::vector<uint32_t> script_;
};

struct LineTransport {
    virtual ~LineTransport() = default;
    virtual bool write_line(const std::string& line) = 0;
    virtual bool read_line(std::string& line) = 0;  // false at end of stream
};

// Client side of "GEN <max_new> <tok>..." -> "T <tok>"* then exactly one "DONE" (or "ERR <msg>").
class LineEngine : public GenEngine {
public:
    explicit LineEngine(LineTransport& t) : t_(t) {}
    GenResult generate(const std::vector<uint32_t>& prompt, uint64_t max_new, const TokenSink& sink) override {
        GenResult r;
        if (prompt.empty() || max_new == 0) { r.error = "empty prompt or max_new == 0"; return r; }
        std::string line = "GEN " + std::to_string(max_new);
        for (uint32_t t : prompt) line += " " + std::to_string(t);
        if (!t_.write_line(line)) { r.error = "engine write failed"; return r; }
        uint32_t last = 0;
        for (;;) {
            std::string in;
            if (!t_.read_line(in)) { r.error = "engine closed before DONE"; return r; }
            if (in == "DONE") break;
            if (in.compare(0, 4, "ERR ") == 0) { r.error = in.substr(4); return r; }
            if (in.compare(0, 2, "T ") != 0) { r.error = "bad engine line"; return r; }
            char* end = nullptr;
            const long long v = std::strtoll(in.c_str() + 2, &end, 10);
            if (end == in.c_str() + 2 || *end != '\0' || v < 0) { r.error = "bad token line"; return r; }
            if (r.n >= max_new) { r.error = "engine exceeded max_new"; return r; }
            last = (uint32_t)v;
            ++r.n;
            sink(last);
        }
        r.ok = true;
        r.stopped = r.n < max_new || (r.n > 0 && last == kEos);
        return r;
    }
private:
    LineTransport& t_;
};

// ---- vision cache -----------------------------------------------------------------------------------------------------
// Image source -> number of image tokens; at most 64 entries (oldest evicted).
class ImageEmbedder {
public:
    explicit ImageEmbedder(std::function<uint32_t(const std::string&)> compute) : compute_(std::move(compute)) {}
    uint32_t tokens_for(const std::string& source) {
        if (const uint32_t* c = cache_.get(source)) { ++hits_; return *c; }
        ++misses_;
        const uint32_t n = compute_(source);
        cache_.put(source, n);
        return n;
    }
    size_t size() const { return cache_.size(); }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
private:
    std::function<uint32_t(const std::string&)> compute_;
    serve_server::ImageCache<uint32_t> cache_;
    uint64_t hits_ = 0, misses_ = 0;
};

// ---- service ------------------------------------------------------------------------------------------------------------
struct Prepared {
    bool ok = false;
    int status = 200;
    std::string error;
    std::vector<uint32_t> ids;
    uint64_t max_new = 0;
};
struct Completion {
    bool ok = false;
    int status = 200;
    std::string error;
    uint64_t prompt_tokens = 0, completion_tokens = 0;
    bool stopped = false;
    uint32_t calls = 0;
    std::vector<ChatEvent> events;
};

class ChatService {
public:
    ChatService(GenEngine& e, ImageEmbedder& v, uint64_t max_ctx) : eng_(e), vis_(v), ctx_(max_ctx) {}
    uint64_t context() const { return ctx_; }

    Prepared prepare(const std::vector<Message>& ms, const TemplateKwargs& kw, uint64_t max_new) {
        Prepared p;
        p.max_new = max_new;
        if (max_new == 0) { p.status = 400; p.error = "max tokens must be positive"; return p; }
        std::vector<uint32_t> counts;
        for (const std::string& src : collect_images(ms)) {
            if (!image_source(src).ok) { p.status = 400; p.error = "unsupported image source"; return p; }
            counts.push_back(vis_.tokens_for(src));
        }
        std::vector<uint32_t> ids = ByteCodec::encode_with_pad(tpl_.render(ms, kw), tpl_.image_pad);
        ids = expand_pads(ids, kImagePad, counts);
        if (!serve_engine::admit(ids.size(), max_new, ctx_)) {
            p.status = 400;
            p.error = "prompt (" + std::to_string(ids.size()) + ") + max tokens (" + std::to_string(max_new) +
                      ") exceed the context (" + std::to_string(ctx_) + ")";
            return p;
        }
        p.ok = true;
        p.ids = std::move(ids);
        return p;
    }

    // FIFO: requests enter the engine strictly in arrival order, one at a time.
    Completion run(const Prepared& p, bool start_in_reasoning = true) {
        Completion c;
        if (!p.ok) { c.status = p.status; c.error = p.error; return c; }
        Ticket t(*this);
        StreamDecoder dec;
        ReplyParser parser(start_in_reasoning);
        auto add = [&](std::vector<ChatEvent> evs) {
            for (ChatEvent& e : evs) {
                if (!c.events.empty() && e.kind != ChatEvent::Call && c.events.back().kind == e.kind) c.events.back().text += e.text;
                else c.events.push_back(std::move(e));
            }
        };
        const GenResult g = eng_.generate(p.ids, p.max_new, [&](uint32_t tok) {
            if (tok == kEos) return;
            add(parser.feed(dec.push(tok)));
        });
        if (!g.ok) { c.status = 500; c.error = g.error; return c; }
        add(parser.feed(dec.flush()));
        add(parser.finish());
        c.ok = true;
        c.prompt_tokens = p.ids.size();
        c.completion_tokens = g.n;
        c.stopped = g.stopped;
        c.calls = parser.calls();
        return c;
    }
    uint64_t served() const { return served_; }

private:
    GenEngine& eng_;
    ImageEmbedder& vis_;
    uint64_t ctx_;
    ChatTemplate tpl_;
    std::mutex m_;
    std::condition_variable cv_;
    uint64_t next_ticket_ = 0, now_serving_ = 0, served_ = 0;
    struct Ticket {
        ChatService& s;
        explicit Ticket(ChatService& x) : s(x) {
            std::unique_lock<std::mutex> l(s.m_);
            const uint64_t mine = s.next_ticket_++;
            s.cv_.wait(l, [&] { return s.now_serving_ == mine; });
        }
        ~Ticket() {
            std::lock_guard<std::mutex> l(s.m_);
            ++s.now_serving_; ++s.served_;
            s.cv_.notify_all();
        }
    };
};

// ---- JSON writers --------------------------------------------------------------------------------------------------------
inline std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o.push_back((char)c);
        }
    }
    return o + "\"";
}
inline std::string args_json(const ToolInvocation& c) {
    std::string o = "{";
    for (size_t i = 0; i < c.args.size(); ++i) { if (i) o += ","; o += json_str(c.args[i].first) + ":" + json_str(c.args[i].second); }
    return o + "}";
}
inline std::string openai_finish_name(const Completion& c) {
    return serve_engine::finish_name(serve_engine::openai_finish(c.calls, c.stopped));
}
inline std::string anthropic_stop_name(const Completion& c) {
    return serve_engine::stop_name(serve_engine::anthropic_stop(c.calls > 0, c.stopped));
}
inline std::string usage_json(const Completion& c) {
    return "{\"prompt_tokens\":" + std::to_string(c.prompt_tokens) + ",\"completion_tokens\":" + std::to_string(c.completion_tokens) +
           ",\"total_tokens\":" + std::to_string(serve_engine::total_tokens(c.prompt_tokens, c.completion_tokens)) + "}";
}

// OpenAI non-streaming body.
inline std::string openai_body(const std::string& id, const std::string& model, const Completion& c) {
    std::string content, reasoning, calls;
    uint32_t k = 0;
    for (const ChatEvent& e : c.events) {
        if (e.kind == ChatEvent::Content) content += e.text;
        else if (e.kind == ChatEvent::Reasoning) reasoning += e.text;
        else {
            if (k) calls += ",";
            calls += "{\"id\":\"call_" + std::to_string(k) + "\",\"type\":\"function\",\"function\":{\"name\":" + json_str(e.call.name) +
                     ",\"arguments\":" + json_str(args_json(e.call)) + "}}";
            ++k;
        }
    }
    std::string msg = "{\"role\":\"assistant\",\"content\":" + json_str(content);
    if (!reasoning.empty()) msg += ",\"reasoning_content\":" + json_str(reasoning);
    if (k) msg += ",\"tool_calls\":[" + calls + "]";
    msg += "}";
    return "{\"id\":" + json_str(id) + ",\"object\":\"chat.completion\",\"model\":" + json_str(model) + ",\"choices\":[{\"index\":0,\"message\":" +
           msg + ",\"finish_reason\":" + json_str(openai_finish_name(c)) + "}],\"usage\":" + usage_json(c) + "}";
}

// OpenAI streaming: one "data: ..." frame per chunk, then the [DONE] sentinel.
inline std::vector<std::string> openai_stream(const std::string& id, const std::string& model, const Completion& c) {
    std::vector<std::string> out;
    auto frame = [&](const std::string& delta, const std::string& fin) {
        out.push_back("data: {\"id\":" + json_str(id) + ",\"object\":\"chat.completion.chunk\",\"model\":" + json_str(model) +
                      ",\"choices\":[{\"index\":0,\"delta\":" + delta + ",\"finish_reason\":" + fin + "}]}\n\n");
    };
    frame("{\"role\":\"assistant\"}", "null");
    uint32_t k = 0;
    for (const ChatEvent& e : c.events) {
        if (e.kind == ChatEvent::Content) frame("{\"content\":" + json_str(e.text) + "}", "null");
        else if (e.kind == ChatEvent::Reasoning) frame("{\"reasoning_content\":" + json_str(e.text) + "}", "null");
        else {
            frame("{\"tool_calls\":[{\"index\":" + std::to_string(k) + ",\"id\":\"call_" + std::to_string(k) + "\",\"type\":\"function\",\"function\":{\"name\":" +
                  json_str(e.call.name) + ",\"arguments\":" + json_str(args_json(e.call)) + "}}]}", "null");
            ++k;
        }
    }
    frame("{}", json_str(openai_finish_name(c)));
    out.push_back("data: [DONE]\n\n");
    return out;
}

// Anthropic non-streaming body.
inline std::string anthropic_body(const std::string& id, const std::string& model, const Completion& c) {
    std::string blocks;
    uint32_t k = 0;
    for (const ChatEvent& e : c.events) {
        if (!blocks.empty()) blocks += ",";
        if (e.kind == ChatEvent::Reasoning) blocks += "{\"type\":\"thinking\",\"thinking\":" + json_str(e.text) + "}";
        else if (e.kind == ChatEvent::Content) blocks += "{\"type\":\"text\",\"text\":" + json_str(e.text) + "}";
        else blocks += "{\"type\":\"tool_use\",\"id\":\"toolu_" + std::to_string(k++) + "\",\"name\":" + json_str(e.call.name) + ",\"input\":" + args_json(e.call) + "}";
    }
    return "{\"id\":" + json_str(id) + ",\"type\":\"message\",\"role\":\"assistant\",\"model\":" + json_str(model) + ",\"content\":[" + blocks +
           "],\"stop_reason\":" + json_str(anthropic_stop_name(c)) + ",\"usage\":{\"input_tokens\":" + std::to_string(c.prompt_tokens) +
           ",\"output_tokens\":" + std::to_string(c.completion_tokens) + "}}";
}

// Anthropic streaming: message_start, one block per event, message_delta, message_stop.
inline std::vector<std::string> anthropic_stream(const std::string& id, const std::string& model, const Completion& c) {
    std::vector<std::string> out;
    auto ev = [&](const std::string& name, const std::string& data) { out.push_back("event: " + name + "\ndata: " + data + "\n\n"); };
    ev("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":" + json_str(id) + ",\"type\":\"message\",\"role\":\"assistant\",\"model\":" + json_str(model) +
       ",\"content\":[],\"usage\":{\"input_tokens\":" + std::to_string(c.prompt_tokens) + ",\"output_tokens\":0}}}");
    uint32_t idx = 0, k = 0;
    for (const ChatEvent& e : c.events) {
        const std::string i = std::to_string(idx);
        if (e.kind == ChatEvent::Call) {
            ev("content_block_start", "{\"type\":\"content_block_start\",\"index\":" + i + ",\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_" + std::to_string(k++) +
               "\",\"name\":" + json_str(e.call.name) + ",\"input\":{}}}");
            ev("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + i + ",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":" + json_str(args_json(e.call)) + "}}");
        } else {
            const bool th = e.kind == ChatEvent::Reasoning;
            ev("content_block_start", "{\"type\":\"content_block_start\",\"index\":" + i + ",\"content_block\":{\"type\":\"" + (th ? "thinking\",\"thinking\":\"\"" : "text\",\"text\":\"\"") + "}}");
            ev("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + i + ",\"delta\":{\"type\":\"" + (th ? "thinking_delta\",\"thinking\":" : "text_delta\",\"text\":") + json_str(e.text) + "}}");
        }
        ev("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":" + i + "}");
        ++idx;
    }
    ev("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":" + json_str(anthropic_stop_name(c)) + "},\"usage\":{\"output_tokens\":" + std::to_string(c.completion_tokens) + "}}");
    ev("message_stop", "{\"type\":\"message_stop\"}");
    return out;
}

// ---- HTTP-level server ----------------------------------------------------------------------------------------------------
struct ApiRequest {
    std::string route;   // "/v1/chat/completions" or "/v1/messages"
    std::string key;     // presented API key
    std::string model = "baloon";
    std::string system;  // Anthropic only
    std::vector<Message> messages;
    std::string effort = "medium";
    uint64_t max_tokens = 0, max_completion_tokens = 0;
    bool stream = false;
};
struct ApiResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;                  // non-streaming
    std::vector<std::string> frames;   // streaming
};

inline ApiResponse api_error(int status, const std::string& msg) {
    ApiResponse r;
    r.status = status;
    r.body = "{\"error\":{\"message\":" + json_str(msg) + "}}";
    return r;
}

class HttpServer {
public:
    HttpServer(ChatService& s, std::string api_key) : svc_(s), key_(std::move(api_key)) {}
    ApiResponse handle(const ApiRequest& rq) {
        if (!serve_server::authorized(key_, rq.key)) return api_error(401, "invalid API key");
        const bool oa = rq.route == "/v1/chat/completions", an = rq.route == "/v1/messages";
        if (!oa && !an) return api_error(404, "unknown route");
        std::optional<std::vector<Message>> ms = oa ? from_openai(rq.messages) : from_anthropic(rq.system, rq.messages);
        if (!ms) return api_error(400, "unsupported role");
        const TemplateKwargs kw = template_kwargs(rq.effort);
        if (!kw.ok) return api_error(400, "unknown reasoning effort");
        const uint64_t max_new = serve_server::pick_max(rq.max_completion_tokens, rq.max_tokens);
        const Prepared p = svc_.prepare(*ms, kw, max_new);
        if (!p.ok) return api_error(p.status, p.error);
        const Completion c = svc_.run(p, kw.enable_thinking);
        if (!c.ok) return api_error(c.status, c.error);
        const std::string id = (oa ? "chatcmpl-" : "msg_") + std::to_string(svc_.served());
        ApiResponse r;
        if (rq.stream) {
            r.content_type = "text/event-stream";
            r.frames = oa ? openai_stream(id, rq.model, c) : anthropic_stream(id, rq.model, c);
        } else {
            r.body = oa ? openai_body(id, rq.model, c) : anthropic_body(id, rq.model, c);
        }
        return r;
    }
private:
    ChatService& svc_;
    std::string key_;
};

using Handler = std::function<ApiResponse(const ApiRequest&)>;
inline Handler build_handler(HttpServer& s) {
    return [&s](const ApiRequest& r) { return s.handle(r); };
}
// Serve a batch of requests in arrival order (the socket-free serving loop).
inline std::vector<ApiResponse> run_requests(const Handler& h, const std::vector<ApiRequest>& rs) {
    std::vector<ApiResponse> out;
    out.reserve(rs.size());
    for (const ApiRequest& r : rs) out.push_back(h(r));
    return out;
}

// Environment for the engine worker process: the base environment with the GPU list pinned.
inline std::vector<std::string> worker_env(const std::vector<std::string>& base, const std::string& gpus) {
    std::vector<std::string> out;
    for (const std::string& e : base) if (e.rfind("CUDA_VISIBLE_DEVICES=", 0) != 0) out.push_back(e);
    if (!gpus.empty()) out.push_back("CUDA_VISIBLE_DEVICES=" + gpus);
    return out;
}

// Stdio front: `--ctx N --key K --gpus 0,1 --script TEXT`; each stdin line is a user turn answered on stdout (OpenAI body).
inline int serve_main(int argc, char** argv, std::istream& in = std::cin, std::ostream& out = std::cout) {
    uint64_t ctx = 4096;
    std::string key, gpus, script = "</think>ok";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](std::string& dst) { if (i + 1 >= argc) return false; dst = argv[++i]; return true; };
        std::string v;
        if (a == "--ctx") { if (!val(v) || v.empty() || v.find_first_not_of("0123456789") != std::string::npos) return 2; ctx = std::strtoull(v.c_str(), nullptr, 10); }
        else if (a == "--key") { if (!val(key)) return 2; }
        else if (a == "--gpus") { if (!val(gpus)) return 2; }
        else if (a == "--script") { if (!val(script)) return 2; }
        else return 2;
    }
    if (ctx == 0) return 2;
    ScriptedEngine eng(ByteCodec::encode(script));
    ImageEmbedder vis([](const std::string&) { return 16u; });
    ChatService svc(eng, vis, ctx);
    HttpServer srv(svc, key);
    const Handler h = build_handler(srv);
    std::string line;
    while (std::getline(in, line)) {
        ApiRequest r;
        r.route = "/v1/chat/completions";
        r.key = key;
        r.max_tokens = 256;
        Message m; m.role = "user"; m.parts = parts_of(line);
        r.messages.push_back(m);
        const ApiResponse resp = h(r);
        out << resp.status << " " << resp.body << "\n";
    }
    return 0;
}

}  // namespace serve_api
}  // namespace baloon

