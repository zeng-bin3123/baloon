// Unit 27 tests: invariants of the serving layer (chat frontend + engine/service/HTTP front).
#include <chrono>
#include <cstdio>
#include <deque>
#include <random>
#include <sstream>
#include <thread>
#include "baloon/serve_http.hpp"

using namespace baloon::serve_api;
namespace sf = baloon::serve_frontend;
namespace se = baloon::serve_engine;

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static std::string join(const std::vector<ChatEvent>& v, ChatEvent::Kind k) {
    std::string s; for (auto& e : v) if (e.kind == k) s += e.text; return s;
}
static std::vector<ChatEvent> merged(const std::vector<ChatEvent>& v) {
    std::vector<ChatEvent> o;
    for (auto& e : v) { if (!o.empty() && e.kind != ChatEvent::Call && o.back().kind == e.kind) o.back().text += e.text; else o.push_back(e); }
    return o;
}
static Message msg(const char* role, const std::string& text) { Message m; m.role = role; m.parts = parts_of(text); return m; }
static Message image_msg(const std::string& src) { Message m; m.role = "user"; Part p; p.image = true; p.source = src; m.parts.push_back(p); return m; }

static void test_budget() {
    CHECK(std::string(budget_effort_name(0)) == "low"); CHECK(std::string(budget_effort_name(2047)) == "low");
    CHECK(std::string(budget_effort_name(2048)) == "medium"); CHECK(std::string(budget_effort_name(8191)) == "medium");
    CHECK(std::string(budget_effort_name(8192)) == "xhigh"); CHECK(std::string(budget_effort_name(1u << 30)) == "xhigh");
    int prev = 0;
    for (uint64_t n = 0; n < 20000; ++n) {
        const std::string s = budget_effort_name(n);
        const int r = s == "low" ? 0 : s == "medium" ? 1 : 2;
        CHECK(r >= prev); prev = r;
    }
    CHECK(template_kwargs("high").effort == "xhigh" && template_kwargs("high").enable_thinking);
    CHECK(template_kwargs("low").effort == "low");
    CHECK(!template_kwargs("none").enable_thinking && template_kwargs("none").effort.empty() && template_kwargs("none").ok);
    CHECK(!template_kwargs("turbo").ok);
}

static void test_tool_call_strip() {
    auto c = decode_tool_call("<function=get>\n<parameter=a>\nx\n</parameter>\n<parameter=b>\n\n\ny\n\n\n</parameter>\n<parameter=c></parameter>\n</function>");
    CHECK(c && c->name == "get" && c->args.size() == 3);
    if (c && c->args.size() == 3) {
        CHECK(c->args[0].second == "x");
        CHECK(c->args[1].second == "\n\ny\n\n");  // exactly one newline lost at each end
        CHECK(c->args[2].second.empty());
    }
    CHECK(!decode_tool_call("no function here")); CHECK(!decode_tool_call("<function=>"));
    CHECK(!decode_tool_call("<function=f><parameter=a>unterminated"));
    CHECK(strip_value("\n") == "" && strip_value("\n\n") == "" && strip_value("a") == "a" && strip_value("\n\na\n") == "\na");
}

static void test_parser() {
    const std::string input = "think a <b> more</think>Hello <tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>tail<x";
    ReplyParser whole;
    auto ref = whole.feed(input); auto fin = whole.finish(); ref.insert(ref.end(), fin.begin(), fin.end());
    ref = merged(ref);
    CHECK(join(ref, ChatEvent::Reasoning) == "think a <b> more");
    CHECK(join(ref, ChatEvent::Content) == "Hello tail<x");
    CHECK(whole.calls() == 1 && !whole.bad_call() && !whole.truncated_call());
    std::mt19937 rng(11);
    for (int it = 0; it < 3000; ++it) {
        ReplyParser p; std::vector<ChatEvent> got; size_t pos = 0;
        while (pos < input.size()) {
            const size_t n = 1 + rng() % 7; const std::string ch = input.substr(pos, n); pos += ch.size();
            auto ev = p.feed(ch);
            // never a partial tag: whatever is held is a proper prefix of a tag
            CHECK(p.held() < std::string("</tool_call>").size());
            for (auto& e : ev) if (e.kind != ChatEvent::Call) {
                CHECK(e.text.find("</think>") == std::string::npos && e.text.find("<tool_call>") == std::string::npos);
            }
            got.insert(got.end(), ev.begin(), ev.end());
        }
        auto f = p.finish(); got.insert(got.end(), f.begin(), f.end());
        got = merged(got);
        CHECK(got.size() == ref.size());
        if (got.size() == ref.size()) for (size_t i = 0; i < got.size(); ++i) {
            CHECK(got[i].kind == ref[i].kind && got[i].text == ref[i].text);
            if (got[i].kind == ChatEvent::Call) CHECK(got[i].call == ref[i].call);
        }
    }
    // a tag split across deltas is never shown: held + emitted == buffer
    { ReplyParser p; auto a = p.feed("abc</thi"); CHECK(join(a, ChatEvent::Reasoning) == "abc" && p.held() == 5);
      auto b = p.feed("nk>x"); CHECK(join(b, ChatEvent::Content) == "x" && p.state() == sf::PState::Content); }
    // an unfinished call is never emitted
    { ReplyParser p(false); auto a = p.feed("hi<tool_call><function=f>"); auto b = p.finish();
      CHECK(join(a, ChatEvent::Content) == "hi" && p.truncated_call() && p.calls() == 0); CHECK(b.empty()); }
    // reasoning -> content -> call -> content only; no way back to reasoning
    { ReplyParser p; p.feed("a</think>b"); p.feed("</think>c"); CHECK(p.state() == sf::PState::Content); }
    // started in content: no reasoning at all
    { ReplyParser p(false); auto a = p.feed("plain </think> text"); auto b = p.finish(); a.insert(a.end(), b.begin(), b.end());
      CHECK(join(a, ChatEvent::Reasoning).empty() && join(a, ChatEvent::Content) == "plain </think> text"); }
}

static void test_messages() {
    auto o = from_openai({msg("developer", "d"), msg("function", "t"), msg("user", "u"), msg("assistant", "")});
    CHECK(o && o->size() == 3 && (*o)[0].role == "system" && (*o)[1].role == "tool" && (*o)[2].role == "user");
    CHECK(!from_openai({msg("wizard", "x")}));
    auto a = from_anthropic("sys", {msg("user", "u")});
    CHECK(a && a->size() == 2 && (*a)[0].role == "system" && text_of((*a)[0]) == "sys");
    auto a2 = from_anthropic("", {msg("user", "u")}); CHECK(a2 && a2->size() == 1);
    CHECK(!from_anthropic("", {msg("system", "x")}));
    Message m = msg("user", "hello"); Part p; p.image = true; p.source = "http://x/y.png"; m.parts.push_back(p);
    CHECK(has_image(m) && !has_image(msg("user", "t")) && text_of(m) == "hello");
    CHECK(collect_images({msg("user", "a"), m, image_msg("data:image/png;base64,AAAA")}).size() == 2);
    CHECK(collect_images({m, image_msg("data:image/png;base64,AAAA")})[1] == "data:image/png;base64,AAAA");
    auto s1 = image_source("data:image/png;base64,QUJD"); CHECK(s1.ok && s1.inline_data && s1.payload == "QUJD");
    auto s2 = image_source("https://h/i.png"); CHECK(s2.ok && !s2.inline_data && s2.payload == "https://h/i.png");
    CHECK(!image_source("ftp://x").ok && !image_source("data:image/png,raw").ok && !image_source("data:image/png;base64,").ok);
    // template: one image pad per image, thinking suffix by mode
    ChatTemplate t; TemplateKwargs on = template_kwargs("low"), off = template_kwargs("none");
    const std::string r = t.render({msg("user", "hi"), image_msg("http://a/b")}, on);
    CHECK(r.find("Reasoning effort: low") != std::string::npos);
    size_t pads = 0; for (size_t p2 = r.find("<|image_pad|>"); p2 != std::string::npos; p2 = r.find("<|image_pad|>", p2 + 1)) ++pads;
    CHECK(pads == 1);
    CHECK(r.size() >= 8 && r.substr(r.size() - 8) == "<think>\n");
    const std::string r2 = t.render({msg("user", "hi")}, off);
    CHECK(r2.find("Reasoning effort") == std::string::npos && r2.find("</think>") != std::string::npos);
}

static void test_expand() {
    const uint32_t P = kImagePad;
    CHECK(expand_pads({1, P, 2, P, 3}, P, {}) == (std::vector<uint32_t>{1, P, 2, P, 3}));   // no images: unchanged
    CHECK(expand_pads({1, P, 2, P, 3}, P, {3, 2}) == (std::vector<uint32_t>{1, P, P, P, 2, P, P, 3}));
    CHECK(expand_pads({1, P, 2}, P, {3, 9}) == (std::vector<uint32_t>{1, P, P, P, 2}));
    CHECK(expand_pads({1, 2, 3}, P, {5}) == (std::vector<uint32_t>{1, 2, 3}));
    std::mt19937 rng(5);
    for (int it = 0; it < 500; ++it) {
        std::vector<uint32_t> ids, cs; size_t pads = 0;
        for (int i = 0; i < 30; ++i) { bool pd = rng() % 4 == 0; ids.push_back(pd ? P : rng() % 200); pads += pd; }
        size_t want = 0; for (size_t i = 0; i < pads; ++i) { cs.push_back(rng() % 5); want += cs.back(); }
        auto o = expand_pads(ids, P, cs); size_t got = 0; for (auto t : o) got += t == P;
        CHECK(got == want);
    }
}

static void test_decoder() {
    const std::string text = "héllo wörld ✓ 日本語 😀 end";
    StreamDecoder d; std::string acc;
    for (unsigned char c : text) {
        const size_t before = d.sent();
        std::string delta = d.push(c);
        CHECK(d.sent() >= before && d.sent() <= d.decoded());
        CHECK(d.sent() == before + delta.size());          // next delta starts exactly where this one ended
        CHECK(utf8_complete_prefix(delta) == delta.size());   // never half a character
        acc += delta;
    }
    acc += d.flush();
    CHECK(acc == text);
    CHECK(ByteCodec::decode(ByteCodec::encode(text)) == text);
    CHECK(utf8_complete_prefix("a\xE2\x9C") == 1 && utf8_complete_prefix("a\xE2\x9C\x93") == 4 && utf8_complete_prefix("\xF0\x9F\x98") == 0);
    StreamDecoder e; e.push(0xE2); CHECK(e.sent() == 0); CHECK(e.flush().size() == 1);   // invalid tail still flushed at the end
    StreamDecoder f; CHECK(f.push(kEos).empty());
    const auto w = ByteCodec::encode_with_pad("a<|image_pad|>b<|image_pad|>", "<|image_pad|>");
    CHECK(w == (std::vector<uint32_t>{'a', kImagePad, 'b', kImagePad}));
}

struct Pipe : LineTransport {
    std::deque<std::string> in; std::vector<std::string> sent;
    bool write_line(const std::string& l) override { sent.push_back(l); return true; }
    bool read_line(std::string& l) override { if (in.empty()) return false; l = in.front(); in.pop_front(); return true; }
};
static void test_line_engine() {
    { Pipe p; p.in = {se::fmt_token('a'), se::fmt_token('b'), se::fmt_done()}; LineEngine e(p);
      std::vector<uint32_t> got; auto r = e.generate({1, 2, 3}, 5, [&](uint32_t t) { got.push_back(t); });
      CHECK(r.ok && r.n == 2 && r.stopped && got.size() == 2);
      auto g = se::parse_gen(p.sent.at(0)); CHECK(g.ok && g.max_new == 5 && g.tokens.size() == 3 && g.tokens[2] == 3); }
    { Pipe p; p.in = {"T 1", "T 2", "T 3", "DONE"}; LineEngine e(p);                       // exactly max_new: length stop
      auto r = e.generate({1}, 3, [](uint32_t) {}); CHECK(r.ok && r.n == 3 && !r.stopped); }
    { Pipe p; p.in = {"T 1", "T 2", "T 3", "DONE"}; LineEngine e(p); uint64_t seen = 0;     // overrun is an error, never delivered
      auto r = e.generate({1}, 2, [&](uint32_t) { ++seen; }); CHECK(!r.ok && seen <= 2); }
    { Pipe p; p.in = {"T 1", "T 256", "DONE"}; LineEngine e(p); auto r = e.generate({1}, 2, [](uint32_t) {}); CHECK(r.ok && r.stopped); }
    { Pipe p; p.in = {"T 1", "ERR boom"}; LineEngine e(p); auto r = e.generate({1}, 4, [](uint32_t) {}); CHECK(!r.ok && r.error == "boom"); }
    { Pipe p; p.in = {"T 1"}; LineEngine e(p); auto r = e.generate({1}, 4, [](uint32_t) {}); CHECK(!r.ok); }       // closed before DONE
    { Pipe p; p.in = {"X 1", "DONE"}; LineEngine e(p); CHECK(!e.generate({1}, 4, [](uint32_t) {}).ok); }
    { Pipe p; LineEngine e(p); CHECK(!e.generate({}, 4, [](uint32_t) {}).ok && !e.generate({1}, 0, [](uint32_t) {}).ok && p.sent.empty()); }
}

static void test_cache() {
    uint32_t calls = 0; ImageEmbedder v([&](const std::string& s) { ++calls; return (uint32_t)s.size(); });
    for (int i = 0; i < 200; ++i) { v.tokens_for("img" + std::to_string(i)); CHECK(v.size() <= 64); }
    CHECK(v.size() == 64);
    const uint32_t c0 = calls;
    v.tokens_for("img199"); CHECK(calls == c0);            // newest still cached
    v.tokens_for("img0"); CHECK(calls == c0 + 1);          // oldest long evicted
    v.tokens_for("img137"); CHECK(calls == c0 + 1);        // img0 evicted img136; 137..199 survive
    v.tokens_for("img136"); CHECK(calls == c0 + 2);        // the oldest was dropped first
    ImageEmbedder w([&](const std::string&) { ++calls; return 7u; });
    for (int i = 0; i < 64; ++i) w.tokens_for(std::to_string(i));
    const uint32_t c1 = calls; w.tokens_for("64"); CHECK(w.size() == 64); w.tokens_for("0"); CHECK(calls == c1 + 2);   // 0 was oldest: evicted
}

static ApiRequest req(const std::string& route, const std::string& text, uint64_t maxn) {
    ApiRequest r; r.route = route; r.max_tokens = maxn; r.messages.push_back(msg("user", text)); return r;
}
static void test_admission_and_finish() {
    const std::string call = "</think>Sure.<tool_call>\n<function=f>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>";
    // context boundary: probe the prompt length, then admit exactly at the limit and reject one below
    ScriptedEngine probe(ByteCodec::encode("</think>ok")); ImageEmbedder vis([](const std::string&) { return 4u; });
    ChatService big(probe, vis, 100000);
    const TemplateKwargs kw = template_kwargs("medium");
    const Prepared p0 = big.prepare({msg("user", "hello")}, kw, 10);
    CHECK(p0.ok); const uint64_t L = p0.ids.size();
    { ChatService s(probe, vis, L + 10); CHECK(s.prepare({msg("user", "hello")}, kw, 10).ok);
      const uint64_t calls0 = probe.calls; Prepared r = s.prepare({msg("user", "hello")}, kw, 11);
      CHECK(!r.ok && r.status == 400); CHECK(!s.run(r).ok && probe.calls == calls0);          // rejected, engine untouched
      CHECK(!ChatService(probe, vis, L + 9).prepare({msg("user", "hello")}, kw, 10).ok);
      CHECK(!s.prepare({msg("user", "hello")}, kw, 0).ok); }
    // images: one pad per image expands to the image token count
    { ScriptedEngine e(ByteCodec::encode("</think>x")); ChatService s(e, vis, 100000);
      Prepared a = s.prepare({msg("user", "p")}, kw, 4), b = s.prepare({msg("user", "p"), image_msg("https://i/1.png")}, kw, 4);
      auto cnt = [](const Prepared& p) { size_t n = 0; for (auto t : p.ids) n += t == kImagePad; return n; };
      CHECK(cnt(a) == 0 && cnt(b) == 4); CHECK(!s.prepare({image_msg("ftp://bad")}, kw, 4).ok); }
    // finish reasons through the full path
    { ScriptedEngine e(ByteCodec::encode(call)); ChatService s(e, vis, 100000); HttpServer srv(s, ""); 
      auto o = srv.handle(req("/v1/chat/completions", "go", 200)); CHECK(o.status == 200 && o.body.find("\"finish_reason\":\"tool_calls\"") != std::string::npos);
      CHECK(o.body.find("\"name\":\"f\"") != std::string::npos && o.body.find("\\\"k\\\":\\\"v\\\"") != std::string::npos);
      auto a = srv.handle(req("/v1/messages", "go", 200)); CHECK(a.body.find("\"stop_reason\":\"tool_use\"") != std::string::npos);
      CHECK(a.body.find("\"type\":\"tool_use\"") != std::string::npos);
      auto ol = srv.handle(req("/v1/chat/completions", "go", 20));       // cut off before the stop token -> length
      CHECK(ol.status == 200 && ol.body.find("\"finish_reason\":\"length\"") != std::string::npos && ol.body.find("tool_calls\"") == std::string::npos);
      auto al = srv.handle(req("/v1/messages", "go", 20)); CHECK(al.body.find("\"stop_reason\":\"max_tokens\"") != std::string::npos);
      auto os = srv.handle([&] { auto r = req("/v1/chat/completions", "go", 200); r.stream = true; return r; }());
      CHECK(os.content_type == "text/event-stream" && os.frames.size() >= 3 && os.frames.back() == "data: [DONE]\n\n");
      CHECK(os.frames[os.frames.size() - 2].find("tool_calls\"}") != std::string::npos);
      auto as = srv.handle([&] { auto r = req("/v1/messages", "go", 200); r.stream = true; return r; }());
      CHECK(as.frames.front().find("event: message_start") == 0 && as.frames.back().find("message_stop") != std::string::npos);
    }
    { ScriptedEngine e(ByteCodec::encode("</think>plain")); ChatService s(e, vis, 100000); HttpServer srv(s, "");
      CHECK(srv.handle(req("/v1/chat/completions", "go", 200)).body.find("\"finish_reason\":\"stop\"") != std::string::npos);
      CHECK(srv.handle(req("/v1/messages", "go", 200)).body.find("\"stop_reason\":\"end_turn\"") != std::string::npos);
      auto r = srv.handle(req("/v1/chat/completions", "go", 200));
      CHECK(r.body.find("\"total_tokens\"") != std::string::npos);
      CHECK(srv.handle(req("/v1/nope", "go", 5)).status == 404);
      auto bad = req("/v1/chat/completions", "go", 5); bad.effort = "turbo"; CHECK(srv.handle(bad).status == 400);
      auto big = req("/v1/chat/completions", "go", 5); big.max_tokens = 0; big.max_completion_tokens = 0; CHECK(srv.handle(big).status == 200);   // defaults to 1024
      { // max tokens default to 1024 when neither field is set: the context boundary is exactly prompt + 1024
        ScriptedEngine pe(ByteCodec::encode("</think>ok")); ChatService ref(pe, vis, 100000);
        const uint64_t Lp = ref.prepare({msg("user", "go")}, template_kwargs("medium"), 1).ids.size();
        auto z = req("/v1/chat/completions", "go", 0);
        ChatService fits(pe, vis, Lp + 1024), tight(pe, vis, Lp + 1023);
        CHECK(HttpServer(fits, "").handle(z).status == 200); CHECK(HttpServer(tight, "").handle(z).status == 400);
        z.max_completion_tokens = 7; ChatService small(pe, vis, Lp + 7); CHECK(HttpServer(small, "").handle(z).status == 200);   // wins over max_tokens
      }
      auto huge = req("/v1/chat/completions", "go", 100000000); CHECK(srv.handle(huge).status == 400); }
    // usage = prompt + completion, completion includes the stop token
    { ScriptedEngine e(ByteCodec::encode("</think>ab")); ChatService s(e, vis, 100000);
      Prepared p = s.prepare({msg("user", "q")}, kw, 50); Completion c = s.run(p);
      CHECK(c.ok && c.completion_tokens == 10 + 1 && c.stopped && c.prompt_tokens == p.ids.size()); }
}

static void test_auth() {
    ScriptedEngine e(ByteCodec::encode("</think>ok")); ImageEmbedder vis([](const std::string&) { return 1u; }); ChatService s(e, vis, 100000);
    { HttpServer open(s, ""); auto r = req("/v1/chat/completions", "x", 8); r.key = "anything"; CHECK(open.handle(r).status == 200); r.key = ""; CHECK(open.handle(r).status == 200); }
    { HttpServer locked(s, "secret"); auto r = req("/v1/chat/completions", "x", 8);
      CHECK(locked.handle(r).status == 401); r.key = "secre"; CHECK(locked.handle(r).status == 401); r.key = "secret!"; CHECK(locked.handle(r).status == 401);
      r.key = "secret"; CHECK(locked.handle(r).status == 200);
      auto m = req("/v1/messages", "x", 8); CHECK(locked.handle(m).status == 401); m.key = "secret"; CHECK(locked.handle(m).status == 200); }
}

struct SlowEngine : GenEngine {
    std::mutex m; std::vector<int> order; int active = 0, max_active = 0;
    GenResult generate(const std::vector<uint32_t>& prompt, uint64_t, const TokenSink& sink) override {
        { std::lock_guard<std::mutex> l(m); ++active; max_active = std::max(max_active, active); order.push_back((int)prompt.back()); }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        sink(kEos);
        { std::lock_guard<std::mutex> l(m); --active; }
        GenResult r; r.ok = true; r.n = 1; r.stopped = true; return r;
    }
};
static void test_fifo() {
    SlowEngine e; ImageEmbedder vis([](const std::string&) { return 1u; }); ChatService s(e, vis, 100000);
    std::vector<Prepared> ps;
    for (int i = 0; i < 8; ++i) { Prepared p; p.ok = true; p.max_new = 4; p.ids = {1, (uint32_t)(10 + i)}; ps.push_back(p); }
    std::vector<std::thread> th;
    for (int i = 0; i < 8; ++i) { th.emplace_back([&, i] { s.run(ps[i]); }); std::this_thread::sleep_for(std::chrono::milliseconds(8)); }
    for (auto& t : th) t.join();
    CHECK(e.max_active == 1);
    bool inorder = e.order.size() == 8; for (int i = 0; i < 8 && inorder; ++i) inorder = e.order[i] == 10 + i;
    CHECK(inorder); CHECK(s.served() == 8);
}

static void test_misc() {
    CHECK(json_str("a\"b\\c\n\x01") == "\"a\\\"b\\\\c\\n\\u0001\"");
    auto env = worker_env({"A=1", "CUDA_VISIBLE_DEVICES=7", "B=2"}, "0,1");
    CHECK(env.size() == 3 && env[2] == "CUDA_VISIBLE_DEVICES=0,1" && env[0] == "A=1");
    CHECK(worker_env({"A=1", "CUDA_VISIBLE_DEVICES=7"}, "").size() == 1);
    CHECK(worker_env({"CUDA_VISIBLE_DEVICES=7"}, "").size() == 0);
    ScriptedEngine e(ByteCodec::encode("</think>ok")); ImageEmbedder vis([](const std::string&) { return 1u; }); ChatService s(e, vis, 100000);
    HttpServer srv(s, ""); auto rs = run_requests(build_handler(srv), {req("/v1/chat/completions", "a", 8), req("/v1/nope", "b", 8)});
    CHECK(rs.size() == 2 && rs[0].status == 200 && rs[1].status == 404);
    std::istringstream in("hello\nworld\n"); std::ostringstream out; const char* av[] = {"x", "--ctx", "4096"};
    CHECK(serve_main(3, const_cast<char**>(av), in, out) == 0);
    CHECK(out.str().find("200 {") == 0 && out.str().find("\n200 {") != std::string::npos);
    const char* bad[] = {"x", "--ctx", "abc"}; CHECK(serve_main(3, const_cast<char**>(bad), in, out) == 2);
    const char* bad2[] = {"x", "--wat"}; CHECK(serve_main(2, const_cast<char**>(bad2), in, out) == 2);
    const char* bad3[] = {"x", "--ctx", "0"}; CHECK(serve_main(3, const_cast<char**>(bad3), in, out) == 2);
}

int main() {
    test_budget(); test_tool_call_strip(); test_parser(); test_messages(); test_expand(); test_decoder();
    test_line_engine(); test_cache(); test_admission_and_finish(); test_auth(); test_fifo(); test_misc();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    if (g_fail) return 1;
    std::printf("UNIT 27 TESTS PASSED\n");
    return 0;
}
