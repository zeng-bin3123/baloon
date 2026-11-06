// Unit 19 tests: installer decisions (setup_tool.hpp) and the streaming chat client (chat_cli.hpp).
#include <cstdio>
#include <set>
#include <string>
#include <vector>
#include "baloon/chat_cli.hpp"
#include "baloon/setup_tool.hpp"

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)
using namespace baloon;

static void test_text() {
    CHECK(setup::ask("", true) && !setup::ask("", false));
    CHECK(setup::ask("Y", false) && setup::ask(" yes \n", false) && !setup::ask("No", true) && !setup::ask("n", true));
    CHECK(setup::ask("maybe", true) && !setup::ask("maybe", false));
    CHECK(setup::out("abc \n\r\n") == "abc" && setup::out("  a b") == "  a b" && setup::out("\n") == "");
    CHECK(setup::step(2, 5, "build") == "[2/5] build");
    CHECK(setup::mark(true) != setup::mark(false));
    CHECK(setup::ok("x") != setup::warn("x") && setup::warn("x") != setup::fail("x"));
}

static void test_memory() {
    CHECK(setup::ram_gb(1ull << 30) == 1.0 && setup::ram_gb(3ull << 29) == 1.5 && setup::free_gb(0) == 0.0);
    const std::string mi = "MemTotal:       528000000 kB\nMemFree:  1 kB\nMemAvailable:   370000000 kB\n";
    CHECK(setup::meminfo_bytes(mi, "MemTotal") == 528000000ull * 1024);
    CHECK(setup::meminfo_bytes(mi, "MemAvailable") == 370000000ull * 1024);
    CHECK(setup::meminfo_bytes(mi, "Mem") == 0);          // key must match whole
    CHECK(setup::meminfo_bytes(mi, "Nope") == 0 && setup::meminfo_bytes("MemTotal: abc kB", "MemTotal") == 0);
}

static std::string cpuinfo_with(const std::string& flags) {
    return "processor\t: 0\nmodel name\t: INTEL(R) XEON(R) GOLD 6530\nflags\t\t: " + flags + "\n\nprocessor\t: 1\nmodel name\t: other\nflags\t\t: sse\n";
}
static void test_cpu() {
    const std::string base = "fpu avx avx2 fma f16c avx512f avx512bw avx512dq avx512vl avx512_vbmi avx512_vnni";
    setup::CpuInfo c = setup::cpu_info(cpuinfo_with(base));
    CHECK(c.model == "INTEL(R) XEON(R) GOLD 6530" && c.logical == 2);          // first model name, every processor counted
    CHECK(c.avx2 && c.fma && c.f16c && c.avx512_full);                         // flags of the FIRST processor only
    // dropping any single required flag breaks the full-AVX-512 verdict
    const char* need[] = {"avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512_vbmi", "avx512_vnni", "f16c", "fma"};
    for (const char* n : need) {
        std::string fl; for (const std::string& f : setup::split(base, ' ')) if (f != n) fl += f + " ";
        CHECK(!setup::cpu_info(cpuinfo_with(fl)).avx512_full);
    }
    // "avx512f" must not be satisfied by a longer flag containing it as a prefix of another word
    CHECK(!setup::cpu_info(cpuinfo_with("avx2 fma f16c avx512fp16 avx512bw avx512dq avx512vl avx512_vbmi avx512_vnni")).avx512_full);
    // a flag present only on a LATER processor must not count (flags of the first processor decide)
    CHECK(!setup::cpu_info("processor\t: 0\nflags\t: avx2\n\nprocessor\t: 1\nflags\t: " + base + "\n").avx512_full);
    CHECK(!setup::cpu_info("processor\t: 0\nflags\t: avx2\n\nprocessor\t: 1\nflags\t: " + base + "\n").fma);
    c = setup::cpu_info(cpuinfo_with("avx2 fma"));
    CHECK(c.avx2 && c.fma && !c.f16c && !c.avx512_full);
    c = setup::cpu_info("");
    CHECK(c.model.empty() && c.logical == 0 && !c.avx2);
}

static void test_gpu() {
    CHECK(setup::driver_major("3.2.1") == 3 && setup::driver_major(" 535.104.05 ") == 535 && setup::driver_major("12") == 12);
    CHECK(setup::driver_major("") == -1 && setup::driver_major("x.1") == -1 && setup::driver_major(".5") == -1);
    const std::string csv = "name, memory.total [MiB], driver_version\nIluvatar BI-V100, 32768 MiB, 3.2.1\nIluvatar BI-V100, 32768 MiB, 3.2.1\nbad row\nfoo, 12 GB, 1.0\n, 5 MiB, 1\n";
    const std::vector<setup::GpuInfo> g = setup::gpu_info(csv);
    CHECK(g.size() == 2 && g[0].name == "Iluvatar BI-V100" && g[0].mib == 32768 && g[1].driver == "3.2.1");
    CHECK(setup::gpu_info("").empty());
}

static void test_tools() {
    std::set<std::string> fs = {"/usr/bin/cmake", "/opt/b/nvcc", "/usr/local/corex/bin/nvcc", "/vs/Community/VC/Auxiliary/Build/vcvars64.bat", "/vs/BuildTools/VC/Auxiliary/Build/vcvars64.bat"};
    setup::Exists ex = [&](const std::string& p) { return fs.count(p) != 0; };
    CHECK(setup::find_tool("cmake", "/bin::/usr/bin:/opt", ex) == "/usr/bin/cmake");   // empty entries skipped, order kept
    CHECK(setup::find_tool("nvcc", "/opt/b:/usr/local/corex/bin", ex) == "/opt/b/nvcc");
    CHECK(setup::find_tool("gcc", "/bin:/usr/bin", ex).empty());
    CHECK(setup::find_nvcc("/usr/local/corex", "/opt/b", ex) == "/usr/local/corex/bin/nvcc");  // root wins over PATH
    CHECK(setup::find_nvcc("/nonexistent", "/opt/b", ex) == "/opt/b/nvcc");
    CHECK(setup::find_nvcc("", "", ex).empty());
    CHECK(setup::find_vcvars({"/vs"}, ex) == "/vs/Community/VC/Auxiliary/Build/vcvars64.bat");   // edition order
    CHECK(setup::find_vcvars({"/none"}, ex).empty());
    const std::vector<std::string> d = setup::cuda_lib_dirs("/r");
    CHECK(d.size() == 2 && d[0] == "/r/lib64" && d[1] == "/r/lib");
}

static void test_configs() {
    CHECK(setup::installed_configs({"engine-cuda", "x", "engine-avx2", "engine-bogus", "engine-avx2"}) == (std::vector<std::string>{"avx2", "cuda"}));
    CHECK(setup::installed_configs({}).empty());
    setup::CpuInfo none, avx2; avx2.avx2 = avx2.fma = avx2.f16c = true;
    setup::CpuInfo full = avx2; full.avx512_full = true;
    std::vector<setup::GpuInfo> gpu(1), nogpu;
    CHECK(setup::choose_config(full, gpu) == "cuda" && setup::choose_config(none, gpu) == "cuda");
    CHECK(setup::choose_config(full, nogpu) == "avx512" && setup::choose_config(avx2, nogpu) == "avx2" && setup::choose_config(none, nogpu) == "portable");
    setup::CpuInfo partial; partial.avx2 = true;   // AVX2 without FMA/F16C is not enough
    CHECK(setup::choose_config(partial, nogpu) == "portable");
}

static bool has(const std::vector<std::string>& v, const std::string& s) { for (auto& x : v) if (x == s) return true; return false; }
static void test_plans() {
    setup::Runner r;
    CHECK(setup::run(r, "false") == 0 && r.log.size() == 1);            // dry: recorded, never executed
    setup::Runner live; live.dry = false;
    CHECK(setup::run(live, "true") == 0 && setup::run(live, "exit 3") != 0);
    CHECK(setup::shq("a'b") == "'a'\\''b'" && setup::shq("x y") == "'x y'");
    setup::Runner q;
    setup::download(q, "http://h/f", "/t/f");
    CHECK(q.log.back().find("'http://h/f'") != std::string::npos && q.log.back().find("-o '/t/f'") != std::string::npos);
    setup::pip_install(q, {}); CHECK(q.log.size() == 1);                 // nothing to install: no command
    setup::pip_install(q, {"numpy", "x y"}); CHECK(q.log.back().find("pip install") != std::string::npos && q.log.back().find("'x y'") != std::string::npos);
    setup::Runner b;
    CHECK(setup::build_engine(b, "/r", "cuda") == 0);
    CHECK(b.log.size() == 3 && b.log[0].find("git clone") == 0 && b.log[1].find("cmake -S '/r/llama.cpp' -B '/r/build-cuda'") == 0 && b.log[2].find("cmake --build") == 0);
    CHECK(b.log[1].find("-DGGML_CUDA=ON") != std::string::npos);
    for (const char* c : {"portable", "avx2", "avx512"}) CHECK(has(setup::cmake_flags(c), "-DGGML_CUDA=OFF"));
    CHECK(has(setup::cmake_flags("portable"), "-DGGML_NATIVE=OFF") && has(setup::cmake_flags("avx512"), "-DGGML_NATIVE=ON"));
    for (const char* c : {"portable", "avx2", "avx512", "cuda"}) {
        const auto f = setup::cmake_flags(c);
        CHECK(has(f, "-DBUILD_SHARED_LIBS=OFF") && has(f, "-DGGML_BACKEND_DL=OFF") && has(f, "-DLLAMA_BUILD_TOOLS=OFF") && has(f, "-DLLAMA_BUILD_MTMD=ON"));
    }
    setup::Runner p; setup::get_prebuilt(p, "avx2", "http://h", "/d");
    CHECK(p.log.size() == 2 && p.log[0].find("engine-avx2.tar.gz") != std::string::npos && p.log[1].find("tar -xzf") == 0);
    setup::Runner t; setup::install_build_tools(t); CHECK(t.log.size() == 2 && t.log[1].find("cmake") != std::string::npos);
    const std::string cuda = setup::write_run_script("/r", "cuda", "m.gguf"), cpu = setup::write_run_script("/r", "avx2", "m.gguf");
    CHECK(cuda.find("LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex/lib:") != std::string::npos && cpu.find("LD_LIBRARY_PATH") == std::string::npos);
    CHECK(cuda.find("build-cuda/bin/engine") != std::string::npos && cuda.find("\"$@\"") != std::string::npos && cuda.find("#!/bin/sh") == 0);
    // a failing step stops the chain: with a failing first command nothing after it is recorded (harmless commands only)
    setup::Runner fl; fl.dry = false;
    CHECK(setup::run(fl, "exit 1") != 0 && fl.log.size() == 1);
}

static std::string ev(const std::string& content) { return "data: {\"choices\":[{\"delta\":{\"content\":\"" + content + "\"}}]}\n\n"; }
static void test_stream() {
    std::vector<std::string> got;
    std::string all = chat::stream(ev("Hel") + ": keep-alive\n" + "event: x\n" + ev("lo") + "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n" + ev("") + "data: [DONE]\n" + ev("never"),
                                   [&](const std::string& d) { got.push_back(d); });
    CHECK(all == "Hello" && got == (std::vector<std::string>{"Hel", "lo"}));
    CHECK(chat::stream(ev("a\\nb\\\"c\\\\d\\/e\\t")) == "a\nb\"c\\d/e\t");
    CHECK(chat::stream(ev("\\u00e9\\u4e2d\\ud83d\\ude00")) == "\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80");
    CHECK(chat::stream(ev("\\ud83dX")) == "\xEF\xBF\xBDX");                         // lone surrogate -> U+FFFD
    CHECK(chat::stream(ev("ab") + "data: {\"choices\":[{\"delta\":{\"content\":\"cd\"}}]}") == "ab");   // unterminated line not released
    CHECK(chat::stream("data:{\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\r\ndata: garbage\ndata: {\"delta\":{\"content\":\"bad\\q\"}}\n") == "x");
    CHECK(chat::stream("").empty() && chat::stream("data: [DONE]\n").empty());
    // chunking invariance: releasing line by line equals the whole text
    std::string whole = ev("one ") + ev("two ") + ev("three"), acc;
    CHECK(chat::stream(whole) == "one two three");
}
static void test_request() {
    const std::string b = chat::build_request("m\"x", {{"user", "hi \"there\"\n"}, {"assistant", "ok"}}, 64);
    CHECK(b == "{\"model\":\"m\\\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"hi \\\"there\\\"\\n\"},{\"role\":\"assistant\",\"content\":\"ok\"}],\"max_tokens\":64,\"stream\":true}");
    CHECK(chat::build_request("m", {}, 1, false) == "{\"model\":\"m\",\"messages\":[],\"max_tokens\":1,\"stream\":false}");
    CHECK(chat::json_escape(std::string("\x01", 1)) == "\\u0001");
    CHECK(chat::json_escape("a\tb\rc\nd") == "a\\tb\\rc\\nd");   // short escapes, not \\u00XX
    // escape then stream-parse round trip for arbitrary text
    for (const std::string& s : {std::string("plain"), std::string("q\"b\\n\n\t\r"), std::string("\x1f\x07 end"), std::string("\xC3\xA9 \xF0\x9F\x98\x80")})
        CHECK(chat::stream(ev(chat::json_escape(s))) == s);
}

int main() {
    test_text(); test_memory(); test_cpu(); test_gpu(); test_tools(); test_configs(); test_plans(); test_stream(); test_request();
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("UNIT19 TESTS FAILED\n"); return 1; }
    std::printf("UNIT19 TESTS PASSED\n");
    return 0;
}
