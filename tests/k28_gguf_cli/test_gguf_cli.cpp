// Unit p45808a8a9c tests: the gguf_reader command line (src/artifact/gguf_reader.cpp), driven in-process through run().
#define BALOON_GGUF_CLI_MAIN_DISABLED 1
#include "../../src/artifact/gguf_reader.cpp"
#include <unistd.h>

using baloon::gguf_reader::kMagic;
using Bytes = std::vector<uint8_t>;
static long g_checks = 0, g_fail = 0;
#define CK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); } } while (0)

static void put(Bytes& b, uint64_t v, int n) { for (int i = 0; i < n; ++i) b.push_back((uint8_t)(v >> (8 * i))); }
static void pstr(Bytes& b, const std::string& s) { put(b, s.size(), 8); b.insert(b.end(), s.begin(), s.end()); }
static void kv_str(Bytes& b, const std::string& k, const std::string& v) { pstr(b, k); put(b, 8, 4); pstr(b, v); }
static void kv_u32(Bytes& b, const std::string& k, uint32_t v) { pstr(b, k); put(b, 4, 4); put(b, v, 4); }

// arch qwen3moe-like file: 2 tensors (F32 [4] at 0, Q2_0 [64,2] at 32 -> 36 bytes), alignment 32.
static Bytes build(const std::string& arch, uint32_t experts, uint32_t used) {
    Bytes b; put(b, kMagic, 4); put(b, 3, 4); put(b, 2, 8); put(b, 6, 8);
    kv_str(b, "general.architecture", arch);
    kv_u32(b, arch + ".expert_count", experts);
    kv_u32(b, arch + ".expert_used_count", used);
    pstr(b, "tok.scores"); put(b, 9, 4); put(b, 6, 4); put(b, 10, 8);   // 10 floats -> truncated display
    for (int i = 0; i < 10; ++i) put(b, 0x3f800000u, 4);
    kv_str(b, "note", "a\"b\\c\nd");
    pstr(b, "flag"); put(b, 7, 4); put(b, 1, 1);
    pstr(b, "w.f32"); put(b, 1, 4); put(b, 4, 8); put(b, 0, 4); put(b, 0, 8);
    pstr(b, "w.q2"); put(b, 2, 4); put(b, 64, 8); put(b, 2, 8); put(b, 42, 4); put(b, 32, 8);
    while (b.size() % 32) b.push_back(0);
    b.resize(b.size() + 32 + 36, 0);
    return b;
}

struct Res { int rc; std::string out, err; };
static std::string slurp(std::FILE* f) {
    std::string s; std::rewind(f); char buf[4096]; size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    return s;
}
static Res call(std::vector<std::string> args) {
    std::vector<char*> av; for (auto& a : args) av.push_back(&a[0]);
    std::FILE* o = std::tmpfile(); std::FILE* e = std::tmpfile();
    Res r; r.rc = baloon::gguf_cli::run((int)av.size(), av.data(), o, e);
    r.out = slurp(o); r.err = slurp(e); std::fclose(o); std::fclose(e); return r;
}
static std::string write_file(const std::string& name, const Bytes& b) {
    const std::string p = "/tmp/k28_" + std::to_string((long)getpid()) + "_" + name;
    std::FILE* f = std::fopen(p.c_str(), "wb"); std::fwrite(b.data(), 1, b.size(), f); std::fclose(f); return p;
}

int main() {
    const std::string good = write_file("good.gguf", build("qwen3moe", 4, 2));
    // info: layout invariants (data_start aligned, counts, tensor byte total = 16 + 36)
    Res r = call({"x", good, "info"});
    CK(r.rc == 0);
    CK(r.out.find("version: 3\n") == 0);
    CK(r.out.find("alignment: 32\n") != std::string::npos);
    CK(r.out.find("metadata: 6\n") != std::string::npos && r.out.find("tensors: 2\n") != std::string::npos);
    CK(r.out.find("architecture: qwen3moe\n") != std::string::npos);
    CK(r.out.find("tensor_bytes: 52\n") != std::string::npos);
    {   // data_start printed is a multiple of the alignment
        const size_t p = r.out.find("data_start: "); CK(p != std::string::npos);
        const unsigned long long ds = std::strtoull(r.out.c_str() + p + 12, nullptr, 10);
        CK(ds > 0 && ds % 32 == 0);
    }
    // meta: file order, types, array truncation, string escaping
    r = call({"x", good, "meta"});
    CK(r.rc == 0);
    CK(r.out.find("general.architecture [string] = \"qwen3moe\"\n") == 0);
    CK(r.out.find("qwen3moe.expert_count [u32] = 4\n") != std::string::npos);
    CK(r.out.find("tok.scores [array] = [1, 1, 1, 1, 1, 1, 1, 1, ...] (10 elements of f32)\n") != std::string::npos);
    CK(r.out.find("note [string] = \"a\\\"b\\\\c\\nd\"\n") != std::string::npos);
    CK(r.out.find("flag [bool] = true\n") != std::string::npos);
    CK(r.out.find("expert_used_count") < r.out.find("tok.scores") && r.out.find("tok.scores") < r.out.find("note"));
    // get
    r = call({"x", good, "get", "qwen3moe.expert_used_count"}); CK(r.rc == 0 && r.out == "2\n");
    r = call({"x", good, "get", "nope"}); CK(r.rc == 1 && r.out.empty() && !r.err.empty());
    // tensors: Q2_0 is type 42 with 64-element/18-byte blocks, offsets are relative to the data section
    r = call({"x", good, "tensors"});
    CK(r.rc == 0);
    CK(r.out == "w.f32 F32 [4] 0 16\nw.q2 Q2_0 [64, 2] 32 36\n");
    // check
    r = call({"x", good, "check", "qwen3moe"}); CK(r.rc == 0 && r.out == "ok\n");
    r = call({"x", good, "check", "llama"}); CK(r.rc == 1 && r.out.empty());
    const std::string bad_experts = write_file("e8.gguf", build("qwen3moe", 8, 2));
    r = call({"x", bad_experts, "check", "qwen3moe"}); CK(r.rc == 1 && r.err.find("expert_count mismatch") != std::string::npos);
    const std::string bad_used = write_file("u5.gguf", build("qwen3moe", 4, 5));
    r = call({"x", bad_used, "check", "qwen3moe"}); CK(r.rc == 1 && r.err.find("expert_used_count") != std::string::npos);
    const std::string llama = write_file("llama.gguf", build("llama", 0, 0));
    r = call({"x", llama, "check", "llama"}); CK(r.rc == 0);   // the qwen guard applies only to qwen*
    // malformed inputs exit 1, usage errors exit 2
    r = call({"x", "/nonexistent/k28.gguf", "info"}); CK(r.rc == 1 && r.err.find("cannot open") != std::string::npos);
    Bytes magic = build("qwen3moe", 4, 2); magic[0] ^= 1;
    r = call({"x", write_file("magic.gguf", magic), "info"}); CK(r.rc == 1 && r.err.find("not a valid GGUF") != std::string::npos);
    Bytes cut = build("qwen3moe", 4, 2); cut.resize(cut.size() - 1);
    r = call({"x", write_file("cut.gguf", cut), "tensors"}); CK(r.rc == 1);   // last tensor no longer fits
    r = call({"x", write_file("empty.gguf", Bytes()), "info"}); CK(r.rc == 1);
    r = call({"x"}); CK(r.rc == 2);
    r = call({"x", good}); CK(r.rc == 2);
    r = call({"x", good, "bogus"}); CK(r.rc == 2);
    r = call({"x", good, "get"}); CK(r.rc == 2);
    r = call({"x", good, "info", "extra"}); CK(r.rc == 2);
    r = call({"x", "/tmp", "info"}); CK(r.rc == 1);   // a directory is not a file
    std::printf("checks: %ld, failures: %ld\n", g_checks, g_fail);
    if (g_fail) return 1;
    std::printf("UNIT 28 TESTS PASSED\n");
    return 0;
}
