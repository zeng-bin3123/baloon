// gguf_reader.cpp - command line inspector for GGUF v2/v3 files, built on baloon/gguf_reader.hpp.
//   gguf_reader <file> info            version, alignment, data offset, counts, architecture
//   gguf_reader <file> meta            every metadata key (file order) with type and value
//   gguf_reader <file> get <key>       one metadata value (exit 1 when absent)
//   gguf_reader <file> tensors         tensor directory: name type shape offset bytes
//   gguf_reader <file> check <arch>    architecture check; for "qwen4exp" also the expert-count guard
// Exit codes: 0 ok, 1 file unreadable / malformed / key absent / check failed, 2 usage error.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "baloon/gguf_reader.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace baloon {
namespace gguf_cli {

using namespace gguf_reader;

// Read-only view of a whole file: mmap on POSIX, buffered read elsewhere.
struct FileImage {
    const uint8_t* data = nullptr; uint64_t size = 0;
    std::vector<uint8_t> owned; void* map = nullptr; size_t map_len = 0;
    FileImage() = default;
    FileImage(const FileImage&) = delete; FileImage& operator=(const FileImage&) = delete;
    ~FileImage() {
#ifndef _WIN32
        if (map) munmap(map, map_len);
#endif
    }
    bool open(const char* path) {
#ifndef _WIN32
        const int fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) { ::close(fd); return false; }
        size = (uint64_t)st.st_size;
        if (size == 0) { ::close(fd); data = owned.data(); return true; }
        map = mmap(nullptr, (size_t)size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (map == MAP_FAILED) { map = nullptr; return false; }
        map_len = (size_t)size; data = (const uint8_t*)map; return true;
#else
        std::FILE* f = std::fopen(path, "rb");
        if (!f) return false;
        uint8_t buf[65536]; size_t got;
        while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) owned.insert(owned.end(), buf, buf + got);
        std::fclose(f); data = owned.data(); size = owned.size(); return true;
#endif
    }
};

inline const char* meta_type_name(MetaType t) {
    switch (t) {
        case MetaType::U8: return "u8"; case MetaType::I8: return "i8"; case MetaType::U16: return "u16";
        case MetaType::I16: return "i16"; case MetaType::U32: return "u32"; case MetaType::I32: return "i32";
        case MetaType::F32: return "f32"; case MetaType::Bool: return "bool"; case MetaType::String: return "string";
        case MetaType::Array: return "array"; case MetaType::U64: return "u64"; case MetaType::I64: return "i64";
        case MetaType::F64: return "f64";
    }
    return "?";
}

// Render a value on one line. Strings are escaped; arrays show the first 8 elements and the total count.
inline std::string render(const MetaValue& v) {
    char b[64];
    switch (v.type) {
        case MetaType::U8: case MetaType::U16: case MetaType::U32: case MetaType::U64:
            std::snprintf(b, sizeof b, "%" PRIu64, v.u); return b;
        case MetaType::I8: case MetaType::I16: case MetaType::I32: case MetaType::I64:
            std::snprintf(b, sizeof b, "%" PRId64, v.i); return b;
        case MetaType::F32: case MetaType::F64: std::snprintf(b, sizeof b, "%.9g", v.f); return b;
        case MetaType::Bool: return v.b ? "true" : "false";
        case MetaType::String: {
            std::string o = "\"";
            for (unsigned char ch : v.s) {
                if (ch == '"' || ch == '\\') { o += '\\'; o += (char)ch; }
                else if (ch == '\n') o += "\\n";
                else if (ch < 0x20 || ch == 0x7f) { std::snprintf(b, sizeof b, "\\x%02x", ch); o += b; }
                else o += (char)ch;
            }
            return o + "\"";
        }
        case MetaType::Array: {
            std::string o = "[";
            const size_t shown = v.arr.size() < 8 ? v.arr.size() : 8;
            for (size_t i = 0; i < shown; ++i) { if (i) o += ", "; o += render(v.arr[i]); }
            if (v.count > shown) o += ", ...";
            std::snprintf(b, sizeof b, "] (%llu elements of %s)", (unsigned long long)v.count, meta_type_name(v.elem));
            return o + b;
        }
    }
    return "?";
}

inline std::string shape_str(const TensorInfo& t) {
    std::string s = "[";
    for (size_t i = 0; i < t.shape.size(); ++i) { if (i) s += ", "; s += std::to_string(t.shape[i]); }
    return s + "]";
}

inline int run(int argc, char** argv, std::FILE* out, std::FILE* err) {
    const char* prog = argc > 0 ? argv[0] : "gguf_reader";
    auto usage = [&]() {
        std::fprintf(err, "usage: %s <file.gguf> info|meta|tensors | get <key> | check <arch>\n", prog);
        return 2;
    };
    if (argc < 3) return usage();
    const std::string cmd = argv[2];
    const bool takes_arg = cmd == "get" || cmd == "check";
    if (!(takes_arg || cmd == "info" || cmd == "meta" || cmd == "tensors")) return usage();
    if (argc != (takes_arg ? 4 : 3)) return usage();

    FileImage img;
    if (!img.open(argv[1])) { std::fprintf(err, "cannot open %s\n", argv[1]); return 1; }
    auto f = GgufFile::parse(img.data, img.size);
    if (!f) { std::fprintf(err, "%s: not a valid GGUF v2/v3 file\n", argv[1]); return 1; }

    if (cmd == "info") {
        std::fprintf(out, "version: %u\nalignment: %u\ndata_start: %" PRIu64 "\nmetadata: %zu\ntensors: %zu\n",
                     f->version, f->alignment, f->data_start, f->meta_order.size(), f->tensors.size());
        auto a = f->get_string("general.architecture");
        std::fprintf(out, "architecture: %s\n", a ? a->c_str() : "(none)");
        uint64_t total = 0; for (auto& t : f->tensors) total += t.nbytes();
        std::fprintf(out, "tensor_bytes: %" PRIu64 "\n", total);
    } else if (cmd == "meta") {
        for (auto& k : f->meta_order)
            std::fprintf(out, "%s [%s] = %s\n", k.c_str(), meta_type_name(f->meta.at(k).type), render(f->meta.at(k)).c_str());
    } else if (cmd == "get") {
        auto m = f->find_meta(argv[3]);
        if (!m) { std::fprintf(err, "key not found: %s\n", argv[3]); return 1; }
        std::fprintf(out, "%s\n", render(*m).c_str());
    } else if (cmd == "tensors") {
        for (auto& t : f->tensors)
            std::fprintf(out, "%s %s %s %" PRIu64 " %" PRIu64 "\n", t.name.c_str(), ggml_type_name(t.type),
                         shape_str(t).c_str(), t.offset, t.nbytes());
    } else {  // check
        const std::string want = argv[3];
        if (!check_architecture(*f, want)) { std::fprintf(err, "architecture mismatch (want %s)\n", want.c_str()); return 1; }
        if (want == kGuardArch) {
            std::string why;
            if (!Qwen4ExpGuard{}.check(*f, &why)) { std::fprintf(err, "guard failed: %s\n", why.c_str()); return 1; }
        }
        std::fprintf(out, "ok\n");
    }
    return 0;
}

}  // namespace gguf_cli
}  // namespace baloon

#ifndef BALOON_GGUF_CLI_MAIN_DISABLED
#define BALOON_GGUF_CLI_MAIN_DISABLED 0
#endif
#if !BALOON_GGUF_CLI_MAIN_DISABLED
int main(int argc, char** argv) { return baloon::gguf_cli::run(argc, argv, stdout, stderr); }
#endif
