// load_main.cpp - loads a weight directory (index.txt + weights.bin + shapes.txt), verifies the per-layer layout,
// uploads every dense BF16 matrix to a device and verifies the resident bytes against the host copy.
// usage: load_main <dir> [--device D] [--geometry layers,embd,heads,head_dim,kv_heads]
//        load_main --selftest [--device D]     (generates a small model in a temp dir first)
// shapes.txt: one "name rows cols" per line (cols 0 = vector). exit 0 ok, 1 failure, 2 usage.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "baloon/core_weights.hpp"
#include "baloon/rt_device.hpp"
#include "baloon/rt_kernels.hpp"
#include "baloon/rt_layout.hpp"
#include "baloon/rt_native.hpp"
#include "baloon/rt_pinned.hpp"
#include "baloon/rt_testdata.hpp"

using namespace baloon;
using namespace baloon::rt;

static bool read_file(const std::string& p, std::string* out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::stringstream ss; ss << f.rdbuf(); *out = ss.str();
    return true;
}

static bool parse_shapes(const std::string& text, LayoutIndex* idx, ShapeMap* dense) {
    std::istringstream in(text); std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line); std::string name; unsigned long long r, c;
        if (!(ls >> name >> r >> c)) return false;
        (*idx)[name] = TensorShape{r, c};
        if (c != 0) (*dense)[name] = {r, c};
    }
    return true;
}

// Builds a tiny model whose layout satisfies check_all for `g`. Every tensor is raw BF16 (index kind 4).
static bool write_selftest_model(const std::string& dir, const LayoutGeometry& g) {
    XorRng rng(11);
    std::string index, shapes; std::vector<uint8_t> blob;
    auto add = [&](const std::string& name, uint64_t rows, uint64_t cols) {
        const uint64_t n = rows * (cols ? cols : 1);
        std::vector<uint16_t> v = make_bf16(rng, n);
        index += name + " 4 " + std::to_string(blob.size()) + " " + std::to_string(n * 2) + " " + std::to_string(n * 2) + " 0 0 0 0\n";
        shapes += name + " " + std::to_string(rows) + " " + std::to_string(cols) + "\n";
        blob.insert(blob.end(), (const uint8_t*)v.data(), (const uint8_t*)v.data() + n * 2);
    };
    for (int l = 0; l < g.n_layers; ++l) {
        LayerView v; v.layer = l;
        add(v.name("attn_norm"), g.n_embd, 0); add(v.name("ffn_norm"), g.n_embd, 0);
        if (g.is_qsa(l)) {
            add(v.name("attn_q"), g.n_head * g.head_dim, g.n_embd);
            add(v.name("attn_k"), g.n_head_kv * g.head_dim, g.n_embd);
            add(v.name("attn_v"), g.n_head_kv * g.head_dim, g.n_embd);
            add(v.name("attn_output"), g.n_embd, g.n_head * g.head_dim);
        } else { add(v.name("ssm_in"), 2 * g.n_embd, g.n_embd); add(v.name("ssm_out"), g.n_embd, 2 * g.n_embd); }
    }
    std::ofstream(dir + "/index.txt") << index;
    std::ofstream(dir + "/shapes.txt") << shapes;
    std::ofstream bin(dir + "/weights.bin", std::ios::binary); bin.write((const char*)blob.data(), (std::streamsize)blob.size());
    return (bool)bin;
}

int main(int argc, char** argv) {
    std::string dir; int dev = 0; bool self = false;
    LayoutGeometry g;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--selftest")) self = true;
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--geometry") && i + 1 < argc) {
            unsigned long long a, b, c, d, e;
            if (std::sscanf(argv[++i], "%llu,%llu,%llu,%llu,%llu", &a, &b, &c, &d, &e) != 5) { std::fprintf(stderr, "bad --geometry\n"); return 2; }
            g.n_layers = (int)a; g.n_embd = b; g.n_head = c; g.head_dim = d; g.n_head_kv = e;
        } else if (argv[i][0] != '-' && dir.empty()) dir = argv[i];
        else { std::fprintf(stderr, "usage: %s <dir> [--device D] [--geometry L,E,H,D,K] | --selftest\n", argv[0]); return 2; }
    }
    if (self) {
        g.n_layers = 8; g.n_embd = 16; g.n_head = 2; g.head_dim = 8; g.n_head_kv = 1;
        char tmpl[] = "/tmp/rt_load_selftest_XXXXXX";
        if (!mkdtemp(tmpl)) { std::fprintf(stderr, "mkdtemp failed\n"); return 1; }
        dir = tmpl;
        if (!write_selftest_model(dir, g)) { std::fprintf(stderr, "cannot write selftest model\n"); return 1; }
    }
    if (dir.empty()) { std::fprintf(stderr, "usage: %s <dir> | --selftest\n", argv[0]); return 2; }
    if (dev < 0 || dev >= device_count()) { std::fprintf(stderr, "bad device %d\n", dev); return 1; }

    std::string index_text, shape_text, err;
    if (!read_file(dir + "/index.txt", &index_text) || !read_file(dir + "/shapes.txt", &shape_text)) { std::fprintf(stderr, "cannot read index.txt / shapes.txt in %s\n", dir.c_str()); return 1; }
    LayoutIndex lidx; ShapeMap dense;
    if (!parse_shapes(shape_text, &lidx, &dense)) { std::fprintf(stderr, "bad shapes.txt\n"); return 1; }
    if (!check_all(lidx, g, &err)) { std::fprintf(stderr, "layout check failed: %s\n", err.c_str()); return 1; }

    core::WeightTable table;
    if (!table.place(index_text, {}, &err)) { std::fprintf(stderr, "place: %s\n", err.c_str()); return 1; }
    PinArena host;
    if (!host.reserve(table.pool_bytes() ? table.pool_bytes() : 1, 2)) { std::fprintf(stderr, "cannot reserve host arena\n"); return 1; }
    core::WeightTable loaded;
    if (!loaded.load(dir, host.data(), host.bytes(), {}, &err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    DenseBank bank;
    if (!bank.load(loaded, host.data(), dense, dev, &err)) { std::fprintf(stderr, "upload: %s\n", err.c_str()); return 1; }

    // every served tensor must read back identical to the host bytes
    size_t served = 0;
    for (const std::string& n : bank.served_names()) {
        DenseView v; const core::WIndexRow* row = loaded.find(n);
        if (!bank.find(n, &v) || !row) { std::fprintf(stderr, "served name %s not found\n", n.c_str()); return 1; }
        std::vector<uint8_t> back(v.rows * v.cols * 2);
        if (cudaMemcpy(back.data(), v.dev, back.size(), cudaMemcpyDeviceToHost) != cudaSuccess) { std::fprintf(stderr, "readback failed\n"); return 1; }
        if (fnv1a64(back.data(), back.size()) != fnv1a64(host.data() + row->arena_off, back.size())) { std::fprintf(stderr, "resident bytes of %s differ\n", n.c_str()); return 1; }
        ++served;
    }
    size_t want = 0; for (const auto& kv : dense) { const core::WIndexRow* r = loaded.find(kv.first); if (r && eligible(*r)) ++want; }
    if (served != want) { std::fprintf(stderr, "served %zu tensors, expected %zu\n", served, want); return 1; }
    std::printf("loaded %zu rows, %llu pool bytes, %zu dense tensors resident (%llu bytes) on device %d\n", loaded.rows().size(),
                (unsigned long long)loaded.pool_bytes(), served, (unsigned long long)bank.resident_bytes(), dev);
    std::printf("LOAD OK\n");
    return 0;
}
