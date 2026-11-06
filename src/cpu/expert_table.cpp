// expert_table.cpp - native expert table validation (see include/baloon/expert_table.hpp).
#include "baloon/expert_table.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace baloon {
namespace etable {

bool contiguous(uint64_t n_expert, const std::vector<LayerEntry>& layers, uint64_t* total) {
    uint64_t cur = 0;
    for (const LayerEntry& l : layers) {
        if (l.offset != cur) return false;      // gap or overlap
        cur += l.bytes * n_expert;
    }
    if (total) *total = cur;
    return true;
}

uint64_t max_blob(const std::vector<LayerEntry>& layers) {
    uint64_t m = 0;
    for (const LayerEntry& l : layers) m = std::max(m, l.bytes);
    return m;
}

ExpertTable default_table(uint64_t n_layers, uint64_t n_expert) {
    ExpertTable t;
    t.n_layers = n_layers; t.n_expert = n_expert; t.by_default = true;
    for (uint64_t l = 0; l < n_layers; ++l) {
        LayerEntry e;
        e.offset = l * n_expert * kern::q2geo::BLOB; e.bytes = kern::q2geo::BLOB; e.gu_type = 42; e.d_type = 42;
        t.layers.push_back(e);
    }
    t.total = default_total(n_layers, n_expert, kern::q2geo::BLOB);
    t.max_blob = n_layers ? kern::q2geo::BLOB : 0;
    return t;
}

static bool next_data_line(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        const size_t h = line.find('#');
        if (h != std::string::npos) line.erase(h);
        if (line.find_first_not_of(" \t\r") != std::string::npos) return true;
    }
    return false;
}

bool parse_expert_table(const std::string& text, ExpertTable& out, std::string& err) {
    out = ExpertTable{};
    std::istringstream in(text);
    std::string line;
    if (!next_data_line(in, line)) { err = "empty table"; return false; }
    ExpertTable t;
    {
        std::istringstream h(line);
        std::string magic, ver;
        long long nl = -1, ne = -1;
        if (!(h >> magic >> ver >> nl >> ne) || magic != "native_experts" || (ver != "v1" && ver != "v2") || nl <= 0 || ne <= 0) {
            err = "bad header"; return false;
        }
        t.v2 = ver == "v2"; t.n_layers = (uint64_t)nl; t.n_expert = (uint64_t)ne;
    }
    for (uint64_t l = 0; l < t.n_layers; ++l) {
        if (!next_data_line(in, line)) { err = "missing layer line " + std::to_string(l); return false; }
        std::istringstream s(line);
        LayerEntry e;
        long long off = -1, by = -1, gt = 0, dt = 0;
        if (!(s >> off >> by >> gt >> dt) || off < 0 || by <= 0) { err = "bad layer line " + std::to_string(l); return false; }
        e.offset = (uint64_t)off; e.bytes = (uint64_t)by; e.gu_type = (int)gt; e.d_type = (int)dt;
        if (t.v2) {
            for (int k = 0; k < 3; ++k) {
                long long g = -1;
                if (!(s >> g) || g < 0) { err = "bad gguf offset on layer " + std::to_string(l); return false; }
                t.gguf.resize(std::max<size_t>(t.gguf.size(), gguf_slot(l, k) + 1));
                t.gguf[gguf_slot(l, k)] = (uint64_t)g;
            }
        }
        t.layers.push_back(e);
    }
    if (next_data_line(in, line)) { err = "trailing data after the last layer"; return false; }
    if (!contiguous(t.n_expert, t.layers, &t.total)) { err = "layers do not tile the pack (gap or overlap)"; return false; }
    t.max_blob = max_blob(t.layers);
    out = t;
    return true;
}

bool load_expert_table(const std::string& path, uint64_t n_layers, uint64_t n_expert, ExpertTable& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { out = default_table(n_layers, n_expert); return true; }   // no native_experts.txt: the fixed-size layout
    std::stringstream ss; ss << f.rdbuf();
    if (!parse_expert_table(ss.str(), out, err)) return false;
    if (out.n_layers != n_layers || out.n_expert != n_expert) { err = "table geometry does not match the model"; out = ExpertTable{}; return false; }
    return true;
}

}  // namespace etable
}  // namespace baloon
