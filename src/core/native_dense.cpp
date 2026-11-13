// native_dense.cpp - NativeDense: loads the dense BF16 rows of a WeightTable onto one device (reference layout).
#include "baloon/u11_native_dense.hpp"

namespace baloon {
namespace u11 {

namespace {
bool set_err(std::string* err, const std::string& m) { if (err) *err = m; return false; }
struct DevScope {   // selects a device and restores the previous one on exit
    int prev = 0; bool ok = false;
    explicit DevScope(int ord) { cudaGetDevice(&prev); ok = cudaSetDevice(ord) == cudaSuccess; if (!ok) cudaGetLastError(); }
    ~DevScope() { cudaSetDevice(prev); }
};
}  // namespace

bool eligible(const core::WIndexRow& r) { return r.placed && r.engine_kind == 1 && r.dst_bytes > 0 && r.dst_bytes % 2 == 0; }

bool eligible(const std::string& name, bool include_ple_key) {
    if (name.compare(0, 4, "blk.") != 0) return false;
    if (include_ple_key && name == "blk.1.ple_key.weight") return true;
    for (const char* s : kNativeSuffixes) {
        const size_t n = std::char_traits<char>::length(s);
        if (name.size() >= n && name.compare(name.size() - n, n, s) == 0) return true;
    }
    return false;
}

NativeDense::~NativeDense() { clear(); }

void NativeDense::clear() {
    if (!e_.empty() || !o_.empty()) { DevScope sc(ord_); e_.clear(); o_.clear(); }
    scratch_in_ = 0;
    bytes_ = 0;
}

bool NativeDense::load(const core::WeightTable& t, const uint8_t* host_arena, const ShapeMap& shapes, int ordinal, std::string* err) {
    clear();
    if (!host_arena) return set_err(err, "no host arena");
    std::vector<Pending> pend;
    for (const auto& r : t.rows()) {
        if (!eligible(r)) continue;
        auto it = shapes.find(r.name);
        if (it == shapes.end()) continue;
        const uint64_t rows = it->second.first, cols = it->second.second;
        if (rows == 0 || cols == 0 || rows > UINT64_MAX / cols / 2 || rows * cols * 2 != r.dst_bytes)
            return set_err(err, "shape of " + r.name + " does not match its byte count");
        Pending p; p.name = r.name; p.host = reinterpret_cast<const uint16_t*>(host_arena + r.arena_off); p.rows = rows; p.cols = cols;
        pend.push_back(std::move(p));
    }
    DevScope sc(ordinal);
    if (!sc.ok) return set_err(err, "cannot select device");
    ord_ = ordinal;
    std::map<std::string, Entry> staged;    // every upload finishes before anything is published
    uint64_t staged_bytes = 0;
    for (const auto& p : pend) {
        void* d = nullptr;
        const uint64_t nb = p.rows * p.cols * 2;
        if (cudaMalloc(&d, (size_t)nb) != cudaSuccess) { cudaGetLastError(); return set_err(err, "device allocation failed for " + p.name); }
        Entry en; en.buf.reset(d); en.rows = p.rows; en.cols = p.cols;
        if (cudaMemcpy(d, p.host, (size_t)nb, cudaMemcpyHostToDevice) != cudaSuccess) { cudaGetLastError(); return set_err(err, "upload failed for " + p.name); }
        staged[p.name] = std::move(en); staged_bytes += nb;
    }
    e_ = std::move(staged); bytes_ = staged_bytes;   // publish the complete batch
    return true;
}

bool NativeDense::attach(const uint8_t* data, uint64_t payload, const std::vector<GgufEntry>& dir, bool include_ple_key,
                         int ordinal, OverrideTable& tbl, std::string* err) {
    if (!data) return set_err(err, "no data section");
    if (!check_directory(dir, payload, err)) return false;      // uploaded or not, every entry is checked
    std::vector<GgufEntry> want;
    for (const auto& e : dir) if (eligible(e.name, include_ple_key)) want.push_back(e);
    std::vector<uint64_t> widths;
    for (const auto& e : want) widths.push_back(e.ne[0]);
    DevScope sc(ordinal);
    if (!sc.ok) return set_err(err, "cannot select device");
    struct Up { std::string name; Entry en; OverrideRef ref; uint64_t bytes = 0; };
    std::vector<Up> batch;
    const bool ok = check_all(want, batch, [&](const GgufEntry& e, Up& u, std::string* er) {
        uint64_t nb = 0;
        if (!entry_bytes(e, &nb, er)) return false;
        void* d = nullptr;
        if (nb && cudaMalloc(&d, (size_t)nb) != cudaSuccess) { cudaGetLastError(); return set_err(er, "device allocation failed for " + e.name); }
        u.en.buf.reset(d);
        if (nb && cudaMemcpy(d, data + e.offset, (size_t)nb, cudaMemcpyHostToDevice) != cudaSuccess) { cudaGetLastError(); return set_err(er, "upload failed for " + e.name); }
        uint64_t rows = 1; for (size_t i = 1; i < e.ne.size(); ++i) rows *= e.ne[i];   // guarded by entry_bytes
        u.name = e.name; u.en.rows = rows; u.en.cols = e.ne[0]; u.bytes = nb;
        u.ref = OverrideRef{d, rows, e.ne[0], e.block_elems, e.block_bytes};
        return true;
    }, err);
    publish(tbl, ok, batch, [&](OverrideTable& t, std::vector<Up>& b) {
        ord_ = ordinal;
        for (auto& u : b) { t[u.name] = u.ref; bytes_ += u.bytes; o_[u.name] = std::move(u.en); }
        scratch_in_ = std::max(scratch_in_, max_in(widths));
    });
    return ok;   // on failure `batch` frees every staged upload and `tbl` is unchanged
}

std::vector<std::string> NativeDense::served_names() const {
    std::vector<std::string> v;
    for (const auto& kv : e_) v.push_back(kv.first);   // std::map iterates in sorted order
    return v;
}

bool NativeDense::find(const std::string& name, const uint16_t** dev, uint64_t* rows, uint64_t* cols) const {
    auto it = e_.find(name);
    if (it == e_.end()) return false;
    if (dev) *dev = static_cast<const uint16_t*>(it->second.buf.get());
    if (rows) *rows = it->second.rows;
    if (cols) *cols = it->second.cols;
    return true;
}

}  // namespace u11
}  // namespace baloon
