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

NativeDense::~NativeDense() { clear(); }

void NativeDense::clear() {
    if (!e_.empty()) { DevScope sc(ord_); e_.clear(); }
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
    for (const auto& p : pend) {
        void* d = nullptr;
        const uint64_t nb = p.rows * p.cols * 2;
        if (cudaMalloc(&d, (size_t)nb) != cudaSuccess) { cudaGetLastError(); clear(); return set_err(err, "device allocation failed for " + p.name); }
        Entry en; en.buf.reset(d); en.rows = p.rows; en.cols = p.cols;
        if (cudaMemcpy(d, p.host, (size_t)nb, cudaMemcpyHostToDevice) != cudaSuccess) { cudaGetLastError(); clear(); return set_err(err, "upload failed for " + p.name); }
        e_[p.name] = std::move(en); bytes_ += nb;
    }
    return true;
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
