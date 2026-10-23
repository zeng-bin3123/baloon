#pragma once
// strata/artifact/gguf_reader.hpp — Minimal GGUF tensor reader
// Target: Iluvatar BI-V100 (ivcore10)
//
// Reads tensor metadata and raw bytes from a GGUF file.
// Used by native_head and native_dense_replace to load weights
// in their original GGUF quantization format.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata {
namespace artifact {

// GGUF tensor entry (metadata only)
struct GgufTensor {
    std::string name;
    int type = -1;           // ggml type id
    int64_t ne[4] = {};      // dimensions
    int n_dims = 0;
    uint64_t offset = 0;     // byte offset into data section
    uint64_t nbytes = 0;     // total bytes for this tensor
};

// Minimal GGUF reader — reads header + tensor metadata, provides
// random-access to tensor data by name.
class GgufReader {
public:
    explicit GgufReader(const std::string& path);
    ~GgufReader();

    bool ok() const { return ok_; }
    const std::string& error() const { return err_; }

    // Find a tensor by name. Returns nullptr if not found.
    const GgufTensor* find(const std::string& name) const;

    // Get total bytes for a named tensor.
    uint64_t tensor_bytes(const std::string& name) const;

    // Get the ggml type id for a named tensor.
    int tensor_type(const std::string& name) const;

    // Read tensor data into a caller-supplied buffer.
    // Returns true on success, false on I/O error.
    bool read_tensor(const std::string& name, void* dst, uint64_t max_bytes) const;

    // Number of tensors in the file.
    size_t tensor_count() const { return tensors_.size(); }

    // Iterate all tensors.
    const std::vector<GgufTensor>& tensors() const { return tensors_; }

private:
    bool parse_header();
    bool read_string(std::string& out);
    bool skip_value(uint32_t vtype);

    std::string path_;
    std::FILE* fp_ = nullptr;
    bool ok_ = false;
    std::string err_;
    uint64_t data_offset_ = 0;  // byte offset where tensor data begins

    std::vector<GgufTensor> tensors_;
    std::unordered_map<std::string, size_t> name_to_idx_;
};

}  // namespace artifact
}  // namespace strata

