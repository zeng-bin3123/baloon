#pragma once
// strata/core/native_head.hpp — IQ-format head weight loader and embedding
// Target: Iluvatar BI-V100 (ivcore10)
//
// NativeHead: loads the language model head (final vocabulary projection)
// in native IQ quantization from a GGUF file. Avoids the pack's canonical
// format for the head tensor.
//
// NativeEmbed: loads the token embedding table from a GGUF file and
// provides single-token and batched gather operations.

#include <cstdint>
#include <string>

namespace strata {

// Language model head (output projection) in native IQ format.
// Loads from a GGUF file, allocates GPU memory for both the weight
// data and a scratch buffer for the IQ kernel.
class NativeHead {
public:
    NativeHead() = default;
    ~NativeHead();

    // Non-copyable
    NativeHead(const NativeHead&) = delete;
    NativeHead& operator=(const NativeHead&) = delete;

    // Load the head weight from a GGUF file.
    // tensor_name: the name of the output.weight tensor in the GGUF.
    bool load(const std::string& gguf_path, const std::string& tensor_name,
              std::string& err);

    // Run the head projection: logits = head(mixed)
    // mixed: [n_embd] f32 input
    // logits: [n_vocab] f32 output
    // stream: CUDA stream (nullable)
    bool run(const float* mixed, float* logits, void* stream,
             std::string& err) const;

    uint64_t weight_bytes() const { return weight_bytes_; }
    bool loaded() const { return weights_ != nullptr; }
    const void* weights() const { return weights_; }
    int type() const { return type_; }
    int64_t n_vocab() const { return n_vocab_; }
    int64_t n_embd() const { return n_embd_; }

    // Number of tensors loaded (always 1 if loaded).
    size_t tensor_count() const { return loaded() ? 1 : 0; }

private:
    void* weights_ = nullptr;
    void* scratch_ = nullptr;
    uint64_t weight_bytes_ = 0;
    uint64_t scratch_bytes_ = 0;
    int64_t n_vocab_ = 0;
    int64_t n_embd_ = 0;
    int type_ = -1;
};

// Token embedding table in native format.
class NativeEmbed {
public:
    NativeEmbed() = default;
    ~NativeEmbed();

    NativeEmbed(const NativeEmbed&) = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;

    // Load from GGUF.
    bool load(const std::string& gguf_path, int64_t n_embd, int64_t n_vocab,
              std::string& err);

    // Gather embedding rows for multiple tokens → device output.
    void gather_dev(const int32_t* tokens, int64_t n_tok, float* out,
                    void* stream) const;

    // Gather a single token's embedding → host output.
    void gather_one(int64_t token, float* out, void* stream) const;

    uint64_t bytes() const { return bytes_; }

private:
    void* host_ = nullptr;        // host copy of embedding table
    const void* dev_ = nullptr;   // device copy
    uint64_t bytes_ = 0;
    size_t row_bytes_ = 0;
    int64_t n_embd_ = 0;
    int64_t n_vocab_ = 0;
    int type_ = -1;
};

}  // namespace strata

