// strata/artifact/gguf_reader.cpp — Minimal GGUF tensor reader
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>

namespace strata {
namespace artifact {

// GGUF magic and version
static constexpr uint32_t GGUF_MAGIC = 0x46475547u;  // "GGUF"

// GGUF value types for skipping metadata
enum GgufValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// Bytes-per-element for known ggml types (indices 0..31)
static const uint64_t GGML_TYPE_SIZE[] = {
    4,    // 0: F32
    2,    // 1: F16
    18,   // 2: Q4_0 (block of 32 elems = 2+16 bytes)
    20,   // 3: Q4_1
    18,   // 4: Q4_2 (deprecated)
    20,   // 5: Q4_3 (deprecated)
    18,   // 6: Q5_0
    22,   // 7: Q5_1
    34,   // 8: Q8_0
    40,   // 9: Q8_1
    0,    // 10: Q2_K
    0,    // 11: Q3_K
    0,    // 12: Q4_K
    0,    // 13: Q5_K
    0,    // 14: Q6_K
    0,    // 15: Q8_K
    2,    // 16: IQ2_XXS
    2,    // 17: IQ2_XS
    2,    // 18: IQ3_XXS
    2,    // 19: IQ1_S
    0,    // 20: IQ4_NL
    0,    // 21: IQ3_S
    2,    // 22: IQ2_S
    0,    // 23: IQ4_XS
    1,    // 24: I8
    2,    // 25: I16
    4,    // 26: I32
    8,    // 27: I64
    8,    // 28: F64
    0,    // 29: IQ1_M
    1,    // 30: BF16
    0,    // 31: Q2_0
};

static const uint64_t GGML_BLOCK_SIZE[] = {
    1,    // 0: F32
    1,    // 1: F16
    32,   // 2: Q4_0
    32,   // 3: Q4_1
    32,   // 4: Q4_2
    32,   // 5: Q4_3
    32,   // 6: Q5_0
    32,   // 7: Q5_1
    32,   // 8: Q8_0
    32,   // 9: Q8_1
    256,  // 10: Q2_K
    256,  // 11: Q3_K
    256,  // 12: Q4_K
    256,  // 13: Q5_K
    256,  // 14: Q6_K
    256,  // 15: Q8_K
    256,  // 16: IQ2_XXS
    256,  // 17: IQ2_XS
    256,  // 18: IQ3_XXS
    256,  // 19: IQ1_S
    32,   // 20: IQ4_NL
    256,  // 21: IQ3_S
    256,  // 22: IQ2_S
    256,  // 23: IQ4_XS
    1,    // 24: I8
    1,    // 25: I16
    1,    // 26: I32
    1,    // 27: I64
    1,    // 28: F64
    256,  // 29: IQ1_M
    1,    // 30: BF16
    4,    // 31: Q2_0
};

GgufReader::GgufReader(const std::string& path) : path_(path) {
    fp_ = std::fopen(path.c_str(), "rb");
    if (!fp_) {
        err_ = "cannot open " + path;
        return;
    }
    ok_ = parse_header();
    if (!ok_ && fp_) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
}

GgufReader::~GgufReader() {
    if (fp_) std::fclose(fp_);
}

bool GgufReader::read_string(std::string& out) {
    uint64_t len = 0;
    if (std::fread(&len, 8, 1, fp_) != 1) return false;
    if (len > 65536) { err_ = "string too long"; return false; }
    out.resize((size_t)len);
    if (len > 0 && std::fread(&out[0], 1, (size_t)len, fp_) != (size_t)len) return false;
    return true;
}

static size_t value_type_size(uint32_t vtype) {
    switch (vtype) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

bool GgufReader::skip_value(uint32_t vtype) {
    if (vtype == GGUF_TYPE_STRING) {
        std::string dummy;
        return read_string(dummy);
    }
    if (vtype == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        if (std::fread(&elem_type, 4, 1, fp_) != 1) return false;
        if (std::fread(&count, 8, 1, fp_) != 1) return false;
        for (uint64_t i = 0; i < count; ++i) {
            if (!skip_value(elem_type)) return false;
        }
        return true;
    }
    size_t sz = value_type_size(vtype);
    if (sz == 0) { err_ = "unknown gguf value type"; return false; }
    return std::fseek(fp_, (long)sz, SEEK_CUR) == 0;
}

bool GgufReader::parse_header() {
    // Read magic
    uint32_t magic = 0;
    if (std::fread(&magic, 4, 1, fp_) != 1) { err_ = "cannot read magic"; return false; }
    if (magic != GGUF_MAGIC) { err_ = "not a GGUF file"; return false; }

    // Version
    uint32_t version = 0;
    if (std::fread(&version, 4, 1, fp_) != 1) { err_ = "cannot read version"; return false; }
    if (version < 2 || version > 3) { err_ = "unsupported GGUF version"; return false; }

    // Counts
    uint64_t n_tensors = 0, n_kv = 0;
    if (std::fread(&n_tensors, 8, 1, fp_) != 1) { err_ = "cannot read n_tensors"; return false; }
    if (std::fread(&n_kv, 8, 1, fp_) != 1) { err_ = "cannot read n_kv"; return false; }

    // Skip metadata KV pairs
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        if (!read_string(key)) { err_ = "cannot read kv key"; return false; }
        uint32_t vtype = 0;
        if (std::fread(&vtype, 4, 1, fp_) != 1) { err_ = "cannot read kv type"; return false; }
        if (!skip_value(vtype)) { err_ = "cannot skip kv value for key: " + key; return false; }
    }

    // Read tensor info
    tensors_.resize((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        auto& t = tensors_[i];
        if (!read_string(t.name)) { err_ = "cannot read tensor name"; return false; }
        uint32_t n_dims = 0;
        if (std::fread(&n_dims, 4, 1, fp_) != 1) { err_ = "cannot read n_dims"; return false; }
        t.n_dims = (int)n_dims;
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim = 0;
            if (std::fread(&dim, 8, 1, fp_) != 1) { err_ = "cannot read dim"; return false; }
            t.ne[d] = (int64_t)dim;
        }
        uint32_t type = 0;
        if (std::fread(&type, 4, 1, fp_) != 1) { err_ = "cannot read type"; return false; }
        t.type = (int)type;
        if (std::fread(&t.offset, 8, 1, fp_) != 1) { err_ = "cannot read offset"; return false; }

        // Compute nbytes
        int64_t n_elems = 1;
        for (int d = 0; d < t.n_dims; ++d) n_elems *= t.ne[d];
        if (type < 32 && GGML_BLOCK_SIZE[type] > 0) {
            uint64_t n_blocks = ((uint64_t)n_elems + GGML_BLOCK_SIZE[type] - 1) / GGML_BLOCK_SIZE[type];
            t.nbytes = n_blocks * GGML_TYPE_SIZE[type];
        } else {
            t.nbytes = (uint64_t)n_elems * 4;  // fallback: f32
        }

        name_to_idx_[t.name] = i;
    }

    // Data section starts at current position, aligned to 32 bytes
    long pos = std::ftell(fp_);
    data_offset_ = ((uint64_t)pos + 31) & ~31ull;

    return true;
}

const GgufTensor* GgufReader::find(const std::string& name) const {
    auto it = name_to_idx_.find(name);
    return it != name_to_idx_.end() ? &tensors_[it->second] : nullptr;
}

uint64_t GgufReader::tensor_bytes(const std::string& name) const {
    auto t = find(name);
    return t ? t->nbytes : 0;
}

int GgufReader::tensor_type(const std::string& name) const {
    auto t = find(name);
    return t ? t->type : -1;
}

bool GgufReader::read_tensor(const std::string& name, void* dst, uint64_t max_bytes) const {
    auto t = find(name);
    if (!t) return false;
    uint64_t to_read = std::min(t->nbytes, max_bytes);
    if (std::fseek(fp_, (long)(data_offset_ + t->offset), SEEK_SET) != 0) return false;
    return std::fread(dst, 1, (size_t)to_read, fp_) == (size_t)to_read;
}

}  // namespace artifact
}  // namespace strata

