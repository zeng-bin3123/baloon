// safetensors_loader.hpp — loader for the 8 tensor groups used by the ported kernels.
// Header-only. Provides sf_group_id() (classify a tensor name into 0-7 or -1) and
// sf_map_to_device() (mmap a tensor from a shard and cudaMemcpy H2D) for the 8 BF16
// tensor groups that pcd027bc02f and u12d8dd5102 kernels operate on.
// No external JSON library dependency — uses minimal inline parsing.
#pragma once
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <string>
#include <map>
#include <dirent.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

namespace baloon {
namespace sf {

// A tensor's location inside a safetensors shard file.
struct SfTensorInfo {
    uint64_t data_offset = 0;   // byte offset from start of data section
    uint64_t nbytes      = 0;   // total byte count
    uint64_t elem_count  = 0;   // number of elements
    char     dtype[8]    = {};  // e.g. "BF16"
};

// Returns the group id (0-7) for a named tensor group, or -1 if not one of the 8.
// The 8 groups correspond to the kernel units' BF16 weight tensors:
//   0: lm_head.weight                                   (pcd027bc02f bf16_gemv_warp_kernel)
//   1: model.language_model.layers.*.mlp.experts.down_proj  (u12d8dd5102 down_grouped_kernel)
//   2: model.language_model.layers.*.mlp.experts.gate_up_proj (u12d8dd5102 gu_grouped_kernel)
//   3: model.language_model.layers.*.self_attn.q_proj.weight  (pcd027bc02f bf16_gemv_warp_kernel)
//   4: model.language_model.norm.weight                 (pcd027bc02f rms_norm_weighted_kernel)
//   5: mtp.layers.*.mlp.experts.down_proj               (u12d8dd5102 down_grouped_kernel)
//   6: mtp.layers.*.mlp.experts.gate_up_proj            (u12d8dd5102 gu_grouped_kernel)
//   7: mtp.layers.*.self_attn.q_proj.weight             (pcd027bc02f bf16_gemv_warp_kernel)
inline int sf_group_id(const char* name) {
    if (!name) return -1;
    if (std::strcmp(name, "lm_head.weight") == 0)                        return 0;
    if (std::strstr(name, ".mlp.experts.down_proj")  && std::strstr(name, "model.language_model.layers.")) return 1;
    if (std::strstr(name, ".mlp.experts.gate_up_proj") && std::strstr(name, "model.language_model.layers.")) return 2;
    if (std::strstr(name, ".self_attn.q_proj.weight") && std::strstr(name, "model.language_model.layers.")) return 3;
    if (std::strcmp(name, "model.language_model.norm.weight") == 0)      return 4;
    if (std::strstr(name, "mtp.layers.") && std::strstr(name, ".mlp.experts.down_proj"))    return 5;
    if (std::strstr(name, "mtp.layers.") && std::strstr(name, ".mlp.experts.gate_up_proj")) return 6;
    if (std::strstr(name, "mtp.layers.") && std::strstr(name, ".self_attn.q_proj.weight"))  return 7;
    return -1;
}

// Minimal safetensors header parse: extract tensor name → {data_offset, nbytes, dtype}.
// Returns the number of bytes consumed from buf (the metadata JSON + 8-byte length prefix),
// or 0 on parse failure.  Populates 'out' with every BF16 tensor whose group id >= 0.
inline uint64_t sf_parse_header(const uint8_t* buf, size_t buf_size,
                                 std::map<std::string, SfTensorInfo>& out) {
    if (buf_size < 8) return 0;
    uint64_t json_len = 0;
    for (int i = 0; i < 8; ++i) json_len |= (uint64_t)buf[i] << (8 * i);
    if (json_len > buf_size - 8 || json_len == 0) return 0;
    const char* js = reinterpret_cast<const char*>(buf + 8);
    const char* end = js + json_len;
    // Minimal state-machine parser: find "name":{"dtype":...,"data_offsets":[start,end]}
    // We only extract tensors where sf_group_id(name) >= 0.
    const char* p = js;
    while (p < end) {
        // Find next "key" at the top level (name)
        const char* q = p;
        while (q < end && *q != '"') ++q;
        if (q >= end) break;
        ++q; // skip opening "
        const char* name_start = q;
        while (q < end && *q != '"') ++q;
        if (q >= end) break;
        std::string tensor_name(name_start, q - name_start);
        p = q + 1; // past closing "
        if (tensor_name == "__metadata__") {
            // skip the entire metadata object
            while (p < end && *p != '{') ++p;
            int depth = 0;
            while (p < end) { if (*p == '{') ++depth; else if (*p == '}') { --depth; if (!depth) { ++p; break; } } ++p; }
            continue;
        }
        if (sf_group_id(tensor_name.c_str()) < 0) {
            // skip to next top-level entry
            while (p < end && *p != '{') ++p;
            int depth = 0;
            while (p < end) { if (*p == '{') ++depth; else if (*p == '}') { --depth; if (!depth) { ++p; break; } } ++p; }
            continue;
        }
        // Find dtype — portable: scan byte by byte rather than memmem
        const char* dtype_p = nullptr;
        for (const char* s = p; s + 7 <= end; ++s)
            if (s[0]=='"'&&s[1]=='d'&&s[2]=='t'&&s[3]=='y'&&s[4]=='p'&&s[5]=='e'&&s[6]=='"')
                { dtype_p = s; break; }
        if (!dtype_p || dtype_p > end - 10) { p = dtype_p ? dtype_p + 1 : end; continue; }
        const char* dq1 = dtype_p + 7;
        while (dq1 < end && *dq1 != '"') ++dq1;
        if (dq1 >= end) break;
        ++dq1;
        const char* dq2 = dq1;
        while (dq2 < end && *dq2 != '"') ++dq2;
        char dtype_str[8] = {};
        size_t dlen = (size_t)(dq2 - dq1) < 7 ? (size_t)(dq2 - dq1) : 7;
        std::memcpy(dtype_str, dq1, dlen);
        // Find data_offsets — portable scan
        const char* off_p = nullptr;
        static const char kOff[] = "\"data_offsets\"";
        for (const char* s = p; s + 14 <= end; ++s)
            if (std::memcmp(s, kOff, 14) == 0) { off_p = s; break; }
        if (!off_p || off_p > end - 14) { p = off_p ? off_p + 1 : end; continue; }
        const char* ob = off_p + 14;
        while (ob < end && *ob != '[') ++ob;
        if (ob >= end) break;
        ++ob;
        uint64_t off_start = 0, off_end = 0;
        while (ob < end && (*ob == ' ' || *ob == '\t')) ++ob;
        while (ob < end && *ob >= '0' && *ob <= '9') off_start = off_start * 10 + (uint64_t)(*ob++ - '0');
        while (ob < end && *ob != ',') ++ob; ++ob;
        while (ob < end && (*ob == ' ' || *ob == '\t')) ++ob;
        while (ob < end && *ob >= '0' && *ob <= '9') off_end = off_end * 10 + (uint64_t)(*ob++ - '0');
        SfTensorInfo ti;
        ti.data_offset = off_start;
        ti.nbytes = off_end - off_start;
        std::memcpy(ti.dtype, dtype_str, sizeof(ti.dtype));
        ti.elem_count = (strcmp(dtype_str, "BF16") == 0 || strcmp(dtype_str, "F16") == 0)
                        ? ti.nbytes / 2 : ti.nbytes / 4;
        out[tensor_name] = ti;
        p = ob;
    }
    return 8 + json_len;
}

// Scan all .safetensors shards in model_dir and collect entries for the 8 tensor groups.
// Returns true if at least one shard was opened; populates found with the named tensors.
inline bool sf_scan_shards(const char* model_dir,
                            std::map<std::string, std::pair<std::string, SfTensorInfo>>& found,
                            std::string* err_out) {
    if (!model_dir) { if (err_out) *err_out = "null model_dir"; return false; }
    DIR* d = opendir(model_dir);
    if (!d) { if (err_out) *err_out = std::string("cannot open ") + model_dir; return false; }
    bool any = false;
    struct dirent* de;
    while ((de = readdir(d))) {
        const char* n = de->d_name;
        size_t ln = strlen(n);
        if (ln <= 12 || strcmp(n + ln - 12, ".safetensors") != 0 || strstr(n, "index")) continue;
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", model_dir, n);
        int fd = open(path, O_RDONLY);
        if (fd < 0) continue;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 8) { close(fd); continue; }
        void* m = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (m == MAP_FAILED) continue;
        std::map<std::string, SfTensorInfo> shard_tensors;
        uint64_t hdr_sz = sf_parse_header(static_cast<const uint8_t*>(m),
                                           (size_t)st.st_size, shard_tensors);
        munmap(m, (size_t)st.st_size);
        if (hdr_sz > 0) {
            any = true;
            for (auto& kv : shard_tensors)
                found[kv.first] = {std::string(path), kv.second};
        }
    }
    closedir(d);
    return any;
}

}  // namespace sf
}  // namespace baloon

