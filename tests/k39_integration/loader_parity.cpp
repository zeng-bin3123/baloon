// loader_parity.cpp — invariant tests for safetensors_loader.hpp (sf_group_id, sf_parse_header).
// No GPU required: all tests run on the host.  Builds as a standalone executable.
#include "baloon/safetensors_loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <map>
#include <vector>

static long g_checks = 0, g_fail = 0;
#define LP(x) do { ++g_checks; if (!(x)) { ++g_fail; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)

static void test_group_id() {
    using namespace baloon::sf;
    // the 8 known groups
    LP(sf_group_id("lm_head.weight") == 0);
    LP(sf_group_id("model.language_model.layers.0.mlp.experts.down_proj") == 1);
    LP(sf_group_id("model.language_model.layers.39.mlp.experts.down_proj") == 1);
    LP(sf_group_id("model.language_model.layers.0.mlp.experts.gate_up_proj") == 2);
    LP(sf_group_id("model.language_model.layers.5.self_attn.q_proj.weight") == 3);
    LP(sf_group_id("model.language_model.norm.weight") == 4);
    LP(sf_group_id("mtp.layers.0.mlp.experts.down_proj") == 5);
    LP(sf_group_id("mtp.layers.0.mlp.experts.gate_up_proj") == 6);
    LP(sf_group_id("mtp.layers.0.self_attn.q_proj.weight") == 7);
    // groups 8-11: layernorm weights
    LP(sf_group_id("model.language_model.layers.0.input_layernorm.weight") == 8);
    LP(sf_group_id("model.language_model.layers.39.input_layernorm.weight") == 8);
    LP(sf_group_id("model.language_model.layers.0.post_attention_layernorm.weight") == 9);
    LP(sf_group_id("mtp.layers.0.input_layernorm.weight") == 10);
    LP(sf_group_id("mtp.layers.0.post_attention_layernorm.weight") == 11);
    // non-members return -1
    LP(sf_group_id("model.language_model.embed_tokens.weight") == -1);
    LP(sf_group_id("model.language_model.layers.0.linear_attn.A_log") == -1);
    LP(sf_group_id("model.visual.patch_embed.proj.weight") == -1);
    LP(sf_group_id(nullptr) == -1);
    LP(sf_group_id("") == -1);
}

static void test_parse_header() {
    using namespace baloon::sf;
    // Build a minimal safetensors JSON for two tensors (one in group, one not)
    const char* json =
        "{\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[248320,2048],"
        "\"data_offsets\":[0,1017978880]},"
        "\"model.language_model.embed_tokens.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[248320,2048],\"data_offsets\":[1017978880,2035957760]}}";
    uint64_t json_len = (uint64_t)strlen(json);
    std::vector<uint8_t> buf(8 + json_len);
    for (int i = 0; i < 8; ++i) buf[i] = (uint8_t)(json_len >> (8 * i));
    std::memcpy(buf.data() + 8, json, json_len);

    std::map<std::string, SfTensorInfo> out;
    uint64_t consumed = sf_parse_header(buf.data(), buf.size(), out);
    LP(consumed == 8 + json_len);
    // only lm_head.weight should appear (group id 0); embed_tokens is -1
    LP(out.size() == 1);
    LP(out.count("lm_head.weight") == 1);
    LP(out.count("model.language_model.embed_tokens.weight") == 0);
    if (out.count("lm_head.weight")) {
        LP(strcmp(out["lm_head.weight"].dtype, "BF16") == 0);
        LP(out["lm_head.weight"].data_offset == 0);
        LP(out["lm_head.weight"].nbytes == 1017978880ULL);
    }

    // too-short buffer
    LP(sf_parse_header(buf.data(), 4, out) == 0);
    // empty json_len
    std::vector<uint8_t> bad(8, 0);
    LP(sf_parse_header(bad.data(), bad.size(), out) == 0);
}

static void test_validate_shape() {
    using namespace baloon::sf;
    SfTensorInfo ti; std::memset(&ti, 0, sizeof(ti));
    std::string err;

    // group 0: lm_head.weight [248320,2048] BF16 → 248320*2048*2 bytes
    std::strncpy(ti.dtype, "BF16", sizeof(ti.dtype));
    ti.nbytes = 248320ULL * 2048 * 2;
    LP(sf_validate_bf16_shape("lm_head.weight", ti, 0, &err));

    // wrong dtype → false
    std::strncpy(ti.dtype, "F32", sizeof(ti.dtype));
    LP(!sf_validate_bf16_shape("lm_head.weight", ti, 0, &err));

    // correct dtype, wrong nbytes → false
    std::strncpy(ti.dtype, "BF16", sizeof(ti.dtype));
    ti.nbytes = 42;
    LP(!sf_validate_bf16_shape("lm_head.weight", ti, 0, &err));

    // group 1: mlp.experts.down_proj [256,2048,512] BF16
    ti.nbytes = 256ULL * 2048 * 512 * 2;
    LP(sf_validate_bf16_shape("model.language_model.layers.0.mlp.experts.down_proj", ti, 1, &err));

    // group 2: mlp.experts.gate_up_proj [256,1024,2048] BF16
    ti.nbytes = 256ULL * 1024 * 2048 * 2;
    LP(sf_validate_bf16_shape("model.language_model.layers.0.mlp.experts.gate_up_proj", ti, 2, &err));

    // group 3: self_attn.q_proj.weight [8192,2048] BF16
    ti.nbytes = 8192ULL * 2048 * 2;
    LP(sf_validate_bf16_shape("model.language_model.layers.5.self_attn.q_proj.weight", ti, 3, &err));

    // group 4: model.language_model.norm.weight [2048] BF16
    ti.nbytes = 2048ULL * 2;
    LP(sf_validate_bf16_shape("model.language_model.norm.weight", ti, 4, &err));

    // group 8: input_layernorm.weight [2048] BF16
    LP(sf_validate_bf16_shape("model.language_model.layers.0.input_layernorm.weight", ti, 8, &err));

    // group 9: post_attention_layernorm.weight [2048] BF16
    LP(sf_validate_bf16_shape("model.language_model.layers.0.post_attention_layernorm.weight", ti, 9, &err));

    // group 10: mtp input_layernorm.weight [2048] BF16
    LP(sf_validate_bf16_shape("mtp.layers.0.input_layernorm.weight", ti, 10, &err));

    // group 11: mtp post_attention_layernorm.weight [2048] BF16
    LP(sf_validate_bf16_shape("mtp.layers.0.post_attention_layernorm.weight", ti, 11, &err));

    // out-of-range gid → false
    LP(!sf_validate_bf16_shape("lm_head.weight", ti, -1, &err));
    LP(!sf_validate_bf16_shape("lm_head.weight", ti, 12, &err));
}

int main() {
    test_group_id();
    test_parse_header();
    test_validate_shape();
    std::printf("loader_parity: checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("loader_parity FAILED\n"); return 1; }
    std::printf("loader_parity PASSED\n"); return 0;
}

