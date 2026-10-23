// tests/test_unit3_weights.cpp — Smoke test for unit 3 weight modules
#include "strata/core/weights.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <unistd.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++tests_passed; } \
    else { ++tests_failed; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

int main() {
    printf("=== strata unit 3 weights smoke test ===\n");

    // ── 1. WeightRef defaults ────────────────────────────────────────
    {
        strata::WeightRef ref;
        CHECK(ref.data == nullptr, "WeightRef default data null");
        CHECK(ref.total_bytes == 0, "WeightRef default total_bytes 0");
        CHECK(ref.ne0 == 0, "WeightRef default ne0 0");
        CHECK(ref.ne1 == 0, "WeightRef default ne1 0");
        CHECK(ref.kind == strata::WeightKind::Verbatim, "WeightRef default kind Verbatim");
        CHECK(ref.code_bits == 0, "WeightRef default code_bits 0");
        CHECK(!ref.quantized(), "WeightRef default not quantized");
        CHECK(!ref.wants_q8k(), "WeightRef default not wants_q8k");
        CHECK(ref.native_data == nullptr, "WeightRef default native_data null");
        CHECK(ref.native_type == -1, "WeightRef default native_type -1");
    }

    // ── 2. WeightTable::find returns null on empty ───────────────────
    {
        strata::WeightTable wt;
        const strata::WeightRef* r = wt.find("nonexistent");
        CHECK(r == nullptr, "find returns null on empty table");
    }

    // ── 3. pool_bytes fails on missing dir ───────────────────────────
    {
        std::string err;
        uint64_t bytes = 0;
        bool ok = strata::WeightTable::pool_bytes("/tmp/no_such_dir_12345", bytes, err);
        CHECK(!ok, "pool_bytes fails on missing dir");
        CHECK(!err.empty(), "pool_bytes sets error on failure");
    }

    // ── 4. load fails on missing dir ─────────────────────────────────
    {
        strata::WeightTable wt;
        std::string err;
        bool ok = wt.load("/tmp/no_such_dir_12345", nullptr, 0, err);
        CHECK(!ok, "load fails on missing dir");
        CHECK(!err.empty(), "load sets error on failure");
    }

    // ── 5. f16 conversions round-trip ────────────────────────────────
    {
        // Normal-range fp16 values: exact or <0.1% relative error.
        // Subnormal fp16 values (|v| < 6.1e-5) have much larger quantization
        // error — that is expected, not a bug.
        float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f, 0.125f};
        bool all_ok = true;
        for (float v : vals) {
            uint16_t h = strata::kernels::f16_from_f32(v);
            float back = strata::kernels::f32_from_f16(h);
            if (v == 0.0f) {
                if (back != 0.0f) all_ok = false;
            } else {
                float err = std::fabs(back - v) / std::fabs(v);
                if (err > 0.001f) all_ok = false;
            }
        }
        CHECK(all_ok, "f16 conversions round-trip");
    }

    // ── 6. bf16_gemv basic test ──────────────────────────────────────
    {
        // 2×3 matrix, BF16 values
        // BF16 of 1.0 = 0x3F80 (upper 16 bits of f32 1.0)
        uint16_t W[] = {0x3F80, 0x3F80, 0x3F80,   // row 0: [1, 1, 1]
                        0x4000, 0x4000, 0x4000};   // row 1: [2, 2, 2]
        uint16_t x[] = {0x3F80, 0x3F80, 0x3F80};   // [1, 1, 1]
        float out[2] = {0, 0};
        strata::bf16_gemv(W, x, out, 2, 3);
        CHECK(std::fabs(out[0] - 3.0f) < 0.01f, "bf16_gemv row 0 = 3.0");
        CHECK(std::fabs(out[1] - 6.0f) < 0.01f, "bf16_gemv row 1 = 6.0");
    }

    // ── 7. NativeHead defaults ───────────────────────────────────────
    {
        strata::NativeHead head;
        CHECK(!head.loaded(), "NativeHead default not loaded");
        CHECK(head.weights() == nullptr, "NativeHead default weights null");
        CHECK(head.type() == -1, "NativeHead default type -1");
        CHECK(head.weight_bytes() == 0, "NativeHead default bytes 0");
        CHECK(head.tensor_count() == 0, "NativeHead default tensor_count 0");
    }

    // ── 8. NativeHead load fails on missing file ─────────────────────
    {
        strata::NativeHead head;
        std::string err;
        bool ok = head.load("/tmp/no_such_file.gguf", "output.weight", err);
        CHECK(!ok, "NativeHead load fails on missing file");
        CHECK(!err.empty(), "NativeHead load sets error");
    }

    // ── 9. NativeHead run fails when not loaded ──────────────────────
    {
        strata::NativeHead head;
        std::string err;
        float dummy_in = 0, dummy_out = 0;
        bool ok = head.run(&dummy_in, &dummy_out, nullptr, err);
        CHECK(!ok, "NativeHead run fails when not loaded");
    }

    // ── 10. NativeEmbed defaults ─────────────────────────────────────
    {
        strata::NativeEmbed embed;
        CHECK(embed.bytes() == 0, "NativeEmbed default bytes 0");
    }

    // ── 11. GgufReader fails on missing file ─────────────────────────
    {
        strata::artifact::GgufReader reader("/tmp/no_such_file.gguf");
        CHECK(!reader.ok(), "GgufReader fails on missing file");
        CHECK(!reader.error().empty(), "GgufReader sets error");
    }

    // ── 12. replace_native fails on missing GGUF ─────────────────────
    {
        strata::WeightTable wt;
        std::string err;
        int n = strata::replace_native(wt, "/tmp/no_such_file.gguf", false, err);
        CHECK(n == -1, "replace_native fails on missing GGUF");
        CHECK(!err.empty(), "replace_native sets error");
    }

    // ── 13. replace_native returns 0 on empty table ──────────────────
    {
        // Create a fake GGUF file that's valid but empty
        // (We can't easily test this without a real GGUF, so we verify error path)
        strata::WeightTable wt;
        std::string err;
        // With a valid but non-GGUF file
        FILE* f = fopen("/tmp/test_not_gguf.bin", "wb");
        if (f) {
            uint32_t magic = 0x12345678;
            fwrite(&magic, 4, 1, f);
            fclose(f);
            int n = strata::replace_native(wt, "/tmp/test_not_gguf.bin", false, err);
            CHECK(n == -1, "replace_native fails on non-GGUF file");
            unlink("/tmp/test_not_gguf.bin");
        } else {
            CHECK(true, "replace_native non-GGUF test skipped (no /tmp)");
        }
    }

    // ── 14. LoadReport defaults ──────────────────────────────────────
    {
        strata::LoadReport lr;
        CHECK(lr.tensors == 0, "LoadReport default tensors 0");
        CHECK(lr.arena_bytes == 0, "LoadReport default arena_bytes 0");
        CHECK(lr.re_rounded == 0, "LoadReport default re_rounded 0");
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
