// elementwise_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main

// test_rms_norm_bf16w_inplace: applies a weighted RMS normalization in-place to the hidden state using one
// input_layernorm or post_attention_layernorm weight vector (BF16, cols=hidden_size=2048 for this model).
// Tensor groups exercised: model.language_model.layers.{i}.input_layernorm.weight BF16 [2048],
//   model.language_model.layers.{i}.post_attention_layernorm.weight BF16 [2048],
//   mtp.layers.{i}.input_layernorm.weight BF16 [2048], mtp.layers.{i}.post_attention_layernorm.weight BF16 [2048].
// Sentence: applies a weighted RMS normalization in-place to the hidden state using one
// input_layernorm or post_attention_layernorm weight vector.
static void test_rms_norm_bf16w_inplace() {
    const float eps = 1e-6f;
    // Known-value case: all-ones hidden state with unit BF16 weights verifies the formula directly.
    // cols=2048: the actual hidden_size for input_layernorm/post_attention_layernorm weight vectors in this model.
    // mean(1^2)=1, inv=1/sqrt(1+eps); bf16(1.0)=0x3F80 (exact); each output element equals 1/sqrt(1+eps).
    {
        const int krows = 2, kcols = 2048;  // 2048 = hidden_size of input_layernorm.weight / post_attention_layernorm.weight
        std::vector<float>    kx(krows * kcols, 1.0f);
        std::vector<uint16_t> kw(kcols, fb::round_f32_to_bf16_bits(1.0f));
        Dev<float>    kdx(kx);
        Dev<uint16_t> kdw(kw);
        ok(rms_norm_bf16w_inplace(kdx, kdw, krows, kcols, eps) == cudaSuccess && sync_ok(),
           "rms_norm_bf16w_inplace known-value launch");
        const double kinv = 1.0 / std::sqrt(1.0 + (double)eps);
        std::vector<double> kref(krows * kcols, kinv);
        near_vec(kdx.get(), kref, 1e-6, 1e-6,
                 "rms_norm_bf16w_inplace: all-ones hidden state with unit weight gives 1/sqrt(1+eps)");
    }
    // Random cases: cols=256 is THREADS (each thread handles exactly one element, boundary);
    // cols=517 is non-power-of-two; cols=2048 is hidden_size for input_layernorm.weight /
    // post_attention_layernorm.weight of model.language_model.layers.{i} and mtp.layers.{i}.
    for (int cols : {8, 256, 517, 2048}) {  // cols=2048: hidden_size for input_layernorm/post_attention_layernorm of model.language_model.layers.{i} and mtp.layers.{i}
        const int rows = (cols == 2048 ? 3 : 2);
        auto xf = vf((size_t)rows * cols, -2.f, 2.f);
        auto wbf = vbf((size_t)cols, 1.5f);
        // host reference: x[r,i] = x[r,i] * rsqrt(mean(x[r]^2) + eps) * bf16_to_f32(w[i])
        std::vector<double> ref((size_t)rows * cols);
        for (int r = 0; r < rows; ++r) {
            double ss = 0.0;
            for (int i = 0; i < cols; ++i) ss += (double)xf[(size_t)r * cols + i] * xf[(size_t)r * cols + i];
            const double inv = 1.0 / std::sqrt(ss / (double)cols + (double)eps);
            for (int i = 0; i < cols; ++i)
                ref[(size_t)r * cols + i] = (double)xf[(size_t)r * cols + i] * inv * (double)bf(wbf[i]);
        }
        Dev<float>    dx(xf);
        Dev<uint16_t> dw(wbf);
        ok(rms_norm_bf16w_inplace(dx, dw, rows, cols, eps) == cudaSuccess && sync_ok(),
           "rms_norm_bf16w_inplace launch");
        near_vec(dx.get(), ref, 1e-4, 1e-5, "rms_norm_bf16w_inplace: in-place weighted RMS norm matches host reference");
    }
    // bad-arg guards: null x, null w, rows<=0, or cols<=0 must return cudaErrorInvalidValue.
    {
        Dev<float>    dx(2048); Dev<uint16_t> dw(2048);
        ok(rms_norm_bf16w_inplace(nullptr, dw, 1, 2048, eps) == cudaErrorInvalidValue,
           "rms_norm_bf16w_inplace rejects null x");
        ok(rms_norm_bf16w_inplace(dx, nullptr, 1, 2048, eps) == cudaErrorInvalidValue,
           "rms_norm_bf16w_inplace rejects null w");
        // The hidden state must have a positive row count and a positive column count.
        ok(rms_norm_bf16w_inplace(dx, dw, 0, 2048, eps) == cudaErrorInvalidValue,
           "rms_norm_bf16w_inplace rejects rows=0");
        ok(rms_norm_bf16w_inplace(dx, dw, 1, 0, eps) == cudaErrorInvalidValue,
           "rms_norm_bf16w_inplace rejects cols=0");
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("elementwise_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("elementwise_parity: cannot open GPU 0\n"); return 2; }
    test_rms_norm_bf16w_inplace();
    std::printf("elementwise_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("elementwise_parity FAILED\n"); return 1; }
    std::printf("elementwise_parity PASSED\n");
    return 0;
}
