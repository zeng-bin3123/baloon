// test_native_rope.cu — smoke test for src/kernels/cuda/native_rope.cu on the real GPU.
// Opens /dev/iluvatar[0-9]+ via cudaSetDevice, launches apply, verifies correctness.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "baloon/kg_kernels.hpp"

using namespace baloon::kg;

static int g_fail = 0, g_checks = 0;
static void ok(bool c, const char* msg) {
    ++g_checks;
    if (!c) { ++g_fail; std::printf("FAIL: %s\n", msg); }
}
static bool check_cuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { std::printf("CUDA error %s: %s\n", msg, cudaGetErrorString(e)); ++g_fail; return false; }
    return true;
}

template<class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t n_) : n(n_) { cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T)); cudaMemset(p, 0, (n ? n : 1) * sizeof(T)); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { if (n) cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice); }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> v(n); if (n) cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }
    operator T*() const { return p; }
};

// Host reference: NeoX-style rotation using component comp[i] from pos3[tok*3..tok*3+2]
static void rope_ref(std::vector<float>& x, const std::vector<int32_t>& pos3,
                     int n_tok, int nh, int hd, int rot, float theta, const int* comp) {
    const int half = rot / 2;
    for (int t = 0; t < n_tok; ++t)
        for (int h = 0; h < nh; ++h)
            for (int i = 0; i < half; ++i) {
                const double ang = (double)pos3[t * 3 + comp[i]] *
                                   std::pow((double)theta, -2.0 * i / rot);
                const double c = std::cos(ang), s = std::sin(ang);
                float* row = &x[((size_t)t * nh + h) * hd];
                const double a = row[i], b = row[i + half];
                row[i]        = (float)(a * c - b * s);
                row[i + half] = (float)(a * s + b * c);
            }
}

static bool nearly(float a, float b, float rtol=1e-4f, float atol=1e-4f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}
static void near_vec(const std::vector<float>& a, const std::vector<float>& b, const char* name) {
    bool ok = a.size() == b.size();
    for (size_t i = 0; ok && i < a.size(); ++i) ok = nearly(a[i], b[i]);
    ::ok(ok, name);
}

int main() {
    // Open the GPU device
    if (!check_cuda(cudaSetDevice(0), "cudaSetDevice")) return 1;
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    std::printf("GPU %d: %s\n", dev, prop.name);

    const int n_tok = 4, nh = 3, hd = 64, rot = 32, half = rot / 2;
    const float theta = 10000.f;
    // pos3: each token has 3 position components
    std::vector<int32_t> pos3 = {0,1,2, 3,4,5, 7,8,9, 15,16,17};
    // random-ish x
    std::vector<float> x((size_t)n_tok * nh * hd);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (float)(((int)i % 17) - 8) * 0.1f;

    // ---- test 1: default sections (all frequencies use component 0 = plain RoPE) ----
    mrope_table_set(nullptr);
    Dev<float> dx(x); Dev<int32_t> dp3(pos3);
    cudaError_t rc = native_rope_apply(dx, dp3, n_tok, nh, hd, rot, theta);
    ok(rc == cudaSuccess, "native_rope_apply default launch");

    std::vector<int> comp0(half, 0);
    std::vector<float> ref = x;
    rope_ref(ref, pos3, n_tok, nh, hd, rot, theta, comp0.data());
    near_vec(dx.get(), ref, "default native_rope equals host reference");

    // ---- test 2: multi-section RoPE {8, 4, 4} ----
    const int sec[3] = {8, 4, 4};
    mrope_table_set(sec);
    for (int i = 0; i < half; ++i) {
        const int exp = i < 8 ? 0 : (i < 12 ? 1 : 2);
        ok(mrope_table(i, half) == exp, "mrope_table section boundary");
    }
    Dev<float> dm(x);
    ok(native_rope_apply(dm, dp3, n_tok, nh, hd, rot, theta) == cudaSuccess, "mrope launch");
    std::vector<int> comp2(half);
    for (int i = 0; i < half; ++i) comp2[i] = mrope_table(i, half);
    ref = x; rope_ref(ref, pos3, n_tok, nh, hd, rot, theta, comp2.data());
    near_vec(dm.get(), ref, "mrope matches host reference with 3-axis sections");

    // ---- test 3: bad sections (sum != rot/2) are rejected ----
    const int bad[3] = {8, 4, 3};
    mrope_table_set(bad);
    Dev<float> db(x);
    ok(native_rope_apply(db, dp3, n_tok, nh, hd, rot, theta) != cudaSuccess,
       "mismatched sections rejected");
    mrope_table_set(nullptr);

    // ---- test 4: enable/disable flag ----
    native_rope_set_enabled(false);
    ok(!native_rope_enabled(), "set_enabled(false) works");
    native_rope_set_enabled(true);
    ok(native_rope_enabled(),  "set_enabled(true) works");

    // ---- test 5: validation rejects odd rot ----
    Dev<float> dv(x);
    ok(native_rope_apply(dv, dp3, n_tok, nh, hd, 33, theta) != cudaSuccess, "odd rot rejected");

    // ---- test 6: overlaps check - null pointer ----
    ok(native_rope_apply(nullptr, dp3, n_tok, nh, hd, rot, theta) != cudaSuccess,
       "null x rejected");

    std::printf("%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail == 0) std::printf("NATIVE ROPE UNIT TEST PASSED\n");
    else             std::printf("NATIVE ROPE UNIT TEST FAILED\n");
    return g_fail ? 1 : 0;
}
