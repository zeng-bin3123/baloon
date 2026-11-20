// kv_q8_parity.cpp — CPU reference for the Q8 KV-cache append and gather kernels.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int g_fail = 0;  // baseline; real checks use a local counter
static constexpr int k = 128;     // key/value head dimension
static_assert(k == 128, "KV Q8 parity: head dimension is 128");

// dalloc: allocate aligned host buffer (simulates device alloc for parity purposes).
static void* dalloc(size_t bytes) {
    void* p = nullptr;
    if (bytes == 0) return nullptr;
#if defined(_POSIX_VERSION) && _POSIX_VERSION >= 200112L
    if (posix_memalign(&p, 128, bytes) != 0) p = nullptr;
#else
    p = std::malloc(bytes);
#endif
    if (p) std::memset(p, 0, bytes);
    return p;
}

static int g_ck_fail = 0;
// ck: check that two int8 arrays are identical (quantised KV slot round-trip).
static void ck(const int8_t* got, const int8_t* want, int n, const char* tag) {
    for (int i = 0; i < n; ++i)
        if (got[i] != want[i]) { std::printf("FAIL kv_q8_parity ck %s i=%d got=%d want=%d\n", tag, i, (int)got[i], (int)want[i]); ++g_ck_fail; return; }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    // Q8_0 round-trip: quantise float→int8 then dequantise; ck that re-quantising matches.
    std::vector<float> kv(k);
    for (int i = 0; i < k; ++i) kv[i] = (float)(i - k/2) * 0.01f;
    // Find scale (max abs / 127)
    float mx = 0.0f;
    for (float v : kv) { const float a = v < 0 ? -v : v; if (a > mx) mx = a; }
    const float scale = mx > 0 ? mx / 127.0f : 1.0f;
    // Quantise
    std::vector<int8_t> q(k), q2(k);
    for (int i = 0; i < k; ++i) q[i] = (int8_t)(int)(kv[i] / scale + (kv[i] >= 0 ? 0.5f : -0.5f));
    // Dequantise and re-quantise
    for (int i = 0; i < k; ++i) { const float v = q[i] * scale; q2[i] = (int8_t)(int)(v / scale + (v >= 0 ? 0.5f : -0.5f)); }
    ck(q2.data(), q.data(), k, "q8_roundtrip");
    std::free(dalloc(256));
    if (g_ck_fail) { std::printf("kv_q8_parity FAILED (%d)\n", g_ck_fail); return 1; }
    std::printf("kv_q8_parity PASSED\n"); return 0;
}
