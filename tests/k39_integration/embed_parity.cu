// embed_parity.cu - BF16 embedding gather reference: looks up each input token in
// embed_tokens (BF16 [vocab, hidden_size]) via gather_rows16_kernel and verifies
// the BF16 hidden-state vector is copied bit-exactly into the output buffer.
// Tests gather_rows16 at width=2048 (Qwen3.5-MoE hidden_size for embed_tokens.weight).
// gather_rows16 is declared in prefill_kernels_api.hpp and compiled from
// src/prefill/kernels.cu as a separate object (linked in by build.sh).
#include "baloon/prefill_kernels_api.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define EP_CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "embed_parity cuda error '%s' at line %d\n", \
        cudaGetErrorString(_e), __LINE__); std::exit(2); } } while (0)

template<typename T>
static T* ep_up(const std::vector<T>& h) {
    T* d; EP_CK(cudaMalloc((void**)&d, h.size() * sizeof(T)));
    EP_CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template<typename T>
static std::vector<T> ep_dn(T* d, size_t n) {
    EP_CK(cudaDeviceSynchronize());
    std::vector<T> h(n);
    EP_CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
template<typename T>
static T* ep_alloc(size_t n) {
    T* d; EP_CK(cudaMalloc((void**)&d, n * sizeof(T)));
    EP_CK(cudaMemset(d, 0, n * sizeof(T))); return d;
}

static uint64_t ep_rng = 0xc0ffee12ab345678ull;
static uint16_t ep_rand16() {
    ep_rng ^= ep_rng << 13; ep_rng ^= ep_rng >> 7; ep_rng ^= ep_rng << 17;
    return (uint16_t)(ep_rng & 0xFFFFu);
}

using namespace baloon::pk;

// test_embed_gather16 — gathers BF16 embedding rows by token ID from embed_tokens.weight
// into the initial hidden-state buffer for each position in the batch.
// Sentence: each output row is a bit-exact copy of the indexed table row (gather_rows16_kernel).
static int test_embed_gather16(int* checks_out, int* bad_out) {
    int bad = 0, checks = 0;

    // looks up each input token in embed_tokens (BF16 table, width=2048 = hidden_size)
    {
        const int vocab = 128, width = 2048, n_toks = 23;
        std::vector<uint16_t> table((size_t)vocab * width);
        for (auto& v : table) v = ep_rand16();
        std::vector<int32_t> ids(n_toks);
        for (int i = 0; i < n_toks; ++i) {
            ep_rng ^= ep_rng << 13; ep_rng ^= ep_rng >> 7; ep_rng ^= ep_rng << 17;
            ids[i] = (int32_t)((ep_rng >> 1) % vocab);
        }

        uint16_t* dt = ep_up(table);
        uint16_t* dd = ep_alloc<uint16_t>((size_t)n_toks * width);
        int32_t*  di = ep_up(ids);

        // gather_rows16_kernel stages the BF16 hidden-state vector for each token
        int rc = gather_rows16(dt, di, dd, n_toks, width);
        ++checks;
        if (rc != 0) {
            std::fprintf(stderr, "embed_parity: gather_rows16 rc=%d (expected 0)\n", rc);
            ++bad;
        }

        // CPU reference: each output row is an exact copy of the indexed table row
        auto out = ep_dn(dd, (size_t)n_toks * width);
        size_t mismatches = 0;
        for (int r = 0; r < n_toks; ++r) {
            if (std::memcmp(&out[(size_t)r * width],
                            &table[(size_t)ids[r] * width],
                            (size_t)width * 2) != 0) {
                ++mismatches;
            }
        }
        ++checks;
        if (mismatches != 0) {
            std::fprintf(stderr, "embed_parity: %zu/%d row(s) differ from table\n",
                         mismatches, n_toks);
            ++bad;
        }

        // width must be a multiple of 8 (uint4-granular)
        ++checks;
        if (gather_rows16(dt, di, dd, n_toks, 100) != -1) {
            std::fprintf(stderr, "embed_parity: expected -1 for width=100\n");
            ++bad;
        }

        cudaFree(dt); cudaFree(dd); cudaFree(di);
    }

    // also confirm the model's exact geometry: 2048 / 8 = 256 uint4 per row
    {
        const int width = 2048, rows = 1;
        std::vector<uint16_t> src((size_t)1 * width);
        for (auto& v : src) v = ep_rand16();
        std::vector<int32_t> idx = {0};
        uint16_t* ds = ep_up(src);
        uint16_t* dd = ep_alloc<uint16_t>((size_t)rows * width);
        int32_t*  di = ep_up(idx);
        ++checks;
        if (gather_rows16(ds, di, dd, rows, width) != 0) {
            std::fprintf(stderr, "embed_parity: width=2048 single-row failed\n"); ++bad;
        }
        auto out = ep_dn(dd, (size_t)rows * width);
        ++checks;
        if (std::memcmp(out.data(), src.data(), (size_t)width * 2) != 0) {
            std::fprintf(stderr, "embed_parity: width=2048 single-row content mismatch\n"); ++bad;
        }
        cudaFree(ds); cudaFree(dd); cudaFree(di);
    }

    *checks_out = checks;
    *bad_out    = bad;
    return bad;
}

// e2e_embed_main: strong symbol exported for baloon_e2e_run_all() (baloon_e2e_shim.cu / e2e_all_check.cu).
// kg_kernels.hpp carries an inline stub { return 0; } so the shared library links even when
// embed_parity.cu is absent; this strong definition overrides it and runs the real gather_rows16 test.
// Sentence: embed_parity.cu exports e2e_embed_main as a strong symbol so the shared library
// can call it without relying on -Dmain=e2e_embed_main renaming (unreliable on COREX clang-16).
int e2e_embed_main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::fprintf(stderr, "embed_parity: no GPU device\n"); return 2;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "embed_parity: cannot open GPU 0\n"); return 2;
    }
    int checks = 0, bad = 0;
    test_embed_gather16(&checks, &bad);
    std::fprintf(stderr, "embed_parity: checks=%d failures=%d\n", checks, bad);
    return bad ? 1 : 0;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::fprintf(stderr, "embed_parity: no GPU device\n"); return 2;
    }
    EP_CK(cudaSetDevice(0));
    int checks = 0, bad = 0;
    test_embed_gather16(&checks, &bad);
    std::printf("embed_parity: checks=%d failures=%d\n", checks, bad);
    if (bad) { std::printf("embed_parity FAILED\n"); return 1; }
    std::printf("embed_parity PASSED\n"); return 0;
}
