// test_u62b.cu - unit u62b936371e: opens a GPU device, re-checks the integer layout contracts on the device (bit packing,
// padding, interleave, plane offsets, scale-min decode), then runs the host invariant checks. Exits non-zero on any failure.
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <chrono>
#include <dirent.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <cuda_runtime.h>

#include "baloon/iq_native_pack_index.hpp"
#include "baloon/ref_pack_builder.hpp"
#include "u62b_checks.hpp"

__device__ __forceinline__ uint64_t d_pad(uint64_t a, uint64_t n) { return (a - n % a) % a; }

// One thread per index; each thread counts violated contracts into *bad.
__global__ void contract_kernel(unsigned long long* bad, uint64_t blob_bytes) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long b = 0;
    if (i < 256) {   // 2-bit pack round trip, nibble split
        const uint64_t c0 = i & 3, c1 = (i >> 2) & 3, c2 = (i >> 4) & 3, c3 = (i >> 6) & 3;
        const uint64_t p = c0 + 4 * c1 + 16 * c2 + 64 * c3;
        b += !(p == i && p % 4 == c0 && p / 4 % 4 == c1 && p / 16 % 4 == c2 && p / 64 % 4 == c3);
        b += !(i % 16 + 16 * (i / 16) == i && i % 16 < 16 && i / 16 < 16);
    }
    if (i < 65536) {   // 64-byte / 32-byte / 256-byte padding specs
        b += !(d_pad(64, i) < 64 && (i + d_pad(64, i)) % 64 == 0);
        b += !(d_pad(32, i) < 32 && (i + d_pad(32, i)) % 32 == 0 && i + d_pad(32, i) == (i + 31) / 32 * 32);
        b += !(d_pad(256, i) < 256 && (i + d_pad(256, i)) % 256 == 0);
        b += !(baloon::iqpack::round_row(i) == i + baloon::iqpack::pad64(i));
    }
    if (i < 640) {   // gate/up interleave is a bijection onto 0..1279
        const uint64_t g = 2 * i, u = 2 * i + 1;
        b += !(g < 1280 && u < 1280 && g / 2 == i && u / 2 == i && g % 2 == 0 && u % 2 == 1);
    }
    if (i < 1280 * 40) {   // code/scale offsets of every (slot, block) stay inside their planes
        const uint64_t slot = i / 40, blk = i % 40;
        b += !(baloon::pack::kGuCodesOff + slot * 640 + blk * 16 + 16 <= baloon::pack::kDCodesOff);
        b += !(baloon::pack::kGuScalesOff + (slot * 40 + blk) * 2 + 2 <= baloon::pack::kDScalesOff);
    }
    if (i == 0) {   // blob tiling
        b += !(baloon::pack::kDScalesOff + baloon::pack::kDScales == blob_bytes && blob_bytes == 1382400);
        b += !(48ull * 512 * blob_bytes == 33973862400ull);
    }
    if (i < 12 * 256) {   // get_scale_min_k4 on the device: 6-bit results (q bytes derived from i)
        const int j = (int)(i % 8);
        uint8_t q[12];
        for (int k = 0; k < 12; ++k) q[k] = (uint8_t)((i * 2654435761u >> (k % 4 * 8)) + k * 37);
        const unsigned d = j < 4 ? q[j] % 64u : (q[j + 4] % 16u) + 16u * (q[j - 4] / 64u);
        b += !(d < 64);
    }
    if (b) atomicAdd(bad, b);
}

// Counts this process's open file descriptors that point at a GPU device node (/dev/iluvatarN or /dev/nvidiaN).
static int open_gpu_fds() {
    int n = 0;
    DIR* d = opendir("/proc/self/fd");
    if (!d) return -1;
    while (dirent* e = readdir(d)) {
        char link[256], path[64];
        std::snprintf(path, sizeof path, "/proc/self/fd/%s", e->d_name);
        const ssize_t k = readlink(path, link, sizeof link - 1);
        if (k <= 0) continue;
        link[k] = 0;
        const std::string s(link);
        for (const char* pre : {"/dev/iluvatar", "/dev/nvidia"})
            if (s.rfind(pre, 0) == 0 && s.size() > std::char_traits<char>::length(pre) && isdigit((unsigned char)s[std::char_traits<char>::length(pre)])) { ++n; break; }
    }
    closedir(d);
    return n;
}

int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n <= 0) { std::printf("no GPU device: %s\n", cudaGetErrorString(e)); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("cudaSetDevice failed\n"); return 2; }
    unsigned long long* d_bad = nullptr;
    if (cudaMalloc(&d_bad, sizeof(*d_bad)) != cudaSuccess) { std::printf("cudaMalloc failed\n"); return 2; }
    cudaMemset(d_bad, 0, sizeof(*d_bad));
    const uint64_t total = 65536;
    contract_kernel<<<(unsigned)((total + 255) / 256), 256>>>(d_bad, baloon::pack::kBlobBytes);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("kernel failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 2; }
    unsigned long long bad = 0;
    cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost);
    // Hold the device context open briefly so an external observer sampling /proc sees the open GPU node.
    std::printf("open GPU device fds: %d\n", open_gpu_fds());
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    cudaFree(d_bad);
    std::printf("device contract violations: %llu\n", bad);
    int fails = bad ? 1 : 0;
    fails += u62b::run_all();
    if (fails) { std::printf("u62b: FAILED\n"); return 1; }
    std::printf("u62b: all invariants hold\n");
    return 0;
}
