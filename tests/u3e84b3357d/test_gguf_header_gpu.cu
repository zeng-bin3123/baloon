// test_gguf_header_gpu.cu - unit u3e84b3357d, device side of GgufHeaderParsing.lean:
//   block_geometry, row_bytes (theory rowBytes) and align_up (theory alignUp) are __host__ __device__ in
//   include/baloon/gguf_reader.hpp. On every usable GPU this test checks
//   (1) the device evaluates the geometry table, rowBytes and alignUp exactly as the host does (all type ids 0..255);
//   (2) a parsed multi-tensor GGUF image copied to the device is addressed correctly there: a kernel that locates
//       every tensor at data_start + offset with size nbytes (computed on the device from type and element count)
//       reads exactly the marker bytes the host wrote at the start and the end of each tensor.
#include <cuda_runtime.h>

#include "baloon/gguf_reader.hpp"
#include "gguf_fixture.hpp"

#include <cstdio>
#include <vector>

using namespace baloon::gguf_reader;

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 2; } } while (0)

// (1) per type id: el, by, row_bytes for a few row sizes, align_up of a few positions
constexpr int kTypes = 256, kNs = 4, kPos = 8;
__constant__ uint64_t d_ns[kNs] = {256, 2560, 4096, 1ull << 20};
__constant__ uint64_t d_pos[kPos] = {0, 1, 31, 32, 33, 1000, 4097, (1ull << 33) + 5};
__global__ void geometry_kernel(uint64_t* out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= kTypes) return;
    uint64_t* o = out + (size_t)t * (2 + kNs + kPos);
    const BlockGeometry g = block_geometry((uint32_t)t);
    o[0] = g.el; o[1] = g.by;
    for (int i = 0; i < kNs; ++i) o[2 + i] = g.el ? row_bytes(g.el, g.by, d_ns[i]) : 0;
    const uint64_t align = (uint64_t)t + 1;                        // alignments 1..256
    for (int i = 0; i < kPos; ++i) o[2 + kNs + i] = align_up(d_pos[i], align);
}

// (2) per tensor: device-computed start/size, and the marker bytes found there
struct DevTensor { uint32_t type; uint64_t elements; uint64_t offset; };
__global__ void locate_kernel(const uint8_t* img, uint64_t header_end, uint64_t align, const DevTensor* ts, int nt,
                              uint64_t* start, uint64_t* size, uint8_t* first, uint8_t* last) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nt) return;
    const BlockGeometry g = block_geometry(ts[i].type);
    const uint64_t ds = align_up(header_end, align);
    start[i] = ds + ts[i].offset;
    size[i] = row_bytes(g.el, g.by, ts[i].elements);
    first[i] = img[start[i]];
    last[i] = img[start[i] + size[i] - 1];
}

static int run_device(int dev) {
    CUDA_OK(cudaSetDevice(dev));
    const long f0 = u3e::g_fail;
    // (1)
    const size_t stride = 2 + kNs + kPos;
    uint64_t* d_out = nullptr; CUDA_OK(cudaMalloc(&d_out, kTypes * stride * sizeof(uint64_t)));
    geometry_kernel<<<(kTypes + 63) / 64, 64>>>(d_out);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint64_t> got(kTypes * stride);
    CUDA_OK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_out));
    const uint64_t ns[kNs] = {256, 2560, 4096, 1ull << 20};
    const uint64_t pos[kPos] = {0, 1, 31, 32, 33, 1000, 4097, (1ull << 33) + 5};
    for (int t = 0; t < kTypes; ++t) {
        const uint64_t* o = &got[t * stride];
        const BlockGeometry g = block_geometry((uint32_t)t);
        CHECK(o[0] == g.el && o[1] == g.by);
        for (int i = 0; i < kNs; ++i) CHECK(o[2 + i] == (g.el ? row_bytes(g.el, g.by, ns[i]) : 0));
        for (int i = 0; i < kPos; ++i) {
            const uint64_t a = (uint64_t)t + 1, s = o[2 + kNs + i];
            CHECK(s == align_up(pos[i], a) && s % a == 0 && s >= pos[i] && s < pos[i] + a);
        }
    }
    CHECK(got[42 * stride] == 64 && got[42 * stride + 1] == 18 && got[42 * stride + 2 + 1] == 720);   // Q2_0, 2560-row

    // (2) a GGUF image with tensors of several types and a non-default alignment
    const uint32_t types[] = {42, 0, 2, 12, 30, 23, 8, 14};
    const uint64_t blocks[] = {40, 7, 3, 2, 5, 1, 9, 4};
    const int nt = 8;
    for (uint64_t align : {32ull, 64ull, 256ull}) {
        u3e::W w; w.u(kMagic, 4); w.u(3, 4); w.u(nt, 8); w.u(3, 8);
        w.s("general.architecture"); w.u(8, 4); w.s("qwen4exp");
        w.s("general.alignment"); w.u(4, 4); w.u(align, 4);
        w.s("general.name"); w.u(8, 4); w.s("gpu fixture");
        uint64_t off = 0; std::vector<uint64_t> offs, sizes;
        for (int i = 0; i < nt; ++i) {
            const BlockGeometry g = block_geometry(types[i]);
            const uint64_t el = g.el * blocks[i], by = row_bytes(g.el, g.by, el);
            w.s("t" + std::to_string(i)); w.u(1, 4); w.u(el, 8); w.u(types[i], 4); w.u(off, 8);
            offs.push_back(off); sizes.push_back(by); off = align_up(off + by, align);
        }
        const uint64_t header_end = w.b.size();
        std::vector<uint8_t> img = w.b; img.resize(align_up(header_end, align) + off, 0);
        const uint64_t ds = align_up(header_end, align);
        for (int i = 0; i < nt; ++i) { img[ds + offs[i]] = (uint8_t)(0xA0 + i); img[ds + offs[i] + sizes[i] - 1] = (uint8_t)(0xC0 + i); }
        auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f.has_value() && f->data_start == ds && f->alignment == align && f->tensors.size() == (size_t)nt);
        if (!f) continue;
        std::vector<DevTensor> ht;
        for (auto& t : f->tensors) ht.push_back({t.type, t.elements(), t.offset});
        uint8_t* d_img; DevTensor* d_ts; uint64_t *d_start, *d_size; uint8_t *d_first, *d_last;
        CUDA_OK(cudaMalloc(&d_img, img.size())); CUDA_OK(cudaMalloc(&d_ts, nt * sizeof(DevTensor)));
        CUDA_OK(cudaMalloc(&d_start, nt * 8)); CUDA_OK(cudaMalloc(&d_size, nt * 8));
        CUDA_OK(cudaMalloc(&d_first, nt)); CUDA_OK(cudaMalloc(&d_last, nt));
        CUDA_OK(cudaMemcpy(d_img, img.data(), img.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_ts, ht.data(), nt * sizeof(DevTensor), cudaMemcpyHostToDevice));
        locate_kernel<<<1, 64>>>(d_img, header_end, f->alignment, d_ts, nt, d_start, d_size, d_first, d_last);
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
        std::vector<uint64_t> st(nt), sz(nt); std::vector<uint8_t> fi(nt), la(nt);
        CUDA_OK(cudaMemcpy(st.data(), d_start, nt * 8, cudaMemcpyDeviceToHost)); CUDA_OK(cudaMemcpy(sz.data(), d_size, nt * 8, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(fi.data(), d_first, nt, cudaMemcpyDeviceToHost)); CUDA_OK(cudaMemcpy(la.data(), d_last, nt, cudaMemcpyDeviceToHost));
        for (int i = 0; i < nt; ++i) {
            CHECK(st[i] == f->data_start + f->tensors[i].offset && sz[i] == f->tensors[i].nbytes() && sz[i] == sizes[i]);
            CHECK(fi[i] == 0xA0 + i && la[i] == 0xC0 + i);
        }
        cudaFree(d_img); cudaFree(d_ts); cudaFree(d_start); cudaFree(d_size); cudaFree(d_first); cudaFree(d_last);
    }
    std::printf("[%s] device %d: geometry/rowBytes/alignUp on device, tensor addressing in a device image\n",
                u3e::g_fail == f0 ? " ok " : "FAIL", dev);
    return 0;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no GPU device\n"); return 2; }
    for (int d = 0; d < ndev; ++d) if (int rc = run_device(d)) return rc;
    std::printf("u3e84b3357d gpu: %d device(s), %ld checks, %ld failures\n", ndev, u3e::g_checks, u3e::g_fail);
    if (u3e::g_fail == 0) std::printf("U3E84B3357D GPU TESTS PASSED\n");
    return u3e::g_fail ? 1 : 0;
}
