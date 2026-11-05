// Shared helpers of the unit-14 GPU tests (host side only).
#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "baloon/f16_conversion.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/scalar_dequantizers.hpp"

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 80) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %d at %s:%d  %s\n", (int)e_, __FILE__, __LINE__, #x); std::exit(2); } } while (0)

template <class T> static T* dalloc(size_t n) { T* p = nullptr; CUOK(cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T))); return p; }
template <class T> static T* dup(const std::vector<T>& v) { T* p = dalloc<T>(v.size()); if (!v.empty()) CUOK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> fetch(const T* p, size_t n) { std::vector<T> v(n); if (n) CUOK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
static void sync() { CUOK(cudaGetLastError()); CUOK(cudaDeviceSynchronize()); }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t x) : s(x * 0x9E3779B97F4A7C15ull + 99) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t below(uint32_t n) { return (uint32_t)(next() >> 16) % n; }
    double uni() { return (double)(next() >> 11) / 9007199254740992.0; }
    float sym(float a) { return (float)((uni() * 2.0 - 1.0) * a); }
};
static uint16_t h_of(float f) { return baloon::f16::float_to_half(f); }
static float f_of(uint16_t h) { return baloon::f16::half_to_float(h); }
static bool near_tol(double got, double want, double abs_tol, double rel_tol) { return std::fabs(got - want) <= abs_tol + rel_tol * std::fabs(want); }
