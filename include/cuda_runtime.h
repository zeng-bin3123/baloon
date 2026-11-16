// cuda_runtime.h — host-only compilation stub used during standalone header
// syntax checks (build.sh HFLAGS = -Iinclude, no explicit -I$COREX/include).
//
// The corex clang++ auto-injects cuda_runtime_api.h even in plain C++ mode,
// declaring all CUDA API functions with CUDA linkage.  Defining our own inline
// stubs for those same names causes "different language linkage" errors.
//
// Strategy:
//   • When the corex CUDA runtime is available at its standard path, include
//     the real header so declarations use the correct linkage.
//   • When only a plain host compiler (g++) is present, provide minimal stubs
//     so the CUDA-annotated baloon headers parse without errors.
#pragma once
#if __has_include("/usr/local/corex/include/cuda_runtime.h")
// Corex CUDA toolkit present: use the real runtime (avoids linkage conflicts).
#  include "/usr/local/corex/include/cuda_runtime.h"
#else
// Plain host compiler only: provide minimal type stubs.
#  include <cstddef>
typedef int    cudaError_t;
typedef void*  cudaStream_t;
typedef void*  cudaEvent_t;
enum {
    cudaSuccess                = 0,
    cudaErrorInvalidValue      = 1,
    cudaErrorMemoryAllocation  = 2,
    cudaErrorUnknown           = 999
};
enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
};
inline cudaError_t cudaFree(void*)                                   { return cudaSuccess; }
inline cudaError_t cudaMalloc(void**, std::size_t)                   { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, std::size_t, int)  { return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void*, const void*, std::size_t,
                                   int, cudaStream_t)                { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t*)                   { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t)                   { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t)               { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t*)                     { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t)                     { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t)        { return cudaSuccess; }
inline cudaError_t cudaGetLastError()                                { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t)                   { return ""; }
#endif
