#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Host-only stubs for CUDA qualifiers (g++ builds without actual CUDA)
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

typedef void* cudaStream_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;
typedef void* cudaEvent_t;

enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidValue = 1, cudaErrorNotReady = 600 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };

inline const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "error"; }
inline cudaError_t cudaMalloc(void** p, size_t s) { *p = ::malloc(s); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
inline cudaError_t cudaFree(void* p) { ::free(p); return cudaSuccess; }
inline cudaError_t cudaMallocHost(void** p, size_t s) { *p = ::malloc(s); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
inline cudaError_t cudaFreeHost(void* p) { ::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { if(d&&s&&n) ::memcpy(d,s,n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t=nullptr) { if(d&&s&&n) ::memcpy(d,s,n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t=nullptr) { if(p&&n) ::memset(p,v,n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* p, int v, size_t n) { if(p&&n) ::memset(p,v,n); return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned int) { static char ev; *e = &ev; return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { static char ev; *e = &ev; return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t=nullptr) { return cudaSuccess; }
inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { if(ms) *ms=0; return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, int) { return cudaSuccess; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) { return cudaSuccess; }
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, void*, char*, size_t) { return cudaSuccess; }
inline cudaError_t cudaGraphGetNodes(cudaGraph_t, void*, size_t*) { return cudaSuccess; }
inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) { if(free) *free=32ULL*1024*1024*1024; if(total) *total=32ULL*1024*1024*1024; return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline cudaError_t cudaHostRegister(void*, size_t, unsigned int) { return cudaSuccess; }
inline cudaError_t cudaHostUnregister(void*) { return cudaSuccess; }
inline cudaError_t cudaHostGetDevicePointer(void** d, void* h, unsigned int) { *d=h; return cudaSuccess; }
#define cudaHostRegisterDefault 0
#define cudaHostRegisterPortable 1
#define cudaHostRegisterMapped 2
#define cudaStreamCaptureMode int
#define cudaStreamCaptureModeGlobal 0
#define cudaEventDisableTiming 2
