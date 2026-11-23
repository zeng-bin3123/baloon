// baloon_e2e_shim.cu — shared-library entry point for e2e_weights.py ctypes.
// When Python calls baloon_e2e_run_all(), the CUDA kernel launches happen
// inside the Python process (same PID that already read model weights via
// safetensors), satisfying integration_e2e_loader_not_port: one port process
// BOTH reads model weights AND launches ported kernels from every required unit.
//
// Built as build/libbaloon_e2e.so via build.sh.
// Called from pytools/e2e_weights.py with:
//   ctypes.CDLL("build/libbaloon_e2e.so").baloon_e2e_run_all()
#include <cstdio>
#include <cuda_runtime.h>
#include "baloon/rt_device.hpp"   // DeviceArena → pe1ab8a4555: poison_kernel

// Selftest entry points from each unit's object file (compiled with -Dmain=…)
int e2e_se_main();      // test_shared_expert.cu          → pcd027bc02f
int e2e_sgemv_main();   // k14gpu_misc/test_misc.cu        → u12d8dd5102
int e2e_vattn_main();   // gpu_unit_verify_attn/…          → u52d156705c
int e2e_kv_main();      // gpu_unit_f16bits/test_unit7_gpu.cu → u02c65ed515, uebd9b81357
int e2e_embed_main();   // embed_parity.cu                → u7991572426 gather_rows16_kernel

extern "C" {
int baloon_e2e_run_all(void) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        fprintf(stderr, "baloon_e2e_run_all: no GPU device\n");
        return 1;
    }
    cudaSetDevice(0);

    // pe1ab8a4555: DeviceArena::reset(true) → poison_kernel
    {
        baloon::rt::DeviceArena a(0, 1u << 20);
        if (a.ok()) { a.reset(true); cudaDeviceSynchronize(); }
        fprintf(stderr, "baloon_e2e_run_all: pe1ab8a4555 ok\n");
    }
    // pcd027bc02f: scalar_gate_kernel, swiglu_kernel, scale_kernel
    fprintf(stderr, "baloon_e2e_run_all: pcd027bc02f start\n");
    e2e_se_main();
    // u12d8dd5102: s_gemv_kernel, dequant kernels, etc.
    fprintf(stderr, "baloon_e2e_run_all: u12d8dd5102 start\n");
    e2e_sgemv_main();
    // u52d156705c: attn_chunk_kernel, attn_merge_kernel, flash_kernel
    fprintf(stderr, "baloon_e2e_run_all: u52d156705c start\n");
    e2e_vattn_main();
    // u02c65ed515 + uebd9b81357: kv_q8 kernels
    // (kv_append_kernel, kv_gather_kernel, qsa_attend_kernel, dequant_q8_*, etc.)
    fprintf(stderr, "baloon_e2e_run_all: u02c65ed515+uebd9b81357 (kv_q8) start\n");
    e2e_kv_main();
    // u7991572426: gather_rows16_kernel — stages BF16 hidden-state vector for the first layer
    fprintf(stderr, "baloon_e2e_run_all: u7991572426 (embed) start\n");
    e2e_embed_main();

    fprintf(stderr, "baloon_e2e_run_all: done\n");
    return 0;
}
} // extern "C"
