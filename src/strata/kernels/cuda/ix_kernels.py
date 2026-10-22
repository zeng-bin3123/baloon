#!/usr/bin/env python3
"""ix_kernels.py — Python ctypes wrapper for BI-V100 CUDA kernels.

Usage:
    import ix_kernels
    ix_kernels.load("/path/to/libix_kernels.so")
    ix_kernels.infccl_init(4, [0, 1, 2, 3], max_elems=1024*1024)
    ix_kernels.ix_mmvq_q8_0(weights, activations, output, n_in, n_out, stream)
"""

import ctypes
import ctypes.util
import os
from typing import Optional

_lib: Optional[ctypes.CDLL] = None


def load(path: str = "libix_kernels.so") -> ctypes.CDLL:
    """Load the shared library. Call once at startup."""
    global _lib
    _lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    _setup_signatures()
    return _lib


def _setup_signatures():
    """Set up ctypes function signatures for type checking."""
    L = _lib

    # infccl
    L.infccl_init.restype = ctypes.c_int
    L.infccl_init.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int64]

    L.infccl_destroy.restype = None
    L.infccl_destroy.argtypes = []

    L.infccl_allreduce.restype = ctypes.c_int
    L.infccl_allreduce.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int64]

    L.infccl_allreduce_inplace.restype = ctypes.c_int
    L.infccl_allreduce_inplace.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]

    # mmvq
    L.ix_mmvq_q8_0.restype = None
    L.ix_mmvq_q8_0.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p
    ]

    L.ix_mmvq_q4k.restype = None
    L.ix_mmvq_q4k.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p
    ]

    L.ix_quantize_q8_1.restype = None
    L.ix_quantize_q8_1.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p
    ]

    L.ix_swiglu_quantize_q8_1.restype = None
    L.ix_swiglu_quantize_q8_1.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_void_p
    ]

    # router
    L.ix_router_top_k.restype = None
    L.ix_router_top_k.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    ]

    L.ix_router_bf16_gemv.restype = None
    L.ix_router_bf16_gemv.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]

    L.ix_router_and_shared_gate.restype = None
    L.ix_router_and_shared_gate.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]

    # moe
    L.ix_moe_combine.restype = None
    L.ix_moe_combine.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]

    L.ix_gated_residual.restype = None
    L.ix_gated_residual.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
    ]

    L.ix_expert_grouped.restype = None
    L.ix_expert_grouped.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]

    # bf16
    L.ix_bf16_gemv.restype = None
    L.ix_bf16_gemv.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]

    L.ix_rms_norm.restype = None
    L.ix_rms_norm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_int, ctypes.c_void_p
    ]

    # gdn
    L.ix_gdn_conv_silu.restype = None
    L.ix_gdn_conv_silu.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]

    L.ix_gdn_step.restype = None
    L.ix_gdn_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]


# ─── Convenience wrappers using torch tensors ──────────────────────

def torch_infccl_allreduce_inplace(tensor, rank: int):
    """AllReduce a torch.cuda tensor in-place."""
    assert _lib is not None, "Call ix_kernels.load() first"
    _lib.infccl_allreduce_inplace(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_int64(tensor.numel()),
        ctypes.c_int(rank)
    )


def torch_mmvq_q8_0(weights, activations, output, stream=None):
    """Q8_0 × Q8_1 GEMV on torch tensors."""
    assert _lib is not None
    n_out = output.shape[0]
    n_in = activations.numel() // (activations.element_size() + 2)  # Q8_1 blocks
    n_in *= 32  # elements per block
    s = ctypes.c_void_p(stream) if stream else ctypes.c_void_p(0)
    _lib.ix_mmvq_q8_0(
        ctypes.c_void_p(weights.data_ptr()),
        ctypes.c_void_p(activations.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_int64(n_in), ctypes.c_int64(n_out), s
    )
