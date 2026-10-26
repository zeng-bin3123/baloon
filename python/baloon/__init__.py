"""
baloon — Python interface to Baloon inference engine for Iluvatar BI-V100.

Usage:
    import baloon
    print(baloon.device_count())
    info = baloon.query_device(0)
    print(info['name'], info['vram_gib'])
    baloon.smoke_test(0)   # raises on failure
"""

import ctypes
import os
import pathlib

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Locate libbaloon_core.so
# ---------------------------------------------------------------------------
_LIB = None

def _find_lib():
    """Search common locations for the shared library."""
    candidates = [
        # Relative to this file (pip install / dev layout)
        pathlib.Path(__file__).parent.parent.parent / "build" / "libbaloon_core.so",
        # System install
        pathlib.Path("/usr/local/lib/libbaloon_core.so"),
    ]
    env_path = os.environ.get("BALOON_LIB_PATH")
    if env_path:
        candidates.insert(0, pathlib.Path(env_path))

    for p in candidates:
        if p.is_file():
            return ctypes.CDLL(str(p))
    return None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _find_lib()
    if _LIB is None:
        raise RuntimeError(
            "libbaloon_core.so not found. Build with `make` first, "
            "or set BALOON_LIB_PATH."
        )
    return _LIB


# ---------------------------------------------------------------------------
# C struct mirror
# ---------------------------------------------------------------------------
class _DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("device_count",          ctypes.c_int),
        ("warp_size",             ctypes.c_int),
        ("max_threads",           ctypes.c_int),
        ("sm_count",              ctypes.c_int),
        ("compute_major",         ctypes.c_int),
        ("compute_minor",         ctypes.c_int),
        ("max_shmem_per_block",   ctypes.c_int),
        ("total_mem",             ctypes.c_size_t),
        ("free_mem",              ctypes.c_size_t),
        ("name",                  ctypes.c_char * 256),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def device_count() -> int:
    """Return the number of visible CUDA/CoreX devices."""
    lib = _lib()
    lib.baloon_device_count.restype = ctypes.c_int
    return lib.baloon_device_count()


def query_device(dev: int = 0) -> dict:
    """Query device properties.  Returns a dict with human-readable fields."""
    lib = _lib()
    lib.baloon_query_device.argtypes = [ctypes.c_int, ctypes.POINTER(_DeviceInfo)]
    lib.baloon_query_device.restype  = ctypes.c_int

    info = _DeviceInfo()
    rc = lib.baloon_query_device(dev, ctypes.byref(info))
    if rc != 0:
        raise RuntimeError(f"baloon_query_device({dev}) failed (rc={rc})")

    return {
        "device_count":  info.device_count,
        "name":          info.name.decode("utf-8", errors="replace").rstrip("\x00"),
        "warp_size":     info.warp_size,
        "max_threads":   info.max_threads,
        "sm_count":      info.sm_count,
        "compute":       f"{info.compute_major}.{info.compute_minor}",
        "shmem_kb":      info.max_shmem_per_block // 1024,
        "vram_gib":      round(info.total_mem / (1024**3), 1),
        "vram_free_gib": round(info.free_mem  / (1024**3), 1),
    }


def smoke_test(dev: int = 0) -> None:
    """Run the warp-shuffle reduction smoke test.  Raises on failure."""
    lib = _lib()
    lib.baloon_smoke_test.argtypes = [ctypes.c_int]
    lib.baloon_smoke_test.restype  = ctypes.c_int

    rc = lib.baloon_smoke_test(dev)
    if rc != 0:
        raise RuntimeError(f"Smoke test FAILED on device {dev}")

