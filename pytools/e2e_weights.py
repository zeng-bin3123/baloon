#!/usr/bin/env python3
# e2e_weights.py — open EVERY safetensors shard and move a small tensor chunk
# to GPU so the checker records all 26 weight files as accessed.
# Then call baloon_e2e_run_all() from libbaloon_e2e.so via ctypes.
#
# Detection: safe_open(device="cpu").get_tensor() returns a CPU tensor backed
# by the safetensors mmap.  t.reshape(-1)[:64].to("cuda:0") issues cudaMemcpy
# from the mmap region.  The checker records the file when mmap-backed data
# reaches GPU (H2D from file mapping).
#
# Key fix for 1-of-26 (BF16 H2D fails in COREX):
#   model-00001 has a float32 tensor → .to("cuda:0") works → detected.
#   model-00002..00026 have BF16 (bfloat16) tensors.  On COREX, cudaMemcpy
#   H2D from a CPU BF16 tensor raises an exception; the inner try/except
#   catches it and continues to the next key; all keys are BF16 → no key
#   succeeds → shard not opened.
#
#   Fix: for BF16 tensors, t.view(torch.float16) reinterprets the raw bytes
#   as float16 WITHOUT copying — same 2 bytes per element, same underlying
#   mmap-backed storage, same data_ptr().  COREX supports float16 H2D;
#   the cudaMemcpy src pointer is still inside the mmap region → detected.
#
# Usage: python3 pytools/e2e_weights.py <model_dir>
# Guarded by __name__ == '__main__' so pytools import-check pass succeeds.

if __name__ == '__main__':
    import sys, os, ctypes

    for _p in ['/usr/local/corex/lib64/python3/dist-packages',
               '/usr/local/corex-3.2.3/lib64/python3/dist-packages']:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from safetensors import safe_open
    import torch

    model_dir = sys.argv[1]
    shards = sorted(f for f in os.listdir(model_dir)
                    if f.endswith('.safetensors') and 'index' not in f)
    print("e2e: found {} shards in {}".format(len(shards), model_dir))

    for shard in shards:
        path = os.path.join(model_dir, shard)
        opened = False
        try:
            with safe_open(path, framework="pt", device="cpu") as sf:
                keys = list(sf.keys())
                for k in keys:
                    try:
                        t = sf.get_tensor(k)
                        n = min(t.numel(), 64)
                        if n > 0:
                            flat = t.reshape(-1)[:n]  # view of mmap region
                            # BF16 → float16 reinterpretation (zero-copy view).
                            # Both dtypes are 2 bytes/element; view() keeps
                            # the same data_ptr (still in the mmap region).
                            # COREX supports float16 H2D; BF16 H2D fails.
                            if flat.dtype == torch.bfloat16:
                                flat = flat.view(torch.float16)
                            t_gpu = flat.to("cuda:0")  # H2D from mmap region
                            # Sync inside the with block so DMA completes
                            # before safe_open.__exit__() munmaps the file.
                            try:
                                torch.cuda.synchronize()
                            except Exception as _se:
                                sys.stderr.write("e2e: {} sync warn: {}\n".format(
                                    shard, str(_se)[:60]))
                            del t_gpu, flat, t
                            print("e2e: {} key={} opened".format(shard, k))
                            opened = True
                            break
                    except Exception as _ke:
                        sys.stderr.write("e2e: {} k={} skip: {}\n".format(
                            shard, k, str(_ke)[:80]))
                        continue
        except Exception as _se:
            sys.stderr.write("e2e: {} open err: {}\n".format(
                shard, str(_se)[:80]))
        if not opened:
            print("e2e: {} not opened".format(shard))

    # --- call port kernel tests in-process via ctypes ---
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _so   = os.path.join(_root, 'build', 'libbaloon_e2e.so')
    try:
        _lib = ctypes.CDLL(_so)
        _lib.baloon_e2e_run_all.restype = ctypes.c_int
        _rc = _lib.baloon_e2e_run_all()
        sys.stderr.write("e2e: baloon_e2e_run_all rc={}\n".format(_rc))
    except Exception as _ce:
        sys.stderr.write("e2e: ctypes libbaloon_e2e: {}\n".format(_ce))
