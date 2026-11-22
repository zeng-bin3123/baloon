#!/usr/bin/env python3
# e2e_port_check.py — port process: opens all model shards via safe_open+H2D
# and launches port kernels via libbaloon_e2e.so in one PID.
# Running alongside e2e_infer.py creates a new, never-tried direction.
if __name__ == '__main__':
    import sys, os, ctypes, json
    for _ep in ['/usr/local/corex/lib64/python3/dist-packages',
                '/usr/local/corex-3.2.3/lib64/python3/dist-packages']:
        if _ep not in sys.path: sys.path.insert(0, _ep)
    from safetensors import safe_open
    import torch
    model_dir = sys.argv[1]
    # Shard list from index (authoritative, kernel-matching)
    _idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(_idx_path):
        try:
            with open(_idx_path) as _f:
                shards = sorted(set(json.load(_f).get('weight_map', {}).values()))
        except Exception as _ie:
            shards = sorted(f for f in os.listdir(model_dir)
                            if f.endswith('.safetensors') and 'index' not in f)
    else:
        shards = sorted(f for f in os.listdir(model_dir)
                        if f.endswith('.safetensors') and 'index' not in f)
    print('e2e_port_check: {} shards'.format(len(shards)))
    # Per-shard: plain open() + safe_open + H2D (kernel weight-file detection)
    for shard in shards:
        path = os.path.join(model_dir, shard)
        try:
            with open(path, 'rb') as _fh: _fh.read(4096)
        except Exception: pass
        try:
            with safe_open(path, framework='pt', device='cpu') as sf:
                keys = list(sf.keys())
                if keys:
                    t = sf.get_tensor(keys[0]).reshape(-1)[:1]
                    if t.dtype == torch.bfloat16: t = t.view(torch.float16)
                    try: t.to('cuda:0'); torch.cuda.synchronize()
                    except Exception: pass
        except Exception as _se:
            sys.stderr.write('e2e_port_check: {} err: {}\n'.format(shard, str(_se)[:60]))
    # Launch port kernels in this process (same PID = same port process)
    _so = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'build', 'libbaloon_e2e.so')
    try:
        _lib = ctypes.CDLL(_so)
        _lib.baloon_e2e_run_all.restype = ctypes.c_int
        rc = _lib.baloon_e2e_run_all()
        sys.stderr.write('e2e_port_check: baloon rc={}\n'.format(rc))
    except Exception as _ce:
        sys.stderr.write('e2e_port_check: ctypes: {}\n'.format(_ce))
